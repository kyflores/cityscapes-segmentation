import os
import argparse
import json

import torch
import torch.nn.functional as F
import torchvision.io as tio
import torchvision.utils as tu
import torchvision.transforms.v2 as tv2
import torchmetrics as tm

from tqdm import tqdm
from torchinfo import summary
import matplotlib.pyplot as plt

import dataset
import fcn
import unet

# See https://docs.pytorch.org/docs/stable/notes/cuda.html for what this does
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# https://arxiv.org/pdf/1708.02002
# https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289/2
def focal_loss(inputs, target, alpha, gamma, ignore_index=-100):
    ce = F.cross_entropy(inputs, target, reduction="none", ignore_index=ignore_index)
    pt = torch.exp(-ce)
    focal_loss = (alpha * (1 - pt) ** gamma * ce).mean()  # mean over the batch
    return focal_loss


# Adapted from https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
def dice_loss(preds, targets, ignore_index=-100):
    b, num_classes, h, w = preds.shape

    preds = preds.softmax(dim=1)
    target_dist = (
        F.one_hot(targets.squeeze(1).long().to(preds.device), num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )

    assert target_dist.shape == preds.shape

    dice = torch.zeros(b, dtype=torch.float32, device=preds.device)
    for c in range(num_classes):
        if c == ignore_index:
            continue
        intersection = (2 * preds[:, c] * target_dist[:, c]).sum(dim=(-2, -1))
        union = (preds[:, c] + target_dist[:, c]).sum(dim=(-2, -1))

        dice += (intersection + 1) / (union + 1)

    return 1 - dice.mean() / num_classes


def train(opt, hparams):
    device = opt.device

    dim = hparams["dim"]
    batchsize = hparams["batchsize"]
    start_epoch = 0
    epochs = hparams["epochs"]
    lr = hparams["lr"]
    lr_floor = hparams["lr_floor"]
    encoder = hparams["encoder"]

    train_tfs = tv2.Compose(
        [
            tv2.RandomHorizontalFlip(0.5),
            tv2.RandomResizedCrop((dim, dim)),
            tv2.ToDtype(torch.float32, scale=True),
            # This guy MUST come before the normalize.
            tv2.ColorJitter(
                brightness=(0.875, 1.125),
                contrast=(0.9, 1.1),
                saturation=(0.9, 1.1),
                hue=None,
            ),
            # Since we use pretrained resnet, we also want to use imagenet normalization.
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfs = tv2.Compose(
        [
            tv2.RandomCrop((dim, dim)),  # Crop val to keep the image size small.
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train = dataset.CityScapesDataset(opt.path, "train", tfs=train_tfs)
    val = dataset.CityScapesDataset(opt.path, "val", tfs=val_tfs)
    test = dataset.CityScapesDataset(opt.path, "test", tfs=val_tfs)

    cpus = os.cpu_count()
    if cpus is None:
        cpus = 4

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batchsize, num_workers=cpus, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batchsize, num_workers=cpus, shuffle=True
    )

    num_classes = train.classes
    # model = unet.UnetSeg(3, num_classes, filts=32).to(device).train()
    model = fcn.Fcn(3, num_classes, encoder=encoder).to(device).train()
    summary(model, (1, 3, dim, dim))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_floor
    )

    if opt.checkpoint:
        tmp = torch.load(opt.checkpoint)
        model.load_state_dict(tmp["model"])
        if not opt.weights_only:
            optimizer.load_state_dict(tmp["optim"])
            scheduler.load_state_dict(tmp["sched"])
            start_epoch = tmp["epoch"]
            epochs = tmp["total_epochs"]
        else:
            print("Don't load opt")

    train_loss_plot = []
    loss_plot = []
    try:
        scaler = torch.amp.grad_scaler.GradScaler(device=device)
        for epoch in range(start_epoch, epochs):
            model.train()

            cel_losses = []
            focal_losses = []
            dice_losses = []
            for i, (images, targets, _) in enumerate(tqdm(train_loader)):
                images = images.to(device)
                # CrossEntropy allows 2D map of IDs rather than one-hot encoded map
                target_ids = targets.squeeze(1).long().to(device)

                # This expands the targets to a full distribution
                # target_dist = F.one_hot(target_ids.squeeze(1).long().to(device)).permute(0, 3, 1, 2).float()

                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=hparams["use_amp"]
                ):
                    outs = model(images)

                    cel = F.cross_entropy(outs, target_ids)
                    focal = focal_loss(outs, target_ids, alpha=0.25, gamma=2.0)
                    dice = dice_loss(outs, target_ids)

                    loss = torch.zeros_like(cel)
                    if hparams["cross_entropy_loss"]:
                        loss = loss + cel

                    if hparams["focal_loss"]:
                        loss = loss + focal

                    if hparams["dice_loss"]:
                        loss = loss + dice

                scaler.scale(loss).backward()

                cel_losses.append(cel.cpu().item())
                focal_losses.append(focal.cpu().item())
                dice_losses.append(dice.cpu().item())

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            cel_loss_avg = torch.Tensor(cel_losses).mean().item()
            focal_loss_avg = torch.Tensor(focal_losses).mean().item()
            dice_loss_avg = torch.Tensor(dice_losses).mean().item()

            train_loss_plot.append(focal_loss_avg + dice_loss_avg)
            print(
                "Epoch: {}, Cross Entropy: {:.4f}, Focal: {:.4f}, Dice: {:.4f}".format(
                    epoch, cel_loss_avg, focal_loss_avg, dice_loss_avg
                )
            )
            scheduler.step()

            eval_loss = 0
            if (epoch) % 5 == 0:
                losses = []
                model.eval()
                corrects = []
                print("Running val...")
                with torch.no_grad():
                    for i, (images, targets, _) in enumerate(tqdm(val_loader)):
                        images = images.to(device)
                        targets = targets.squeeze(1).long().to(device)
                        outs = model(images)

                        num_correct = torch.count_nonzero(
                            targets == outs.softmax(dim=1).max(dim=1)[1]
                        )
                        frac_correct = (
                            100 * (num_correct / targets.numel()).float().cpu().item()
                        )
                        corrects.append(frac_correct)

                        loss = focal_loss(outs, targets, alpha=0.25, gamma=2.0)
                        losses.append(loss.cpu().item())

                eval_loss = torch.tensor(losses).mean().item()
                correct = torch.tensor(corrects).mean().item()
                print(
                    "Epoch: {}, Focal Loss: {:.4f}. {:.2f}% pixelwise accuracy".format(
                        epoch, eval_loss, correct
                    )
                )
                print("Current LR is {}".format(scheduler.get_last_lr()))

                torch.save(
                    {
                        "epoch": epoch,
                        "total_epochs": epochs,
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "sched": scheduler.state_dict(),
                    },
                    f"checkpoint_{epoch}.pth",
                )
            loss_plot.append(eval_loss)

    except KeyboardInterrupt:
        pass

    plt.plot(loss_plot)
    plt.plot(train_loss_plot)
    plt.show()
    torch.save(
        {
            "epoch": epoch,
            "total_epochs": epochs,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
        },
        "checkpoint_last.pth",
    )


def visualize(opt, hparams):
    device = opt.device
    encoder = hparams["encoder"]

    val_tfs = tv2.Compose(
        [
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val = dataset.CityScapesDataset(opt.path, "val", tfs=val_tfs)
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=1,
    )

    tmp = torch.load(opt.checkpoint)
    model = fcn.Fcn(3, val.classes, encoder=encoder)
    model.load_state_dict(tmp["model"])
    model = model.to(device=device)

    os.makedirs("vis", exist_ok=True)
    os.makedirs("eval_result", exist_ok=True)

    ioumetric = tm.JaccardIndex(
        task="multiclass", num_classes=val.classes, ignore_index=19
    )
    with torch.no_grad():
        ious = []
        for ix, (img, lbl, name) in enumerate(tqdm(val_loader)):
            nm = os.path.splitext(os.path.basename(name[0]))[0]
            city, seq, frame, _ = nm.split("_")

            pred = model(img.to(device))
            pred = pred.softmax(dim=1).max(dim=1)[1].cpu()

            ious.append(ioumetric(pred.unsqueeze(0), lbl).cpu().float().item())

            img = img.squeeze(0)
            lbl = lbl.squeeze(0)

            out = torch.cat(
                [
                    255 * (0.45 + torch.mean(img, dim=0, keepdim=True) * 0.225),
                    10 * lbl,
                    10 * pred,
                ],
                dim=1,
            ).squeeze()

            tio.write_jpeg(
                out.clip(0, 255).to(torch.uint8).unsqueeze(0).expand(3, -1, -1),
                f"vis/{city}_{seq}_{frame}.png".format(ix),
            )

            eval_result = val.class_to_orig_id(pred)
            tio.write_png(
                eval_result.cpu().to(torch.uint8),
                f"eval_result/{city}_{seq}_{frame}.png",
            )

        print("Average IOU {}".format(torch.tensor(ious).mean().item()))


def export(opt, hparams):
    import onnx

    encoder = hparams["encoder"]

    val = dataset.CityScapesDataset(opt.path, "val")

    tmp = torch.load(opt.checkpoint)
    model = fcn.Fcn(3, val.classes, encoder=encoder)
    model.load_state_dict(tmp["model"])
    model = model

    print("Exporting with size {}".format(opt.export_dim))
    onnx_program = torch.onnx.export(
        model,
        (torch.randn(1, 3, opt.export_dim[0], opt.export_dim[1]),),
        f"cityscapes_{encoder}.onnx",
        input_names=[
            "csinputs",
        ],
        output_names=[
            "csoutputs",
        ],
        dynamo=False,  # This is broken in the NVIDIA Container?
        # opset_version=18,  # Might need to set explicitly if targetting an older framework.
    )

    onnx_model = onnx.load(f"cityscapes_{encoder}.onnx")
    onnx.checker.check_model(onnx_model, full_check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cityscapes Segmentation")
    parser.add_argument(
        "-a",
        "--action",
        help="What task to perform",
        default="train",
        choices=["train", "vis", "export"],
    )
    parser.add_argument(
        "-d", "--device", help="Compute device to run on", default="auto"
    )
    parser.add_argument(
        "-c", "--checkpoint", help="Checkpoint path to restore", default=""
    )
    parser.add_argument(
        "--weights-only",
        help="Load weights from checkpoint, but not optimizer state, epoch, LR",
        action="store_true",
    )
    parser.add_argument(
        "-p", "--path", help="Path to cityscapes root", default="./cityscapes"
    )
    parser.add_argument(
        "--hparams", help="Hyperparameters json", default="./hparams.json"
    )
    parser.add_argument(
        "--export-dim", help="Output dimension to export with", default="512,512"
    )
    opt = parser.parse_args()

    opt.export_dim = [int(x) for x in opt.export_dim.split(",")]
    assert len(opt.export_dim) == 2

    with open(opt.hparams, "r") as f:
        hparams = json.load(f)

    if opt.device == "auto":
        opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.action == "train":
        train(opt, hparams)
    elif opt.action == "vis":
        visualize(opt, hparams)
    elif opt.action == "export":
        export(opt, hparams)
