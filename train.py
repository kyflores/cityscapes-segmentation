import os
import random
import argparse

import torch
import torchvision.io as tio
import torchvision.utils as tu
import torchvision.transforms.v2 as tv2
import torchmetrics as tm

from tqdm import tqdm
import matplotlib.pyplot as plt

import dataset
import unet


def random_grid(imgs):
    grid = tu.make_grid(imgs)
    return grid.permute(1, 2, 0)


def train(opt):
    device = opt.device

    # TODO pull this from a config file
    dim = 512
    batchsize = 16
    start_epoch = 0
    epochs = 150
    lr = 0.02
    lr_floor = 1e-5

    train_tfs = tv2.Compose(
        [
            # tv2.Resize((512, 256)), # 1/4 scale, cityscapes images are big.
            tv2.Resize((1024, 512)),  # 1/2 scale, cityscapes images are big.
            tv2.RandomHorizontalFlip(0.5),
            tv2.RandomResizedCrop((dim, dim)),
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfs = tv2.Compose(
        [
            # tv2.Resize((512, 256)),
            tv2.Resize((1024, 512)),
            tv2.RandomCrop((dim, dim)),
            tv2.ToDtype(torch.float32, scale=True),
            tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train = dataset.CityScapesDataset(opt.path, "train", tfs=train_tfs)
    val = dataset.CityScapesDataset(opt.path, "val", tfs=val_tfs)
    test = dataset.CityScapesDataset(opt.path, "test", tfs=val_tfs)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batchsize, num_workers=8, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batchsize, num_workers=8, shuffle=True
    )

    # Sample the dataset to make sure our dataloader works.
    # choices = random.choices(train, k=4)
    # imgs = torch.stack([x[0] for x in choices])
    # masks = torch.stack([x[1] for x in choices])
    # fig, axes = plt.subplots(2, 1)
    # axes[0].imshow(random_grid(imgs))
    # axes[1].imshow(random_grid(masks))
    # plt.show()

    # TODO see https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    num_classes = train.categories
    model = unet.UnetSeg(3, num_classes, filts=32).to(device).train()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr_floor
    )

    if opt.checkpoint:
        tmp = torch.load(opt.checkpoint)
        model.load_state_dict(tmp["model"])
        optimizer.load_state_dict(tmp["optim"])
        scheduler.load_state_dict(tmp["sched"])
        start_epoch = tmp["epoch"]
        epochs = tmp["total_epochs"]

    lossfn = torch.nn.CrossEntropyLoss(ignore_index=255)

    train_loss_plot = []
    loss_plot = []
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            train_losses = []
            for i, (images, targets) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                images = images.to(device)
                # CrossEntropy allows 2D map of IDs rather than one-hot encoded map
                targets = targets.squeeze(1).long().to(device)

                outs = model(images)
                loss = lossfn(outs, targets)
                loss.backward()
                train_losses.append(loss.cpu().item())
                optimizer.step()

            train_loss = torch.Tensor(train_losses).mean().item()
            train_loss_plot.append(train_loss)
            print("Epoch: {}, Train Loss: {:.4f}".format(epoch, train_loss))
            scheduler.step()

            if (epoch) % 5 == 0:
                losses = []
                model.eval()
                corrects = []
                total = len(val)
                print("Running val...")
                with torch.no_grad():
                    for i, (images, targets) in enumerate(tqdm(val_loader)):
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

                        loss = lossfn(outs, targets)
                        losses.append(loss.cpu().item())

                eval_loss = torch.tensor(losses).mean().item()
                correct = torch.tensor(corrects).mean().item()
                print(
                    "Epoch: {}, Eval Loss: {:.4f}. {:.2f}% pixelwise accuracy".format(
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


def visualize(opt):
    device = opt.device

    val_tfs = tv2.Compose([tv2.ToDtype(torch.float32, scale=True)])
    val = dataset.CityScapesDataset(opt.path, "val", tfs=val_tfs)
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=1,
    )

    tmp = torch.load(opt.checkpoint)
    model = unet.UnetSeg(3, val.categories, filts=32)
    model.load_state_dict(tmp["model"])
    model = model.to(device)

    os.makedirs("vis", exist_ok=True)

    ioumetric = tm.JaccardIndex(task="multiclass", num_classes=val.categories)
    with torch.no_grad():
        ious = []
        for ix, (img, lbl) in enumerate(tqdm(val_loader)):
            pred = model(img.to(device))
            pred = pred.softmax(dim=1).max(dim=1)[1].cpu()

            ious.append(ioumetric(pred.unsqueeze(0), lbl).cpu().float().item())

            img = img.squeeze(0)
            lbl = lbl.squeeze(0)

            out = torch.cat(
                [255 * torch.mean(img, dim=0, keepdim=True), 10 * lbl, 10 * pred], dim=1
            ).squeeze()

            tio.write_jpeg(
                out.clip(0, 255).to(torch.uint8).unsqueeze(0).expand(3, -1, -1),
                "vis/{}.jpg".format(ix),
            )

        print("Average IOU {}".format(torch.tensor(ious).mean().item()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cityscapes Segmentation")
    parser.add_argument(
        "-a",
        "--action",
        help="What task to perform",
        default="train",
        choices=["train", "vis"],
    )
    parser.add_argument(
        "-d", "--device", help="Compute device to run on", default="auto"
    )
    parser.add_argument(
        "-c", "--checkpoint", help="Checkpoint path to restore", default=""
    )
    parser.add_argument(
        "-p", "--path", help="Path to cityscapes root", default="./cityscapes"
    )
    opt = parser.parse_args()

    if opt.device == "auto":
        opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    if opt.action == "train":
        train(opt)
    elif opt.action == "vis":
        visualize(opt)
