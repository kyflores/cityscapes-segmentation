import os
import random
import argparse

import torch
import torchvision.io as tio
import torchvision.utils as tu
import torchvision.transforms.v2 as tv2

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
    epochs = 100
    lr = 0.01
    lr_floor = 1e-5

    train_tfs = tv2.Compose(
        [
            tv2.RandomHorizontalFlip(0.5),
            tv2.RandomResizedCrop((dim, dim)),
            tv2.ToDtype(torch.float32, scale=True),
        ]
    )

    val_tfs = tv2.Compose(
        [
            tv2.RandomCrop((dim, dim)),
            tv2.ToDtype(torch.float32, scale=True),
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
    choices = random.choices(train, k=4)
    imgs = torch.stack([x[0] for x in choices])
    masks = torch.stack([x[1] for x in choices])
    fig, axes = plt.subplots(2, 1)
    axes[0].imshow(random_grid(imgs))
    axes[1].imshow(random_grid(masks))
    plt.show()

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

    lossfn = torch.nn.CrossEntropyLoss(ignore_index=255)

    train_loss_plot = []
    loss_plot = []
    try:
        for epoch in range(epochs):
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
            print("Train Loss {}: {} ".format(epoch, train_loss))
            scheduler.step()

            if epoch % 5 == 0:
                losses = []
                model.eval()
                correct = 0
                total = len(val)
                print("Running val...")
                with torch.no_grad():
                    for i, (images, targets) in enumerate(tqdm(val_loader)):
                        images = images.to(device)
                        targets = targets.squeeze(1).long().to(device)
                        outs = model(images)

                        if False:
                            out_img = outs.softmax(dim=1).max(dim=1)[1].cpu()
                            fig, axes = plt.subplots(2, 1)
                            axes[0].imshow(random_grid(30 * targets.cpu()[0]))
                            axes[1].imshow(random_grid(30 * out_img[0]))
                            plt.show()

                        loss = lossfn(outs, targets)
                        losses.append(loss.cpu().item())

                eval_loss = torch.Tensor(losses).mean().item()
                print("Eval Loss {}: {} ".format(epoch, eval_loss))
                print("Current LR is {}".format(scheduler.get_last_lr()))

                torch.save(
                    {
                        "epoch": epoch,
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


def visualize(opt):
    device = opt["device"]

    val_tfs = tv2.Compose([tv2.ToDtype(torch.float32, scale=True)])
    val = dataset.CityScapesDataset(
        "/home/kyle/Documents/cityscapes", "val", tfs=val_tfs
    )
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=1,
    )

    tmp = torch.load(opt["checkpoint"])
    model = unet.UnetSeg(3, val.categories, filts=32)
    model.load_state_dict(tmp["model"])
    model = model.to(device)

    os.makedirs("eval", exist_ok=True)

    with torch.no_grad():
        for ix, (img, lbl) in enumerate(tqdm(val_loader)):
            pred = model(img.to(device))
            pred = pred.softmax(dim=1).max(dim=1)[1].cpu()

            img = img.squeeze(0)
            lbl = lbl.squeeze(0).float()

            out = torch.cat(
                [255 * torch.mean(img, dim=0, keepdim=True), 10 * lbl, 10 * pred], dim=1
            ).squeeze()

            tio.write_jpeg(
                out.clip(0, 255).to(torch.uint8).unsqueeze(0).expand(3, -1, -1),
                "eval/{}.jpg".format(ix),
            )


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
