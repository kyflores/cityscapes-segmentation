import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as tud

import torchvision as tv
import torchvision.tv_tensors as tvtensor
import torchvision.transforms.v2 as tv2

import labels

TRAIN_SUBSETS = [
    "aachen",
    "bochum",
    "bremen",
    "cologne",
    "darmstadt",
    "dusseldorf",
    "erfurt",
    "hamburg",
    "hanover",
    "jena",
    "krefeld",
    "monchengladbach",
    "strasbourg",
    "stuttgart",
    "tubingen",
    "ulm",
    "weimar",
    "zurich",
]

VAL_SUBSETS = ["frankfurt", "lindau", "munster"]

TEST_SUBSETS = ["berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"]


class CityScapesDataset(tud.Dataset):
    def __init__(self, root: str, split: str, tfs: tv2.Compose | None = None):
        super().__init__()

        self.root = root
        self.split = split
        self.tfs = tfs

        # TODO see https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        self.classes = 20
        self.categories = 8

        self.BACKGROUND = 19

        if split == "train":
            subsets = TRAIN_SUBSETS
        elif split == "val":
            subsets = VAL_SUBSETS
        elif split == "test":
            subsets = TEST_SUBSETS

        self.img_names = []
        self.lbl_names = []
        for s in subsets:
            curr_img_path = os.path.join(self.root, "leftImg8bit", self.split, s)
            curr_lbl_path = os.path.join(self.root, "gtFine", self.split, s)
            imgs = glob.glob(os.path.join(curr_img_path, "*.png"))
            self.img_names += imgs
            for i in imgs:
                nm = os.path.splitext(os.path.basename(i))[0]
                _, seq, frame, _ = nm.split("_")
                lbl_nm = os.path.join(
                    curr_lbl_path, f"{s}_{seq}_{frame}_gtFine_labelIds.png"
                )
                self.lbl_names.append(lbl_nm)

        assert all([os.path.isfile(x) for x in self.img_names])
        assert all([os.path.isfile(x) for x in self.lbl_names])
        assert len(self.img_names) == len(self.lbl_names)

    def __len__(self):
        return len(self.img_names)

    def id_to_classes(self, mask):
        for lb in labels.labels:
            if (lb.trainId != 255) and (lb.trainId != -1):
                mask[mask == lb.id] = lb.trainId
            else:
                # Assign 19 to background or "everything else"
                mask[mask == lb.id] = self.BACKGROUND
        return mask

    def class_to_orig_id(self, mask):
        m = torch.zeros_like(mask)
        for lb in labels.labels:
            if (lb.trainId != 255) and (lb.trainId != -1):
                tmp = torch.zeros_like(mask)
                tmp[mask == lb.trainId] = lb.id
                m += tmp

        m[m == self.BACKGROUND] = 0

        return m

    def __getitem__(self, index) -> tuple[tvtensor.Image, tvtensor.Mask, str]:
        name = self.img_names[index]
        img = tv.io.decode_image(self.img_names[index])
        img = tvtensor.Image(img)

        mask = tv.io.decode_image(self.lbl_names[index])
        mask = self.id_to_classes(mask)
        mask = tvtensor.Mask(mask)

        if self.tfs is None:
            return img, mask, name
        else:
            return (*(self.tfs((img, mask))), name)


if __name__ == "__main__":
    cspath = "/home/kyle/Documents/cityscapes"
    train = CityScapesDataset(cspath, "train")
    print(len(train))

    val = CityScapesDataset(cspath, "val")
    print(len(val))

    test = CityScapesDataset(cspath, "test")
    print(len(test))

    cc = tv2.CenterCrop(100)
    a, b = cc(train[0])
    print(a.shape, b.shape)
