import os
import glob
import torch
import torch.nn as nn
import torch.utils.data as tud

import torchvision as tv
import torchvision.tv_tensors as tvtensor
import torchvision.transforms.v2 as tv2

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
        self.classes = 19
        self.categories = 8

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

    # Cityscapes has object classes, but also category classes. For instance,
    # bus and car have different ids but belong to the same category.
    # Solve a simpler problem and reduce the dataset to categories only.
    def _simplify_mask_to_categories(self, mask):
        # void
        mask[mask < 7] = 0
        # flat
        mask[(mask >= 7) & (mask < 11)] = 1
        # construction
        mask[(mask >= 11) & (mask < 17)] = 2
        # object
        mask[(mask >= 17) & (mask < 21)] = 3
        # nature
        mask[(mask >= 21) & (mask < 23)] = 4
        # sky
        mask[(mask == 23)] = 5
        # human
        mask[(mask >= 24) & (mask < 26)] = 6
        # vehicle
        mask[(mask >= 26)] = 7
        return mask

    def _id_to_classes(self, mask):
        import cityscapesscripts as css

        # TODO

    def __getitem__(self, index) -> tuple[tvtensor.Image, tvtensor.Mask]:
        img = tv.io.decode_image(self.img_names[index])
        img = tvtensor.Image(img)

        mask = tv.io.decode_image(self.lbl_names[index])
        mask = self._simplify_mask_to_categories(mask)
        mask = tvtensor.Mask(mask)

        if self.tfs is None:
            return img, mask
        else:
            return self.tfs((img, mask))


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
