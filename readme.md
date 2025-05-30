# CityScapes Segmentation
Train semantic segmentation on the cityscapes dataset

## Data
Unpack the separate parts of cityscapes into one directory. For instance...
```
.
├── gtFine
│   ├── test
│   ├── train
│   └── val
├── leftImg8bit
│   ├── test
│   ├── train
│   └── val
├── license.txt
└── README
```

Cityscapes has object classes, but also category classes. For instance, bus and car have different ids but belong to the same category.
This repo solves a simpler problem and reduces the dataset to categories only, of which there are only 8.

## Usage
```
pip install -r requirements.txt

# Train the model.
python train.py -a train -p /path/to/cityscapes

# Train the model, restoring from a checkpoint
python train.py -a train -p /path/to/cityscapes -a /path/to/checkpoint.pth

# View outputs. Creates a folder called vis and writes to it
python train.py -a vis -p /path/to/cityscpaes -a /path/to/checkpoint.pth
```

## Model
The model can be replaced by anything that has the following in/out shapes.
* Input: `[B, 3, H, W]`
* Output: `[B, NUM_CATEGORIES, H, W]`

