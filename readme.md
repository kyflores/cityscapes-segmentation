# CityScapes Segmentation
Train semantic segmentation on the cityscapes dataset
![](imgs/sample_small.png)

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
See [this file](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py) for the label IDs.

Many of the cityscapes classes (trainId=255 or -1) are not used as per the instructions, so they are assigned trainId=19 which we call background.


## Usage
```
pip install -r requirements.txt

# Train the model.
python train.py -a train -p /path/to/cityscapes

# Train the model, restoring from a checkpoint
python train.py -a train -p /path/to/cityscapes -a /path/to/checkpoint.pth

# View outputs. Creates a folder called vis and writes to it
# Also creates `eval_result/` which can be passed to the official eval script.
python train.py -a vis -p /path/to/cityscpaes -a /path/to/checkpoint.pth
```

## Eval
```
# First install cityscapesscripts

CITYSCAPES_RESULTS="$(pwd)/eval_result/" CITYSCAPES_DATASET="$(pwd)/../cityscapes/" csEvalPixelLevelSemanticLabeling
```

## Model
The default configuration is resnet18 with pretrained imagenet weights with some
upscaler blocks after to form a decoder back to the source resolution

The model can be replaced by anything that has the following in/out shapes.
* Input: `[B, 3, H, W]`
* Output: `[B, NUM_CATEGORIES, H, W]`

## Other
I originally used SGD with LR=1e-2, but my results improved drastically with AdamW and LR=5e-4
