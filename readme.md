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
Desktop w/ dGPU:
```
virtualenv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

# Train the model.
python train.py -a train -p /path/to/cityscapes

# Train the model, restoring from a checkpoint
python train.py -a train -p /path/to/cityscapes -c /path/to/checkpoint.pth

# View outputs. Creates a folder called vis and writes to it
# Also creates `eval_result/` which can be passed to the official eval script.
python train.py -a vis -p /path/to/cityscapes -c /path/to/checkpoint.pth

# Export an onnx model
python train.py -a export -p /path/to/cityscapes

```

Orin Nano:
Here's how to set up the Orin for inference with TensorRT
```
sudo apt install nvidia-tensorrt-dev
# Use system-site-packages to pick up tensorrt from the system install.
virtualenv myvenv --system-site-packages
source myvenv/bin/activate

export PIP_INDEX_URL=http://jetson.webredirect.org/jp6/cu126
export PIP_TRUSTED_HOST=jetson.webredirect.org
pip install -r requirements.txt

# To test compiled model...
/usr/src/tensorrt/bin/trtexec --onnx=~/Downloads/cityscapes_resnet18.onnx --fp16 --saveEngine=cityscapes_trt.engine
```

## Eval
```
pip install cityscapesscripts

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
