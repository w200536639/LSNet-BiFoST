# LSNet-BiFoST Tree-Height Regression Branch

This folder contains the tree-height regression branch of the LSNet-BiFoST framework.
The model estimates tree height from UAV RGB imagery and the corresponding crown mask. The input consists of four channels: RGB image channels and one crown-mask channel.

## 1. Project Overview

The tree-height regression branch is designed to predict a continuous tree-height map. It uses RGB imagery and crown-mask information as input, and learns from processed tree-height raster labels.

Main features:

* RGB + crown mask four-channel input
* Single-channel tree-height regression output
* FocalNet-based encoder
* Optional SVIT module
* Optional BIE module
* Mask-guided loss calculation
* Support for mixed-precision training
* Automatic handling of invalid NoData values in height maps

## 2. Directory Structure

The dataset should be organized as follows:

```text
TreeHeightDataset/
├── rgb/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── mask/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── heights/
│   ├── image_001_processed.tif
│   ├── image_002_processed.tif
│   └── ...
├── train.txt
└── val.txt
```

Each line in `train.txt` and `val.txt` should contain the file prefix without suffix. For example:

```text
image_001
image_002
image_003
```

For each prefix, the program will automatically read:

```text
rgb/image_001.jpg
mask/image_001.png
heights/image_001_processed.tif
```

## 3. Environment Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```text
torch
torchvision
tensorboard
scipy
numpy
matplotlib
opencv-python
tqdm
Pillow
h5py
labelme==3.16.7
tifffile
scikit-image
scikit-learn
```

If CUDA is used, it is recommended to install `torch` and `torchvision` according to the official PyTorch installation command that matches your CUDA version.

## 4. Main Files

```text
train.py
nets/lsnet_bifost_height_regression.py
nets/unet_training.py
utils/utils_fit.py
utils/callbacks.py
utils/utils.py
```

Main functions:

* `train.py`: main training script for tree-height regression
* `LSNetBiFoSTHeightRegression`: tree-height regression model
* `fit_one_epoch_reg`: one-epoch training and validation loop
* `RgbMaskHeightDataset`: dataset class for RGB + mask + height map loading

## 5. Data Format

### 5.1 RGB Images

RGB images should be stored in:

```text
TreeHeightDataset/rgb/
```

Default suffix:

```text
.jpg
```

The image is read as RGB and normalized to `[0, 1]`.

### 5.2 Crown Masks

Crown masks should be stored in:

```text
TreeHeightDataset/mask/
```

Default suffix:

```text
.png
```

Mask values greater than 0 are treated as foreground crown pixels:

```python
mask = mask > 0
```

The mask is used in two ways:

1. As the fourth input channel
2. As part of the valid loss mask

### 5.3 Height Labels

Height labels should be stored in:

```text
TreeHeightDataset/heights/
```

Default suffix:

```text
_processed.tif
```

The height label is a single-band floating-point raster. Invalid NoData values such as:

```text
-3.4028234663852886e+38
```

are converted to `NaN` and excluded from loss calculation.

## 6. Important Configuration

The main configuration is located at the top of `train.py`.

### 6.1 Model Configuration

```python
BACKBONE = "focal_s"
IN_CHANNELS = 4
OUT_CHANNELS = 1
PRETRAINED = False
MODEL_PATH = r""
```

Available backbone options:

```text
focal_t
focal_s
focal_b
```

If training from scratch, keep:

```python
MODEL_PATH = r""
```

If resuming training, set it to an existing checkpoint:

```python
MODEL_PATH = r"logs_reg_rgb_mask/focal_s_svit1_bie1_f161_f321/last_epoch_weights.pth"
```

### 6.2 Ablation Switches

```python
USE_SVIT = True
USE_BIE = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True
SVIT_STOKEN_SIZE = (4, 4)
SVIT_HEADS = 8
SVIT_N_ITER = 1
```

These switches control whether the SVIT and BIE modules are used.

### 6.3 Input Size

```python
INPUT_SHAPE = [640, 640]
```

The height and width must be divisible by 32.

### 6.4 Height Range

```python
VALID_HEIGHT_MIN = 0.0
VALID_HEIGHT_MAX = 6.0
NORMALIZE_MODE = "minmax"
```

The model only uses valid height pixels within the specified range. Invalid values and NoData pixels are excluded.

### 6.5 Training Schedule

```python
INIT_EPOCH = 0
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_TRAIN = True
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4
```

The default training strategy uses two stages:

1. Freeze training stage: epoch 0–50
2. Unfreeze training stage: epoch 50–300

### 6.6 Optimizer and Learning Rate

```python
INIT_LR = 1e-5
MIN_LR = 1e-7
OPTIMIZER_TYPE = "adam"
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
LR_DECAY_TYPE = "cos"
```

### 6.7 Loss Function

```python
LOSS_TYPE = "combined"
LOSS_ALPHA = 0.5
```

Supported loss types:

```text
mse
mae
combined
```

The combined loss uses both MSE and MAE.

## 7. Training

Run:

```bash
python train.py
```

The program will:

1. Read `train.txt` and `val.txt`
2. Estimate the valid height range from height maps
3. Build the LSNet-BiFoST height regression model
4. Load checkpoint weights if `MODEL_PATH` is provided
5. Train the regression branch
6. Save logs and model weights

## 8. Output Files

Training outputs are saved to:

```text
logs_reg_rgb_mask/
```

The run folder is automatically named according to model settings, for example:

```text
logs_reg_rgb_mask/focal_s_svit1_bie1_f161_f321/
```

Common output files:

```text
best_epoch_weights.pth
last_epoch_weights.pth
best_epoch_xxx_valloss_xxxx_valmae_xxxx.pth
epxxx-lossxxx-valxxx.pth
```

Description:

* `best_epoch_weights.pth`: latest best model
* `last_epoch_weights.pth`: latest epoch model, useful for resuming training
* `best_epoch_xxx_...pth`: historical best model with epoch and metrics
* `epxxx-lossxxx-valxxx.pth`: periodically saved checkpoint

## 9. TensorBoard Logs

Loss logs are saved under the run directory:

```text
logs_reg_rgb_mask/focal_s_svit1_bie1_f161_f321/loss_xxxxx/
```

Start TensorBoard with:

```bash
tensorboard --logdir logs_reg_rgb_mask
```

Then open the displayed local URL in your browser.

## 10. NoData Handling

Some GeoTIFF height maps may contain extreme NoData values such as:

```text
-3.4028234663852886e+38
```

The training code converts these values to `NaN`. These invalid pixels are not used in height normalization or loss calculation.

The valid loss mask is calculated as:

```text
crown mask × valid height mask
```

Only pixels that satisfy all conditions below are used for training:

```text
inside crown mask
finite height value
height >= VALID_HEIGHT_MIN
height <= VALID_HEIGHT_MAX
```

## 11. Common Problems

### 11.1 Checkpoint does not exist

Message:

```text
Checkpoint does not exist: model_data/best_epoch_weights.pth
```

This means the specified weight file does not exist. If training from scratch, set:

```python
MODEL_PATH = r""
```

If resuming training, set `MODEL_PATH` to a real checkpoint path.

### 11.2 GDAL_NODATA warning

Message:

```text
parsing GDAL_NODATA tag raised ValueError
```

This is caused by extreme NoData values in GeoTIFF files. The modified code safely converts them to `NaN`, so they will not affect training.

### 11.3 Progress bar keeps printing new lines

If the console keeps printing progress lines, use the modified `utils_fit.py` with single-line progress display. In PyCharm, also enable:

```text
Run/Debug Configurations → Emulate terminal in output console
```

If the console still does not support single-line refresh, increase:

```python
PROGRESS_REFRESH_STEPS = 50
PROGRESS_REFRESH_SECONDS = 10.0
```

### 11.4 Input type and weight type mismatch

Message:

```text
Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

This usually occurs when writing the TensorBoard model graph using CPU dummy input while the model is already on GPU. It does not affect the main training process. Disable model graph writing or ensure the dummy input is moved to the same device as the model.

## 12. Recommended Training Settings

For the current tree-height regression branch, the recommended default settings are:

```python
BACKBONE = "focal_s"
IN_CHANNELS = 4
OUT_CHANNELS = 1
INPUT_SHAPE = [640, 640]

USE_SVIT = True
USE_BIE = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True

INIT_LR = 1e-5
MIN_LR = 1e-7
OPTIMIZER_TYPE = "adam"
LR_DECAY_TYPE = "cos"

LOSS_TYPE = "combined"
LOSS_ALPHA = 0.5

FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4
```

## 13. Notes

* Keep RGB images, masks, and height maps spatially aligned.
* The file prefix in `train.txt` and `val.txt` must match the filenames in all three folders.
* Height maps should be floating-point GeoTIFF files.
* Invalid or NoData height pixels should not be encoded as valid values.
* The crown mask is required because it provides spatial guidance for the height regression branch.
* The model output is a normalized height map. During evaluation or visualization, convert it back to meters using the recorded `height_min` and `height_max`.

## 14. Citation / Description

This branch is part of the LSNet-BiFoST framework for UAV-based individual tree-crown segmentation and tree-height estimation. The height regression branch uses crown-mask-guided spatial constraints to reduce background interference and improve tree-height estimation.
