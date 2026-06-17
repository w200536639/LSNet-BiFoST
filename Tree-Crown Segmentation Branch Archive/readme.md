# LSNet-BiFoST Tree-Crown Segmentation Branch

This folder contains the tree-crown segmentation branch of the LSNet-BiFoST framework.
The model performs binary semantic segmentation of individual tree crowns from UAV RGB imagery.

## 1. Project Overview

The tree-crown segmentation branch is used to extract crown regions from UAV RGB images.
It adopts an LSNet-based encoder and integrates boundary-enhancement and hierarchical-pooling attention modules to improve crown boundary delineation and small-object segmentation.

Main features:

* UAV RGB image input
* Binary tree-crown segmentation
* LSNet-based encoder
* Optional BIE module
* Optional HPA module
* VOC-style dataset format
* Support for training, evaluation, prediction, and Grad-CAM visualization

## 2. Directory Structure

The dataset should be organized in VOC format:

```text
VOCdevkit/
└── VOC2007/
    ├── JPEGImages/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ...
    ├── SegmentationClass/
    │   ├── image_001.png
    │   ├── image_002.png
    │   └── ...
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

Each line in `train.txt` and `val.txt` should contain the image prefix without suffix. For example:

```text
image_001
image_002
image_003
```

For each prefix, the program will read:

```text
VOCdevkit/VOC2007/JPEGImages/image_001.jpg
VOCdevkit/VOC2007/SegmentationClass/image_001.png
```

## 3. Environment Requirements

Install dependencies:

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
scikit-image
scikit-learn
```

For CUDA users, install `torch` and `torchvision` according to the official PyTorch command that matches your CUDA version.

## 4. Main Files

```text
train.py
predict_folder_segmentation.py
evaluate_segmentation.py
explain_gradcam_segmentation.py
nets/lsnet_bifost_segmentation.py
nets/unet_training.py
utils/dataloader.py
utils/utils_fit.py
utils/callbacks.py
utils/utils.py
```

Main functions:

* `train.py`: training script for tree-crown segmentation
* `predict_folder_segmentation.py`: predict all images in a folder
* `evaluate_segmentation.py`: evaluate segmentation performance
* `explain_gradcam_segmentation.py`: Grad-CAM visualization for segmentation branch
* `LSNetBiFoSTSegmentation`: main segmentation model
* `UnetDataset`: VOC-style segmentation dataset loader

## 5. Data Format

### 5.1 RGB Images

RGB images should be stored in:

```text
VOCdevkit/VOC2007/JPEGImages/
```

Supported image formats usually include:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
```

Images are read as RGB and resized to the configured input size.

### 5.2 Segmentation Masks

Segmentation masks should be stored in:

```text
VOCdevkit/VOC2007/SegmentationClass/
```

For binary segmentation:

```text
0 = background
1 or non-zero value = tree crown
```

If the original mask uses a specific label value, the data loader or evaluation script should convert the target label to foreground class `1`.

## 6. Important Configuration

The main configuration is located at the top of `train.py`.

### 6.1 Model Configuration

```python
NUM_CLASSES = 2
BACKBONE = "lsnet_b"
PRETRAINED = False
MODEL_PATH = r""
```

Available backbone options:

```text
lsnet_t
lsnet_s
lsnet_b
```

If training from scratch, keep:

```python
MODEL_PATH = r""
```

If resuming training, set it to an existing checkpoint:

```python
MODEL_PATH = r"logs/xxx/last_epoch_weights.pth"
```

### 6.2 Ablation Switches

```python
USE_BIE = True
USE_HPA = True
```

These switches control whether the BIE and HPA modules are enabled.

Recommended full model setting:

```python
USE_BIE = True
USE_HPA = True
```

For ablation experiments:

```python
USE_BIE = False
USE_HPA = False
```

or:

```python
USE_BIE = True
USE_HPA = False
```

or:

```python
USE_BIE = False
USE_HPA = True
```

### 6.3 Input Size

```python
INPUT_SHAPE = [640, 640]
```

The height and width should be divisible by 32.

### 6.4 Training Schedule

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

### 6.5 Optimizer and Learning Rate

```python
INIT_LR = 1e-5
MIN_LR = 1e-7
OPTIMIZER_TYPE = "adamw"
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
LR_DECAY_TYPE = "cos"
```

### 6.6 Loss Function

The segmentation branch commonly uses a combination of cross-entropy loss and Dice loss.

Typical setting:

```python
DICE_LOSS = True
FOCAL_LOSS = False
```

## 7. Training

Run:

```bash
python train.py
```

The program will:

1. Read `train.txt` and `val.txt`
2. Load RGB images and segmentation masks
3. Build the LSNet-BiFoST segmentation model
4. Load checkpoint weights if `MODEL_PATH` is provided
5. Train the segmentation branch
6. Save logs and model weights

## 8. Output Files

Training outputs are usually saved to:

```text
logs/
```

or a customized save directory, for example:

```text
logs_seg/
logs_lsnet_bifost/
```

Common output files:

```text
best_epoch_weights.pth
last_epoch_weights.pth
epxxx-lossxxx-valxxx.pth
```

Description:

* `best_epoch_weights.pth`: latest best model
* `last_epoch_weights.pth`: latest epoch model, useful for resuming training
* `epxxx-lossxxx-valxxx.pth`: periodically saved checkpoint

## 9. Prediction

To predict all images in a folder, use:

```bash
python predict_folder_segmentation.py
```

Default input folder:

```text
input_images/
```

Default output folder:

```text
prediction_results/
```

Typical outputs:

```text
prediction_results/
├── binary_masks/
├── probability_maps/
├── red_masks/
├── red_overlays/
└── visualizations/
```

Output description:

* `binary_masks`: 0/255 binary masks
* `probability_maps`: foreground probability maps
* `red_masks`: red foreground masks
* `red_overlays`: red masks overlaid on original RGB images
* `visualizations`: combined visualization figures

## 10. Evaluation

To evaluate segmentation performance, use:

```bash
python evaluate_segmentation.py
```

The evaluation script can calculate metrics such as:

```text
Precision
Recall
F1-score
Overall IoU
TP
FP
FN
```

Instance-level evaluation usually requires:

```python
MATCH_IOU_THRESHOLD = 0.5
INSTANCE_MIN_AREA = 5
```

The foreground class is usually:

```python
FOREGROUND_CLASS_ID = 1
```

If the ground-truth mask uses a specific label value, set the corresponding target label value in the evaluation script.

## 11. Grad-CAM Visualization

To visualize model attention for the segmentation branch, use:

```bash
python explain_gradcam_segmentation.py
```

Typical target layers include:

```text
hpa16
hpa32
up4_conv
up3_conv
up2_conv
up1_conv
out_head
final_logits
```

Outputs are saved to:

```text
gradcam_lsnet_bifost/
```

Typical outputs:

```text
gradcam_lsnet_bifost/
├── triplet/
├── hpa16/
├── hpa32/
├── up4_conv/
├── up3_conv/
├── up2_conv/
├── up1_conv/
├── out_head/
└── final_logits/
```

Each visualization usually includes:

```text
Input RGB
Grad-CAM heatmap
Overlay
```

## 12. Common Problems

### 12.1 val.txt encoding error

If an error occurs when reading `val.txt`, use encoding fallback:

```python
encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
```

This is useful when image names contain Chinese characters or full-width brackets.

### 12.2 No supported images found

Message:

```text
No supported images found in: input_images
```

Check whether the input folder exists and whether image suffixes are supported:

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
```

### 12.3 Mask values are not 0/1

If the ground-truth mask uses values such as `38` for tree crown, convert the target label to foreground class `1` before training or evaluation.

Example:

```python
mask = (mask == TARGET_LABEL_VALUE).astype(np.uint8)
```

or for binary masks:

```python
mask = (mask > 0).astype(np.uint8)
```

### 12.4 Checkpoint does not exist

Message:

```text
Checkpoint does not exist
```

If training from scratch, keep:

```python
MODEL_PATH = r""
```

If resuming training, set it to an existing checkpoint path.

### 12.5 Progress bar keeps printing new lines

If the progress bar keeps printing new lines in PyCharm, enable:

```text
Run/Debug Configurations → Emulate terminal in output console
```

If it still prints new lines, use a simplified one-line progress display in `utils_fit.py`.

## 13. Recommended Training Settings

Recommended default settings for the full tree-crown segmentation branch:

```python
NUM_CLASSES = 2
BACKBONE = "lsnet_b"
INPUT_SHAPE = [640, 640]

USE_BIE = True
USE_HPA = True

INIT_LR = 1e-5
MIN_LR = 1e-7
OPTIMIZER_TYPE = "adamw"
LR_DECAY_TYPE = "cos"

FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4

DICE_LOSS = True
FOCAL_LOSS = False
```

## 14. Notes

* Keep images and masks spatially aligned.
* The prefixes in `train.txt` and `val.txt` must match image and mask filenames.
* For binary segmentation, the model output has two classes: background and tree crown.
* The predicted tree-crown mask can be used as spatial guidance for the tree-height regression branch.
* For publication figures, use the overlay and Grad-CAM results to show model attention and segmentation behavior.

## 15. Description

This branch is part of the LSNet-BiFoST framework for UAV-based tree-crown segmentation and tree-height estimation. The segmentation branch extracts individual tree-crown regions from UAV RGB imagery and provides mask guidance for the tree-height regression branch.
