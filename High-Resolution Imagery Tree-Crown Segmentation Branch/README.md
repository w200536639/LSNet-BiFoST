# High-Resolution Multispectral Crown-Guided Tree-Height Regression

This repository provides the high-resolution multispectral tree-height regression branch. It uses high-resolution multispectral imagery, crown masks, and processed tree-height maps to train and validate a crown-guided tree-height regression model.

The model input is a 9-channel tensor composed of 8 multispectral bands and 1 crown-mask channel. The output is a single-channel tree-height map. The current network uses a FocalNet encoder and SVIT high-level attention modules. The BIE module has been removed in this version.

## 1. Project Structure

Recommended project structure:

```text
project_root/
├── nets/
│   ├── focal_svit_crown_segmentation.py
│   ├── unet_training.py
│   └── ...
├── utils/
│   ├── callbacks.py
│   ├── utils.py
│   ├── utils_fit.py
│   └── ...
├── TreeHeightDataset/
│   ├── RSImages/
│   │   ├── sample_001.tif
│   │   └── ...
│   ├── mask/
│   │   ├── sample_001.png
│   │   └── ...
│   ├── heights/
│   │   ├── sample_001_processed.tif
│   │   └── ...
│   ├── train.txt
│   └── val.txt
├── TreeHeightDataset_AUG/
│   ├── RSImages/
│   ├── mask/
│   ├── heights/
│   ├── train.txt
│   └── val.txt
├── augment_ms_mask_height_dataset.py
├── train_ms_mask_reg.py
├── validate_ms_mask_treewise.py
├── requirements.txt
└── README.md
```

## 2. Environment

Recommended environment:

```text
Python 3.10
PyTorch
CUDA 11.8
cuDNN 8.5.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Recommended `requirements.txt`:

```txt
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
tifffile
rasterio
pandas
openpyxl
```

If `rasterio` fails to install on Windows, install it with conda first:

```bash
conda install -c conda-forge rasterio
```

Then run:

```bash
pip install -r requirements.txt
```

## 3. Dataset Format

The original dataset should be organized as follows:

```text
TreeHeightDataset/
├── RSImages/
├── mask/
├── heights/
├── train.txt
└── val.txt
```

### 3.1 Multispectral Images

The multispectral images should be placed in:

```text
TreeHeightDataset/RSImages/
```

Example:

```text
TreeHeightDataset/RSImages/sample_001.tif
```

The default image format is:

```text
.tif
```

The default number of multispectral channels is:

```text
8
```

For WorldView-3 multispectral imagery, a common band order is:

```text
Coastal, Blue, Green, Yellow, Red, RedEdge, NIR1, NIR2
```

### 3.2 Crown Masks

The crown masks should be placed in:

```text
TreeHeightDataset/mask/
```

Example:

```text
TreeHeightDataset/mask/sample_001.png
```

The mask should be a binary mask:

```text
0 = background
1 = crown
```

If the original mask uses 255 as foreground, the augmentation script will convert all non-zero mask values to 1.

### 3.3 Height Maps

The processed tree-height maps should be placed in:

```text
TreeHeightDataset/heights/
```

Example:

```text
TreeHeightDataset/heights/sample_001_processed.tif
```

The height map should be a single-band floating-point GeoTIFF. Invalid or NoData values will be converted to invalid pixels and excluded from loss and metric calculations.

### 3.4 Train and Validation Splits

The split files should be:

```text
TreeHeightDataset/train.txt
TreeHeightDataset/val.txt
```

Each line contains the sample name without extension:

```text
sample_001
sample_002
sample_003
```

The corresponding files should be:

```text
RSImages/sample_001.tif
mask/sample_001.png
heights/sample_001_processed.tif
```

## 4. Data Augmentation

The augmentation script is:

```text
augment_ms_mask_height_dataset.py
```

Run:

```bash
python augment_ms_mask_height_dataset.py
```

Input directory:

```text
TreeHeightDataset/
```

Output directory:

```text
TreeHeightDataset_AUG/
```

The script crops large images into 640 × 640 patches and applies discrete geometric augmentation to the training split.

Output structure:

```text
TreeHeightDataset_AUG/
├── RSImages/
├── mask/
├── heights/
├── train.txt
└── val.txt
```

The augmentation methods include:

```text
hflip   horizontal flip
vflip   vertical flip
rot90   90-degree rotation
rot180  180-degree rotation
rot270  270-degree rotation
```

These operations are implemented using array indexing only. No arbitrary rotation, scaling, shearing, interpolation, or pixel-value perturbation is applied.

The output masks are saved as:

```text
0 = background
1 = crown
```

The output height NoData value is:

```text
-9999.0
```

## 5. Model

The main model file is:

```text
nets/focal_svit_crown_segmentation.py
```

The main model class is:

```python
FocalSVITCrownSegmentationNet
```

Recommended import:

```python
from nets.focal_svit_crown_segmentation import FocalSVITCrownSegmentationNet
```

Model initialization example:

```python
model = FocalSVITCrownSegmentationNet(
    num_classes=1,
    pretrained=False,
    backbone="focal_s",
    in_channels=9,
    use_svit=True,
    svit_on_f16=True,
    svit_on_f32=True,
    svit_stoken_size=(4, 4),
    svit_heads=8,
    svit_n_iter=1,
)
```

The model input is:

```text
8 multispectral bands + 1 crown mask = 9 channels
```

The model output is:

```text
1-channel normalized tree-height map
```

The BIE module has been removed. Therefore, `use_bie` is no longer required in the training or validation scripts.

## 6. Training

The training script is:

```text
train_ms_mask_reg.py
```

Run:

```bash
python train_ms_mask_reg.py
```

Important training configuration:

```python
dataset_path = "TreeHeightDataset_AUG"

ms_dir = "RSImages"
mask_dir = "mask"
height_dir = "heights"

ms_suffix = ".tif"
mask_suffix = ".png"
height_suffix = "_processed.tif"

backbone = "focal_s"
in_channels = 9
out_channels = 1

use_svit = True
svit_on_f16 = True
svit_on_f32 = True
```

The default input size is:

```python
input_shape = [640, 640]
```

The default training strategy is:

```text
Freeze epochs: 50
Total epochs: 300
Optimizer: Adam
Initial learning rate: 1e-5
Minimum learning rate: 1e-7
Loss: combined regression loss
```

The training script automatically calculates the valid tree-height range from the training and validation height maps. The range is saved to:

```text
height_range.txt
```

Example output path:

```text
logs_reg_ms_mask_focalsvit_no_bie/loss_xxxxxxxx/height_range.txt
```

This file is required for validation and prediction.

## 7. Height Normalization

During training, height values are normalized as:

```text
height_norm = (height - height_min) / (height_max - height_min)
```

Invalid height pixels are set to 0 after normalization and excluded from loss calculation.

The valid loss mask is defined as:

```text
loss_mask = crown_mask × valid_height_mask
```

Therefore, the model is optimized mainly within valid crown regions.

## 8. Validation and Prediction

The validation script is:

```text
validate_ms_mask_treewise.py
```

Run:

```bash
python validate_ms_mask_treewise.py
```

Important validation configuration:

```python
MODEL_PATH = r"model_data/best_epoch_xxx.pth"
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie"
DATASET_PATH = "TreeHeightDataset_AUG"
VAL_TXT = os.path.join(DATASET_PATH, "val.txt")
```

`HEIGHT_RANGE_PATH` can be either a specific file:

```python
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie/loss_xxxxxxxx/height_range.txt"
```

or a log root directory:

```python
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie"
```

If a directory is provided, the validation script automatically searches for the latest `height_range.txt`.

## 9. Tree-Wise Height Extraction

The validation script performs tree-wise evaluation based on connected crown regions.

For each connected crown region:

```text
true tree height = maximum true height within the crown region
predicted tree height = maximum predicted height within the crown region
```

Very small crown regions can be filtered by:

```python
MIN_TREE_PIXELS = 5
```

The model prediction can optionally be clipped to the training height range:

```python
CLIP_PRED_TO_RANGE = True
```

## 10. Evaluation Metrics

The validation script reports tree-wise regression metrics:

```text
MAE
RMSE
Bias
R²
Pearson r²
```

Definitions:

```text
MAE  = mean absolute error
RMSE = root mean square error
Bias = mean prediction error
R²   = coefficient of determination
r²   = squared Pearson correlation coefficient
```

The main evaluation unit is an individual tree crown, not a single pixel.

## 11. Output Files

The default validation output directory is:

```text
validate_results_ms_treewise_focalsvit_no_bie/
```

Output structure:

```text
validate_results_ms_treewise_focalsvit_no_bie/
├── pred_tif/
│   ├── sample_001_pred_treeMaxFilled.tif
│   └── ...
├── visuals/
│   ├── sample_001_visual.png
│   └── ...
├── tables/
│   └── tree-level detailed results
├── tree-wise true-predicted height table
└── metrics_summary.txt
```

The `pred_tif` folder stores tree-wise maximum-height-filled prediction maps.

The `visuals` folder stores visualization results including:

```text
input image
true tree height
predicted tree height
prediction error
```

The `metrics_summary.txt` file stores the overall tree-wise metrics.

The Excel files store tree-level true height, predicted height, and prediction error.

## 12. Notes on Weight Loading

The model loads weights by matching parameter names and tensor shapes.

If the input channel number changes, such as from 3 to 9, the first convolution layer may not be loaded. This is normal.

If the BIE module was present in an older checkpoint, BIE-related parameters will be skipped because BIE has been removed in the current version. This is also normal.

If many encoder parameters cannot be loaded, check whether the backbone name and network structure are consistent.

## 13. Recommended Running Order

A typical workflow is:

```bash
python augment_ms_mask_height_dataset.py
python train_ms_mask_reg.py
python validate_ms_mask_treewise.py
```

Corresponding stages:

```text
1. Generate augmented training and validation patches
2. Train the crown-guided tree-height regression model
3. Validate tree-wise height estimation accuracy
```

## 14. Application

This branch is designed for crown-guided tree-height estimation using high-resolution multispectral imagery. It can support individual tree-level height mapping, crown-level ecological analysis, and downstream forest or desert vegetation structure assessment.
