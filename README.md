# LSNet-BiFoST: Tree-Crown Segmentation and Crown-Guided Tree-Height Estimation

This repository provides the source code, model configuration files, and trained model weights for the LSNet-BiFoST framework. The framework was developed for individual tree-crown segmentation and crown-guided tree-height estimation of *Haloxylon ammodendron* using UAV RGB imagery and high-resolution remote-sensing imagery.

The repository contains four main branches:

```text
LSNet-BiFoST/
├── High-Resolution Imagery Tree-Crown Segmentation Branch/
├── High-Resolution Imagery Tree-Height Estimation Branch/
├── Tree-Crown Segmentation Branch Archive/
├── Tree-Height Estimation Branch Archive/
├── .gitattributes
├── .gitignore
└── README.md
```

## 1. Overview

LSNet-BiFoST is a dual-branch deep-learning framework designed for:

1. Individual tree-crown segmentation;
2. Crown-guided tree-height estimation;
3. Reducing background interference in tree-height regression by using crown masks as spatial guidance.

The UAV-based branch uses RGB image patches, tree-crown masks, and processed tree-height raster labels. The high-resolution imagery branch supports multispectral imagery and is designed for transfer or extended evaluation on high-resolution satellite imagery.

## 2. Repository Structure

### 2.1 High-Resolution Imagery Tree-Crown Segmentation Branch

```text
High-Resolution Imagery Tree-Crown Segmentation Branch/
```

This folder contains the code for high-resolution multispectral tree-crown segmentation.

Main files:

```text
train.py
predict.py
requirements.txt
README.md
nets/high_resolution_crown_segmentation.py
model_data/best_epoch_weights.pth
```

The input is a high-resolution multispectral image patch, and the output is a binary tree-crown segmentation mask.

### 2.2 High-Resolution Imagery Tree-Height Estimation Branch

```text
High-Resolution Imagery Tree-Height Estimation Branch/
```

This folder contains the code for high-resolution multispectral crown-guided tree-height regression.

Main files:

```text
train.py
predict.py
requirements.txt
README.md
nets/focal_svit_crown_segmentation.py
model_data/best_epoch_weights.pth
```

The input is composed of multispectral image bands and a crown-mask channel. The output is a single-channel tree-height map.

### 2.3 Tree-Crown Segmentation Branch Archive

```text
Tree-Crown Segmentation Branch Archive/
```

This folder contains the UAV RGB tree-crown segmentation branch used in the LSNet-BiFoST framework.

Main files:

```text
train.py
predict.py
evaluate_segmentation.py
explain_gradcam_segmentation.py
requirements.txt
readme.md
nets/lsnet_bifost_segmentation.py
model_data/best_epoch_weights.pth
```

This branch is used for semantic segmentation of individual *Haloxylon ammodendron* crowns from UAV RGB image patches.

### 2.4 Tree-Height Estimation Branch Archive

```text
Tree-Height Estimation Branch Archive/
```

This folder contains the UAV RGB crown-guided tree-height regression branch.

Main files:

```text
train.py
predict.py
evaluate_height_regression.py
explain_height_regression_gradcam.py
requirements.txt
README.md
nets/lsnet_bifost_height_regression.py
model_data/best_epoch_weights.pth
```

This branch uses UAV RGB images and crown masks to estimate individual tree height.

## 3. Environment

Recommended environment:

```text
Python 3.10
PyTorch
CUDA 11.8
cuDNN 8.5.0
```

Install dependencies inside each branch folder:

```bash
cd "Tree-Crown Segmentation Branch Archive"
pip install -r requirements.txt
```

or:

```bash
cd "Tree-Height Estimation Branch Archive"
pip install -r requirements.txt
```

A general dependency list includes:

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
tifffile
rasterio
pandas
openpyxl
```

If `rasterio` fails to install on Windows, it is recommended to install it with conda:

```bash
conda install -c conda-forge rasterio
```

## 4. Data Availability

The UAV RGB image patches, tree-crown masks, processed tree-height raster labels, and training/validation split files used in this study are publicly available from Zenodo:

```text
https://doi.org/10.5281/zenodo.20728941
```

The uploaded dataset contains two compressed archives:

```text
VOCdevkit.zip
TreeHeightDataset.zip
```

`VOCdevkit.zip` contains the tree-crown segmentation dataset organized in VOC format, including UAV RGB image patches, segmentation masks, and training/validation split files.

`TreeHeightDataset.zip` contains the tree-height regression dataset, including RGB image patches, crown masks, processed height maps, and training/validation split files.

The high-resolution satellite imagery used in the high-resolution imagery branch cannot be publicly shared because the original commercial satellite data are subject to data-use and redistribution restrictions. Therefore, the raw WorldView-3 and other restricted high-resolution remote-sensing images are not included in this repository. The preprocessing workflow, annotation protocol, model configuration files, trained weights, and redistributable derived labels are provided where permitted.

## 5. Trained Model Weights

The trained model weights are included in each branch under:

```text
model_data/best_epoch_weights.pth
```

The `.pth` files are managed using Git LFS. After cloning this repository, make sure Git LFS is installed and pull the model weights with:

```bash
git lfs install
git lfs pull
```

To check whether the weights are correctly tracked by Git LFS:

```bash
git lfs ls-files
```

## 6. UAV Tree-Crown Segmentation

Enter the UAV tree-crown segmentation branch:

```bash
cd "Tree-Crown Segmentation Branch Archive"
```

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

Evaluate segmentation performance:

```bash
python evaluate_segmentation.py
```

This branch reports segmentation metrics such as:

```text
Precision
Recall
F1-score
Overall IoU
mIoU
```

## 7. UAV Crown-Guided Tree-Height Estimation

Enter the UAV tree-height estimation branch:

```bash
cd "Tree-Height Estimation Branch Archive"
```

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

Evaluate tree-height estimation performance:

```bash
python evaluate_height_regression.py
```

This branch reports regression metrics such as:

```text
MAE
RMSE
Bias
R²
Pearson r²
```

## 8. High-Resolution Imagery Tree-Crown Segmentation

Enter the high-resolution imagery tree-crown segmentation branch:

```bash
cd "High-Resolution Imagery Tree-Crown Segmentation Branch"
```

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

This branch is designed for tree-crown segmentation using high-resolution multispectral remote-sensing imagery.

## 9. High-Resolution Imagery Tree-Height Estimation

Enter the high-resolution imagery tree-height estimation branch:

```bash
cd "High-Resolution Imagery Tree-Height Estimation Branch"
```

Train the model:

```bash
python train.py
```

Run prediction:

```bash
python predict.py
```

This branch uses high-resolution multispectral imagery and crown masks for crown-guided tree-height regression.

## 10. Dataset Organization

### 10.1 Tree-Crown Segmentation Dataset

The UAV tree-crown segmentation dataset follows the VOC-style structure:

```text
VOCdevkit/
└── VOC2007/
    ├── RSImages/
    ├── SegmentationClass/
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

### 10.2 Tree-Height Regression Dataset

The UAV tree-height regression dataset is organized as:

```text
TreeHeightDataset/
├── RSImages/
├── mask/
├── heights/
├── train.txt
└── val.txt
```

where:

```text
RSImages/   UAV RGB image patches
mask/       tree-crown masks
heights/    processed tree-height raster labels
train.txt   training split
val.txt     validation split
```

## 11. Notes

1. The UAV datasets are publicly available from Zenodo.
2. The raw high-resolution satellite imagery cannot be redistributed due to commercial data-use restrictions.
3. The trained `.pth` files are stored using Git LFS.
4. The model weights should be placed in the corresponding `model_data/` folder before prediction or evaluation.
5. Each branch contains its own `README.md` or `readme.md` with more detailed instructions.
6. The dataset and code are intended for research use in UAV remote sensing, individual tree-crown segmentation, tree-height estimation, semantic segmentation, deep learning, and ecological monitoring in arid environments.

## 12. Recommended Citation

If this repository is useful for your research, please cite the corresponding paper and the Zenodo dataset.

Dataset DOI:

```text
10.5281/zenodo.20728941
```

## 13. License

This repository is released for academic and research purposes. Users should follow the license terms of the repository and respect the data-use restrictions of commercial high-resolution satellite imagery.
