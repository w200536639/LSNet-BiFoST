# Focal-SVIT Multispectral Crown-Guided Tree-Height Regression

本项目用于基于高分辨率多光谱遥感影像、树冠掩膜和树高栅格图进行树高回归建模。输入为 8 波段高分 / WorldView-3 / GF 类多光谱影像与树冠 mask 拼接形成的 9 通道数据，输出为单通道树高预测图。模型采用 FocalNet 编码器与 SVIT 高层注意力模块，BIE 模块已删除。

## 1. Project Structure

推荐项目目录如下：

```text
project_root/
├── nets/
│   ├── focal_svit_crown_segmentation.py
│   └── unet_training.py
├── utils/
│   ├── callbacks.py
│   ├── utils.py
│   └── utils_fit.py
├── TreeHeightDataset/
│   ├── RSImages/
│   │   ├── xxx.tif
│   │   └── ...
│   ├── mask/
│   │   ├── xxx.png
│   │   └── ...
│   ├── heights/
│   │   ├── xxx_processed.tif
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

Python 版本建议使用：

```text
Python 3.10
```

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 内容如下：

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

如果 Windows 下 `rasterio` 安装失败，建议使用 conda 安装：

```bash
conda install -c conda-forge rasterio
```

然后再执行：

```bash
pip install -r requirements.txt
```

## 3. Dataset Format

原始数据集目录应为：

```text
TreeHeightDataset/
├── RSImages/
├── mask/
├── heights/
├── train.txt
└── val.txt
```

其中：

```text
RSImages/   存放 8 波段多光谱影像，格式为 .tif
mask/       存放树冠二值掩膜，格式为 .png
heights/    存放树高图，格式为 *_processed.tif
train.txt   训练集样本名
val.txt     验证集样本名
```

样本命名需要保持一致。例如：

```text
RSImages/sample_001.tif
mask/sample_001.png
heights/sample_001_processed.tif
```

对应的 `train.txt` 或 `val.txt` 中写：

```text
sample_001
```

也可以写：

```text
sample_001.tif
```

代码会自动去除扩展名。

## 4. Data Augmentation

数据增强脚本为：

```text
augment_ms_mask_height_dataset.py
```

该脚本会将原始数据裁剪为 640 × 640 的 patch，并进行离散几何增强。

运行：

```bash
python augment_ms_mask_height_dataset.py
```

输入目录：

```text
TreeHeightDataset/
```

输出目录：

```text
TreeHeightDataset_AUG/
```

增强方式包括：

```text
hflip   水平翻转
vflip   垂直翻转
rot90   旋转 90 度
rot180  旋转 180 度
rot270  旋转 270 度
```

这些增强均为纯索引操作，不进行任意角度旋转、缩放、切变或像元值扰动，因此不会引入插值误差。

输出 mask 保存为：

```text
0 = background
1 = crown
```

不是 255。

## 5. Model

模型文件为：

```text
nets/focal_svit_crown_segmentation.py
```

主类名：

```python
FocalSVITCrownSegmentationNet
```

当前模型结构：

```text
Input:  9 channels
        8 multispectral bands + 1 crown mask

Encoder:
        FocalNet encoder

High-level attention:
        SVIT on f16 and f32

Decoder:
        standard upsampling and skip-connection fusion

Output:
        1-channel normalized tree-height map
```

当前版本已删除 BIE 模块。

## 6. Training

训练脚本为：

```text
train_ms_mask_reg.py
```

运行：

```bash
python train_ms_mask_reg.py
```

训练脚本中需要确认以下配置：

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

训练输入为：

```text
8-band multispectral image + crown mask
```

即：

```text
8 + 1 = 9 channels
```

训练输出为：

```text
1-channel normalized height map
```

训练过程中会自动统计树高范围，并在日志目录中保存：

```text
height_range.txt
```

例如：

```text
logs_reg_ms_mask_focalsvit_no_bie/loss_xxxxxxxx/height_range.txt
```

该文件在预测验证时需要使用。

## 7. Validation and Prediction

预测与逐棵树验证脚本为：

```text
validate_ms_mask_treewise.py
```

运行：

```bash
python validate_ms_mask_treewise.py
```

需要确认以下配置：

```python
MODEL_PATH = r"model_data/best_epoch_xxx.pth"
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie"
DATASET_PATH = "TreeHeightDataset_AUG"
VAL_TXT = os.path.join(DATASET_PATH, "val.txt")
```

`HEIGHT_RANGE_PATH` 可以填写具体文件：

```python
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie/loss_xxxxxxxx/height_range.txt"
```

也可以填写日志根目录：

```python
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie"
```

如果填写目录，程序会自动查找最新的 `height_range.txt`。

## 8. Evaluation Metrics

验证脚本会基于树冠 mask 连通域进行逐棵树高度提取。

每棵树的高度定义为：

```text
tree height = maximum canopy height within one connected crown region
```

即每个树冠连通域中：

```text
真实高度 = 该树冠区域内真实树高最大值
预测高度 = 该树冠区域内预测树高最大值
```

最终输出以下指标：

```text
MAE
RMSE
Bias
R²
Pearson r²
```

其中：

```text
MAE  = mean absolute error
RMSE = root mean square error
Bias = mean prediction error
R²   = coefficient of determination
r²   = squared Pearson correlation coefficient
```

## 9. Output Files

验证结果默认保存到：

```text
validate_results_ms_treewise_focalsvit_no_bie/
```

输出内容包括：

```text
pred_tif/
    每张图像的逐树最大高度填充预测图

visuals/
    输入图像、真实树高、预测树高和误差图

tables/
    逐图逐棵树详细结果表

逐棵树真实值-预测值.xlsx
metrics_summary.txt
```

其中：

```text
metrics_summary.txt
```

保存总体精度指标。

```text
逐棵树真实值-预测值.xlsx
```

保存每棵树的真实高度、预测高度和误差。

## 10. Notes

1. 训练、验证和增强脚本中的数据目录必须保持一致。
2. 增强后的 mask 已保存为 0/1，不应再按 255 作为前景处理。
3. 树高图中的 NoData 会被转换为无效像元，不参与损失和指标计算。
4. 当前网络已删除 BIE，因此旧权重中的 BIE 相关参数无法加载是正常现象。
5. 如果输入通道数从 3 变为 9，第一层卷积权重无法完全加载也是正常现象。
6. 如果大量 encoder 权重无法加载，需要检查模型结构、backbone 名称和权重文件是否一致。

## 11. Recommended Running Order

完整流程如下：

```bash
python augment_ms_mask_height_dataset.py
python train_ms_mask_reg.py
python validate_ms_mask_treewise.py
```

对应顺序为：

```text
1. 数据增强
2. 模型训练
3. 逐棵树预测验证