import os
import glob
import warnings
import logging
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure
import tifffile
from PIL import Image

from nets.focal_svit_crown_segmentation import FocalSVITCrownSegmentationNet
from utils.utils import seed_everything, worker_init_fn


# ============================================================
# Suppress tifffile NoData warnings
# ============================================================
logging.getLogger("tifffile").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=r".*GDAL_NODATA.*")
warnings.filterwarnings("ignore", message=r".*not castable.*")


# ============================================================
# Configuration
# ============================================================
MODEL_PATH = r"model_data/best_epoch_146_treeR2_0.7793_treeRMSE_0.6106.pth"

# 可以是具体文件，也可以是日志目录。
# 如果填写目录，代码会自动递归查找最新的 height_range.txt。
HEIGHT_RANGE_PATH = r"logs_reg_ms_mask_focalsvit_no_bie"

DATASET_PATH = "TreeHeightDataset"
VAL_TXT = os.path.join(DATASET_PATH, "val.txt")

MS_DIR = "RSImages"
MASK_DIR = "mask"
HEIGHT_DIR = "heights"

MS_SUFFIX = ".tif"
MASK_SUFFIX = ".png"
HEIGHT_SUFFIX = "_processed.tif"

MS_IN_CHANNELS = 8
MS_NORM_MODE = "percentile"          # percentile / max / none
MS_PERCENTILE = (2, 98)

BACKBONE = "focal_s"
IN_CHANNELS = 9                      # 8 bands + crown mask

USE_SVIT = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True
SVIT_STOKEN_SIZE = (4, 4)
SVIT_HEADS = 8
SVIT_N_ITER = 1

INPUT_SHAPE = (640, 640)
BATCH_SIZE = 1
NUM_WORKERS = 0

DEVICE = "cuda"
SEED = 11

SAVE_DIR = "validate_results_ms_treewise_focalsvit_no_bie"

MIN_TREE_PIXELS = 5

CLIP_PRED_TO_RANGE = True

NODATA_ABS_THRESHOLD = 1.0e20


# ============================================================
# Metrics
# ============================================================
def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.size < 2:
        return np.nan

    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))

    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot


def pearson_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if y_true.size < 2:
        return np.nan

    vt = y_true - np.mean(y_true)
    vp = y_pred - np.mean(y_pred)

    denom = np.sqrt(np.sum(vt * vt)) * np.sqrt(np.sum(vp * vp))

    if denom <= 0:
        return np.nan

    r = float(np.sum(vt * vp) / denom)

    return r * r


def compute_basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    error = y_pred - y_true

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))
    r2 = float(r2_score(y_true, y_pred))
    pr2 = float(pearson_r2(y_true, y_pred))

    return mae, rmse, bias, r2, pr2


# ============================================================
# Safe IO
# ============================================================
def safe_name(name):
    name = str(name)
    name = name.replace("（", "(").replace("）", ")")

    illegal_chars = '<>:"/\\|?*'

    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


def read_height_tif_safely(height_path):
    if not os.path.exists(height_path):
        raise FileNotFoundError(f"[Height tif not found] {height_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        height = tifffile.imread(height_path)

    height = np.asarray(height)

    if height.ndim == 3:
        if height.shape[0] == 1:
            height = height[0]
        elif height.shape[-1] == 1:
            height = height[..., 0]
        else:
            height = height[0]

    height = height.astype(np.float32, copy=False)

    height[np.abs(height) >= NODATA_ABS_THRESHOLD] = np.nan
    height[~np.isfinite(height)] = np.nan

    return height.astype(np.float32, copy=False)


def load_height_range(height_range_path):
    """
    height_range_path can be:
        1. a file path ending with height_range.txt
        2. a directory containing height_range.txt
        3. a log root directory containing multiple loss_xxx/height_range.txt
    """
    candidate_file = None

    if os.path.isfile(height_range_path):
        candidate_file = height_range_path

    elif os.path.isdir(height_range_path):
        files = glob.glob(
            os.path.join(height_range_path, "**", "height_range.txt"),
            recursive=True,
        )

        if len(files) == 0:
            raise FileNotFoundError(
                f"在目录中没有找到 height_range.txt: {height_range_path}"
            )

        files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
        candidate_file = files[0]

    else:
        raise FileNotFoundError(f"未找到 height range 路径: {height_range_path}")

    print(f"使用 height_range 文件: {candidate_file}")

    height_min = None
    height_max = None

    with open(candidate_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("height_min"):
                height_min = float(line.split("=")[-1])

            elif line.startswith("height_max"):
                height_max = float(line.split("=")[-1])

    if height_min is None or height_max is None:
        raise ValueError(f"height_range.txt 内容不完整: {candidate_file}")

    print(f"高度范围: {height_min:.4f} ~ {height_max:.4f} m")

    return float(height_min), float(height_max), candidate_file


def read_val_lines(val_txt):
    if not os.path.exists(val_txt):
        raise FileNotFoundError(f"未找到 val.txt: {val_txt}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(val_txt, "r", encoding=encoding) as file:
                return [line.strip() for line in file if line.strip()]
        except UnicodeDecodeError:
            continue

    with open(val_txt, "r", encoding="utf-8", errors="ignore") as file:
        return [line.strip() for line in file if line.strip()]


# ============================================================
# Tree-wise extraction
# ============================================================
def fill_tree_canopy_with_max(pred_np, true_np, valid_mask_np, min_pixels=5):
    """
    Connected crown regions are extracted from valid_mask_np.

    For each tree region:
        true tree height = max true height in the region
        pred tree height = max predicted height in the region

    Returns:
        pred_filled
        true_filled
        tree_true
        tree_pred
        tree_records
    """
    pred_np = pred_np.astype(np.float32)
    true_np = true_np.astype(np.float32)
    valid_mask_np = valid_mask_np.astype(np.bool_)

    valid = (
        valid_mask_np
        & np.isfinite(true_np)
        & np.isfinite(pred_np)
        & (true_np > 0.0)
    )

    pred_valid = np.where(valid, pred_np, np.nan)
    true_valid = np.where(valid, true_np, np.nan)

    labeled = measure.label(valid, connectivity=2)

    tree_pairs = []
    tree_records = []

    pred_filled = np.full_like(pred_valid, np.nan, dtype=np.float32)
    true_filled = np.full_like(true_valid, np.nan, dtype=np.float32)

    out_tree_id = 0

    for region_id in range(1, int(labeled.max()) + 1):
        region = labeled == region_id
        pixel_count = int(np.sum(region))

        if pixel_count < int(min_pixels):
            continue

        region_pred = pred_valid[region]
        region_true = true_valid[region]

        region_pred = region_pred[np.isfinite(region_pred)]
        region_true = region_true[np.isfinite(region_true)]

        if region_pred.size == 0 or region_true.size == 0:
            continue

        max_pred = float(np.nanmax(region_pred))
        max_true = float(np.nanmax(region_true))

        if not np.isfinite(max_true) or max_true <= 0.0:
            continue

        out_tree_id += 1

        tree_pairs.append((max_true, max_pred))

        pred_filled[region] = max_pred
        true_filled[region] = max_true

        ys, xs = np.where(region)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))

        tree_records.append(
            {
                "tree_local_id": out_tree_id,
                "region_id": int(region_id),
                "pixel_count": int(pixel_count),
                "center_y": cy,
                "center_x": cx,
                "true_height_m": max_true,
                "pred_height_m": max_pred,
                "error_m": max_pred - max_true,
            }
        )

    if tree_pairs:
        tree_true = np.array([pair[0] for pair in tree_pairs], dtype=np.float32)
        tree_pred = np.array([pair[1] for pair in tree_pairs], dtype=np.float32)
    else:
        tree_true = np.array([], dtype=np.float32)
        tree_pred = np.array([], dtype=np.float32)

    return pred_filled, true_filled, tree_true, tree_pred, tree_records


# ============================================================
# Dataset
# ============================================================
class MsMaskValDataset(torch.utils.data.Dataset):
    """
    Validation dataset.

    Input:
        image_9ch: 8-band image + crown mask

    Output:
        height_norm: normalized height map
        loss_mask: crown mask × valid height mask
    """

    def __init__(
        self,
        annotation_lines,
        input_shape=(640, 640),
        dataset_path="TreeHeightDataset",
        ms_dir="RSImages",
        mask_dir="mask",
        height_dir="heights",
        ms_suffix=".tif",
        mask_suffix=".png",
        height_suffix="_processed.tif",
        height_min=0.0,
        height_max=6.0,
        normalize_mode="minmax",
        ms_norm_mode="percentile",
        ms_in_channels=8,
        ms_percentile=(2, 98),
    ):
        super().__init__()

        self.lines = list(annotation_lines)
        self.input_shape = tuple(input_shape)
        self.dataset_path = dataset_path

        self.ms_dir = ms_dir
        self.mask_dir = mask_dir
        self.height_dir = height_dir

        self.ms_suffix = ms_suffix
        self.mask_suffix = mask_suffix
        self.height_suffix = height_suffix

        self.height_min = float(height_min)
        self.height_max = float(height_max)
        self.normalize_mode = str(normalize_mode).lower().strip()

        self.ms_norm_mode = str(ms_norm_mode).lower().strip()
        self.ms_in_channels = int(ms_in_channels)
        self.ms_percentile = tuple(ms_percentile)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        raw = self.lines[idx].strip()
        prefix = os.path.splitext(raw)[0]

        ms_path = os.path.join(
            self.dataset_path,
            self.ms_dir,
            prefix + self.ms_suffix,
        )
        mask_path = os.path.join(
            self.dataset_path,
            self.mask_dir,
            prefix + self.mask_suffix,
        )
        height_path = os.path.join(
            self.dataset_path,
            self.height_dir,
            prefix + self.height_suffix,
        )

        if not os.path.exists(ms_path):
            raise FileNotFoundError(f"[MS tif not found] {ms_path}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"[Mask not found] {mask_path}")

        if not os.path.exists(height_path):
            raise FileNotFoundError(f"[Height tif not found] {height_path}")

        ms_img = self.read_ms_hwc(ms_path, out_channels=self.ms_in_channels)

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        height = read_height_tif_safely(height_path)

        common_h = min(ms_img.shape[0], mask.shape[0], height.shape[0])
        common_w = min(ms_img.shape[1], mask.shape[1], height.shape[1])

        ms_img = ms_img[:common_h, :common_w, :]
        mask = mask[:common_h, :common_w]
        height = height[:common_h, :common_w]

        ms_img, mask, height = self.center_crop(
            ms_img,
            mask,
            height,
            self.input_shape,
        )

        valid_height_mask = (
            np.isfinite(height)
            & (height >= self.height_min)
            & (height <= self.height_max)
        ).astype(np.float32)

        loss_mask = mask * valid_height_mask

        height_norm = self.normalize_height(height)

        mask_channel = np.expand_dims(mask, axis=-1)
        image_9ch = np.concatenate([ms_img, mask_channel], axis=-1)
        image_9ch = np.transpose(image_9ch, (2, 0, 1)).astype(np.float32)

        height_norm = np.expand_dims(height_norm, axis=0).astype(np.float32)
        loss_mask = np.expand_dims(loss_mask, axis=0).astype(np.float32)

        return (
            torch.from_numpy(image_9ch).float(),
            torch.from_numpy(height_norm).float(),
            torch.from_numpy(loss_mask).float(),
            prefix,
        )

    def center_crop(self, ms, mask, height, target_size):
        target_h, target_w = target_size
        image_h, image_w, _ = ms.shape

        if image_h < target_h or image_w < target_w:
            raise ValueError(
                f"图像尺寸 {(image_h, image_w)} 小于裁剪尺寸 {(target_h, target_w)}。"
            )

        top = max(0, (image_h - target_h) // 2)
        left = max(0, (image_w - target_w) // 2)

        ms = ms[top: top + target_h, left: left + target_w, :]
        mask = mask[top: top + target_h, left: left + target_w]
        height = height[top: top + target_h, left: left + target_w]

        return ms, mask, height

    def normalize_height(self, height_np):
        valid_mask = (
            np.isfinite(height_np)
            & (height_np >= self.height_min)
            & (height_np <= self.height_max)
        )

        height = height_np.copy()

        if not np.any(valid_mask):
            return np.zeros_like(height, dtype=np.float32)

        if self.normalize_mode == "zscore":
            mean_value = np.nanmean(height[valid_mask])
            std_value = np.nanstd(height[valid_mask]) + 1e-6
            height = (height - mean_value) / std_value
        else:
            height = (height - self.height_min) / (
                self.height_max - self.height_min + 1e-6
            )
            height = np.clip(height, 0.0, 1.0)

        height[~valid_mask] = 0.0

        return height.astype(np.float32)

    def read_ms_hwc(self, path, out_channels=8):
        arr = tifffile.imread(path)

        if arr.ndim == 2:
            arr = arr[None, ...]

        elif arr.ndim == 3:
            if arr.shape[0] in [1, 3, 4, 8, 9, 16] and arr.shape[1] > 32 and arr.shape[2] > 32:
                pass
            else:
                arr = np.transpose(arr, (2, 0, 1))

        else:
            raise ValueError(f"Unsupported tif shape: {arr.shape}")

        arr = arr.astype(np.float32)

        channels, height, width = arr.shape

        if channels >= out_channels:
            arr = arr[:out_channels]
        else:
            pad = np.zeros(
                (out_channels - channels, height, width),
                dtype=np.float32,
            )
            arr = np.concatenate([arr, pad], axis=0)

        if self.ms_norm_mode == "none":
            pass

        elif self.ms_norm_mode == "max":
            max_value = float(np.max(arr)) + 1e-6
            arr = arr / max_value

        elif self.ms_norm_mode == "percentile":
            low_percentile, high_percentile = self.ms_percentile

            output = np.zeros_like(arr, dtype=np.float32)

            for band_id in range(arr.shape[0]):
                band = arr[band_id]

                low = np.percentile(band, low_percentile)
                high = np.percentile(band, high_percentile)

                if high - low < 1e-6:
                    output[band_id] = 0.0
                else:
                    output[band_id] = (band - low) / (high - low)

            arr = np.clip(output, 0.0, 1.0)

        else:
            raise ValueError(f"Unsupported MS_NORM_MODE: {self.ms_norm_mode}")

        return np.transpose(arr, (1, 2, 0)).astype(np.float32)


def ms_mask_collate(batch):
    images, heights, masks, names = zip(*batch)
    return torch.stack(images), torch.stack(heights), torch.stack(masks), list(names)


# ============================================================
# Model
# ============================================================
def build_model(device):
    model = FocalSVITCrownSegmentationNet(
        num_classes=1,
        backbone=BACKBONE,
        in_channels=IN_CHANNELS,
        use_svit=USE_SVIT,
        svit_on_f16=SVIT_ON_F16,
        svit_on_f32=SVIT_ON_F32,
        svit_stoken_size=SVIT_STOKEN_SIZE,
        svit_heads=SVIT_HEADS,
        svit_n_iter=SVIT_N_ITER,
    )

    if hasattr(model, "get_model_profile"):
        print("[Model Profile]", model.get_model_profile())

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"未找到权重: {MODEL_PATH}")

    print(f"加载权重: {MODEL_PATH}")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state = {}

    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        clean_state[new_key] = value

    model_state = model.state_dict()

    loaded = 0
    skipped = 0
    skipped_keys = []

    for key, value in clean_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
        else:
            skipped += 1
            skipped_keys.append(key)

    model.load_state_dict(model_state, strict=False)

    print(f"加载完成: loaded={loaded}, skipped={skipped}")

    if skipped_keys:
        print("Skipped keys examples:", skipped_keys[:30])

    model = model.to(device)
    model.eval()

    return model


# ============================================================
# Visualization
# ============================================================
def make_rgb_visual_from_input(img_tensor_chw):
    """
    Use the first three bands only for visualization.
    """
    img_np = img_tensor_chw[:3].astype(np.float32)
    img_vis = np.transpose(img_np, (1, 2, 0))

    vmin = float(np.nanmin(img_vis))
    vmax = float(np.nanmax(img_vis))

    if vmax > vmin:
        img_vis = (img_vis - vmin) / (vmax - vmin + 1e-6)

    img_vis = np.clip(img_vis, 0.0, 1.0)

    return img_vis


def visualize_one(
    name,
    image_vis,
    true_filled,
    pred_filled,
    diff_map,
    height_min,
    height_max,
    out_dir,
):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(image_vis)
    axes[0].set_title("Input")
    axes[0].axis("off")

    im1 = axes[1].imshow(
        true_filled,
        cmap="viridis_r",
        vmin=height_min,
        vmax=height_max,
    )
    axes[1].set_title("True tree max")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        pred_filled,
        cmap="viridis_r",
        vmin=height_min,
        vmax=height_max,
    )
    axes[2].set_title("Pred tree max")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(
        diff_map,
        cmap="coolwarm_r",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[3].set_title("Error: Pred - True")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{safe_name(name)}_visual.png")
    plt.savefig(save_path, dpi=250)
    plt.close(fig)


# ============================================================
# Main validation
# ============================================================
def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    height_min, height_max, height_range_file = load_height_range(HEIGHT_RANGE_PATH)

    val_lines = read_val_lines(VAL_TXT)
    print(f"验证样本数: {len(val_lines)}")

    val_dataset = MsMaskValDataset(
        annotation_lines=val_lines,
        input_shape=INPUT_SHAPE,
        dataset_path=DATASET_PATH,
        ms_dir=MS_DIR,
        mask_dir=MASK_DIR,
        height_dir=HEIGHT_DIR,
        ms_suffix=MS_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        height_suffix=HEIGHT_SUFFIX,
        height_min=height_min,
        height_max=height_max,
        normalize_mode="minmax",
        ms_norm_mode=MS_NORM_MODE,
        ms_in_channels=MS_IN_CHANNELS,
        ms_percentile=MS_PERCENTILE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ms_mask_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=SEED),
    )

    model = build_model(device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    tif_dir = os.path.join(SAVE_DIR, "pred_tif")
    vis_dir = os.path.join(SAVE_DIR, "visuals")
    table_dir = os.path.join(SAVE_DIR, "tables")

    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    all_pairs = []
    all_records = []

    scale = height_max - height_min

    for step, (images, true_heights, valid_masks, names) in enumerate(
        tqdm(val_loader, desc="Validating")
    ):
        images = images.to(device).float()

        with torch.no_grad():
            pred_norm = model(images)

        if CLIP_PRED_TO_RANGE:
            pred_norm = torch.clamp(pred_norm, 0.0, 1.0)

        pred_np = pred_norm.detach().cpu().numpy().astype(np.float32) * scale + height_min
        true_np = true_heights.detach().cpu().numpy().astype(np.float32) * scale + height_min
        mask_np = valid_masks.detach().cpu().numpy().astype(np.bool_)

        images_np = images.detach().cpu().numpy().astype(np.float32)

        for batch_id in range(pred_np.shape[0]):
            name = names[batch_id]

            pred_filled, true_filled, tree_true, tree_pred, tree_records = fill_tree_canopy_with_max(
                pred_np[batch_id, 0],
                true_np[batch_id, 0],
                mask_np[batch_id, 0],
                min_pixels=MIN_TREE_PIXELS,
            )

            if tree_true.size > 0:
                all_pairs.extend(list(zip(tree_true.tolist(), tree_pred.tolist())))

                for record in tree_records:
                    record["image_name"] = name
                    all_records.append(record)

            tifffile.imwrite(
                os.path.join(tif_dir, f"{safe_name(name)}_pred_treeMaxFilled.tif"),
                pred_filled.astype(np.float32),
            )

            valid = mask_np[batch_id, 0] & np.isfinite(true_filled) & np.isfinite(pred_filled)
            diff_map = np.where(valid, pred_filled - true_filled, np.nan).astype(np.float32)

            image_vis = make_rgb_visual_from_input(images_np[batch_id])

            visualize_one(
                name=name,
                image_vis=image_vis,
                true_filled=true_filled,
                pred_filled=pred_filled,
                diff_map=diff_map,
                height_min=height_min,
                height_max=height_max,
                out_dir=vis_dir,
            )

    if len(all_pairs) == 0:
        print("\n[!] 没有提取到有效树。请检查 mask、height 或 MIN_TREE_PIXELS。")
        return

    all_true = np.array([pair[0] for pair in all_pairs], dtype=np.float32)
    all_pred = np.array([pair[1] for pair in all_pairs], dtype=np.float32)

    mae, rmse, bias, r2, pr2 = compute_basic_metrics(all_true, all_pred)

    print("\n==== 逐棵树 max canopy height 总体指标 ====")
    print(f"trees = {len(all_pairs)}")
    print(f"MAE   = {mae:.4f} m")
    print(f"RMSE  = {rmse:.4f} m")
    print(f"Bias  = {bias:.4f} m")
    print(f"R²    = {r2:.4f}")
    print(f"r²    = {pr2:.4f}")

    df = pd.DataFrame(
        {
            "树序号": np.arange(1, len(all_pairs) + 1),
            "真实高度(m)": np.round(all_true, 4),
            "预测高度(m)": np.round(all_pred, 4),
            "误差(预测-真实,m)": np.round(all_pred - all_true, 4),
        }
    )

    excel_path = os.path.join(SAVE_DIR, "逐棵树真实值-预测值.xlsx")
    df.to_excel(excel_path, index=False, engine="openpyxl")

    detail_df = pd.DataFrame(all_records)
    detail_path = os.path.join(table_dir, "逐图逐棵树详细结果.xlsx")
    detail_df.to_excel(detail_path, index=False, engine="openpyxl")

    metrics_path = os.path.join(SAVE_DIR, "metrics_summary.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("network=FocalSVITCrownSegmentationNet\n")
        file.write("use_bie=False\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"HEIGHT_RANGE_FILE={height_range_file}\n")
        file.write(f"DATASET_PATH={DATASET_PATH}\n")
        file.write(f"VAL_TXT={VAL_TXT}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"IN_CHANNELS={IN_CHANNELS}\n")
        file.write(f"USE_SVIT={USE_SVIT}\n")
        file.write(f"SVIT_ON_F16={SVIT_ON_F16}\n")
        file.write(f"SVIT_ON_F32={SVIT_ON_F32}\n")
        file.write(f"MIN_TREE_PIXELS={MIN_TREE_PIXELS}\n")
        file.write(f"CLIP_PRED_TO_RANGE={CLIP_PRED_TO_RANGE}\n")
        file.write(f"height_min={height_min:.6f}\n")
        file.write(f"height_max={height_max:.6f}\n")
        file.write(f"trees={len(all_pairs)}\n")
        file.write(f"MAE={mae:.6f}\n")
        file.write(f"RMSE={rmse:.6f}\n")
        file.write(f"Bias={bias:.6f}\n")
        file.write(f"R2={r2:.6f}\n")
        file.write(f"Pearson_r2={pr2:.6f}\n")

    print(f"\n[√] 输出完成: {SAVE_DIR}")
    print(f" - 预测 tif: {tif_dir}")
    print(f" - 可视化: {vis_dir}")
    print(f" - Excel: {excel_path}")
    print(f" - 逐树详细表: {detail_path}")
    print(f" - 指标汇总: {metrics_path}")


if __name__ == "__main__":
    seed_everything(SEED)
    main()