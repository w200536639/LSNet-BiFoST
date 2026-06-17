import os
from functools import partial
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image
from skimage import measure
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nets.lsnet_bifost_height_regression import LSNetBiFoSTHeightRegression
from utils.utils import seed_everything, worker_init_fn


# ============================================================
# Manual configuration
# ============================================================

# Checkpoint and height range
MODEL_PATH = r"logs_reg_rgb_mask/全开/best_epoch_066_valloss_0.0576_valmae_0.5085.pth"
HEIGHT_RANGE_PATH = r"logs_reg_rgb_mask/height_range.txt"

# Dataset
DATASET_PATH = "TreeHeightDataset"
RGB_DIR = "rgb"
MASK_DIR = "mask"
HEIGHT_DIR = "heights"

RGB_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"
HEIGHT_SUFFIX = "_processed.tif"

VAL_TXT = "val.txt"

# Model structure
BACKBONE = "focal_s"        # focal_t / focal_s / focal_b
IN_CHANNELS = 4             # RGB(3) + crown mask(1)
OUT_CHANNELS = 1

USE_SVIT = True
USE_BIE = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True
SVIT_STOKEN_SIZE = (4, 4)
SVIT_HEADS = 8
SVIT_N_ITER = 1

# Input and device
INPUT_SHAPE = (640, 640)
BATCH_SIZE = 1
NUM_WORKERS = 0
DEVICE = "cuda"             # cuda / cpu
SEED = 11

# Height normalization
NORMALIZE_MODE = "minmax"   # minmax is recommended for this script

# Tree-level height statistic
# Options:
#   max          : maximum height within each crown
#   p95          : 95th percentile height within each crown
#   p98          : 98th percentile height within each crown
#   top1_percent : mean of top 1% height pixels within each crown
HEIGHT_STATISTIC = "max"

MIN_TREE_AREA = 5
VALID_TRUE_HEIGHT_MIN = 0.0
VALID_TRUE_HEIGHT_MAX = 6.0

# Prediction post-processing
CLIP_PRED_TO_HEIGHT_RANGE = False

# Output
SAVE_DIR = "validate_results_height"
SAVE_PRED_TIF = True
SAVE_VISUALIZATION = True
SAVE_TRUE_PRED_MAPS = True
SAVE_EXCEL = True
SAVE_CSV = True


# ============================================================
# Dataset
# ============================================================
class RgbMaskHeightEvalDataset(Dataset):
    """
    Evaluation dataset for tree-height regression.

    Input:
        RGB image + crown mask -> 4-channel tensor.

    Target:
        height map and crown mask.

    The crop/pad strategy is consistent with the validation setting of the
    training dataset: center crop if the image is larger than INPUT_SHAPE,
    and pad if the image is smaller.
    """

    def __init__(
        self,
        annotation_lines: List[str],
        input_shape: Tuple[int, int] = (640, 640),
        dataset_path: str = "TreeHeightDataset",
        rgb_dir: str = "rgb",
        mask_dir: str = "mask",
        height_dir: str = "heights",
        rgb_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        height_suffix: str = "_processed.tif",
        height_min: float = 0.0,
        height_max: float = 6.0,
        normalize_mode: str = "minmax",
    ):
        super().__init__()

        self.annotation_lines = annotation_lines
        self.input_shape = input_shape

        self.dataset_path = dataset_path
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.height_dir = height_dir

        self.rgb_suffix = rgb_suffix
        self.mask_suffix = mask_suffix
        self.height_suffix = height_suffix

        self.height_min = height_min
        self.height_max = height_max
        self.normalize_mode = normalize_mode

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        prefix = self.annotation_lines[index].strip()

        rgb_path = os.path.join(
            self.dataset_path,
            self.rgb_dir,
            prefix + self.rgb_suffix,
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

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")
        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Height map not found: {height_path}")

        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        height = tifffile.imread(height_path).astype(np.float32)

        rgb, mask, height = self.center_crop_or_pad(
            rgb=rgb,
            mask=mask,
            height=height,
            target_size=self.input_shape,
        )

        height_norm = self.normalize_height(height)

        mask_channel_last = np.expand_dims(mask, axis=-1)
        input_4ch = np.concatenate([rgb, mask_channel_last], axis=-1)
        input_4ch = np.transpose(input_4ch, (2, 0, 1))

        height_norm = np.expand_dims(height_norm, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.from_numpy(input_4ch).float(),
            torch.from_numpy(height_norm).float(),
            torch.from_numpy(mask).float(),
            prefix,
        )

    @staticmethod
    def pad_to_size(rgb, mask, height, target_size):
        target_h, target_w = target_size
        h, w, _ = rgb.shape

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_h == 0 and pad_w == 0:
            return rgb, mask, height

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        rgb = np.pad(
            rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=128.0 / 255.0,
        )
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=0.0,
        )
        height = np.pad(
            height,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=np.nan,
        )

        return rgb, mask, height

    def center_crop_or_pad(self, rgb, mask, height, target_size):
        target_h, target_w = target_size

        rgb, mask, height = self.pad_to_size(
            rgb=rgb,
            mask=mask,
            height=height,
            target_size=target_size,
        )

        h, w, _ = rgb.shape

        top = max(0, (h - target_h) // 2)
        left = max(0, (w - target_w) // 2)

        rgb = rgb[top: top + target_h, left: left + target_w, :]
        mask = mask[top: top + target_h, left: left + target_w]
        height = height[top: top + target_h, left: left + target_w]

        return rgb, mask, height

    def normalize_height(self, height_map):
        finite_mask = np.isfinite(height_map)
        height = height_map.copy()

        if not np.any(finite_mask):
            return np.zeros_like(height, dtype=np.float32)

        if self.normalize_mode == "zscore":
            mean_value = np.nanmean(height[finite_mask])
            std_value = np.nanstd(height[finite_mask]) + 1e-6
            height = (height - mean_value) / std_value

        else:
            denominator = self.height_max - self.height_min
            if denominator <= 1e-6:
                denominator = 1.0

            height = (height - self.height_min) / (denominator + 1e-6)
            height = np.clip(height, 0.0, 1.0)

        height[~finite_mask] = 0.0

        return height.astype(np.float32)


def rgb_mask_height_collate(batch):
    images, heights, masks, names = zip(*batch)
    return torch.stack(images), torch.stack(heights), torch.stack(masks), list(names)


# ============================================================
# Basic utilities
# ============================================================
def validate_input_shape(input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width].")

    height, width = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Both height and width in INPUT_SHAPE must be divisible by 32.")


def read_split_file(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    try:
        with open(split_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]
    except UnicodeDecodeError:
        with open(split_path, "r", encoding="gbk", errors="ignore") as file:
            lines = [line.strip() for line in file if line.strip()]

    return lines


def read_height_range(height_range_path):
    """
    Read height_min and height_max from height_range.txt.

    Supported formats:
        height_min=0.1234
        height_max=5.6789
    """
    if not os.path.exists(height_range_path):
        raise FileNotFoundError(f"Height range file not found: {height_range_path}")

    height_min = None
    height_max = None

    with open(height_range_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip().lower()
            value = value.strip()

            if key == "height_min":
                height_min = float(value)
            elif key == "height_max":
                height_max = float(value)

    if height_min is None or height_max is None:
        raise ValueError(
            f"Cannot parse height_min and height_max from: {height_range_path}"
        )

    if height_max <= height_min:
        raise ValueError(
            f"Invalid height range: height_min={height_min}, height_max={height_max}"
        )

    return height_min, height_max


def calculate_r2(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1.0 - ss_res / ss_tot


def calculate_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if not np.any(valid_mask):
        return {
            "count": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "r2": np.nan,
        }

    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    errors = y_pred - y_true

    return {
        "count": len(y_true),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
        "r2": float(calculate_r2(y_true, y_pred)),
    }


def denormalize_height(height_norm, height_min, height_max):
    scale = height_max - height_min
    return height_norm * scale + height_min


def compute_region_height(values, statistic="max"):
    """
    Compute one tree-level height statistic from crown pixels.
    """
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    if statistic == "max":
        return float(np.nanmax(values))

    if statistic == "p95":
        return float(np.nanpercentile(values, 95))

    if statistic == "p98":
        return float(np.nanpercentile(values, 98))

    if statistic == "top1_percent":
        sorted_values = np.sort(values)
        top_k = max(1, int(np.ceil(sorted_values.size * 0.01)))
        return float(np.mean(sorted_values[-top_k:]))

    raise ValueError(
        f"Unsupported HEIGHT_STATISTIC: {statistic}. "
        "Choose from max, p95, p98, top1_percent."
    )


# ============================================================
# Tree-level extraction
# ============================================================
def extract_tree_height_pairs(
    pred_map,
    true_map,
    mask_map,
    min_tree_area=5,
    height_statistic="max",
    valid_true_height_min=0.0,
    valid_true_height_max=6.0,
):
    """
    Extract tree-level true/predicted height pairs from one image.

    Each connected component in the crown mask is treated as one tree.
    """
    pred_map = pred_map.astype(np.float32)
    true_map = true_map.astype(np.float32)
    mask_map = mask_map.astype(bool)

    valid_mask = (
        mask_map
        & np.isfinite(true_map)
        & np.isfinite(pred_map)
        & (true_map > valid_true_height_min)
        & (true_map <= valid_true_height_max)
    )

    labeled = measure.label(valid_mask, connectivity=2)

    pred_filled = np.full_like(pred_map, np.nan, dtype=np.float32)
    true_filled = np.full_like(true_map, np.nan, dtype=np.float32)

    tree_records = []

    for region_id in range(1, int(labeled.max()) + 1):
        region_mask = labeled == region_id

        area = int(np.sum(region_mask))
        if area < min_tree_area:
            continue

        region_true = true_map[region_mask]
        region_pred = pred_map[region_mask]

        region_true = region_true[np.isfinite(region_true)]
        region_pred = region_pred[np.isfinite(region_pred)]

        region_true = region_true[
            (region_true > valid_true_height_min)
            & (region_true <= valid_true_height_max)
        ]

        if region_true.size == 0 or region_pred.size == 0:
            continue

        true_height = compute_region_height(region_true, statistic=height_statistic)
        pred_height = compute_region_height(region_pred, statistic=height_statistic)

        if not np.isfinite(true_height) or not np.isfinite(pred_height):
            continue

        true_filled[region_mask] = true_height
        pred_filled[region_mask] = pred_height

        tree_records.append(
            {
                "region_id": region_id,
                "area_pixels": area,
                "true_height_m": true_height,
                "pred_height_m": pred_height,
                "error_m": pred_height - true_height,
            }
        )

    return pred_filled, true_filled, tree_records


# ============================================================
# Checkpoint loading
# ============================================================
def load_checkpoint_flexible(model, checkpoint_path, device):
    """
    Load checkpoint weights flexibly.

    Supports:
        1. raw state_dict
        2. {'state_dict': state_dict}
        3. {'model_state_dict': state_dict}
        4. DataParallel checkpoints with 'module.' prefix
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading model weights: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    normalized_state = {}
    for key, value in checkpoint.items():
        normalized_key = key[len("module."):] if key.startswith("module.") else key
        normalized_state[normalized_key] = value

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    matched_state = {}

    for key, value in normalized_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)

    print(f"Loaded keys: {len(loaded_keys)}")
    print(f"Skipped keys: {len(skipped_keys)}")
    if len(skipped_keys) > 0:
        print("Skipped key examples:", skipped_keys[:10])


def build_model(device):
    model = LSNetBiFoSTHeightRegression(
        num_classes=OUT_CHANNELS,
        backbone=BACKBONE,
        in_channels=IN_CHANNELS,
        use_svit=USE_SVIT,
        use_bie=USE_BIE,
        svit_on_f16=SVIT_ON_F16,
        svit_on_f32=SVIT_ON_F32,
        svit_stoken_size=SVIT_STOKEN_SIZE,
        svit_heads=SVIT_HEADS,
        svit_n_iter=SVIT_N_ITER,
    )

    load_checkpoint_flexible(model, MODEL_PATH, device)

    model = model.to(device)
    model.eval()

    if hasattr(model, "get_ablation_config"):
        print("[Model ablation config]:", model.get_ablation_config())

    return model


# ============================================================
# Visualization
# ============================================================
def save_height_maps_and_visualization(
    image_tensor,
    true_filled,
    pred_filled,
    save_dir,
    image_name,
    height_min,
    height_max,
    metrics_for_image,
):
    """
    Save:
        1. combined visualization
        2. true filled height map
        3. predicted filled height map
    """
    visual_dir = os.path.join(save_dir, "visualizations")
    true_dir = os.path.join(save_dir, "true_maps")
    pred_dir = os.path.join(save_dir, "pred_maps")

    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(true_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    rgb = image_tensor[:3].cpu().numpy().transpose(1, 2, 0)
    rgb = np.clip(rgb, 0.0, 1.0)

    error_map = pred_filled - true_filled
    error_map = np.where(np.isfinite(true_filled), error_map, np.nan)

    mae = metrics_for_image["mae"]
    rmse = metrics_for_image["rmse"]
    r2 = metrics_for_image["r2"]
    count = metrics_for_image["count"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    im1 = axes[1].imshow(
        true_filled,
        cmap="viridis_r",
        vmin=height_min,
        vmax=height_max,
    )
    axes[1].set_title("True Height")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(
        pred_filled,
        cmap="viridis_r",
        vmin=height_min,
        vmax=height_max,
    )
    axes[2].set_title(f"Pred Height\nRMSE={rmse:.3f} m")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    im3 = axes[3].imshow(
        error_map,
        cmap="coolwarm_r",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[3].set_title(f"Error\nN={count}, MAE={mae:.3f}, R2={r2:.3f}")
    axes[3].axis("off")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(visual_dir, f"{image_name}_visual.png"), dpi=300)
    plt.close(fig)

    if SAVE_TRUE_PRED_MAPS:
        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(
            true_filled,
            cmap="viridis_r",
            vmin=height_min,
            vmax=height_max,
        )
        axis.set_title("True Height")
        axis.axis("off")
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        plt.savefig(
            os.path.join(true_dir, f"{image_name}_true.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(
            pred_filled,
            cmap="viridis_r",
            vmin=height_min,
            vmax=height_max,
        )
        axis.set_title("Predicted Height")
        axis.axis("off")
        plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        plt.savefig(
            os.path.join(pred_dir, f"{image_name}_pred.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_summary_plots(all_true, all_pred, save_dir):
    errors = all_pred - all_true

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    bias = float(np.mean(errors))
    r2 = float(calculate_r2(all_true, all_pred))

    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=40, color="skyblue", edgecolor="k", alpha=0.85)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.axvline(
        bias,
        color="red",
        linestyle="-",
        linewidth=1.2,
        label=f"Mean Bias={bias:.3f} m",
    )
    plt.xlabel("Prediction Error (m)")
    plt.ylabel("Tree Count")
    plt.title("Distribution of Prediction Errors")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.text(
        0.02,
        0.95,
        f"RMSE = {rmse:.4f} m\nMAE = {mae:.4f} m\nBias = {bias:.4f} m",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_histogram.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    hexbin = plt.hexbin(all_true, all_pred, gridsize=40, cmap="viridis", mincnt=1)
    colorbar = plt.colorbar(hexbin)
    colorbar.set_label("Point density")

    min_value = float(min(all_true.min(), all_pred.min()))
    max_value = float(max(all_true.max(), all_pred.max()))

    plt.plot([min_value, max_value], [min_value, max_value], "r", linewidth=1.2)
    plt.xlabel("True Height (m)")
    plt.ylabel("Predicted Height (m)")
    plt.title("Predicted vs. True Tree Heights")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.text(
        min_value + (max_value - min_value) * 0.05,
        max_value - (max_value - min_value) * 0.10,
        f"R2 = {r2:.4f}\nRMSE = {rmse:.4f} m\nMAE = {mae:.4f} m\nBias = {bias:.4f} m",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predicted_vs_true_hexbin.png"), dpi=300)
    plt.close()


# ============================================================
# Main evaluation
# ============================================================
def evaluate_height_regression():
    validate_input_shape(INPUT_SHAPE)
    seed_everything(SEED)

    use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")
    print("\n[Height-regression evaluation configuration]")
    print(f"  backbone         = {BACKBONE}")
    print(f"  in_channels      = {IN_CHANNELS}")
    print(f"  use_svit         = {USE_SVIT}")
    print(f"  use_bie          = {USE_BIE}")
    print(f"  svit_on_f16      = {SVIT_ON_F16}")
    print(f"  svit_on_f32      = {SVIT_ON_F32}")
    print(f"  height_statistic = {HEIGHT_STATISTIC}")
    print(f"  model_path       = {MODEL_PATH}\n")

    height_min, height_max = read_height_range(HEIGHT_RANGE_PATH)
    print(f"Height range: {height_min:.4f} m to {height_max:.4f} m")

    val_lines = read_split_file(os.path.join(DATASET_PATH, VAL_TXT))
    print(f"Validation samples: {len(val_lines)}")

    val_dataset = RgbMaskHeightEvalDataset(
        annotation_lines=val_lines,
        input_shape=INPUT_SHAPE,
        dataset_path=DATASET_PATH,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        height_dir=HEIGHT_DIR,
        rgb_suffix=RGB_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        height_suffix=HEIGHT_SUFFIX,
        height_min=height_min,
        height_max=height_max,
        normalize_mode=NORMALIZE_MODE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=rgb_mask_height_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=SEED),
    )

    model = build_model(device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    pred_tif_dir = os.path.join(SAVE_DIR, "pred_tif")
    if SAVE_PRED_TIF:
        os.makedirs(pred_tif_dir, exist_ok=True)

    all_tree_records = []
    per_image_records = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
            images, true_heights_norm, masks, names = batch

            images = images.float().to(device)
            true_heights_norm = true_heights_norm.float().to(device)
            masks = masks.float().to(device)

            pred_heights_norm = model(images)

            pred_heights = denormalize_height(
                pred_heights_norm.detach().cpu().numpy().astype(np.float32),
                height_min,
                height_max,
            )
            true_heights = denormalize_height(
                true_heights_norm.detach().cpu().numpy().astype(np.float32),
                height_min,
                height_max,
            )
            masks_np = masks.detach().cpu().numpy().astype(bool)

            if CLIP_PRED_TO_HEIGHT_RANGE:
                pred_heights = np.clip(pred_heights, height_min, height_max)

            for sample_index in range(pred_heights.shape[0]):
                image_name = names[sample_index]

                pred_map = pred_heights[sample_index, 0]
                true_map = true_heights[sample_index, 0]
                mask_map = masks_np[sample_index, 0]

                pred_filled, true_filled, tree_records = extract_tree_height_pairs(
                    pred_map=pred_map,
                    true_map=true_map,
                    mask_map=mask_map,
                    min_tree_area=MIN_TREE_AREA,
                    height_statistic=HEIGHT_STATISTIC,
                    valid_true_height_min=VALID_TRUE_HEIGHT_MIN,
                    valid_true_height_max=VALID_TRUE_HEIGHT_MAX,
                )

                image_true = np.array(
                    [record["true_height_m"] for record in tree_records],
                    dtype=np.float32,
                )
                image_pred = np.array(
                    [record["pred_height_m"] for record in tree_records],
                    dtype=np.float32,
                )

                image_metrics = calculate_regression_metrics(image_true, image_pred)

                per_image_records.append(
                    {
                        "image_name": image_name,
                        "tree_count": image_metrics["count"],
                        "mae_m": image_metrics["mae"],
                        "rmse_m": image_metrics["rmse"],
                        "bias_m": image_metrics["bias"],
                        "r2": image_metrics["r2"],
                    }
                )

                for record in tree_records:
                    record = dict(record)
                    record["image_name"] = image_name
                    all_tree_records.append(record)

                if SAVE_PRED_TIF:
                    tifffile.imwrite(
                        os.path.join(pred_tif_dir, f"{image_name}_pred.tif"),
                        pred_filled.astype(np.float32),
                    )

                if SAVE_VISUALIZATION:
                    save_height_maps_and_visualization(
                        image_tensor=images[sample_index].detach().cpu(),
                        true_filled=true_filled,
                        pred_filled=pred_filled,
                        save_dir=SAVE_DIR,
                        image_name=image_name,
                        height_min=height_min,
                        height_max=height_max,
                        metrics_for_image=image_metrics,
                    )

    if len(all_tree_records) == 0:
        print("\n[Warning] No valid tree crowns were found.")
        return

    all_true = np.array(
        [record["true_height_m"] for record in all_tree_records],
        dtype=np.float32,
    )
    all_pred = np.array(
        [record["pred_height_m"] for record in all_tree_records],
        dtype=np.float32,
    )

    overall_metrics = calculate_regression_metrics(all_true, all_pred)

    print("\n==== Overall Height-Regression Metrics ====")
    print(f"Valid tree count: {overall_metrics['count']}")
    print(f"MAE  = {overall_metrics['mae']:.4f} m")
    print(f"RMSE = {overall_metrics['rmse']:.4f} m")
    print(f"Bias = {overall_metrics['bias']:.4f} m")
    print(f"R2   = {overall_metrics['r2']:.4f}")
    print(f"Height statistic: {HEIGHT_STATISTIC}")
    print(f"Minimum tree area: {MIN_TREE_AREA} pixels")

    tree_df = pd.DataFrame(all_tree_records)
    tree_df = tree_df[
        [
            "image_name",
            "region_id",
            "area_pixels",
            "true_height_m",
            "pred_height_m",
            "error_m",
        ]
    ]

    image_df = pd.DataFrame(per_image_records)

    if SAVE_CSV:
        tree_df.to_csv(
            os.path.join(SAVE_DIR, "tree_height_true_pred_pairs.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        image_df.to_csv(
            os.path.join(SAVE_DIR, "per_image_height_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    if SAVE_EXCEL:
        excel_path = os.path.join(SAVE_DIR, "tree_height_evaluation_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            tree_df.to_excel(writer, sheet_name="tree_pairs", index=False)
            image_df.to_excel(writer, sheet_name="per_image_metrics", index=False)

            summary_df = pd.DataFrame(
                [
                    {
                        "valid_tree_count": overall_metrics["count"],
                        "mae_m": overall_metrics["mae"],
                        "rmse_m": overall_metrics["rmse"],
                        "bias_m": overall_metrics["bias"],
                        "r2": overall_metrics["r2"],
                        "height_statistic": HEIGHT_STATISTIC,
                        "min_tree_area": MIN_TREE_AREA,
                        "model_path": MODEL_PATH,
                        "height_range_path": HEIGHT_RANGE_PATH,
                        "backbone": BACKBONE,
                        "use_svit": USE_SVIT,
                        "use_bie": USE_BIE,
                        "svit_on_f16": SVIT_ON_F16,
                        "svit_on_f32": SVIT_ON_F32,
                    }
                ]
            )
            summary_df.to_excel(writer, sheet_name="summary", index=False)

    with open(os.path.join(SAVE_DIR, "metrics_summary.txt"), "w", encoding="utf-8") as file:
        file.write("==== Overall Height-Regression Metrics ====\n")
        file.write(f"Valid tree count: {overall_metrics['count']}\n")
        file.write(f"MAE={overall_metrics['mae']:.6f}\n")
        file.write(f"RMSE={overall_metrics['rmse']:.6f}\n")
        file.write(f"Bias={overall_metrics['bias']:.6f}\n")
        file.write(f"R2={overall_metrics['r2']:.6f}\n")
        file.write(f"Height statistic={HEIGHT_STATISTIC}\n")
        file.write(f"Minimum tree area={MIN_TREE_AREA}\n")
        file.write(f"Model path={MODEL_PATH}\n")
        file.write(f"Height range path={HEIGHT_RANGE_PATH}\n")
        file.write("\n==== Model Configuration ====\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"IN_CHANNELS={IN_CHANNELS}\n")
        file.write(f"USE_SVIT={USE_SVIT}\n")
        file.write(f"USE_BIE={USE_BIE}\n")
        file.write(f"SVIT_ON_F16={SVIT_ON_F16}\n")
        file.write(f"SVIT_ON_F32={SVIT_ON_F32}\n")

    save_summary_plots(all_true, all_pred, SAVE_DIR)

    print(f"\nResults saved to: {SAVE_DIR}")
    print("Generated files:")
    print(" - metrics_summary.txt")
    print(" - tree_height_true_pred_pairs.csv")
    print(" - per_image_height_metrics.csv")
    print(" - tree_height_evaluation_results.xlsx")
    print(" - error_histogram.png")
    print(" - predicted_vs_true_hexbin.png")


if __name__ == "__main__":
    evaluate_height_regression()