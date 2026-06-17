import os
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from PIL import Image
from matplotlib.ticker import FuncFormatter
from skimage import measure
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nets.lsnet_bifost_height_regression import LSNetBiFoSTHeightRegression
from utils.utils import seed_everything


# ============================================================
# Manual configuration
# ============================================================

# Checkpoint and height range
MODEL_PATH = r"model_data/best_epoch_weights.pth"
HEIGHT_RANGE_PATH = r"logs_reg_rgb_mask/height_range.txt"

# Dataset
DATASET_PATH = "TreeHeightDataset"
RGB_DIR = "rgb"
MASK_DIR = "mask"

RGB_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"

# Optional file list.
# If None, all images in RGB_DIR will be used.
FILE_LIST_TXT = None
# FILE_LIST_TXT = "predict.txt"

# Model structure
BACKBONE = "focal_s"          # focal_t / focal_s / focal_b
IN_CHANNELS = 4               # RGB(3) + crown mask(1)
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
DEVICE = "cuda"               # cuda / cpu
SEED = 11

# Prediction setting
MIN_TREE_AREA = 5
CLIP_PRED_TO_HEIGHT_RANGE = False

# Preprocessing mode:
#   "center_crop_pad" is consistent with validation in train_height_regression.py.
#   "resize" directly resizes RGB and mask to INPUT_SHAPE.
PREPROCESS_MODE = "center_crop_pad"     # center_crop_pad / resize

# Output
SAVE_DIR = "results_predict_height"
SAVE_PRED_TIF = True
SAVE_PRED_PNG = True
SAVE_TREE_TABLE = True

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# Dataset
# ============================================================
class RgbMaskHeightPredictDataset(Dataset):
    """
    Prediction dataset for tree-height regression.

    Input:
        RGB image + crown mask -> 4-channel tensor.

    Output:
        image tensor, mask tensor, image prefix, and preprocessing metadata.
    """

    def __init__(
        self,
        dataset_path: str,
        file_list: List[str],
        input_shape: Tuple[int, int] = (640, 640),
        rgb_dir: str = "rgb",
        mask_dir: str = "mask",
        rgb_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        preprocess_mode: str = "center_crop_pad",
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.file_list = file_list
        self.input_shape = input_shape

        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir

        self.rgb_suffix = rgb_suffix
        self.mask_suffix = mask_suffix

        self.preprocess_mode = preprocess_mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        prefix = self.file_list[index].strip()

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

        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB image not found: {rgb_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask image not found: {mask_path}")

        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.float32)

        original_h, original_w = mask.shape

        if self.preprocess_mode == "resize":
            rgb, mask, meta = self.resize_to_input_shape(
                rgb=rgb,
                mask=mask,
                target_size=self.input_shape,
                original_h=original_h,
                original_w=original_w,
            )
        elif self.preprocess_mode == "center_crop_pad":
            rgb, mask, meta = self.center_crop_or_pad(
                rgb=rgb,
                mask=mask,
                target_size=self.input_shape,
                original_h=original_h,
                original_w=original_w,
            )
        else:
            raise ValueError(
                f"Unsupported PREPROCESS_MODE: {self.preprocess_mode}. "
                "Use 'center_crop_pad' or 'resize'."
            )

        mask_channel_last = np.expand_dims(mask, axis=-1)
        input_4ch = np.concatenate([rgb, mask_channel_last], axis=-1)
        input_4ch = np.transpose(input_4ch, (2, 0, 1))

        mask = np.expand_dims(mask, axis=0)

        return (
            torch.from_numpy(input_4ch).float(),
            torch.from_numpy(mask).float(),
            prefix,
            meta,
        )

    @staticmethod
    def resize_to_input_shape(rgb, mask, target_size, original_h, original_w):
        target_h, target_w = target_size

        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        rgb_resized = Image.fromarray(rgb_uint8).resize(
            (target_w, target_h),
            Image.BILINEAR,
        )
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (target_w, target_h),
            Image.NEAREST,
        )

        rgb = np.array(rgb_resized, dtype=np.float32) / 255.0
        mask = (np.array(mask_resized, dtype=np.uint8) > 0).astype(np.float32)

        meta = {
            "mode": "resize",
            "original_h": original_h,
            "original_w": original_w,
            "target_h": target_h,
            "target_w": target_w,
            "pad_top": 0,
            "pad_left": 0,
            "crop_top": 0,
            "crop_left": 0,
        }

        return rgb, mask, meta

    @staticmethod
    def pad_to_size(rgb, mask, target_size):
        target_h, target_w = target_size
        h, w, _ = rgb.shape

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_h == 0 and pad_w == 0:
            meta_pad = {
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
            }
            return rgb, mask, meta_pad

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

        meta_pad = {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }

        return rgb, mask, meta_pad

    def center_crop_or_pad(self, rgb, mask, target_size, original_h, original_w):
        target_h, target_w = target_size

        rgb, mask, meta_pad = self.pad_to_size(
            rgb=rgb,
            mask=mask,
            target_size=target_size,
        )

        padded_h, padded_w, _ = rgb.shape

        crop_top = max(0, (padded_h - target_h) // 2)
        crop_left = max(0, (padded_w - target_w) // 2)

        rgb = rgb[crop_top: crop_top + target_h, crop_left: crop_left + target_w, :]
        mask = mask[crop_top: crop_top + target_h, crop_left: crop_left + target_w]

        meta = {
            "mode": "center_crop_pad",
            "original_h": original_h,
            "original_w": original_w,
            "target_h": target_h,
            "target_w": target_w,
            "padded_h": padded_h,
            "padded_w": padded_w,
            "crop_top": crop_top,
            "crop_left": crop_left,
            **meta_pad,
        }

        return rgb, mask, meta


def predict_collate(batch):
    images, masks, prefixes, metas = zip(*batch)

    return (
        torch.stack(images),
        torch.stack(masks),
        list(prefixes),
        list(metas),
    )


# ============================================================
# Utilities
# ============================================================
def validate_input_shape(input_shape):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width].")

    height, width = input_shape

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Both height and width in INPUT_SHAPE must be divisible by 32.")


def read_height_range(height_range_path):
    """
    Read height_min and height_max from height_range.txt.

    Supported format:
        height_min=0.000000
        height_max=5.873200
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


def read_file_list_from_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File list does not exist: {txt_path}")

    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            file_list = [line.strip() for line in file if line.strip()]
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="gbk", errors="ignore") as file:
            file_list = [line.strip() for line in file if line.strip()]

    return file_list


def collect_file_list(dataset_path, rgb_dir, rgb_suffix, file_list_txt=None):
    if file_list_txt is not None:
        if os.path.isabs(file_list_txt):
            txt_path = file_list_txt
        else:
            txt_path = os.path.join(dataset_path, file_list_txt)

        return read_file_list_from_txt(txt_path)

    rgb_dir_full = os.path.join(dataset_path, rgb_dir)

    if not os.path.exists(rgb_dir_full):
        raise FileNotFoundError(f"RGB folder does not exist: {rgb_dir_full}")

    file_list = []

    for file_name in os.listdir(rgb_dir_full):
        suffix = Path(file_name).suffix.lower()

        if rgb_suffix is not None:
            if suffix == rgb_suffix.lower():
                file_list.append(Path(file_name).stem)
        else:
            if suffix in SUPPORTED_IMAGE_EXTENSIONS:
                file_list.append(Path(file_name).stem)

    file_list = sorted(file_list)

    if len(file_list) == 0:
        raise FileNotFoundError(f"No images found in: {rgb_dir_full}")

    return file_list


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
    matched_state = {}
    loaded_keys = []
    skipped_keys = []

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
        pretrained=False,
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
# Tree-level post-processing
# ============================================================
def fill_tree_canopy_with_max_pred(
    pred_map,
    mask_map,
    min_tree_area=5,
):
    """
    Fill each connected crown component with its maximum predicted height.

    Args:
        pred_map: predicted height map in meters.
        mask_map: binary crown mask, 1 for crown and 0 for background.
        min_tree_area: minimum component area in pixels.

    Returns:
        pred_filled: height map filled by tree-level maximum height.
        tree_records: per-tree prediction records.
    """
    pred_map = pred_map.astype(np.float32)
    mask_map = mask_map.astype(bool)

    valid_mask = mask_map & np.isfinite(pred_map) & (pred_map > 0.0)

    labeled = measure.label(valid_mask, connectivity=2)
    pred_filled = np.full_like(pred_map, np.nan, dtype=np.float32)

    tree_records = []

    for region_id in range(1, int(labeled.max()) + 1):
        region_mask = labeled == region_id
        area = int(np.sum(region_mask))

        if area < min_tree_area:
            continue

        region_pred = pred_map[region_mask]
        region_pred = region_pred[np.isfinite(region_pred)]
        region_pred = region_pred[region_pred > 0.0]

        if region_pred.size == 0:
            continue

        max_pred = float(np.nanmax(region_pred))

        if not np.isfinite(max_pred) or max_pred <= 0.0:
            continue

        pred_filled[region_mask] = max_pred

        tree_records.append(
            {
                "region_id": region_id,
                "area_pixels": area,
                "pred_height_m": max_pred,
            }
        )

    return pred_filled, tree_records


# ============================================================
# Visualization and saving
# ============================================================
def save_prediction_png(
    pred_filled,
    save_path,
    height_min,
    height_max,
    title="Predicted Tree Height",
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axis = plt.subplots(figsize=(6, 5))

    image = axis.imshow(
        pred_filled,
        cmap="viridis_r",
        vmin=height_min,
        vmax=height_max,
    )

    axis.set_title(title)
    axis.axis("off")

    plt.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        format=FuncFormatter(lambda value, _: f"{value:.2f}"),
    )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main prediction pipeline
# ============================================================
def predict_tree_height_fill():
    validate_input_shape(INPUT_SHAPE)
    seed_everything(SEED)

    use_cuda = DEVICE == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Using device: {device}")
    print("\n[Height-regression prediction configuration]")
    print(f"  backbone        = {BACKBONE}")
    print(f"  in_channels     = {IN_CHANNELS}")
    print(f"  use_svit        = {USE_SVIT}")
    print(f"  use_bie         = {USE_BIE}")
    print(f"  svit_on_f16     = {SVIT_ON_F16}")
    print(f"  svit_on_f32     = {SVIT_ON_F32}")
    print(f"  preprocess_mode = {PREPROCESS_MODE}")
    print(f"  model_path      = {MODEL_PATH}\n")

    height_min, height_max = read_height_range(HEIGHT_RANGE_PATH)
    height_scale = height_max - height_min

    print(f"Height range: {height_min:.4f} m to {height_max:.4f} m")

    file_list = collect_file_list(
        dataset_path=DATASET_PATH,
        rgb_dir=RGB_DIR,
        rgb_suffix=RGB_SUFFIX,
        file_list_txt=FILE_LIST_TXT,
    )

    print(f"Prediction samples: {len(file_list)}")

    dataset = RgbMaskHeightPredictDataset(
        dataset_path=DATASET_PATH,
        file_list=file_list,
        input_shape=INPUT_SHAPE,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        rgb_suffix=RGB_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        preprocess_mode=PREPROCESS_MODE,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=predict_collate,
    )

    model = build_model(device)

    os.makedirs(SAVE_DIR, exist_ok=True)

    tif_dir = os.path.join(SAVE_DIR, "pred_tif")
    png_dir = os.path.join(SAVE_DIR, "pred_png")

    if SAVE_PRED_TIF:
        os.makedirs(tif_dir, exist_ok=True)
    if SAVE_PRED_PNG:
        os.makedirs(png_dir, exist_ok=True)

    all_tree_records = []

    with torch.no_grad():
        for images, masks, prefixes, metas in tqdm(dataloader, desc="Predicting"):
            images = images.float().to(device)
            masks = masks.float().to(device)

            pred_norm = model(images)
            pred_norm_np = pred_norm.detach().cpu().numpy().astype(np.float32)
            mask_np = masks.detach().cpu().numpy().astype(bool)

            pred_height_np = pred_norm_np * height_scale + height_min

            if CLIP_PRED_TO_HEIGHT_RANGE:
                pred_height_np = np.clip(pred_height_np, height_min, height_max)

            for sample_index in range(pred_height_np.shape[0]):
                prefix = prefixes[sample_index]

                pred_map = pred_height_np[sample_index, 0]
                mask_map = mask_np[sample_index, 0]

                pred_map = np.where(mask_map, pred_map, np.nan)

                pred_filled, tree_records = fill_tree_canopy_with_max_pred(
                    pred_map=pred_map,
                    mask_map=mask_map,
                    min_tree_area=MIN_TREE_AREA,
                )

                for record in tree_records:
                    record = dict(record)
                    record["image_name"] = prefix
                    all_tree_records.append(record)

                if SAVE_PRED_TIF:
                    tif_path = os.path.join(tif_dir, f"{prefix}_filled_pred.tif")
                    tifffile.imwrite(tif_path, pred_filled.astype(np.float32))

                if SAVE_PRED_PNG:
                    png_path = os.path.join(png_dir, f"{prefix}_filled_pred.png")
                    save_prediction_png(
                        pred_filled=pred_filled,
                        save_path=png_path,
                        height_min=height_min,
                        height_max=height_max,
                        title="Predicted Tree Height",
                    )

    if SAVE_TREE_TABLE:
        tree_table_path = os.path.join(SAVE_DIR, "predicted_tree_heights.csv")
        tree_df = pd.DataFrame(all_tree_records)

        if len(tree_df) > 0:
            tree_df = tree_df[
                [
                    "image_name",
                    "region_id",
                    "area_pixels",
                    "pred_height_m",
                ]
            ]

        tree_df.to_csv(tree_table_path, index=False, encoding="utf-8-sig")

    with open(os.path.join(SAVE_DIR, "prediction_config.txt"), "w", encoding="utf-8") as file:
        file.write("==== Height-Regression Prediction Configuration ====\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"HEIGHT_RANGE_PATH={HEIGHT_RANGE_PATH}\n")
        file.write(f"DATASET_PATH={DATASET_PATH}\n")
        file.write(f"RGB_DIR={RGB_DIR}\n")
        file.write(f"MASK_DIR={MASK_DIR}\n")
        file.write(f"RGB_SUFFIX={RGB_SUFFIX}\n")
        file.write(f"MASK_SUFFIX={MASK_SUFFIX}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"IN_CHANNELS={IN_CHANNELS}\n")
        file.write(f"OUT_CHANNELS={OUT_CHANNELS}\n")
        file.write(f"USE_SVIT={USE_SVIT}\n")
        file.write(f"USE_BIE={USE_BIE}\n")
        file.write(f"SVIT_ON_F16={SVIT_ON_F16}\n")
        file.write(f"SVIT_ON_F32={SVIT_ON_F32}\n")
        file.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        file.write(f"PREPROCESS_MODE={PREPROCESS_MODE}\n")
        file.write(f"HEIGHT_MIN={height_min:.6f}\n")
        file.write(f"HEIGHT_MAX={height_max:.6f}\n")
        file.write(f"MIN_TREE_AREA={MIN_TREE_AREA}\n")

    print("\n[√] Tree-height prediction finished.")
    print(f"Results saved to: {SAVE_DIR}")
    print(f" - TIF results: {tif_dir}")
    print(f" - PNG results: {png_dir}")
    print(f" - Tree table: {os.path.join(SAVE_DIR, 'predicted_tree_heights.csv')}")


if __name__ == "__main__":
    predict_tree_height_fill()