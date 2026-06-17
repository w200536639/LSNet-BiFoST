import os
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

from skimage.measure import label
from skimage.filters import sobel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.utils import seed_everything
from nets.high_resolution_crown_segmentation import HighResolutionCrownSegmentationNet


# ============================================================
# Optional libraries
# ============================================================
try:
    import rasterio
except Exception:
    rasterio = None

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
# Manual configuration
# ============================================================
SEED = 11
CUDA = True
FP16 = False
DETERMINISTIC = True

NUM_CLASSES = 2
BACKBONE = "lsnet_b"
PRETRAINED = False
MODEL_PATH = r"model_data/best_epoch_weights.pth"

INPUT_SHAPE = [640, 640]

VOC_ROOT = r"VOCdevkit"
IMAGE_SET = r"VOC2007/ImageSets/Segmentation/val.txt"

# High-resolution multispectral image configuration
IMAGE_DIR_NAME = "RSImages"
IMAGE_SUFFIX = ".tif"
IN_CHANNELS = 8

NORM_MODE = "percentile"      # percentile / max / max_dtype / none

WV3_BAND_NAMES = [
    "Coastal",
    "Blue",
    "Green",
    "Yellow",
    "Red",
    "RedEdge",
    "NIR1",
    "NIR2",
]

WV3_BAND_ORDER = None

# Ground truth configuration
# None = automatically detect foreground value from masks.
# If your GT mask is 0/1, use 1.
# If your GT mask is 0/38, use 38.
TARGET_LABEL_VALUE = None

IGNORE_VALUE = None
AUTO_INFER_IGNORE_255 = True

# Evaluation and visualization
VISUALIZATION_FOLDER = r"results"
METRICS_FILE = r"results/metrics.txt"

MASK_ALPHA = 0.12
MATCH_IOU_THR = 0.5

# WV3 pseudo-RGB visualization bands.
# Python index starts from 0.
# For WV3 order [Coastal, Blue, Green, Yellow, Red, RedEdge, NIR1, NIR2],
# (4, 2, 1) means Red-Green-Blue.
RGB_VIS_BANDS = (4, 2, 1)

BATCH_SIZE = 1
NUM_WORKERS = 0

USE_BIE = 1
USE_HPA = 1

RUN_TAG = ""


# ============================================================
# Basic utilities
# ============================================================
def read_text_lines_with_fallback(txt_path):
    """Read val.txt with encoding fallback."""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Image set file does not exist: {txt_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(txt_path, "r", encoding=encoding) as file:
                return [line.strip().split()[0] for line in file if line.strip()]
        except UnicodeDecodeError:
            continue

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as file:
        return [line.strip().split()[0] for line in file if line.strip()]


def read_mask_u8(path):
    """
    Safely read mask.

    If mask is P mode, keep palette index values.
    Otherwise, convert to grayscale.
    """
    mask = Image.open(path)

    if mask.mode == "P":
        return np.array(mask, dtype=np.uint8)

    return np.array(mask.convert("L"), dtype=np.uint8)


def scan_mask_values(mask_dir, ids, scan_max=300):
    """Scan unique mask values."""
    counts = Counter()
    scanned = 0

    for name in ids[:min(scan_max, len(ids))]:
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            continue

        arr = read_mask_u8(mask_path)

        unique_values, frequencies = np.unique(arr, return_counts=True)

        for value, frequency in zip(unique_values, frequencies):
            counts[int(value)] += int(frequency)

        scanned += 1

    return counts, scanned


def auto_detect_foreground_value(mask_dir, ids, ignore_value=None, scan_max=300):
    """
    Automatically detect foreground raw value.

    Rule:
        choose the most frequent non-zero and non-ignore value.
    """
    counts = Counter()

    for name in ids[:min(scan_max, len(ids))]:
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            continue

        arr = read_mask_u8(mask_path)

        unique_values, frequencies = np.unique(arr, return_counts=True)

        for value, frequency in zip(unique_values, frequencies):
            value = int(value)

            if value == 0:
                continue

            if ignore_value is not None and value == int(ignore_value):
                continue

            counts[value] += int(frequency)

    if len(counts) == 0:
        return 1, counts

    return int(counts.most_common(1)[0][0]), counts


def safe_filename(name):
    """Make safe filename for Windows."""
    name = str(name)
    name = name.replace("（", "(").replace("）", ")")

    illegal_chars = '<>:"/\\|?*'

    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


# ============================================================
# Image reading and preprocessing
# ============================================================
def normalize_multispectral_chw(arr_chw, norm_mode):
    """Normalize multispectral image in CHW format."""
    norm_mode = str(norm_mode).lower().strip()

    arr_chw = arr_chw.astype(np.float32)

    if norm_mode == "none":
        return arr_chw

    if norm_mode == "max":
        max_value = float(np.max(arr_chw)) + 1e-6
        return arr_chw / max_value

    if norm_mode == "max_dtype":
        max_value = float(np.max(arr_chw))

        if max_value > 2000:
            return arr_chw / 65535.0

        return arr_chw / (max_value + 1e-6)

    if norm_mode == "percentile":
        output = np.zeros_like(arr_chw, dtype=np.float32)

        for band_id in range(arr_chw.shape[0]):
            band = arr_chw[band_id]

            low = np.percentile(band, 2)
            high = np.percentile(band, 98)

            if high - low < 1e-6:
                output[band_id] = 0.0
            else:
                output[band_id] = (band - low) / (high - low)

        return np.clip(output, 0.0, 1.0)

    raise ValueError(f"Unsupported NORM_MODE: {norm_mode}")


def read_multispectral_tif_hwc(path, in_channels=8, norm_mode="percentile", band_order=None):
    """
    Read high-resolution multispectral GeoTIFF as HWC float32.
    """
    if rasterio is None:
        raise ImportError("Reading GeoTIFF requires rasterio. Please install rasterio first.")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file does not exist: {path}")

    with rasterio.open(path) as src:
        band_count = src.count
        use_channels = min(int(in_channels), int(band_count))
        arr = src.read(list(range(1, use_channels + 1)))

    if arr.shape[0] < int(in_channels):
        pad = np.zeros(
            (
                int(in_channels) - arr.shape[0],
                arr.shape[1],
                arr.shape[2],
            ),
            dtype=arr.dtype,
        )
        arr = np.concatenate([arr, pad], axis=0)

    arr = arr.astype(np.float32)

    if band_order is not None:
        band_order = list(band_order)
        band_order = [int(np.clip(i, 0, arr.shape[0] - 1)) for i in band_order]

        if len(band_order) < arr.shape[0]:
            rest = [i for i in range(arr.shape[0]) if i not in band_order]
            band_order = band_order + rest

        arr = arr[band_order[:arr.shape[0]]]

    arr = normalize_multispectral_chw(arr, norm_mode)

    return np.transpose(arr, (1, 2, 0)).astype(np.float32)


def resize_image_hwc_float(image_hwc, size_wh):
    """
    Resize HWC float32 image.

    Args:
        size_wh: (width, height)
    """
    width, height = size_wh

    if cv2 is not None:
        return cv2.resize(
            image_hwc,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

    channels = image_hwc.shape[2]
    output = np.zeros((height, width, channels), dtype=np.float32)

    for channel_id in range(channels):
        band = image_hwc[:, :, channel_id]
        band_16 = np.clip(band * 65535.0, 0, 65535).astype(np.uint16)

        pil_image = Image.fromarray(band_16, mode="I;16")
        resized = pil_image.resize((width, height), Image.BICUBIC)

        output[:, :, channel_id] = np.array(resized, dtype=np.float32) / 65535.0

    return output


def resize_mask_hw_u8(mask_hw, size_wh):
    """Resize mask using nearest interpolation."""
    width, height = size_wh
    pil_mask = Image.fromarray(mask_hw.astype(np.uint8))
    return np.array(pil_mask.resize((width, height), Image.NEAREST), dtype=np.uint8)


def letterbox_np(image_hwc, out_hw, pad_val):
    """
    Letterbox resize for HWC multispectral image.

    Returns:
        canvas
        metadata: (nh, nw, top, left)
    """
    out_h, out_w = out_hw
    image_h, image_w = image_hwc.shape[:2]
    channels = image_hwc.shape[2]

    scale = min(out_w / image_w, out_h / image_h)

    new_w = int(image_w * scale)
    new_h = int(image_h * scale)

    resized = resize_image_hwc_float(image_hwc, (new_w, new_h))

    canvas = np.ones((out_h, out_w, channels), dtype=np.float32) * float(pad_val)

    top = (out_h - new_h) // 2
    left = (out_w - new_w) // 2

    canvas[top: top + new_h, left: left + new_w, :] = resized

    return canvas, (new_h, new_w, top, left)


def letterbox_mask(mask_hw, out_hw, pad_val=0):
    """
    Letterbox resize for mask.
    """
    out_h, out_w = out_hw
    mask_h, mask_w = mask_hw.shape[:2]

    scale = min(out_w / mask_w, out_h / mask_h)

    new_w = int(mask_w * scale)
    new_h = int(mask_h * scale)

    resized = resize_mask_hw_u8(mask_hw, (new_w, new_h))

    canvas = np.ones((out_h, out_w), dtype=np.uint8) * int(pad_val)

    top = (out_h - new_h) // 2
    left = (out_w - new_w) // 2

    canvas[top: top + new_h, left: left + new_w] = resized

    return canvas, (new_h, new_w, top, left)


def to_vis_rgb_uint8(canvas_hwc, rgb_vis_bands=(4, 2, 1)):
    """
    Convert multispectral HWC image to pseudo RGB uint8.
    """
    channels = canvas_hwc.shape[2]

    band_r, band_g, band_b = rgb_vis_bands

    band_r = int(np.clip(band_r, 0, channels - 1))
    band_g = int(np.clip(band_g, 0, channels - 1))
    band_b = int(np.clip(band_b, 0, channels - 1))

    rgb = canvas_hwc[:, :, [band_r, band_g, band_b]].astype(np.float32)

    output = np.zeros_like(rgb, dtype=np.float32)

    for channel_id in range(3):
        band = rgb[:, :, channel_id]

        low = np.percentile(band, 2)
        high = np.percentile(band, 98)

        if high - low < 1e-6:
            output[:, :, channel_id] = 0.0
        else:
            output[:, :, channel_id] = (band - low) / (high - low)

    output = np.clip(output, 0.0, 1.0) * 255.0

    return output.astype(np.uint8)


# ============================================================
# Metrics
# ============================================================
def calculate_instance_metrics(predicted, ground_truth_raw, target_label_value, ignore_value=None):
    """
    Instance-level metric by one-to-one matching.

    TP:
        pred object and GT object IoU >= MATCH_IOU_THR.

    FP:
        unmatched predicted objects.

    FN:
        unmatched GT objects.
    """
    ground_truth_raw = ground_truth_raw.astype(np.int32)
    predicted = predicted.astype(np.int32)

    if ignore_value is not None:
        valid = ground_truth_raw != int(ignore_value)
    else:
        valid = np.ones_like(ground_truth_raw, dtype=bool)

    gt_binary = (
        (ground_truth_raw == int(target_label_value))
        & valid
    ).astype(np.uint8)

    pred_binary = (
        (predicted == 1)
        & valid
    ).astype(np.uint8)

    gt_labels = label(gt_binary, connectivity=2)
    pred_labels = label(pred_binary, connectivity=2)

    num_gt_objects = int(gt_labels.max())
    num_pred_objects = int(pred_labels.max())

    if num_gt_objects == 0 and num_pred_objects == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_gt_objects": 0,
            "num_pred_objects": 0,
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
            "valid": valid,
        }

    if num_gt_objects == 0 and num_pred_objects > 0:
        return {
            "tp": 0,
            "fp": num_pred_objects,
            "fn": 0,
            "num_gt_objects": num_gt_objects,
            "num_pred_objects": num_pred_objects,
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
            "valid": valid,
        }

    if num_gt_objects > 0 and num_pred_objects == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": num_gt_objects,
            "num_gt_objects": num_gt_objects,
            "num_pred_objects": num_pred_objects,
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
            "valid": valid,
        }

    gt_masks = {}
    gt_areas = {}

    for gt_id in range(1, num_gt_objects + 1):
        gt_mask = gt_labels == gt_id
        area = int(gt_mask.sum())

        if area > 0:
            gt_masks[gt_id] = gt_mask
            gt_areas[gt_id] = area

    pred_masks = {}
    pred_areas = {}

    for pred_id in range(1, num_pred_objects + 1):
        pred_mask = pred_labels == pred_id
        area = int(pred_mask.sum())

        if area > 0:
            pred_masks[pred_id] = pred_mask
            pred_areas[pred_id] = area

    matched_pairs = []

    for gt_id, gt_mask in gt_masks.items():
        gt_area = gt_areas[gt_id]

        for pred_id, pred_mask in pred_masks.items():
            intersection = int((gt_mask & pred_mask).sum())

            if intersection == 0:
                continue

            union = gt_area + pred_areas[pred_id] - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou >= MATCH_IOU_THR:
                matched_pairs.append((gt_id, pred_id, iou))

    matched_pairs.sort(key=lambda item: item[2], reverse=True)

    matched_gt = set()
    matched_pred = set()
    tp = 0

    for gt_id, pred_id, _ in matched_pairs:
        if gt_id in matched_gt or pred_id in matched_pred:
            continue

        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        tp += 1

    fn = num_gt_objects - tp
    fp = num_pred_objects - tp
    fp = max(fp, 0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gt_objects": num_gt_objects,
        "num_pred_objects": num_pred_objects,
        "gt_binary": gt_binary,
        "pred_binary": pred_binary,
        "valid": valid,
    }


def calculate_pixel_iou(pred_binary, gt_binary, valid_mask=None):
    """Return intersection and union."""
    if valid_mask is None:
        valid_mask = np.ones_like(gt_binary, dtype=bool)

    pred = (pred_binary == 1) & valid_mask
    gt = (gt_binary == 1) & valid_mask

    intersection = int((pred & gt).sum())
    union = int((pred | gt).sum())

    return intersection, union


def safe_iou_for_miou(intersection, union, pred_has_fg, gt_has_fg):
    """
    Per-image IoU for mIoU.
    """
    if union > 0:
        return intersection / union

    if (not pred_has_fg) and (not gt_has_fg):
        return 1.0

    return 0.0


# ============================================================
# Visualization
# ============================================================
def visualize_and_save(original_rgb_u8, metrics, save_path):
    """
    Save visualization:
        GT: yellow
        Prediction: light blue
        Prediction boundary: red
    """
    gt_binary = metrics["gt_binary"]
    pred_binary = metrics["pred_binary"]
    valid = metrics.get("valid", None)

    vis = original_rgb_u8.copy()

    if valid is not None:
        vis[~valid] = 0

    pred_color = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    pred_color[pred_binary == 1] = [135, 206, 250]

    gt_color = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    gt_color[gt_binary == 1] = [255, 215, 0]

    edges = sobel(pred_binary) > 0

    blended = vis.astype(np.float32)

    blended[pred_binary == 1] = (
        blended[pred_binary == 1] * (1.0 - MASK_ALPHA)
        + pred_color[pred_binary == 1] * MASK_ALPHA
    )

    blended[gt_binary == 1] = (
        blended[gt_binary == 1] * (1.0 - MASK_ALPHA)
        + gt_color[gt_binary == 1] * MASK_ALPHA
    )

    blended[edges] = [255, 0, 0]
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(blended)
    ax.set_title(
        f"GT: {metrics['num_gt_objects']} | "
        f"Pred: {metrics['num_pred_objects']} | "
        f"TP: {metrics['tp']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}"
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Dataset
# ============================================================
class HighResolutionEvalDataset(Dataset):
    """
    Dataset for high-resolution crown segmentation prediction and evaluation.

    This dataset requires:
        image file
        ground-truth mask

    Therefore it can calculate metrics.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        input_shape,
        image_set_file,
        image_suffix=".tif",
        in_channels=8,
        norm_mode="percentile",
        target_label_value=None,
        ignore_value=None,
        auto_infer_ignore_255=True,
        rgb_vis_bands=(4, 2, 1),
        band_order=None,
    ):
        super().__init__()

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_shape = input_shape
        self.image_suffix = str(image_suffix)
        self.in_channels = int(in_channels)
        self.norm_mode = str(norm_mode).lower().strip()
        self.rgb_vis_bands = tuple(rgb_vis_bands)
        self.band_order = band_order

        self.ids = read_text_lines_with_fallback(image_set_file)

        print(f"加载数据集: {image_set_file}, 样本数量: {len(self.ids)}")

        self.ignore_value = None if ignore_value is None else int(ignore_value)

        if self.ignore_value is None and auto_infer_ignore_255:
            counts, _ = scan_mask_values(self.mask_dir, self.ids, scan_max=200)

            if 255 in counts:
                self.ignore_value = 255
                print("[Eval] AUTO infer ignore_value=255 because 255 exists in masks.")

        if target_label_value is None:
            detected_value, foreground_counts = auto_detect_foreground_value(
                mask_dir=self.mask_dir,
                ids=self.ids,
                ignore_value=self.ignore_value,
                scan_max=300,
            )
            self.target_label_value = int(detected_value)
            print(f"[Eval] AUTO detect target_label_value={self.target_label_value}")

            if len(foreground_counts) > 0:
                print(f"[Eval] Non-zero top values: {foreground_counts.most_common(10)}")

        else:
            self.target_label_value = int(target_label_value)

        counts, scanned = scan_mask_values(self.mask_dir, self.ids, scan_max=300)

        print(f"[Eval][MaskScan] scanned={scanned}, top20={counts.most_common(20)}")
        print(f"[Eval][MaskScan] all_keys={sorted(counts.keys())}")
        print(f"[Eval] final target_label_value={self.target_label_value}, ignore_value={self.ignore_value}")

        self.total_gt_objects = self.count_total_gt_objects()

    def __len__(self):
        return len(self.ids)

    def count_total_gt_objects(self):
        total_gt_objects = 0

        for index, name in enumerate(self.ids):
            mask_path = os.path.join(self.mask_dir, f"{name}.png")

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

            mask_raw = read_mask_u8(mask_path)

            mask_full, _ = letterbox_mask(
                mask_raw,
                (self.input_shape[0], self.input_shape[1]),
                pad_val=0,
            )

            if self.ignore_value is not None:
                valid = mask_full != int(self.ignore_value)
            else:
                valid = np.ones_like(mask_full, dtype=bool)

            gt_binary = (
                (mask_full == int(self.target_label_value))
                & valid
            ).astype(np.uint8)

            gt_labels = label(gt_binary, connectivity=2)
            num_gt_objects = int(gt_labels.max())

            total_gt_objects += num_gt_objects

            if index < 3:
                print(
                    f"[GT Count] {name}: "
                    f"instances={num_gt_objects}, "
                    f"raw_unique={np.unique(mask_raw)[:20].tolist()}"
                )

        print(f"所有图像中的真实目标实例总数 = {total_gt_objects}")

        return int(total_gt_objects)

    def __getitem__(self, index):
        name = self.ids[index]

        image_path = os.path.join(self.image_dir, f"{name}{self.image_suffix}")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")

        image_hwc = read_multispectral_tif_hwc(
            image_path,
            in_channels=self.in_channels,
            norm_mode=self.norm_mode,
            band_order=self.band_order,
        )

        image_full, image_meta = letterbox_np(
            image_hwc,
            (self.input_shape[0], self.input_shape[1]),
            pad_val=0.0,
        )

        mask_raw = read_mask_u8(mask_path)

        mask_full, mask_meta = letterbox_mask(
            mask_raw,
            (self.input_shape[0], self.input_shape[1]),
            pad_val=0,
        )

        if image_meta != mask_meta:
            raise RuntimeError(
                f"Image and mask letterbox metadata mismatch for {name}: "
                f"image_meta={image_meta}, mask_meta={mask_meta}"
            )

        new_h, new_w, top, left = image_meta

        image_tensor = image_full.astype(np.float32).transpose(2, 0, 1)
        vis_rgb = to_vis_rgb_uint8(image_full, self.rgb_vis_bands)

        if index < 2:
            print(
                f"[EvalSample] {name}: "
                f"image_shape={image_hwc.shape}, "
                f"mask_unique_raw={np.unique(mask_raw)[:30].tolist()}, "
                f"mask_unique_letterbox={np.unique(mask_full)[:30].tolist()}"
            )

        return (
            torch.from_numpy(image_tensor).float(),
            torch.from_numpy(mask_full).long(),
            torch.tensor(new_h).long(),
            torch.tensor(new_w).long(),
            torch.tensor(top).long(),
            torch.tensor(left).long(),
            torch.from_numpy(vis_rgb).byte(),
            name,
        )


# ============================================================
# Model loading
# ============================================================
def load_checkpoint_strict_flexible(model, checkpoint_path, device):
    """
    Flexible checkpoint loading.

    Class name and file name changes do not affect loading.
    The state_dict is matched by module attribute names.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在或路径为空: {checkpoint_path}")

    print(f"Load weights: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    normalized_state = {}

    for key, value in state_dict.items():
        normalized_key = key[len("module."):] if key.startswith("module.") else key
        normalized_state[normalized_key] = value

    model_state = model.state_dict()

    load_keys = []
    miss_keys = []

    for key, value in normalized_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            load_keys.append(key)
        else:
            miss_keys.append(key)

    model.load_state_dict(model_state, strict=False)

    print("Successful Load Key Num:", len(load_keys))
    print("Fail To Load Key Num:", len(miss_keys))

    if len(miss_keys) > 0:
        print("Fail keys examples:", miss_keys[:50])

    print(
        "\n提示：final/head 或第一层输入通道不一致导致没载入通常可以接受；"
        "如果大量 encoder 权重没载入，才需要检查模型结构。\n"
    )


def build_model(device):
    """Build high-resolution crown segmentation model."""
    model = HighResolutionCrownSegmentationNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
        in_channels=IN_CHANNELS,
    ).to(device)

    model.eval()

    if hasattr(model, "get_model_profile"):
        print("[Model Profile]", model.get_model_profile())
    else:
        print("[Model] HighResolutionCrownSegmentationNet")

    load_checkpoint_strict_flexible(model, MODEL_PATH, device)

    return model


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    seed_everything(SEED)

    if DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device("cuda" if (torch.cuda.is_available() and CUDA) else "cpu")

    print(f"使用设备: {device}")

    auto_suffix = f"{BACKBONE}_bie{USE_BIE}_hpa{USE_HPA}_highres_c{IN_CHANNELS}"
    run_tag = RUN_TAG.strip() or auto_suffix

    print(f"[Run Tag] {run_tag}")

    net = build_model(device)

    voc2007_dir = os.path.join(VOC_ROOT, "VOC2007")
    image_dir = os.path.join(voc2007_dir, IMAGE_DIR_NAME)
    mask_dir = os.path.join(voc2007_dir, "SegmentationClass")
    image_set_file = os.path.join(VOC_ROOT, IMAGE_SET)

    for path in [voc2007_dir, image_dir, mask_dir, image_set_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"路径不存在: {path}")

    val_dataset = HighResolutionEvalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_shape=INPUT_SHAPE,
        image_set_file=image_set_file,
        image_suffix=IMAGE_SUFFIX,
        in_channels=IN_CHANNELS,
        norm_mode=NORM_MODE,
        target_label_value=TARGET_LABEL_VALUE,
        ignore_value=IGNORE_VALUE,
        auto_infer_ignore_255=AUTO_INFER_IGNORE_255,
        rgb_vis_bands=RGB_VIS_BANDS,
        band_order=WV3_BAND_ORDER,
    )

    target_value = val_dataset.target_label_value
    ignore_value = val_dataset.ignore_value

    print(f"[Eval] using target_label_value={target_value}, ignore_value={ignore_value}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"验证集加载完成，样本数: {len(val_dataset)}")

    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_FILE) or ".", exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_intersection = 0
    total_union = 0

    per_image_ious = []
    image_metrics = []

    print(f"开始验证，共 {len(val_loader)} 个 batch，batch_size={BATCH_SIZE}")

    use_amp = FP16 and device.type == "cuda"

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(val_loader, desc="Predict + Evaluate")):
            images, masks_raw, new_h, new_w, top, left, vis_rgb, name = batch

            if isinstance(name, (list, tuple)):
                image_name = name[0]
            else:
                image_name = str(name)

            images = images.to(device, non_blocking=True)

            if use_amp:
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                    try:
                        with torch.amp.autocast("cuda", enabled=True):
                            outputs = net(images)
                    except TypeError:
                        with torch.cuda.amp.autocast():
                            outputs = net(images)
                else:
                    with torch.cuda.amp.autocast():
                        outputs = net(images)
            else:
                outputs = net(images)

            prediction_full = torch.softmax(outputs, dim=1).argmax(dim=1)
            prediction_full = prediction_full.detach().cpu().numpy()[0].astype(np.uint8)

            new_h_value = int(new_h[0].item())
            new_w_value = int(new_w[0].item())
            top_value = int(top[0].item())
            left_value = int(left[0].item())

            prediction = prediction_full[
                top_value: top_value + new_h_value,
                left_value: left_value + new_w_value,
            ]

            gt_full = masks_raw.numpy()[0].astype(np.uint8)
            ground_truth = gt_full[
                top_value: top_value + new_h_value,
                left_value: left_value + new_w_value,
            ]

            original_rgb = vis_rgb.numpy()[0]
            original_rgb = original_rgb[
                top_value: top_value + new_h_value,
                left_value: left_value + new_w_value,
            ]

            metrics = calculate_instance_metrics(
                predicted=prediction,
                ground_truth_raw=ground_truth,
                target_label_value=target_value,
                ignore_value=ignore_value,
            )

            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

            intersection, union = calculate_pixel_iou(
                metrics["pred_binary"],
                metrics["gt_binary"],
                valid_mask=metrics.get("valid", None),
            )

            total_intersection += intersection
            total_union += union

            image_iou_for_miou = safe_iou_for_miou(
                intersection=intersection,
                union=union,
                pred_has_fg=bool(metrics["pred_binary"].sum() > 0),
                gt_has_fg=bool(metrics["gt_binary"].sum() > 0),
            )

            per_image_ious.append(float(image_iou_for_miou))

            image_iou_raw = intersection / union if union != 0 else 0.0

            image_metrics.append(
                {
                    "image_name": image_name,
                    "num_gt_objects": metrics["num_gt_objects"],
                    "num_pred_objects": metrics["num_pred_objects"],
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "intersection": intersection,
                    "union": union,
                    "iou_raw": float(image_iou_raw),
                    "iou_for_miou": float(image_iou_for_miou),
                }
            )

            if batch_index < 5 or (batch_index + 1) % 10 == 0:
                print(
                    f"图像 {batch_index + 1}/{len(val_loader)} [{image_name}]: "
                    f"GT={metrics['num_gt_objects']}, "
                    f"Pred={metrics['num_pred_objects']}, "
                    f"TP={metrics['tp']}, "
                    f"FP={metrics['fp']}, "
                    f"FN={metrics['fn']}, "
                    f"inter/union={intersection}/{union}, "
                    f"IoU_raw={image_iou_raw:.4f}, "
                    f"IoU_for_mIoU={image_iou_for_miou:.4f}"
                )

            safe_name = safe_filename(image_name)
            save_path = os.path.join(
                VISUALIZATION_FOLDER,
                f"{run_tag}_{safe_name}.png",
            )

            visualize_and_save(original_rgb, metrics, save_path)

    # ========================================================
    # Final metrics
    # ========================================================
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0

    if precision + recall > 0:
        f1_score = 2.0 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    overall_iou = total_intersection / total_union if total_union else 0.0
    miou = float(np.mean(per_image_ious)) if len(per_image_ious) > 0 else 0.0

    print("\n==== 高分多光谱树冠分割指标：Instance + Pixel ====")
    print(f"真实目标实例总数: {val_dataset.total_gt_objects}")
    print(f"[Instance] TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"[Instance] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
    print(f"[Pixel] Overall IoU: {overall_iou:.4f} | mIoU: {miou:.4f}")
    print(f"IoU 匹配阈值 MATCH_IOU_THR: {MATCH_IOU_THR}")
    print(f"target_label_value={target_value}, ignore_value={ignore_value}")
    print(f"消融开关: BIE={USE_BIE}, HPA={USE_HPA}")
    print(f"Band names={WV3_BAND_NAMES}")
    print(f"RGB_VIS_BANDS={RGB_VIS_BANDS}, norm={NORM_MODE}")

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        file.write(f"[Run Tag] {run_tag}\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"target_label_value={target_value}, ignore_value={ignore_value}\n")
        file.write(f"真实目标实例总数: {val_dataset.total_gt_objects}\n")

        file.write(f"[Instance] TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        file.write(f"[Instance] Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}\n")

        file.write(f"[Pixel] Overall_IoU={overall_iou:.4f}\n")
        file.write(f"[Pixel] mIoU={miou:.4f}\n")

        file.write(f"匹配阈值 MATCH_IOU_THR={MATCH_IOU_THR}\n")
        file.write(f"消融: BIE={USE_BIE}, HPA={USE_HPA}\n\n")

        file.write("==== 每张图像的指标 ====\n")
        file.write("image_name,gt_num,pred_num,tp,fp,fn,intersection,union,IoU_raw,IoU_for_mIoU\n")

        for item in image_metrics:
            file.write(
                f"{item['image_name']},"
                f"{item['num_gt_objects']},"
                f"{item['num_pred_objects']},"
                f"{item['tp']},"
                f"{item['fp']},"
                f"{item['fn']},"
                f"{item['intersection']},"
                f"{item['union']},"
                f"{item['iou_raw']:.4f},"
                f"{item['iou_for_miou']:.4f}\n"
            )

        file.write("\n==== 配置参数 ====\n")
        file.write(f"SEED={SEED}\n")
        file.write(f"CUDA={CUDA}, FP16={FP16}, DETERMINISTIC={DETERMINISTIC}\n")
        file.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        file.write(f"VOC_ROOT={VOC_ROOT}\n")
        file.write(f"IMAGE_SET={IMAGE_SET}\n")
        file.write(f"IMAGE_DIR_NAME={IMAGE_DIR_NAME}, IMAGE_SUFFIX={IMAGE_SUFFIX}\n")
        file.write(f"IN_CHANNELS={IN_CHANNELS}, NORM_MODE={NORM_MODE}\n")
        file.write(f"WV3_BAND_ORDER={WV3_BAND_ORDER}\n")
        file.write(f"RGB_VIS_BANDS={RGB_VIS_BANDS}\n")
        file.write(f"WV3_BAND_NAMES={WV3_BAND_NAMES}\n")

    print(f"\n详细结果已保存至: {METRICS_FILE}")
    print(f"可视化结果已保存至: {VISUALIZATION_FOLDER}")
