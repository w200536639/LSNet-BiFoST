import csv
import os
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from skimage.filters import sobel
from skimage.measure import label
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nets.lsnet_bifost_segmentation import LSNetBiFoSTSegmentation
from utils.utils import seed_everything


# ============================================================
# Configuration
# ============================================================
SEED = 11
USE_CUDA = True
USE_FP16 = False
USE_DETERMINISTIC = True

NUM_CLASSES = 2
FOREGROUND_CLASS_ID = 1

BACKBONE = "lsnet_b"
PRETRAINED = False
MODEL_PATH = r"model_data/best_epoch_weights.pth"
INPUT_SHAPE = [640, 640]

VOC_ROOT = r"VOCdevkit"
IMAGE_SET = r"VOC2007/ImageSets/Segmentation/val.txt"

RESULTS_DIR = r"results"
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
METRICS_TXT = os.path.join(RESULTS_DIR, "metrics.txt")
METRICS_CSV = os.path.join(RESULTS_DIR, "per_image_metrics.csv")

MASK_ALPHA = 0.12
MATCH_IOU_THRESHOLD = 0.5
INSTANCE_MIN_AREA = 5

# ============================================================
# Ground-truth label setting
# ============================================================
# 如果你的标注掩膜是 0/1，设为 1。
# 如果你的标注掩膜是 0/38，设为 38。
# 如果不确定，设为 None，代码会自动从 val.txt 前若干张标注里识别最常见的非 0 标签。
TARGET_LABEL_VALUE = None

AUTODETECT_SCAN_MAX = 200

BATCH_SIZE = 1
NUM_WORKERS = 0

USE_BIE = 1
USE_HPA = 1

RUN_TAG = ""


# ============================================================
# File utilities
# ============================================================
def read_lines_with_fallback(file_path: str) -> List[str]:
    """
    Read text lines with multiple encoding fallbacks.

    This fixes:
        UnicodeDecodeError: 'utf-8' codec can't decode byte ...
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "ansi"]

    last_error = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                lines = [line.strip().split()[0] for line in file if line.strip()]
            print(f"[Read txt] {file_path} 使用编码: {encoding}")
            return lines
        except UnicodeDecodeError as error:
            last_error = error
            continue
        except LookupError:
            continue

    print(f"[Warning] 常规编码读取失败，使用 utf-8 + ignore 强制读取: {file_path}")
    if last_error is not None:
        print(f"[Warning] Last decode error: {last_error}")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = [line.strip().split()[0] for line in file if line.strip()]

    return lines


def find_file_by_id(folder: str, image_id: str, preferred_ext: str = ".png") -> str:
    """
    Find a file by image_id with flexible suffix.

    This supports jpg/png/tif naming differences.
    """
    preferred_path = os.path.join(folder, image_id + preferred_ext)
    if os.path.exists(preferred_path):
        return preferred_path

    supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    for ext in supported_exts:
        path = os.path.join(folder, image_id + ext)
        if os.path.exists(path):
            return path

    if os.path.isdir(folder):
        for file_name in os.listdir(folder):
            stem, _ = os.path.splitext(file_name)
            if stem == image_id:
                return os.path.join(folder, file_name)

    raise FileNotFoundError(f"Cannot find file for image_id={image_id} in folder={folder}")


# ============================================================
# Preprocessing utilities
# ============================================================
def letterbox_image(image: Image.Image, size_hw: Tuple[int, int]):
    """
    Resize an RGB image with unchanged aspect ratio and pad it to size_hw.

    Args:
        image: PIL RGB image.
        size_hw: Target size as (height, width).

    Returns:
        canvas: Letterboxed RGB image.
        resized_width: Width after resizing.
        resized_height: Height after resizing.
        top: Top padding.
        left: Left padding.
    """
    target_h, target_w = size_hw
    original_w, original_h = image.size

    scale = min(target_w / original_w, target_h / original_h)
    resized_w = int(original_w * scale)
    resized_h = int(original_h * scale)

    resized_image = image.resize((resized_w, resized_h), Image.BICUBIC)

    canvas = Image.new("RGB", (target_w, target_h), (128, 128, 128))

    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2

    canvas.paste(resized_image, (left, top))

    return canvas, resized_w, resized_h, top, left


def letterbox_mask(mask: Image.Image, size_hw: Tuple[int, int]):
    """
    Resize a label mask with unchanged aspect ratio and pad it to size_hw.

    Padding value is 0 and treated as background.
    """
    target_h, target_w = size_hw
    original_w, original_h = mask.size

    scale = min(target_w / original_w, target_h / original_h)
    resized_w = int(original_w * scale)
    resized_h = int(original_h * scale)

    resized_mask = mask.resize((resized_w, resized_h), Image.NEAREST)

    canvas = Image.new("L", (target_w, target_h), 0)

    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2

    canvas.paste(resized_mask, (left, top))

    return canvas, resized_w, resized_h, top, left


def validate_input_shape(input_shape: List[int]):
    """Check whether input shape is valid."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width].")

    height, width = input_shape

    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Height and width in INPUT_SHAPE must be divisible by 32.")


def build_run_tag() -> str:
    """Build evaluation run tag."""
    auto_tag = f"{BACKBONE}_bie{USE_BIE}_hpa{USE_HPA}"
    return RUN_TAG.strip() or auto_tag


# ============================================================
# Target-label detection
# ============================================================
def autodetect_target_label_value(
    image_ids: List[str],
    mask_dir: str,
    scan_max: int = 200,
) -> int:
    """
    Automatically detect the foreground label value from validation masks.

    It chooses the most frequent non-zero pixel value among scanned masks.
    """
    value_counter = {}

    scan_ids = image_ids[: min(scan_max, len(image_ids))]

    for image_id in scan_ids:
        try:
            mask_path = find_file_by_id(mask_dir, image_id, preferred_ext=".png")
        except FileNotFoundError:
            continue

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        values, counts = np.unique(mask, return_counts=True)

        for value, count in zip(values, counts):
            value = int(value)
            count = int(count)

            if value == 0:
                continue

            value_counter[value] = value_counter.get(value, 0) + count

    if len(value_counter) == 0:
        print("[Auto target label] 没有检测到非 0 标签，默认使用 1。")
        return 1

    sorted_values = sorted(value_counter.items(), key=lambda item: item[1], reverse=True)
    target_value = int(sorted_values[0][0])

    print(f"[Auto target label] 自动识别 target_label_value = {target_value}")
    print(f"[Auto target label] 非 0 像素 Top 值及计数: {sorted_values[:5]}")

    return target_value


# ============================================================
# Metric utilities
# ============================================================
def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area."""
    if min_area <= 1:
        return binary_mask.astype(np.uint8)

    labeled_mask = label(binary_mask.astype(np.uint8), connectivity=2)
    cleaned_mask = np.zeros_like(binary_mask, dtype=np.uint8)

    for component_id in range(1, int(labeled_mask.max()) + 1):
        component = labeled_mask == component_id

        if int(component.sum()) >= min_area:
            cleaned_mask[component] = 1

    return cleaned_mask


def calculate_instance_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    target_label_value: int,
    foreground_class_id: int = FOREGROUND_CLASS_ID,
    match_iou_threshold: float = MATCH_IOU_THRESHOLD,
    min_area: int = INSTANCE_MIN_AREA,
) -> Dict:
    """
    Calculate instance-level TP, FP, and FN using one-to-one greedy matching.

    A predicted instance and a GT instance are matched if IoU >= match_iou_threshold.
    Each GT and each prediction can be matched at most once.
    """
    gt_binary = (ground_truth == target_label_value).astype(np.uint8)
    pred_binary = (prediction == foreground_class_id).astype(np.uint8)

    gt_binary = remove_small_components(gt_binary, min_area=min_area)
    pred_binary = remove_small_components(pred_binary, min_area=min_area)

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
        }

    if num_gt_objects == 0 and num_pred_objects > 0:
        return {
            "tp": 0,
            "fp": num_pred_objects,
            "fn": 0,
            "num_gt_objects": 0,
            "num_pred_objects": num_pred_objects,
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
        }

    if num_gt_objects > 0 and num_pred_objects == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": num_gt_objects,
            "num_gt_objects": num_gt_objects,
            "num_pred_objects": 0,
            "gt_binary": gt_binary,
            "pred_binary": pred_binary,
        }

    gt_masks = {}
    gt_areas = {}

    for gt_id in range(1, num_gt_objects + 1):
        gt_component = gt_labels == gt_id
        gt_area = int(gt_component.sum())

        if gt_area > 0:
            gt_masks[gt_id] = gt_component
            gt_areas[gt_id] = gt_area

    pred_masks = {}
    pred_areas = {}

    for pred_id in range(1, num_pred_objects + 1):
        pred_component = pred_labels == pred_id
        pred_area = int(pred_component.sum())

        if pred_area > 0:
            pred_masks[pred_id] = pred_component
            pred_areas[pred_id] = pred_area

    candidate_pairs = []

    for gt_id, gt_mask in gt_masks.items():
        gt_area = gt_areas[gt_id]

        for pred_id, pred_mask in pred_masks.items():
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())

            if intersection == 0:
                continue

            union = gt_area + pred_areas[pred_id] - intersection
            iou = intersection / union if union > 0 else 0.0

            if iou >= match_iou_threshold:
                candidate_pairs.append((gt_id, pred_id, iou))

    candidate_pairs.sort(key=lambda item: item[2], reverse=True)

    matched_gt = set()
    matched_pred = set()
    true_positive = 0

    for gt_id, pred_id, _ in candidate_pairs:
        if gt_id in matched_gt or pred_id in matched_pred:
            continue

        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        true_positive += 1

    false_negative = num_gt_objects - true_positive
    false_positive = num_pred_objects - true_positive

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "num_gt_objects": num_gt_objects,
        "num_pred_objects": num_pred_objects,
        "gt_binary": gt_binary,
        "pred_binary": pred_binary,
    }


def calculate_pixel_intersection_union(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
):
    """Calculate foreground-pixel intersection and union."""
    intersection = int(np.logical_and(pred_binary == 1, gt_binary == 1).sum())
    union = int(np.logical_or(pred_binary == 1, gt_binary == 1).sum())

    return intersection, union


def safe_divide(numerator: float, denominator: float) -> float:
    """Safe division."""
    return numerator / denominator if denominator != 0 else 0.0


# ============================================================
# Visualization
# ============================================================
def visualize_and_save(original_image: np.ndarray, metrics: Dict, save_path: str):
    """Save overlay visualization of prediction, GT, and predicted boundary."""
    gt_binary = metrics["gt_binary"]
    pred_binary = metrics["pred_binary"]

    prediction_color = np.zeros((*pred_binary.shape, 3), dtype=np.uint8)
    prediction_color[pred_binary == 1] = [135, 206, 250]

    gt_color = np.zeros((*gt_binary.shape, 3), dtype=np.uint8)
    gt_color[gt_binary == 1] = [255, 215, 0]

    boundary_mask = sobel(pred_binary) > 0
    boundary_color = np.zeros_like(prediction_color)
    boundary_color[boundary_mask] = [255, 0, 0]

    blended = original_image.astype(np.float32)

    blended[pred_binary == 1] = (
        blended[pred_binary == 1] * (1.0 - MASK_ALPHA)
        + prediction_color[pred_binary == 1] * MASK_ALPHA
    )
    blended[gt_binary == 1] = (
        blended[gt_binary == 1] * (1.0 - MASK_ALPHA)
        + gt_color[gt_binary == 1] * MASK_ALPHA
    )
    blended[boundary_mask] = boundary_color[boundary_mask]

    blended = np.clip(blended, 0, 255).astype(np.uint8)

    fig, axis = plt.subplots(figsize=(10, 10))
    axis.imshow(blended)
    axis.set_title(
        f"GT: {metrics['num_gt_objects']} | "
        f"Pred: {metrics['num_pred_objects']} | "
        f"TP: {metrics['tp']} | "
        f"FP: {metrics['fp']} | "
        f"FN: {metrics['fn']}"
    )
    axis.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Dataset
# ============================================================
class VOCSegmentationEvalDataset(Dataset):
    """VOC-style segmentation dataset for evaluation."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        input_shape: List[int],
        image_set_file: str,
        target_label_value: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_shape = input_shape

        self.image_ids = self._load_image_set(image_set_file)

        if target_label_value is None:
            self.target_label_value = autodetect_target_label_value(
                image_ids=self.image_ids,
                mask_dir=self.mask_dir,
                scan_max=AUTODETECT_SCAN_MAX,
            )
        else:
            self.target_label_value = int(target_label_value)
            print(f"[Target label] 使用手动设置 target_label_value = {self.target_label_value}")

        print(f"Loaded dataset: {image_set_file}, samples: {len(self.image_ids)}")

        self.total_gt_objects = self._count_total_gt_objects()

    @staticmethod
    def _load_image_set(image_set_file: str):
        return read_lines_with_fallback(image_set_file)

    def _count_total_gt_objects(self):
        total_gt_objects = 0

        for index, image_id in enumerate(self.image_ids):
            mask_path = find_file_by_id(self.mask_dir, image_id, preferred_ext=".png")
            mask = Image.open(mask_path).convert("L")

            mask_canvas, resized_w, resized_h, top, left = letterbox_mask(
                mask,
                (self.input_shape[0], self.input_shape[1]),
            )

            mask_array = np.array(mask_canvas, dtype=np.uint8)
            valid_mask = mask_array[top: top + resized_h, left: left + resized_w]

            gt_binary = (valid_mask == self.target_label_value).astype(np.uint8)
            gt_binary = remove_small_components(
                gt_binary,
                min_area=INSTANCE_MIN_AREA,
            )

            gt_labels = label(gt_binary, connectivity=2)
            num_gt_objects = int(gt_labels.max())
            total_gt_objects += num_gt_objects

            if index < 5:
                print(
                    f"Image {index + 1}/{len(self.image_ids)}: "
                    f"id={image_id}, GT instances = {num_gt_objects}"
                )

        print(f"Total GT instances in all images = {total_gt_objects}")

        return total_gt_objects

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        image_path = find_file_by_id(self.image_dir, image_id, preferred_ext=".jpg")
        mask_path = find_file_by_id(self.mask_dir, image_id, preferred_ext=".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_canvas, resized_w, resized_h, top, left = letterbox_image(
            image,
            (self.input_shape[0], self.input_shape[1]),
        )

        mask_canvas, mask_resized_w, mask_resized_h, mask_top, mask_left = letterbox_mask(
            mask,
            (self.input_shape[0], self.input_shape[1]),
        )

        if (resized_w, resized_h, top, left) != (
            mask_resized_w,
            mask_resized_h,
            mask_top,
            mask_left,
        ):
            raise RuntimeError(
                f"Image and mask letterbox parameters are inconsistent for {image_id}."
            )

        image_array = np.asarray(image_canvas, dtype=np.float32) / 255.0
        image_array = image_array.transpose(2, 0, 1)

        mask_array = np.asarray(mask_canvas, dtype=np.uint8)

        if index < 3:
            print(
                f"Sample {index}: "
                f"image_id={image_id}, "
                f"mask_shape={mask_array.shape}, "
                f"unique_values={np.unique(mask_array)}"
            )

        return (
            torch.tensor(image_array, dtype=torch.float32),
            torch.tensor(mask_array, dtype=torch.uint8),
            torch.tensor(resized_h, dtype=torch.int32),
            torch.tensor(resized_w, dtype=torch.int32),
            torch.tensor(top, dtype=torch.int32),
            torch.tensor(left, dtype=torch.int32),
            image_id,
        )


# ============================================================
# Checkpoint loading
# ============================================================
def load_checkpoint_flexible(model: torch.nn.Module, checkpoint_path: str, device):
    """
    Load checkpoint weights flexibly.

    Supports:
        1. raw state_dict;
        2. {"state_dict": state_dict};
        3. {"model_state_dict": state_dict};
        4. DataParallel checkpoints with "module." prefix.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: checkpoint does not exist: {checkpoint_path}. Skip loading.")
        return

    print(f"Load weights from: {checkpoint_path}")

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
    loaded_keys = []
    skipped_keys = []

    for key, value in normalized_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state, strict=False)

    print("\nSuccessful Load Key:", str(loaded_keys)[:500], "……")
    print("Successful Load Key Num:", len(loaded_keys))

    if skipped_keys:
        print("\nFail To Load Key:", str(skipped_keys)[:500], "……")
        print("Fail To Load Key Num:", len(skipped_keys))

    print(
        "\n\033[1;33;44mNote: it is normal if the output head is not loaded; "
        "it is usually problematic if many backbone keys are not loaded.\033[0m"
    )


# ============================================================
# Main evaluation
# ============================================================
if __name__ == "__main__":
    validate_input_shape(INPUT_SHAPE)

    seed_everything(SEED)

    if USE_DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and USE_CUDA else "cpu"
    )

    print(f"Using device: {device}")

    run_tag = build_run_tag()
    print(f"[Run Tag] {run_tag}")

    model = LSNetBiFoSTSegmentation(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
    ).to(device)

    model.eval()

    load_checkpoint_flexible(model, MODEL_PATH, device)

    voc2007_dir = os.path.join(VOC_ROOT, "VOC2007")
    image_dir = os.path.join(voc2007_dir, "JPEGImages")
    mask_dir = os.path.join(voc2007_dir, "SegmentationClass")
    image_set_file = os.path.join(VOC_ROOT, IMAGE_SET)

    for path in [voc2007_dir, image_dir, mask_dir, image_set_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

    val_dataset = VOCSegmentationEvalDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        input_shape=INPUT_SHAPE,
        image_set_file=image_set_file,
        target_label_value=TARGET_LABEL_VALUE,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Validation set loaded. Samples: {len(val_dataset)}")
    print(f"Target label value: {val_dataset.target_label_value}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_intersection = 0
    total_union = 0

    per_image_metrics = []

    print(f"Start evaluation: {len(val_loader)} batches, batch_size={BATCH_SIZE}")

    use_amp = USE_FP16 and device.type == "cuda"

    with torch.no_grad():
        for batch_index, batch_data in enumerate(tqdm(val_loader)):
            images, masks, resized_h, resized_w, top, left, image_ids = batch_data

            images = images.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

            batch_size = predictions.shape[0]

            for sample_index in range(batch_size):
                image_id = image_ids[sample_index]

                h = int(resized_h[sample_index].item())
                w = int(resized_w[sample_index].item())
                y0 = int(top[sample_index].item())
                x0 = int(left[sample_index].item())

                prediction = predictions[sample_index]
                prediction = prediction[y0: y0 + h, x0: x0 + w]

                ground_truth = masks[sample_index].numpy()
                ground_truth = ground_truth[y0: y0 + h, x0: x0 + w]

                original = images[sample_index].detach().cpu().numpy().transpose(1, 2, 0)
                original = (original * 255).astype(np.uint8)
                original = original[y0: y0 + h, x0: x0 + w]

                metrics = calculate_instance_metrics(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    target_label_value=val_dataset.target_label_value,
                    foreground_class_id=FOREGROUND_CLASS_ID,
                    match_iou_threshold=MATCH_IOU_THRESHOLD,
                    min_area=INSTANCE_MIN_AREA,
                )

                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]

                intersection, union = calculate_pixel_intersection_union(
                    metrics["pred_binary"],
                    metrics["gt_binary"],
                )

                total_intersection += intersection
                total_union += union

                image_iou = safe_divide(intersection, union)

                per_image_metrics.append(
                    {
                        "image_id": image_id,
                        "num_gt_objects": metrics["num_gt_objects"],
                        "num_pred_objects": metrics["num_pred_objects"],
                        "tp": metrics["tp"],
                        "fp": metrics["fp"],
                        "fn": metrics["fn"],
                        "iou": image_iou,
                    }
                )

                global_image_index = len(per_image_metrics)

                if global_image_index <= 5 or global_image_index % 10 == 0:
                    print(
                        f"Image {global_image_index}/{len(val_dataset)}: "
                        f"id={image_id}, "
                        f"GT={metrics['num_gt_objects']}, "
                        f"Pred={metrics['num_pred_objects']}, "
                        f"TP={metrics['tp']}, "
                        f"FP={metrics['fp']}, "
                        f"FN={metrics['fn']}, "
                        f"IoU={image_iou:.4f}"
                    )

                visualization_path = os.path.join(
                    VISUALIZATION_DIR,
                    f"{run_tag}_{image_id}.png",
                )

                visualize_and_save(
                    original_image=original,
                    metrics=metrics,
                    save_path=visualization_path,
                )

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    f1_score = safe_divide(2.0 * precision * recall, precision + recall)

    overall_iou = safe_divide(total_intersection, total_union)

    mean_image_iou = (
        float(np.mean([item["iou"] for item in per_image_metrics]))
        if len(per_image_metrics) > 0
        else 0.0
    )

    print("\n==== Segmentation Metrics ====")
    print(f"Total GT instances: {val_dataset.total_gt_objects}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1_score:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    print(f"Mean Image IoU: {mean_image_iou:.4f}")
    print(f"Target label value: {val_dataset.target_label_value}")
    print(f"Instance match IoU threshold: {MATCH_IOU_THRESHOLD}")
    print(f"Instance min area: {INSTANCE_MIN_AREA}")
    print(f"Ablation switches: BIE={USE_BIE}, HPA={USE_HPA}")

    with open(METRICS_TXT, "w", encoding="utf-8") as file:
        file.write(f"[Run Tag] {run_tag}\n")
        file.write(f"Total_GT_instances={val_dataset.total_gt_objects}\n")
        file.write(f"TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        file.write(f"Precision={precision:.4f}\n")
        file.write(f"Recall={recall:.4f}\n")
        file.write(f"F1={f1_score:.4f}\n")
        file.write(f"Overall_IoU={overall_iou:.4f}\n")
        file.write(f"Mean_Image_IoU={mean_image_iou:.4f}\n")
        file.write(f"TARGET_LABEL_VALUE={val_dataset.target_label_value}\n")
        file.write(f"MATCH_IOU_THRESHOLD={MATCH_IOU_THRESHOLD}\n")
        file.write(f"INSTANCE_MIN_AREA={INSTANCE_MIN_AREA}\n")
        file.write(f"Ablation: BIE={USE_BIE}, HPA={USE_HPA}\n\n")

        file.write("Configuration:\n")
        file.write(f"SEED={SEED}\n")
        file.write(f"USE_CUDA={USE_CUDA}, USE_FP16={USE_FP16}\n")
        file.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        file.write(f"FOREGROUND_CLASS_ID={FOREGROUND_CLASS_ID}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        file.write(f"VOC_ROOT={VOC_ROOT}\n")
        file.write(f"IMAGE_SET={IMAGE_SET}\n")

    with open(METRICS_CSV, "w", newline="", encoding="utf-8-sig") as csv_file:
        fieldnames = [
            "image_id",
            "num_gt_objects",
            "num_pred_objects",
            "tp",
            "fp",
            "fn",
            "iou",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for item in per_image_metrics:
            writer.writerow(item)

    print(f"\nDetailed metrics saved to: {METRICS_TXT}")
    print(f"Per-image metrics saved to: {METRICS_CSV}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
