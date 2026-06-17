import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

from nets.lsnet_bifost_segmentation import LSNetBiFoSTSegmentation
from utils.utils import seed_everything


# ============================================================
# Manual configuration
# ============================================================

SEED = 11
USE_CUDA = True
USE_FP16 = False
USE_DETERMINISTIC = True

# ---------------- Model setting ----------------
NUM_CLASSES = 2
FOREGROUND_CLASS_ID = 1

BACKBONE = "lsnet_b"
PRETRAINED = False
MODEL_PATH = r"model_data/best_epoch_weights.pth"
INPUT_SHAPE = [640, 640]

USE_BIE = 1
USE_HPA = 1

# ---------------- Input / output folders ----------------
INPUT_IMAGE_DIR = r"input_images"
OUTPUT_DIR = r"prediction_results"

# ---------------- Prediction setting ----------------
PREDICTION_MODE = "argmax"       # argmax / threshold
FOREGROUND_THRESHOLD = 0.5
RECURSIVE = False

SAVE_BINARY_MASK = True
SAVE_PROBABILITY = True
SAVE_COLOR_MASK = True
SAVE_OVERLAY = True
SAVE_RAW_NPY = False

# 红色掩膜透明度
OVERLAY_ALPHA = 0.35

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# Basic utilities
# ============================================================
def validate_input_shape(input_shape: List[int]):
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width].")

    height, width = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Height and width in INPUT_SHAPE must be divisible by 32.")


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = name.replace("（", "(").replace("）", ")")

    illegal_chars = '<>:"/\\|?*'
    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


def collect_image_paths(image_dir: str, recursive: bool = False) -> List[str]:
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Input image folder does not exist: {image_dir}")

    image_paths = []

    if recursive:
        for extension in SUPPORTED_EXTENSIONS:
            image_paths.extend(image_dir.rglob(f"*{extension}"))
            image_paths.extend(image_dir.rglob(f"*{extension.upper()}"))
    else:
        for extension in SUPPORTED_EXTENSIONS:
            image_paths.extend(image_dir.glob(f"*{extension}"))
            image_paths.extend(image_dir.glob(f"*{extension.upper()}"))

    image_paths = sorted(list(set([str(path) for path in image_paths])))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    return image_paths


# ============================================================
# Image preprocessing
# ============================================================
def letterbox_image(image: Image.Image, size_hw: Tuple[int, int]):
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


def preprocess_image(image_path: str, input_shape: List[int]):
    image = Image.open(image_path).convert("RGB")

    original_w, original_h = image.size

    canvas, resized_w, resized_h, top, left = letterbox_image(
        image,
        (input_shape[0], input_shape[1]),
    )

    image_array = np.asarray(canvas, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0)

    meta = {
        "image_path": image_path,
        "image_name": Path(image_path).stem,
        "original_w": original_w,
        "original_h": original_h,
        "resized_w": resized_w,
        "resized_h": resized_h,
        "top": top,
        "left": left,
        "letterbox_rgb": np.asarray(canvas, dtype=np.uint8),
        "original_rgb": np.asarray(image, dtype=np.uint8),
    }

    return image_tensor, meta


def remove_letterbox_and_resize(array_2d: np.ndarray, meta: Dict, interpolation) -> np.ndarray:
    top = meta["top"]
    left = meta["left"]
    resized_h = meta["resized_h"]
    resized_w = meta["resized_w"]
    original_h = meta["original_h"]
    original_w = meta["original_w"]

    valid = array_2d[top: top + resized_h, left: left + resized_w]

    restored = cv2.resize(
        valid,
        (original_w, original_h),
        interpolation=interpolation,
    )

    return restored


# ============================================================
# Model loading
# ============================================================
def load_checkpoint_flexible(model: torch.nn.Module, checkpoint_path: str, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

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

    print(f"Successful Load Key Num: {len(loaded_keys)}")
    print(f"Fail To Load Key Num: {len(skipped_keys)}")

    if len(skipped_keys) > 0:
        print("Skipped key examples:", skipped_keys[:10])


def build_model(device):
    model = LSNetBiFoSTSegmentation(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
    ).to(device)

    model.eval()

    load_checkpoint_flexible(model, MODEL_PATH, device)

    return model


# ============================================================
# Prediction utilities
# ============================================================
def predict_one_image(model, image_tensor, device, use_fp16=False):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(image_tensor)
        else:
            outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        foreground_prob = probabilities[:, FOREGROUND_CLASS_ID, :, :]

        if PREDICTION_MODE == "threshold":
            pred_class = (foreground_prob >= FOREGROUND_THRESHOLD).long()
        elif PREDICTION_MODE == "argmax":
            pred_class = torch.argmax(probabilities, dim=1)
        else:
            raise ValueError("PREDICTION_MODE must be 'argmax' or 'threshold'.")

    pred_class_np = pred_class[0].detach().cpu().numpy().astype(np.uint8)
    foreground_prob_np = foreground_prob[0].detach().cpu().numpy().astype(np.float32)

    return pred_class_np, foreground_prob_np


def make_color_mask(binary_mask: np.ndarray) -> np.ndarray:
    """
    Convert binary mask to RGB color mask.

    Background: black
    Foreground: red
    """
    color_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    # 前景统一使用红色掩膜
    color_mask[binary_mask > 0] = [255, 0, 0]

    return color_mask


def make_overlay(original_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Overlay red predicted crown mask on original RGB image.
    """
    color_mask = make_color_mask(binary_mask)

    overlay = original_rgb.astype(np.float32).copy()
    foreground = binary_mask > 0

    overlay[foreground] = (
        original_rgb[foreground].astype(np.float32) * (1.0 - alpha)
        + color_mask[foreground].astype(np.float32) * alpha
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def save_gray_png(array_2d: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(array_2d.astype(np.uint8), mode="L").save(save_path)


def save_rgb_png(array_rgb: np.ndarray, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(array_rgb.astype(np.uint8), mode="RGB").save(save_path)


def save_visualization(original_rgb, binary_mask, probability_map, overlay_rgb, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 二值掩膜也用红色显示
    red_mask = make_color_mask(binary_mask)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(red_mask)
    axes[1].set_title("Predicted Mask (Red)")
    axes[1].axis("off")

    im = axes[2].imshow(probability_map, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Foreground Probability")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Red Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main prediction
# ============================================================
def run_folder_prediction():
    validate_input_shape(INPUT_SHAPE)
    seed_everything(SEED)

    if USE_DETERMINISTIC:
        cudnn.benchmark = False
        cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and USE_CUDA else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Input folder: {INPUT_IMAGE_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Prediction mode: {PREDICTION_MODE}")
    print("Foreground mask color: red")

    model = build_model(device)

    image_paths = collect_image_paths(INPUT_IMAGE_DIR, recursive=RECURSIVE)
    print(f"Found images: {len(image_paths)}")

    mask_dir = os.path.join(OUTPUT_DIR, "binary_masks")
    prob_dir = os.path.join(OUTPUT_DIR, "probability_maps")
    color_dir = os.path.join(OUTPUT_DIR, "red_masks")
    overlay_dir = os.path.join(OUTPUT_DIR, "red_overlays")
    visual_dir = os.path.join(OUTPUT_DIR, "visualizations")
    raw_dir = os.path.join(OUTPUT_DIR, "raw_npy")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SAVE_BINARY_MASK:
        os.makedirs(mask_dir, exist_ok=True)
    if SAVE_PROBABILITY:
        os.makedirs(prob_dir, exist_ok=True)
    if SAVE_COLOR_MASK:
        os.makedirs(color_dir, exist_ok=True)
    if SAVE_OVERLAY:
        os.makedirs(overlay_dir, exist_ok=True)
    if SAVE_RAW_NPY:
        os.makedirs(raw_dir, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Predicting folder images"):
        image_tensor, meta = preprocess_image(image_path, INPUT_SHAPE)

        pred_class, foreground_prob = predict_one_image(
            model=model,
            image_tensor=image_tensor,
            device=device,
            use_fp16=USE_FP16,
        )

        pred_class_original = remove_letterbox_and_resize(
            pred_class,
            meta,
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)

        foreground_prob_original = remove_letterbox_and_resize(
            foreground_prob,
            meta,
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)

        binary_mask = (pred_class_original == FOREGROUND_CLASS_ID).astype(np.uint8)
        binary_mask_255 = binary_mask * 255

        probability_255 = np.uint8(np.clip(foreground_prob_original, 0.0, 1.0) * 255)

        original_rgb = meta["original_rgb"]

        red_mask = make_color_mask(binary_mask)
        red_overlay = make_overlay(
            original_rgb=original_rgb,
            binary_mask=binary_mask,
            alpha=OVERLAY_ALPHA,
        )

        image_name = sanitize_filename(meta["image_name"])

        if SAVE_BINARY_MASK:
            save_gray_png(
                binary_mask_255,
                os.path.join(mask_dir, f"{image_name}_mask.png"),
            )

        if SAVE_PROBABILITY:
            save_gray_png(
                probability_255,
                os.path.join(prob_dir, f"{image_name}_prob.png"),
            )

        if SAVE_COLOR_MASK:
            save_rgb_png(
                red_mask,
                os.path.join(color_dir, f"{image_name}_red_mask.png"),
            )

        if SAVE_OVERLAY:
            save_rgb_png(
                red_overlay,
                os.path.join(overlay_dir, f"{image_name}_red_overlay.png"),
            )

        save_visualization(
            original_rgb=original_rgb,
            binary_mask=binary_mask,
            probability_map=foreground_prob_original,
            overlay_rgb=red_overlay,
            save_path=os.path.join(visual_dir, f"{image_name}_visual.png"),
        )

        if SAVE_RAW_NPY:
            np.save(
                os.path.join(raw_dir, f"{image_name}_pred_class.npy"),
                pred_class_original.astype(np.uint8),
            )
            np.save(
                os.path.join(raw_dir, f"{image_name}_foreground_prob.npy"),
                foreground_prob_original.astype(np.float32),
            )

    config_path = os.path.join(OUTPUT_DIR, "prediction_config.txt")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write("==== LSNet-BiFoST Folder Prediction Configuration ====\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"INPUT_IMAGE_DIR={INPUT_IMAGE_DIR}\n")
        file.write(f"OUTPUT_DIR={OUTPUT_DIR}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        file.write(f"FOREGROUND_CLASS_ID={FOREGROUND_CLASS_ID}\n")
        file.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        file.write(f"USE_BIE={USE_BIE}\n")
        file.write(f"USE_HPA={USE_HPA}\n")
        file.write(f"PREDICTION_MODE={PREDICTION_MODE}\n")
        file.write(f"FOREGROUND_THRESHOLD={FOREGROUND_THRESHOLD}\n")
        file.write(f"FOREGROUND_COLOR=red RGB(255,0,0)\n")
        file.write(f"OVERLAY_ALPHA={OVERLAY_ALPHA}\n")
        file.write(f"RECURSIVE={RECURSIVE}\n")

    print("\nPrediction finished.")
    print(f"Results saved to: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Binary masks: {mask_dir}")
    print(f"Probability maps: {prob_dir}")
    print(f"Red masks: {color_dir}")
    print(f"Red overlays: {overlay_dir}")
    print(f"Visualizations: {visual_dir}")


if __name__ == "__main__":
    run_folder_prediction()