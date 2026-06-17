import csv
import gc
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nets.lsnet_bifost_height_regression import LSNetBiFoSTHeightRegression
from utils.utils import seed_everything, worker_init_fn


# ============================================================
# Matplotlib settings
# ============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
    }
)


# ============================================================
# Manual configuration
# ============================================================

# Checkpoint
MODEL_PATH = r"logs_reg_rgb_mask/focal_s_svit1_bie1_f161_f321/best_epoch_weights.pth"

# Dataset
DATASET_PATH = r"TreeHeightDataset"
VAL_LIST_TXT = r"val.txt"

RGB_DIR = "rgb"
MASK_DIR = "mask"

RGB_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"

# Model structure; must match the training checkpoint
BACKBONE = "focal_s"          # focal_t / focal_s / focal_b
NUM_CLASSES = 1               # height regression output channel
IN_CHANNELS = 4               # RGB(3) + crown mask(1)

USE_BIE = True
USE_SVIT = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True
SVIT_STOKEN_SIZE = (4, 4)
SVIT_HEADS = 8
SVIT_N_ITER = 1

# Input and device
INPUT_SHAPE = (640, 640)      # (H, W)
BATCH_SIZE = 1                # Grad-CAM is recommended to use batch_size=1
NUM_WORKERS = 0
USE_CUDA = True
SEED = 11

# Preprocessing mode:
#   center_crop_pad: consistent with height-regression validation pipeline
#   resize: directly resize image and mask to INPUT_SHAPE
PREPROCESS_MODE = "center_crop_pad"

# Grad-CAM settings
MAX_IMAGES = 236

# score mode:
#   masked_mean: mean predicted height over crown mask
#   masked_max : maximum predicted height over crown mask
#   all_mean   : mean predicted height over whole map
SCORE_MODE = "masked_mean"

# Target layers.
# You may keep several layers for comparison.
# Recommended:
#   encoder.stage3 : high-level FocalNet feature at /16
#   encoder.stage4 : high-level FocalNet feature at /32
#   svit16         : SVIT output at /16, if enabled
#   svit32         : SVIT output at /32, if enabled
#   up3.conv       : decoder middle-level feature
#   up2.conv       : decoder fine-level feature
#   out_head       : high-resolution feature before final regression head
TARGET_LAYER_NAMES = [
    "encoder.stage3",
    "encoder.stage4",
    "svit16",
    "svit32",
    "up4.conv",
    "up3.conv",
    "up2.conv",
    "up1.conv",
    "out_head",
]

# Output
SAVE_ROOT = "gradcam_height_valset"
SAVE_RGB_PER_LAYER = True
SAVE_SCORE_CSV = True

SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# ============================================================
# File and path utilities
# ============================================================
def sanitize_filename(name: str) -> str:
    """Make filename safe for Windows/Linux."""
    illegal = '<>:"/\\|?*'
    for char in illegal:
        name = name.replace(char, "_")
    return name.rstrip(" .")


def sanitize_layer_name(layer_name: str) -> str:
    """Convert module path to folder-friendly name."""
    return sanitize_filename(layer_name.replace(".", "_"))


def read_split_file(split_path: str) -> List[str]:
    """Read image names from split file."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    try:
        with open(split_path, "r", encoding="utf-8") as file:
            lines = [line.strip().split()[0] for line in file if line.strip()]
    except UnicodeDecodeError:
        with open(split_path, "r", encoding="gbk", errors="ignore") as file:
            lines = [line.strip().split()[0] for line in file if line.strip()]

    return lines


def find_file_by_prefix(folder: str, prefix: str, preferred_suffix: str = None):
    """Find file by prefix and optional preferred suffix."""
    if not os.path.isdir(folder):
        return None

    if preferred_suffix is not None:
        candidate = os.path.join(folder, prefix + preferred_suffix)
        if os.path.exists(candidate):
            return candidate

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, prefix + extension)
        if os.path.exists(candidate):
            return candidate

    for file_name in os.listdir(folder):
        if Path(file_name).stem == prefix:
            return os.path.join(folder, file_name)

    return None


def validate_input_shape(input_shape: Tuple[int, int]):
    """Validate input shape."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width] or (height, width).")

    height, width = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError("Both height and width in INPUT_SHAPE must be divisible by 32.")


# ============================================================
# Dataset
# ============================================================
class RgbMaskGradCAMDataset(Dataset):
    """
    Dataset for Grad-CAM explanation of the height-regression branch.

    Input:
        RGB image + crown mask -> 4-channel tensor.
    """

    def __init__(
        self,
        image_names: List[str],
        dataset_path: str,
        input_shape: Tuple[int, int],
        rgb_dir: str = "rgb",
        mask_dir: str = "mask",
        rgb_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        preprocess_mode: str = "center_crop_pad",
    ):
        super().__init__()

        self.image_names = image_names
        self.dataset_path = dataset_path
        self.input_shape = input_shape

        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir

        self.rgb_suffix = rgb_suffix
        self.mask_suffix = mask_suffix

        self.preprocess_mode = preprocess_mode

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]

        rgb_folder = os.path.join(self.dataset_path, self.rgb_dir)
        mask_folder = os.path.join(self.dataset_path, self.mask_dir)

        rgb_path = find_file_by_prefix(
            folder=rgb_folder,
            prefix=image_name,
            preferred_suffix=self.rgb_suffix,
        )
        mask_path = find_file_by_prefix(
            folder=mask_folder,
            prefix=image_name,
            preferred_suffix=self.mask_suffix,
        )

        if rgb_path is None:
            raise FileNotFoundError(f"RGB image not found for {image_name} in {rgb_folder}")
        if mask_path is None:
            raise FileNotFoundError(f"Mask image not found for {image_name} in {mask_folder}")

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

        mask_channel = np.expand_dims(mask, axis=-1)
        input_4ch = np.concatenate([rgb, mask_channel], axis=-1)
        input_4ch = np.transpose(input_4ch, (2, 0, 1))

        meta["rgb_path"] = rgb_path
        meta["mask_path"] = mask_path
        meta["image_name"] = image_name

        return torch.from_numpy(input_4ch).float(), image_name, meta

    @staticmethod
    def resize_to_input_shape(rgb, mask, target_size, original_h, original_w):
        target_h, target_w = target_size

        rgb_uint8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        rgb_resized = Image.fromarray(rgb_uint8).resize((target_w, target_h), Image.BILINEAR)

        mask_uint8 = np.uint8(mask * 255)
        mask_resized = Image.fromarray(mask_uint8).resize((target_w, target_h), Image.NEAREST)

        rgb = np.array(rgb_resized, dtype=np.float32) / 255.0
        mask = (np.array(mask_resized, dtype=np.uint8) > 0).astype(np.float32)

        meta = {
            "mode": "resize",
            "original_h": original_h,
            "original_w": original_w,
            "target_h": target_h,
            "target_w": target_w,
            "crop_top": 0,
            "crop_left": 0,
            "pad_top": 0,
            "pad_left": 0,
        }

        return rgb, mask, meta

    @staticmethod
    def pad_to_size(rgb, mask, target_size):
        target_h, target_w = target_size
        h, w, _ = rgb.shape

        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_h == 0 and pad_w == 0:
            return rgb, mask, {
                "pad_top": 0,
                "pad_bottom": 0,
                "pad_left": 0,
                "pad_right": 0,
            }

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


def gradcam_collate(batch):
    images, image_names, metas = zip(*batch)
    return torch.stack(images, dim=0), list(image_names), list(metas)


# ============================================================
# Model utilities
# ============================================================
def set_inplace_false(model: nn.Module):
    """Disable inplace activations to make backward hooks safer."""
    for module in model.modules():
        if hasattr(module, "inplace") and module.inplace:
            module.inplace = False


def get_module_by_name(model: nn.Module, module_name: str):
    """Get submodule by dot-separated name."""
    current_module = model

    for name in module_name.split("."):
        if not hasattr(current_module, name):
            return None
        current_module = getattr(current_module, name)

    if not isinstance(current_module, nn.Module):
        return None

    return current_module


def filter_existing_target_layers(model: nn.Module, target_layer_names: List[str]):
    """Keep only existing layers."""
    target_layers = []

    for layer_name in target_layer_names:
        module = get_module_by_name(model, layer_name)

        if module is None:
            print(f"[Skip] target layer not found or disabled: {layer_name}")
            continue

        target_layers.append((layer_name, module))

    if len(target_layers) == 0:
        available_names = [name for name, _ in model.named_modules()]
        preview = "\n".join(available_names[:160])
        raise RuntimeError(
            "No valid target layer found. Please check TARGET_LAYER_NAMES.\n"
            f"Available module names include:\n{preview}"
        )

    return target_layers


def load_checkpoint_flexible(model: nn.Module, checkpoint_path: str, device):
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
    """Build model and load checkpoint."""
    model = LSNetBiFoSTHeightRegression(
        num_classes=NUM_CLASSES,
        pretrained=False,
        backbone=BACKBONE,
        in_channels=IN_CHANNELS,
        use_bie=USE_BIE,
        use_svit=USE_SVIT,
        svit_on_f16=SVIT_ON_F16,
        svit_on_f32=SVIT_ON_F32,
        svit_stoken_size=SVIT_STOKEN_SIZE,
        svit_heads=SVIT_HEADS,
        svit_n_iter=SVIT_N_ITER,
    )

    set_inplace_false(model)
    load_checkpoint_flexible(model, MODEL_PATH, device)

    model = model.to(device)
    model.eval()

    if hasattr(model, "get_ablation_config"):
        print("[Model ablation config]:", model.get_ablation_config())

    return model


# ============================================================
# Grad-CAM for regression
# ============================================================
class GradCAMRegression:
    """
    Grad-CAM for height-regression output.

    The scalar target score can be:
        masked_mean: mean predicted height over crown mask
        masked_max : maximum predicted height over crown mask
        all_mean   : mean predicted height over whole output map
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, layer_name: str):
        self.model = model
        self.target_layer = target_layer
        self.layer_name = layer_name

        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        gradient = grad_output[0]
        self.gradients = gradient

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def build_score(self, prediction: torch.Tensor, input_tensor: torch.Tensor, score_mode: str):
        """
        Build scalar target score for regression Grad-CAM.
        """
        if prediction.dim() == 3:
            prediction = prediction.unsqueeze(1)

        crown_mask = (input_tensor[:, 3:4, :, :] > 0.5).float()

        if score_mode == "masked_max":
            if crown_mask.sum() > 0:
                masked_prediction = prediction.clone()
                masked_prediction[crown_mask < 0.5] = -1e6
                score = masked_prediction.max()
            else:
                score = prediction.max()

        elif score_mode == "masked_mean":
            if crown_mask.sum() > 0:
                denominator = crown_mask.sum().clamp_min(1.0)
                score = (prediction * crown_mask).sum() / denominator
            else:
                score = prediction.mean()

        elif score_mode == "all_mean":
            score = prediction.mean()

        else:
            raise ValueError(
                f"Unsupported SCORE_MODE: {score_mode}. "
                "Choose from masked_mean, masked_max, all_mean."
            )

        return score

    def generate(self, input_tensor: torch.Tensor, score_mode: str = "masked_mean"):
        """
        Args:
            input_tensor: [1, 4, H, W]

        Returns:
            cam: [H, W], normalized to [0, 1]
            score: scalar regression score used for backpropagation
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("Grad-CAM requires batch_size=1.")
        if input_tensor.size(1) < 4:
            raise ValueError("Height-regression Grad-CAM requires 4-channel input: RGB + mask.")

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        prediction = self.model(input_tensor)

        score = self.build_score(
            prediction=prediction,
            input_tensor=input_tensor,
            score_mode=score_mode,
        )

        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                f"Grad-CAM failed for layer {self.layer_name}: "
                "activations or gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.clamp(0.0, 1.0)

        return cam.detach().cpu().numpy().astype(np.float32), float(score.detach().cpu().item())


# ============================================================
# Visualization utilities
# ============================================================
def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert [C,H,W] tensor to RGB uint8 image."""
    image = image_tensor.detach().cpu().numpy()
    image = image[:3]
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)

    return (image * 255).astype(np.uint8)


def colorize_cam(cam: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Colorize CAM array in [0,1] to RGB heatmap."""
    cam = np.clip(cam, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    heatmap = (cmap(cam)[..., :3] * 255).astype(np.uint8)

    return heatmap


def overlay_cam_on_rgb(cam: np.ndarray, rgb_image: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay CAM heatmap on RGB image."""
    heatmap = colorize_cam(cam, cmap_name="magma")
    overlay = (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay


def save_rgb_image(image: np.ndarray, save_path: str):
    """Save RGB image without axis."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(3.2, 3.2), dpi=300)
    axis = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    axis.imshow(image)
    axis.axis("off")
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_cam_with_colorbar(cam: np.ndarray, save_path: str):
    """Save CAM with colorbar."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(3.4, 3.2), dpi=300)
    axis = fig.add_axes([0.0, 0.0, 0.82, 1.0])
    axis.axis("off")

    image = axis.imshow(cam, cmap="magma", vmin=0.0, vmax=1.0)

    colorbar_axis = fig.add_axes([0.85, 0.08, 0.04, 0.84])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.ax.tick_params(labelsize=7)

    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ============================================================
# Main pipeline
# ============================================================
def run_gradcam_height_on_valset():
    validate_input_shape(INPUT_SHAPE)
    seed_everything(SEED)

    device = torch.device("cuda" if (USE_CUDA and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    print("\n[Height-regression Grad-CAM configuration]")
    print(f"  model_path      = {MODEL_PATH}")
    print(f"  dataset_path    = {DATASET_PATH}")
    print(f"  val_list_txt    = {VAL_LIST_TXT}")
    print(f"  backbone        = {BACKBONE}")
    print(f"  in_channels     = {IN_CHANNELS}")
    print(f"  use_svit        = {USE_SVIT}")
    print(f"  use_bie         = {USE_BIE}")
    print(f"  svit_on_f16     = {SVIT_ON_F16}")
    print(f"  svit_on_f32     = {SVIT_ON_F32}")
    print(f"  score_mode      = {SCORE_MODE}")
    print(f"  preprocess_mode = {PREPROCESS_MODE}\n")

    model = build_model(device)

    target_layers = filter_existing_target_layers(
        model=model,
        target_layer_names=TARGET_LAYER_NAMES,
    )

    print("Target layers:")
    for layer_name, module in target_layers:
        print(f"  - {layer_name}: {module.__class__.__name__}")

    val_list_path = VAL_LIST_TXT if os.path.isabs(VAL_LIST_TXT) else os.path.join(DATASET_PATH, VAL_LIST_TXT)
    val_names = read_split_file(val_list_path)

    if MAX_IMAGES is not None:
        val_names = val_names[:MAX_IMAGES]

    print(f"\nNumber of images to explain: {len(val_names)}")

    dataset = RgbMaskGradCAMDataset(
        image_names=val_names,
        dataset_path=DATASET_PATH,
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
        collate_fn=gradcam_collate,
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None,
    )

    os.makedirs(SAVE_ROOT, exist_ok=True)

    rgb_root = os.path.join(SAVE_ROOT, "rgb")
    cam_root = os.path.join(SAVE_ROOT, "cam")
    overlay_root = os.path.join(SAVE_ROOT, "overlay")

    os.makedirs(rgb_root, exist_ok=True)
    os.makedirs(cam_root, exist_ok=True)
    os.makedirs(overlay_root, exist_ok=True)

    for layer_name, _ in target_layers:
        safe_layer_name = sanitize_layer_name(layer_name)

        if SAVE_RGB_PER_LAYER:
            os.makedirs(os.path.join(rgb_root, safe_layer_name), exist_ok=True)

        os.makedirs(os.path.join(cam_root, safe_layer_name), exist_ok=True)
        os.makedirs(os.path.join(overlay_root, safe_layer_name), exist_ok=True)

    score_records = []

    processed_count = 0

    print("\nStart generating Grad-CAM maps for height regression...")

    for images, image_names, metas in tqdm(dataloader, desc="Grad-CAM height"):
        if images.size(0) != 1:
            raise ValueError("This script requires BATCH_SIZE=1 for Grad-CAM.")

        input_tensor = images.to(device, dtype=torch.float32)
        image_name = image_names[0]
        safe_image_name = sanitize_filename(image_name)

        rgb_image = tensor_to_rgb(images[0])

        for layer_name, layer_module in target_layers:
            safe_layer_name = sanitize_layer_name(layer_name)

            gradcam = GradCAMRegression(
                model=model,
                target_layer=layer_module,
                layer_name=layer_name,
            )

            cam, score = gradcam.generate(
                input_tensor=input_tensor,
                score_mode=SCORE_MODE,
            )

            overlay = overlay_cam_on_rgb(
                cam=cam,
                rgb_image=rgb_image,
                alpha=0.55,
            )

            if SAVE_RGB_PER_LAYER:
                rgb_save_path = os.path.join(
                    rgb_root,
                    safe_layer_name,
                    f"{safe_image_name}.png",
                )
            else:
                rgb_save_path = os.path.join(
                    rgb_root,
                    f"{safe_image_name}.png",
                )

            cam_save_path = os.path.join(
                cam_root,
                safe_layer_name,
                f"{safe_image_name}.png",
            )
            overlay_save_path = os.path.join(
                overlay_root,
                safe_layer_name,
                f"{safe_image_name}.png",
            )

            save_rgb_image(rgb_image, rgb_save_path)
            save_cam_with_colorbar(cam, cam_save_path)
            save_rgb_image(overlay, overlay_save_path)

            score_records.append(
                {
                    "image_name": image_name,
                    "layer_name": layer_name,
                    "score_mode": SCORE_MODE,
                    "score": score,
                }
            )

            gradcam.remove_hooks()

            del gradcam, cam, overlay
            gc.collect()

            if device.type == "cuda":
                torch.cuda.empty_cache()

        processed_count += 1

    if SAVE_SCORE_CSV:
        csv_path = os.path.join(SAVE_ROOT, "gradcam_scores.csv")

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["image_name", "layer_name", "score_mode", "score"],
            )
            writer.writeheader()
            for record in score_records:
                writer.writerow(record)

    config_path = os.path.join(SAVE_ROOT, "gradcam_config.txt")
    with open(config_path, "w", encoding="utf-8") as file:
        file.write("==== Height-Regression Grad-CAM Configuration ====\n")
        file.write(f"MODEL_PATH={MODEL_PATH}\n")
        file.write(f"DATASET_PATH={DATASET_PATH}\n")
        file.write(f"VAL_LIST_TXT={VAL_LIST_TXT}\n")
        file.write(f"RGB_DIR={RGB_DIR}\n")
        file.write(f"MASK_DIR={MASK_DIR}\n")
        file.write(f"RGB_SUFFIX={RGB_SUFFIX}\n")
        file.write(f"MASK_SUFFIX={MASK_SUFFIX}\n")
        file.write(f"BACKBONE={BACKBONE}\n")
        file.write(f"NUM_CLASSES={NUM_CLASSES}\n")
        file.write(f"IN_CHANNELS={IN_CHANNELS}\n")
        file.write(f"USE_BIE={USE_BIE}\n")
        file.write(f"USE_SVIT={USE_SVIT}\n")
        file.write(f"SVIT_ON_F16={SVIT_ON_F16}\n")
        file.write(f"SVIT_ON_F32={SVIT_ON_F32}\n")
        file.write(f"SVIT_STOKEN_SIZE={SVIT_STOKEN_SIZE}\n")
        file.write(f"SVIT_HEADS={SVIT_HEADS}\n")
        file.write(f"SVIT_N_ITER={SVIT_N_ITER}\n")
        file.write(f"INPUT_SHAPE={INPUT_SHAPE}\n")
        file.write(f"SCORE_MODE={SCORE_MODE}\n")
        file.write(f"PREPROCESS_MODE={PREPROCESS_MODE}\n")
        file.write(f"TARGET_LAYER_NAMES={TARGET_LAYER_NAMES}\n")
        file.write(f"MAX_IMAGES={MAX_IMAGES}\n")

    print(f"\n[Done] Processed {processed_count} images.")
    print(f"Results saved to: {os.path.abspath(SAVE_ROOT)}")
    print(f"  - RGB     : {os.path.abspath(rgb_root)}")
    print(f"  - CAM     : {os.path.abspath(cam_root)}")
    print(f"  - Overlay : {os.path.abspath(overlay_root)}")
    print(f"  - Scores  : {os.path.abspath(os.path.join(SAVE_ROOT, 'gradcam_scores.csv'))}")


if __name__ == "__main__":
    run_gradcam_height_on_valset()
