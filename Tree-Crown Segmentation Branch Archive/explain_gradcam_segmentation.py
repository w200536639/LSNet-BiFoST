import os
import gc
from functools import partial
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# Matplotlib style
# ============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
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
# Project imports
# ============================================================
from nets.lsnet_bifost_segmentation import LSNetBiFoSTSegmentation
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn


# ============================================================
# Global configuration
# ============================================================

CUDA = True
SEED = 11

# Keep these consistent with train.py
NUM_CLASSES = 2
INPUT_SHAPE = [640, 640]
BACKBONE = "lsnet_b"          # lsnet_t / lsnet_s / lsnet_b
USE_BIE = True
USE_HPA = True

# Checkpoint
MODEL_PATH = r"model_data\best_epoch_weights.pth"

# Dataset
VOC_PATH = r"VOCdevkit"
VAL_TXT = os.path.join(
    VOC_PATH,
    "VOC2007",
    "ImageSets",
    "Segmentation",
    "val.txt",
)

BATCH_SIZE = 1
NUM_WORKERS = 0

# Class index
TREE_CLASS_IDX = 1            # 0 background, 1 Haloxylon crown

# 0 means all validation images.
# For debugging, set MAX_IMAGES = 5 or 10.
MAX_IMAGES = 236

# Output
SAVE_ROOT = "gradcam_lsnet_bifost"
PAPER_TAG = "LSNet-BiFoST tree-crown segmentation Grad-CAM"

# If True, a failed layer will be skipped instead of stopping the program.
SKIP_FAILED_LAYERS = True

# If True, the script will first test each target layer using the first validation image.
AUTO_FILTER_LAYERS = True


# ============================================================
# Safe title function
# ============================================================
def add_paper_title(fig, text: str, y: float = 1.02):
    """
    Add a figure-level title without using fontdict['size'].

    This avoids the Matplotlib error:
        TypeError: Got both 'size' and 'fontsize',
        which are aliases of one another.
    """
    fig.suptitle(
        text,
        fontsize=11,
        fontfamily="Times New Roman",
        y=y,
    )


# ============================================================
# Grad-CAM class
# ============================================================
class GradCAM:
    """
    Grad-CAM for segmentation.

    This version registers the gradient hook directly on the target layer
    output, which is more stable in recent PyTorch versions.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module, layer_name: str):
        self.model = model
        self.target_layer = target_layer
        self.layer_name = layer_name

        self.activations = None
        self.gradients = None
        self._last_score = None

        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            self.activations = None
            self.gradients = None
            return

        if output.dim() != 4:
            self.activations = None
            self.gradients = None
            return

        self.activations = output

        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor, class_idx: int):
        """
        Args:
            input_tensor: [1, C, H, W]
            class_idx: target class index. For tree crown, use 1.

        Returns:
            cam: [H, W], values in [0, 1].
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError("Grad-CAM requires input tensor with shape [1, C, H, W].")

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.detach()
        input_tensor.requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)

            if output.dim() == 4:
                logits = output[:, class_idx, :, :]

                probabilities = torch.softmax(output, dim=1)
                foreground_prob = probabilities[:, class_idx, :, :]

                foreground_mask = foreground_prob >= 0.5

                if foreground_mask.sum() > 0:
                    score = logits[foreground_mask].mean()
                else:
                    score = logits.mean()

            elif output.dim() == 2:
                score = output[:, class_idx].mean()
            else:
                raise RuntimeError(
                    f"Unsupported model output shape for Grad-CAM: {tuple(output.shape)}"
                )

            self._last_score = float(score.detach().cpu().item())
            score.backward(retain_graph=False)

        if self.activations is None:
            raise RuntimeError(
                f"Layer {self.layer_name}: activations were not captured. "
                "The layer may not output a 4D tensor or may not be used in forward."
            )

        if self.gradients is None:
            raise RuntimeError(
                f"Layer {self.layer_name}: gradients were not captured. "
                "The layer may not contribute to the selected target score."
            )

        activations = self.activations
        gradients = self.gradients

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1)
        cam = F.relu(cam)[0]

        cam_min = cam.min()
        cam_max = cam.max()

        cam = cam - cam_min
        if (cam_max - cam_min) > 1e-6:
            cam = cam / (cam_max - cam_min)

        cam = cam.clamp(0.0, 1.0)

        return cam.detach().cpu().numpy().astype(np.float32)

    def remove_hooks(self):
        self.fwd_handle.remove()


# ============================================================
# Helper functions
# ============================================================
def tensor_to_rgb(img_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert image tensor [C,H,W] in [0,1] to RGB uint8 image.
    """
    img = img_tensor.detach().cpu().numpy()

    if img.shape[0] > 3:
        img = img[:3]

    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    return img


def visualize_cam_nature(cam: np.ndarray, rgb_img: np.ndarray, alpha: float = 0.55):
    """
    Nature-style Grad-CAM visualization.

    Uses 'magma' colormap and creates:
        heatmap
        overlay
    """
    height, width, _ = rgb_img.shape

    cam_norm = np.clip(cam, 0.0, 1.0)
    cam_uint8 = (cam_norm * 255).astype(np.uint8)

    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize((width, height), Image.BILINEAR)
    )
    cam_resized = cam_resized.astype(np.float32) / 255.0

    cmap = plt.get_cmap("magma")
    heatmap = cmap(cam_resized)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (
        (1.0 - alpha) * rgb_img.astype(np.float32)
        + alpha * heatmap.astype(np.float32)
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return heatmap, overlay


def get_decoder_target_layers(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """
    Get decoder layers for LSNet-BiFoST segmentation branch.

    Target layers:
        up4.conv
        up3.conv
        up2.conv
        up1.conv
        out_head
        final
    """
    layers = []

    for attr_name in ["up4", "up3", "up2", "up1"]:
        if hasattr(model, attr_name):
            block = getattr(model, attr_name)
            if hasattr(block, "conv"):
                layers.append((block.conv, f"{attr_name}_conv"))

    if hasattr(model, "out_head"):
        layers.append((model.out_head, "out_head"))

    if hasattr(model, "final"):
        layers.append((model.final, "final_logits"))

    if len(layers) == 0:
        raise ValueError(
            "Cannot find decoder layers: up4/up3/up2/up1/out_head/final. "
            "Please check nets/lsnet_bifost_segmentation.py."
        )

    return layers


def get_all_recommended_layers(model: nn.Module) -> List[Tuple[nn.Module, str]]:
    """
    Recommended layers for paper visualization.

    This function keeps decoder-focused layers and also adds HPA layers if available.
    """
    layers = []

    if hasattr(model, "hpa16"):
        layers.append((model.hpa16, "hpa16"))

    if hasattr(model, "hpa32"):
        layers.append((model.hpa32, "hpa32"))

    layers.extend(get_decoder_target_layers(model))

    return layers


def load_val_list(val_path: str):
    """
    Read val.txt with encoding fallback.
    """
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(val_path, "r", encoding=encoding) as file:
                lines = [line.strip() for line in file if line.strip()]
            print(f"val.txt 使用编码: {encoding}")
            return lines
        except UnicodeDecodeError:
            continue

    with open(val_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = [line.strip() for line in file if line.strip()]

    print("警告: val.txt 无法以常规编码完全读取，已使用 utf-8 + ignore。")
    return lines


def safe_filename(name: str):
    """
    Make a safe filename for Windows.
    """
    name = name.replace("（", "(").replace("）", ")")
    illegal_chars = '<>:"/\\|?*'

    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


def load_checkpoint_flexible(model: nn.Module, model_path: str, device):
    """
    Flexible checkpoint loading.

    Supports:
        raw state_dict
        {'state_dict': state_dict}
        {'model_state_dict': state_dict}
        DataParallel state_dict with module. prefix
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    print(f"加载模型权重: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    clean_state = {}

    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith("module.") else key
        clean_state[new_key] = value

    model_state = model.state_dict()
    load_keys = []
    skip_keys = []

    for key, value in clean_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            load_keys.append(key)
        else:
            skip_keys.append(key)

    model.load_state_dict(model_state, strict=False)

    print(f"  成功加载 {len(load_keys)} 个权重，跳过 {len(skip_keys)} 个。")

    if len(skip_keys) > 0:
        print("  跳过的 key 示例:", skip_keys[:10])


def build_model(device):
    """
    Build LSNet-BiFoST segmentation model.
    """
    model = LSNetBiFoSTSegmentation(
        num_classes=NUM_CLASSES,
        pretrained=False,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
    )

    for module in model.modules():
        if hasattr(module, "inplace") and module.inplace:
            module.inplace = False

    load_checkpoint_flexible(model, MODEL_PATH, device)

    model.to(device)
    model.eval()

    return model


def ensure_tensor(images):
    """
    Ensure DataLoader output is torch.Tensor.

    This makes the script compatible with both old and new dataloader.py.
    """
    if isinstance(images, torch.Tensor):
        return images

    if isinstance(images, np.ndarray):
        return torch.from_numpy(images)

    return torch.as_tensor(images)


def test_layer_available(
    model: nn.Module,
    target_layer: nn.Module,
    layer_name: str,
    input_tensor: torch.Tensor,
):
    """
    Test whether a target layer can produce Grad-CAM.
    """
    cam_engine = GradCAM(model, target_layer, layer_name)

    try:
        _ = cam_engine.generate(input_tensor, TREE_CLASS_IDX)
        cam_engine.remove_hooks()
        del cam_engine
        return True, ""

    except RuntimeError as error:
        cam_engine.remove_hooks()
        del cam_engine
        return False, str(error)


def filter_available_layers(
    model: nn.Module,
    target_layers: List[Tuple[nn.Module, str]],
    sample_tensor: torch.Tensor,
):
    """
    Automatically filter out layers that cannot capture activations/gradients.
    """
    if not AUTO_FILTER_LAYERS:
        return target_layers

    active_layers = []

    print("\n开始检测目标层是否可用于 Grad-CAM：")

    for layer, layer_name in target_layers:
        ok, error_message = test_layer_available(
            model=model,
            target_layer=layer,
            layer_name=layer_name,
            input_tensor=sample_tensor,
        )

        if ok:
            print(f"  [OK] {layer_name}")
            active_layers.append((layer, layer_name))
        else:
            print(f"  [Skip] {layer_name}: {error_message}")

        gc.collect()
        if sample_tensor.is_cuda:
            torch.cuda.empty_cache()

    if len(active_layers) == 0:
        raise RuntimeError("所有目标层都无法生成 Grad-CAM，请检查模型结构或目标层设置。")

    return active_layers


# ============================================================
# Main pipeline
# ============================================================
def run_gradcam_on_valset():
    seed_everything(SEED)

    device = torch.device("cuda" if (CUDA and torch.cuda.is_available()) else "cpu")
    print(f"使用设备: {device}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_model(device)

    # --------------------------------------------------------
    # Validation list
    # --------------------------------------------------------
    if not os.path.exists(VAL_TXT):
        raise FileNotFoundError(f"未找到验证集列表: {VAL_TXT}")

    val_lines = load_val_list(VAL_TXT)

    if MAX_IMAGES is not None and MAX_IMAGES > 0:
        val_lines_used = val_lines[:MAX_IMAGES]
    else:
        val_lines_used = val_lines

    print(f"\n验证样本数: {len(val_lines)}")
    print(f"本次 Grad-CAM 样本数: {len(val_lines_used)}")

    # --------------------------------------------------------
    # Dataset and DataLoader
    # --------------------------------------------------------
    val_dataset = UnetDataset(
        annotation_lines=val_lines_used,
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        train=False,
        dataset_path=VOC_PATH,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=SEED),
    )

    # --------------------------------------------------------
    # Target layers
    # --------------------------------------------------------
    target_layers = get_all_recommended_layers(model)

    print("\n初始 Grad-CAM 目标层：")
    for _, name in target_layers:
        print("  -", name)

    # Use the first validation image to filter usable layers.
    first_batch = next(iter(val_loader))
    first_images = first_batch[0] if isinstance(first_batch, (list, tuple)) else first_batch
    first_images = ensure_tensor(first_images).float().to(device)

    sample_tensor = first_images[0:1]

    target_layers = filter_available_layers(
        model=model,
        target_layers=target_layers,
        sample_tensor=sample_tensor,
    )

    print("\n最终用于 Grad-CAM 的目标层：")
    for _, name in target_layers:
        print("  -", name)

    # --------------------------------------------------------
    # Output folders
    # --------------------------------------------------------
    os.makedirs(SAVE_ROOT, exist_ok=True)

    triplet_dir = os.path.join(SAVE_ROOT, "triplet")
    os.makedirs(triplet_dir, exist_ok=True)

    # --------------------------------------------------------
    # Grad-CAM loop
    # --------------------------------------------------------
    img_counter = 0
    line_idx = 0

    print("\n开始对验证集进行 Grad-CAM 可解释性分析...")

    for step, batch in enumerate(tqdm(val_loader, desc="Grad-CAM on val")):
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch

        images = ensure_tensor(images).float()

        batch_size = images.size(0)

        for b in range(batch_size):
            if line_idx >= len(val_lines_used):
                break

            line = val_lines_used[line_idx]
            image_id = line.split()[0]
            line_idx += 1

            img_counter += 1

            safe_image_id = safe_filename(image_id)

            input_tensor = images[b:b + 1].to(device, dtype=torch.float32)
            rgb_vis = tensor_to_rgb(images[b])

            with torch.no_grad():
                logits = model(input_tensor)
                prob = torch.softmax(logits, dim=1)[0]
                tree_prob = prob[TREE_CLASS_IDX].mean().item()

            for layer, layer_name in target_layers:
                cam_engine = GradCAM(model, layer, layer_name)

                try:
                    cam_tree = cam_engine.generate(input_tensor, TREE_CLASS_IDX)

                    heatmap, overlay = visualize_cam_nature(
                        cam_tree,
                        rgb_vis,
                        alpha=0.55,
                    )

                    layer_root = os.path.join(SAVE_ROOT, layer_name)
                    rgb_dir = os.path.join(layer_root, "rgb")
                    cam_dir = os.path.join(layer_root, "cam")
                    overlay_dir = os.path.join(layer_root, "overlay")

                    for directory in [layer_root, rgb_dir, cam_dir, overlay_dir]:
                        os.makedirs(directory, exist_ok=True)

                    base_name = f"val_{img_counter:04d}_{safe_image_id}_{layer_name}"

                    # ------------------------------------------------
                    # 1. Triplet figure: RGB / Grad-CAM / Overlay
                    # ------------------------------------------------
                    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

                    axes[0].imshow(rgb_vis)
                    axes[0].set_title("Input RGB", fontsize=8)
                    axes[0].axis("off")

                    im1 = axes[1].imshow(
                        cam_tree,
                        cmap="magma",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    axes[1].set_title(
                        f"Grad-CAM (tree crown)\n[{layer_name}]",
                        fontsize=8,
                    )
                    axes[1].axis("off")

                    cbar = fig.colorbar(
                        im1,
                        ax=axes[1],
                        fraction=0.046,
                        pad=0.04,
                    )
                    cbar.ax.tick_params(labelsize=7)

                    axes[2].imshow(overlay)
                    axes[2].set_title(
                        f"Overlay\nMean p(tree)={tree_prob:.2f}",
                        fontsize=8,
                    )
                    axes[2].axis("off")

                    add_paper_title(fig, PAPER_TAG, y=1.02)

                    plt.tight_layout()

                    triplet_path = os.path.join(
                        triplet_dir,
                        base_name + "_triplet.png",
                    )
                    fig.savefig(triplet_path, bbox_inches="tight")
                    plt.close(fig)

                    # ------------------------------------------------
                    # 2. Single RGB
                    # ------------------------------------------------
                    fig_rgb, ax_rgb = plt.subplots(1, 1, figsize=(3, 3))
                    ax_rgb.imshow(rgb_vis)
                    ax_rgb.axis("off")
                    add_paper_title(fig_rgb, PAPER_TAG, y=0.98)
                    plt.tight_layout()

                    rgb_path = os.path.join(rgb_dir, base_name + "_rgb.png")
                    fig_rgb.savefig(rgb_path, bbox_inches="tight")
                    plt.close(fig_rgb)

                    # ------------------------------------------------
                    # 3. Single Grad-CAM with colorbar and no title
                    # ------------------------------------------------
                    fig_cam, ax_cam = plt.subplots(1, 1, figsize=(3, 3))
                    im_cam = ax_cam.imshow(
                        cam_tree,
                        cmap="magma",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    ax_cam.axis("off")
                    fig_cam.colorbar(im_cam, fraction=0.046, pad=0.04)
                    plt.tight_layout()

                    cam_path = os.path.join(cam_dir, base_name + "_cam.png")
                    fig_cam.savefig(cam_path, bbox_inches="tight")
                    plt.close(fig_cam)

                    # ------------------------------------------------
                    # 4. Single Overlay
                    # ------------------------------------------------
                    fig_overlay, ax_overlay = plt.subplots(1, 1, figsize=(3, 3))
                    ax_overlay.imshow(overlay)
                    ax_overlay.axis("off")
                    add_paper_title(fig_overlay, PAPER_TAG, y=0.98)
                    plt.tight_layout()

                    overlay_path = os.path.join(
                        overlay_dir,
                        base_name + "_overlay.png",
                    )
                    fig_overlay.savefig(overlay_path, bbox_inches="tight")
                    plt.close(fig_overlay)

                    del cam_tree, heatmap, overlay

                except RuntimeError as error:
                    print(
                        f"\n[Warning] Skip failed layer: "
                        f"image={image_id}, layer={layer_name}"
                    )
                    print(f"Reason: {error}")

                    if not SKIP_FAILED_LAYERS:
                        raise error

                finally:
                    cam_engine.remove_hooks()
                    del cam_engine
                    gc.collect()

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

        if line_idx >= len(val_lines_used):
            break

    print(f"\n[完成] 共从验证集中抽取 {img_counter} 张图像做 Grad-CAM 分析。")
    print(f"结果已保存到: {os.path.abspath(SAVE_ROOT)}")


if __name__ == "__main__":
    run_gradcam_on_valset()
