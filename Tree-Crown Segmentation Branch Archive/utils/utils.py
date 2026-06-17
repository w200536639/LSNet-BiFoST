# utils/utils.py
import os
import random
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file


# ---------------------------------------------------------#
#   Convert image to RGB.
#   This prevents errors caused by grayscale or palette images.
# ---------------------------------------------------------#
def cvtColor(image):
    """
    Convert PIL image to RGB.

    Args:
        image: PIL.Image or image-like object.

    Returns:
        RGB PIL.Image.
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image

    return image.convert("RGB")


# ---------------------------------------------------#
#   Resize image while keeping aspect ratio.
#   The remaining area is filled with gray.
# ---------------------------------------------------#
def resize_image(image: Image.Image, size: Sequence[int]):
    """
    Resize image with unchanged aspect ratio and gray padding.

    Args:
        image: PIL image.
        size: target size as (width, height).

    Returns:
        new_image: resized and padded image.
        nw: resized width.
        nh: resized height.
    """
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


# ---------------------------------------------------#
#   Get current learning rate.
# ---------------------------------------------------#
def get_lr(optimizer):
    """Get learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# ---------------------------------------------------#
#   Set random seed for reproducibility.
# ---------------------------------------------------#
def seed_everything(seed: int = 11):
    """
    Set random seed for Python, NumPy, and PyTorch.

    Args:
        seed: random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   Worker initialization function for DataLoader.
# ---------------------------------------------------#
def worker_init_fn(worker_id: int, rank: int, seed: int):
    """
    Initialize random seed for each DataLoader worker.

    Args:
        worker_id: worker id.
        rank: distributed rank.
        seed: base seed.
    """
    worker_seed = rank + seed + worker_id

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------#
#   Image preprocessing.
# ---------------------------------------------------#
def preprocess_input(
    image,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
):
    """
    Unified image preprocessing function.

    Supports:
        - PIL.Image
        - numpy array, usually HWC
        - torch.Tensor, CHW / NCHW / HWC / NHWC

    Default behavior:
        - Convert to float32
        - Normalize 0-255 to 0-1
        - Do not apply mean/std normalization

    Important:
        This version is compatible with NumPy 2.x.
        Do not use np.array(..., copy=False), because NumPy 2.x may raise:
        ValueError: Unable to avoid copy while creating an array as requested.

    Args:
        image: input image.
        mean: optional normalization mean.
        std: optional normalization std.

    Returns:
        Preprocessed image with the same general layout as input.
    """
    # --------------------------------------------------------
    # torch.Tensor input
    # --------------------------------------------------------
    if isinstance(image, torch.Tensor):
        x = image.float()

        if x.numel() > 0 and x.max() > 1.0:
            x = x / 255.0

        if mean is not None and std is not None:
            mean_tensor = torch.tensor(mean, dtype=x.dtype, device=x.device)
            std_tensor = torch.tensor(std, dtype=x.dtype, device=x.device)

            # NCHW
            if x.dim() == 4 and x.shape[1] == len(mean):
                m = mean_tensor.view(1, len(mean), 1, 1)
                s = std_tensor.view(1, len(std), 1, 1)
                x = (x - m) / s

            # NHWC
            elif x.dim() == 4 and x.shape[-1] == len(mean):
                m = mean_tensor.view(1, 1, 1, len(mean))
                s = std_tensor.view(1, 1, 1, len(std))
                x = (x - m) / s

            # CHW
            elif x.dim() == 3 and x.shape[0] == len(mean):
                m = mean_tensor.view(len(mean), 1, 1)
                s = std_tensor.view(len(std), 1, 1)
                x = (x - m) / s

            # HWC
            elif x.dim() == 3 and x.shape[-1] == len(mean):
                m = mean_tensor.view(1, 1, len(mean))
                s = std_tensor.view(1, 1, len(std))
                x = (x - m) / s

        return x

    # --------------------------------------------------------
    # PIL.Image or numpy input
    # --------------------------------------------------------
    # NumPy 2.x compatible:
    # np.asarray allows a copy when needed.
    if isinstance(image, Image.Image):
        x = np.asarray(image, dtype=np.float32)
    else:
        x = np.asarray(image, dtype=np.float32)

    if x.size > 0 and np.nanmax(x) > 1.0:
        x = x / 255.0

    if mean is not None and std is not None:
        mean_array = np.asarray(mean, dtype=np.float32)
        std_array = np.asarray(std, dtype=np.float32)

        # HWC
        if x.ndim == 3 and x.shape[-1] == len(mean_array):
            mean_array = mean_array.reshape(1, 1, len(mean_array))
            std_array = std_array.reshape(1, 1, len(std_array))
            x = (x - mean_array) / std_array

        # CHW
        elif x.ndim == 3 and x.shape[0] == len(mean_array):
            mean_array = mean_array.reshape(len(mean_array), 1, 1)
            std_array = std_array.reshape(len(std_array), 1, 1)
            x = (x - mean_array) / std_array

    return x.astype(np.float32, copy=False)


# ---------------------------------------------------#
#   Show configuration.
# ---------------------------------------------------#
def show_config(**kwargs):
    """Print configuration table."""
    print("Configurations:")
    print("-" * 70)
    print("|%25s | %40s|" % ("keys", "values"))
    print("-" * 70)

    for key, value in kwargs.items():
        print("|%25s | %40s|" % (str(key), str(value)))

    print("-" * 70)


# ---------------------------------------------------#
#   Download pretrained weights.
# ---------------------------------------------------#
def download_weights(backbone: str, model_dir: str = "./model_data") -> Optional[str]:
    """
    Download pretrained weights to ./model_data.

    Supported backbones:
        vgg
        resnet50
        mobilenetv3
        mobilenet

    Args:
        backbone: backbone name.
        model_dir: target directory.

    Returns:
        Local path of downloaded or existing weight file.
        Returns None if the backbone is unsupported or download fails.
    """
    backbone = str(backbone).lower()

    download_urls = {
        "vgg": "https://download.pytorch.org/models/vgg16-397923af.pth",
        "resnet50": "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
        "mobilenetv3": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        "mobilenet": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    }

    if backbone not in download_urls:
        print(f"[download_weights] No download URL is configured for backbone: {backbone}. Skipped.")
        return None

    # Anchor relative path to project root.
    # utils/utils.py -> project root is one level above utils/.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if os.path.isabs(model_dir):
        target_dir = model_dir
    else:
        target_dir = os.path.abspath(os.path.join(project_root, model_dir))

    os.makedirs(target_dir, exist_ok=True)

    url = download_urls[backbone]
    filename = os.path.basename(url)
    dst_path = os.path.join(target_dir, filename)

    if os.path.exists(dst_path):
        print(f"[download_weights] Existing file found: {dst_path}. Skipped download.")
        return dst_path

    try:
        print(f"[download_weights] Downloading {backbone} weights:")
        print(f"[download_weights] URL: {url}")
        print(f"[download_weights] Save to: {dst_path}")

        download_url_to_file(url, dst_path, progress=True)

        print(f"[download_weights] Saved to: {dst_path}")
        return dst_path

    except Exception as error:
        print(f"[download_weights] Download failed: {error}")
        return None