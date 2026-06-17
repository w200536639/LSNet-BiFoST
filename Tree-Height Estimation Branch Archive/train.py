import datetime
import logging
import os
import random
import warnings
from functools import partial

import numpy as np
import tifffile
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from nets.lsnet_bifost_height_regression import LSNetBiFoSTHeightRegression
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch_reg


# ============================================================
# Suppress tifffile GDAL_NODATA warning
# ============================================================
logging.getLogger("tifffile").setLevel(logging.CRITICAL)

warnings.filterwarnings("ignore", message=r".*GDAL_NODATA.*")
warnings.filterwarnings("ignore", message=r".*not castable to float32.*")


# ============================================================
# Manual configuration
# ============================================================

# Device and reproducibility
USE_CUDA = True
SEED = 11
USE_DISTRIBUTED = False
USE_SYNC_BN = True
USE_FP16 = True
USE_CUDNN_BENCHMARK = True

# Model
BACKBONE = "focal_s"          # focal_t / focal_s / focal_b
IN_CHANNELS = 4               # RGB(3) + crown mask(1)
OUT_CHANNELS = 1              # single-channel height regression
PRETRAINED = False

# 如果没有树高回归分支的预训练权重，保持为空即可。
# 如果要断点续训，改成真实存在的树高权重路径，例如：
# MODEL_PATH = r"logs_reg_rgb_mask/focal_s_svit1_bie1_f161_f321/last_epoch_weights.pth"
MODEL_PATH = r""

# Ablation switches
USE_SVIT = True
USE_BIE = True
SVIT_ON_F16 = True
SVIT_ON_F32 = True
SVIT_STOKEN_SIZE = (4, 4)
SVIT_HEADS = 8
SVIT_N_ITER = 1

# Input
INPUT_SHAPE = [640, 640]

# Dataset
DATASET_PATH = "TreeHeightDataset"
RGB_DIR = "rgb"
MASK_DIR = "mask"
HEIGHT_DIR = "heights"

RGB_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"
HEIGHT_SUFFIX = "_processed.tif"

TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"

# Height normalization
VALID_HEIGHT_MIN = 0.0
VALID_HEIGHT_MAX = 6.0
NORMALIZE_MODE = "minmax"     # minmax / zscore

# 用于识别 CHM/树高 tif 中的极端 NoData 值，例如 -3.402823466e+38
NODATA_ABS_THRESHOLD = 1.0e20

# For min-max normalization, the range is estimated from these split files.
HEIGHT_RANGE_TXT_FILES = ["train.txt", "val.txt"]

# Training schedule
INIT_EPOCH = 0
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_TRAIN = True
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4

# Optimizer and learning rate
INIT_LR = 1e-5
MIN_LR = 1e-7
OPTIMIZER_TYPE = "adam"       # adam / adamw / sgd
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
LR_DECAY_TYPE = "cos"

# Loss
LOSS_TYPE = "combined"        # mse / mae / combined
LOSS_ALPHA = 0.5

# Logging
SAVE_PERIOD = 10
BASE_SAVE_DIR = "logs_reg_rgb_mask"
RUN_TAG = ""

# DataLoader
NUM_WORKERS = 0


# ============================================================
# Height map utilities
# ============================================================
def read_height_map_safely(height_path):
    """
    Safely read tree-height / CHM GeoTIFF.

    Extreme NoData values such as -3.402823466e+38 are converted to NaN.
    NaN pixels are ignored by valid-height masks and do not participate in loss.
    """
    if not os.path.exists(height_path):
        raise FileNotFoundError(f"Height file does not exist: {height_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        height_map = tifffile.imread(height_path)

    height_map = np.asarray(height_map)

    if height_map.ndim == 3:
        if height_map.shape[0] == 1:
            height_map = height_map[0]
        elif height_map.shape[-1] == 1:
            height_map = height_map[..., 0]
        else:
            height_map = height_map[0]

    height_map = height_map.astype(np.float32, copy=False)

    height_map[np.abs(height_map) >= NODATA_ABS_THRESHOLD] = np.nan
    height_map[~np.isfinite(height_map)] = np.nan

    return height_map.astype(np.float32, copy=False)


def build_valid_height_mask(height_map, valid_height_min=0.0, valid_height_max=6.0):
    """Return a float32 valid-height mask."""
    valid_mask = (
        np.isfinite(height_map)
        & (height_map >= valid_height_min)
        & (height_map <= valid_height_max)
    )
    return valid_mask.astype(np.float32)


# ============================================================
# Dataset
# ============================================================
class RgbMaskHeightDataset(torch.utils.data.Dataset):
    """
    Dataset for tree-height regression using RGB + crown mask as input.

    Input:
        RGB image:      H x W x 3
        crown mask:     H x W
        model input:    4 x H x W

    Target:
        height map:     1 x H x W
        valid mask:     1 x H x W

    The returned mask is:
        crown mask * valid height mask

    Therefore CHM/height NoData pixels do not participate in loss.
    """

    def __init__(
        self,
        annotation_lines,
        input_shape=(640, 640),
        train=True,
        dataset_path="TreeHeightDataset",
        rgb_dir="rgb",
        mask_dir="mask",
        height_dir="heights",
        rgb_suffix=".jpg",
        mask_suffix=".png",
        height_suffix=".tif",
        height_min=0.0,
        height_max=6.0,
        valid_height_min=0.0,
        valid_height_max=6.0,
        normalize_mode="minmax",
    ):
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

        self.dataset_path = dataset_path
        self.rgb_dir = rgb_dir
        self.mask_dir = mask_dir
        self.height_dir = height_dir

        self.rgb_suffix = rgb_suffix
        self.mask_suffix = mask_suffix
        self.height_suffix = height_suffix

        self.height_min = height_min
        self.height_max = height_max
        self.valid_height_min = valid_height_min
        self.valid_height_max = valid_height_max
        self.normalize_mode = normalize_mode

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        prefix = self.annotation_lines[index].strip().split()[0]

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
            raise FileNotFoundError(f"RGB file does not exist: {rgb_path}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        if not os.path.exists(height_path):
            raise FileNotFoundError(f"Height file does not exist: {height_path}")

        rgb_image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        mask_image = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask_image = (mask_image > 0).astype(np.float32)

        height_map = read_height_map_safely(height_path)

        if rgb_image.shape[:2] != mask_image.shape[:2]:
            raise RuntimeError(
                f"RGB and mask shape mismatch for {prefix}: "
                f"rgb={rgb_image.shape}, mask={mask_image.shape}"
            )

        if height_map.shape[:2] != mask_image.shape[:2]:
            raise RuntimeError(
                f"Height and mask shape mismatch for {prefix}: "
                f"height={height_map.shape}, mask={mask_image.shape}"
            )

        rgb_image, mask_image, height_map = self.crop_or_pad(
            rgb=rgb_image,
            mask=mask_image,
            height=height_map,
            target_size=self.input_shape,
            random_enable=self.train,
        )

        valid_height_mask = build_valid_height_mask(
            height_map,
            valid_height_min=self.valid_height_min,
            valid_height_max=self.valid_height_max,
        )

        loss_mask = (mask_image > 0).astype(np.float32) * valid_height_mask
        height_norm = self.normalize_height(height_map)

        mask_channel = np.expand_dims(mask_image.astype(np.float32), axis=-1)
        input_4ch = np.concatenate([rgb_image, mask_channel], axis=-1)
        input_4ch = np.transpose(input_4ch, (2, 0, 1)).astype(np.float32)

        height_norm = np.expand_dims(height_norm, axis=0).astype(np.float32)
        loss_mask = np.expand_dims(loss_mask, axis=0).astype(np.float32)

        return (
            torch.from_numpy(input_4ch).float(),
            torch.from_numpy(height_norm).float(),
            torch.from_numpy(loss_mask).float(),
            prefix,
        )

    @staticmethod
    def pad_to_size(rgb, mask, height, target_size):
        """Pad RGB, mask, and height map if they are smaller than target size."""
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

    def crop_or_pad(self, rgb, mask, height, target_size, random_enable=True):
        """
        Crop or pad RGB, mask, and height map to target size.

        Training:
            random crop.
        Validation:
            center crop.
        """
        target_h, target_w = target_size

        rgb, mask, height = self.pad_to_size(
            rgb=rgb,
            mask=mask,
            height=height,
            target_size=target_size,
        )

        h, w, _ = rgb.shape

        if random_enable and h > target_h and w > target_w:
            top = random.randint(0, h - target_h)
            left = random.randint(0, w - target_w)
        else:
            top = max(0, (h - target_h) // 2)
            left = max(0, (w - target_w) // 2)

        rgb = rgb[top: top + target_h, left: left + target_w, :]
        mask = mask[top: top + target_h, left: left + target_w]
        height = height[top: top + target_h, left: left + target_w]

        return rgb, mask, height

    def normalize_height(self, height_map):
        """
        Normalize height map while preserving floating-point height information.

        Invalid pixels are set to 0 after normalization.
        """
        valid_mask = build_valid_height_mask(
            height_map,
            valid_height_min=self.valid_height_min,
            valid_height_max=self.valid_height_max,
        ).astype(bool)

        height = height_map.copy().astype(np.float32)

        if not np.any(valid_mask):
            return np.zeros_like(height, dtype=np.float32)

        if self.normalize_mode == "zscore":
            mean_value = np.nanmean(height[valid_mask])
            std_value = np.nanstd(height[valid_mask]) + 1e-6
            height = (height - mean_value) / std_value
        else:
            denominator = self.height_max - self.height_min
            if denominator <= 1e-6:
                denominator = 1.0

            height = (height - self.height_min) / (denominator + 1e-6)
            height = np.clip(height, 0.0, 1.0)

        height[~valid_mask] = 0.0

        return height.astype(np.float32)


def rgb_mask_height_collate(batch):
    images, heights, masks, names = zip(*batch)
    return torch.stack(images), torch.stack(heights), torch.stack(masks), list(names)


# ============================================================
# Utility functions
# ============================================================
def read_split_file(split_path):
    """Read image prefixes from a split file with encoding fallback."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(split_path, "r", encoding=encoding) as file:
                return [line.strip().split()[0] for line in file if line.strip()]
        except UnicodeDecodeError:
            continue

    with open(split_path, "r", encoding="utf-8", errors="ignore") as file:
        return [line.strip().split()[0] for line in file if line.strip()]


def validate_input_shape(input_shape):
    """Validate whether input shape is [H, W] and divisible by 32."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be [height, width].")

    height, width = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            "Both height and width in INPUT_SHAPE must be divisible by 32, "
            "for example [512, 512], [640, 640], or [1024, 1024]."
        )


def build_run_tag():
    """Build run tag from model and ablation settings."""
    auto_tag = (
        f"{BACKBONE}"
        f"_svit{int(USE_SVIT)}"
        f"_bie{int(USE_BIE)}"
        f"_f16{int(SVIT_ON_F16)}"
        f"_f32{int(SVIT_ON_F32)}"
    )
    return RUN_TAG.strip() or auto_tag


def calculate_height_range(
    dataset_path,
    height_dir,
    height_suffix,
    split_files,
    valid_height_min=0.0,
    valid_height_max=6.0,
):
    """Estimate valid height range from height maps."""
    height_dir_full = os.path.join(dataset_path, height_dir)

    global_min = np.inf
    global_max = -np.inf
    total_valid_count = 0
    missing_count = 0

    print(
        f"Height filtering rule: keep finite values within "
        f"[{valid_height_min}, {valid_height_max}] m."
    )

    for split_file in split_files:
        split_path = os.path.join(dataset_path, split_file)
        prefixes = read_split_file(split_path)

        for prefix in prefixes:
            height_path = os.path.join(height_dir_full, prefix + height_suffix)
            if not os.path.exists(height_path):
                missing_count += 1
                continue

            height_map = read_height_map_safely(height_path)
            valid_mask = build_valid_height_mask(
                height_map,
                valid_height_min=valid_height_min,
                valid_height_max=valid_height_max,
            ).astype(bool)

            if not np.any(valid_mask):
                continue

            local_min = float(np.min(height_map[valid_mask]))
            local_max = float(np.max(height_map[valid_mask]))

            global_min = min(global_min, local_min)
            global_max = max(global_max, local_max)
            total_valid_count += int(np.sum(valid_mask))

    if missing_count > 0:
        print(f"Warning: missing height files: {missing_count}")

    if total_valid_count == 0:
        print(
            "Warning: no valid height pixels found. "
            f"Fallback to [{valid_height_min}, {valid_height_max}] m."
        )
        global_min = valid_height_min
        global_max = valid_height_max

    print(f"Valid height pixels: {total_valid_count}")
    print(f"Height min: {global_min:.3f} m, height max: {global_max:.3f} m")

    return global_min, global_max


def load_matching_weights(model, checkpoint_path, device, local_rank=0):
    """
    Load only shape-matched checkpoint weights.

    Supports:
        1. raw state_dict
        2. {'state_dict': state_dict}
        3. {'model_state_dict': state_dict}
        4. DataParallel checkpoints with 'module.' prefix
    """
    if checkpoint_path in ["", None]:
        if local_rank == 0:
            print("MODEL_PATH is empty. Training will start from scratch.")
        return

    if not os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Checkpoint does not exist: {checkpoint_path}. Training will continue without loading.")
        return

    if local_rank == 0:
        print(f"Load weights from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    normalized_checkpoint = {}
    for key, value in checkpoint.items():
        normalized_key = key[len("module."):] if key.startswith("module.") else key
        normalized_checkpoint[normalized_key] = value

    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    matched_state = {}

    for key, value in normalized_checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model_state.update(matched_state)
    model.load_state_dict(model_state, strict=False)

    if local_rank == 0:
        print("\nSuccessful Load Key:", str(loaded_keys)[:500], "……")
        print("Successful Load Key Num:", len(loaded_keys))
        print("\nFail To Load Key:", str(skipped_keys)[:500], "……")
        print("Fail To Load Key Num:", len(skipped_keys))
        print(
            "\n\033[1;33;44mNote: it is normal if the final regression head is not loaded; "
            "it is usually problematic if many encoder/backbone keys are not loaded.\033[0m"
        )


def build_optimizer(model, init_lr):
    """Build optimizer from configuration."""
    optimizer_dict = {
        "adam": optim.Adam(
            model.parameters(),
            init_lr,
            betas=(MOMENTUM, 0.999),
            weight_decay=WEIGHT_DECAY,
        ),
        "adamw": optim.AdamW(
            model.parameters(),
            init_lr,
            betas=(MOMENTUM, 0.999),
            weight_decay=WEIGHT_DECAY,
        ),
        "sgd": optim.SGD(
            model.parameters(),
            init_lr,
            momentum=MOMENTUM,
            nesterov=True,
            weight_decay=WEIGHT_DECAY,
        ),
    }

    if OPTIMIZER_TYPE not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")

    return optimizer_dict[OPTIMIZER_TYPE]


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    train_sampler,
    val_sampler,
    shuffle_train,
    rank,
):
    """Build training and validation dataloaders."""
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=rgb_mask_height_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=rgb_mask_height_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
    )

    return train_loader, val_loader


def has_invalid_loss(loss_history):
    """Check whether latest loss contains NaN or Inf."""
    if loss_history is None:
        return False

    last_train_loss = loss_history.losses[-1] if len(loss_history.losses) > 0 else None

    if hasattr(loss_history, "val_loss") and len(loss_history.val_loss) > 0:
        last_val_loss = loss_history.val_loss[-1]
    else:
        last_val_loss = None

    train_invalid = last_train_loss is not None and (
        np.isnan(last_train_loss) or np.isinf(last_train_loss)
    )
    val_invalid = last_val_loss is not None and (
        np.isnan(last_val_loss) or np.isinf(last_val_loss)
    )

    return train_invalid or val_invalid


def build_grad_scaler(use_fp16, use_cuda):
    """Build AMP GradScaler."""
    if not use_fp16 or not use_cuda:
        return None

    try:
        from torch.amp import GradScaler as TorchAmpGradScaler

        try:
            return TorchAmpGradScaler("cuda")
        except TypeError:
            return TorchAmpGradScaler()

    except Exception:
        from torch.cuda.amp import GradScaler as CudaAmpGradScaler

        return CudaAmpGradScaler()


def create_loss_history(log_dir, model, input_shape):
    """Create LossHistory."""
    return LossHistory(log_dir, model, input_shape=input_shape)


# ============================================================
# Main training pipeline
# ============================================================
if __name__ == "__main__":
    validate_input_shape(INPUT_SHAPE)
    seed_everything(SEED)

    run_tag = build_run_tag()
    save_dir = os.path.join(BASE_SAVE_DIR, run_tag)
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------------
    # Device and distributed setup
    # --------------------------------------------------------
    num_gpus = torch.cuda.device_count()
    use_cuda = USE_CUDA and torch.cuda.is_available()

    if USE_DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)

        if local_rank == 0:
            print(f"[{os.getpid()}] rank={rank}, local_rank={local_rank}, training...")
            print("GPU Device Count:", num_gpus)

    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        local_rank = 0
        rank = 0

    if local_rank == 0:
        print("\n[Height-regression branch configuration]")
        print(f"  backbone       = {BACKBONE}")
        print(f"  in_channels    = {IN_CHANNELS}")
        print(f"  out_channels   = {OUT_CHANNELS}")
        print(f"  use_svit       = {USE_SVIT}")
        print(f"  use_bie        = {USE_BIE}")
        print(f"  svit_on_f16    = {SVIT_ON_F16}")
        print(f"  svit_on_f32    = {SVIT_ON_F32}")
        print(f"  run_tag        = {run_tag}")
        print(f"  save_dir       = {save_dir}\n")

    # --------------------------------------------------------
    # Height range
    # --------------------------------------------------------
    height_min, height_max = calculate_height_range(
        dataset_path=DATASET_PATH,
        height_dir=HEIGHT_DIR,
        height_suffix=HEIGHT_SUFFIX,
        split_files=HEIGHT_RANGE_TXT_FILES,
        valid_height_min=VALID_HEIGHT_MIN,
        valid_height_max=VALID_HEIGHT_MAX,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = LSNetBiFoSTHeightRegression(
        num_classes=OUT_CHANNELS,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        in_channels=IN_CHANNELS,
        use_bie=USE_BIE,
        use_svit=USE_SVIT,
        svit_on_f16=SVIT_ON_F16,
        svit_on_f32=SVIT_ON_F32,
        svit_stoken_size=SVIT_STOKEN_SIZE,
        svit_heads=SVIT_HEADS,
        svit_n_iter=SVIT_N_ITER,
    ).train()

    if not PRETRAINED:
        weights_init(model)

    load_matching_weights(
        model=model,
        checkpoint_path=MODEL_PATH,
        device=device,
        local_rank=local_rank,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------
    scaler = build_grad_scaler(use_fp16=USE_FP16, use_cuda=use_cuda)

    model_train = model.train()

    # --------------------------------------------------------
    # SyncBN and parallel training
    # --------------------------------------------------------
    if USE_SYNC_BN and num_gpus > 1 and USE_DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif USE_SYNC_BN and local_rank == 0:
        print("SyncBN is only effective in distributed multi-GPU training; it is ignored here.")

    if use_cuda:
        if USE_DISTRIBUTED:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = USE_CUDNN_BENCHMARK
            cudnn.deterministic = not USE_CUDNN_BENCHMARK
            model_train = model_train.cuda()

    # --------------------------------------------------------
    # Read dataset split
    # --------------------------------------------------------
    train_lines = read_split_file(os.path.join(DATASET_PATH, TRAIN_TXT))
    val_lines = read_split_file(os.path.join(DATASET_PATH, VAL_TXT))

    num_train = len(train_lines)
    num_val = len(val_lines)

    # --------------------------------------------------------
    # Check height precision and normalization
    # --------------------------------------------------------
    if num_val > 0 and local_rank == 0:
        print("\n[Checking height precision]")
        check_dataset = RgbMaskHeightDataset(
            annotation_lines=val_lines,
            input_shape=INPUT_SHAPE,
            train=False,
            dataset_path=DATASET_PATH,
            rgb_dir=RGB_DIR,
            mask_dir=MASK_DIR,
            height_dir=HEIGHT_DIR,
            rgb_suffix=RGB_SUFFIX,
            mask_suffix=MASK_SUFFIX,
            height_suffix=HEIGHT_SUFFIX,
            height_min=height_min,
            height_max=height_max,
            valid_height_min=VALID_HEIGHT_MIN,
            valid_height_max=VALID_HEIGHT_MAX,
            normalize_mode=NORMALIZE_MODE,
        )

        sample_image, sample_height, sample_mask, sample_name = check_dataset[0]
        height_array = sample_height.numpy().flatten()
        valid_array = sample_mask.numpy().flatten() > 0

        print(f"Sample: {sample_name}")
        print(f"Input dtype: {sample_image.dtype}")
        print(f"Height dtype: {sample_height.dtype}")
        print(f"Mask dtype: {sample_mask.dtype}")
        print(f"Valid training pixels: {int(valid_array.sum())}")

        if np.any(valid_array):
            print(
                f"Height min/max in valid mask: "
                f"{np.nanmin(height_array[valid_array]):.4f} / "
                f"{np.nanmax(height_array[valid_array]):.4f}"
            )
            print(
                f"Height values are integer-like: "
                f"{np.allclose(height_array[valid_array], np.round(height_array[valid_array]))}\n"
            )
        else:
            print("Warning: this sample has no valid height pixels inside crown mask.\n")

    if local_rank == 0:
        show_config(
            num_classes=OUT_CHANNELS,
            backbone=BACKBONE,
            model_path=MODEL_PATH,
            input_shape=INPUT_SHAPE,
            Init_Epoch=INIT_EPOCH,
            Freeze_Epoch=FREEZE_EPOCH,
            UnFreeze_Epoch=UNFREEZE_EPOCH,
            Freeze_batch_size=FREEZE_BATCH_SIZE,
            Unfreeze_batch_size=UNFREEZE_BATCH_SIZE,
            Freeze_Train=FREEZE_TRAIN,
            Init_lr=INIT_LR,
            Min_lr=MIN_LR,
            optimizer_type=OPTIMIZER_TYPE,
            momentum=MOMENTUM,
            lr_decay_type=LR_DECAY_TYPE,
            save_period=SAVE_PERIOD,
            save_dir=save_dir,
            num_workers=NUM_WORKERS,
            num_train=num_train,
            num_val=num_val,
        )

    # --------------------------------------------------------
    # Dataset and DataLoader
    # --------------------------------------------------------
    train_dataset = RgbMaskHeightDataset(
        annotation_lines=train_lines,
        input_shape=INPUT_SHAPE,
        train=True,
        dataset_path=DATASET_PATH,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        height_dir=HEIGHT_DIR,
        rgb_suffix=RGB_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        height_suffix=HEIGHT_SUFFIX,
        height_min=height_min,
        height_max=height_max,
        valid_height_min=VALID_HEIGHT_MIN,
        valid_height_max=VALID_HEIGHT_MAX,
        normalize_mode=NORMALIZE_MODE,
    )

    val_dataset = RgbMaskHeightDataset(
        annotation_lines=val_lines,
        input_shape=INPUT_SHAPE,
        train=False,
        dataset_path=DATASET_PATH,
        rgb_dir=RGB_DIR,
        mask_dir=MASK_DIR,
        height_dir=HEIGHT_DIR,
        rgb_suffix=RGB_SUFFIX,
        mask_suffix=MASK_SUFFIX,
        height_suffix=HEIGHT_SUFFIX,
        height_min=height_min,
        height_max=height_max,
        valid_height_min=VALID_HEIGHT_MIN,
        valid_height_max=VALID_HEIGHT_MAX,
        normalize_mode=NORMALIZE_MODE,
    )

    is_unfrozen = False

    if FREEZE_TRAIN:
        model.freeze_backbone()

    batch_size = FREEZE_BATCH_SIZE if FREEZE_TRAIN else UNFREEZE_BATCH_SIZE

    init_lr_fit = INIT_LR
    min_lr_fit = MIN_LR

    optimizer = build_optimizer(model, init_lr_fit)
    lr_scheduler_func = get_lr_scheduler(
        LR_DECAY_TYPE,
        init_lr_fit,
        min_lr_fit,
        UNFREEZE_EPOCH,
    )

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("The dataset is too small for training. Please add more samples.")

    if USE_DISTRIBUTED:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        batch_size = batch_size // num_gpus
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        shuffle_train=shuffle_train,
        rank=rank,
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d_%H%M%S_%f")
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)

        loss_history = create_loss_history(log_dir, model, INPUT_SHAPE)

        with open(os.path.join(log_dir, "height_range.txt"), "w", encoding="utf-8") as file:
            file.write(f"height_min={height_min:.6f}\n")
            file.write(f"height_max={height_max:.6f}\n")
            file.write(f"valid_height_min={VALID_HEIGHT_MIN:.6f}\n")
            file.write(f"valid_height_max={VALID_HEIGHT_MAX:.6f}\n")
            file.write(f"normalize_mode={NORMALIZE_MODE}\n")
            file.write(f"nodata_abs_threshold={NODATA_ABS_THRESHOLD:.6e}\n")

        if hasattr(model, "get_ablation_config"):
            exp_config = model.get_ablation_config()
        else:
            exp_config = {
                "backbone": BACKBONE,
                "use_bie": USE_BIE,
                "use_svit": USE_SVIT,
                "svit_on_f16": SVIT_ON_F16,
                "svit_on_f32": SVIT_ON_F32,
            }

        with open(os.path.join(log_dir, "exp_config.txt"), "w", encoding="utf-8") as file:
            for key, value in exp_config.items():
                file.write(f"{key}={value}\n")
            file.write(f"in_channels={IN_CHANNELS}\n")
            file.write(f"out_channels={OUT_CHANNELS}\n")
            file.write(f"loss_type={LOSS_TYPE}\n")
            file.write(f"loss_alpha={LOSS_ALPHA}\n")
            file.write(f"model_path={MODEL_PATH}\n")

    else:
        loss_history = None

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    if local_rank == 0:
        print("\nStart height-regression training...")

    for epoch in range(INIT_EPOCH, UNFREEZE_EPOCH):
        if epoch >= FREEZE_EPOCH and not is_unfrozen and FREEZE_TRAIN:
            batch_size = UNFREEZE_BATCH_SIZE

            init_lr_fit = INIT_LR
            min_lr_fit = MIN_LR

            lr_scheduler_func = get_lr_scheduler(
                LR_DECAY_TYPE,
                init_lr_fit,
                min_lr_fit,
                UNFREEZE_EPOCH,
            )

            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("The dataset is too small for training. Please add more samples.")

            if USE_DISTRIBUTED:
                batch_size = batch_size // num_gpus

            train_loader, val_loader = build_dataloaders(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                batch_size=batch_size,
                train_sampler=train_sampler,
                val_sampler=val_sampler,
                shuffle_train=shuffle_train,
                rank=rank,
            )

            is_unfrozen = True

        if USE_DISTRIBUTED and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch_reg(
            model_train,
            model,
            loss_history,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            train_loader,
            val_loader,
            UNFREEZE_EPOCH,
            use_cuda,
            LOSS_TYPE,
            LOSS_ALPHA,
            USE_FP16,
            scaler,
            SAVE_PERIOD,
            save_dir,
            local_rank,
            height_min=height_min,
            height_max=height_max,
        )

        if local_rank == 0 and has_invalid_loss(loss_history):
            print(
                "NaN/Inf loss detected. Training stopped early. "
                "Please resume from the latest valid checkpoint."
            )
            break

        if USE_DISTRIBUTED:
            dist.barrier()

    if local_rank == 0 and loss_history is not None and hasattr(loss_history, "writer"):
        loss_history.writer.close()

    if local_rank == 0:
        print("Height-regression training finished.")
