import datetime
import os
from functools import partial
import random
import warnings
import logging

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import tifffile

from nets.focal_svit_crown_segmentation import FocalSVITCrownSegmentationNet
from nets.unet_training import (
    get_lr_scheduler,
    set_optimizer_lr,
    weights_init,
    MSE_Loss,
    MAE_Loss,
    Combined_Reg_Loss,
)
from utils.callbacks import LossHistory
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch_reg


# ============================================================
# Suppress tifffile GDAL_NODATA warnings
# ============================================================
logging.getLogger("tifffile").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=r".*GDAL_NODATA.*")
warnings.filterwarnings("ignore", message=r".*not castable.*")


# ============================================================
# Safe height reading
# ============================================================
NODATA_ABS_THRESHOLD = 1.0e20


def read_height_tif_safely(height_path):
    """
    Read height map safely.

    Very large NoData values such as -3.402823e+38 will be converted to NaN.
    """
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


# ============================================================
# 8-band multispectral image + crown mask dataset
# ============================================================
class MsMaskDataset(torch.utils.data.Dataset):
    """
    Input:
        8-band multispectral tif + crown mask png -> 9-channel tensor

    Supervision:
        height tif -> single-channel height regression target

    Returned:
        image_9ch:  [9, H, W]
        height:     [1, H, W]
        loss_mask:  [1, H, W]
        prefix:     sample name
    """

    def __init__(
        self,
        annotation_lines,
        input_shape=(640, 640),
        num_classes=1,
        train=True,
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

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train
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
        return self.length

    def __getitem__(self, index):
        raw = self.annotation_lines[index].strip()
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

        # Read multispectral image
        ms_img = self.read_ms_hwc(
            ms_path,
            out_channels=self.ms_in_channels,
        )

        # Read crown mask
        mask_img = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask_img = (mask_img > 0).astype(np.float32)

        # Read height map
        height_np = read_height_tif_safely(height_path)

        # Align to common size
        common_h = min(ms_img.shape[0], mask_img.shape[0], height_np.shape[0])
        common_w = min(ms_img.shape[1], mask_img.shape[1], height_np.shape[1])

        ms_img = ms_img[:common_h, :common_w, :]
        mask_img = mask_img[:common_h, :common_w]
        height_np = height_np[:common_h, :common_w]

        # Synchronized crop
        ms_img, mask_img, height_np = self.random_crop(
            ms_img,
            mask_img,
            height_np,
            self.input_shape,
            self.train,
        )

        # Valid height mask
        valid_height_mask = (
            np.isfinite(height_np)
            & (height_np >= self.height_min)
            & (height_np <= self.height_max)
        ).astype(np.float32)

        # Loss mask = crown mask × valid height mask
        loss_mask = mask_img * valid_height_mask

        # Normalize height
        height_norm = self.normalize_height(height_np)

        # 8 bands + crown mask = 9 channels
        mask_channel = np.expand_dims(mask_img, axis=-1)
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

    def read_ms_hwc(self, path, out_channels=8):
        """
        Read multispectral tif.

        Supports:
            H x W
            C x H x W
            H x W x C

        Output:
            H x W x C float32
        """
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

        norm_mode = self.ms_norm_mode

        if norm_mode == "none":
            pass

        elif norm_mode == "max":
            max_value = float(np.max(arr)) + 1e-6
            arr = arr / max_value

        elif norm_mode == "percentile":
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
            raise ValueError(f"Unsupported ms_norm_mode: {self.ms_norm_mode}")

        return np.transpose(arr, (1, 2, 0)).astype(np.float32)

    def random_crop(self, ms, mask, height, target_size, random_enable=True):
        target_h, target_w = target_size
        image_h, image_w, _ = ms.shape

        if image_h < target_h or image_w < target_w:
            raise ValueError(
                f"Image is smaller than input_shape. "
                f"Image size=({image_h}, {image_w}), input_shape=({target_h}, {target_w})."
            )

        if random_enable and image_h > target_h and image_w > target_w:
            top = random.randint(0, image_h - target_h)
            left = random.randint(0, image_w - target_w)
        else:
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


def ms_mask_collate(batch):
    images, heights, masks, names = zip(*batch)
    return torch.stack(images), torch.stack(heights), torch.stack(masks), list(names)


# ============================================================
# Height range statistics
# ============================================================
def calculate_height_range(dataset_path, height_dir, height_suffix, txt_files):
    height_dir_full = os.path.join(dataset_path, height_dir)

    valid_height_min = 0.0
    valid_height_max = 6.0

    global_min = np.inf
    global_max = -np.inf
    total_valid_count = 0

    print(f"过滤规则：仅保留 {valid_height_min}-{valid_height_max} 米范围内的树高值")

    for txt_file in txt_files:
        txt_path = os.path.join(dataset_path, txt_file)

        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"txt 文件不存在: {txt_path}")

        with open(txt_path, "r", encoding="gbk", errors="ignore") as file:
            prefixes = [line.strip() for line in file if line.strip()]

        for raw in prefixes:
            prefix = os.path.splitext(raw)[0]
            height_path = os.path.join(height_dir_full, f"{prefix}{height_suffix}")

            if not os.path.exists(height_path):
                continue

            height_np = read_height_tif_safely(height_path)

            valid = (
                np.isfinite(height_np)
                & (height_np >= valid_height_min)
                & (height_np <= valid_height_max)
            )

            if np.any(valid):
                global_min = min(global_min, float(np.min(height_np[valid])))
                global_max = max(global_max, float(np.max(height_np[valid])))
                total_valid_count += int(np.sum(valid))

    if total_valid_count == 0:
        print("未统计到有效树高值，使用默认范围 0.0-6.0 m")
        return 0.0, 6.0

    print(f"有效像元数：{total_valid_count}")
    print(f"树高最小值：{global_min:.3f} m，最大值：{global_max:.3f} m")

    return float(global_min), float(global_max)


def assert_input_shape(shape):
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("input_shape 必须是长度为 2 的 [H, W]。")

    if shape[0] % 32 != 0 or shape[1] % 32 != 0:
        raise ValueError("input_shape 的高宽必须为 32 的倍数。")


def load_matching_weights(model, model_path, device, local_rank=0):
    """
    Flexible checkpoint loading.

    It only loads keys whose names and shapes match the current model.
    """
    if model_path in ["", None]:
        if local_rank == 0:
            print("model_path 为空，从头训练。")
        return

    if not os.path.exists(model_path):
        if local_rank == 0:
            print(f"权重文件不存在: {model_path}，从头训练。")
        return

    if local_rank == 0:
        print(f"加载预训练权重：{model_path}")

    try:
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        pretrained_dict = torch.load(model_path, map_location=device)

    if isinstance(pretrained_dict, dict):
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        elif "model_state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["model_state_dict"]

    normalized_dict = {}

    for key, value in pretrained_dict.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        normalized_dict[new_key] = value

    model_dict = model.state_dict()

    load_key = []
    no_load_key = []
    temp_dict = {}

    for key, value in normalized_dict.items():
        if key in model_dict and np.shape(model_dict[key]) == np.shape(value):
            temp_dict[key] = value
            load_key.append(key)
        else:
            no_load_key.append(key)

    model_dict.update(temp_dict)
    model.load_state_dict(model_dict, strict=False)

    if local_rank == 0:
        print("\nSuccessful Load Key Num:", len(load_key))
        print("Fail To Load Key Num:", len(no_load_key))

        if len(no_load_key) > 0:
            print("Fail keys examples:", no_load_key[:30])

        print(
            "\n\033[1;33;44m提示：输入 stem 因通道不同没载入是正常现象；"
            "删除 BIE 后，旧权重中的 bie 相关 key 没载入也是正常现象。"
            "如果 encoder 大面积没载入才需要检查。\033[0m"
        )


# ============================================================
# Main training
# ============================================================
if __name__ == "__main__":
    # ---------------- Basic config ---------------- #
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = True
    fp16 = True

    # ---------------- Model config ---------------- #
    backbone = "focal_s"
    in_channels = 9
    out_channels = 1
    pretrained = False
    model_path = ""

    use_svit = True
    svit_on_f16 = True
    svit_on_f32 = True

    input_shape = [640, 640]
    assert_input_shape(input_shape)

    # ---------------- Dataset paths ---------------- #
    dataset_path = "TreeHeightDataset"
    ms_dir = "RSImages"
    mask_dir = "mask"
    height_dir = "heights"

    ms_suffix = ".tif"
    mask_suffix = ".png"
    height_suffix = "_processed.tif"

    txt_files = ["train.txt", "val.txt"]

    MS_IN_CHANNELS = 8
    MS_NORM_MODE = "percentile"

    # ---------------- Training strategy ---------------- #
    Init_Epoch = 0
    Freeze_Epoch = 50
    UnFreeze_Epoch = 300
    Freeze_Train = True
    Freeze_batch_size = 2
    Unfreeze_batch_size = 2

    # ---------------- Optimizer and scheduler ---------------- #
    Init_lr = 1e-5
    Min_lr = 1e-7
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = "cos"
    save_period = 500
    save_dir = "logs_reg_ms_mask_focalsvit_no_bie"

    # ---------------- Loss ---------------- #
    loss_type = "combined"
    loss_alpha = 0.5
    num_workers = 0

    seed_everything(seed)

    # ---------------- Device / distributed ---------------- #
    ngpus_per_node = torch.cuda.device_count()

    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if (torch.cuda.is_available() and Cuda) else "cpu")
        local_rank = 0
        rank = 0

    if local_rank == 0:
        print("\n[模块开关配置]")
        print(f"  network     = FocalSVITCrownSegmentationNet")
        print(f"  backbone    = {backbone}")
        print(f"  use_svit    = {use_svit}")
        print(f"  svit_on_f16 = {svit_on_f16}")
        print(f"  svit_on_f32 = {svit_on_f32}")
        print(f"  use_bie     = False")
        print(f"  in_channels = {in_channels} (8band + mask)")
        print(f"  ms_dir      = {ms_dir}\n")

    # ---------------- Height range ---------------- #
    height_min, height_max = calculate_height_range(
        dataset_path,
        height_dir,
        height_suffix,
        txt_files,
    )

    # ---------------- Model ---------------- #
    model = FocalSVITCrownSegmentationNet(
        num_classes=out_channels,
        pretrained=pretrained,
        backbone=backbone,
        in_channels=in_channels,
        use_svit=use_svit,
        svit_on_f16=svit_on_f16,
        svit_on_f32=svit_on_f32,
        svit_stoken_size=(4, 4),
        svit_heads=8,
        svit_n_iter=1,
    )

    if local_rank == 0:
        if hasattr(model, "get_model_profile"):
            print("[Model Profile]", model.get_model_profile())

    if not pretrained:
        weights_init(model)

    load_matching_weights(
        model=model,
        model_path=model_path,
        device=device,
        local_rank=local_rank,
    )

    # ---------------- AMP GradScaler ---------------- #
    scaler = None

    if fp16:
        try:
            from torch.amp import GradScaler as _GradScaler

            try:
                scaler = _GradScaler("cuda")
            except TypeError:
                scaler = _GradScaler()

        except Exception:
            from torch.cuda.amp import GradScaler as _GradScaler

            scaler = _GradScaler()

    model_train = model.train()

    # ---------------- SyncBN and parallel ---------------- #
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)

    elif sync_bn and local_rank == 0:
        print("SyncBN 仅在分布式多卡时有效；当前已忽略。")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------- Read train / val split ---------------- #
    with open(os.path.join(dataset_path, "train.txt"), "r", encoding="gbk", errors="ignore") as file:
        train_lines = [line.strip() for line in file if line.strip()]

    with open(os.path.join(dataset_path, "val.txt"), "r", encoding="gbk", errors="ignore") as file:
        val_lines = [line.strip() for line in file if line.strip()]

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=out_channels,
            backbone=backbone,
            model_path=model_path,
            input_shape=input_shape,
            Init_Epoch=Init_Epoch,
            Freeze_Epoch=Freeze_Epoch,
            UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size,
            Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr,
            Min_lr=Min_lr,
            optimizer_type=optimizer_type,
            momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period,
            save_dir=save_dir,
            num_workers=num_workers,
            num_train=num_train,
            num_val=num_val,
        )

    # ---------------- Dataset check ---------------- #
    print("\n[验证输入输出形状]")

    ds_check = MsMaskDataset(
        val_lines,
        input_shape,
        num_classes=out_channels,
        train=False,
        dataset_path=dataset_path,
        ms_dir=ms_dir,
        mask_dir=mask_dir,
        height_dir=height_dir,
        ms_suffix=ms_suffix,
        mask_suffix=mask_suffix,
        height_suffix=height_suffix,
        height_min=height_min,
        height_max=height_max,
        ms_norm_mode=MS_NORM_MODE,
        ms_in_channels=MS_IN_CHANNELS,
    )

    sample_img, sample_height, sample_mask, name = ds_check[0]

    if local_rank == 0:
        print(f"样本: {name}")
        print(f"image shape: {tuple(sample_img.shape)} (expect [9,H,W])")
        print(f"height shape: {tuple(sample_height.shape)} (expect [1,H,W])")
        print(f"loss mask shape: {tuple(sample_mask.shape)} (expect [1,H,W])")
        print(f"image min/max: {sample_img.min().item():.4f}/{sample_img.max().item():.4f}")
        print(f"height min/max: {sample_height.min().item():.4f}/{sample_height.max().item():.4f}")
        print(f"loss mask sum: {sample_mask.sum().item():.1f}")

    train_dataset = MsMaskDataset(
        train_lines,
        input_shape,
        num_classes=out_channels,
        train=True,
        dataset_path=dataset_path,
        ms_dir=ms_dir,
        mask_dir=mask_dir,
        height_dir=height_dir,
        ms_suffix=ms_suffix,
        mask_suffix=mask_suffix,
        height_suffix=height_suffix,
        height_min=height_min,
        height_max=height_max,
        ms_norm_mode=MS_NORM_MODE,
        ms_in_channels=MS_IN_CHANNELS,
    )

    val_dataset = MsMaskDataset(
        val_lines,
        input_shape,
        num_classes=out_channels,
        train=False,
        dataset_path=dataset_path,
        ms_dir=ms_dir,
        mask_dir=mask_dir,
        height_dir=height_dir,
        ms_suffix=ms_suffix,
        mask_suffix=mask_suffix,
        height_suffix=height_suffix,
        height_min=height_min,
        height_max=height_max,
        ms_norm_mode=MS_NORM_MODE,
        ms_in_channels=MS_IN_CHANNELS,
    )

    # ---------------- DataLoader and training control ---------------- #
    UnFreeze_flag = False

    if Freeze_Train:
        model.freeze_backbone()

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    Init_lr_fit = Init_lr
    Min_lr_fit = Min_lr

    optimizer = {
        "adam": optim.Adam(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "adamw": optim.AdamW(
            model.parameters(),
            Init_lr_fit,
            betas=(momentum, 0.999),
            weight_decay=weight_decay,
        ),
        "sgd": optim.SGD(
            model.parameters(),
            Init_lr_fit,
            momentum=momentum,
            nesterov=True,
            weight_decay=weight_decay,
        ),
    }[optimizer_type]

    lr_scheduler_func = get_lr_scheduler(
        lr_decay_type,
        Init_lr_fit,
        Min_lr_fit,
        UnFreeze_Epoch,
    )

    epoch_step = num_train // batch_size
    epoch_step_val = max(1, num_val // batch_size)

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            shuffle=False,
        )

        batch_size = batch_size // ngpus_per_node
        shuffle_train = False

    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    gen = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ms_mask_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=ms_mask_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
    )

    # ---------------- Logging ---------------- #
    if local_rank == 0:
        time_str = datetime.datetime.strftime(
            datetime.datetime.now(),
            "%Y%m%d_%H%M%S_%f",
        )

        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        os.makedirs(log_dir, exist_ok=True)

        loss_history = LossHistory(
            log_dir,
            model,
            input_shape=input_shape,
        )

        with open(os.path.join(log_dir, "height_range.txt"), "w", encoding="utf-8") as file:
            file.write(f"height_min={height_min:.4f}\n")
            file.write(f"height_max={height_max:.4f}\n")

        with open(os.path.join(log_dir, "exp_config.txt"), "w", encoding="utf-8") as file:
            file.write("network=FocalSVITCrownSegmentationNet\n")
            file.write(f"backbone={backbone}\n")
            file.write("use_bie=False\n")
            file.write(f"use_svit={use_svit}\n")
            file.write(f"svit_on_f16={svit_on_f16}\n")
            file.write(f"svit_on_f32={svit_on_f32}\n")
            file.write(f"in_channels={in_channels}\n")
            file.write(f"ms_dir={ms_dir}\n")
            file.write(f"ms_suffix={ms_suffix}\n")
            file.write(f"ms_in_channels={MS_IN_CHANNELS}\n")
            file.write(f"ms_norm={MS_NORM_MODE}\n")
            file.write(f"loss_type={loss_type}\n")
            file.write(f"loss_alpha={loss_alpha}\n")

    else:
        loss_history = None

    # ---------------- Training loop ---------------- #
    print("\n开始训练...")

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            Init_lr_fit = Init_lr
            Min_lr_fit = Min_lr

            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type,
                Init_lr_fit,
                Min_lr_fit,
                UnFreeze_Epoch,
            )

            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = max(1, num_val // batch_size)

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

            if distributed:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=ms_mask_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
            )

            gen_val = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=ms_mask_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed),
            )

            UnFreeze_flag = True

        if distributed and train_sampler is not None:
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
            gen,
            gen_val,
            UnFreeze_Epoch,
            Cuda,
            loss_type,
            loss_alpha,
            fp16,
            scaler,
            save_period,
            save_dir,
            local_rank,
            height_min=height_min,
            height_max=height_max,
        )

        if distributed:
            dist.barrier()

    if local_rank == 0 and loss_history is not None and hasattr(loss_history, "writer"):
        loss_history.writer.close()

    print("训练完成！")