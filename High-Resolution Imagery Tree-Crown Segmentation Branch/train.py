import datetime
import os
from functools import partial
from collections import Counter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from nets.high_resolution_crown_segmentation import HighResolutionCrownSegmentationNet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


# =========================
# 手动配置区
# =========================
CUDA                = True
SEED                = 11
DISTRIBUTED         = False
SYNC_BN             = True
FP16                = False
EARLY_STOP_PATIENCE = None

NUM_CLASSES         = 2
BACKBONE            = "lsnet_b"
PRETRAINED          = False
MODEL_PATH          = r"logs/lsnet_b_bie1_hpa1_c8/best_f1_epoch_004.pth"
INPUT_SHAPE         = [640, 640]

# 多光谱 / 高分影像配置
VOC_PATH            = "VOCdevkit"
IMAGE_DIR_NAME      = "RSImages"
IMAGE_SUFFIX        = ".tif"
IN_CHANNELS         = 8
NORM_MODE           = "percentile"   # percentile / max / none

# 前景像元值：None = 自动识别
TARGET_LABEL_VALUE  = None

# ignore 像元值：没有就 None；常见 VOC ignore 是 255
MASK_IGNORE_VALUE   = None

# 如果 ignore 不确定，建议开启自动推断
AUTO_INFER_IGNORE_255 = True

# 训练策略
INIT_EPOCH          = 0
FREEZE_EPOCH        = 50
UNFREEZE_EPOCH      = 300
FREEZE_TRAIN        = True
FREEZE_BATCH_SIZE   = 4
UNFREEZE_BATCH_SIZE = 4

# 学习率 / 优化器
INIT_LR             = 1e-4
MIN_LR_RATIO        = 0.01
OPTIMIZER_TYPE      = "adamw"        # adam / adamw / sgd
MOMENTUM            = 0.9
WEIGHT_DECAY        = 1e-4
LR_DECAY_TYPE       = "cos"

# 日志 / 评估
SAVE_PERIOD         = 300
BASE_SAVE_DIR       = "logs"
EVAL_FLAG           = True
EVAL_PERIOD         = 1

# 损失
DICE_LOSS           = True
FOCAL_LOSS          = False

NUM_WORKERS         = 0

USE_BIE             = 1
USE_HPA             = 1
RUN_TAG             = ""


def assert_input_shape(shape):
    """检查输入尺寸是否为 [H, W] 且高宽为 32 的倍数。"""
    if not (isinstance(shape, (list, tuple)) and len(shape) == 2):
        raise ValueError("INPUT_SHAPE 必须是长度为 2 的 [H, W]。")

    if shape[0] % 32 != 0 or shape[1] % 32 != 0:
        raise ValueError("INPUT_SHAPE 的高和宽必须为 32 的倍数。")


def read_text_lines_with_fallback(txt_path):
    """读取 train.txt / val.txt，兼容中文路径和 GBK 编码。"""
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"划分文件不存在: {txt_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(txt_path, "r", encoding=encoding) as file:
                return [line.strip() for line in file if line.strip()]
        except UnicodeDecodeError:
            continue

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as file:
        return [line.strip() for line in file if line.strip()]


def read_mask_u8(mask_path):
    """
    安全读取 mask。

    P 模式：
        直接读取索引值，例如 38。

    其他模式：
        转为 L 灰度图。
    """
    mask = Image.open(mask_path)

    if mask.mode == "P":
        arr = np.array(mask, dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L"), dtype=np.uint8)

    return arr


def scan_mask_values(lines, mask_dir, scan_max=500):
    """
    扫描 mask 的唯一值分布。

    用于确认：
        背景值
        前景值
        ignore 值
    """
    counts = Counter()
    scanned = 0

    names = [line.strip().split()[0] for line in lines[:scan_max] if line.strip()]

    for name in names:
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            continue

        arr = read_mask_u8(mask_path)

        unique_values, frequencies = np.unique(arr, return_counts=True)

        for value, count in zip(unique_values, frequencies):
            counts[int(value)] += int(count)

        scanned += 1

    return counts, scanned


def auto_detect_fg_value(lines, mask_dir, scan_max=200, ignore_value=None):
    """
    自动识别前景像元值。

    规则：
        统计非 0、非 ignore 的像元值；
        取出现次数最多的值作为前景。
    """
    counts = Counter()
    scanned = 0

    names = [line.strip().split()[0] for line in lines[:scan_max] if line.strip()]

    for name in names:
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            continue

        arr = read_mask_u8(mask_path)

        unique_values, frequencies = np.unique(arr, return_counts=True)

        for value, count in zip(unique_values, frequencies):
            value = int(value)

            if value == 0:
                continue

            if ignore_value is not None and value == int(ignore_value):
                continue

            counts[value] += int(count)

        scanned += 1

    if scanned == 0 or len(counts) == 0:
        return 1, counts, scanned

    foreground_value = counts.most_common(1)[0][0]

    return int(foreground_value), counts, scanned


def compute_cls_weights(
    lines,
    mask_dir,
    fg_value,
    sample_cap=800,
    beta=0.999,
    ignore_value=None,
):
    """
    计算二分类类别权重。

    只统计：
        0 = background
        fg_value = foreground

    其他值默认不参与统计。
    """
    counts = np.zeros(2, np.float64)

    sample_count = min(sample_cap, len(lines))
    names = [line.strip().split()[0] for line in lines[:sample_count] if line.strip()]

    for name in names:
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            continue

        arr = read_mask_u8(mask_path)

        if ignore_value is not None:
            valid = arr != int(ignore_value)
        else:
            valid = np.ones_like(arr, dtype=bool)

        is_background = (arr == 0) & valid
        is_foreground = (arr == int(fg_value)) & valid

        counts[0] += is_background.sum()
        counts[1] += is_foreground.sum()

    effective_number = (1.0 - np.power(beta, counts)) / (1.0 - beta + 1e-12)

    weights = 1.0 / np.maximum(effective_number, 1e-12)
    weights = weights / weights.sum()
    weights = np.clip(weights, 0.2, 0.8).astype(np.float32)

    return weights


def load_matching_weights(model, model_path, device, local_rank=0):
    """
    加载形状匹配的权重。

    注意：
        类名和文件名变化不会影响权重加载；
        state_dict 主要根据模块属性名匹配，例如：
            encoder
            hpa16
            hpa32
            up4
            up3
            up2
            up1
            out_head
            final
    """
    if model_path in ["", None]:
        if local_rank == 0:
            print("MODEL_PATH is empty. Training will start from scratch.")
        return

    if not os.path.exists(model_path):
        if local_rank == 0:
            print(f"Checkpoint does not exist: {model_path}. Training will continue without loading.")
        return

    if local_rank == 0:
        print(f"Load weights: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

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

    load_key = []
    no_load_key = []
    temp_dict = {}

    for key, value in normalized_checkpoint.items():
        if key in model_state and np.shape(model_state[key]) == np.shape(value):
            temp_dict[key] = value
            load_key.append(key)
        else:
            no_load_key.append(key)

    model_state.update(temp_dict)
    model.load_state_dict(model_state, strict=False)

    if local_rank == 0:
        print("\nSuccessful Load Key Num:", len(load_key))
        print("Fail To Load Key Num:", len(no_load_key))

        if len(no_load_key) > 0:
            print("Fail To Load Key Examples:", no_load_key[:20])

        print(
            "\n\033[1;33;44m提示：final/head 或输入第一层没载入通常可能正常；"
            "如果大量 encoder 权重没载入才需要检查模型结构。\033[0m"
        )


def initialize_segmentation_head_prior(model, num_classes, local_rank=0):
    """
    初始化输出头前景先验。

    注意：
        如果已经加载了训练好的权重，一般不建议重新清零 final。
        因此这里只在新训练时使用更合理。
    """
    try:
        with torch.no_grad():
            prior_fg = 0.01
            logit_prior = float(np.log(prior_fg / max(1e-8, 1.0 - prior_fg)))

            model.final.weight.zero_()

            if model.final.bias is None:
                model.final.bias = torch.nn.Parameter(torch.zeros(num_classes))

            model.final.bias.data[:] = 0.0

            if num_classes >= 2:
                model.final.bias.data[1] = logit_prior

            if local_rank == 0:
                print(f"[Init head prior] set foreground bias to {logit_prior:.3f} (prior={prior_fg})")

    except Exception as error:
        if local_rank == 0:
            print(f"[Init head prior] skip: {error}")


if __name__ == "__main__":
    assert_input_shape(INPUT_SHAPE)

    auto_suffix = f"{BACKBONE}_bie{USE_BIE}_hpa{USE_HPA}_c{IN_CHANNELS}"
    run_tag = RUN_TAG.strip() or auto_suffix

    save_dir = os.path.join(BASE_SAVE_DIR, run_tag)
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(SEED)

    ngpus_per_node = torch.cuda.device_count()

    if DISTRIBUTED:
        dist.init_process_group(backend="nccl")

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)

        if local_rank == 0:
            print(f"[{os.getpid()}] rank={rank}, local_rank={local_rank}, training...")
            print("GPU Device Count:", ngpus_per_node)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() and CUDA else "cpu")
        local_rank = 0
        rank = 0

    # =========================
    # 模型
    # =========================
    model = HighResolutionCrownSegmentationNet(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
        in_channels=IN_CHANNELS,
    ).train()

    if local_rank == 0:
        if hasattr(model, "get_model_profile"):
            print("[Model Profile]", model.get_model_profile())
        else:
            print("[Model] HighResolutionCrownSegmentationNet")

    if not PRETRAINED:
        weights_init(model)

    # 加载权重
    load_matching_weights(
        model=model,
        model_path=MODEL_PATH,
        device=device,
        local_rank=local_rank,
    )

    # 如果是从头训练，可以初始化 final 前景先验。
    # 如果 MODEL_PATH 有效，避免覆盖已加载的 final 权重。
    if MODEL_PATH in ["", None] or not os.path.exists(MODEL_PATH):
        initialize_segmentation_head_prior(
            model=model,
            num_classes=NUM_CLASSES,
            local_rank=local_rank,
        )

    # =========================
    # 日志
    # =========================
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=INPUT_SHAPE)
    else:
        loss_history = None

    # =========================
    # AMP
    # =========================
    scaler = None

    if FP16:
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

    # =========================
    # SyncBN / 并行
    # =========================
    if SYNC_BN and ngpus_per_node > 1 and DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif SYNC_BN and local_rank == 0:
        print("SyncBN 仅在分布式多卡时有效；当前已忽略。")

    if CUDA:
        if DISTRIBUTED:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = False
            cudnn.deterministic = True
            model_train = model_train.cuda()

    # =========================
    # 数据划分
    # =========================
    train_txt = os.path.join(VOC_PATH, "VOC2007", "ImageSets", "Segmentation", "train.txt")
    val_txt = os.path.join(VOC_PATH, "VOC2007", "ImageSets", "Segmentation", "val.txt")

    train_lines = read_text_lines_with_fallback(train_txt)
    val_lines = read_text_lines_with_fallback(val_txt)

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=NUM_CLASSES,
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
            Min_lr=INIT_LR * MIN_LR_RATIO,
            optimizer_type=OPTIMIZER_TYPE,
            momentum=MOMENTUM,
            lr_decay_type=LR_DECAY_TYPE,
            save_period=SAVE_PERIOD,
            save_dir=save_dir,
            num_workers=NUM_WORKERS,
            num_train=num_train,
            num_val=num_val,
        )

        print(f"[Ablation] use_bie={USE_BIE}, use_hpa={USE_HPA}")
        print(f"[Run Tag] {run_tag}")
        print(
            f"[High-resolution imagery] "
            f"image_dir={IMAGE_DIR_NAME}, "
            f"suffix={IMAGE_SUFFIX}, "
            f"in_channels={IN_CHANNELS}, "
            f"norm={NORM_MODE}"
        )

    # =========================
    # mask 值扫描
    # =========================
    mask_dir = os.path.join(VOC_PATH, "VOC2007", "SegmentationClass")

    if local_rank == 0:
        train_counts, train_scanned = scan_mask_values(train_lines, mask_dir, scan_max=500)
        val_counts, val_scanned = scan_mask_values(val_lines, mask_dir, scan_max=500)

        print(f"[MaskScan][train] scanned={train_scanned}, top20={train_counts.most_common(20)}")
        print(f"[MaskScan][train] all_keys={sorted(train_counts.keys())}")
        print(f"[MaskScan][val]   scanned={val_scanned}, top20={val_counts.most_common(20)}")
        print(f"[MaskScan][val]   all_keys={sorted(val_counts.keys())}")

    # =========================
    # 自动推断 ignore=255
    # =========================
    if MASK_IGNORE_VALUE is None and AUTO_INFER_IGNORE_255:
        try:
            tmp_counts, _ = scan_mask_values(train_lines, mask_dir, scan_max=200)

            if 255 in tmp_counts:
                MASK_IGNORE_VALUE = 255

                if local_rank == 0:
                    print("[Mask] AUTO infer ignore_index=255 because 255 exists in train masks.")

        except Exception as error:
            if local_rank == 0:
                print(f"[Mask] AUTO infer ignore_index failed: {error}")

    # =========================
    # 自动识别前景像元值
    # =========================
    if TARGET_LABEL_VALUE is None:
        fg_value, fg_counts, scanned = auto_detect_fg_value(
            train_lines,
            mask_dir,
            scan_max=200,
            ignore_value=MASK_IGNORE_VALUE,
        )

        if local_rank == 0:
            print(f"[Mask] auto-detect fg_raw_value = {fg_value} (scanned={scanned})")

            if len(fg_counts) > 0:
                print(f"[Mask] non-zero top values: {fg_counts.most_common(10)}")

    else:
        fg_value = int(TARGET_LABEL_VALUE)

        if local_rank == 0:
            print(f"[Mask] use specified fg_raw_value = {fg_value}")

    if local_rank == 0:
        print(f"[Mask] final fg_raw_value={fg_value}, ignore_raw_value={MASK_IGNORE_VALUE}")

    # =========================
    # Dataset
    # =========================
    train_dataset = UnetDataset(
        train_lines,
        INPUT_SHAPE,
        NUM_CLASSES,
        True,
        VOC_PATH,
        image_dir=IMAGE_DIR_NAME,
        image_suffix=IMAGE_SUFFIX,
        in_channels=IN_CHANNELS,
        norm_mode=NORM_MODE,
        target_label_value=fg_value,
        ignore_index=MASK_IGNORE_VALUE,
        debug_first_n=5 if local_rank == 0 else 0,
    )

    val_dataset = UnetDataset(
        val_lines,
        INPUT_SHAPE,
        NUM_CLASSES,
        False,
        VOC_PATH,
        image_dir=IMAGE_DIR_NAME,
        image_suffix=IMAGE_SUFFIX,
        in_channels=IN_CHANNELS,
        norm_mode=NORM_MODE,
        target_label_value=fg_value,
        ignore_index=MASK_IGNORE_VALUE,
        debug_first_n=5 if local_rank == 0 else 0,
    )

    # =========================
    # 类别权重
    # =========================
    cls_weights = compute_cls_weights(
        train_lines,
        mask_dir,
        fg_value=fg_value,
        sample_cap=800,
        beta=0.999,
        ignore_value=MASK_IGNORE_VALUE,
    )

    if local_rank == 0:
        print(f"[Auto cls_weights] -> {cls_weights}")

    # =========================
    # 冻结 / 解冻
    # =========================
    unfreeze_flag = False

    if FREEZE_TRAIN:
        model.freeze_backbone()

    batch_size = FREEZE_BATCH_SIZE if FREEZE_TRAIN else UNFREEZE_BATCH_SIZE

    init_lr_fit = INIT_LR
    min_lr_fit = INIT_LR * MIN_LR_RATIO

    optimizer_dict = {
        "adam": optim.Adam(
            model.parameters(),
            init_lr_fit,
            betas=(MOMENTUM, 0.999),
            weight_decay=WEIGHT_DECAY,
        ),
        "adamw": optim.AdamW(
            model.parameters(),
            init_lr_fit,
            betas=(MOMENTUM, 0.999),
            weight_decay=WEIGHT_DECAY,
        ),
        "sgd": optim.SGD(
            model.parameters(),
            init_lr_fit,
            momentum=MOMENTUM,
            nesterov=True,
            weight_decay=WEIGHT_DECAY,
        ),
    }

    if OPTIMIZER_TYPE not in optimizer_dict:
        raise ValueError(f"Unsupported optimizer type: {OPTIMIZER_TYPE}")

    optimizer = optimizer_dict[OPTIMIZER_TYPE]

    lr_scheduler_func = get_lr_scheduler(
        LR_DECAY_TYPE,
        init_lr_fit,
        min_lr_fit,
        UNFREEZE_EPOCH,
    )

    epoch_step = num_train // batch_size
    epoch_step_val = max(1, num_val // batch_size)

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续训练，请扩充数据集。")

    # =========================
    # DataLoader
    # =========================
    if DISTRIBUTED:
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
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=unet_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=unet_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
    )

    # =========================
    # EvalCallback
    # =========================
    if local_rank == 0:
        eval_callback = EvalCallback(
            model=model,
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            val_lines=val_lines,
            VOCdevkit_path=VOC_PATH,
            log_dir=save_dir,
            Cuda=CUDA,
            eval_flag=EVAL_FLAG,
            period=EVAL_PERIOD,
            image_dir=IMAGE_DIR_NAME,
            image_suffix=IMAGE_SUFFIX,
            in_channels=IN_CHANNELS,
            norm_mode=NORM_MODE,
            target_label_value=fg_value,
            ignore_index=MASK_IGNORE_VALUE,
            area_thr=0,
            perim_thr=0,
            circ_thr=0.0,
            save_visual_topk=0,
            debug_first_n=5,
            autodetect_scan_max=200,
            iou_thr=0.5,
        )
    else:
        eval_callback = None

    # =========================
    # 训练循环
    # =========================
    best_f1_seen = -1.0
    no_improve = 0

    for epoch in range(INIT_EPOCH, UNFREEZE_EPOCH):
        if epoch >= FREEZE_EPOCH and (not unfreeze_flag) and FREEZE_TRAIN:
            batch_size = UNFREEZE_BATCH_SIZE

            init_lr_fit = INIT_LR
            min_lr_fit = INIT_LR * MIN_LR_RATIO

            lr_scheduler_func = get_lr_scheduler(
                LR_DECAY_TYPE,
                init_lr_fit,
                min_lr_fit,
                UNFREEZE_EPOCH,
            )

            model.unfreeze_backbone()

            epoch_step = num_train // batch_size
            epoch_step_val = max(1, num_val // batch_size)

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续训练，请扩充数据集。")

            if DISTRIBUTED:
                batch_size = batch_size // ngpus_per_node

            gen = DataLoader(
                train_dataset,
                shuffle=shuffle_train,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
                collate_fn=unet_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
            )

            gen_val = DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                drop_last=False,
                collate_fn=unet_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
            )

            unfreeze_flag = True

        if DISTRIBUTED and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            gen,
            gen_val,
            UNFREEZE_EPOCH,
            CUDA,
            DICE_LOSS,
            FOCAL_LOSS,
            cls_weights,
            NUM_CLASSES,
            FP16,
            scaler,
            SAVE_PERIOD,
            save_dir,
            local_rank,
            ignore_raw_value=MASK_IGNORE_VALUE,
            fg_raw_value=fg_value,
        )

        # NaN / Inf 保护
        if local_rank == 0 and loss_history is not None:
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

            if train_invalid or val_invalid:
                print("检测到 Loss 为 NaN/Inf，提前停止训练。")
                break

        # 早停
        if local_rank == 0 and eval_callback is not None and EARLY_STOP_PATIENCE is not None:
            current_best = getattr(eval_callback, "best_f1", None)

            if current_best is not None:
                if current_best > best_f1_seen + 1e-8:
                    best_f1_seen = current_best
                    no_improve = 0
                else:
                    no_improve += 1

                    if no_improve >= EARLY_STOP_PATIENCE:
                        print(
                            f"早停触发：F1 在 {EARLY_STOP_PATIENCE} 个 epoch 内未提升 "
                            f"(best_f1={best_f1_seen:.4f})。"
                        )
                        break

        if DISTRIBUTED:
            dist.barrier()

    if local_rank == 0 and loss_history is not None and hasattr(loss_history, "writer"):
        loss_history.writer.close()
