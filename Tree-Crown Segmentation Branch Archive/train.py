import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

from nets.lsnet_bifost_segmentation import LSNetBiFoSTSegmentation
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import (
    UnetDataset as SegmentationDataset,
    unet_dataset_collate as segmentation_dataset_collate,
)
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


# ============================================================
# Manual configuration
# ============================================================

# Device and reproducibility
USE_CUDA = True
SEED = 11
USE_DISTRIBUTED = False
USE_SYNC_BN = True
USE_FP16 = False
EARLY_STOP_PATIENCE = None

# Dataset and model
NUM_CLASSES = 2
BACKBONE = "lsnet_b"  # options: lsnet_t / lsnet_s / lsnet_b
PRETRAINED = False
MODEL_PATH = r"model_data/best_epoch_weights.pth"  # "" means training from scratch
INPUT_SHAPE = [640, 640]

# Training schedule
INIT_EPOCH = 0
FREEZE_EPOCH = 50
UNFREEZE_EPOCH = 300
FREEZE_TRAIN = True
FREEZE_BATCH_SIZE = 4
UNFREEZE_BATCH_SIZE = 4

# Optimizer and learning rate
INIT_LR = 1e-4
MIN_LR_RATIO = 0.01
OPTIMIZER_TYPE = "adamw"  # adam / adamw / sgd
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_DECAY_TYPE = "cos"  # cos / step

# Logging and evaluation
SAVE_PERIOD = 20
BASE_SAVE_DIR = "logs"
EVAL_FLAG = True
EVAL_PERIOD = 1

# Loss functions
DICE_LOSS = True
FOCAL_LOSS = False

# Dataset and DataLoader
NUM_WORKERS = 0
VOC_PATH = "VOCdevkit"

# Ablation switches
USE_BIE = 1
USE_HPA = 1

# Optional experiment tag
RUN_TAG = ""

# Output-head prior initialization
# Important:
# If MODEL_PATH is a trained checkpoint, keep this False.
# Otherwise, setting it True will re-initialize model.final after loading.
INIT_HEAD_PRIOR = False
FOREGROUND_PRIOR = 0.01


# ============================================================
# Utility functions
# ============================================================
def validate_input_shape(input_shape):
    """Validate whether input shape is [H, W] and divisible by 32."""
    if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
        raise ValueError("INPUT_SHAPE must be a list or tuple with two elements: [height, width].")

    height, width = input_shape
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            "Both height and width in INPUT_SHAPE must be divisible by 32, "
            "for example [512, 512], [640, 640], or [1024, 1024]."
        )


def build_run_tag():
    """Build a readable run tag from backbone and ablation switches."""
    auto_suffix = f"{BACKBONE}_bie{USE_BIE}_hpa{USE_HPA}"
    return RUN_TAG.strip() or auto_suffix


def load_matching_weights(model, model_path, device, local_rank=0):
    """
    Load only shape-matched weights.

    This keeps the script robust when the output head or minor settings differ.
    """
    if model_path in ["", None]:
        return

    if local_rank == 0:
        print(f"Load weights from: {model_path}")

    model_state = model.state_dict()

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    loaded_keys = []
    skipped_keys = []
    matched_state = {}

    for key, value in checkpoint.items():
        if key in model_state and np.shape(model_state[key]) == np.shape(value):
            matched_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model_state.update(matched_state)
    model.load_state_dict(model_state)

    if local_rank == 0:
        print("\nSuccessful Load Key:", str(loaded_keys)[:500], "……")
        print("Successful Load Key Num:", len(loaded_keys))
        print("\nFail To Load Key:", str(skipped_keys)[:500], "……")
        print("Fail To Load Key Num:", len(skipped_keys))
        print(
            "\n\033[1;33;44mNote: it is normal if the output head is not loaded; "
            "it is usually problematic if many backbone keys are not loaded.\033[0m"
        )


def initialize_output_head_prior(model, num_classes, foreground_prior=0.01, local_rank=0):
    """
    Initialize the segmentation output head with a foreground prior.

    Use this only for training from scratch. Do not use it after loading a trained checkpoint.
    """
    try:
        with torch.no_grad():
            logit_prior = float(np.log(foreground_prior / max(1e-8, 1.0 - foreground_prior)))

            model.final.weight.zero_()

            if model.final.bias is None:
                model.final.bias = torch.nn.Parameter(torch.zeros(num_classes))

            model.final.bias.data[:] = 0.0
            if num_classes >= 2:
                model.final.bias.data[1] = logit_prior

            if local_rank == 0:
                print(
                    f"[Init head prior] foreground bias set to {logit_prior:.3f} "
                    f"(prior={foreground_prior})"
                )

    except Exception as error:
        if local_rank == 0:
            print(f"[Init head prior] skipped because no compatible .final layer was found: {error}")


def compute_class_weights(lines, mask_dir, sample_cap=800, beta=0.999):
    """Estimate background/foreground class weights from mask samples."""
    class_counts = np.zeros(2, dtype=np.float64)
    num_samples = min(sample_cap, len(lines))
    image_names = [line.strip() for line in lines[:num_samples]]

    for image_name in image_names:
        mask_path = os.path.join(mask_dir, f"{image_name}.png")
        if not os.path.exists(mask_path):
            continue

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        class_counts[0] += (mask == 0).sum()
        class_counts[1] += (mask == 1).sum()

    effective_num = (1.0 - np.power(beta, class_counts)) / (1.0 - beta + 1e-12)
    class_weights = 1.0 / np.maximum(effective_num, 1e-12)
    class_weights = class_weights / class_weights.sum()
    class_weights = np.clip(class_weights, 0.2, 0.8).astype(np.float32)

    return class_weights


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
        collate_fn=segmentation_dataset_collate,
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
        collate_fn=segmentation_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=SEED),
    )

    return train_loader, val_loader


def has_invalid_loss(loss_history):
    """Check whether latest training or validation loss is NaN/Inf."""
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

    if USE_DISTRIBUTED:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)

        if local_rank == 0:
            print(f"[{os.getpid()}] rank={rank}, local_rank={local_rank}, training...")
            print("GPU Device Count:", num_gpus)

    else:
        device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
        local_rank = 0
        rank = 0

    # --------------------------------------------------------
    # Build LSNet-BiFoST segmentation model
    # --------------------------------------------------------
    model = LSNetBiFoSTSegmentation(
        num_classes=NUM_CLASSES,
        pretrained=PRETRAINED,
        backbone=BACKBONE,
        use_bie=bool(USE_BIE),
        use_hpa=bool(USE_HPA),
    ).train()

    if not PRETRAINED:
        weights_init(model)

    load_matching_weights(
        model=model,
        model_path=MODEL_PATH,
        device=device,
        local_rank=local_rank,
    )

    if INIT_HEAD_PRIOR:
        initialize_output_head_prior(
            model=model,
            num_classes=NUM_CLASSES,
            foreground_prior=FOREGROUND_PRIOR,
            local_rank=local_rank,
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=INPUT_SHAPE)
    else:
        loss_history = None

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------
    scaler = None
    if USE_FP16:
        try:
            from torch.amp import GradScaler as _GradScaler

            scaler = _GradScaler()
        except Exception:
            from torch.cuda.amp import GradScaler as _GradScaler

            scaler = _GradScaler()

    model_train = model.train()

    # --------------------------------------------------------
    # SyncBN and parallel training
    # --------------------------------------------------------
    if USE_SYNC_BN and num_gpus > 1 and USE_DISTRIBUTED:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif USE_SYNC_BN and local_rank == 0:
        print("SyncBN is only effective in distributed multi-GPU training; it is ignored here.")

    if USE_CUDA:
        if USE_DISTRIBUTED:
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

    # --------------------------------------------------------
    # Dataset split
    # --------------------------------------------------------
    train_txt = os.path.join(VOC_PATH, "VOC2007/ImageSets/Segmentation/train.txt")
    val_txt = os.path.join(VOC_PATH, "VOC2007/ImageSets/Segmentation/val.txt")

    with open(train_txt, "r") as file:
        train_lines = file.readlines()

    with open(val_txt, "r") as file:
        val_lines = file.readlines()

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

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------
    mask_dir = os.path.join(VOC_PATH, "VOC2007/SegmentationClass")
    class_weights = compute_class_weights(train_lines, mask_dir)

    if local_rank == 0:
        print(f"[Auto class weights] -> {class_weights}")

    # --------------------------------------------------------
    # Dataset and DataLoader
    # --------------------------------------------------------
    is_unfrozen = False

    if FREEZE_TRAIN:
        model.freeze_backbone()

    batch_size = FREEZE_BATCH_SIZE if FREEZE_TRAIN else UNFREEZE_BATCH_SIZE

    init_lr_fit = INIT_LR
    min_lr_fit = INIT_LR * MIN_LR_RATIO

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

    train_dataset = SegmentationDataset(
        train_lines,
        INPUT_SHAPE,
        NUM_CLASSES,
        True,
        VOC_PATH,
    )
    val_dataset = SegmentationDataset(
        val_lines,
        INPUT_SHAPE,
        NUM_CLASSES,
        False,
        VOC_PATH,
    )

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
    # Evaluation callback
    # --------------------------------------------------------
    if local_rank == 0:
        eval_callback = EvalCallback(
            model=model,
            input_shape=INPUT_SHAPE,
            num_classes=NUM_CLASSES,
            val_lines=val_lines,
            VOCdevkit_path=VOC_PATH,
            log_dir=save_dir,
            Cuda=USE_CUDA,
            eval_flag=EVAL_FLAG,
            period=EVAL_PERIOD,
            target_label_value=None,
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

    # --------------------------------------------------------
    # Training loop with NaN/Inf protection and optional early stopping
    # --------------------------------------------------------
    best_f1_seen = -1.0
    no_improve_count = 0

    for epoch in range(INIT_EPOCH, UNFREEZE_EPOCH):
        if epoch >= FREEZE_EPOCH and not is_unfrozen and FREEZE_TRAIN:
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

        fit_one_epoch(
            model_train,
            model,
            loss_history,
            eval_callback,
            optimizer,
            epoch,
            epoch_step,
            epoch_step_val,
            train_loader,
            val_loader,
            UNFREEZE_EPOCH,
            USE_CUDA,
            DICE_LOSS,
            FOCAL_LOSS,
            class_weights,
            NUM_CLASSES,
            USE_FP16,
            scaler,
            SAVE_PERIOD,
            save_dir,
            local_rank,
        )

        if local_rank == 0 and has_invalid_loss(loss_history):
            print(
                "⚠️ NaN/Inf loss detected. Training stopped early. "
                "Please resume from the latest best_f1 or best_epoch checkpoint."
            )
            break

        if local_rank == 0 and eval_callback is not None and EARLY_STOP_PATIENCE is not None:
            current_best_f1 = getattr(eval_callback, "best_f1", None)

            if current_best_f1 is not None:
                if current_best_f1 > best_f1_seen + 1e-8:
                    best_f1_seen = current_best_f1
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count >= EARLY_STOP_PATIENCE:
                    print(
                        f"⏹ Early stopping triggered: F1 did not improve for "
                        f"{EARLY_STOP_PATIENCE} epochs. best_f1={best_f1_seen:.4f}"
                    )
                    break

        if USE_DISTRIBUTED:
            dist.barrier()

    if local_rank == 0 and loss_history is not None and hasattr(loss_history, "writer"):
        loss_history.writer.close()