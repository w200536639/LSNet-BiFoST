import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch

from nets.unet_training import MSE_Loss, MAE_Loss, Combined_Reg_Loss
from utils.utils import get_lr


# ============================================================
# Progress display settings
# ============================================================
# True：显示单行进度；False：只在每个 epoch 结束后打印结果
SHOW_PROGRESS = True

# 每隔多少个 iteration 刷新一次进度。
# 如果你的 PyCharm 仍然换行，可以把它改大，例如 20 或 50。
PROGRESS_REFRESH_STEPS = 5

# 至少间隔多少秒刷新一次。
PROGRESS_REFRESH_SECONDS = 1.0

# 每次打印时清空行宽，避免残留字符
PROGRESS_LINE_WIDTH = 140


# ============================================================
# Warning filter
# ============================================================
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*",
)


# ============================================================
# Helper functions
# ============================================================
def _safe_average(total_value, count):
    return total_value / max(1, count)


def _unpack_regression_batch(batch):
    """
    兼容：
        images, heights, masks
        images, heights, masks, names
    """
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        images, heights_norm, masks, _ = batch
    elif isinstance(batch, (list, tuple)) and len(batch) == 3:
        images, heights_norm, masks = batch
    else:
        raise ValueError(
            "fit_one_epoch_reg expects batch to be "
            "(images, heights, masks) or (images, heights, masks, names)."
        )

    return images, heights_norm, masks


def _move_to_device(images, heights_norm, masks, cuda, local_rank):
    if cuda:
        images = images.cuda(local_rank, non_blocking=True)
        heights_norm = heights_norm.cuda(local_rank, non_blocking=True)
        masks = masks.cuda(local_rank, non_blocking=True)

    return images, heights_norm, masks


def _binarize_mask_if_needed(masks):
    """
    保证 mask 为 0/1。
    返回：
        masks
        changed
    """
    with torch.no_grad():
        is_binary = torch.all((masks == 0) | (masks == 1))

    if not bool(is_binary):
        return (masks > 0.5).float(), True

    return masks, False


def _calc_regression_loss(outputs_norm, heights_norm, masks, loss_type, loss_alpha):
    if loss_type == "mse":
        return MSE_Loss(outputs_norm, heights_norm, masks)

    if loss_type == "mae":
        return MAE_Loss(outputs_norm, heights_norm, masks)

    if loss_type == "combined":
        return Combined_Reg_Loss(
            outputs_norm,
            heights_norm,
            masks,
            alpha=loss_alpha,
        )

    raise ValueError(f"不支持的损失类型：{loss_type}")


def _autocast_context(cuda, fp16):
    """
    兼容新版/旧版 PyTorch 的 AMP autocast。

    新版优先使用：
        torch.amp.autocast("cuda")

    旧版自动回退：
        torch.cuda.amp.autocast()
    """
    if not cuda or not fp16:
        return nullcontext()

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=True)
        except TypeError:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.cuda.amp.autocast(enabled=True)


def _clear_progress_line():
    """
    清空当前控制台行。
    """
    sys.stdout.write("\r" + " " * PROGRESS_LINE_WIDTH + "\r")
    sys.stdout.flush()


def _print_progress(
    phase,
    epoch,
    Epoch,
    current,
    total,
    loss_value,
    mae_value,
    lr_value=None,
    force=False,
    last_print_time=None,
):
    """
    单行刷新进度，避免 tqdm 在 PyCharm 中刷屏和出现乱码进度条。

    返回：
        当前刷新时间，供下一次节流使用。
    """
    if not SHOW_PROGRESS:
        return last_print_time

    now_time = time.time()

    if last_print_time is None:
        last_print_time = 0.0

    should_print = (
        force
        or current == 1
        or current >= total
        or current % max(1, PROGRESS_REFRESH_STEPS) == 0
        or (now_time - last_print_time) >= PROGRESS_REFRESH_SECONDS
    )

    if not should_print:
        return last_print_time

    percent = 100.0 * current / max(1, total)

    if lr_value is None:
        text = (
            f"{phase} Epoch {epoch + 1}/{Epoch} "
            f"{current}/{total} "
            f"{percent:6.2f}% "
            f"loss={loss_value:.4f} "
            f"MAE={mae_value:.4f} m"
        )
    else:
        text = (
            f"{phase} Epoch {epoch + 1}/{Epoch} "
            f"{current}/{total} "
            f"{percent:6.2f}% "
            f"loss={loss_value:.4f} "
            f"MAE={mae_value:.4f} m "
            f"lr={lr_value:.2e}"
        )

    if len(text) > PROGRESS_LINE_WIDTH - 2:
        text = text[:PROGRESS_LINE_WIDTH - 5] + "..."

    sys.stdout.write("\r" + text.ljust(PROGRESS_LINE_WIDTH))
    sys.stdout.flush()

    return now_time


def _finish_progress_line():
    """
    结束当前单行进度，换到下一行。
    """
    if SHOW_PROGRESS:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _save_model_weights(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)


# ============================================================
# Main function
# ============================================================
def fit_one_epoch_reg(
    model_train,
    model,
    loss_history,
    optimizer,
    epoch,
    epoch_step,
    epoch_step_val,
    gen,
    gen_val,
    Epoch,
    cuda,
    loss_type,
    loss_alpha,
    fp16,
    scaler,
    save_period,
    save_dir,
    local_rank=0,
    height_min=0.0,
    height_max=6.0,
):
    """
    单个 epoch 的训练 / 验证循环，适用于树高回归分支。

    这版特点：
        1. 不使用 tqdm，避免 PyCharm 控制台刷屏；
        2. 进度在一行内刷新；
        3. 不使用已弃用的 torch.cuda.amp.autocast 写法；
        4. batch 内不 print，避免冲乱进度条。
    """

    scale_range = height_max - height_min
    if abs(scale_range) <= 1e-12:
        scale_range = 1.0

    total_loss = 0.0
    total_mse_norm = 0.0
    total_mae_norm = 0.0
    total_mse_raw = 0.0
    total_mae_raw = 0.0

    val_total_loss = 0.0
    val_total_mse_norm = 0.0
    val_total_mae_norm = 0.0
    val_total_mse_raw = 0.0
    val_total_mae_raw = 0.0

    train_mask_warning_printed = False
    val_mask_warning_printed = False

    # ========================================================
    # Train
    # ========================================================
    model_train.train()

    if local_rank == 0:
        _clear_progress_line()

    train_last_print_time = None

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, heights_norm, masks = _unpack_regression_batch(batch)

        images, heights_norm, masks = _move_to_device(
            images=images,
            heights_norm=heights_norm,
            masks=masks,
            cuda=cuda,
            local_rank=local_rank,
        )

        masks, mask_changed = _binarize_mask_if_needed(masks)

        if local_rank == 0 and mask_changed and not train_mask_warning_printed:
            _finish_progress_line()
            print("[Warning] Train mask is not binary. It has been binarized automatically.")
            train_mask_warning_printed = True

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(cuda=cuda, fp16=fp16):
            outputs_norm = model_train(images)
            loss = _calc_regression_loss(
                outputs_norm=outputs_norm,
                heights_norm=heights_norm,
                masks=masks,
                loss_type=loss_type,
                loss_alpha=loss_alpha,
            )

        if fp16 and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            mse_norm = MSE_Loss(outputs_norm, heights_norm, masks)
            mae_norm = MAE_Loss(outputs_norm, heights_norm, masks)

            outputs_raw = outputs_norm * scale_range + height_min
            heights_raw = heights_norm * scale_range + height_min

            mse_raw = MSE_Loss(outputs_raw, heights_raw, masks)
            mae_raw = MAE_Loss(outputs_raw, heights_raw, masks)

        total_loss += float(loss.item())
        total_mse_norm += float(mse_norm.item())
        total_mae_norm += float(mae_norm.item())
        total_mse_raw += float(mse_raw.item())
        total_mae_raw += float(mae_raw.item())

        if local_rank == 0:
            current_iter = iteration + 1

            train_last_print_time = _print_progress(
                phase="Train",
                epoch=epoch,
                Epoch=Epoch,
                current=current_iter,
                total=epoch_step,
                loss_value=_safe_average(total_loss, current_iter),
                mae_value=_safe_average(total_mae_raw, current_iter),
                lr_value=get_lr(optimizer),
                force=False,
                last_print_time=train_last_print_time,
            )

    if local_rank == 0:
        _print_progress(
            phase="Train",
            epoch=epoch,
            Epoch=Epoch,
            current=epoch_step,
            total=epoch_step,
            loss_value=_safe_average(total_loss, epoch_step),
            mae_value=_safe_average(total_mae_raw, epoch_step),
            lr_value=get_lr(optimizer),
            force=True,
            last_print_time=train_last_print_time,
        )
        _finish_progress_line()

    # ========================================================
    # Validation
    # ========================================================
    model_train.eval()

    if local_rank == 0:
        _clear_progress_line()

    val_last_print_time = None

    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            images, heights_norm, masks = _unpack_regression_batch(batch)

            images, heights_norm, masks = _move_to_device(
                images=images,
                heights_norm=heights_norm,
                masks=masks,
                cuda=cuda,
                local_rank=local_rank,
            )

            masks, mask_changed = _binarize_mask_if_needed(masks)

            if local_rank == 0 and mask_changed and not val_mask_warning_printed:
                _finish_progress_line()
                print("[Warning] Val mask is not binary. It has been binarized automatically.")
                val_mask_warning_printed = True

            outputs_norm = model_train(images)

            loss = _calc_regression_loss(
                outputs_norm=outputs_norm,
                heights_norm=heights_norm,
                masks=masks,
                loss_type=loss_type,
                loss_alpha=loss_alpha,
            )

            mse_norm = MSE_Loss(outputs_norm, heights_norm, masks)
            mae_norm = MAE_Loss(outputs_norm, heights_norm, masks)

            outputs_raw = outputs_norm * scale_range + height_min
            heights_raw = heights_norm * scale_range + height_min

            mse_raw = MSE_Loss(outputs_raw, heights_raw, masks)
            mae_raw = MAE_Loss(outputs_raw, heights_raw, masks)

            val_total_loss += float(loss.item())
            val_total_mse_norm += float(mse_norm.item())
            val_total_mae_norm += float(mae_norm.item())
            val_total_mse_raw += float(mse_raw.item())
            val_total_mae_raw += float(mae_raw.item())

            if local_rank == 0:
                current_iter = iteration + 1

                val_last_print_time = _print_progress(
                    phase="Val  ",
                    epoch=epoch,
                    Epoch=Epoch,
                    current=current_iter,
                    total=epoch_step_val,
                    loss_value=_safe_average(val_total_loss, current_iter),
                    mae_value=_safe_average(val_total_mae_raw, current_iter),
                    lr_value=None,
                    force=False,
                    last_print_time=val_last_print_time,
                )

    if local_rank == 0:
        _print_progress(
            phase="Val  ",
            epoch=epoch,
            Epoch=Epoch,
            current=epoch_step_val,
            total=epoch_step_val,
            loss_value=_safe_average(val_total_loss, epoch_step_val),
            mae_value=_safe_average(val_total_mae_raw, epoch_step_val),
            lr_value=None,
            force=True,
            last_print_time=val_last_print_time,
        )
        _finish_progress_line()

    # ========================================================
    # Logging and saving
    # ========================================================
    if local_rank == 0:
        train_loss = _safe_average(total_loss, epoch_step)
        val_loss = _safe_average(val_total_loss, epoch_step_val)

        train_mse_norm = _safe_average(total_mse_norm, epoch_step)
        train_mae_norm = _safe_average(total_mae_norm, epoch_step)
        train_mse_m = _safe_average(total_mse_raw, epoch_step)
        train_mae_m = _safe_average(total_mae_raw, epoch_step)

        val_mse_norm = _safe_average(val_total_mse_norm, epoch_step_val)
        val_mae_norm = _safe_average(val_total_mae_norm, epoch_step_val)
        val_mse_m = _safe_average(val_total_mse_raw, epoch_step_val)
        val_mae_m = _safe_average(val_total_mae_raw, epoch_step_val)

        if loss_history is not None:
            loss_history.append_loss(epoch + 1, train_loss, val_loss)

        print(
            f"Epoch {epoch + 1}/{Epoch} | "
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_MAE={train_mae_m:.4f} m, val_MAE={val_mae_m:.4f} m"
        )

        print(
            f"                "
            f"train_MSE_norm={train_mse_norm:.4f}, "
            f"train_MAE_norm={train_mae_norm:.4f}, "
            f"val_MSE_norm={val_mse_norm:.4f}, "
            f"val_MAE_norm={val_mae_norm:.4f}"
        )

        # ----------------------------------------------------
        # Periodic save
        # ----------------------------------------------------
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            save_path = os.path.join(
                save_dir,
                f"ep{epoch + 1:03d}-loss{train_loss:.3f}-val{val_loss:.3f}.pth",
            )
            _save_model_weights(model, save_path)

        # ----------------------------------------------------
        # Best model save
        # ----------------------------------------------------
        is_best = False

        if loss_history is None:
            is_best = True
        elif len(loss_history.val_loss) <= 1:
            is_best = True
        else:
            previous_best = min(loss_history.val_loss[:-1])
            if val_loss <= previous_best:
                is_best = True

        if is_best:
            print("Save best model to best_epoch_weights.pth")

            best_latest_path = os.path.join(save_dir, "best_epoch_weights.pth")
            _save_model_weights(model, best_latest_path)

            best_hist_path = os.path.join(
                save_dir,
                f"best_epoch_{epoch + 1:03d}_valloss_{val_loss:.4f}_valmae_{val_mae_m:.4f}.pth",
            )
            _save_model_weights(model, best_hist_path)

            print(f"Historical best model saved as: {os.path.basename(best_hist_path)}")

        # ----------------------------------------------------
        # Last model save
        # ----------------------------------------------------
        last_path = os.path.join(save_dir, "last_epoch_weights.pth")
        _save_model_weights(model, last_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()