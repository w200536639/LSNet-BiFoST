import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from nets.unet_training import MSE_Loss, MAE_Loss, Combined_Reg_Loss
from utils.utils import get_lr


def _calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """逐棵树 R²（和你验证脚本一致）"""
    if y_true is None or y_pred is None:
        return np.nan
    if len(y_true) < 2:
        return np.nan
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    y_mean = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _extract_tree_max_pairs(pred_raw: np.ndarray,
                            true_raw: np.ndarray,
                            mask_np: np.ndarray,
                            min_pixels: int = 5):
    """
    输入单张图的 pred/true/mask (H,W)，输出该图所有树的 (true_max, pred_max)。
    - valid: mask==1 且 true>0
    - connected components: 8连通
    """
    pred_raw = pred_raw.astype(np.float32)
    true_raw = true_raw.astype(np.float32)
    mask_np  = mask_np.astype(np.uint8)

    valid = (mask_np > 0) & (true_raw > 0.0)
    if not np.any(valid):
        return [], []

    # 连通域标记（每个连通域当作一棵树）
    num_labels, labels = cv2.connectedComponents(valid.astype(np.uint8), connectivity=8)

    t_list, p_list = [], []
    for lab in range(1, num_labels):
        region = (labels == lab)
        if region.sum() < min_pixels:
            continue

        region_true = true_raw[region]
        region_pred = pred_raw[region]

        # 取 max（等价于你验证脚本的 canopy max）
        max_true = float(np.max(region_true))
        max_pred = float(np.max(region_pred))

        if max_true <= 0.0:
            continue

        t_list.append(max_true)
        p_list.append(max_pred)

    return t_list, p_list


def fit_one_epoch_reg(model_train, model, loss_history, optimizer, epoch,
                      epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,
                      loss_type, loss_alpha, fp16, scaler, save_period, save_dir,
                      local_rank=0, height_min=0.0, height_max=6.0):
    """
    单个 epoch 的训练/验证循环（树高回归）
    ✅ best 逻辑：用“逐棵树 max 高度”的 R² 最大来选最优模型（替代 val_loss 最小）
    """
    scale_range = height_max - height_min if (height_max - height_min) != 0 else 1.0

    def calc_loss(outputs_norm, heights_norm, masks):
        if loss_type == "mse":
            return MSE_Loss(outputs_norm, heights_norm, masks)
        elif loss_type == "mae":
            return MAE_Loss(outputs_norm, heights_norm, masks)
        elif loss_type == "combined":
            return Combined_Reg_Loss(outputs_norm, heights_norm, masks, alpha=loss_alpha)
        else:
            raise ValueError(f"不支持的损失类型：{loss_type}")

    # ---------------- 训练统计 ----------------
    total_loss = total_mse_norm = total_mae_norm = total_mse_raw = total_mae_raw = 0.0

    if local_rank == 0:
        print('开始训练...')
        train_pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch} - train', mininterval=0.3, ncols=100)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            images, heights_norm, masks, _ = batch
        else:
            images, heights_norm, masks = batch

        if cuda:
            images       = images.cuda(local_rank, non_blocking=True)
            heights_norm = heights_norm.cuda(local_rank, non_blocking=True)
            masks        = masks.cuda(local_rank, non_blocking=True)

        if not torch.all((masks == 0) | (masks == 1)):
            masks = (masks > 0.5).float()
            if local_rank == 0:
                print(f"警告：训练批次{iteration}的掩码非二值，已自动二值化")

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=fp16):
            outputs_norm = model_train(images)
            loss = calc_loss(outputs_norm, heights_norm, masks)

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

        total_loss += loss.item()
        total_mse_norm += mse_norm.item()
        total_mae_norm += mae_norm.item()
        total_mse_raw += mse_raw.item()
        total_mae_raw += mae_raw.item()

        if local_rank == 0:
            train_pbar.set_postfix({
                'Loss(norm)': f"{total_loss / (iteration + 1):.4f}",
                'MAE(m)': f"{total_mae_raw / (iteration + 1):.4f}",
                'lr': f"{get_lr(optimizer):.6f}"
            })
            train_pbar.update(1)

    if local_rank == 0:
        train_pbar.close()
        print('训练结束')
        print('开始验证...')
        val_pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch} - val', mininterval=0.3, ncols=100)

    # ---------------- 验证统计（像素级 + 逐棵树级）----------------
    val_total_loss = val_total_mse_norm = val_total_mae_norm = val_total_mse_raw = val_total_mae_raw = 0.0

    # 逐棵树统计容器：收集所有树的 true_max/pred_max
    all_tree_true = []
    all_tree_pred = []

    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            if isinstance(batch, (list, tuple)) and len(batch) == 4:
                images, heights_norm, masks, _ = batch
            else:
                images, heights_norm, masks = batch

            if cuda:
                images       = images.cuda(local_rank, non_blocking=True)
                heights_norm = heights_norm.cuda(local_rank, non_blocking=True)
                masks        = masks.cuda(local_rank, non_blocking=True)

            if not torch.all((masks == 0) | (masks == 1)):
                masks = (masks > 0.5).float()
                if local_rank == 0:
                    print(f"警告：验证批次{iteration}的掩码非二值，已自动二值化")

            outputs_norm = model_train(images)

            loss = calc_loss(outputs_norm, heights_norm, masks)
            mse_norm = MSE_Loss(outputs_norm, heights_norm, masks)
            mae_norm = MAE_Loss(outputs_norm, heights_norm, masks)

            outputs_raw = outputs_norm * scale_range + height_min
            heights_raw = heights_norm * scale_range + height_min
            mse_raw = MSE_Loss(outputs_raw, heights_raw, masks)
            mae_raw = MAE_Loss(outputs_raw, heights_raw, masks)

            val_total_loss += loss.item()
            val_total_mse_norm += mse_norm.item()
            val_total_mae_norm += mae_norm.item()
            val_total_mse_raw += mse_raw.item()
            val_total_mae_raw += mae_raw.item()

            # ===== 逐棵树 max 提取（CPU numpy）=====
            # shapes: [B,1,H,W]
            pred_np = outputs_raw.detach().float().cpu().numpy()
            true_np = heights_raw.detach().float().cpu().numpy()
            mask_np = masks.detach().float().cpu().numpy()

            B = pred_np.shape[0]
            for bi in range(B):
                t_list, p_list = _extract_tree_max_pairs(
                    pred_np[bi, 0], true_np[bi, 0], mask_np[bi, 0]
                )
                if len(t_list) > 0:
                    all_tree_true.extend(t_list)
                    all_tree_pred.extend(p_list)

            if local_rank == 0:
                val_pbar.set_postfix({
                    'ValLoss(norm)': f"{val_total_loss / (iteration + 1):.4f}",
                    'ValMAE(m)': f"{val_total_mae_raw / (iteration + 1):.4f}"
                })
                val_pbar.update(1)

    if local_rank == 0:
        val_pbar.close()
        print('验证结束')

        # -------- epoch 平均像素级指标 --------
        train_loss = total_loss / max(1, epoch_step)
        val_loss   = val_total_loss / max(1, epoch_step_val)
        train_mae_m = total_mae_raw / max(1, epoch_step)
        val_mae_m   = val_total_mae_raw / max(1, epoch_step_val)

        # -------- 逐棵树指标（R²/MAE/RMSE/Bias）--------
        if len(all_tree_true) >= 2:
            t = np.array(all_tree_true, dtype=np.float32)
            p = np.array(all_tree_pred, dtype=np.float32)
            err = p - t
            val_tree_mae  = float(np.mean(np.abs(err)))
            val_tree_rmse = float(np.sqrt(np.mean(err ** 2)))
            val_tree_bias = float(np.mean(err))
            val_tree_r2   = float(_calc_r2(t, p))
            tree_cnt      = int(len(t))
        else:
            val_tree_mae = np.nan
            val_tree_rmse = np.nan
            val_tree_bias = np.nan
            val_tree_r2 = np.nan
            tree_cnt = 0

        # 记录到 LossHistory（不改 LossHistory 也能用：动态挂属性）
        loss_history.append_loss(epoch + 1, train_loss, val_loss)
        if not hasattr(loss_history, "val_tree_r2"):
            loss_history.val_tree_r2 = []
        loss_history.val_tree_r2.append(val_tree_r2)

        print(f'Epoch: {epoch + 1}/{Epoch}')
        print(f'训练损失(归一化): {train_loss:.4f} | 训练MAE(米): {train_mae_m:.4f}')
        print(f'验证损失(归一化): {val_loss:.4f} | 验证MAE(米): {val_mae_m:.4f}')
        print(f'[逐棵树指标] trees={tree_cnt} | R2={val_tree_r2:.4f} | RMSE={val_tree_rmse:.4f} | MAE={val_tree_mae:.4f} | Bias={val_tree_bias:.4f}')

        # ================== 周期性保存（不变） ==================
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f'ep{epoch+1:03d}-loss{train_loss:.3f}-val{val_loss:.3f}-r2{val_tree_r2:.4f}.pth')
            )

        # ================== ✅ 最佳模型保存：按 val_tree_r2 最大 ==================
        # 初始化 best_r2
        if not hasattr(loss_history, "best_tree_r2"):
            loss_history.best_tree_r2 = -np.inf

        # 只有当 R2 可用时才更新 best
        if not np.isnan(val_tree_r2) and val_tree_r2 >= loss_history.best_tree_r2:
            loss_history.best_tree_r2 = val_tree_r2
            print(f'保存最佳模型到 best_epoch_weights.pth（按逐棵树R²） best_r2={val_tree_r2:.4f}')

            best_latest_path = os.path.join(save_dir, "best_epoch_weights.pth")
            torch.save(model.state_dict(), best_latest_path)

            best_hist_path = os.path.join(
                save_dir,
                f"best_epoch_{epoch+1:03d}_treeR2_{val_tree_r2:.4f}_treeRMSE_{val_tree_rmse:.4f}.pth"
            )
            torch.save(model.state_dict(), best_hist_path)
            print(f"另存历史最佳模型为: {os.path.basename(best_hist_path)}")

        # 始终保存 last_epoch（断点续练）
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

        # 可选：写一份当前epoch的树级指标
        with open(os.path.join(save_dir, "val_tree_metrics.txt"), "a", encoding="utf-8") as f:
            f.write(
                f"epoch={epoch+1:03d} trees={tree_cnt} "
                f"treeR2={val_tree_r2:.6f} treeRMSE={val_tree_rmse:.6f} "
                f"treeMAE={val_tree_mae:.6f} treeBias={val_tree_bias:.6f} "
                f"valLoss={val_loss:.6f}\n"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
