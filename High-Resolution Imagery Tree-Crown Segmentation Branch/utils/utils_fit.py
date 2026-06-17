# utils/utils_fit.py
import os
import torch
import torch.nn.functional as F
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def _auto_detect_fg_value_from_mask(pngs_cpu: torch.Tensor, ignore_raw=None):
    """
    pngs_cpu: [B,H,W] on CPU
    返回：fg_raw（int 或 None）
    规则：统计非0、非ignore 的像元值，取出现最多的那个作为前景值
    """
    uniq, cnt = torch.unique(pngs_cpu, return_counts=True)
    uniq = uniq.tolist()
    cnt = cnt.tolist()

    pairs = []
    for u, c in zip(uniq, cnt):
        u = int(u)
        if u == 0:
            continue
        if ignore_raw is not None and u == int(ignore_raw):
            continue
        pairs.append((u, int(c)))

    if not pairs:
        return None

    pairs.sort(key=lambda x: x[1], reverse=True)
    return int(pairs[0][0])


def _remap_pngs_and_build_onehot(
    pngs: torch.Tensor,
    num_classes: int,
    fg_raw: int = None,
    ignore_raw: int = None
):
    """
    把原始 mask 像元值 -> 训练用类别索引：
    - 背景: 0 -> 0
    - 前景: fg_raw -> 1
    - ignore_raw 或其它值 -> num_classes (作为 ignore_index)
    同时构建 one-hot labels: [B,H,W,num_classes+1]
    """
    if fg_raw is None:
        # 没前景：全部背景（极端情况）
        png_idx = torch.zeros_like(pngs, dtype=torch.long)
    else:
        fg_raw = int(fg_raw)
        png_idx = torch.full_like(pngs, fill_value=num_classes, dtype=torch.long)  # 默认 ignore
        png_idx = torch.where(pngs == 0, torch.zeros_like(png_idx), png_idx)       # 背景=0
        png_idx = torch.where(pngs == fg_raw, torch.ones_like(png_idx), png_idx)   # 前景=1

        if ignore_raw is not None:
            png_idx = torch.where(
                pngs == int(ignore_raw),
                torch.full_like(png_idx, num_classes),
                png_idx
            )

    labels = F.one_hot(png_idx.clamp(0, num_classes), num_classes=num_classes + 1).float()
    return png_idx, labels


def fit_one_epoch(
    model_train, model, loss_history, eval_callback, optimizer, epoch,
    epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda,
    dice_loss, focal_loss, cls_weights, num_classes,
    fp16, scaler, save_period, save_dir, local_rank=0,
    ignore_raw_value=None,
    fg_raw_value=None,          # ✅ 新增：外部传入前景原始像元值（推荐来自 train.py 的 fg_value）
):
    """
    ✅ 推荐策略：
    - train.py 先全局统计 fg_raw_value（例如38）
    - fit 内部固定使用 fg_raw_value remap，保证训练/验证一致
    - 若 fg_raw_value=None，才 fallback 用 batch 自动识别（兼容旧逻辑）
    """
    total_loss = 0.0
    total_f_score = 0.0
    val_loss = 0.0
    val_f_score = 0.0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # weights
    weights = torch.from_numpy(cls_weights)
    if cuda:
        weights = weights.cuda(local_rank, non_blocking=True)

    # ✅ 本 epoch 固定 fg_raw（优先用外部传入）
    fg_raw_used = fg_raw_value

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, _ = batch
        with torch.no_grad():
            if cuda:
                imgs = imgs.cuda(local_rank, non_blocking=True)
                pngs = pngs.cuda(local_rank, non_blocking=True)

        # fallback：如果没传 fg_raw_value 才自动识别（只做一次）
        if fg_raw_used is None:
            with torch.no_grad():
                fg_raw_used = _auto_detect_fg_value_from_mask(
                    pngs.detach().cpu(), ignore_raw=ignore_raw_value
                )
            if local_rank == 0:
                print(f"[Train] Auto-detect fg_raw_value = {fg_raw_used} (ignore_raw={ignore_raw_value})")

        pngs_idx, labels = _remap_pngs_and_build_onehot(
            pngs, num_classes=num_classes, fg_raw=fg_raw_used, ignore_raw=ignore_raw_value
        )
        if cuda:
            labels = labels.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs_idx, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs_idx, weights, num_classes=num_classes)

            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

        else:
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs_idx, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs_idx, weights, num_classes=num_classes)

                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # DEBUG（前两步）
        if local_rank == 0 and epoch == 0 and iteration < 2:
            with torch.no_grad():
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_fore = (pred_argmax != 0).sum().item()
                gt_fore = (pngs_idx == 1).sum().item()
                gt_ign = (pngs_idx == num_classes).sum().item()
                print(f"[Train-DEBUG] step {iteration+1}: pred_fore={pred_fore}, gt_fore={gt_fore}, gt_ignore={gt_ign}, fg_raw={fg_raw_used}")

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------- #
    # Validation
    # ---------------------- #
    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break

            imgs, pngs, _ = batch
            if cuda:
                imgs = imgs.cuda(local_rank, non_blocking=True)
                pngs = pngs.cuda(local_rank, non_blocking=True)

            # ✅ val 必须用同一个 fg_raw_used
            if fg_raw_used is None:
                fg_raw_used = _auto_detect_fg_value_from_mask(
                    pngs.detach().cpu(), ignore_raw=ignore_raw_value
                )
                if local_rank == 0:
                    print(f"[Val] Auto-detect fg_raw_value = {fg_raw_used} (ignore_raw={ignore_raw_value})")

            pngs_idx, labels = _remap_pngs_and_build_onehot(
                pngs, num_classes=num_classes, fg_raw=fg_raw_used, ignore_raw=ignore_raw_value
            )
            if cuda:
                labels = labels.cuda(local_rank, non_blocking=True)

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs_idx, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs_idx, weights, num_classes=num_classes)

            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)

            _f_score = f_score(outputs, labels)

            if local_rank == 0 and iteration < 2 and epoch == 0:
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_fore = (pred_argmax != 0).sum().item()
                gt_fore = (pngs_idx == 1).sum().item()
                gt_ign = (pngs_idx == num_classes).sum().item()
                print(f"[Val-DEBUG] step {iteration+1}: pred_fore={pred_fore}, gt_fore={gt_fore}, gt_ignore={gt_ign}, fg_raw={fg_raw_used}")

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{
                    'val_loss': val_loss / (iteration + 1),
                    'f_score': val_f_score / (iteration + 1),
                    'lr': get_lr(optimizer)
                })
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch, model_train)

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            save_path = os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
            torch.save(model.state_dict(), save_path)

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(
    model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda,
    dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir,
    local_rank=0, ignore_raw_value=None, fg_raw_value=None
):
    total_loss = 0.0
    total_f_score = 0.0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    weights = torch.from_numpy(cls_weights)
    if cuda:
        weights = weights.cuda(local_rank, non_blocking=True)

    fg_raw_used = fg_raw_value

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, _ = batch
        with torch.no_grad():
            if cuda:
                imgs = imgs.cuda(local_rank, non_blocking=True)
                pngs = pngs.cuda(local_rank, non_blocking=True)

        if fg_raw_used is None:
            fg_raw_used = _auto_detect_fg_value_from_mask(
                pngs.detach().cpu(), ignore_raw=ignore_raw_value
            )
            if local_rank == 0:
                print(f"[Train(no-val)] Auto-detect fg_raw_value = {fg_raw_used} (ignore_raw={ignore_raw_value})")

        pngs_idx, labels = _remap_pngs_and_build_onehot(
            pngs, num_classes=num_classes, fg_raw=fg_raw_used, ignore_raw=ignore_raw_value
        )
        if cuda:
            labels = labels.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs_idx, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs_idx, weights, num_classes=num_classes)

            if dice_loss:
                loss = loss + Dice_loss(outputs, labels)

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs_idx, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs_idx, weights, num_classes=num_classes)

                if dice_loss:
                    loss = loss + Dice_loss(outputs, labels)

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            save_path = os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step))
            torch.save(model.state_dict(), save_path)

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
