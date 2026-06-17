import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                  gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes,
                  fp16, scaler, save_period, save_dir, local_rank=0):
    """
    说明：
    - DataLoader 已在 dataloader.py 中完成 letterbox + preprocess_input，
      因此此处直接 forward，不再做任何额外预处理。
    - 为了排查“验证回调中预测全背景”的现象，这里加入前几步 DEBUG 统计：
      统计 (argmax != 0) 的前景像素数量与 GT 的前景像素数量对比。
    """
    total_loss      = 0.0
    total_f_score   = 0.0

    val_loss        = 0.0
    val_f_score     = 0.0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels = batch  # imgs: [B,3,H,W]  pngs: [B,H,W]  labels: [B,H,W,num_classes+1 (onehot)]
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank, non_blocking=True)
                pngs    = pngs.cuda(local_rank, non_blocking=True)
                labels  = labels.cuda(local_rank, non_blocking=True)
                weights = weights.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ---------------------- #
        #   前向 + 损失
        # ---------------------- #
        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            # 使用新的 AMP API
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # ---------------------- #
        #   DEBUG：前几步检查前景像素
        # ---------------------- #
        if local_rank == 0 and epoch == 0 and iteration < 2:
            with torch.no_grad():
                pred_argmax = torch.argmax(outputs, dim=1)  # [B,H,W]
                pred_fore   = (pred_argmax != 0).sum().item()
                gt_fore     = (pngs != 0).sum().item()
                print(f"[Train-DEBUG] step {iteration+1}: pred_fore={pred_fore}, gt_fore={gt_fore}")

        total_loss    += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score'   : total_f_score / (iteration + 1),
                'lr'        : get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------- #
    #   验证
    # ---------------------- #
    model_train.eval()
    with torch.no_grad():
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            imgs, pngs, labels = batch
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank, non_blocking=True)
                pngs    = pngs.cuda(local_rank, non_blocking=True)
                labels  = labels.cuda(local_rank, non_blocking=True)
                weights = weights.cuda(local_rank, non_blocking=True)

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            _f_score = f_score(outputs, labels)

            # 验证阶段 DEBUG（仅前两步）
            if local_rank == 0 and iteration < 2 and epoch == 0:
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_fore   = (pred_argmax != 0).sum().item()
                gt_fore     = (pngs != 0).sum().item()
                print(f"[Val-DEBUG] step {iteration+1}: pred_fore={pred_fore}, gt_fore={gt_fore}")

            val_loss    += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{
                    'val_loss' : val_loss / (iteration + 1),
                    'f_score'  : val_f_score / (iteration + 1),
                    'lr'       : get_lr(optimizer)
                })
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # 记录 Loss
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

        # 调用实例级评估（callbacks.EvalCallback）
        eval_callback.on_epoch_end(epoch + 1, model_train)

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # ----------------------------------------------- #
        #   保存权值
        # ----------------------------------------------- #
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            save_path = os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val))
            torch.save(model.state_dict(), save_path)

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda,
                         dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir,
                         local_rank=0):
    """
    无验证集版本。接口保持与原实现一致。
    """
    total_loss      = 0.0
    total_f_score   = 0.0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank, non_blocking=True)
                pngs    = pngs.cuda(local_rank, non_blocking=True)
                labels  = labels.cuda(local_rank, non_blocking=True)
                weights = weights.cuda(local_rank, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast('cuda'):
                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # DEBUG：首两个 step 打印前景像素对比
        if local_rank == 0 and epoch == 0 and iteration < 2:
            with torch.no_grad():
                pred_argmax = torch.argmax(outputs, dim=1)
                pred_fore   = (pred_argmax != 0).sum().item()
                gt_fore     = (pngs != 0).sum().item()
                print(f"[Train(no-val)-DEBUG] step {iteration+1}: pred_fore={pred_fore}, gt_fore={gt_fore}")

        total_loss    += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'f_score'   : total_f_score / (iteration + 1),
                'lr'        : get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss / epoch_step)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        # ----------------------------------------------- #
        #   保存权值
        # ----------------------------------------------- #
        if (epoch + 1) % save_period == 0 or (epoch + 1) == Epoch:
            save_path = os.path.join(save_dir, 'ep%03d-loss%.3f.pth' % ((epoch + 1), total_loss / epoch_step))
            torch.save(model.state_dict(), save_path)

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
