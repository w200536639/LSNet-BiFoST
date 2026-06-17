import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# 树高回归核心损失函数（仅计算梭梭树冠区域）
# --------------------------
def _ensure_dims_and_type(pred, label, crown_mask):
    """
    确保 pred/label/mask 的维度与类型兼容：
      - 将 [B,H,W] -> [B,1,H,W]
      - 转为 float32（避免 uint8 / float16 在 isnan 或插值时报错）
    返回 (pred, label, crown_mask)
    """
    if crown_mask.dim() == 3:
        crown_mask = crown_mask.unsqueeze(1)
    if label.dim() == 3:
        label = label.unsqueeze(1)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)

    pred = pred.float()
    label = label.float()
    crown_mask = crown_mask.float()
    return pred, label, crown_mask


def MSE_Loss(pred, label, crown_mask, reduction='mean'):
    """
    per-sample MSE over valid pixels (tree crown & non-NaN label).
    reduction: 'mean' (default) -> mean over batch of per-sample means,
               'sum' -> global sum over batch.
    """
    pred, label, crown_mask = _ensure_dims_and_type(pred, label, crown_mask)

    # resize pred/mask to label size when needed
    n, c, h, w = pred.size()
    _, _, ht, wt = label.size()
    if h != ht or w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
        crown_mask = F.interpolate(crown_mask, size=(ht, wt), mode="nearest")
        crown_mask = (crown_mask > 0.5).float().detach()

    # valid mask: crown AND label not NaN
    label_non_nan_mask = (~torch.isnan(label)).float()
    valid_mask = crown_mask * label_non_nan_mask  # shape [B,1,H,W]

    # ensure valid_mask has at least one valid pixel per sample by clamp later
    label_valid = torch.nan_to_num(label, nan=0.0)

    sq_err = ((pred - label_valid) ** 2) * valid_mask  # zeros outside valid region

    if reduction == 'mean':
        # per-sample sum / per-sample valid count, then mean over batch
        valid_counts = valid_mask.view(n, -1).sum(dim=1)  # [B]
        valid_counts = torch.clamp(valid_counts, min=1.0)
        per_sample_mse = sq_err.view(n, -1).sum(dim=1) / valid_counts  # [B]
        return per_sample_mse.mean()
    elif reduction == 'sum':
        return sq_err.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def MAE_Loss(pred, label, crown_mask, reduction='mean'):
    """
    per-sample MAE over valid pixels (tree crown & non-NaN label).
    """
    pred, label, crown_mask = _ensure_dims_and_type(pred, label, crown_mask)

    n, c, h, w = pred.size()
    _, _, ht, wt = label.size()
    if h != ht or w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
        crown_mask = F.interpolate(crown_mask, size=(ht, wt), mode="nearest")
        crown_mask = (crown_mask > 0.5).float().detach()

    label_non_nan_mask = (~torch.isnan(label)).float()
    valid_mask = crown_mask * label_non_nan_mask

    label_valid = torch.nan_to_num(label, nan=0.0)
    abs_err = torch.abs(pred - label_valid) * valid_mask

    if reduction == 'mean':
        valid_counts = valid_mask.view(n, -1).sum(dim=1)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        per_sample_mae = abs_err.view(n, -1).sum(dim=1) / valid_counts
        return per_sample_mae.mean()
    elif reduction == 'sum':
        return abs_err.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class CombinedRegLoss(nn.Module):
    """
    Combined MSE + MAE with weight alpha (alpha * MSE + (1-alpha) * MAE).
    Both MSE and MAE follow per-sample mean logic.
    """
    def __init__(self, alpha=0.5):
        super(CombinedRegLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, label, crown_mask):
        mse = MSE_Loss(pred, label, crown_mask, reduction='mean')
        mae = MAE_Loss(pred, label, crown_mask, reduction='mean')
        return self.alpha * mse + (1 - self.alpha) * mae


# For backward compatibility with existing imports:
def Combined_Reg_Loss(pred, label, crown_mask, alpha=0.5):
    return CombinedRegLoss(alpha=alpha)(pred, label, crown_mask)


# --------------------------
# 训练辅助功能（保持不变，微调）
# --------------------------
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
        elif 'BatchNorm2d' in classname:
            try:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0.0)
            except Exception:
                pass

    print(f'initialize network with {init_type} type')
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters,
                     warmup_iters_ratio=0.05, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.05, step_num=10):
    """
    返回一个函数 func(iters) -> lr。
    注意：warmup_total_iters 以比例计算（避免极小 warmup）。
    """
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr_out = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr_out = min_lr
        else:
            lr_out = min_lr + 0.5 * (lr - min_lr) * (
                    1.0 + math.cos(
                math.pi * (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr_out

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = max(0, int(iters // step_size))
        out_lr = lr * (decay_rate ** n)
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = max(1, int(warmup_iters_ratio * total_iters))
        warmup_lr_start = max(int(warmup_lr_ratio * lr), 1e-6)
        no_aug_iter = max(1, int(no_aug_iter_ratio * total_iters))
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / max(1, (step_num - 1)))
        step_size = max(1, total_iters / step_num)
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
