# utils/utils_metrics.py
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import Counter


# ============================================================
# ✅ 安全读取 mask（关键：P 模式不要 convert('L')）
# - P: 直接得到索引值（例如 38）
# - L: 灰度
# ============================================================
def read_mask_u8(path):
    m = Image.open(path)
    if m.mode == "P":
        arr = np.array(m, dtype=np.uint8)
    else:
        arr = np.array(m.convert("L"), dtype=np.uint8)
    return arr


# ============================================================
# ✅ 自动识别前景像元值（例如 38）
# 规则：统计非0且非ignore的值，像元数最多者作为前景
# ============================================================
def auto_detect_target_label_value(gt_paths, scan_max=200, ignore_index=None):
    counts = Counter()
    scanned = 0
    for p in gt_paths[: min(scan_max, len(gt_paths))]:
        if not os.path.exists(p):
            continue
        arr = read_mask_u8(p)
        uniq, freq = np.unique(arr, return_counts=True)
        for u, c in zip(uniq, freq):
            u = int(u)
            if u == 0:
                continue
            if ignore_index is not None and u == int(ignore_index):
                continue
            counts[u] += int(c)
        scanned += 1

    if scanned == 0 or len(counts) == 0:
        return 1, counts, scanned
    target = counts.most_common(1)[0][0]
    return int(target), counts, scanned


# ============================================================
# ✅ GT 映射：原始像元值 -> 训练用类别索引
# 输出范围：
#   0..num_classes-1 为有效类别
#   -1 为 ignore（不参与统计）
# 默认按“二分类：背景0，前景=target_label_value->1”处理
# ============================================================
def map_gt_to_train_ids(label_raw, num_classes, target_label_value=None, ignore_index=None):
    label_raw = label_raw.astype(np.int32)

    # 情况A：GT 本来就是 0..C-1 的索引标签（VOC/常规语义分割）
    if label_raw.max() < num_classes and label_raw.min() >= 0:
        out = label_raw.copy()
        if ignore_index is not None:
            out[out == int(ignore_index)] = -1
        return out

    # 情况B：二分类 raw 值（0 / 38 / 255 ...）
    if num_classes == 2:
        out = np.zeros_like(label_raw, dtype=np.int32)

        # ignore
        if ignore_index is not None:
            out[label_raw == int(ignore_index)] = -1

        valid = np.ones_like(label_raw, dtype=bool)
        if ignore_index is not None:
            valid &= (label_raw != int(ignore_index))

        if target_label_value is None:
            # 非0（且非ignore）都当作前景
            out[(label_raw != 0) & valid] = 1
        else:
            # 仅 target 当前景，其它非0非target -> ignore
            out[(label_raw == int(target_label_value)) & valid] = 1
            other = (label_raw != 0) & (label_raw != int(target_label_value)) & valid
            out[other] = -1

        return out

    # 情况C：多类别 raw 值但不是 0..C-1（不常见），默认：除了背景0外都 ignore
    out = label_raw.copy()
    out[:] = -1
    if ignore_index is not None:
        out[label_raw == int(ignore_index)] = -1
    out[label_raw == 0] = 0
    return out


# ============================================================
# ✅ Pred 映射：读取预测图 -> 类别索引
# - 若 pred 里含有 ignore_id（比如 num_classes），可以映射为 -1
# ============================================================
def map_pred_to_train_ids(pred_raw, num_classes, pred_ignore_index=None):
    pred_raw = pred_raw.astype(np.int32)
    out = pred_raw.copy()

    if pred_ignore_index is not None:
        out[out == int(pred_ignore_index)] = -1

    out[(out < 0) | (out >= num_classes)] = -1
    return out


# ============================================================
# ✅ 正确的 f_score / Dice-Fβ：
# - 使用 argmax 做 hard prediction（而不是 per-channel threshold）
# - 默认不统计背景类（class 0），二分类只统计前景（class 1）
# - 支持 target 最后一通道为 ignore（one-hot 的 C+1）
# ============================================================
def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    """
    inputs: [N, C, H, W]
    target: [N, H, W, C+1] (最后一类为 ignore) 或 [N, H, W, C]
    返回：对 (1..C-1) 的 mean F_beta（排除背景0；二分类就是前景1）
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()

    # ✅ 修复：只要 H 或 W 不一致都要插值
    if (h != ht) or (w != wt):
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # prob: [N, HW, C]
    prob = torch.softmax(inputs.permute(0, 2, 3, 1).contiguous().view(n, -1, c), dim=-1)

    # target: [N, HW, ct]
    t = target.view(n, -1, ct)

    # valid mask：如果 ct==c+1，最后一通道为 ignore
    if ct == c + 1:
        valid = 1.0 - t[..., c]  # [N, HW]
        t_cls = t[..., :c]       # [N, HW, C]
    else:
        valid = torch.ones((n, t.size(1)), device=t.device, dtype=t.dtype)
        t_cls = t[..., :c]

    # pred hard: argmax -> onehot [N, HW, C]
    pred = torch.argmax(prob, dim=-1)  # [N, HW]
    pred_oh = F.one_hot(pred, num_classes=c).float()

    # mask valid
    v = valid.unsqueeze(-1)  # [N, HW, 1]
    pred_oh = pred_oh * v
    t_cls = t_cls * v

    # 只算非背景类：1..C-1
    if c <= 1:
        return torch.tensor(0.0, device=inputs.device)

    cls_ids = list(range(1, c))
    # tp/fp/fn: [C]
    tp = torch.sum(t_cls * pred_oh, dim=[0, 1])
    fp = torch.sum(pred_oh, dim=[0, 1]) - tp
    fn = torch.sum(t_cls, dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # 只取前景类（排除背景0）
    score_fg = score[cls_ids]
    return torch.mean(score_fg)


# ============================================================
# ✅ 混淆矩阵统计（支持 ignore = -1）
# a: GT index，b: pred index（均为 int32，ignore 为 -1）
# ============================================================
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n) & (b >= 0) & (b < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


# ============================================================
# ✅ mIoU 计算：支持 raw GT=0/38/(255)
# - target_label_value: 例如 38；None 表示自动识别
# - ignore_index: GT 忽略像元值（例如255）
# - pred_ignore_index: pred 里如果用 num_classes 表示 ignore（例如2），可传 num_classes
# ============================================================
def compute_mIoU(
    gt_dir,
    pred_dir,
    png_name_list,
    num_classes,
    name_classes=None,
    target_label_value=None,
    ignore_index=None,
    pred_ignore_index=None,
    autodetect_scan_max=200,
):
    print('Num classes', num_classes)

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    if num_classes == 2 and target_label_value is None:
        detected, counts, scanned = auto_detect_target_label_value(
            gt_imgs, scan_max=autodetect_scan_max, ignore_index=ignore_index
        )
        target_label_value = detected
        print(f"[mIoU] auto-detect target_label_value={target_label_value} (scanned={scanned})")
        if len(counts) > 0:
            print(f"[mIoU] non-zero top values: {counts.most_common(5)}")

    for ind in range(len(gt_imgs)):
        gt_path = gt_imgs[ind]
        pr_path = pred_imgs[ind]

        if (not os.path.exists(gt_path)) or (not os.path.exists(pr_path)):
            print(f"Skipping missing file: gt={gt_path}, pred={pr_path}")
            continue

        pred_raw = read_mask_u8(pr_path)
        label_raw = read_mask_u8(gt_path)

        if len(label_raw.flatten()) != len(pred_raw.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label_raw.flatten()), len(pred_raw.flatten()), gt_path, pr_path
                )
            )
            continue

        label = map_gt_to_train_ids(
            label_raw,
            num_classes=num_classes,
            target_label_value=target_label_value,
            ignore_index=ignore_index
        )
        pred = map_pred_to_train_ids(
            pred_raw,
            num_classes=num_classes,
            pred_ignore_index=pred_ignore_index
        )

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            ))

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] +
                  ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) +
                  '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) +
                  '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) +
          '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) +
          '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int64), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(
        IoUs, name_classes,
        "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100),
        "Intersection over Union",
        os.path.join(miou_out_path, "mIoU.png"),
        tick_font_size=tick_font_size, plt_show=True
    )
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(
        PA_Recall, name_classes,
        "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100),
        "Pixel Accuracy",
        os.path.join(miou_out_path, "mPA.png"),
        tick_font_size=tick_font_size, plt_show=False
    )
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(
        PA_Recall, name_classes,
        "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100),
        "Recall",
        os.path.join(miou_out_path, "Recall.png"),
        tick_font_size=tick_font_size, plt_show=False
    )
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(
        Precision, name_classes,
        "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
        "Precision",
        os.path.join(miou_out_path, "Precision.png"),
        tick_font_size=tick_font_size, plt_show=False
    )
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
