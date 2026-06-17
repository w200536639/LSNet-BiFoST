from utils.utils import preprocess_input
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from skimage.measure import label, regionprops
from skimage.filters import sobel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from collections import Counter


# -----------------------------
# 训练损失记录（原样保留）
# -----------------------------
class LossHistory:
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag
        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except Exception:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        os.makedirs(self.log_dir, exist_ok=True)
        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a") as f:
            f.write(str(loss) + "\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a") as f:
                f.write(str(val_loss) + "\n")
        self.writer.add_scalar("loss", loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar("val_loss", val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        import scipy.signal as signal
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')
        try:
            num = 5 if len(self.losses) < 25 else 15
            plt.plot(iters, signal.savgol_filter(self.losses, num, 3),
                     'green', linestyle='--', linewidth=2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, signal.savgol_filter(self.val_loss, num, 3),
                         '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except Exception:
            pass
        plt.grid(True)
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla(); plt.close("all")


# -----------------------------
# 工具：读取 mask 为类别索引数组（兼容 P / L）
# -----------------------------
def _read_mask_array(mask_path):
    m = Image.open(mask_path)
    if m.mode == 'P':
        arr = np.array(m, dtype=np.uint8)
    else:
        arr = np.array(m.convert('L'), dtype=np.uint8)
    return arr, m.mode

def _resize_mask_preserve_index(mask_pil, size_wh):
    if mask_pil.mode == 'P':
        return np.array(mask_pil.resize(size_wh, Image.NEAREST), dtype=np.uint8)
    else:
        return np.array(mask_pil.convert('L').resize(size_wh, Image.NEAREST), dtype=np.uint8)


# -----------------------------
# 实例级指标（带过滤与DEBUG）
# -----------------------------
def _filter_pred_instances(pred_bin, area_thr, per_thr, circ_thr, debug=False):
    labeled = label(pred_bin)
    props = regionprops(labeled)
    filtered = np.zeros_like(pred_bin)
    for prop in props:
        area = prop.area
        per  = prop.perimeter
        circ = (4 * np.pi * area) / (per ** 2) if per > 0 else 0
        if (area >= area_thr) and (per >= per_thr) and (circ >= circ_thr):
            filtered[labeled == prop.label] = 1
    num_before = int(labeled.max())
    num_after  = int(label(filtered).max())
    if debug:
        print(f"    - 预测连通域：过滤前 {num_before}，过滤后 {num_after} "
              f"(area>={area_thr}, per>={per_thr}, circ>={circ_thr})")
    return filtered, num_before, num_after


def _greedy_one_to_one_match_iou(pred_labels, gt_labels, iou_thr=0.5):
    """
    对连通域标签图进行 IoU 贪心一对一匹配。
    返回 (tp, fp, fn)；保证 tp <= min(#pred, #gt) 且 fp, fn >= 0。
    """
    P = int(pred_labels.max())
    G = int(gt_labels.max())

    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    # 面积（像素数）
    pred_areas = np.bincount(pred_labels.ravel(), minlength=P+1)[1:]  # drop background
    gt_areas   = np.bincount(gt_labels.ravel(),   minlength=G+1)[1:]

    # 仅统计 pred>0 且 gt>0 的交集像素，并构造交集矩阵 (P x G)
    pr = pred_labels.ravel()
    gt = gt_labels.ravel()
    mask = (pr > 0) & (gt > 0)
    if mask.any():
        pp = pr[mask] - 1
        gg = gt[mask] - 1
        k = pp * G + gg
        inter = np.bincount(k, minlength=P * G).reshape(P, G)
    else:
        inter = np.zeros((P, G), dtype=np.int64)

    # IoU
    union = pred_areas[:, None] + gt_areas[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / np.maximum(1, union)

    # 贪心：按 IoU 从大到小选择不冲突的配对
    pairs = np.argwhere(iou >= iou_thr)
    tp = 0
    if pairs.size > 0:
        order = np.argsort(iou[pairs[:, 0], pairs[:, 1]])[::-1]
        used_p = np.zeros(P, dtype=bool)
        used_g = np.zeros(G, dtype=bool)
        for k in order:
            p, g = pairs[k]
            if not used_p[p] and not used_g[g]:
                used_p[p] = True
                used_g[g] = True
                tp += 1

    fp = P - tp
    fn = G - tp
    return int(tp), int(max(0, fp)), int(max(0, fn))


def _instance_metrics(pred_mask, gt_mask, target_label=38,
                      area_thr=0, per_thr=0, circ_thr=0.0,
                      iou_thr=0.5,
                      debug=False):
    """
    将实例级 TP/FP/FN 计算改为：IoU>=iou_thr 的一对一贪心匹配；
    保留原有过滤与 DEBUG 打印。
    """
    dbg = {}

    # 1) GT（仅 target_label）
    gt_bin = (gt_mask == target_label).astype(np.uint8)
    dbg["gt_unique"] = np.unique(gt_mask).tolist()
    dbg["gt_target_pixels"] = int(gt_bin.sum())
    if debug:
        print(f"    - GT 唯一值: {dbg['gt_unique']}, 目标像素数(=={target_label}): {dbg['gt_target_pixels']}")

    # 2) 预测：前景=非0
    pred_bin = (pred_mask != 0).astype(np.uint8)
    dbg["pred_fore_pixels"] = int(pred_bin.sum())
    if debug:
        print(f"    - 预测前景像素数: {dbg['pred_fore_pixels']}")

    # 3) 过滤预测实例
    filtered_pred, n_before, n_after = _filter_pred_instances(
        pred_bin, area_thr, per_thr, circ_thr, debug=debug
    )
    dbg["pred_cc_before"] = n_before
    dbg["pred_cc_after"]  = n_after

    if (n_after == 0) and (dbg["pred_fore_pixels"] > 0) and (area_thr>0 or per_thr>0 or circ_thr>0):
        if debug:
            print("    🔁 过滤后实例=0，但前景像素>0，自动降级为 '不过滤' 再评估一次。")
        filtered_pred = pred_bin
        n_after = int(label(filtered_pred).max())
        dbg["pred_cc_after"] = n_after

    # 4) 连通域标签
    pred_labels = label(filtered_pred, connectivity=2)
    gt_labels   = label(gt_bin,       connectivity=2)
    num_pred    = int(pred_labels.max())
    num_gt      = int(gt_labels.max())
    dbg["num_pred_instances"] = num_pred
    dbg["num_gt_instances"]   = num_gt

    # 5) 一对一 IoU 匹配
    tp, fp, fn = _greedy_one_to_one_match_iou(pred_labels, gt_labels, iou_thr=iou_thr)
    dbg["tp_fp_fn"] = (tp, fp, fn)
    dbg["all_bg"] = (dbg["pred_fore_pixels"] == 0)

    if debug:
        print(f"    - IoU阈值: {iou_thr}, 匹配后 TP={tp}, FP={fp}, FN={fn}")

    return tp, fp, fn, dbg


# -----------------------------
# 验证回调（实例级 F1）
# -----------------------------
class EvalCallback:
    def __init__(self,
                 model,
                 input_shape,
                 num_classes,
                 val_lines,
                 VOCdevkit_path,
                 log_dir,
                 Cuda,
                 eval_flag=True,
                 period=1,
                 target_label_value=None,
                 area_thr=0,
                 perim_thr=0,
                 circ_thr=0.0,
                 save_visual_topk=0,
                 debug_first_n=0,
                 autodetect_scan_max=200,
                 iou_thr=0.5):
        """
        target_label_value=None 时，将自动扫描验证集前 autodetect_scan_max 张标注，
        统计非 0 像素值出现次数，取出现次数最多者作为前景标签。
        """
        self.model       = model
        self.input_shape = input_shape       # (H, W)
        self.num_classes = num_classes
        self.val_lines   = val_lines
        self.voc_root    = VOCdevkit_path
        self.log_dir     = log_dir
        self.cuda        = Cuda
        self.eval_flag   = eval_flag
        self.period      = period

        self.area_thr  = area_thr
        self.perim_thr = perim_thr
        self.circ_thr  = circ_thr
        self.iou_thr   = iou_thr

        self.best_f1   = 0.0
        self.metrics_log = os.path.join(self.log_dir, "best_f1_summary.txt")

        self.save_visual_topk = int(save_visual_topk)
        self.vis_dir = os.path.join(self.log_dir, "vis")
        if self.save_visual_topk > 0:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.debug_first_n     = int(debug_first_n)
        self.autodetect_scan_max = int(autodetect_scan_max)

        # 自动识别前景标签
        if target_label_value is None:
            self.target_label_value = self._auto_detect_target_label()
        else:
            self.target_label_value = int(target_label_value)
            print(f"[EvalCallback] 使用指定 target_label_value = {self.target_label_value}")

    def _auto_detect_target_label(self):
        ids = [line.strip() for line in self.val_lines]
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")
        counts = Counter()
        scanned = 0

        for name in ids[:self.autodetect_scan_max]:
            mask_p = os.path.join(mask_dir, f"{name}.png")
            if not os.path.exists(mask_p):
                continue
            arr, mode = _read_mask_array(mask_p)
            uniq, freq = np.unique(arr, return_counts=True)
            for u, c in zip(uniq, freq):
                if u != 0:
                    counts[int(u)] += int(c)
            scanned += 1

        if scanned == 0:
            print("[EvalCallback] ⚠️ 自动识别失败：未能扫描到有效标注文件，回退 target_label_value=1")
            return 1

        if not counts:
            print("[EvalCallback] ⚠️ 自动识别：非 0 像素不存在，怀疑你的标注只有背景；回退 target_label_value=1")
            return 1

        target = counts.most_common(1)[0][0]
        print(f"[EvalCallback] 自动识别 target_label_value = {target}（统计自前 {scanned} 张标注）")
        topk = counts.most_common(5)
        print(f"[EvalCallback] 非 0 像素 Top 值及计数（最多 5 个）：{topk}")
        return int(target)

    def _letterbox(self, img_pil):
        iw, ih = img_pil.size
        W = self.input_shape[1]
        H = self.input_shape[0]
        scale = min(W / iw, H / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = img_pil.resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (W, H), (128, 128, 128))
        cx, cy = (W - nw) // 2, (H - nh) // 2
        canvas.paste(resized, (cx, cy))
        return canvas, (cx, cy, nw, nh)

    def _visualize(self, original_np, pred_bin, gt_bin, save_path, tp, fp, fn):
        edges = sobel(pred_bin) > 0
        gt_color = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
        gt_color[gt_bin == 1] = [255, 215, 0]
        pred_color = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
        pred_color[pred_bin == 1] = [135, 206, 250]

        alpha = 0.2
        blended = original_np.astype(np.float32).copy()
        blended[pred_bin == 1] = blended[pred_bin == 1] * (1 - alpha) + pred_color[pred_bin == 1] * alpha
        blended[gt_bin == 1]   = blended[gt_bin == 1] * (1 - alpha) + gt_color[gt_bin == 1]   * alpha
        blended[edges] = [255, 0, 0]
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(blended)
        plt.title(f"TP={tp}, FP={fp}, FN={fn}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    def on_epoch_end(self, epoch, model_eval=None):
        if (not self.eval_flag) or ((epoch + 1) % self.period != 0):
            return

        print(f"\n[EvalCallback] 🚀 开始验证 Epoch {epoch + 1}（target_label_value={self.target_label_value}）")
        model = self.model if model_eval is None else model_eval
        model.eval()

        ids = [line.strip() for line in self.val_lines]
        img_dir  = os.path.join(self.voc_root, "VOC2007/JPEGImages")
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")

        tp_sum = fp_sum = fn_sum = 0
        saved_vis = 0

        H, W = self.input_shape[0], self.input_shape[1]

        with torch.inference_mode():
            for idx, name in enumerate(tqdm(ids, desc="Validating", ncols=100)):
                img_p  = os.path.join(img_dir,  f"{name}.jpg")
                mask_p = os.path.join(mask_dir, f"{name}.png")
                if (not os.path.exists(img_p)) or (not os.path.exists(mask_p)):
                    continue

                img  = Image.open(img_p).convert("RGB")
                mask_pil = Image.open(mask_p)  # 可能是 P 或 L

                # letterbox 到 (H, W)
                canvas, (cx, cy, nw, nh) = self._letterbox(img)

                # 归一化与训练一致
                img_np = np.array(canvas, dtype=np.float32)
                img_np = preprocess_input(img_np)

                x = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
                if self.cuda:
                    x = x.cuda(non_blocking=True)

                # 模型输出整幅画布
                out = model(x)
                pred_full = torch.softmax(out, dim=1).argmax(dim=1).cpu().numpy()[0]  # (H, W)

                # 构造整幅 GT 画布并贴中间
                gt_resized = _resize_mask_preserve_index(mask_pil, (nw, nh))          # (nh, nw)
                gt_full = np.zeros((H, W), dtype=np.uint8)
                gt_full[cy:cy+nh, cx:cx+nw] = gt_resized

                # 在整幅画布上评估
                pred = pred_full
                gt   = gt_full

                # 可选 DEBUG：对比整幅与裁剪后前景像素（只看前几张）
                debug_this = (idx < self.debug_first_n)
                if debug_this:
                    pred_crop = pred_full[cy:cy+nh, cx:cx+nw]
                    print(f"\n[DEBUG] 样本 {idx+1}: {name}")
                    print(f"    - pred_full_fore = {int((pred_full != 0).sum())}, "
                          f"pred_crop_fore = {int((pred_crop != 0).sum())}, "
                          f"gt_crop_fore = {int((gt_resized == self.target_label_value).sum())}")

                tp, fp, fn, dbg = _instance_metrics(
                    pred, gt,
                    target_label=self.target_label_value,
                    area_thr=self.area_thr,
                    per_thr=self.perim_thr,
                    circ_thr=self.circ_thr,
                    iou_thr=self.iou_thr,
                    debug=debug_this
                )
                if debug_this and dbg["all_bg"]:
                    print("    ⚠️ 预测几乎全背景（前景像素数为 0）。")

                tp_sum += tp; fp_sum += fp; fn_sum += fn

                # 可视化（可选）
                if self.save_visual_topk > 0 and saved_vis < self.save_visual_topk:
                    original_np = np.array(canvas)  # 直接用整幅画布可视化
                    pred_bin = (pred != 0).astype(np.uint8)
                    gt_bin   = (gt == self.target_label_value).astype(np.uint8)
                    save_path = os.path.join(self.vis_dir, f"epoch_{epoch+1:03d}_{name}.png")
                    self._visualize(original_np, pred_bin, gt_bin, save_path, tp, fp, fn)
                    saved_vis += 1

        # 安全分母，避免 NaN/Inf
        precision = tp_sum / max(1, tp_sum + fp_sum)
        recall    = tp_sum / max(1, tp_sum + fn_sum)
        f1        = (2 * precision * recall) / max(1e-12, (precision + recall))

        print(f"📈 Epoch {epoch + 1} → TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

        with open(self.metrics_log, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch + 1}: TP={tp_sum}, FP={fp_sum}, FN={fn_sum}, "
                    f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")

        if f1 > self.best_f1:
            self.best_f1 = f1
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_path = os.path.join(self.log_dir, f"best_f1_epoch_{epoch + 1:03d}.pth")
            torch.save(state_dict, save_path)
            print(f"🎯 发现更优模型 (F1={f1:.4f})，已保存：{save_path}")

        model.train()

    def on_train_end(self):
        print("✅ 训练完成，EvalCallback 已结束。")
