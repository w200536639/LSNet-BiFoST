# utils/callbacks.py
from utils.utils import preprocess_input
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
from skimage.measure import label, regionprops
from skimage.filters import sobel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from collections import Counter

# ✅ 多光谱 tif 建议用 rasterio
try:
    import rasterio
except Exception:
    rasterio = None


# -----------------------------
# 训练损失记录（兼容多通道）
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

        # ✅ 自动推断输入通道（兼容 DataParallel）
        in_ch = 3
        try:
            m = model.module if hasattr(model, "module") else model
            in_ch = int(getattr(m, "in_channels", 3))
        except Exception:
            pass

        try:
            dummy_input = torch.randn(2, in_ch, input_shape[0], input_shape[1])
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
    P = int(pred_labels.max())
    G = int(gt_labels.max())

    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    pred_areas = np.bincount(pred_labels.ravel(), minlength=P+1)[1:]
    gt_areas   = np.bincount(gt_labels.ravel(),   minlength=G+1)[1:]

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

    union = pred_areas[:, None] + gt_areas[None, :] - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / np.maximum(1, union)

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


def _instance_metrics(pred_mask,
                      gt_mask,
                      target_label=38,
                      area_thr=0,
                      per_thr=0,
                      circ_thr=0.0,
                      iou_thr=0.5,
                      ignore_index=None,
                      debug=False):
    """
    pred_mask: 模型输出的类别索引 (0/1/...)
    gt_mask  : 可能是原始像元值(0/38/255...)，也可能是已映射的训练id(0/1/2...)
    """
    dbg = {}

    gt_mask = np.array(gt_mask)
    pred_mask = np.array(pred_mask)

    if ignore_index is not None:
        ignore_index = int(ignore_index)
        valid = (gt_mask != ignore_index)
        dbg["ignore_index"] = ignore_index
        dbg["ignored_pixels"] = int((~valid).sum())
    else:
        valid = np.ones_like(gt_mask, dtype=bool)
        dbg["ignore_index"] = None
        dbg["ignored_pixels"] = 0

    dbg["gt_unique"] = np.unique(gt_mask).tolist()

    # -------------------------
    # ✅ 关键修复：GT 前景 label 自适应
    # -------------------------
    target_label = int(target_label)
    used_label = target_label

    gt_bin = ((gt_mask == used_label) & valid).astype(np.uint8)
    gt_pixels = int(gt_bin.sum())

    # 情况 A：你想用 38，但 GT 实际是 0/1（最常见的“评估为0”原因）
    if gt_pixels == 0 and used_label != 1:
        if 1 in dbg["gt_unique"]:
            used_label = 1
            gt_bin = ((gt_mask == used_label) & valid).astype(np.uint8)
            gt_pixels = int(gt_bin.sum())

    # 情况 B：既不是 38 也不是 1，但 valid 区域里确实有前景（比如别的label）
    if gt_pixels == 0:
        vals = gt_mask[valid]
        vals = vals[vals != 0]
        if ignore_index is not None:
            vals = vals[vals != ignore_index]
        if vals.size > 0:
            # 用出现最多的非零值当作前景（兜底）
            uniq, cnt = np.unique(vals, return_counts=True)
            used_label = int(uniq[np.argmax(cnt)])
            gt_bin = ((gt_mask == used_label) & valid).astype(np.uint8)
            gt_pixels = int(gt_bin.sum())

    dbg["gt_used_label"] = int(used_label)
    dbg["gt_target_pixels"] = int(gt_pixels)

    if debug:
        print(f"    - GT 唯一值: {dbg['gt_unique']}, ignore={dbg['ignore_index']}, "
              f"ignored_pixels={dbg['ignored_pixels']}, "
              f"目标像素数(=={dbg['gt_used_label']}): {dbg['gt_target_pixels']}")

    # pred 前景：只在 valid 范围内
    pred_bin = ((pred_mask != 0) & valid).astype(np.uint8)
    dbg["pred_fore_pixels"] = int(pred_bin.sum())
    if debug:
        print(f"    - 预测前景像素数(valid内): {dbg['pred_fore_pixels']}")

    filtered_pred, n_before, n_after = _filter_pred_instances(
        pred_bin, area_thr, per_thr, circ_thr, debug=debug
    )
    dbg["pred_cc_before"] = int(n_before)
    dbg["pred_cc_after"]  = int(n_after)

    if (n_after == 0) and (dbg["pred_fore_pixels"] > 0) and (area_thr > 0 or per_thr > 0 or circ_thr > 0):
        if debug:
            print("    🔁 过滤后实例=0，但前景像素>0，自动降级为 '不过滤' 再评估一次。")
        filtered_pred = pred_bin
        n_after = int(label(filtered_pred).max())
        dbg["pred_cc_after"] = int(n_after)

    pred_labels = label(filtered_pred, connectivity=2)
    gt_labels   = label(gt_bin,       connectivity=2)

    num_pred = int(pred_labels.max())
    num_gt   = int(gt_labels.max())
    dbg["num_pred_instances"] = num_pred
    dbg["num_gt_instances"]   = num_gt

    tp, fp, fn = _greedy_one_to_one_match_iou(pred_labels, gt_labels, iou_thr=iou_thr)
    dbg["tp_fp_fn"] = (int(tp), int(fp), int(fn))
    dbg["all_bg"] = (dbg["pred_fore_pixels"] == 0)

    if debug:
        print(f"    - IoU阈值: {iou_thr}, 匹配后 TP={tp}, FP={fp}, FN={fn}")

    return int(tp), int(fp), int(fn), dbg



# -----------------------------
# 验证回调（实例级 F1）— 支持多光谱
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
                 iou_thr=0.5,
                 # ✅ 多光谱输入配置
                 image_dir="JPEGImages",
                 image_suffix=".jpg",
                 in_channels=3,
                 norm_mode="percentile",
                 # ✅ ignore 像元值（例如 255）；None 表示不忽略
                 ignore_index=None,
                 # ✅ 可视化用：选择3个波段合成RGB（0-based）
                 rgb_vis_bands=(2, 1, 0)):

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

        self.image_dir = image_dir
        self.image_suffix = str(image_suffix)
        self.in_channels = int(in_channels)
        self.norm_mode = str(norm_mode).lower().strip()
        self.rgb_vis_bands = tuple(rgb_vis_bands)

        self.ignore_index = None if ignore_index is None else int(ignore_index)

        self.best_f1   = 0.0
        self.metrics_log = os.path.join(self.log_dir, "best_f1_summary.txt")

        self.save_visual_topk = int(save_visual_topk)
        self.vis_dir = os.path.join(self.log_dir, "vis")
        if self.save_visual_topk > 0:
            os.makedirs(self.vis_dir, exist_ok=True)

        self.debug_first_n       = int(debug_first_n)
        self.autodetect_scan_max = int(autodetect_scan_max)

        if target_label_value is None:
            self.target_label_value = self._auto_detect_target_label()
        else:
            self.target_label_value = int(target_label_value)
            print(f"[EvalCallback] 使用指定 target_label_value = {self.target_label_value}")

        print(f"[EvalCallback] image_dir={self.image_dir}, suffix={self.image_suffix}, "
              f"in_channels={self.in_channels}, norm_mode={self.norm_mode}, "
              f"rgb_vis_bands={self.rgb_vis_bands}, ignore_index={self.ignore_index}")

    def _auto_detect_target_label(self):
        """
        自动识别“前景像元值”：统计 val 标注中 (非0且非ignore) 的像元值频次，取最多的那个。
        """
        ids = [line.strip() for line in self.val_lines]
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")
        counts = Counter()
        scanned = 0

        for name in ids[:self.autodetect_scan_max]:
            mask_p = os.path.join(mask_dir, f"{name}.png")
            if not os.path.exists(mask_p):
                continue
            arr, _ = _read_mask_array(mask_p)
            uniq, freq = np.unique(arr, return_counts=True)
            for u, c in zip(uniq, freq):
                u = int(u)
                if u == 0:
                    continue
                if (self.ignore_index is not None) and (u == self.ignore_index):
                    continue
                counts[u] += int(c)
            scanned += 1

        if scanned == 0:
            print("[EvalCallback] ⚠️ 自动识别失败：未能扫描到有效标注文件，回退 target_label_value=1")
            return 1

        if not counts:
            print("[EvalCallback] ⚠️ 自动识别：非 0 且非 ignore 的像元不存在，怀疑你的标注只有背景；回退 target_label_value=1")
            return 1

        target = counts.most_common(1)[0][0]
        print(f"[EvalCallback] 自动识别 target_label_value = {target}（统计自前 {scanned} 张标注）")
        print(f"[EvalCallback] 非 0/非 ignore 像元 Top 值及计数（最多 5 个）：{counts.most_common(5)}")
        return int(target)

    # -----------------------------
    # ✅ 读图（RGB / tif 多波段）
    # -----------------------------
    def _read_image_hwc(self, img_path):
        use_tif = str(self.image_suffix).lower() in [".tif", ".tiff"]

        if use_tif:
            if rasterio is None:
                raise ImportError("读取 tif 需要 rasterio：pip install rasterio")
            with rasterio.open(img_path) as src:
                arr = src.read()  # (C,H,W) 原始 dtype

            orig_dtype = arr.dtype

            # 截断/补齐到 in_channels
            if arr.shape[0] >= self.in_channels:
                arr = arr[:self.in_channels]
            else:
                pad = np.zeros((self.in_channels - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)

            arr_f = arr.astype(np.float32)

            if self.norm_mode == "none":
                pass
            elif self.norm_mode == "max":
                if np.issubdtype(orig_dtype, np.integer):
                    maxv = float(np.iinfo(orig_dtype).max)
                else:
                    maxv = float(np.max(arr_f)) + 1e-6
                arr_f = arr_f / maxv
            else:
                out = np.zeros_like(arr_f, dtype=np.float32)
                for b in range(arr_f.shape[0]):
                    v = arr_f[b]
                    lo = np.percentile(v, 2)
                    hi = np.percentile(v, 98)
                    if hi - lo < 1e-6:
                        out[b] = 0.0
                    else:
                        out[b] = (v - lo) / (hi - lo)
                arr_f = np.clip(out, 0.0, 1.0)

            img = np.transpose(arr_f, (1, 2, 0))  # (H,W,C)
            return img

        img = Image.open(img_path).convert("RGB")
        return np.array(img, dtype=np.float32)  # (H,W,3) 0~255

    # -----------------------------
    # ✅ numpy 版 letterbox（支持任意通道）
    # -----------------------------
    def _letterbox_np(self, image_hwc):
        ih, iw = image_hwc.shape[:2]
        C = image_hwc.shape[2]
        W = self.input_shape[1]
        H = self.input_shape[0]

        scale = min(W / iw, H / ih)
        nw, nh = int(iw * scale), int(ih * scale)

        resized = cv2.resize(image_hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)

        is_rgb = (self.in_channels == 3) and (str(self.image_suffix).lower() in [".jpg", ".jpeg", ".png"])
        pad_val = 128.0 if is_rgb else 0.0

        canvas = np.ones((H, W, C), dtype=np.float32) * pad_val
        cx, cy = (W - nw) // 2, (H - nh) // 2
        canvas[cy:cy+nh, cx:cx+nw, :] = resized
        return canvas, (cx, cy, nw, nh)

    # -----------------------------
    # ✅ 多光谱可视化：选 3 波段合成 RGB + 拉伸到 0~255
    # -----------------------------
    def _to_vis_rgb_uint8(self, canvas_hwc):
        is_rgb = (canvas_hwc.shape[2] == 3) and (str(self.image_suffix).lower() in [".jpg", ".jpeg", ".png"])
        if is_rgb:
            return np.clip(canvas_hwc, 0, 255).astype(np.uint8)

        C = canvas_hwc.shape[2]
        b0, b1, b2 = self.rgb_vis_bands
        b0 = int(np.clip(b0, 0, C-1))
        b1 = int(np.clip(b1, 0, C-1))
        b2 = int(np.clip(b2, 0, C-1))
        rgb = canvas_hwc[:, :, [b0, b1, b2]].astype(np.float32)

        out = np.zeros_like(rgb, dtype=np.float32)
        for k in range(3):
            v = rgb[:, :, k]
            lo = np.percentile(v, 2)
            hi = np.percentile(v, 98)
            if hi - lo < 1e-6:
                out[:, :, k] = 0.0
            else:
                out[:, :, k] = (v - lo) / (hi - lo)
        out = np.clip(out, 0.0, 1.0) * 255.0
        return out.astype(np.uint8)

    def _visualize(self, original_rgb_u8, pred_bin, gt_bin, save_path, tp, fp, fn):
        edges = sobel(pred_bin) > 0
        gt_color = np.zeros((*gt_bin.shape, 3), dtype=np.uint8)
        gt_color[gt_bin == 1] = [255, 215, 0]
        pred_color = np.zeros((*pred_bin.shape, 3), dtype=np.uint8)
        pred_color[pred_bin == 1] = [135, 206, 250]

        alpha = 0.2
        blended = original_rgb_u8.astype(np.float32).copy()
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
        img_dir  = os.path.join(self.voc_root, "VOC2007", self.image_dir)
        mask_dir = os.path.join(self.voc_root, "VOC2007/SegmentationClass")

        tp_sum = fp_sum = fn_sum = 0
        saved_vis = 0
        H, W = self.input_shape[0], self.input_shape[1]

        with torch.inference_mode():
            for idx, name in enumerate(tqdm(ids, desc="Validating", ncols=100)):
                img_p  = os.path.join(img_dir,  f"{name}{self.image_suffix}")
                mask_p = os.path.join(mask_dir, f"{name}.png")
                if (not os.path.exists(img_p)) or (not os.path.exists(mask_p)):
                    continue

                img_hwc = self._read_image_hwc(img_p)
                mask_pil = Image.open(mask_p)

                canvas, (cx, cy, nw, nh) = self._letterbox_np(img_hwc)

                is_rgb = (self.in_channels == 3) and (str(self.image_suffix).lower() in [".jpg", ".jpeg", ".png"])
                if is_rgb:
                    img_np = preprocess_input(canvas.astype(np.float32))
                else:
                    img_np = canvas.astype(np.float32)

                x = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).contiguous()
                if self.cuda:
                    x = x.cuda(non_blocking=True)

                out = model(x)
                pred_full = torch.softmax(out, dim=1).argmax(dim=1).cpu().numpy()[0]  # (H, W) 0/1

                # GT 保持原始像元值（0/38/255...），不要强行改成 0/1
                gt_resized = _resize_mask_preserve_index(mask_pil, (nw, nh))
                gt_full = np.zeros((H, W), dtype=np.uint8)
                gt_full[cy:cy+nh, cx:cx+nw] = gt_resized

                pred = pred_full
                gt   = gt_full

                debug_this = (idx < self.debug_first_n)
                if debug_this:
                    pred_crop = pred_full[cy:cy+nh, cx:cx+nw]
                    gt_fg = int(((gt_resized == self.target_label_value) &
                                 (gt_resized != self.ignore_index if self.ignore_index is not None else True)).sum())
                    print(f"\n[DEBUG] 样本 {idx+1}: {name}")
                    print(f"    - pred_full_fore(valid前) = {int((pred_full != 0).sum())}, "
                          f"pred_crop_fore = {int((pred_crop != 0).sum())}, "
                          f"gt_crop_fore(=={self.target_label_value}) = {gt_fg}")

                tp, fp, fn, dbg = _instance_metrics(
                    pred, gt,
                    target_label=self.target_label_value,
                    area_thr=self.area_thr,
                    per_thr=self.perim_thr,
                    circ_thr=self.circ_thr,
                    iou_thr=self.iou_thr,
                    ignore_index=self.ignore_index,
                    debug=debug_this
                )
                if debug_this and dbg["all_bg"]:
                    print("    ⚠️ 预测几乎全背景（valid 内前景像素数为 0）。")

                tp_sum += tp
                fp_sum += fp
                fn_sum += fn

                if self.save_visual_topk > 0 and saved_vis < self.save_visual_topk:
                    original_rgb_u8 = self._to_vis_rgb_uint8(canvas)

                    # 可视化也建议忽略 ignore 区域
                    if self.ignore_index is not None:
                        valid = (gt != self.ignore_index)
                        pred_bin = ((pred != 0) & valid).astype(np.uint8)
                        gt_bin   = ((gt == self.target_label_value) & valid).astype(np.uint8)
                    else:
                        pred_bin = (pred != 0).astype(np.uint8)
                        gt_bin   = (gt == self.target_label_value).astype(np.uint8)

                    save_path = os.path.join(self.vis_dir, f"epoch_{epoch+1:03d}_{name}.png")
                    self._visualize(original_rgb_u8, pred_bin, gt_bin, save_path, tp, fp, fn)
                    saved_vis += 1

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
