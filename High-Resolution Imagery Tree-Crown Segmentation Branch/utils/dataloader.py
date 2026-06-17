import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

# ✅ 多光谱 tif 建议用 rasterio
try:
    import rasterio
except Exception:
    rasterio = None

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    """
    兼容两种输入：
    1) 传统 VOC RGB：VOC2007/JPEGImages/*.jpg (in_channels=3, suffix=.jpg)
    2) 多光谱/遥感：VOC2007/RSImages/*.tif (in_channels=8, suffix=.tif)

    ✅ 关键约定：
    - raw mask 里前景可能是 38（或别的值）
    - 训练/验证喂给网络的 png 必须是 0/1/(ignore=num_classes)
      其中：前景 train_id 固定为 1
    """
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
        image_dir="JPEGImages",
        image_suffix=".jpg",
        in_channels=3,
        norm_mode="percentile",          # "percentile"(推荐) / "max" / "none"
        target_label_value=None,         # ✅ raw 前景像元值；None 则自动扫描识别
        autodetect_scan_max=200,
        ignore_index=None,               # raw mask 里需要忽略的像元值（例如 255）
        debug_first_n=0,                 # ✅ 前 N 个样本打印 raw/mapped unique（默认关）
    ):
        super(UnetDataset, self).__init__()
        self.annotation_lines    = annotation_lines
        self.length              = len(annotation_lines)
        self.input_shape         = input_shape
        self.num_classes         = int(num_classes)
        self.train               = train
        self.dataset_path        = dataset_path

        self.image_dir           = image_dir
        self.image_suffix        = image_suffix
        self.in_channels         = int(in_channels)
        self.norm_mode           = str(norm_mode).lower().strip()

        self.autodetect_scan_max = int(autodetect_scan_max)
        self.ignore_index_value  = ignore_index  # raw mask ignore 值（例如 255）
        self.debug_first_n       = int(debug_first_n)

        # ✅ 训练体系约定：前景训练 id 恒为 1；ignore id 恒为 num_classes
        self.fg_train_id         = 1
        self.ignore_train_id     = self.num_classes

        # ✅ raw 前景像元值（例如 38）
        if target_label_value is None:
            self.fg_raw_value = self._auto_detect_target_label()
        else:
            self.fg_raw_value = int(target_label_value)

        if self.length > 0:
            print(
                f"[UnetDataset] train={self.train} | image_dir={self.image_dir} | suffix={self.image_suffix} | "
                f"in_channels={self.in_channels} | norm_mode={self.norm_mode} | "
                f"fg_raw_value={self.fg_raw_value} -> fg_train_id={self.fg_train_id} | "
                f"ignore_raw={self.ignore_index_value} -> ignore_train_id={self.ignore_train_id}"
            )

    def __len__(self):
        return self.length

    def label_mapping(self):
        """
        给外部模块（比如 EvalCallback）用的映射表，避免把 38/1 搞混。
        """
        return {
            "fg_raw_value": int(self.fg_raw_value),
            "fg_train_id": int(self.fg_train_id),
            "ignore_raw_value": None if self.ignore_index_value is None else int(self.ignore_index_value),
            "ignore_train_id": int(self.ignore_train_id),
        }

    # ------------------------- #
    # ✅ 安全读取 mask（关键修正） #
    # - P 模式：直接 np.array 得到“索引值”(例如38)，不会被 convert("L") 误映射
    # - 其他模式：读取灰度
    # ------------------------- #
    def _read_mask_u8(self, mask_path):
        m = Image.open(mask_path)
        if m.mode == "P":
            arr = np.array(m, dtype=np.uint8)
        else:
            arr = np.array(m.convert("L"), dtype=np.uint8)
        return arr

    # ------------------------- #
    # ✅ 自动识别 raw mask 前景像元值 #
    # ------------------------- #
    def _auto_detect_target_label(self):
        if self.length == 0:
            return 1

        mask_dir = os.path.join(self.dataset_path, "VOC2007", "SegmentationClass")
        counts = {}

        scan_n = min(self.autodetect_scan_max, self.length)
        scanned = 0

        for i in range(scan_n):
            line = self.annotation_lines[i].strip()
            if not line:
                continue
            name = line.split()[0]
            p = os.path.join(mask_dir, name + ".png")
            if not os.path.exists(p):
                continue

            # ✅ 关键：不要 convert("L")，要保留 P 模式索引值
            arr = self._read_mask_u8(p)

            uniq, freq = np.unique(arr, return_counts=True)
            for u, c in zip(uniq, freq):
                u = int(u)
                if u == 0:
                    continue
                if (self.ignore_index_value is not None) and (u == int(self.ignore_index_value)):
                    continue
                counts[u] = counts.get(u, 0) + int(c)

            scanned += 1

        if scanned == 0 or len(counts) == 0:
            print("[UnetDataset] ⚠️ 自动识别前景失败：未扫描到有效非0像元，回退 fg_raw_value=1")
            return 1

        target = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0][0]
        top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"[UnetDataset] ✅ 自动识别 fg_raw_value={target}（扫描 {scanned} 张 mask，Top5={top5}）")
        return int(target)

    # ------------------------- #
    # 读图：支持 jpg / tif 多波段 #
    # ------------------------- #
    def _read_image(self, path):
        suffix = os.path.splitext(path)[1].lower()

        if suffix in [".tif", ".tiff"]:
            if rasterio is None:
                raise ImportError("读取 tif 需要 rasterio：pip install rasterio")

            with rasterio.open(path) as src:
                arr = src.read()  # (C, H, W)

            if arr.shape[0] >= self.in_channels:
                arr = arr[:self.in_channels]
            else:
                pad = np.zeros((self.in_channels - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)

            arr = arr.astype(np.float32)

            if self.norm_mode == "none":
                pass
            elif self.norm_mode == "max":
                maxv = float(np.max(arr)) + 1e-6
                arr = arr / maxv
            else:
                out = np.zeros_like(arr, dtype=np.float32)
                for b in range(arr.shape[0]):
                    v = arr[b]
                    lo = np.percentile(v, 2)
                    hi = np.percentile(v, 98)
                    if hi - lo < 1e-6:
                        out[b] = 0.0
                    else:
                        out[b] = (v - lo) / (hi - lo)
                arr = np.clip(out, 0.0, 1.0)

            img = np.transpose(arr, (1, 2, 0))  # (H,W,C)
            return img.astype(np.float32)

        else:
            img = Image.open(path)
            img = cvtColor(img)
            img = np.array(img, dtype=np.float32)
            return img

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_hwc, label_hw, input_shape, jitter=0.3, random=True):
        h, w = input_shape
        ih, iw = image_hwc.shape[0], image_hwc.shape[1]
        c = image_hwc.shape[2]

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image_resized = cv2.resize(image_hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)
            label_resized = cv2.resize(label_hw, (nw, nh), interpolation=cv2.INTER_NEAREST)

            if self.image_suffix.lower() in [".jpg", ".png", ".jpeg"] and self.in_channels == 3:
                pad_val = 128.0
            else:
                pad_val = 0.0

            new_image = np.ones((h, w, c), dtype=np.float32) * pad_val
            new_label = np.zeros((h, w), dtype=label_hw.dtype)

            dx = (w - nw) // 2
            dy = (h - nh) // 2
            new_image[dy:dy + nh, dx:dx + nw, :] = image_resized
            new_label[dy:dy + nh, dx:dx + nw] = label_resized
            return new_image, new_label

        new_ar = (iw / ih) * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2.0)

        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image_resized = cv2.resize(image_hwc, (nw, nh), interpolation=cv2.INTER_LINEAR)
        label_resized = cv2.resize(label_hw, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if self.rand() < 0.5:
            image_resized = image_resized[:, ::-1, :]
            label_resized = label_resized[:, ::-1]

        dx = int(self.rand(0, max(1, w - nw)))
        dy = int(self.rand(0, max(1, h - nh)))

        if self.image_suffix.lower() in [".jpg", ".png", ".jpeg"] and self.in_channels == 3:
            pad_val = 128.0
        else:
            pad_val = 0.0

        new_image = np.ones((h, w, c), dtype=np.float32) * pad_val
        new_label = np.zeros((h, w), dtype=label_hw.dtype)

        x1 = dx
        y1 = dy
        x2 = min(w, dx + nw)
        y2 = min(h, dy + nh)

        src_x1 = 0
        src_y1 = 0
        src_x2 = x2 - x1
        src_y2 = y2 - y1

        new_image[y1:y2, x1:x2, :] = image_resized[src_y1:src_y2, src_x1:src_x2, :]
        new_label[y1:y2, x1:x2] = label_resized[src_y1:src_y2, src_x1:src_x2]

        return new_image, new_label

    # ------------------------- #
    # ✅ raw mask -> train ids #
    # ------------------------- #
    def _map_mask_to_train_ids(self, mask_hw):
        """
        输出：
        - 0: 背景
        - 1: 前景（raw==fg_raw_value）
        - num_classes: ignore
        """
        mask_hw = np.array(mask_hw)
        if mask_hw.dtype != np.uint8:
            mask_hw = mask_hw.astype(np.uint8)

        out = np.zeros_like(mask_hw, dtype=np.uint8)

        # 前景：raw_fg -> train_id=1
        out[mask_hw == int(self.fg_raw_value)] = self.fg_train_id

        # ignore：raw_ignore -> train_ignore=num_classes
        if self.ignore_index_value is not None:
            out[mask_hw == int(self.ignore_index_value)] = self.ignore_train_id

        # 其他非0非fg（也非ignore）统一当 ignore，避免污染训练
        other = (mask_hw != 0) & (mask_hw != int(self.fg_raw_value))
        if self.ignore_index_value is not None:
            other = other & (mask_hw != int(self.ignore_index_value))
        out[other] = self.ignore_train_id

        return out

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        img_path = os.path.join(self.dataset_path, "VOC2007", self.image_dir, name + self.image_suffix)
        lab_path = os.path.join(self.dataset_path, "VOC2007", "SegmentationClass", name + ".png")

        image = self._read_image(img_path)  # (H,W,C)

        # ✅ 关键：安全读取 raw mask，P 模式保留索引值
        mask_raw = self._read_mask_u8(lab_path)  # raw（可能是 0/38/255...）

        image, mask_raw = self.get_random_data(image, mask_raw, self.input_shape, random=self.train)

        if self.image_suffix.lower() in [".jpg", ".png", ".jpeg"] and self.in_channels == 3:
            image = preprocess_input(np.array(image, np.float64))
        else:
            image = np.array(image, np.float32)

        image = np.transpose(image, (2, 0, 1))  # CHW

        # ✅ 映射到训练 id：0/1/ignore
        png = self._map_mask_to_train_ids(mask_raw)

        # 防御：确保不会超过 ignore id
        png[png > self.num_classes] = self.num_classes

        # one-hot（num_classes+1，最后一类作为 ignore）
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # ✅ 可选 debug：仅打印前 N 个样本一次
        if self.debug_first_n > 0 and index < self.debug_first_n:
            ur = np.unique(mask_raw)
            um = np.unique(png)
            print(f"[UnetDataset-DEBUG] {name}: raw_unique={ur.tolist()} -> mapped_unique={um.tolist()}")

        return image, png, seg_labels


def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
