import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # [H, W]
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 读取图像与标签（原 VOC 路径）
        jpg_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png")
        jpg = Image.open(jpg_path)
        png = Image.open(png_path)

        # 无论训练或验证，统一使用固定的缩放+padding（保证训练/验证一致）
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=False)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """
        NOTE: 为保持训练与验证一致性，此处实现为确定性处理：
        - 等比例缩放
        - 中心对齐灰条填充（灰条128）
        - image: RGB array (uint8)
        - label: L array (uint8)
        返回：(image_array_uint8, label_array_uint8)
        """
        image = cvtColor(image)  # 转为RGB PIL
        label = Image.fromarray(np.array(label))  # 确保为 PIL 格式

        iw, ih = image.size
        h, w = input_shape

        # 等比例缩放到目标框内
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image_resized = image.resize((nw, nh), Image.BICUBIC)
        label_resized = label.resize((nw, nh), Image.NEAREST)

        # 中心粘贴 + 灰条填充 (灰128, label 背景值 0)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), 0)

        dx = (w - nw) // 2
        dy = (h - nh) // 2

        new_image.paste(image_resized, (dx, dy))
        new_label.paste(label_resized, (dx, dy))

        # 返回 numpy uint8 类型（与原实现兼容）
        return np.array(new_image, dtype=np.uint8), np.array(new_label, dtype=np.uint8)


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
