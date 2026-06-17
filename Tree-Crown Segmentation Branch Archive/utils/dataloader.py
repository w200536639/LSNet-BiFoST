import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


# ============================================================
# Basic image utilities
# ============================================================
def cvtColor(image):
    """
    Convert PIL image to RGB.

    This prevents errors caused by grayscale or palette images.
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image

    return image.convert("RGB")


def preprocess_input(image):
    """
    Normalize RGB image to [0, 1].

    Compatible with NumPy 1.x and NumPy 2.x.
    Do not use np.array(..., copy=False).
    """
    image = np.asarray(image, dtype=np.float32)
    image /= 255.0
    return image


def normalize_segmentation_mask(mask, num_classes):
    """
    Normalize segmentation mask labels.

    For binary segmentation:
        background = 0
        foreground = 1

    This supports masks saved as:
        0/1
        0/255
        0/38
        or any nonzero foreground label.
    """
    mask = np.asarray(mask, dtype=np.uint8)

    if num_classes == 2:
        mask = (mask > 0).astype(np.uint8)
    else:
        mask[mask >= num_classes] = num_classes

    return mask


# ============================================================
# Dataset
# ============================================================
class UnetDataset(Dataset):
    def __init__(
        self,
        annotation_lines,
        input_shape,
        num_classes,
        train,
        dataset_path,
    ):
        super(UnetDataset, self).__init__()

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index].strip()
        name = annotation_line.split()[0]

        jpg_path = os.path.join(
            self.dataset_path,
            "VOC2007/JPEGImages",
            name + ".jpg",
        )
        png_path = os.path.join(
            self.dataset_path,
            "VOC2007/SegmentationClass",
            name + ".png",
        )

        if not os.path.exists(jpg_path):
            jpg_path = self.find_image_path(
                folder=os.path.join(self.dataset_path, "VOC2007/JPEGImages"),
                name=name,
            )

        if not os.path.exists(png_path):
            png_path = self.find_mask_path(
                folder=os.path.join(self.dataset_path, "VOC2007/SegmentationClass"),
                name=name,
            )

        jpg = Image.open(jpg_path)
        png = Image.open(png_path)

        jpg, png = self.get_random_data(
            image=jpg,
            label=png,
            input_shape=self.input_shape,
            random=self.train,
        )

        jpg = np.transpose(preprocess_input(jpg), [2, 0, 1])

        png = normalize_segmentation_mask(
            mask=png,
            num_classes=self.num_classes,
        )

        seg_labels = np.eye(self.num_classes + 1, dtype=np.float32)[
            png.reshape([-1])
        ]
        seg_labels = seg_labels.reshape(
            (
                int(self.input_shape[0]),
                int(self.input_shape[1]),
                self.num_classes + 1,
            )
        )

        return jpg, png, seg_labels

    @staticmethod
    def find_image_path(folder, name):
        """
        Find image path when the extension is not .jpg.
        """
        supported_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        for ext in supported_exts:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path

        for file_name in os.listdir(folder):
            if os.path.splitext(file_name)[0] == name:
                return os.path.join(folder, file_name)

        raise FileNotFoundError(f"Cannot find image for: {name} in {folder}")

    @staticmethod
    def find_mask_path(folder, name):
        """
        Find mask path when the extension is not .png.
        """
        supported_exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

        for ext in supported_exts:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path

        for file_name in os.listdir(folder):
            if os.path.splitext(file_name)[0] == name:
                return os.path.join(folder, file_name)

        raise FileNotFoundError(f"Cannot find mask for: {name} in {folder}")

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(
        self,
        image,
        label,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=0.7,
        val=0.3,
        random=True,
    ):
        """
        Data augmentation.

        Args:
            image: PIL RGB image.
            label: PIL mask image.
            input_shape: [height, width].
            random: whether to use random augmentation.
        """
        image = cvtColor(image)
        label = Image.fromarray(np.asarray(label, dtype=np.uint8))

        image_width, image_height = image.size
        input_height, input_width = input_shape

        if not random:
            scale = min(input_width / image_width, input_height / image_height)

            new_width = int(image_width * scale)
            new_height = int(image_height * scale)

            dx = (input_width - new_width) // 2
            dy = (input_height - new_height) // 2

            image = image.resize((new_width, new_height), Image.BICUBIC)
            label = label.resize((new_width, new_height), Image.NEAREST)

            new_image = Image.new("RGB", (input_width, input_height), (128, 128, 128))
            new_label = Image.new("L", (input_width, input_height), 0)

            new_image.paste(image, (dx, dy))
            new_label.paste(label, (dx, dy))

            image_data = np.asarray(new_image, dtype=np.float32)
            label_data = np.asarray(new_label, dtype=np.uint8)

            return image_data, label_data

        # ----------------------------------------------------
        # Random resize and aspect-ratio jitter
        # ----------------------------------------------------
        new_ar = (
            image_width
            / image_height
            * self.rand(1 - jitter, 1 + jitter)
            / self.rand(1 - jitter, 1 + jitter)
        )
        scale = self.rand(0.25, 2.0)

        if new_ar < 1:
            new_height = int(scale * input_height)
            new_width = int(new_height * new_ar)
        else:
            new_width = int(scale * input_width)
            new_height = int(new_width / new_ar)

        new_width = max(1, new_width)
        new_height = max(1, new_height)

        image = image.resize((new_width, new_height), Image.BICUBIC)
        label = label.resize((new_width, new_height), Image.NEAREST)

        # ----------------------------------------------------
        # Random horizontal flip
        # ----------------------------------------------------
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ----------------------------------------------------
        # Random placement
        # ----------------------------------------------------
        dx = int(self.rand(0, input_width - new_width)) if input_width > new_width else 0
        dy = int(self.rand(0, input_height - new_height)) if input_height > new_height else 0

        new_image = Image.new("RGB", (input_width, input_height), (128, 128, 128))
        new_label = Image.new("L", (input_width, input_height), 0)

        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))

        image = new_image
        label = new_label

        image_data = np.asarray(image, dtype=np.uint8)

        # ----------------------------------------------------
        # HSV color augmentation
        # ----------------------------------------------------
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue_channel, sat_channel, val_channel = cv2.split(
            cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV)
        )

        dtype = image_data.dtype
        x = np.arange(0, 256, dtype=r.dtype)

        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge(
            (
                cv2.LUT(hue_channel, lut_hue),
                cv2.LUT(sat_channel, lut_sat),
                cv2.LUT(val_channel, lut_val),
            )
        )
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        label_data = np.asarray(label, dtype=np.uint8)

        return image_data, label_data


# ============================================================
# Collate function
# ============================================================
def unet_dataset_collate(batch):
    """
    Collate function for segmentation training.

    Important:
        This function must return torch.Tensor, not numpy.ndarray.
        Otherwise utils_fit.py will raise:
            AttributeError: 'numpy.ndarray' object has no attribute 'cuda'
    """
    images = []
    pngs = []
    seg_labels = []

    for image, png, labels in batch:
        images.append(image)
        pngs.append(png)
        seg_labels.append(labels)

    images = torch.from_numpy(np.asarray(images, dtype=np.float32))
    pngs = torch.from_numpy(np.asarray(pngs, dtype=np.int64))
    seg_labels = torch.from_numpy(np.asarray(seg_labels, dtype=np.float32))

    return images, pngs, seg_labels
