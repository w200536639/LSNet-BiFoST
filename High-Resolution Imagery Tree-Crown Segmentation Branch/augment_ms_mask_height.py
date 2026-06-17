import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from PIL import Image


# ============================================================
# Manual configuration
# ============================================================
SRC_ROOT = "TreeHeightDataset"
DST_ROOT = "VOCdevkit"

# Source folders
SRC_IMAGE_DIR = "RSImages"
SRC_MASK_DIR = "mask"

# Source file suffixes
IMAGE_SUFFIX = ".tif"
MASK_SUFFIX = ".png"

# Output VOC2007 folders
VOC_YEAR = "VOC2007"
DST_IMAGE_DIR = "RSImages"
DST_MASK_DIR = "SegmentationClass"
DST_IMAGE_SET_DIR = os.path.join("ImageSets", "Segmentation")

# Patch size
PATCH_H = 640
PATCH_W = 640

# Training split augmentation
TRAIN_PATCHES_PER_IMAGE = 4
TRAIN_AUGS_PER_PATCH = 6
TRAIN_KEEP_IDENTITY = True

# Validation split configuration
# 建议验证集不做增强，只裁剪 patch，保证评估更客观。
VAL_PATCHES_PER_IMAGE = 4
VAL_AUGS_PER_PATCH = 0
VAL_KEEP_IDENTITY = True

# Empty-patch filtering
SKIP_EMPTY_PATCH = True
EMPTY_THRESH = 0.001

# Try multiple random crops to avoid empty patches
MAX_CROP_TRIES_PER_PATCH = 30

# Mask value handling
# None = any non-zero value is treated as foreground.
# If your source foreground is 38, set SOURCE_FOREGROUND_VALUE = 38.
SOURCE_FOREGROUND_VALUE = None

# If source masks contain ignore value, set it here, e.g. 255.
# The output mask will still be 0/1 by default.
SOURCE_IGNORE_VALUE = None

# Output mask values
OUTPUT_BACKGROUND_VALUE = 0
OUTPUT_FOREGROUND_VALUE = 1

# Random seed
SEED = 11

# GeoTIFF writing
# For training patches, it is safer not to keep original georeferencing,
# especially after flip/rotation augmentation.
KEEP_GEOREFERENCE = False

# None = no compression.
# You can set "deflate" if your environment supports it.
TIFF_COMPRESS = None

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Directory utilities
# ============================================================
def ensure_voc_dirs(dst_root):
    voc_root = Path(dst_root) / VOC_YEAR

    image_dir = voc_root / DST_IMAGE_DIR
    mask_dir = voc_root / DST_MASK_DIR
    set_dir = voc_root / DST_IMAGE_SET_DIR

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    set_dir.mkdir(parents=True, exist_ok=True)

    return image_dir, mask_dir, set_dir


def read_list(txt_path):
    """
    Read train.txt / val.txt.

    Each line can be:
        image_name
        image_name.tif
        image_name other_info
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"List file does not exist: {txt_path}")

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]

    for encoding in encodings:
        try:
            with open(txt_path, "r", encoding=encoding) as file:
                lines = []
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    first_token = line.split()[0]
                    prefix = os.path.splitext(first_token)[0]
                    lines.append(prefix)
                return lines
        except UnicodeDecodeError:
            continue

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = []
        for line in file:
            line = line.strip()
            if not line:
                continue
            first_token = line.split()[0]
            prefix = os.path.splitext(first_token)[0]
            lines.append(prefix)
        return lines


def write_list(txt_path, lines):
    with open(txt_path, "w", encoding="utf-8") as file:
        for item in lines:
            file.write(str(item) + "\n")


def safe_name(name):
    """Make filename safe for Windows."""
    name = str(name)
    name = name.replace("（", "(").replace("）", ")")

    illegal_chars = '<>:"/\\|?*'

    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


# ============================================================
# IO utilities
# ============================================================
def read_multispectral_tif(image_path):
    """
    Read multispectral image.

    Returns:
        image: C x H x W
        profile: rasterio profile
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile.copy()

    return image, profile


def read_mask(mask_path):
    """
    Read mask safely.

    If mask is palette mode, keep its index values.
    Otherwise, convert to grayscale.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask does not exist: {mask_path}")

    mask = Image.open(mask_path)

    if mask.mode == "P":
        arr = np.array(mask, dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L"), dtype=np.uint8)

    return arr


def convert_mask_to_binary(mask_raw):
    """
    Convert source mask to 0/1 binary mask.

    Output:
        0 = background
        1 = crown foreground
    """
    mask_raw = mask_raw.astype(np.int32)

    if SOURCE_IGNORE_VALUE is not None:
        valid = mask_raw != int(SOURCE_IGNORE_VALUE)
    else:
        valid = np.ones_like(mask_raw, dtype=bool)

    if SOURCE_FOREGROUND_VALUE is None:
        foreground = (mask_raw > 0) & valid
    else:
        foreground = (mask_raw == int(SOURCE_FOREGROUND_VALUE)) & valid

    mask_binary = np.zeros_like(mask_raw, dtype=np.uint8)
    mask_binary[foreground] = OUTPUT_FOREGROUND_VALUE

    return mask_binary


def build_tif_profile(image_c_hw, ref_profile=None):
    """Build output GeoTIFF profile."""
    count, height, width = image_c_hw.shape

    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": int(count),
        "dtype": str(image_c_hw.dtype),
    }

    if TIFF_COMPRESS:
        profile["compress"] = TIFF_COMPRESS

    if KEEP_GEOREFERENCE and ref_profile is not None:
        if ref_profile.get("crs", None) is not None:
            profile["crs"] = ref_profile["crs"]
        if ref_profile.get("transform", None) is not None:
            profile["transform"] = ref_profile["transform"]

    return profile


def save_multispectral_tif(path, image_c_hw, ref_profile=None):
    """Save multispectral patch."""
    image_c_hw = np.ascontiguousarray(image_c_hw)

    profile = build_tif_profile(image_c_hw, ref_profile=ref_profile)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(image_c_hw)


def save_mask_png(path, mask_hw):
    """
    Save binary mask as PNG.

    Important:
        foreground is saved as 1, not 255.
    """
    mask_hw = np.ascontiguousarray(mask_hw.astype(np.uint8))

    mask_out = np.zeros_like(mask_hw, dtype=np.uint8)
    mask_out[mask_hw > 0] = OUTPUT_FOREGROUND_VALUE

    Image.fromarray(mask_out, mode="L").save(path)


# ============================================================
# Discrete geometric augmentation
# ============================================================
DISCRETE_MODES = [
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
]


def apply_discrete_geom(image_c_hw, mask_hw, mode):
    """
    Apply discrete geometric augmentation.

    Only index operations are used.
    No interpolation is introduced.
    """
    if mode == "identity":
        image_aug = image_c_hw
        mask_aug = mask_hw

    elif mode == "hflip":
        image_aug = image_c_hw[:, :, ::-1]
        mask_aug = mask_hw[:, ::-1]

    elif mode == "vflip":
        image_aug = image_c_hw[:, ::-1, :]
        mask_aug = mask_hw[::-1, :]

    elif mode == "rot90":
        image_aug = np.rot90(image_c_hw, 1, axes=(1, 2))
        mask_aug = np.rot90(mask_hw, 1)

    elif mode == "rot180":
        image_aug = np.rot90(image_c_hw, 2, axes=(1, 2))
        mask_aug = np.rot90(mask_hw, 2)

    elif mode == "rot270":
        image_aug = np.rot90(image_c_hw, 3, axes=(1, 2))
        mask_aug = np.rot90(mask_hw, 3)

    else:
        raise ValueError(f"Unsupported augmentation mode: {mode}")

    return np.ascontiguousarray(image_aug), np.ascontiguousarray(mask_aug)


def choose_aug_modes(augs_per_patch):
    """
    Choose augmentation modes.

    If requested augmentations are more than available modes,
    modes may repeat.
    """
    if augs_per_patch <= 0:
        return []

    if augs_per_patch <= len(DISCRETE_MODES):
        return random.sample(DISCRETE_MODES, augs_per_patch)

    modes = []

    while len(modes) < augs_per_patch:
        modes.extend(random.sample(DISCRETE_MODES, len(DISCRETE_MODES)))

    return modes[:augs_per_patch]


# ============================================================
# Patch sampling
# ============================================================
def random_crop_position(height, width):
    top = random.randint(0, height - PATCH_H)
    left = random.randint(0, width - PATCH_W)
    return top, left


def crop_patch(image_c_hw, mask_hw, top, left):
    image_patch = image_c_hw[:, top: top + PATCH_H, left: left + PATCH_W]
    mask_patch = mask_hw[top: top + PATCH_H, left: left + PATCH_W]

    return image_patch, mask_patch


def sample_non_empty_patch(image_c_hw, mask_hw):
    """
    Try to sample a non-empty patch.

    If no valid patch is found after MAX_CROP_TRIES_PER_PATCH,
    return the last sampled patch.
    """
    height, width = mask_hw.shape

    last_image_patch = None
    last_mask_patch = None
    last_top = 0
    last_left = 0

    for _ in range(MAX_CROP_TRIES_PER_PATCH):
        top, left = random_crop_position(height, width)

        image_patch, mask_patch = crop_patch(image_c_hw, mask_hw, top, left)

        last_image_patch = image_patch
        last_mask_patch = mask_patch
        last_top = top
        last_left = left

        foreground_ratio = float((mask_patch > 0).mean())

        if not SKIP_EMPTY_PATCH:
            return image_patch, mask_patch, top, left, foreground_ratio

        if foreground_ratio >= EMPTY_THRESH:
            return image_patch, mask_patch, top, left, foreground_ratio

    return last_image_patch, last_mask_patch, last_top, last_left, float((last_mask_patch > 0).mean())


# ============================================================
# Split processing
# ============================================================
def process_split(
    split_name,
    prefixes,
    image_out_dir,
    mask_out_dir,
    patches_per_image,
    augs_per_patch,
    keep_identity=True,
):
    out_prefixes = []

    stat = defaultdict(int)
    stat["images_total"] = len(prefixes)

    for raw_prefix in prefixes:
        prefix = os.path.splitext(str(raw_prefix).strip())[0]
        prefix = safe_name(prefix)

        image_path = os.path.join(SRC_ROOT, SRC_IMAGE_DIR, prefix + IMAGE_SUFFIX)
        mask_path = os.path.join(SRC_ROOT, SRC_MASK_DIR, prefix + MASK_SUFFIX)

        if not os.path.exists(image_path):
            print(f"[{split_name}] Skip, image missing: {image_path}")
            stat["images_missing"] += 1
            continue

        if not os.path.exists(mask_path):
            print(f"[{split_name}] Skip, mask missing: {mask_path}")
            stat["masks_missing"] += 1
            continue

        image, image_profile = read_multispectral_tif(image_path)
        mask_raw = read_mask(mask_path)
        mask = convert_mask_to_binary(mask_raw)

        # Align to common size
        common_h = min(image.shape[1], mask.shape[0])
        common_w = min(image.shape[2], mask.shape[1])

        image = image[:, :common_h, :common_w]
        mask = mask[:common_h, :common_w]

        if common_h < PATCH_H or common_w < PATCH_W:
            print(f"[{split_name}] Skip, image too small for patch: {prefix} ({common_h}x{common_w})")
            stat["too_small"] += 1
            continue

        stat["images_used"] += 1

        for patch_id in range(patches_per_image):
            stat["patch_try"] += 1

            image_patch, mask_patch, top, left, fg_ratio = sample_non_empty_patch(image, mask)

            if SKIP_EMPTY_PATCH and fg_ratio < EMPTY_THRESH:
                stat["patch_skipped_empty"] += 1
                continue

            stat["patch_kept"] += 1

            # Identity patch
            if keep_identity:
                new_prefix = f"{split_name}_{prefix}_p{patch_id:03d}_id"
                image_save_path = os.path.join(image_out_dir, new_prefix + IMAGE_SUFFIX)
                mask_save_path = os.path.join(mask_out_dir, new_prefix + MASK_SUFFIX)

                save_multispectral_tif(image_save_path, image_patch, ref_profile=image_profile)
                save_mask_png(mask_save_path, mask_patch)

                out_prefixes.append(new_prefix)
                stat["samples_written"] += 1

            # Augmented patches
            aug_modes = choose_aug_modes(augs_per_patch)

            for aug_id, mode in enumerate(aug_modes):
                image_aug, mask_aug = apply_discrete_geom(image_patch, mask_patch, mode)

                new_prefix = f"{split_name}_{prefix}_p{patch_id:03d}_{mode}_a{aug_id:02d}"
                image_save_path = os.path.join(image_out_dir, new_prefix + IMAGE_SUFFIX)
                mask_save_path = os.path.join(mask_out_dir, new_prefix + MASK_SUFFIX)

                save_multispectral_tif(image_save_path, image_aug, ref_profile=image_profile)
                save_mask_png(mask_save_path, mask_aug)

                out_prefixes.append(new_prefix)
                stat["samples_written"] += 1

        print(f"[{split_name}] Done: {prefix}")

    return out_prefixes, dict(stat)


def print_stat(split_name, stat, base_count, output_count):
    print(f"\n[{split_name}] Statistics")
    print(f"Original entries: {base_count}")
    print(f"Output entries:   {output_count}")
    print(f"Images total:     {stat.get('images_total', 0)}")
    print(f"Images used:      {stat.get('images_used', 0)}")
    print(f"Images missing:   {stat.get('images_missing', 0)}")
    print(f"Masks missing:    {stat.get('masks_missing', 0)}")
    print(f"Too small:        {stat.get('too_small', 0)}")
    print(f"Patch tried:      {stat.get('patch_try', 0)}")
    print(f"Patch kept:       {stat.get('patch_kept', 0)}")
    print(f"Patch empty skip: {stat.get('patch_skipped_empty', 0)}")
    print(f"Samples written:  {stat.get('samples_written', 0)}")

    if stat.get("patch_try", 0) > 0:
        kept_ratio = stat.get("patch_kept", 0) / max(1, stat.get("patch_try", 0))
        print(f"Patch kept ratio: {kept_ratio:.4f}")

    if base_count > 0:
        print(f"Expansion ratio:  {output_count / base_count:.2f}x")


# ============================================================
# Main
# ============================================================
def main():
    image_out_dir, mask_out_dir, set_out_dir = ensure_voc_dirs(DST_ROOT)

    train_txt = os.path.join(SRC_ROOT, "train.txt")
    val_txt = os.path.join(SRC_ROOT, "val.txt")

    train_list = read_list(train_txt)
    val_list = read_list(val_txt)

    print("============================================================")
    print("Prepare high-resolution crown segmentation VOC dataset")
    print("============================================================")
    print(f"SRC_ROOT: {SRC_ROOT}")
    print(f"DST_ROOT: {DST_ROOT}")
    print(f"Patch size: {PATCH_H} x {PATCH_W}")
    print(f"Train patches/image: {TRAIN_PATCHES_PER_IMAGE}")
    print(f"Train augmentations/patch: {TRAIN_AUGS_PER_PATCH}")
    print(f"Val patches/image: {VAL_PATCHES_PER_IMAGE}")
    print(f"Val augmentations/patch: {VAL_AUGS_PER_PATCH}")
    print(f"Output mask values: background={OUTPUT_BACKGROUND_VALUE}, foreground={OUTPUT_FOREGROUND_VALUE}")
    print("============================================================\n")

    train_aug, train_stat = process_split(
        split_name="train",
        prefixes=train_list,
        image_out_dir=str(image_out_dir),
        mask_out_dir=str(mask_out_dir),
        patches_per_image=TRAIN_PATCHES_PER_IMAGE,
        augs_per_patch=TRAIN_AUGS_PER_PATCH,
        keep_identity=TRAIN_KEEP_IDENTITY,
    )

    val_aug, val_stat = process_split(
        split_name="val",
        prefixes=val_list,
        image_out_dir=str(image_out_dir),
        mask_out_dir=str(mask_out_dir),
        patches_per_image=VAL_PATCHES_PER_IMAGE,
        augs_per_patch=VAL_AUGS_PER_PATCH,
        keep_identity=VAL_KEEP_IDENTITY,
    )

    train_out_txt = os.path.join(set_out_dir, "train.txt")
    val_out_txt = os.path.join(set_out_dir, "val.txt")

    write_list(train_out_txt, train_aug)
    write_list(val_out_txt, val_aug)

    print("\n============================================================")
    print("Finished")
    print("============================================================")
    print_stat("train", train_stat, len(train_list), len(train_aug))
    print_stat("val", val_stat, len(val_list), len(val_aug))

    print("\nOutput dataset structure:")
    print(f"{DST_ROOT}/{VOC_YEAR}/{DST_IMAGE_DIR}/")
    print(f"{DST_ROOT}/{VOC_YEAR}/{DST_MASK_DIR}/")
    print(f"{DST_ROOT}/{VOC_YEAR}/{DST_IMAGE_SET_DIR}/train.txt")
    print(f"{DST_ROOT}/{VOC_YEAR}/{DST_IMAGE_SET_DIR}/val.txt")

    print("\nImportant:")
    print("1. Output masks are binary masks: 0 = background, 1 = crown.")
    print("2. This script does not write height maps.")
    print("3. In the training script, set:")
    print(f"   VOC_PATH = r'{DST_ROOT}'")
    print(f"   IMAGE_DIR_NAME = '{DST_IMAGE_DIR}'")
    print(f"   IMAGE_SUFFIX = '{IMAGE_SUFFIX}'")
    print("   TARGET_LABEL_VALUE = 1")
    print("   MASK_IGNORE_VALUE = None")
    print("   AUTO_INFER_IGNORE_255 = False")


if __name__ == "__main__":
    main()