import os
import random
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasterio
from PIL import Image


# ============================================================
# Suppress raster/tiff warnings
# ============================================================
logging.getLogger("rasterio").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r".*NotGeoreferencedWarning.*")


# ============================================================
# Manual configuration
# ============================================================
SRC_ROOT = "TreeHeightDataset"
DST_ROOT = "TreeHeightDataset_AUG"

MS_DIR = "RSImages"
MASK_DIR = "mask"
H_DIR = "heights"

MS_SUFFIX = ".tif"
MASK_SUFFIX = ".png"
H_SUFFIX = "_processed.tif"

PATCH_H = 640
PATCH_W = 640

# Training patches
TRAIN_PATCHES_PER_IMAGE = 4
TRAIN_AUGS_PER_PATCH = 6
TRAIN_KEEP_IDENTITY = True

# Validation patches
# 建议验证集不做增强，只裁剪 patch，避免验证集被人为扩增。
VAL_PATCHES_PER_IMAGE = 4
VAL_AUGS_PER_PATCH = 0
VAL_KEEP_IDENTITY = True

# Empty patch filtering
SKIP_EMPTY_PATCH = True
EMPTY_THRESH = 0.001

# When random crop gets empty patch, retry several times
MAX_CROP_TRIES = 30

# Mask conversion
# None = any non-zero value is foreground.
# If source foreground is exactly 38, set SOURCE_FOREGROUND_VALUE = 38.
SOURCE_FOREGROUND_VALUE = None

# If source has ignore value, set it here.
# For height regression, usually keep it as None unless you are sure.
SOURCE_IGNORE_VALUE = None

# Output mask value
# Important: save foreground as 1, not 255.
OUTPUT_BACKGROUND_VALUE = 0
OUTPUT_FOREGROUND_VALUE = 1

# Height NoData
H_NODATA = -9999.0
NODATA_ABS_THRESHOLD = 1.0e20

# GeoTIFF compression
# None = no compression, safer for tifffile reading.
TIFF_COMPRESS = None

# Georeference
# For crop + flip + rotation augmentation, original transform is no longer reliable.
# Keep False for training patches.
KEEP_GEOREFERENCE = False

SEED = 11
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Directory and txt utilities
# ============================================================
def ensure_dirs(root):
    Path(root, MS_DIR).mkdir(parents=True, exist_ok=True)
    Path(root, MASK_DIR).mkdir(parents=True, exist_ok=True)
    Path(root, H_DIR).mkdir(parents=True, exist_ok=True)


def read_list(txt_path):
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
    name = str(name)
    name = name.replace("（", "(").replace("）", ")")

    illegal_chars = '<>:"/\\|?*'

    for char in illegal_chars:
        name = name.replace(char, "_")

    return name.rstrip(" .")


# ============================================================
# Reading utilities
# ============================================================
def read_multispectral(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"MS image not found: {path}")

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()

    return arr, profile


def read_height(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Height tif not found: {path}")

    with rasterio.open(path) as src:
        height = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        height[np.isclose(height, float(nodata))] = np.nan

    height[np.abs(height) >= NODATA_ABS_THRESHOLD] = np.nan
    height[~np.isfinite(height)] = np.nan

    return height.astype(np.float32), profile


def read_mask(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = Image.open(path)

    if mask.mode == "P":
        arr = np.array(mask, dtype=np.uint8)
    else:
        arr = np.array(mask.convert("L"), dtype=np.uint8)

    return arr


def convert_mask_to_binary(mask_raw):
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


# ============================================================
# Saving utilities
# ============================================================
def build_tif_profile(count, height, width, dtype="float32", ref_profile=None, nodata=None):
    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": int(count),
        "dtype": dtype,
    }

    if TIFF_COMPRESS:
        profile["compress"] = TIFF_COMPRESS

    if nodata is not None:
        profile["nodata"] = nodata

    if KEEP_GEOREFERENCE and ref_profile is not None:
        if ref_profile.get("crs", None) is not None:
            profile["crs"] = ref_profile["crs"]

        if ref_profile.get("transform", None) is not None:
            profile["transform"] = ref_profile["transform"]

    return profile


def save_ms_tif(path, ms_c_hw, ref_profile=None):
    ms_c_hw = np.ascontiguousarray(ms_c_hw.astype(np.float32))
    channels, height, width = ms_c_hw.shape

    profile = build_tif_profile(
        count=channels,
        height=height,
        width=width,
        dtype="float32",
        ref_profile=ref_profile,
        nodata=None,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(ms_c_hw)


def save_height_tif(path, height_hw, ref_profile=None):
    height_hw = height_hw.astype(np.float32).copy()
    height_hw[np.isnan(height_hw)] = H_NODATA

    height_hw = np.ascontiguousarray(height_hw)

    height, width = height_hw.shape

    profile = build_tif_profile(
        count=1,
        height=height,
        width=width,
        dtype="float32",
        ref_profile=ref_profile,
        nodata=H_NODATA,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(height_hw, 1)


def save_mask_png(path, mask_hw):
    """
    Save mask as 0/1 PNG.

    0 = background
    1 = crown
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


def apply_discrete_geom(ms_c_hw, mask_hw, height_hw, mode):
    if mode == "identity":
        return (
            np.ascontiguousarray(ms_c_hw),
            np.ascontiguousarray(mask_hw),
            np.ascontiguousarray(height_hw),
        )

    if mode == "hflip":
        return (
            np.ascontiguousarray(ms_c_hw[:, :, ::-1]),
            np.ascontiguousarray(mask_hw[:, ::-1]),
            np.ascontiguousarray(height_hw[:, ::-1]),
        )

    if mode == "vflip":
        return (
            np.ascontiguousarray(ms_c_hw[:, ::-1, :]),
            np.ascontiguousarray(mask_hw[::-1, :]),
            np.ascontiguousarray(height_hw[::-1, :]),
        )

    if mode == "rot90":
        return (
            np.ascontiguousarray(np.rot90(ms_c_hw, 1, axes=(1, 2))),
            np.ascontiguousarray(np.rot90(mask_hw, 1)),
            np.ascontiguousarray(np.rot90(height_hw, 1)),
        )

    if mode == "rot180":
        return (
            np.ascontiguousarray(np.rot90(ms_c_hw, 2, axes=(1, 2))),
            np.ascontiguousarray(np.rot90(mask_hw, 2)),
            np.ascontiguousarray(np.rot90(height_hw, 2)),
        )

    if mode == "rot270":
        return (
            np.ascontiguousarray(np.rot90(ms_c_hw, 3, axes=(1, 2))),
            np.ascontiguousarray(np.rot90(mask_hw, 3)),
            np.ascontiguousarray(np.rot90(height_hw, 3)),
        )

    raise ValueError(f"Unsupported mode: {mode}")


def choose_aug_modes(augs_per_patch):
    if augs_per_patch <= 0:
        return []

    if augs_per_patch <= len(DISCRETE_MODES):
        return random.sample(DISCRETE_MODES, augs_per_patch)

    modes = []

    while len(modes) < augs_per_patch:
        modes.extend(random.sample(DISCRETE_MODES, len(DISCRETE_MODES)))

    return modes[:augs_per_patch]


# ============================================================
# Crop sampling
# ============================================================
def random_crop_position(height, width):
    top = random.randint(0, height - PATCH_H)
    left = random.randint(0, width - PATCH_W)
    return top, left


def center_crop_position(height, width):
    top = max(0, (height - PATCH_H) // 2)
    left = max(0, (width - PATCH_W) // 2)
    return top, left


def grid_crop_positions(height, width, num_patches):
    """
    Deterministic validation crop positions.
    """
    candidates = [
        center_crop_position(height, width),
        (0, 0),
        (0, max(0, width - PATCH_W)),
        (max(0, height - PATCH_H), 0),
        (max(0, height - PATCH_H), max(0, width - PATCH_W)),
    ]

    unique = []
    seen = set()

    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)

    if len(unique) >= num_patches:
        return unique[:num_patches]

    # Fill more deterministic positions if needed
    rows = np.linspace(0, height - PATCH_H, num=max(1, int(np.ceil(np.sqrt(num_patches))))).astype(int)
    cols = np.linspace(0, width - PATCH_W, num=max(1, int(np.ceil(np.sqrt(num_patches))))).astype(int)

    for r in rows:
        for c in cols:
            item = (int(r), int(c))
            if item not in seen:
                unique.append(item)
                seen.add(item)
            if len(unique) >= num_patches:
                return unique[:num_patches]

    return unique[:num_patches]


def crop_patch(ms_c_hw, mask_hw, height_hw, top, left):
    ms_patch = ms_c_hw[:, top: top + PATCH_H, left: left + PATCH_W]
    mask_patch = mask_hw[top: top + PATCH_H, left: left + PATCH_W]
    height_patch = height_hw[top: top + PATCH_H, left: left + PATCH_W]

    return ms_patch, mask_patch, height_patch


def sample_train_patch(ms_c_hw, mask_hw, height_hw):
    """
    Randomly sample a non-empty training patch.
    """
    height, width = mask_hw.shape

    last_patch = None
    last_ratio = 0.0

    for _ in range(MAX_CROP_TRIES):
        top, left = random_crop_position(height, width)

        patch = crop_patch(ms_c_hw, mask_hw, height_hw, top, left)

        _, mask_patch, height_patch = patch

        valid_height = np.isfinite(height_patch)
        foreground_ratio = float(((mask_patch > 0) & valid_height).mean())

        last_patch = patch
        last_ratio = foreground_ratio

        if not SKIP_EMPTY_PATCH:
            return patch, foreground_ratio

        if foreground_ratio >= EMPTY_THRESH:
            return patch, foreground_ratio

    return last_patch, last_ratio


# ============================================================
# Split augmentation
# ============================================================
def augment_split(
    split_name,
    prefixes,
    patches_per_image,
    augs_per_patch,
    keep_identity=True,
    is_train=True,
):
    out_prefixes = []

    stat = defaultdict(int)
    stat["images_total"] = len(prefixes)

    for raw in prefixes:
        prefix = safe_name(os.path.splitext(str(raw).strip())[0])

        ms_path = os.path.join(SRC_ROOT, MS_DIR, prefix + MS_SUFFIX)
        mask_path = os.path.join(SRC_ROOT, MASK_DIR, prefix + MASK_SUFFIX)
        height_path = os.path.join(SRC_ROOT, H_DIR, prefix + H_SUFFIX)

        if not os.path.exists(ms_path):
            print(f"[{split_name}] skip, MS missing: {ms_path}")
            stat["ms_missing"] += 1
            continue

        if not os.path.exists(mask_path):
            print(f"[{split_name}] skip, mask missing: {mask_path}")
            stat["mask_missing"] += 1
            continue

        if not os.path.exists(height_path):
            print(f"[{split_name}] skip, height missing: {height_path}")
            stat["height_missing"] += 1
            continue

        ms, ms_profile = read_multispectral(ms_path)
        height, height_profile = read_height(height_path)

        mask_raw = read_mask(mask_path)
        mask = convert_mask_to_binary(mask_raw)

        common_h = min(ms.shape[1], mask.shape[0], height.shape[0])
        common_w = min(ms.shape[2], mask.shape[1], height.shape[1])

        ms = ms[:, :common_h, :common_w]
        mask = mask[:common_h, :common_w]
        height = height[:common_h, :common_w]

        if common_h < PATCH_H or common_w < PATCH_W:
            print(f"[{split_name}] skip, too small: {prefix} ({common_h}x{common_w})")
            stat["too_small"] += 1
            continue

        stat["images_used"] += 1

        if is_train:
            patch_jobs = []

            for patch_id in range(patches_per_image):
                stat["patch_try"] += 1

                patch, foreground_ratio = sample_train_patch(ms, mask, height)

                if SKIP_EMPTY_PATCH and foreground_ratio < EMPTY_THRESH:
                    stat["patch_skipped_empty"] += 1
                    continue

                patch_jobs.append((patch_id, patch, foreground_ratio))

        else:
            patch_jobs = []

            positions = grid_crop_positions(common_h, common_w, patches_per_image)

            for patch_id, (top, left) in enumerate(positions):
                stat["patch_try"] += 1

                patch = crop_patch(ms, mask, height, top, left)
                _, mask_patch, height_patch = patch

                valid_height = np.isfinite(height_patch)
                foreground_ratio = float(((mask_patch > 0) & valid_height).mean())

                if SKIP_EMPTY_PATCH and foreground_ratio < EMPTY_THRESH:
                    stat["patch_skipped_empty"] += 1
                    continue

                patch_jobs.append((patch_id, patch, foreground_ratio))

        for patch_id, patch, foreground_ratio in patch_jobs:
            ms_patch, mask_patch, height_patch = patch

            stat["patch_kept"] += 1

            if keep_identity:
                new_prefix = f"{split_name}_{prefix}_p{patch_id:03d}_id"

                save_ms_tif(
                    os.path.join(DST_ROOT, MS_DIR, new_prefix + MS_SUFFIX),
                    ms_patch,
                    ref_profile=ms_profile,
                )

                save_mask_png(
                    os.path.join(DST_ROOT, MASK_DIR, new_prefix + MASK_SUFFIX),
                    mask_patch,
                )

                save_height_tif(
                    os.path.join(DST_ROOT, H_DIR, new_prefix + H_SUFFIX),
                    height_patch,
                    ref_profile=height_profile,
                )

                out_prefixes.append(new_prefix)
                stat["samples_written"] += 1

            aug_modes = choose_aug_modes(augs_per_patch)

            for aug_id, mode in enumerate(aug_modes):
                ms_aug, mask_aug, height_aug = apply_discrete_geom(
                    ms_patch,
                    mask_patch,
                    height_patch,
                    mode,
                )

                new_prefix = f"{split_name}_{prefix}_p{patch_id:03d}_{mode}_a{aug_id:02d}"

                save_ms_tif(
                    os.path.join(DST_ROOT, MS_DIR, new_prefix + MS_SUFFIX),
                    ms_aug,
                    ref_profile=ms_profile,
                )

                save_mask_png(
                    os.path.join(DST_ROOT, MASK_DIR, new_prefix + MASK_SUFFIX),
                    mask_aug,
                )

                save_height_tif(
                    os.path.join(DST_ROOT, H_DIR, new_prefix + H_SUFFIX),
                    height_aug,
                    ref_profile=height_profile,
                )

                out_prefixes.append(new_prefix)
                stat["samples_written"] += 1

        print(f"[{split_name}] done: {prefix}")

    return out_prefixes, dict(stat)


def print_stat(split_name, stat, base_count, output_count):
    print(f"\n[{split_name}] statistics")
    print(f"Original entries:       {base_count}")
    print(f"Output entries:         {output_count}")
    print(f"Images total:           {stat.get('images_total', 0)}")
    print(f"Images used:            {stat.get('images_used', 0)}")
    print(f"MS missing:             {stat.get('ms_missing', 0)}")
    print(f"Mask missing:           {stat.get('mask_missing', 0)}")
    print(f"Height missing:         {stat.get('height_missing', 0)}")
    print(f"Too small:              {stat.get('too_small', 0)}")
    print(f"Patch tried:            {stat.get('patch_try', 0)}")
    print(f"Patch kept:             {stat.get('patch_kept', 0)}")
    print(f"Patch skipped empty:    {stat.get('patch_skipped_empty', 0)}")
    print(f"Samples written:        {stat.get('samples_written', 0)}")

    if stat.get("patch_try", 0) > 0:
        kept_ratio = stat.get("patch_kept", 0) / max(1, stat.get("patch_try", 0))
        print(f"Patch kept ratio:       {kept_ratio:.4f}")

    if base_count > 0:
        print(f"Expansion ratio:        {output_count / base_count:.2f}x")


# ============================================================
# Main
# ============================================================
def main():
    ensure_dirs(DST_ROOT)

    train_txt = os.path.join(SRC_ROOT, "train.txt")
    val_txt = os.path.join(SRC_ROOT, "val.txt")

    train_list = read_list(train_txt)
    val_list = read_list(val_txt)

    print("============================================================")
    print("8-band image + mask + height augmentation")
    print("============================================================")
    print(f"SRC_ROOT: {SRC_ROOT}")
    print(f"DST_ROOT: {DST_ROOT}")
    print(f"Patch size: {PATCH_H} x {PATCH_W}")
    print(f"Train patches/image: {TRAIN_PATCHES_PER_IMAGE}")
    print(f"Train augmentations/patch: {TRAIN_AUGS_PER_PATCH}")
    print(f"Val patches/image: {VAL_PATCHES_PER_IMAGE}")
    print(f"Val augmentations/patch: {VAL_AUGS_PER_PATCH}")
    print(f"Mask output: background={OUTPUT_BACKGROUND_VALUE}, foreground={OUTPUT_FOREGROUND_VALUE}")
    print(f"Height nodata output: {H_NODATA}")
    print(f"Keep georeference: {KEEP_GEOREFERENCE}")
    print("============================================================\n")

    train_aug, train_stat = augment_split(
        split_name="train",
        prefixes=train_list,
        patches_per_image=TRAIN_PATCHES_PER_IMAGE,
        augs_per_patch=TRAIN_AUGS_PER_PATCH,
        keep_identity=TRAIN_KEEP_IDENTITY,
        is_train=True,
    )

    val_aug, val_stat = augment_split(
        split_name="val",
        prefixes=val_list,
        patches_per_image=VAL_PATCHES_PER_IMAGE,
        augs_per_patch=VAL_AUGS_PER_PATCH,
        keep_identity=VAL_KEEP_IDENTITY,
        is_train=False,
    )

    write_list(os.path.join(DST_ROOT, "train.txt"), train_aug)
    write_list(os.path.join(DST_ROOT, "val.txt"), val_aug)

    print("\n============================================================")
    print("Finished")
    print("============================================================")

    print_stat("train", train_stat, len(train_list), len(train_aug))
    print_stat("val", val_stat, len(val_list), len(val_aug))

    print("\nOutput dataset:")
    print(f"{DST_ROOT}/{MS_DIR}/")
    print(f"{DST_ROOT}/{MASK_DIR}/")
    print(f"{DST_ROOT}/{H_DIR}/")
    print(f"{DST_ROOT}/train.txt")
    print(f"{DST_ROOT}/val.txt")

    print("\nTraining setting:")
    print(f"dataset_path = r'{DST_ROOT}'")
    print(f"ms_dir = '{MS_DIR}'")
    print(f"mask_dir = '{MASK_DIR}'")
    print(f"height_dir = '{H_DIR}'")
    print(f"ms_suffix = '{MS_SUFFIX}'")
    print(f"mask_suffix = '{MASK_SUFFIX}'")
    print(f"height_suffix = '{H_SUFFIX}'")

    print("\nImportant:")
    print("1. Output masks are saved as 0/1, not 255.")
    print("2. Training split uses random crop + discrete augmentation.")
    print("3. Validation split uses deterministic crop and no augmentation by default.")
    print("4. No arbitrary rotation, scaling, shearing, or pixel-value perturbation is used.")
    print("5. Height NoData values are written as -9999.0.")


if __name__ == "__main__":
    main()