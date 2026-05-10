import os
import cv2
import zlib
import numpy as np
from pathlib import Path

PERM_TABLE = np.array([
    [0, 1, 2],  # RGB
    [0, 2, 1],  # RBG
    [1, 0, 2],  # GRB
    [1, 2, 0],  # GBR
    [2, 0, 1],  # BRG
    [2, 1, 0],  # BGR
], dtype=np.int64)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    arr = img.astype(np.float32)
    mn, mx = arr.min(), arr.max()

    if mx <= mn:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - mn) / (mx - mn) * 255.0
    return np.rint(arr).clip(0, 255).astype(np.uint8)


def _read_as_rgb(input_file: Path) -> np.ndarray:
    img = cv2.imread(str(input_file), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {input_file}")

    img = _to_uint8(img)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"Định dạng ảnh không hỗ trợ: shape={img.shape}, dtype={img.dtype}")

    return img


def _save_rgb(output_file: Path, img_rgb: np.ndarray) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(output_file), img_bgr)
    if not ok:
        raise IOError(f"Không ghi được ảnh ra: {output_file}")


def _negative_positive_transform(img_rgb: np.ndarray, bits: int, rng: np.random.Generator) -> np.ndarray:
    if bits != 8:
        raise ValueError("Code này thiết kế cho ảnh 8-bit, nên bits nên là 8.")

    h, w, c = img_rgb.shape
    xor_value = (1 << bits) - 1  # 255

    knp = rng.integers(0, 2, size=(h, w, c), dtype=np.uint8)
    out = np.bitwise_xor(img_rgb, knp * xor_value)
    return out.astype(np.uint8)


def _color_shuffle(img_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w, _ = img_rgb.shape
    kcs = rng.integers(0, 6, size=(h, w))
    perms = PERM_TABLE[kcs]
    out = np.take_along_axis(img_rgb, perms, axis=2)
    return out.astype(np.uint8)


def _block_statistical_smoothing(img_rgb: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size < 1:
        raise ValueError("block_size phải >= 1")

    h, w, _ = img_rgb.shape
    out = img_rgb.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = out[y:y + block_size, x:x + block_size]

            kss = int(rng.integers(0, 4))  # 0 mean, 1 median, 2 max, 3 min

            if kss == 0:
                value = np.rint(block.mean(axis=(0, 1))).clip(0, 255).astype(np.uint8)
            elif kss == 1:
                value = np.rint(np.median(block, axis=(0, 1))).clip(0, 255).astype(np.uint8)
            elif kss == 2:
                value = block.max(axis=(0, 1)).astype(np.uint8)
            else:
                value = block.min(axis=(0, 1)).astype(np.uint8)

            block[:] = value.reshape(1, 1, 3)

    return out


def encrypt_huang2022_array(
    img_rgb: np.ndarray,
    block_size: int = 4,
    bits: int = 8,
    seed: int | None = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    img_rgb = _to_uint8(img_rgb)
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb phải có shape (H, W, 3)")

    out = _negative_positive_transform(img_rgb, bits=bits, rng=rng)
    out = _color_shuffle(out, rng=rng)
    out = _block_statistical_smoothing(out, block_size=block_size, rng=rng)

    return out.astype(np.uint8)


def _make_per_image_seed(rel_path: Path, base_seed: int | None):
    if base_seed is None:
        return None
    rel_str = rel_path.as_posix()
    crc = zlib.crc32(rel_str.encode("utf-8")) & 0xFFFFFFFF
    return (int(base_seed) + crc) % (2**32)