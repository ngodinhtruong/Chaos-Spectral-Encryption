import os
import cv2
import numpy as np
from pathlib import Path



def negative_positive_transform(img_rgb, rng):
    h, w, c = img_rgb.shape
    negpos_mask = rng.random((h, w, 3)) >= 0.5

    out = img_rgb.copy()
    out[negpos_mask] = np.bitwise_xor(out[negpos_mask], 255).astype(np.uint8)
    return out



def channel_shuffle(img_rgb, rng):
    perms = np.array([
        [0, 1, 2],  # RGB
        [0, 2, 1],  # RBG
        [1, 0, 2],  # GRB
        [1, 2, 0],  # GBR
        [2, 0, 1],  # BRG
        [2, 1, 0],  # BGR
    ], dtype=np.int64)

    h, w, _ = img_rgb.shape
    shuffle_idx = rng.integers(0, 6, size=(h, w))
    selected_perms = perms[shuffle_idx]   # (h, w, 3)

    out = np.take_along_axis(img_rgb, selected_perms, axis=2)
    return out


def block_scramble(img_rgb, rng, block_size=4):
    if img_rgb.ndim == 2:
        img_rgb = img_rgb[:, :, None]
        gray_input = True
    else:
        gray_input = False

    h, w, c = img_rgb.shape

    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size

    if new_h == 0 or new_w == 0:
        raise ValueError(
            f"Ảnh quá nhỏ cho block_size={block_size}, shape={img_rgb.shape}"
        )

    img_crop = img_rgb[:new_h, :new_w].copy()

    nbh = new_h // block_size
    nbw = new_w // block_size
    num_blocks = nbh * nbw

    blocks = []
    for by in range(nbh):
        for bx in range(nbw):
            y0 = by * block_size
            x0 = bx * block_size
            block = img_crop[y0:y0 + block_size, x0:x0 + block_size, :]
            blocks.append(block)

    perm = rng.permutation(num_blocks)

    enc = np.zeros_like(img_crop)
    for out_idx, src_idx in enumerate(perm):
        out_by = out_idx // nbw
        out_bx = out_idx % nbw
        y0 = out_by * block_size
        x0 = out_bx * block_size
        enc[y0:y0 + block_size, x0:x0 + block_size, :] = blocks[src_idx]

    if gray_input:
        enc = enc[:, :, 0]

    return enc

def skk_encrypt_final(img_rgb, seed=8888, block_size=4):
    rng = np.random.default_rng(seed)

    x = negative_positive_transform(img_rgb, rng)
    x = channel_shuffle(x, rng)
    x = block_scramble(x, rng, block_size=block_size)

    return x