import os
import cv2
import numpy as np


def tanaka_encrypt_block_scramble(img, key=8888, block_size=4):

    if img is None:
        raise ValueError("Không đọc được ảnh đầu vào.")

    gray_input = False
    if img.ndim == 2:
        img = img[:, :, None]
        gray_input = True

    h, w, c = img.shape

    new_h = (h // block_size) * block_size
    new_w = (w // block_size) * block_size

    if new_h == 0 or new_w == 0:
        raise ValueError(
            f"Ảnh quá nhỏ so với block_size={block_size}, shape={img.shape}"
        )

    img_crop = img[:new_h, :new_w].copy()

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

    rng = np.random.default_rng(seed=key)
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
