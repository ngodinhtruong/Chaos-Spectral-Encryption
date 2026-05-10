
import os
import cv2
import itertools
import numpy as np
from pathlib import Path

def check_image(img, name="img"):
    if img is None:
        raise ValueError(f"{name} is None. Make sure cv2.imread(...) loaded successfully.")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {img.shape}")

def generate_candidates(pixel):
    p0, p1, p2 = [int(x) for x in pixel]
    perms = np.array([
        [p0, p1, p2],
        [p0, p2, p1],
        [p1, p0, p2],
        [p1, p2, p0],
        [p2, p0, p1],
        [p2, p1, p0],
    ], dtype=np.int16)

    candidates = np.zeros((6, 8, 3), dtype=np.int16)

    for r in range(6):
        base = perms[r]
        for bf in range(8):
            b3 = bf % 2
            b2 = (bf // 2) % 2
            b1 = (bf // 4) % 2

            candidates[r, bf, 0] = base[0] if b1 == 0 else (base[0] ^ 255)
            candidates[r, bf, 1] = base[1] if b2 == 0 else (base[1] ^ 255)
            candidates[r, bf, 2] = base[2] if b3 == 0 else (base[2] ^ 255)

    return candidates

def attack_ciphertext_only_advanced(enc_img: np.ndarray, initial_pixel=(128, 128, 128)) -> np.ndarray:
    check_image(enc_img, "enc_img")
    h, w, _ = enc_img.shape
    dec = np.zeros_like(enc_img)
    dec[0, 0] = np.array(initial_pixel, dtype=np.uint8)

    for j in range(1, w):
        ref = dec[0, j - 1].astype(np.int16)
        cand = generate_candidates(enc_img[0, j])
        diff = np.abs(cand - ref).sum(axis=2)
        r, c = np.unravel_index(np.argmin(diff), diff.shape)
        dec[0, j] = cand[r, c].astype(np.uint8)

    for i in range(1, h):
        for j in range(w):
            ref = dec[i - 1, j].astype(np.int16)
            cand = generate_candidates(enc_img[i, j])
            diff = np.abs(cand - ref).sum(axis=2)
            r, c = np.unravel_index(np.argmin(diff), diff.shape)
            dec[i, j] = cand[r, c].astype(np.uint8)

    return dec