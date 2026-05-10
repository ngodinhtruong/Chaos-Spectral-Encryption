import os
import numpy as np
import cv2
from PIL import Image


def logistic_map_permutation(n, seed, r=3.99):
    x = (seed % 1000) / 1000.0
    if x in [0, 0.25, 0.5, 0.75, 1]: x += 0.123

    for _ in range(100):
        x = r * x * (1 - x)

    sequence = []
    for _ in range(n):
        x = r * x * (1 - x)
        sequence.append(x)

    return np.argsort(sequence)

def dct_process_block(block, seed, flip_prob=0.10, noise_scale=1.0):
    block_f = np.float32(block)
    dct = cv2.dct(block_f)
    h, w = dct.shape

    rng = np.random.default_rng(seed)

    # vùng high-frequency
    hf = (slice(h//2, h), slice(w//2, w))

    # đảo dấu xác suất nhỏ
    flips = rng.choice([-1.0, 1.0], size=dct[hf].shape, p=[flip_prob, 1 - flip_prob]).astype(np.float32)
    dct[hf] *= flips

    # thêm nhiễu Gaussian nhỏ trong miền DCT
    dct[hf] += rng.normal(0, noise_scale, size=dct[hf].shape).astype(np.float32)

    idct = cv2.idct(dct)
    return idct


def chaos_spectral_scramble(img, block_size, seed):
    H, W, C = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size
    n_blocks = h_blocks * w_blocks

    # Cắt ảnh thành các khối
    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            blocks.append(img[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size])

    perm = logistic_map_permutation(len(blocks), seed)

    shuffled = np.zeros_like(img, dtype=np.float32)
    idx = 0

    for i in range(h_blocks):
        for j in range(w_blocks):
            target_block = blocks[perm[idx]]

            processed_block = np.zeros_like(target_block, dtype=np.float32)
            for c in range(C):
                processed_block[:, :, c] = dct_process_block(target_block[:, :, c], seed + idx + c)

            shuffled[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ] = processed_block

            idx += 1

    return np.clip(shuffled, 0, 255).astype(np.uint8)



def advanced_learnable_encrypt(img, k=3, block_xor=4, block_shuffle=16, seed=2024):
    img = img.astype(np.uint8)
    H, W, _ = img.shape

    msb_mask = ((1 << k) - 1) << (8 - k)
    lsb_mask = (1 << (8 - k)) - 1

    msb = (img & msb_mask) >> (8 - k)
    lsb = img & lsb_mask
    enc = np.zeros_like(img)

    for c in range(3):
        c2 = (c + 1) % 3
        m = msb[:, :, c].copy()
        m[:-1, :] ^= msb[1:, :, c]
        m[:, :-1] ^= msb[:, 1:, c]
        m ^= msb[:, :, c2]
        enc[:, :, c] = (m << (8 - k)) | lsb[:, :, c]

    for i in range(0, H, block_xor):
        for j in range(0, W, block_xor):
            if ((i // block_xor) + (j // block_xor)) % 2 == 1:
                lsb_part = enc[i:i+block_xor, j:j+block_xor] & lsb_mask
                msb_part = enc[i:i+block_xor, j:j+block_xor] & msb_mask
                enc[i:i+block_xor, j:j+block_xor] = msb_part | (lsb_mask - lsb_part)


    return chaos_spectral_scramble(enc, block_shuffle, seed)

