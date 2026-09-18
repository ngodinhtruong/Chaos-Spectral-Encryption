from encryption_proposed import *



def msb_lsb_only(img, k=3):
    img = img.astype(np.uint8)

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

    return enc.astype(np.uint8)


def msb_lsb_block_xor(img, k=3, block_xor=4):
    img = img.astype(np.uint8)
    H, W, _ = img.shape

    msb_mask = ((1 << k) - 1) << (8 - k)
    lsb_mask = (1 << (8 - k)) - 1

    enc = msb_lsb_only(img, k=k)

    for i in range(0, H, block_xor):
        for j in range(0, W, block_xor):
            if ((i // block_xor) + (j // block_xor)) % 2 == 1:
                lsb_part = enc[i:i+block_xor, j:j+block_xor] & lsb_mask
                msb_part = enc[i:i+block_xor, j:j+block_xor] & msb_mask
                enc[i:i+block_xor, j:j+block_xor] = msb_part | (lsb_mask - lsb_part)

    return enc.astype(np.uint8)


def random_block_shuffle(img, block_size=16, seed=2024):
    H, W, C = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size

    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            blocks.append(img[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(blocks))

    shuffled = np.zeros_like(img)
    idx = 0

    for i in range(h_blocks):
        for j in range(w_blocks):
            shuffled[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ] = blocks[perm[idx]]
            idx += 1

    return shuffled.astype(np.uint8)


def chaos_shuffle_only(img, block_size=16, seed=2024):
    H, W, C = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size

    blocks = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            blocks.append(img[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ])

    perm = logistic_map_permutation(len(blocks), seed)

    shuffled = np.zeros_like(img)
    idx = 0

    for i in range(h_blocks):
        for j in range(w_blocks):
            shuffled[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ] = blocks[perm[idx]]
            idx += 1

    return shuffled.astype(np.uint8)


def dct_only(img, block_size=16, seed=2024):
    H, W, C = img.shape
    h_blocks = H // block_size
    w_blocks = W // block_size

    out = img.copy().astype(np.float32)

    idx = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = img[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ]

            processed = np.zeros_like(block, dtype=np.float32)

            for c in range(C):
                processed[:, :, c] = dct_process_block(
                    block[:, :, c],
                    seed + idx + c
                )

            out[
                i*block_size:(i+1)*block_size,
                j*block_size:(j+1)*block_size
            ] = processed

            idx += 1

    return np.clip(out, 0, 255).astype(np.uint8)



def ablation_encrypt(img, mode, k=3, block_xor=4, block_shuffle=16, seed=2024):
    if mode == "original":
        return img.astype(np.uint8)

    elif mode == "msb_lsb_only":
        return msb_lsb_only(img, k=k)

    elif mode == "msb_lsb_block_xor":
        return msb_lsb_block_xor(
            img,
            k=k,
            block_xor=block_xor
        )

    elif mode == "msb_lsb_random_shuffle":
        enc = msb_lsb_block_xor(
            img,
            k=k,
            block_xor=block_xor
        )
        return random_block_shuffle(
            enc,
            block_size=block_shuffle,
            seed=seed
        )

    elif mode == "msb_lsb_chaos_shuffle":
        enc = msb_lsb_block_xor(
            img,
            k=k,
            block_xor=block_xor
        )
        return chaos_shuffle_only(
            enc,
            block_size=block_shuffle,
            seed=seed
        )

    elif mode == "msb_lsb_dct_only":
        enc = msb_lsb_block_xor(
            img,
            k=k,
            block_xor=block_xor
        )
        return dct_only(
            enc,
            block_size=block_shuffle,
            seed=seed
        )

    elif mode == "full":
        return advanced_learnable_encrypt(
            img,
            k=k,
            block_xor=block_xor,
            block_shuffle=block_shuffle,
            seed=seed
        )

    else:
        raise ValueError(f"Unknown ablation mode: {mode}")