# ============================================================
# LPIPS from 2 image paths
# input_path  = raw/original image
# output_path = reconstructed / attacked image
# ============================================================

import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import lpips


def _load_img_rgb(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def _img_to_tensor_lpips(img_pil, device):
    """
    LPIPS expects tensor in [-1, 1], shape [1, 3, H, W]
    """
    arr = np.array(img_pil).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device)


def compute_lpips_from_paths(input_path, output_path, net='alex', device=None, resize_to_input=True):
    """
    Args:
        input_path:  path to raw/original image
        output_path: path to reconstructed/attacked image
        net:         'alex', 'vgg', or 'squeeze'
        device:      'cuda' / 'cpu' / None
        resize_to_input: if True, resize output image to match input image size

    Returns:
        lpips_score (float)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_ref = _load_img_rgb(input_path)
    img_out = _load_img_rgb(output_path)

    if resize_to_input and img_ref.size != img_out.size:
        img_out = img_out.resize(img_ref.size, Image.BILINEAR)

    x = _img_to_tensor_lpips(img_ref, device)
    y = _img_to_tensor_lpips(img_out, device)

    model = lpips.LPIPS(net=net).to(device)
    model.eval()

    with torch.no_grad():
        score = model(x, y).item()

    print("=" * 80)
    print("LPIPS evaluation finished")
    print(f"Reference image : {input_path}")
    print(f"Compared image  : {output_path}")
    print(f"Network         : {net}")
    print(f"Device          : {device}")
    print(f"LPIPS score     : {score:.6f}")
    print("=" * 80)

    return score


# =========================
# Example
# =========================
# lp = compute_lpips_from_paths(
#     input_path="/content/raw.png",
#     output_path="/content/reconstructed.png",
#     net="alex"
# )