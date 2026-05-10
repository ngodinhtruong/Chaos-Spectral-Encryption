
import math
import numpy as np
import cv2

def load_image_as_array(path: str, image_size: int = None) -> np.ndarray:
    """
    Đọc ảnh bằng cv2, trả về RGB numpy array uint8
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))

    return img


def avg_pool_3x3_same(img: np.ndarray) -> np.ndarray:
    """
    Average pooling 3x3, stride 1, same padding
    img: (H, W, C)
    """
    h, w, c = img.shape
    padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            patch = padded[i:i+3, j:j+3, :]
            out[i, j, :] = np.mean(patch, axis=(0, 1))

    return out


def compute_ssim_between_two_images(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Tính SSIM giữa 2 ảnh numpy.
    img1, img2: shape (H, W, C), uint8 hoặc float
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape không khớp: {img1.shape} vs {img2.shape}")

    img1 = np.clip(img1.astype(np.float32) / 255.0, 0.0, 1.0)
    img2 = np.clip(img2.astype(np.float32) / 255.0, 0.0, 1.0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = avg_pool_3x3_same(img1)
    mu2 = avg_pool_3x3_same(img2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = avg_pool_3x3_same(img1 * img1) - mu1_sq
    sigma2_sq = avg_pool_3x3_same(img2 * img2) - mu2_sq
    sigma12 = avg_pool_3x3_same(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return float(np.mean(ssim_map))

def compute_ssim_from_paths(path1: str, path2: str, image_size: int = None) -> float:
    img1 = load_image_as_array(path1, image_size=image_size)
    img2 = load_image_as_array(path2, image_size=image_size)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return compute_ssim_between_two_images(img1, img2)