
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

def compute_psnr_between_two_images(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Tính PSNR giữa 2 ảnh numpy uint8 hoặc float.
    img1, img2: shape (H, W, C)
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Shape không khớp: {img1.shape} vs {img2.shape}")

    img1 = np.clip(img1.astype(np.float32) / 255.0, 0.0, 1.0)
    img2 = np.clip(img2.astype(np.float32) / 255.0, 0.0, 1.0)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    return 10.0 * math.log10(1.0 / mse)

def compute_psnr_from_paths(path1: str, path2: str, image_size: int = None) -> float:
    img1 = load_image_as_array(path1, image_size=image_size)
    img2 = load_image_as_array(path2, image_size=image_size)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return compute_psnr_between_two_images(img1, img2)



