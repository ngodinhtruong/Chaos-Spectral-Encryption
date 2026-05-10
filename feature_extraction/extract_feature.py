from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = Path(path).parent.name
        return image, label


def extract_features(loader, model, model_type, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    features = []
    labels = []

    with torch.no_grad():
        for imgs, batch_labels in tqdm(loader, desc="Extract features", leave=False):
            imgs = imgs.to(device, non_blocking=True)

            if model_type == "clip":
                feat = model.encode_image(imgs)
            elif model_type == "dino":
                feat = model(imgs)
            elif model_type == "resnet":
                feat = model(imgs).squeeze()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            feat = feat / feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy())
            labels.extend(batch_labels)

    features = np.concatenate(features, axis=0)
    return features, np.array(labels)


def _path_to_label(path):
    return Path(path).parent.name


def _open_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _hsv_feature(path):
    image = _open_rgb(path)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1, 2],
        mask=None,
        histSize=[8, 8, 8],
        ranges=[0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32), _path_to_label(path)


def _hog_feature(path):
    image = _open_rgb(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    descriptor = hog.compute(resized)
    return descriptor.squeeze().astype(np.float32), _path_to_label(path)


def _run_parallel(fn, paths, max_workers):
    paths = list(paths)
    if len(paths) == 0:
        return np.zeros((0, 0), dtype=np.float32), np.array([]), []

    if max_workers is None or max_workers <= 1:
        results = [fn(path) for path in tqdm(paths, desc=f"Extract {fn.__name__}")]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(fn, paths), total=len(paths), desc=f"Extract {fn.__name__}"))

    features, labels = zip(*results)
    features = np.vstack(features)
    labels = np.array(labels)
    classes = sorted(set(labels.tolist()))
    return features, labels, classes


def extract_hsv_from_paths(paths, max_workers=8):
    return _run_parallel(_hsv_feature, paths, max_workers)


def extract_hog_from_paths(paths, max_workers=8):
    return _run_parallel(_hog_feature, paths, max_workers)
