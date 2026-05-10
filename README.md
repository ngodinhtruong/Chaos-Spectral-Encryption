# Chaos-Spectral-Encryption

A simple pipeline for image encryption and model training.

## Overview
This repo includes:
- preprocessing raw images to 224x224 PNG
- encrypting processed images
- training models with two flows:
  1. `pretrain` using `Pretrain_model.ipynb`
  2. `foundation` using `foundation_model_train.ipynb`

## Setup
1. Install Python 3.10+.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Step 1: Preprocess images
Use `process_data.py` to convert a raw dataset into normalized PNG images.

- Input: image folder dataset with one class per folder
- Output: processed PNG dataset

Example:

```python
from process_data import preprocess_split_mp

src_dir = "data/raw_dataset"
out_dir = "data/new_dataset_processed"
preprocess_split_mp(src_dir, out_dir, max_workers=8)
```

## Step 2: Encrypt images
The `encrypt/` folder contains example encryption functions:
- `encrypt/encryption.py`
- `encrypt/skk_enc.py`
- `encrypt/tanaka_enc.py`
- `encrypt/huangAtel_enc.py`

Use one of these functions to encrypt the processed image dataset and save the result to a new folder.

Simple encryption example:

```python
from pathlib import Path
from PIL import Image
import numpy as np
from encrypt.skk_enc import skk_encrypt_final

src_dir = Path("data/new_dataset_processed")
dst_dir = Path("data/new_dataset_encrypted")
dst_dir.mkdir(parents=True, exist_ok=True)

for img_path in src_dir.rglob("*.png"):
    rel_path = img_path.relative_to(src_dir)
    target_path = dst_dir / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    enc = skk_encrypt_final(img, seed=8888, block_size=4)
    Image.fromarray(enc).save(target_path)
```

## Step 3: Train models
### Pretrain flow
- Notebook: `Pretrain_model.ipynb`
- Support module: `models/pretrain_model.py`

This module includes:
- dataset utilities
- train/validation/test split
- transforms and normalization
- ResNet/VGG model builders
- optimizer, mixup, early stopping
- training and evaluation functions

### Foundation flow
- Notebook: `foundation_model_train.ipynb`
- Support module: `models/foundation_model.py`

This module includes loaders for:
- CLIP
- DINO
- ResNet50 feature extractor

## Recommended order
1. Preprocess raw data with `process_data.py`.
2. Encrypt processed data with a function from `encrypt/`.
3. Run `Pretrain_model.ipynb` for pretrain flow.
4. Run `foundation_model_train.ipynb` for foundation flow.

## Data structure example

```text
data/
  raw_dataset/
  new_dataset_processed/
  new_dataset_encrypted/
```
