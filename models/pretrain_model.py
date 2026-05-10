import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.models as models

from tqdm.auto import tqdm


NUM_CLASSES = 3


def compute_mean_std(dataset_root, indices, image_size=224, batch_size=128, num_workers=4):
    """Compute channel mean and std over a subset of ImageFolder images."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(dataset_root, transform=transform)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    mean = torch.zeros(3)
    sq_mean = torch.zeros(3)
    total = 0

    for images, _ in loader:
        num_pixels = images.shape[0] * images.shape[2] * images.shape[3]
        mean += images.sum(dim=[0, 2, 3])
        sq_mean += (images ** 2).sum(dim=[0, 2, 3])
        total += num_pixels

    mean /= total
    std = (sq_mean / total - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def build_transforms(mean, std, image_size=224, train=False):
    transforms_list = [
        transforms.Resize((image_size, image_size)),
    ]
    if train:
        transforms_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ]
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


def build_dataloaders(dataset_root, train_idx, val_idx, test_idx, mean, std, batch_size=32, num_workers=4):
    dataset = ImageFolder(dataset_root)

    train_dataset = ImageFolder(dataset_root, transform=build_transforms(mean, std, train=True))
    val_dataset = ImageFolder(dataset_root, transform=build_transforms(mean, std, train=False))
    test_dataset = ImageFolder(dataset_root, transform=build_transforms(mean, std, train=False))

    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def create_splits(dataset_root, val_ratio=0.2, test_ratio=0.2, seed=42):
    dataset = ImageFolder(dataset_root)
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    val_size = int(num_samples * val_ratio)
    train_size = num_samples - test_size - val_size

    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size : train_size + val_size].tolist()
    test_idx = indices[train_size + val_size :].tolist()

    return train_idx, val_idx, test_idx


def mixup(x, y, alpha=0.4):
    batch_size = x.size(0)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def resnet50_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    model = models.resnet50(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False
    return model


def vgg19_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    model = models.vgg19(pretrained=pretrained)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier."):
                p.requires_grad = False
    return model


def make_optimizer(model, lr=1e-4, weight_decay=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay)


class EarlyStopping:
    def __init__(self, patience=8, mode="min", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False, True

        improved = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad = 0
            return False, True
        self.bad += 1
        return (self.bad >= self.patience), False


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_mixup=True):
    model.train()
    correct = 0
    total = 0
    loss_sum = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_mixup:
            x, y_a, y_b, lam = mixup(x, y)

        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b) if use_mixup else criterion(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b) if use_mixup else criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=loss.item())

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * y.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes=NUM_CLASSES):
    model.eval()
    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    y_true = torch.cat(all_y, dim=0)
    y_pred = logits.argmax(dim=1)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    total = cm.sum().item()
    correct = cm.diag().sum().item()
    acc = correct / max(total, 1)

    per_class = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        support = cm[c, :].sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class.append({"precision": precision, "recall": recall, "f1": f1, "support": support})

    macro_precision = sum(d["precision"] for d in per_class) / num_classes
    macro_recall = sum(d["recall"] for d in per_class) / num_classes
    macro_f1 = sum(d["f1"] for d in per_class) / num_classes
    total_support = max(sum(d["support"] for d in per_class), 1)
    weighted_precision = sum(d["precision"] * d["support"] for d in per_class) / total_support
    weighted_recall = sum(d["recall"] * d["support"] for d in per_class) / total_support
    weighted_f1 = sum(d["f1"] * d["support"] for d in per_class) / total_support

    return {
        "acc": acc,
        "micro_precision": acc,
        "micro_recall": acc,
        "micro_f1": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "logits": logits,
        "y_true": y_true,
    }


def plot_confusion_matrix(cm, class_names=None, figsize=(6, 6), cmap="Blues"):
    plt.figure(figsize=figsize)
    plt.imshow(cm.numpy(), interpolation="nearest", cmap=cmap)
    plt.title("Confusion matrix")
    plt.colorbar()
    n_classes = cm.shape[0]
    ticks = np.arange(n_classes)
    plt.xticks(ticks, class_names if class_names else ticks, rotation=45)
    plt.yticks(ticks, class_names if class_names else ticks)

    thresh = cm.max().item() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, int(cm[i, j].item()), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

