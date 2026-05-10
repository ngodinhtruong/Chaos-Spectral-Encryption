import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


def run_gradient_inversion_attack(img_path: str, save_dir: str, file_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    IMAGE_SIZE = 224
    NUM_CLASSES = 10
    MAX_ITERS = 800
    LR = 0.1
    TV_WEIGHT = 1e-4
    L2_WEIGHT = 1e-6
    PRINT_EVERY = 50
    USE_SIGN_LABEL_RECOVERY = False

    gt_label = torch.tensor([0], dtype=torch.long, device=device)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    os.makedirs(save_dir, exist_ok=True)

    def total_variation(x: torch.Tensor) -> torch.Tensor:
        dh = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        dw = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return dh + dw

    def gradient_cosine_loss(dummy_grads, target_grads):
        num = 0.0
        den1 = 0.0
        den2 = 0.0
        for dg, tg in zip(dummy_grads, target_grads):
            num = num + (dg * tg).sum()
            den1 = den1 + (dg * dg).sum()
            den2 = den2 + (tg * tg).sum()
        return 1 - num / (torch.sqrt(den1) * torch.sqrt(den2) + 1e-12)

    def compute_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        return torch.mean((img1 - img2) ** 2).item()

    def compute_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
        mse = compute_mse(img1, img2)
        if mse == 0:
            return float("inf")
        return 10.0 * math.log10(1.0 / mse)

    def tensor_to_numpy_image(x: torch.Tensor):
        x = torch.clamp(x.detach().cpu(), 0, 1)
        x = x.squeeze(0).permute(1, 2, 0).numpy()
        return x

    def show_tensor_image(x: torch.Tensor, title: str):
        plt.imshow(tensor_to_numpy_image(x))
        plt.title(title)
        plt.axis("off")

    def save_tensor_image(x: torch.Tensor, save_path: str, normalize=False):
        x = x.detach().cpu().clone()
        if normalize:
            x = x / (x.max() + 1e-8)
        x = torch.clamp(x, 0, 1)
        x = x.squeeze(0).permute(1, 2, 0).numpy()
        img = Image.fromarray((x * 255).astype("uint8"))
        img.save(save_path)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    img = Image.open(img_path).convert("RGB")
    gt_data = transform(img).unsqueeze(0).to(device)

    print("device      :", device)
    print("image path  :", img_path)
    print("gt_data     :", tuple(gt_data.shape))
    print("gt_label    :", gt_label.item())

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    model.zero_grad(set_to_none=True)
    out = model(gt_data)
    target_loss = criterion(out, gt_label)
    target_grads = torch.autograd.grad(target_loss, model.parameters())
    target_grads = [g.detach().clone() for g in target_grads]

    print("Computed target gradients from YOUR image.")

    if USE_SIGN_LABEL_RECOVERY:
        try:
            grad_last_weight = target_grads[-2]
            recovered_label = torch.argmin(torch.sum(grad_last_weight, dim=1)).view(1).long().to(device)
            used_label = recovered_label
            print("Recovered label:", used_label.item())
        except Exception as e:
            print("Label recovery failed, fallback to gt_label:", str(e))
            used_label = gt_label
    else:
        used_label = gt_label
        print("Using ground-truth label:", used_label.item())

    dummy_data = torch.randn_like(gt_data, device=device, requires_grad=True)

    optimizer = optim.Adam([dummy_data], lr=LR)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[MAX_ITERS // 2, int(MAX_ITERS * 0.75)],
        gamma=0.3
    )

    history_psnr = []
    history_abs_psnr = []
    history_loss = []

    best_psnr = -1e9
    best_dummy = None
    best_iter = -1

    for it in range(1, MAX_ITERS + 1):
        optimizer.zero_grad(set_to_none=True)

        dummy_clamped = torch.sigmoid(dummy_data)

        pred = model(dummy_clamped)
        dummy_loss_ce = criterion(pred, used_label)

        dummy_grads = torch.autograd.grad(
            dummy_loss_ce,
            model.parameters(),
            create_graph=True
        )

        grad_loss = gradient_cosine_loss(dummy_grads, target_grads)
        tv_loss = total_variation(dummy_clamped)
        l2_loss = torch.mean(dummy_clamped ** 2)

        total_loss = grad_loss + TV_WEIGHT * tv_loss + L2_WEIGHT * l2_loss
        total_loss.backward()

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            current_dummy = torch.sigmoid(dummy_data)
            current_psnr = compute_psnr(gt_data, current_dummy)

            current_abs_diff = torch.abs(
                torch.clamp(gt_data, 0, 1) - torch.clamp(current_dummy, 0, 1)
            )
            zero_img = torch.zeros_like(current_abs_diff)
            current_abs_psnr = compute_psnr(current_abs_diff, zero_img)

            history_loss.append(total_loss.item())
            history_psnr.append(current_psnr)
            history_abs_psnr.append(current_abs_psnr)

            if current_psnr > best_psnr:
                best_psnr = current_psnr
                best_dummy = current_dummy.detach().clone()
                best_iter = it

        if it % PRINT_EVERY == 0 or it == 1:
            mse_now = compute_mse(gt_data, torch.sigmoid(dummy_data))
            print(
                f"iter={it:04d} | "
                f"total_loss={total_loss.item():.6f} | "
                f"grad_loss={grad_loss.item():.6f} | "
                f"tv={tv_loss.item():.6f} | "
                f"PSNR={current_psnr:.2f} dB | "
                f"Abs-PSNR={current_abs_psnr:.2f} dB | "
                f"MSE={mse_now:.6f}"
            )

    final_dummy = best_dummy
    final_psnr = compute_psnr(gt_data, final_dummy)
    final_mse = compute_mse(gt_data, final_dummy)

    absolute_diff = torch.abs(
        torch.clamp(gt_data, 0, 1) - torch.clamp(final_dummy, 0, 1)
    )
    zero_img = torch.zeros_like(absolute_diff)
    absolute_psnr = compute_psnr(absolute_diff, zero_img)

    print("\n================ FINAL RESULT ================")
    print(f"Best iter     : {best_iter}")
    print(f"Final MSE     : {final_mse:.6f}")
    print(f"Final PSNR    : {final_psnr:.2f} dB")
    print(f"Absolute PSNR : {absolute_psnr:.2f} dB")


    original_path = os.path.join(save_dir, f"{file_name}_original_input.png")
    recovered_path = os.path.join(save_dir, f"{file_name}_recovered.png")
    absolute_diff_path = os.path.join(save_dir, f"{file_name}_absolute_diff.png")
    comparison_path = os.path.join(save_dir, f"{file_name}_comparison.png")

    save_tensor_image(gt_data, original_path, normalize=False)
    save_tensor_image(final_dummy, recovered_path, normalize=False)
    save_tensor_image(absolute_diff, absolute_diff_path, normalize=True)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    show_tensor_image(gt_data, "Original Input")

    plt.subplot(1, 3, 2)
    show_tensor_image(final_dummy, f"Recovered\nPSNR={final_psnr:.2f} dB")

    plt.subplot(1, 3, 3)
    show_tensor_image(
        absolute_diff / (absolute_diff.max() + 1e-8),
        f"Absolute Difference\nAbs-PSNR={absolute_psnr:.2f} dB"
    )

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print(" -", original_path)
    print(" -", recovered_path)
    print(" -", absolute_diff_path)
    print(" -", comparison_path)

    return {
        "best_iter": best_iter,
        "final_psnr": final_psnr,
        "final_mse": final_mse,
        "absolute_psnr": absolute_psnr,
        "original_path": original_path,
        "recovered_path": recovered_path,
        "absolute_diff_path": absolute_diff_path,
        "comparison_path": comparison_path,
    }

