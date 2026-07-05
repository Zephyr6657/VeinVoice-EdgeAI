from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from train_tiny_unet import TinyUNet, dice_score_from_logits, loss_fn, seed_everything


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Tiny U-Net on a small image/mask dataset.")
    parser.add_argument("--dataset", default="datasets/finetune_hand_vein")
    parser.add_argument("--weights", default="runs/tiny_unet/tiny_unet_128_bc8_best.pth")
    parser.add_argument("--out", default="runs/tiny_unet/tiny_unet_128_bc8_finetuned.pth")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class FlatVeinMaskDataset(Dataset):
    def __init__(self, root: Path, imgsz: int, augment: bool = False) -> None:
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        self.imgsz = imgsz
        self.augment = augment
        self.images = sorted(
            path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        missing = [
            image.name
            for image in self.images
            if not (self.masks_dir / f"{image.stem}.png").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} masks missing in {self.masks_dir}; first: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = self.masks_dir / f"{image_path.stem}.png"

        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            if random.random() < 0.5:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-12, 12)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return image_tensor, mask_tensor


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach())
        total_dice += float(dice_score_from_logits(logits.detach(), masks))
        batches += 1

    return total_loss / batches, total_dice / batches


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = Path(args.dataset).resolve()
    weights_path = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    base_channels = int(checkpoint["base_channels"])

    train_base = FlatVeinMaskDataset(dataset_root, imgsz=imgsz, augment=True)
    val_base = FlatVeinMaskDataset(dataset_root, imgsz=imgsz, augment=False)
    val_size = max(1, round(len(train_base) * args.val_ratio))
    train_size = len(train_base) - val_size
    if train_size < 1:
        raise ValueError("Need at least two samples for fine-tuning with validation.")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    train_ds = Subset(train_base, indices[:train_size])
    val_ds = Subset(val_base, indices[train_size:])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_dice = -1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Fine-tuning on {device}: {train_size} train, {val_size} valid, "
        f"imgsz={imgsz}, base_channels={base_channels}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer)
        valid_loss, valid_dice = run_epoch(model, val_loader, device)
        scheduler.step()

        improved = valid_dice > best_dice
        if improved:
            best_dice = valid_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "TinyUNet",
                    "base_channels": base_channels,
                    "imgsz": imgsz,
                    "best_dice": best_dice,
                    "epoch": epoch,
                    "source_weights": str(weights_path),
                    "finetune_dataset": str(dataset_root),
                },
                out_path,
            )

        marker = "*" if improved else " "
        print(
            f"{marker} epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
            f"valid_loss={valid_loss:.4f} valid_dice={valid_dice:.4f} "
            f"best={best_dice:.4f}",
            flush=True,
        )

    print(f"Best fine-tuned checkpoint saved to: {out_path}")


if __name__ == "__main__":
    main()
