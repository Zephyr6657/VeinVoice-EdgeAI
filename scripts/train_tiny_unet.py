from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny U-Net on hand vein masks.")
    parser.add_argument("--dataset", default="datasets/hand_vein")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=160,
        choices=(128, 160, 320, 640),
        help="Square training size. Use 128 or 160 for Ethos-U55-oriented experiments.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--base-channels",
        type=int,
        default=8,
        choices=(4, 8, 16),
        help="Tiny U-Net width. Use 8 or smaller for deployment-oriented experiments.",
    )
    parser.add_argument("--out", default="tiny_unet_best.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--own-sample-weight",
        type=float,
        default=1.0,
        help="Sampling weight for training images whose filename starts with own_.",
    )
    parser.add_argument(
        "--img-sample-weight",
        type=float,
        default=None,
        help="Sampling weight for target-distribution images whose filename starts with own_IMG_.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


class VeinMaskDataset(Dataset):
    def __init__(self, root: Path, split: str, imgsz: int, augment: bool = False) -> None:
        self.images_dir = root / split / "images"
        self.masks_dir = root / split / "masks"
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
        if image.shape[:2] != (self.imgsz, self.imgsz):
            image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                image = np.ascontiguousarray(image[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            if random.random() < 0.8:
                alpha = random.uniform(0.75, 1.35)
                beta = random.uniform(-18.0, 18.0)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return image_tensor, mask_tensor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3_conv = nn.Conv2d(c * 8, c * 4, 1)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2_conv = nn.Conv2d(c * 4, c * 2, 1)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1_conv = nn.Conv2d(c * 2, c, 1)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3_conv(self.up(b)), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2_conv(self.up(d3)), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1_conv(self.up(d2)), e1], dim=1))
        return self.head(d1)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > 0.5).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)
    return ((2 * intersection + eps) / (union + eps)).mean()


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice_loss = 1 - ((2 * intersection + 1e-7) / (union + 1e-7)).mean()
    return bce + dice_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
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
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach())
        total_dice += float(dice_score_from_logits(logits.detach(), masks))
        batches += 1

    return total_loss / batches, total_dice / batches


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    dataset_root = Path(args.dataset).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"

    train_ds = VeinMaskDataset(dataset_root, "train", args.imgsz, augment=True)
    eval_split = "valid" if (dataset_root / "valid" / "images").is_dir() else "test"
    valid_ds = VeinMaskDataset(dataset_root, eval_split, args.imgsz, augment=False)
    train_sampler = None
    train_shuffle = True
    img_sample_weight = args.img_sample_weight
    if img_sample_weight is None:
        img_sample_weight = args.own_sample_weight
    if args.own_sample_weight != 1.0 or img_sample_weight != 1.0:
        sample_weights = [
            img_sample_weight
            if image_path.stem.startswith("own_IMG_")
            else args.own_sample_weight
            if image_path.stem.startswith("own_")
            else 1.0
            for image_path in train_ds.images
        ]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyUNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best_dice = -1.0
    out_path = Path(args.out).resolve()
    print(f"Training on {device}: {len(train_ds)} train, {len(valid_ds)} {eval_split}")
    if train_sampler is not None:
        own_count = sum(1 for image_path in train_ds.images if image_path.stem.startswith("own_"))
        img_count = sum(1 for image_path in train_ds.images if image_path.stem.startswith("own_IMG_"))
        print(
            f"Weighted sampling enabled: own_IMG={img_count}, own_other={own_count - img_count}, "
            f"other={len(train_ds) - own_count}, img_sample_weight={img_sample_weight}, "
            f"own_sample_weight={args.own_sample_weight}"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimizer, scaler)
        valid_loss, valid_dice = run_epoch(model, valid_loader, device)
        scheduler.step()

        improved = valid_dice > best_dice
        if improved:
            best_dice = valid_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "arch": "TinyUNet",
                    "base_channels": args.base_channels,
                    "imgsz": args.imgsz,
                    "best_dice": best_dice,
                    "epoch": epoch,
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

    print(f"Best checkpoint saved to: {out_path}")


if __name__ == "__main__":
    main()
