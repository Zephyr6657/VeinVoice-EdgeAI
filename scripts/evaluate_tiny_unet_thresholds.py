from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from evaluate_tiny_unet_split import SplitDataset, dice_score
from train_tiny_unet import TinyUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search probability thresholds for Tiny U-Net.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="datasets/split_hand_vein")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="runs/tiny_unet/threshold_search.tsv")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    dataset_root = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(weights_path, map_location="cpu")
    model = TinyUNet(base_channels=int(ckpt["base_channels"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SplitDataset(dataset_root, args.split, int(ckpt["imgsz"]))
    probs_and_targets = []
    with torch.no_grad():
        for index in range(len(ds)):
            image_tensor, mask_tensor, stem = ds[index]
            prob = torch.sigmoid(model(image_tensor.unsqueeze(0)))[0, 0].numpy()
            target = mask_tensor[0].numpy() > 0.5
            probs_and_targets.append((stem, prob, target))

    rows = []
    for threshold in args.thresholds:
        dices = [
            dice_score(prob >= threshold, target)
            for _, prob, target in probs_and_targets
        ]
        rows.append(
            {
                "threshold": threshold,
                "mean_dice": float(np.mean(dices)),
                "min_dice": float(np.min(dices)),
                "max_dice": float(np.max(dices)),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["threshold", "mean_dice", "min_dice", "max_dice"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: row["mean_dice"])
    for row in rows:
        print(
            f"threshold={row['threshold']:.2f} "
            f"mean={row['mean_dice']:.4f} min={row['min_dice']:.4f} max={row['max_dice']:.4f}"
        )
    print(
        f"best_threshold: {best['threshold']:.2f} "
        f"mean={best['mean_dice']:.4f} min={best['min_dice']:.4f} max={best['max_dice']:.4f}"
    )
    print(f"output: {out_path.resolve()}")


if __name__ == "__main__":
    main()
