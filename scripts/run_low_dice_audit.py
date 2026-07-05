from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_tiny_unet_split import dice_score
from train_tiny_unet import TinyUNet, VeinMaskDataset


def run_command(command: list[str], log_path: Path | None = None) -> None:
    print(" ".join(command), flush=True)
    if log_path:
        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            process = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
    else:
        process = subprocess.run(command, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def evaluate_rows(weights_path: Path, dataset_root: Path, split: str) -> list[dict[str, object]]:
    ckpt = torch.load(weights_path, map_location="cpu")
    model = TinyUNet(base_channels=int(ckpt.get("base_channels", 8)))
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = VeinMaskDataset(dataset_root, split, int(ckpt.get("imgsz", 128)), augment=False)

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for index in range(len(ds)):
            image_tensor, mask_tensor = ds[index]
            stem = ds.images[index].stem
            prob = torch.sigmoid(model(image_tensor.unsqueeze(0)))[0, 0].numpy()
            pred = (prob >= 0.5).astype(np.uint8)
            target = (mask_tensor[0].numpy() >= 0.5).astype(np.uint8)
            rows.append({"stem": stem, "dice": float(dice_score(pred, target))})
    return rows


def priority_for(count: int, mean_dice: float, min_dice: float) -> str:
    if count >= 2 and mean_dice < 0.55:
        return "high"
    if min_dice < 0.50:
        return "medium"
    if mean_dice < 0.62:
        return "watch"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed low-Dice audit for data1.")
    parser.add_argument("--data", default="data1")
    parser.add_argument("--dataset", default="datasets/split_hand_vein")
    parser.add_argument("--runs-dir", default="runs/tiny_unet")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    python = sys.executable
    data_dir = Path(args.data)
    dataset_root = Path(args.dataset)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    summary_by_stem: dict[str, list[dict[str, object]]] = defaultdict(list)

    for seed in args.seeds:
        run_name = f"data1_audit_seed{seed}"
        weights = runs_dir / f"{run_name}_128_bc8_best.pth"
        viz_dir = runs_dir / f"{run_name}_test_visualizations"
        train_log = runs_dir / f"{run_name}_train.log"

        print(f"\n=== seed {seed} ===", flush=True)
        run_command(
            [
                python,
                "labelme_to_split_dataset.py",
                "--input",
                str(data_dir),
                "--out",
                str(dataset_root),
                "--label",
                "vein",
                "--train-ratio",
                str(args.train_ratio),
                "--valid-ratio",
                "0",
                "--seed",
                str(seed),
                "--overwrite",
            ]
        )
        run_command(
            [
                python,
                "train_tiny_unet.py",
                "--dataset",
                str(dataset_root),
                "--imgsz",
                str(args.imgsz),
                "--base-channels",
                str(args.base_channels),
                "--batch",
                str(args.batch),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--workers",
                "0",
                "--out",
                str(weights),
            ],
            log_path=train_log,
        )
        run_command(
            [
                python,
                "evaluate_tiny_unet_split.py",
                "--weights",
                str(weights),
                "--dataset",
                str(dataset_root),
                "--split",
                "test",
                "--out",
                str(viz_dir),
            ]
        )

        rows = evaluate_rows(weights, dataset_root, "test")
        dices = [float(row["dice"]) for row in rows]
        print(
            f"seed {seed}: test n={len(rows)} mean={np.mean(dices):.4f} "
            f"min={np.min(dices):.4f} max={np.max(dices):.4f}",
            flush=True,
        )
        for row in rows:
            record = {
                "seed": seed,
                "stem": str(row["stem"]),
                "dice": float(row["dice"]),
                "viz_path": str((viz_dir / f"{row['stem']}_viz.png").resolve()),
            }
            all_rows.append(record)
            summary_by_stem[str(row["stem"])].append(record)

    rows_path = runs_dir / "data1_low_dice_audit_all_rows.tsv"
    with rows_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["seed", "stem", "dice", "viz_path"], delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    summary_rows = []
    for stem, records in summary_by_stem.items():
        dices = [float(record["dice"]) for record in records]
        viz_paths = ";".join(str(record["viz_path"]) for record in records)
        summary_rows.append(
            {
                "priority": priority_for(len(records), float(np.mean(dices)), float(np.min(dices))),
                "stem": stem,
                "test_count": len(records),
                "mean_dice": float(np.mean(dices)),
                "min_dice": float(np.min(dices)),
                "max_dice": float(np.max(dices)),
                "seeds": ",".join(str(record["seed"]) for record in records),
                "viz_paths": viz_paths,
            }
        )

    priority_order = {"high": 0, "medium": 1, "watch": 2, "ok": 3}
    summary_rows.sort(key=lambda row: (priority_order[row["priority"]], row["mean_dice"], row["stem"]))

    summary_path = runs_dir / "data1_low_dice_audit.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "priority",
                "stem",
                "test_count",
                "mean_dice",
                "min_dice",
                "max_dice",
                "seeds",
                "viz_paths",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nAll rows: {rows_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    for priority in ("high", "medium", "watch"):
        selected = [row for row in summary_rows if row["priority"] == priority]
        print(f"\n{priority}: {len(selected)}")
        for row in selected[:20]:
            print(
                f"{row['stem']}\tcount={row['test_count']}\t"
                f"mean={row['mean_dice']:.4f}\tmin={row['min_dice']:.4f}"
            )


if __name__ == "__main__":
    main()
