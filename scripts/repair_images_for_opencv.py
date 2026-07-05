import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def cv2_read(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def repair_image(path: Path, backup_dir: Path) -> bool:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / path.name
    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    with Image.open(path) as img:
        img = img.convert("RGB")
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            img.save(path, format="JPEG", quality=95, subsampling=0, optimize=False)
        elif path.suffix.lower() == ".png":
            img.save(path, format="PNG")
        else:
            img.save(path)

    return cv2_read(path) is not None


def main():
    parser = argparse.ArgumentParser(description="Repair images that Pillow can read but OpenCV cannot decode.")
    parser.add_argument("--input", default="数据集", help="Image folder to scan.")
    parser.add_argument(
        "--backup",
        default=None,
        help="Backup folder. Default: <input>/_backup_opencv_unreadable",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    backup_dir = Path(args.backup) if args.backup else input_dir / "_backup_opencv_unreadable"

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    already_ok = []
    repaired = []
    failed = []

    for path in image_paths:
        if cv2_read(path) is not None:
            already_ok.append(path.name)
            continue
        try:
            ok = repair_image(path, backup_dir)
            if ok:
                repaired.append(path.name)
            else:
                failed.append((path.name, "OpenCV still cannot decode after Pillow re-save"))
        except Exception as exc:
            failed.append((path.name, str(exc)))

    print(f"Input: {input_dir.resolve()}")
    print(f"Scanned images: {len(image_paths)}")
    print(f"Already OpenCV-readable: {len(already_ok)}")
    print(f"Repaired and overwritten: {len(repaired)}")
    print(f"Failed: {len(failed)}")
    print(f"Backup folder: {backup_dir.resolve()}")

    if repaired:
        print("\nRepaired files:")
        for name in repaired:
            print(f"  {name}")

    if failed:
        print("\nFailed files:")
        for name, reason in failed:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
