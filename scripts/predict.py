from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "segment" / "hand_vein_yolov8n" / "weights" / "best.pt"
SOURCE = ROOT / "datasets" / "hand_vein" / "test" / "images"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")
    if not SOURCE.exists():
        raise FileNotFoundError(f"Test image directory not found: {SOURCE}")

    model = YOLO(str(WEIGHTS))
    model.predict(
        source=str(SOURCE),
        task="segment",
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(ROOT / "runs" / "segment"),
        name="hand_vein_yolov8n_predict",
    )


if __name__ == "__main__":
    main()
