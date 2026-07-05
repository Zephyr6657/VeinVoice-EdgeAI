from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "hand_vein" / "data.yaml"


def main() -> None:
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data=str(DATA),
        task="segment",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        patience=30,
        amp=True,
        close_mosaic=10,
        project=str(ROOT / "runs" / "segment"),
        name="hand_vein_yolov8n",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
