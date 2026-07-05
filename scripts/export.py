from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "runs" / "segment" / "hand_vein_yolov8n" / "weights" / "best.pt"


def main() -> None:
    if not WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {WEIGHTS}")

    model = YOLO(str(WEIGHTS))
    model.export(format="onnx", imgsz=640, opset=12, simplify=True)


if __name__ == "__main__":
    main()
