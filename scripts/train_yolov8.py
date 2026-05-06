from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n for hard-hat detection.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "yolo" / "hardhat" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "train")
    parser.add_argument("--name", default="hardhat-yolov8n")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save-period", type=int, default=10)
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT / "models" / "best.pt")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        save_period=args.save_period,
    )

    save_dir = Path(results.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    target = args.target if args.target.is_absolute() else PROJECT_ROOT / args.target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, target)
    print(f"Copied trained model to {target.resolve()}")


if __name__ == "__main__":
    main()
