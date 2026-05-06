from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train second-stage helmet/cap/bare-head classifier.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "headwear_cls" / "prepared")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="yolov8n-cls.pt")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs" / "classify")
    parser.add_argument("--name", default="headwear-cls")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT / "models" / "headwear_cls.pt")
    return parser.parse_args()


def main() -> None:
    from ultralytics import YOLO

    args = parse_args()
    data = args.data if args.data.is_absolute() else PROJECT_ROOT / args.data
    target = args.target if args.target.is_absolute() else PROJECT_ROOT / args.target

    model = YOLO(args.model)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
    )

    save_dir = Path(results.save_dir)
    best_weight = save_dir / "weights" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weight, target)
    print(f"Copied trained classifier to {target.resolve()}")


if __name__ == "__main__":
    main()
