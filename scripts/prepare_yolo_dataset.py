from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ["helmet", "head"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Hardhat VOC XML labels to YOLOv8 format.")
    parser.add_argument("--source", type=Path, default=Path("data/raw/Hardhat"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo/hardhat"))
    parser.add_argument("--max-train", type=int, default=900)
    parser.add_argument("--max-val", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def convert_box(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def convert_xml(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    if width <= 0 or height <= 0:
        return []

    rows: list[str] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip().lower()
        if class_name not in CLASS_TO_ID:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(float(width), float(box.findtext("xmax", "0")))
        ymax = min(float(height), float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue

        yolo_box = convert_box((width, height), (xmin, ymin, xmax, ymax))
        rows.append(f"{CLASS_TO_ID[class_name]} " + " ".join(f"{value:.6f}" for value in yolo_box))
    return rows


def collect_pairs(source: Path, split: str) -> list[tuple[Path, Path]]:
    split_root = source / split
    image_dir = split_root / "JPEGImage"
    annotation_dir = split_root / "Annotation"
    pairs: list[tuple[Path, Path]] = []

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        image_path = image_dir / f"{xml_path.stem}.jpg"
        if image_path.exists():
            pairs.append((image_path, xml_path))
    return pairs


def write_split(pairs: list[tuple[Path, Path]], output: Path, split: str) -> int:
    image_out = output / "images" / split
    label_out = output / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_path, xml_path in pairs:
        rows = convert_xml(xml_path)
        if not rows:
            continue
        shutil.copy2(image_path, image_out / image_path.name)
        (label_out / f"{image_path.stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        written += 1
    return written


def write_data_yaml(output: Path) -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    content = (
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/val\n"
        f"names:\n{names}\n"
    )
    (output / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    train_pairs = collect_pairs(args.source, "Train")
    val_pairs = collect_pairs(args.source, "Test")
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)

    if args.max_train > 0:
        train_pairs = train_pairs[: args.max_train]
    if args.max_val > 0:
        val_pairs = val_pairs[: args.max_val]

    if args.output.exists():
        shutil.rmtree(args.output)

    train_count = write_split(train_pairs, args.output, "train")
    val_count = write_split(val_pairs, args.output, "val")
    write_data_yaml(args.output)

    print(f"Prepared YOLO dataset: {args.output}")
    print(f"train images: {train_count}")
    print(f"val images: {val_count}")
    print(f"classes: {', '.join(CLASSES)}")


if __name__ == "__main__":
    main()
