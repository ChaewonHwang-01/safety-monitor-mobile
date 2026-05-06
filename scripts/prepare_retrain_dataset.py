from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ["helmet", "head"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stronger PPE dataset for retraining.")
    parser.add_argument("--hardhat-source", type=Path, default=Path("data/raw/Hardhat"))
    parser.add_argument(
        "--extra-source",
        type=Path,
        default=Path.home() / ".cache" / "kagglehub" / "datasets" / "npk7264" / "helmet-dataset" / "versions" / "1",
    )
    parser.add_argument("--output", type=Path, default=Path("data/yolo/hardhat_plus"))
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


def write_hardhat_split(source: Path, output: Path, source_split: str, target_split: str) -> int:
    image_dir = source / source_split / "JPEGImage"
    annotation_dir = source / source_split / "Annotation"
    image_out = output / "images" / target_split
    label_out = output / "labels" / target_split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for xml_path in sorted(annotation_dir.glob("*.xml")):
        image_path = image_dir / f"{xml_path.stem}.jpg"
        if not image_path.exists():
            continue

        rows = convert_xml(xml_path)
        if not rows:
            continue

        target_name = f"hardhat_{source_split.lower()}_{image_path.name}"
        shutil.copy2(image_path, image_out / target_name)
        (label_out / f"{Path(target_name).stem}.txt").write_text("\n".join(rows), encoding="utf-8")
        written += 1
    return written


def write_extra_no_helmet(source: Path, output: Path, source_split: str, target_split: str) -> int:
    image_dir = source / source_split / "images"
    label_dir = source / source_split / "labels"
    image_out = output / "images" / target_split
    label_out = output / "labels" / target_split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        classes = {line.split()[0] for line in lines}
        if classes != {"1"}:
            continue

        image_candidates = list(image_dir.glob(f"{label_path.stem}.*"))
        if not image_candidates:
            continue

        mapped_rows = []
        for line in lines:
            parts = line.split()
            mapped_rows.append("1 " + " ".join(parts[1:5]))

        image_path = image_candidates[0]
        target_name = f"extra_nohelmet_{source_split}_{image_path.name}"
        shutil.copy2(image_path, image_out / target_name)
        (label_out / f"{Path(target_name).stem}.txt").write_text("\n".join(mapped_rows), encoding="utf-8")
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

    if args.output.exists():
        shutil.rmtree(args.output)

    counts = {
        "hardhat_train": write_hardhat_split(args.hardhat_source, args.output, "Train", "train"),
        "hardhat_val": write_hardhat_split(args.hardhat_source, args.output, "Test", "val"),
        "extra_train": write_extra_no_helmet(args.extra_source, args.output, "train", "train"),
        "extra_valid": write_extra_no_helmet(args.extra_source, args.output, "valid", "val"),
        "extra_test": write_extra_no_helmet(args.extra_source, args.output, "test", "val"),
    }
    write_data_yaml(args.output)

    print(f"Prepared retraining dataset: {args.output}")
    for name, count in counts.items():
        print(f"{name}: {count}")
    print(f"train total: {counts['hardhat_train'] + counts['extra_train']}")
    print(f"val total: {counts['hardhat_val'] + counts['extra_valid'] + counts['extra_test']}")
    print(f"classes: {', '.join(CLASSES)}")


if __name__ == "__main__":
    main()
