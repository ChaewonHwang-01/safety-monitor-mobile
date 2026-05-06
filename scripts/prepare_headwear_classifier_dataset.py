from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = {0: "helmet", 1: "bare_head"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a helmet/cap/bare-head classifier dataset.")
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data" / "yolo" / "hardhat_plus")
    parser.add_argument("--manual", type=Path, default=PROJECT_ROOT / "data" / "headwear_cls" / "manual")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "headwear_cls" / "prepared")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=2400,
        help="Augment smaller manual classes until they have at least this many samples.",
    )
    return parser.parse_args()


def crop_from_yolo(image: Image.Image, label_line: str, padding: float = 0.14) -> tuple[str, Image.Image] | None:
    parts = label_line.strip().split()
    if len(parts) != 5:
        return None

    class_id = int(float(parts[0]))
    if class_id not in CLASS_NAMES:
        return None

    x_center, y_center, box_width, box_height = map(float, parts[1:])
    width, height = image.size
    x1 = int((x_center - box_width / 2) * width)
    y1 = int((y_center - box_height / 2) * height)
    x2 = int((x_center + box_width / 2) * width)
    y2 = int((y_center + box_height / 2) * height)
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    crop = image.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )
    )
    return CLASS_NAMES[class_id], crop


def clear_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ["train", "val"]:
        for label in ["helmet", "cap_hat", "bare_head"]:
            (output / split / label).mkdir(parents=True, exist_ok=True)


def yolo_crops(source: Path) -> list[tuple[str, Image.Image]]:
    crops: list[tuple[str, Image.Image]] = []
    for split in ["train", "val"]:
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.exists() or not label_dir.exists():
            continue

        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                item = crop_from_yolo(image, line)
                if item is not None:
                    crops.append(item)
    return crops


def manual_images(manual: Path) -> list[tuple[str, Image.Image]]:
    items: list[tuple[str, Image.Image]] = []
    if not manual.exists():
        return items

    for label_dir in manual.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for image_path in label_dir.iterdir():
            if image_path.suffix.lower() in SUPPORTED_SUFFIXES:
                items.append((label, Image.open(image_path).convert("RGB")))
    return items


def group_items(items: list[tuple[str, Image.Image]]) -> dict[str, list[Image.Image]]:
    grouped: dict[str, list[Image.Image]] = {}
    for label, image in items:
        grouped.setdefault(label, []).append(image)
    return grouped


def augment_image(image: Image.Image, variant: int) -> Image.Image:
    augmented = image.copy()
    if variant % 2:
        augmented = ImageOps.mirror(augmented)

    angle = [-8, -5, -3, 3, 5, 8][variant % 6]
    augmented = augmented.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)

    brightness = [0.82, 0.9, 1.0, 1.08, 1.18][variant % 5]
    contrast = [0.88, 0.96, 1.05, 1.14][variant % 4]
    color = [0.9, 1.0, 1.12][variant % 3]
    augmented = ImageEnhance.Brightness(augmented).enhance(brightness)
    augmented = ImageEnhance.Contrast(augmented).enhance(contrast)
    augmented = ImageEnhance.Color(augmented).enhance(color)

    if variant % 7 == 0:
        augmented = augmented.filter(ImageFilter.GaussianBlur(radius=0.45))

    width, height = augmented.size
    if min(width, height) > 24 and variant % 3 == 0:
        inset_x = max(1, int(width * 0.04))
        inset_y = max(1, int(height * 0.04))
        augmented = augmented.crop((inset_x, inset_y, width - inset_x, height - inset_y)).resize(
            (width, height),
            Image.Resampling.BICUBIC,
        )

    return augmented


def balance_with_augmentation(images: list[Image.Image], min_per_class: int, max_per_class: int) -> list[Image.Image]:
    if not images:
        return images

    balanced = list(images[:max_per_class])
    variant = 0
    source_index = 0
    target_count = min(min_per_class, max_per_class)
    while len(balanced) < target_count:
        balanced.append(augment_image(images[source_index % len(images)], variant))
        variant += 1
        source_index += 1
    return balanced[:max_per_class]


def save_split(
    yolo_items: list[tuple[str, Image.Image]],
    manual_items: list[tuple[str, Image.Image]],
    output: Path,
    val_ratio: float,
    max_per_class: int,
    min_per_class: int,
) -> None:
    yolo_grouped = group_items(yolo_items)
    manual_grouped = group_items(manual_items)
    labels = sorted(set(yolo_grouped) | set(manual_grouped))

    for label in labels:
        manual_images_for_label = manual_grouped.get(label, [])
        yolo_images_for_label = yolo_grouped.get(label, [])
        random.shuffle(manual_images_for_label)
        random.shuffle(yolo_images_for_label)

        remaining = max(0, max_per_class - len(manual_images_for_label))
        images = manual_images_for_label + yolo_images_for_label[:remaining]
        if label == "cap_hat":
            images = balance_with_augmentation(images, min_per_class, max_per_class)
        val_count = max(1, int(len(images) * val_ratio)) if len(images) >= 5 else 0
        for index, image in enumerate(images):
            split = "val" if index < val_count else "train"
            image.save(output / split / label / f"{label}_{index:05d}.jpg", quality=95)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    clear_output(output)

    yolo_items = yolo_crops(args.source if args.source.is_absolute() else PROJECT_ROOT / args.source)
    manual_items = manual_images(args.manual if args.manual.is_absolute() else PROJECT_ROOT / args.manual)
    save_split(yolo_items, manual_items, output, args.val_ratio, args.max_per_class, args.min_per_class)

    for split in ["train", "val"]:
        counts = {
            label: len(list((output / split / label).glob("*.jpg")))
            for label in ["helmet", "cap_hat", "bare_head"]
        }
        print(f"{split}: {counts}")
    print(f"prepared classifier dataset: {output}")


if __name__ == "__main__":
    main()
