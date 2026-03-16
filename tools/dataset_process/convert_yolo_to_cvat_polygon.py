#!/usr/bin/env python3
import argparse
import os
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def parse_yolo_line(line):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:])
    except ValueError:
        return None
    return cls_id, cx, cy, bw, bh


def bbox_to_polygon_points(cx, cy, bw, bh, img_w, img_h):
    x_min = (cx - bw / 2.0) * img_w
    y_min = (cy - bh / 2.0) * img_h
    x_max = (cx + bw / 2.0) * img_w
    y_max = (cy + bh / 2.0) * img_h

    x_min = max(0.0, min(float(img_w - 1), x_min))
    y_min = max(0.0, min(float(img_h - 1), y_min))
    x_max = max(0.0, min(float(img_w - 1), x_max))
    y_max = max(0.0, min(float(img_h - 1), y_max))

    pts = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in pts)


def find_image_for_label(labels_dir, images_dir, label_file):
    stem = label_file.stem
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_xml(dataset_root, splits, label_name="pallet"):
    root = ET.Element("annotations")
    version = ET.SubElement(root, "version")
    version.text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")

    labels = ET.SubElement(task, "labels")
    label = ET.SubElement(labels, "label")
    name = ET.SubElement(label, "name")
    name.text = label_name

    image_id = 0
    total_polygons = 0
    total_images = 0

    for split in splits:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        label_files = sorted(labels_dir.glob("*.txt"))
        for label_file in label_files:
            image_file = find_image_for_label(labels_dir, images_dir, label_file)
            if image_file is None:
                continue

            with Image.open(image_file) as img:
                img_w, img_h = img.size

            rel_name = image_file.relative_to(dataset_root).as_posix()
            image_el = ET.SubElement(
                root,
                "image",
                {
                    "id": str(image_id),
                    "name": rel_name,
                    "width": str(img_w),
                    "height": str(img_h),
                },
            )
            image_id += 1
            total_images += 1

            with label_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parsed = parse_yolo_line(line)
                    if parsed is None:
                        continue
                    cls_id, cx, cy, bw, bh = parsed

                    if cls_id != 0:
                        continue

                    points = bbox_to_polygon_points(cx, cy, bw, bh, img_w, img_h)
                    ET.SubElement(
                        image_el,
                        "polygon",
                        {
                            "label": label_name,
                            "source": "manual",
                            "occluded": "0",
                            "points": points,
                            "z_order": "0",
                        },
                    )
                    total_polygons += 1

    indent(root)
    return ET.ElementTree(root), total_images, total_polygons


def create_zip(dataset_root, xml_path, output_zip, splits):
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, arcname="annotations.xml")
        for split in splits:
            images_dir = dataset_root / split / "images"
            if not images_dir.exists():
                continue
            for image_file in sorted(images_dir.iterdir()):
                if image_file.suffix.lower() in IMAGE_EXTS:
                    arcname = image_file.relative_to(dataset_root).as_posix()
                    zf.write(image_file, arcname=arcname)


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO bbox labels to CVAT polygons (CVAT for images 1.1)."
    )
    parser.add_argument("--dataset-root", default=".", help="Dataset root directory")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        help="Dataset splits to include",
    )
    parser.add_argument("--label-name", default="pallet", help="Class name")
    parser.add_argument(
        "--output-dir",
        default="cvat_polygon_export",
        help="Output directory for annotations and zip",
    )
    parser.add_argument(
        "--zip-name",
        default="cvat_polygon_dataset.zip",
        help="Output zip filename",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_dir = (dataset_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_tree, total_images, total_polygons = build_xml(
        dataset_root, args.splits, label_name=args.label_name
    )
    xml_path = output_dir / "annotations.xml"
    xml_tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    zip_path = output_dir / args.zip_name
    create_zip(dataset_root, xml_path, zip_path, args.splits)

    print(f"annotations: {xml_path}")
    print(f"zip: {zip_path}")
    print(f"images_in_xml: {total_images}")
    print(f"polygons: {total_polygons}")


if __name__ == "__main__":
    main()
