#!/usr/bin/env python3
"""
Convert a CVAT pose dataset (CVAT for images 1.1 XML + images folder)
into an Ultralytics YOLO pose training dataset.

Input directory example:
    dataset_root/
      annotations.xml
      images/
        *.jpg|*.png|...

Output directory example:
    output_root/
      dataset.yaml
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/

Examples:
    python tools/dataset_process/convert_cvat_pose_to_yolo.py \
        /path/to/cvat_dataset

    python tools/dataset_process/convert_cvat_pose_to_yolo.py \
        /path/to/cvat_dataset \
        -o /path/to/output \
        --val-ratio 0.2 \
        --test-ratio 0.1 \
        --copy
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CVAT pose annotations to Ultralytics YOLO pose format."
    )
    parser.add_argument("input_dir", help="CVAT dataset root containing annotations.xml and images/")
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output YOLO pose dataset directory (default: <input_dir>_yolo_pose)",
    )
    parser.add_argument(
        "--annotations",
        default="annotations.xml",
        help="Annotation XML filename relative to input_dir",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional skeleton label names to export. Defaults to all skeleton labels in XML meta.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Test split ratio. Default: 0.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split. Default: 42",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of creating symlinks.",
    )
    parser.add_argument(
        "--skip-empty-images",
        action="store_true",
        help="Skip images that contain no exported pose instances.",
    )
    return parser.parse_args()


def parse_point_xy(points_attr: str):
    first_pair = points_attr.split(";")[0].strip()
    x_str, y_str = first_pair.split(",")
    return float(x_str), float(y_str)


def cvat_point_visibility(point_el: ET.Element) -> int:
    outside = point_el.get("outside", "0")
    occluded = point_el.get("occluded", "0")
    if outside == "1":
        return 0
    if occluded == "1":
        return 1
    return 2


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalize_xy(x: float, y: float, width: int, height: int):
    return clamp01(x / float(width)), clamp01(y / float(height))


def make_unique_name(name: str, used_names: set[str]) -> str:
    candidate = name
    stem = Path(name).stem
    suffix = Path(name).suffix
    index = 2
    while candidate in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate)
    return candidate


def infer_flip_idx(keypoint_names: list[str]):
    lower_to_index = {name.lower(): i for i, name in enumerate(keypoint_names)}
    flip_idx = []
    paired = False

    for i, name in enumerate(keypoint_names):
        lower = name.lower()
        replacement = None
        if "left" in lower:
            replacement = lower.replace("left", "right")
        elif "right" in lower:
            replacement = lower.replace("right", "left")

        if replacement and replacement in lower_to_index:
            flip_idx.append(lower_to_index[replacement])
            if lower_to_index[replacement] != i:
                paired = True
        else:
            flip_idx.append(i)

    return flip_idx if paired else None


def parse_cvat_meta(root: ET.Element):
    label_meta = OrderedDict()

    # CVAT exports use meta/task (task-level export) or meta/job (job-level export)
    label_els = root.findall("./meta/task/labels/label")
    if not label_els:
        label_els = root.findall("./meta/job/labels/label")

    for label_el in label_els:
        label_name_el = label_el.find("name")
        label_type_el = label_el.find("type")
        if label_name_el is None or label_type_el is None:
            continue
        if label_type_el.text != "skeleton":
            continue

        sublabels = []
        for sublabel_el in label_el.findall("./sublabels/sublabel"):
            name_el = sublabel_el.find("name")
            if name_el is not None and name_el.text:
                sublabels.append(name_el.text)

        label_meta[label_name_el.text] = sublabels

    if not label_meta:
        raise RuntimeError("No skeleton labels found in CVAT XML meta.")
    return label_meta


def parse_skeleton_instance(skeleton_el: ET.Element, keypoint_names: list[str], width: int, height: int):
    point_by_name = {}
    for point_el in skeleton_el.findall("points"):
        point_label = point_el.get("label")
        if point_label:
            point_by_name[point_label] = point_el

    keypoints = []
    visible_bbox_points = []
    all_bbox_points = []

    for keypoint_name in keypoint_names:
        point_el = point_by_name.get(keypoint_name)
        if point_el is None:
            keypoints.extend([0.0, 0.0, 0])
            continue

        x, y = parse_point_xy(point_el.get("points", "0,0"))
        visibility = cvat_point_visibility(point_el)

        # bbox uses real coordinates for all non-missing points
        all_bbox_points.append((x, y))
        if visibility > 0:
            visible_bbox_points.append((x, y))

        # YOLO output: only visible (v=2) keypoints keep coordinates;
        # occluded/outside → (0, 0, 0) to avoid data-augmentation artifacts.
        if visibility == 2:
            x_norm, y_norm = normalize_xy(x, y, width, height)
            keypoints.extend([x_norm, y_norm, 2])
        else:
            keypoints.extend([0.0, 0.0, 0])

    bbox_points = visible_bbox_points or all_bbox_points
    if not bbox_points:
        return None

    x_min = min(x for x, _ in bbox_points)
    x_max = max(x for x, _ in bbox_points)
    y_min = min(y for _, y in bbox_points)
    y_max = max(y for _, y in bbox_points)

    # Keep hard-edge examples near the image border instead of dropping them
    # when only one keypoint is visible or all visible points are collinear.
    box_w = max(1.0, x_max - x_min)
    box_h = max(1.0, y_max - y_min)

    cx = x_min + box_w / 2.0
    cy = y_min + box_h / 2.0
    bbox = [
        clamp01(cx / float(width)),
        clamp01(cy / float(height)),
        clamp01(box_w / float(width)),
        clamp01(box_h / float(height)),
    ]
    return bbox, keypoints


def format_label_line(class_id: int, bbox: list[float], keypoints: list[float | int]):
    values = [class_id, *bbox, *keypoints]
    formatted = []
    for idx, value in enumerate(values):
        if idx >= 5 and (idx - 5) % 3 == 2:
            formatted.append(str(int(value)))
        elif idx == 0:
            formatted.append(str(int(value)))
        else:
            formatted.append(f"{float(value):.6f}")
    return " ".join(formatted)


def split_items(items: list, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to less than 1.")

    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)
    n_train = total - n_val - n_test

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_dataset_yaml(
    output_dir: Path,
    class_names: list[str],
    keypoint_names_by_class: dict[int, list[str]],
    flip_idx: list[int] | None,
    include_test: bool,
):
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if include_test:
        lines.append("test: images/test")

    first_class_kpts = next(iter(keypoint_names_by_class.values()))
    lines.extend(
        [
            "",
            f"kpt_shape: [{len(first_class_kpts)}, 3]",
        ]
    )
    if flip_idx:
        lines.append(f"flip_idx: [{', '.join(str(i) for i in flip_idx)}]")

    lines.extend(["", "names:"])
    for class_id, class_name in enumerate(class_names):
        lines.append(f"  {class_id}: {class_name}")

    lines.extend(["", "kpt_names:"])
    for class_id, keypoint_names in keypoint_names_by_class.items():
        lines.append(f"  {class_id}:")
        for name in keypoint_names:
            lines.append(f"    - {name}")

    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_split_dirs(output_dir: Path, include_test: bool):
    split_names = ["train", "val"] + (["test"] if include_test else [])
    for split in split_names:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def export_image(src: Path, dst: Path, use_copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main():
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    annotations_path = input_dir / args.annotations
    if not annotations_path.is_file():
        raise FileNotFoundError(f"CVAT annotations XML not found: {annotations_path}")

    images_dir = input_dir / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"CVAT images directory not found: {images_dir}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_yolo_pose"
    )

    tree = ET.parse(annotations_path)
    root = tree.getroot()
    label_meta = parse_cvat_meta(root)

    selected_labels = args.labels or list(label_meta.keys())
    missing_labels = [name for name in selected_labels if name not in label_meta]
    if missing_labels:
        raise ValueError(f"Requested labels not found in XML meta: {missing_labels}")

    class_names = list(selected_labels)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    keypoint_names_by_class = {class_to_id[name]: label_meta[name] for name in class_names}

    keypoint_counts = {len(v) for v in keypoint_names_by_class.values()}
    if len(keypoint_counts) != 1:
        raise ValueError(
            "All exported skeleton labels must have the same number of keypoints for YOLO pose training."
        )

    flip_idx = infer_flip_idx(keypoint_names_by_class[0])
    include_test = args.test_ratio > 0
    ensure_split_dirs(output_dir, include_test)

    image_records = []
    stats = defaultdict(int)
    used_output_names = set()

    for image_el in root.findall("image"):
        image_name = image_el.get("name")
        if not image_name:
            continue

        width = int(image_el.get("width", "0"))
        height = int(image_el.get("height", "0"))
        if width <= 0 or height <= 0:
            continue

        image_path = input_dir / image_name
        if not image_path.is_file():
            stats["missing_images"] += 1
            continue

        label_lines = []
        for skeleton_el in image_el.findall("skeleton"):
            label_name = skeleton_el.get("label")
            if label_name not in class_to_id:
                continue

            parsed = parse_skeleton_instance(
                skeleton_el,
                keypoint_names_by_class[class_to_id[label_name]],
                width,
                height,
            )
            if parsed is None:
                stats["skipped_instances"] += 1
                continue

            bbox, keypoints = parsed
            label_lines.append(format_label_line(class_to_id[label_name], bbox, keypoints))
            stats["instances"] += 1

        if args.skip_empty_images and not label_lines:
            stats["skipped_empty_images"] += 1
            continue

        original_name = Path(image_name).name
        unique_name = make_unique_name(original_name, used_output_names)
        image_records.append(
            {
                "src": image_path,
                "name": unique_name,
                "label_lines": label_lines,
            }
        )
        stats["images"] += 1
        if label_lines:
            stats["images_with_labels"] += 1
        else:
            stats["images_without_labels"] += 1

    splits = split_items(image_records, args.val_ratio, args.test_ratio, args.seed)

    for split_name, records in splits.items():
        for record in records:
            image_dst = output_dir / "images" / split_name / record["name"]
            label_dst = output_dir / "labels" / split_name / f"{Path(record['name']).stem}.txt"
            export_image(record["src"], image_dst, use_copy=args.copy)
            label_dst.write_text("\n".join(record["label_lines"]) + ("\n" if record["label_lines"] else ""), encoding="utf-8")

    write_dataset_yaml(
        output_dir=output_dir,
        class_names=class_names,
        keypoint_names_by_class=keypoint_names_by_class,
        flip_idx=flip_idx,
        include_test=include_test,
    )

    print(f"input_dir:            {input_dir}")
    print(f"annotations:          {annotations_path}")
    print(f"output_dir:           {output_dir}")
    print(f"classes:              {len(class_names)}")
    print(f"class_names:          {class_names}")
    print(f"keypoints_per_obj:    {len(keypoint_names_by_class[0])}")
    print(f"flip_idx:             {flip_idx if flip_idx else 'not inferred'}")
    print(f"images_total:         {stats['images']}")
    print(f"images_with_labels:   {stats['images_with_labels']}")
    print(f"images_without_labels:{stats['images_without_labels']}")
    print(f"instances_total:      {stats['instances']}")
    print(f"missing_images:       {stats['missing_images']}")
    print(f"skipped_instances:    {stats['skipped_instances']}")
    print(f"split_train:          {len(splits['train'])}")
    print(f"split_val:            {len(splits['val'])}")
    if include_test:
        print(f"split_test:           {len(splits['test'])}")
    print(f"dataset_yaml:         {output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
