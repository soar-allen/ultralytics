"""
多格式数据导入模块。

支持以下导入格式：
  A. 纯图片目录（无标注）
  B. Thoro COCO 格式（原 convert_thoro_coco_to_cvat_polygon.py 逻辑重构，
     直接解析为 FiftyOne Detection / Keypoint 对象）
  C. Roboflow 导出的 COCO 格式
  D. Roboflow 导出的 YOLO 格式
  E. CVAT 1.1 标准数据集 (XML)
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import fiftyone as fo
import numpy as np

from .config import CONFIG

logger = logging.getLogger(__name__)

IMAGE_EXTS = set(CONFIG.image_extensions)


# ===================================================================
# 通用工具
# ===================================================================

def _collect_images(root: Path, recursive: bool = True) -> list[Path]:
    gen = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in gen if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _build_basename_index(image_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index[p.name].append(p)
    return index


def _resolve_image_path(
    image_obj: dict,
    image_groups_dir: Path,
    basename_index: dict[str, list[Path]],
) -> Optional[Path]:
    """尝试从 COCO image 对象解析图片路径。"""
    for key in ("image_title", "file_name"):
        val = image_obj.get(key)
        if not val:
            continue
        candidate = image_groups_dir / val.replace("\\", "/")
        if candidate.is_file():
            return candidate

    for key in ("image_title", "file_name"):
        val = image_obj.get(key)
        if not val:
            continue
        import os
        matches = basename_index.get(os.path.basename(val), [])
        if len(matches) == 1:
            return matches[0]
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _normalize_bbox(bbox: list, w: int, h: int) -> list[float]:
    """COCO bbox [x,y,w,h] (abs) -> FiftyOne [x,y,w,h] (rel 0~1)."""
    x, y, bw, bh = (float(v) for v in bbox[:4])
    return [
        _clamp(x / w, 0, 1),
        _clamp(y / h, 0, 1),
        _clamp(bw / w, 0, 1),
        _clamp(bh / h, 0, 1),
    ]


def _get_keypoint_names(category: dict, num_kps: int, user_kp_names: dict) -> list[str]:
    cat_name = category.get("name", "")
    if cat_name in user_kp_names:
        names = user_kp_names[cat_name]
        if len(names) >= num_kps:
            return names[:num_kps]
    if "keypoints" in category and len(category["keypoints"]) >= num_kps:
        return category["keypoints"][:num_kps]
    return [f"kp_{i}" for i in range(num_kps)]


# ===================================================================
# A. 纯图片导入
# ===================================================================

def ingest_images(
    ds: fo.Dataset,
    image_dir: str | Path,
    tags: Optional[list[str]] = None,
    recursive: bool = True,
) -> int:
    """导入纯图片目录（无标注），返回新增样本数。"""
    from .data_manager import add_samples_from_dir
    return add_samples_from_dir(ds, image_dir, tags=tags, recursive=recursive)


# ===================================================================
# B. Thoro COCO 格式导入
# ===================================================================

def ingest_thoro_coco(
    ds: fo.Dataset,
    input_dir: str | Path,
    label_field: str = "ground_truth",
    label_map: Optional[dict[str, str]] = None,
    kp_names_map: Optional[dict[str, list[str]]] = None,
    include_bbox: bool = False,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    解析 Thoro COCO 格式并直接映射为 FiftyOne 对象。

    数据集目录结构：
        <input_dir>/
        ├── coco_files/          # *.coco.json
        └── image_groups/        # 图片

    标注映射：
      - keypoints → fo.Keypoint
      - segmentation → fo.Polyline (filled=True)
      - bbox → fo.Detection
    """
    input_dir = Path(input_dir).resolve()
    label_map = label_map or {}
    kp_names_map = kp_names_map or {}

    coco_dir = input_dir / "coco_files"
    image_groups_dir = input_dir / "image_groups"

    if not coco_dir.is_dir():
        raise FileNotFoundError(f"coco_files 目录不存在: {coco_dir}")
    if not image_groups_dir.is_dir():
        raise FileNotFoundError(f"image_groups 目录不存在: {image_groups_dir}")

    coco_jsons = sorted(coco_dir.glob("*.coco.json"))
    if not coco_jsons:
        raise FileNotFoundError(f"在 {coco_dir} 中未找到 .coco.json 文件")

    basename_index = _build_basename_index(image_groups_dir)
    existing_fps = set(ds.values("filepath"))

    cat_kp_cache: dict[int, list[str]] = {}
    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)
        cats = {c["id"]: c for c in data.get("categories", [])}
        for ann in data.get("annotations", []):
            kps = ann.get("keypoints", [])
            cid = ann.get("category_id")
            if kps and cid not in cat_kp_cache and cid in cats:
                num = len(kps) // 3
                cat_kp_cache[cid] = _get_keypoint_names(cats[cid], num, kp_names_map)

    stats = defaultdict(int)
    samples_to_add = []

    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)

        cat_by_id = {c["id"]: c for c in data.get("categories", [])}
        cat_name_by_id = {cid: c.get("name", f"cat_{cid}") for cid, c in cat_by_id.items()}

        ann_by_image: dict[int, list] = defaultdict(list)
        for ann in data.get("annotations", []):
            ann_by_image[ann.get("image_id")].append(ann)

        for img_obj in data.get("images", []):
            w = int(img_obj.get("width", 0) or 0)
            h = int(img_obj.get("height", 0) or 0)
            if w <= 0 or h <= 0:
                stats["skipped_no_size"] += 1
                continue

            src_path = _resolve_image_path(img_obj, image_groups_dir, basename_index)
            if src_path is None:
                stats["images_missing"] += 1
                continue

            fp_str = str(src_path.resolve())
            if fp_str in existing_fps:
                stats["skipped_duplicate"] += 1
                continue
            existing_fps.add(fp_str)

            sample = fo.Sample(filepath=fp_str)
            if tags:
                sample.tags.extend(tags)
            detections = []
            keypoints_list = []
            polylines = []

            for ann in ann_by_image.get(img_obj.get("id"), []):
                cat_id = ann.get("category_id")
                raw_name = cat_name_by_id.get(cat_id, f"cat_{cat_id}")
                label = label_map.get(raw_name, raw_name)

                kps_raw = ann.get("keypoints", [])
                has_kps = bool(kps_raw) and len(kps_raw) >= 3
                seg = ann.get("segmentation")
                has_seg = isinstance(seg, list) and any(
                    isinstance(p, list) and len(p) >= 6 for p in seg
                )
                has_bbox = bool(ann.get("bbox")) and len(ann["bbox"]) >= 4

                emitted_rich = False

                if has_kps:
                    num_kps = len(kps_raw) // 3
                    kp_names = cat_kp_cache.get(cat_id, [f"kp_{i}" for i in range(num_kps)])

                    points = []
                    confidences = []
                    for ki in range(num_kps):
                        kx = _clamp(float(kps_raw[ki * 3]) / w, 0, 1)
                        ky = _clamp(float(kps_raw[ki * 3 + 1]) / h, 0, 1)
                        kv = int(float(kps_raw[ki * 3 + 2]))
                        points.append((kx, ky))
                        confidences.append(float(kv))

                    kp = fo.Keypoint(
                        label=label,
                        points=points,
                        confidence=confidences,
                    )
                    keypoints_list.append(kp)
                    stats["keypoints"] += 1
                    emitted_rich = True

                if has_seg:
                    for poly_pts in seg:
                        if not isinstance(poly_pts, list) or len(poly_pts) < 6:
                            continue
                        norm_pts = []
                        for i in range(0, len(poly_pts), 2):
                            px = _clamp(float(poly_pts[i]) / w, 0, 1)
                            py = _clamp(float(poly_pts[i + 1]) / h, 0, 1)
                            norm_pts.append((px, py))
                        if len(norm_pts) >= 3:
                            polylines.append(fo.Polyline(
                                label=label,
                                points=[norm_pts],
                                closed=True,
                                filled=True,
                            ))
                            stats["polylines"] += 1
                    emitted_rich = True

                if has_bbox and (include_bbox or not emitted_rich):
                    rel = _normalize_bbox(ann["bbox"], w, h)
                    det = fo.Detection(
                        label=label,
                        bounding_box=rel,
                    )
                    detections.append(det)
                    stats["detections"] += 1

            if detections:
                sample[label_field] = fo.Detections(detections=detections)
            if keypoints_list:
                kp_field = f"{label_field}_keypoints"
                sample[kp_field] = fo.Keypoints(keypoints=keypoints_list)
            if polylines:
                poly_field = f"{label_field}_polylines"
                sample[poly_field] = fo.Polylines(polylines=polylines)

            samples_to_add.append(sample)
            stats["images_imported"] += 1

    if samples_to_add:
        ds.add_samples(samples_to_add)

    logger.info("Thoro COCO 导入完成: %s", dict(stats))
    return dict(stats)


# ===================================================================
# C. Roboflow COCO 格式导入
# ===================================================================

def ingest_roboflow_coco(
    ds: fo.Dataset,
    dataset_dir: str | Path,
    label_field: str = "ground_truth",
    label_map: Optional[dict[str, str]] = None,
    splits: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    导入 Roboflow 导出的 COCO 格式数据集。

    Roboflow COCO 结构：
        <dataset_dir>/
        ├── train/
        │   ├── _annotations.coco.json
        │   └── *.jpg / *.png ...
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── ...
        └── test/
            ├── _annotations.coco.json
            └── ...
    """
    dataset_dir = Path(dataset_dir).resolve()
    label_map = label_map or {}
    splits = splits or ["train", "valid", "test"]

    existing_fps = set(ds.values("filepath"))
    stats = defaultdict(int)

    for split in splits:
        split_dir = dataset_dir / split
        ann_file = split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            alt = list(split_dir.glob("*.coco.json")) if split_dir.is_dir() else []
            if alt:
                ann_file = alt[0]
            else:
                continue

        with open(ann_file) as f:
            data = json.load(f)

        cat_by_id = {c["id"]: c.get("name", f"cat_{c['id']}") for c in data.get("categories", [])}

        ann_by_image: dict[int, list] = defaultdict(list)
        for ann in data.get("annotations", []):
            ann_by_image[ann["image_id"]].append(ann)

        samples_to_add = []
        for img_obj in data.get("images", []):
            fname = img_obj.get("file_name", "")
            w = int(img_obj.get("width", 0) or 0)
            h = int(img_obj.get("height", 0) or 0)

            img_path = split_dir / fname
            if not img_path.exists():
                stats["missing"] += 1
                continue

            fp_str = str(img_path.resolve())
            if fp_str in existing_fps:
                stats["skipped_duplicate"] += 1
                continue
            existing_fps.add(fp_str)

            sample_tags = [split]
            if tags:
                sample_tags.extend(tags)
            sample = fo.Sample(filepath=fp_str, tags=sample_tags)
            detections = []
            polylines = []

            for ann in ann_by_image.get(img_obj.get("id"), []):
                raw_label = cat_by_id.get(ann.get("category_id"), "unknown")
                label = label_map.get(raw_label, raw_label)

                seg = ann.get("segmentation")
                has_seg = isinstance(seg, list) and any(
                    isinstance(p, list) and len(p) >= 6 for p in seg
                )

                if has_seg and w > 0 and h > 0:
                    for poly_pts in seg:
                        if not isinstance(poly_pts, list) or len(poly_pts) < 6:
                            continue
                        norm_pts = []
                        for i in range(0, len(poly_pts), 2):
                            norm_pts.append((
                                _clamp(float(poly_pts[i]) / w, 0, 1),
                                _clamp(float(poly_pts[i + 1]) / h, 0, 1),
                            ))
                        if len(norm_pts) >= 3:
                            polylines.append(fo.Polyline(
                                label=label,
                                points=[norm_pts],
                                closed=True,
                                filled=True,
                            ))
                            stats["polylines"] += 1

                bbox = ann.get("bbox")
                if bbox and len(bbox) >= 4 and w > 0 and h > 0:
                    rel = _normalize_bbox(bbox, w, h)
                    det = fo.Detection(label=label, bounding_box=rel)
                    if ann.get("area"):
                        det["area"] = ann["area"]
                    detections.append(det)
                    stats["detections"] += 1

            if detections:
                sample[label_field] = fo.Detections(detections=detections)
            if polylines:
                sample[f"{label_field}_polylines"] = fo.Polylines(polylines=polylines)

            samples_to_add.append(sample)
            stats["images_imported"] += 1

        if samples_to_add:
            ds.add_samples(samples_to_add)
        stats[f"split_{split}"] = len(samples_to_add)

    logger.info("Roboflow COCO 导入完成: %s", dict(stats))
    return dict(stats)


# ===================================================================
# D. Roboflow YOLO 格式导入
# ===================================================================

def _parse_yolo_label_file(
    label_path: Path,
    class_names: list[str],
    label_map: dict[str, str],
) -> tuple[list[fo.Detection], list[fo.Keypoint]]:
    """解析单个 YOLO 标注文件，返回 (detections, keypoints)。"""
    detections = []
    keypoints = []

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            label = label_map.get(label, label)

            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bbox = [
                _clamp(cx - bw / 2, 0, 1),
                _clamp(cy - bh / 2, 0, 1),
                _clamp(bw, 0, 1),
                _clamp(bh, 0, 1),
            ]

            remaining = parts[5:]
            if len(remaining) >= 6 and len(remaining) % 3 == 0:
                num_kps = len(remaining) // 3
                points = []
                confidences = []
                for ki in range(num_kps):
                    kx = float(remaining[ki * 3])
                    ky = float(remaining[ki * 3 + 1])
                    kv = float(remaining[ki * 3 + 2])
                    points.append((_clamp(kx, 0, 1), _clamp(ky, 0, 1)))
                    confidences.append(kv)
                kp = fo.Keypoint(label=label, points=points, confidence=confidences)
                keypoints.append(kp)
            else:
                det = fo.Detection(label=label, bounding_box=bbox)
                detections.append(det)

    return detections, keypoints


def ingest_roboflow_yolo(
    ds: fo.Dataset,
    dataset_dir: str | Path,
    label_field: str = "ground_truth",
    label_map: Optional[dict[str, str]] = None,
    splits: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    导入 Roboflow 导出的 YOLO 格式数据集。

    Roboflow YOLO 结构：
        <dataset_dir>/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    """
    dataset_dir = Path(dataset_dir).resolve()
    label_map = label_map or {}
    splits = splits or ["train", "valid", "test"]

    yaml_file = dataset_dir / "data.yaml"
    class_names = []
    if yaml_file.exists():
        import yaml
        with open(yaml_file) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        if isinstance(names, dict):
            class_names = [names[k] for k in sorted(names.keys())]
        elif isinstance(names, list):
            class_names = names

    existing_fps = set(ds.values("filepath"))
    stats = defaultdict(int)

    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.is_dir():
            continue

        images = _collect_images(img_dir, recursive=False)
        samples_to_add = []

        for img_path in images:
            fp_str = str(img_path.resolve())
            if fp_str in existing_fps:
                stats["skipped_duplicate"] += 1
                continue
            existing_fps.add(fp_str)

            sample_tags = [split]
            if tags:
                sample_tags.extend(tags)
            sample = fo.Sample(filepath=fp_str, tags=sample_tags)

            label_file = lbl_dir / (img_path.stem + ".txt")
            if label_file.exists():
                dets, kps = _parse_yolo_label_file(label_file, class_names, label_map)
                if dets:
                    sample[label_field] = fo.Detections(detections=dets)
                    stats["detections"] += len(dets)
                if kps:
                    sample[f"{label_field}_keypoints"] = fo.Keypoints(keypoints=kps)
                    stats["keypoints"] += len(kps)

            samples_to_add.append(sample)
            stats["images_imported"] += 1

        if samples_to_add:
            ds.add_samples(samples_to_add)
        stats[f"split_{split}"] = len(samples_to_add)

    logger.info("Roboflow YOLO 导入完成: %s", dict(stats))
    return dict(stats)


# ===================================================================
# E. CVAT 1.1 标准数据集导入
# ===================================================================

def ingest_cvat_xml(
    ds: fo.Dataset,
    xml_path: str | Path,
    image_dir: Optional[str | Path] = None,
    label_field: str = "ground_truth",
    label_map: Optional[dict[str, str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    解析 CVAT for images 1.1 XML 格式并导入到 FiftyOne。

    Args:
        ds: 目标 FiftyOne 数据集
        xml_path: CVAT annotations.xml 文件路径
        image_dir: 图片目录（默认为 xml_path 同级的 images/ 目录）
        label_field: 标签字段名
        label_map: 标签映射
    """
    xml_path = Path(xml_path).resolve()
    label_map = label_map or {}

    if image_dir is None:
        image_dir = xml_path.parent / "images"
        if not image_dir.is_dir():
            image_dir = xml_path.parent
    else:
        image_dir = Path(image_dir).resolve()

    tree = ET.parse(xml_path)
    root = tree.getroot()

    skeleton_defs = {}
    for label_el in root.iter("label"):
        name_el = label_el.find("name")
        type_el = label_el.find("type")
        if name_el is not None and type_el is not None and type_el.text == "skeleton":
            sublabels = []
            subs_el = label_el.find("sublabels")
            if subs_el is not None:
                for sl in subs_el.findall("sublabel"):
                    sl_name = sl.find("name")
                    if sl_name is not None:
                        sublabels.append(sl_name.text)
            skeleton_defs[name_el.text] = sublabels

    existing_fps = set(ds.values("filepath"))
    stats = defaultdict(int)
    samples_to_add = []

    for image_el in root.findall("image"):
        img_name = image_el.get("name", "")
        w = int(float(image_el.get("width", "0")))
        h = int(float(image_el.get("height", "0")))

        img_path = image_dir / img_name
        if not img_path.exists():
            for candidate_dir in [xml_path.parent, image_dir.parent]:
                candidate = candidate_dir / img_name
                if candidate.exists():
                    img_path = candidate
                    break

        if not img_path.exists():
            stats["missing"] += 1
            continue

        fp_str = str(img_path.resolve())
        if fp_str in existing_fps:
            stats["skipped_duplicate"] += 1
            continue
        existing_fps.add(fp_str)

        sample = fo.Sample(filepath=fp_str)
        if tags:
            sample.tags.extend(tags)
        detections = []
        keypoints_list = []
        polylines = []

        for box_el in image_el.findall("box"):
            raw_label = box_el.get("label", "unknown")
            label = label_map.get(raw_label, raw_label)
            xtl = float(box_el.get("xtl", 0))
            ytl = float(box_el.get("ytl", 0))
            xbr = float(box_el.get("xbr", 0))
            ybr = float(box_el.get("ybr", 0))

            if w > 0 and h > 0:
                bbox = [xtl / w, ytl / h, (xbr - xtl) / w, (ybr - ytl) / h]
            else:
                bbox = [xtl, ytl, xbr - xtl, ybr - ytl]

            rotation = float(box_el.get("rotation", 0))
            det = fo.Detection(label=label, bounding_box=bbox)
            if rotation != 0:
                det["rotation"] = rotation
            detections.append(det)
            stats["boxes"] += 1

        for poly_el in image_el.findall("polygon"):
            raw_label = poly_el.get("label", "unknown")
            label = label_map.get(raw_label, raw_label)
            pts_str = poly_el.get("points", "")
            if not pts_str:
                continue

            norm_pts = []
            for pt in pts_str.split(";"):
                parts = pt.strip().split(",")
                if len(parts) == 2:
                    px = float(parts[0]) / w if w > 0 else float(parts[0])
                    py = float(parts[1]) / h if h > 0 else float(parts[1])
                    norm_pts.append((_clamp(px, 0, 1), _clamp(py, 0, 1)))

            if len(norm_pts) >= 3:
                polylines.append(fo.Polyline(
                    label=label,
                    points=[norm_pts],
                    closed=True,
                    filled=True,
                ))
                stats["polygons"] += 1

        for polyline_el in image_el.findall("polyline"):
            raw_label = polyline_el.get("label", "unknown")
            label = label_map.get(raw_label, raw_label)
            pts_str = polyline_el.get("points", "")
            if not pts_str:
                continue

            norm_pts = []
            for pt in pts_str.split(";"):
                parts = pt.strip().split(",")
                if len(parts) == 2:
                    px = float(parts[0]) / w if w > 0 else float(parts[0])
                    py = float(parts[1]) / h if h > 0 else float(parts[1])
                    norm_pts.append((_clamp(px, 0, 1), _clamp(py, 0, 1)))

            if len(norm_pts) >= 2:
                polylines.append(fo.Polyline(
                    label=label,
                    points=[norm_pts],
                    closed=False,
                    filled=False,
                ))
                stats["polylines"] += 1

        for skel_el in image_el.findall("skeleton"):
            raw_label = skel_el.get("label", "unknown")
            label = label_map.get(raw_label, raw_label)

            point_elements = skel_el.findall("points")
            if not point_elements:
                continue

            expected_names = skeleton_defs.get(raw_label, [])
            points = []
            confidences = []

            name_to_pt: dict[str, tuple[float, float, float]] = {}
            for pt_el in point_elements:
                pt_label = pt_el.get("label", "")
                pts_str = pt_el.get("points", "")
                outside = pt_el.get("outside", "0") == "1"
                occluded = pt_el.get("occluded", "0") == "1"

                parts = pts_str.split(",")
                if len(parts) != 2:
                    continue
                px = float(parts[0]) / w if w > 0 else float(parts[0])
                py = float(parts[1]) / h if h > 0 else float(parts[1])

                if outside:
                    vis = 0.0
                elif occluded:
                    vis = 1.0
                else:
                    vis = 2.0

                name_to_pt[pt_label] = (_clamp(px, 0, 1), _clamp(py, 0, 1), vis)

            if expected_names:
                for kp_name in expected_names:
                    if kp_name in name_to_pt:
                        px, py, vis = name_to_pt[kp_name]
                        points.append((px, py))
                        confidences.append(vis)
                    else:
                        points.append((0.0, 0.0))
                        confidences.append(0.0)
            else:
                for pt_el in point_elements:
                    pt_label = pt_el.get("label", "")
                    if pt_label in name_to_pt:
                        px, py, vis = name_to_pt[pt_label]
                        points.append((px, py))
                        confidences.append(vis)

            if points:
                kp = fo.Keypoint(
                    label=label,
                    points=points,
                    confidence=confidences,
                )
                keypoints_list.append(kp)
                stats["skeletons"] += 1

        if detections:
            sample[label_field] = fo.Detections(detections=detections)
        if keypoints_list:
            sample[f"{label_field}_keypoints"] = fo.Keypoints(keypoints=keypoints_list)
        if polylines:
            sample[f"{label_field}_polylines"] = fo.Polylines(polylines=polylines)

        samples_to_add.append(sample)
        stats["images_imported"] += 1

    if samples_to_add:
        ds.add_samples(samples_to_add)

    logger.info("CVAT XML 导入完成: %s", dict(stats))
    return dict(stats)
