"""
多格式导出模块。

支持导出格式：
  - YOLO Detect  (纯框)
  - YOLO Pose    (关键点)
  - YOLO OBB     (旋转框)

特性：
  - 按类别过滤导出
  - 只导出有标签的图像
  - 自动生成 data.yaml
  - 支持按比例划分 train/valid/test
"""

from __future__ import annotations

import logging
import random
import shutil
import yaml
from pathlib import Path
from typing import Optional, Union

import cv2
import fiftyone as fo
import numpy as np

logger = logging.getLogger(__name__)

SplitsType = Union[str, dict[str, float]]


# ===================================================================
# 内部工具
# ===================================================================

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_labeled_view(
    ds: fo.Dataset | fo.DatasetView,
    label_field: str,
    classes: Optional[list[str]] = None,
) -> fo.DatasetView:
    """获取有标注的视图，可选按类别过滤。"""
    from fiftyone import ViewField as F

    schema = ds.get_field_schema()
    if label_field not in schema:
        raise ValueError(f"字段 '{label_field}' 不存在于数据集中")

    field = schema[label_field]
    if hasattr(field, 'document_type') and field.document_type:
        if issubclass(field.document_type, fo.Detections):
            sub = f"{label_field}.detections"
        elif issubclass(field.document_type, fo.Keypoints):
            sub = f"{label_field}.keypoints"
        elif issubclass(field.document_type, fo.Polylines):
            sub = f"{label_field}.polylines"
        else:
            sub = label_field
    else:
        sub = label_field

    view = ds.match(F(sub).length() > 0)

    if classes:
        view = view.filter_labels(label_field, F("label").is_in(classes))
        view = view.match(F(sub).length() > 0)

    return view


def load_class_names_from_yaml(data_yaml: str | Path | None) -> list[str]:
    """从 YOLO data.yaml 中读取按 class id 排序的类别名。"""
    if not data_yaml:
        return []
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return []
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        logger.warning("读取类别顺序失败: %s", yaml_path)
        return []

    names = data.get("names", {})
    if isinstance(names, dict):
        items = []
        for k, v in names.items():
            try:
                items.append((int(k), str(v)))
            except (TypeError, ValueError):
                continue
        return [name for _, name in sorted(items)]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return []


def _distinct_label_classes(ds: fo.Dataset | fo.DatasetView, label_field: str) -> list[str]:
    """返回字段中的类别名，按稳定字典序排列。"""
    schema = ds.get_field_schema()
    if label_field not in schema:
        return []

    field = schema[label_field]
    if hasattr(field, "document_type") and field.document_type:
        if issubclass(field.document_type, fo.Detections):
            raw = ds.distinct(f"{label_field}.detections.label")
        elif issubclass(field.document_type, fo.Keypoints):
            raw = ds.distinct(f"{label_field}.keypoints.label")
        elif issubclass(field.document_type, fo.Polylines):
            raw = ds.distinct(f"{label_field}.polylines.label")
        else:
            raw = []
    else:
        raw = []
    return sorted(c for c in raw if c is not None)


def _build_ordered_class_map(
    discovered_classes: list[str],
    classes: Optional[list[str]] = None,
    class_names: Optional[list[str]] = None,
) -> dict[str, int]:
    """构建类别映射，可用既有 class_names 锁定旧 class id 并追加新类别。"""
    export_filter = set(classes or [])
    discovered = [c for c in discovered_classes if not export_filter or c in export_filter]

    ordered: list[str] = []
    seen: set[str] = set()
    for name in class_names or []:
        if name in seen:
            continue
        ordered.append(name)
        seen.add(name)
    for name in discovered:
        if name not in seen:
            ordered.append(name)
            seen.add(name)

    return {name: idx for idx, name in enumerate(ordered)}


def _build_class_map(
    ds: fo.Dataset | fo.DatasetView,
    label_field: str,
    classes: Optional[list[str]] = None,
    class_names: Optional[list[str]] = None,
) -> dict[str, int]:
    """构建 class_name -> class_id 映射。"""
    all_classes = _distinct_label_classes(ds, label_field)
    return _build_ordered_class_map(all_classes, classes=classes, class_names=class_names)


def _write_data_yaml(
    output_dir: Path,
    class_map: dict[str, int],
    splits: list[str],
) -> None:
    """生成 data.yaml 配置文件。"""
    _ensure_dir(output_dir)
    names = {v: k for k, v in class_map.items()}
    data = {
        "path": str(output_dir),
        "names": names,
        "nc": len(class_map),
    }
    for split in splits:
        data[split] = f"{split}/images"

    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def _resolve_splits(
    view: fo.DatasetView,
    splits: SplitsType,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    将 splits 参数解析为 {split_name: [sample_ids]} 映射。

    splits 可以是：
      - str: 所有样本归入该 split（如 "train"）
      - dict[str, float]: 按比例随机划分（如 {"train": 0.8, "valid": 0.1, "test": 0.1}）
    """
    ids = list(view.values("id"))
    if isinstance(splits, str):
        return {splits: ids}

    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    result: dict[str, list[str]] = {}
    offset = 0
    split_names = list(splits.keys())
    for i, name in enumerate(split_names):
        if i == len(split_names) - 1:
            result[name] = ids[offset:]
        else:
            count = round(n * splits[name])
            result[name] = ids[offset : offset + count]
            offset += count
    return result


_YOLO_SPLIT_NAMES = {"valid": "val"}


def _normalize_split_ids(
    split_ids: dict[str, list[str]],
) -> dict[str, list[str]]:
    """将 split 名称映射为 YOLO 惯例（valid → val）。"""
    return {_YOLO_SPLIT_NAMES.get(k, k): v for k, v in split_ids.items()}


def _background_ids(ds: fo.Dataset | fo.DatasetView, background_tag: str | None) -> list[str]:
    if not background_tag:
        return []
    try:
        return list(ds.match_tags(background_tag).values("id"))
    except Exception:
        return []


def _add_background_to_train(
    split_ids: dict[str, list[str]],
    background_ids: list[str],
) -> dict[str, list[str]]:
    """将背景负样本强制放入 train，并从其他 split 去重。"""
    if not background_ids:
        return split_ids
    bg_set = set(background_ids)
    cleaned = {
        split_name: [sid for sid in ids if sid not in bg_set]
        for split_name, ids in split_ids.items()
    }
    train_ids = cleaned.setdefault("train", [])
    seen_train = set(train_ids)
    for sid in background_ids:
        if sid not in seen_train:
            train_ids.append(sid)
            seen_train.add(sid)
    return cleaned


def _copy_image(src: str | Path, dst_dir: Path) -> str:
    """复制图片到目标目录，返回文件名。"""
    src = Path(src)
    dst = dst_dir / src.name
    if dst.exists():
        stem, ext = src.stem, src.suffix
        n = 2
        while (dst_dir / f"{stem}_{n}{ext}").exists():
            n += 1
        dst = dst_dir / f"{stem}_{n}{ext}"
    shutil.copy2(src, dst)
    return dst.name


# ===================================================================
# YOLO Detect 导出
# ===================================================================

def export_yolo_detect(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    label_field: str = "ground_truth",
    classes: Optional[list[str]] = None,
    splits: SplitsType = "train",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    导出 YOLO Detect 格式（纯框）。

    Args:
        splits: "train" 全部导入单 split，或 {"train": 0.8, "valid": 0.1, "test": 0.1} 按比例划分。
    """
    output_dir = Path(output_dir).resolve()
    view = _get_labeled_view(ds, label_field, classes)
    class_map = _build_class_map(ds, label_field, classes, class_names=class_names)

    if not class_map:
        logger.warning("无可导出的类别")
        return {"exported": 0}

    split_ids = _normalize_split_ids(_resolve_splits(view, splits))
    all_split_names = list(split_ids.keys())
    _write_data_yaml(output_dir, class_map, all_split_names)

    total = 0
    per_split = {}
    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = view.select(ids)

        count = 0
        for sample in sub_view.iter_samples(progress=True):
            label_data = sample[label_field]
            if label_data is None:
                continue
            detections = label_data.detections
            if not detections:
                continue

            fname = _copy_image(sample.filepath, img_dir)
            stem = Path(fname).stem

            lines = []
            for det in detections:
                if classes and det.label not in classes:
                    continue
                if det.label not in class_map:
                    continue
                cls_id = class_map[det.label]
                x, y, w, h = det.bounding_box
                cx = x + w / 2
                cy = y + h / 2
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if lines:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count

    logger.info("YOLO Detect 导出: %d 张图片, %d 个类别", total, len(class_map))
    return {"exported": total, "per_split": per_split, "classes": list(class_map.keys()), "output_dir": str(output_dir)}


export_yolov8_detect = export_yolo_detect


# ===================================================================
# YOLO Pose 导出
# ===================================================================

def export_yolo_pose(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    det_field: str = "ground_truth",
    kp_field: str = "ground_truth_keypoints",
    classes: Optional[list[str]] = None,
    splits: SplitsType = "train",
    num_keypoints: Optional[int] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    导出 YOLO Pose 格式（关键点）。

    每行格式: class_id cx cy w h kx1 ky1 kv1 kx2 ky2 kv2 ...
    """
    output_dir = Path(output_dir).resolve()

    has_kp = kp_field in ds.get_field_schema()
    has_det = det_field in ds.get_field_schema()

    if not has_kp:
        logger.warning("关键点字段 '%s' 不存在", kp_field)
        return {"exported": 0}

    from fiftyone import ViewField as F
    view = ds.match(F(f"{kp_field}.keypoints").length() > 0)
    if classes:
        view = view.filter_labels(kp_field, F("label").is_in(classes))
        view = view.match(F(f"{kp_field}.keypoints").length() > 0)

    all_kp_classes = _distinct_label_classes(ds, kp_field)
    class_map = _build_ordered_class_map(all_kp_classes, classes=classes, class_names=class_names)

    if not class_map:
        return {"exported": 0}

    if num_keypoints is None:
        for sample in view.head(10):
            kps = sample[kp_field]
            if kps and kps.keypoints:
                num_keypoints = len(kps.keypoints[0].points)
                break
        if num_keypoints is None:
            num_keypoints = 0

    split_ids = _normalize_split_ids(_resolve_splits(view, splits))
    all_split_names = list(split_ids.keys())

    _ensure_dir(output_dir)
    data_yaml = {
        "path": str(output_dir),
        "names": {v: k for k, v in class_map.items()},
        "nc": len(class_map),
        "kpt_shape": [num_keypoints, 3],
    }
    for sn in all_split_names:
        data_yaml[sn] = f"{sn}/images"
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    total = 0
    per_split = {}
    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = view.select(ids)

        count = 0
        for sample in sub_view.iter_samples(progress=True):
            kps_data = sample[kp_field]
            if kps_data is None or not kps_data.keypoints:
                continue

            det_data = sample[det_field] if has_det and sample[det_field] else None
            det_map: dict[str, list] = {}
            if det_data:
                for det in det_data.detections:
                    det_map.setdefault(det.label, []).append(det.bounding_box)

            fname = _copy_image(sample.filepath, img_dir)
            stem = Path(fname).stem

            lines = []
            for kp in kps_data.keypoints:
                if classes and kp.label not in classes:
                    continue
                if kp.label not in class_map:
                    continue
                cls_id = class_map[kp.label]

                bboxes = det_map.get(kp.label, [])
                if bboxes:
                    bbox = bboxes.pop(0)
                    x, y, w, h = bbox
                else:
                    xs = [p[0] for p in kp.points if p[0] > 0]
                    ys = [p[1] for p in kp.points if p[1] > 0]
                    if xs and ys:
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        margin = 0.02
                        x = max(0, x_min - margin)
                        y = max(0, y_min - margin)
                        w = min(1, x_max - x_min + 2 * margin)
                        h = min(1, y_max - y_min + 2 * margin)
                    else:
                        continue

                cx = x + w / 2
                cy = y + h / 2
                parts = [f"{cls_id}", f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]

                for pi, pt in enumerate(kp.points):
                    kx, ky = pt[0], pt[1]
                    if kp.confidence and pi < len(kp.confidence):
                        kv = kp.confidence[pi]
                    else:
                        kv = 2.0
                    parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{kv:.0f}"])

                lines.append(" ".join(parts))

            if lines:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count

    logger.info("YOLO Pose 导出: %d 张图片", total)
    return {"exported": total, "per_split": per_split, "classes": list(class_map.keys()), "num_keypoints": num_keypoints}


export_yolov8_pose = export_yolo_pose


# ===================================================================
# YOLO Pose 导出（从四边形 Polylines 转换）
# ===================================================================

def _order_quad_tl_tr_br_bl(
    pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """将 4 点排序为 [tl, tr, br, bl]（左上、右上、右下、左下）。"""
    arr = np.array(pts, dtype=np.float64)
    s = arr.sum(axis=1)       # x+y 最小 → 左上, 最大 → 右下
    d = arr[:, 0] - arr[:, 1]  # x-y 最大 → 右上, 最小 → 左下
    tl = arr[int(np.argmin(s))]
    br = arr[int(np.argmax(s))]
    tr = arr[int(np.argmax(d))]
    bl = arr[int(np.argmin(d))]
    return [
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    ]


def _compute_kpt_visibility(
    norm_pts: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    edge_threshold: float = 5.0,
    occlusion_flags: Optional[list[bool]] = None,
) -> list[int]:
    """根据关键点到图像边界的距离和遮挡属性计算可见性。

    优先级：贴近边界 → v=0（坐标归零）> 被遮挡 → v=1（保留坐标）> 可见 → v=2

    Args:
        occlusion_flags: 长度与 norm_pts 相同的布尔列表，True 表示该点被遮挡。
            对应 CVAT 的 1_occu~4_occu 属性（左上/右上/右下/左下）。

    Returns:
        visibility 列表：2=可见, 1=被遮挡, 0=不可见（贴近边界，坐标将归零）
    """
    result: list[int] = []
    for i, (nx, ny) in enumerate(norm_pts):
        px, py = nx * img_w, ny * img_h
        min_dist = min(px, img_w - px, py, img_h - py)
        if min_dist < edge_threshold:
            result.append(0)
        elif occlusion_flags and i < len(occlusion_flags) and occlusion_flags[i]:
            result.append(1)
        else:
            result.append(2)
    return result


def _get_occlusion_flags(poly) -> list[bool]:
    """从 polyline 的 CVAT 属性中提取四角遮挡标记 [tl, tr, br, bl]。

    属性名 1_occu~4_occu 分别对应左上、右上、右下、左下角点。
    """
    flags: list[bool] = []
    for i in range(1, 5):
        attr_name = f"{i}_occu"
        val = None
        try:
            val = getattr(poly, attr_name, None)
        except Exception:
            pass
        if val is None:
            try:
                val = poly.get_attribute_value(attr_name)
            except Exception:
                pass
        if isinstance(val, bool):
            flags.append(val)
        elif isinstance(val, str):
            flags.append(val.lower() in ("true", "1", "yes"))
        else:
            flags.append(bool(val) if val is not None else False)
    return flags


def _get_image_dims(sample) -> tuple[int, int]:
    """获取样本图片尺寸 (width, height)，优先用 metadata，降级读文件。"""
    if sample.metadata and hasattr(sample.metadata, "width"):
        return sample.metadata.width, sample.metadata.height
    try:
        img = cv2.imread(sample.filepath)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass
    return 0, 0


def export_yolo_pose_from_polylines(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    label_field: str = "ground_truth_polylines",
    classes: Optional[list[str]] = None,
    splits: SplitsType = "train",
    edge_threshold: float = 5.0,
    bbox_margin: float = 0.0,
    class_names: Optional[list[str]] = None,
    include_background: bool = True,
    background_tag: str = "background",
) -> dict:
    """
    从四边形 Polylines 导出 YOLO Pose 格式。

    将每个 4 点多边形转换为：
      - 关键点：tl, tr, br, bl（自动排序）
      - 包围框：从 polygon 顶点直接计算（bbox_margin>0 时外扩）
      - 可见性：距图像边界 < edge_threshold 像素 → v=0 坐标归零（不可见），否则 v=2（可见）

    Args:
        label_field: Polylines 类型的标签字段名
        edge_threshold: 关键点距图像边界小于此像素值时标记为不可见（默认 5.0）
        bbox_margin: 包围框外扩的归一化边距（默认 0.0 不外扩）
        splits: 同其他导出函数，支持 str 或 dict 比例划分

    输出 YOLO-Pose 行格式 (kpt_shape=[4,3]):
        cls_id cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
        v=2 可见 | v=0 不可见（坐标归零）
    """
    output_dir = Path(output_dir).resolve()

    from fiftyone import ViewField as F
    view = ds.match(F(f"{label_field}.polylines").length() > 0)
    if classes:
        view = view.filter_labels(label_field, F("label").is_in(classes))
        view = view.match(F(f"{label_field}.polylines").length() > 0)

    all_classes = _distinct_label_classes(ds, label_field)
    class_map = _build_ordered_class_map(all_classes, classes=classes, class_names=class_names)

    if not class_map:
        logger.warning("无可导出的类别")
        return {"exported": 0}

    try:
        view.compute_metadata(overwrite=False)
    except Exception:
        logger.info("compute_metadata 跳过，将从图片文件读取尺寸")

    bg_ids = _background_ids(ds, background_tag) if include_background else []
    bg_set = set(bg_ids)
    export_ids = list(dict.fromkeys(list(view.values("id")) + bg_ids))
    export_view = ds.select(export_ids) if bg_ids else view

    split_ids = _normalize_split_ids(_resolve_splits(view, splits))
    split_ids = _add_background_to_train(split_ids, bg_ids)
    all_split_names = list(split_ids.keys())

    _ensure_dir(output_dir)
    data_yaml = {
        "path": str(output_dir),
        "names": {v: k for k, v in class_map.items()},
        "nc": len(class_map),
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    for sn in all_split_names:
        data_yaml[sn] = f"{sn}/images"
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    total = 0
    total_edge_invisible = 0
    total_occluded = 0
    per_split: dict[str, int] = {}

    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = export_view.select(ids)

        count = 0
        background_count = 0
        for sample in sub_view.iter_samples(progress=True):
            if sample.id in bg_set:
                fname = _copy_image(sample.filepath, img_dir)
                stem = Path(fname).stem
                (lbl_dir / f"{stem}.txt").write_text("")
                count += 1
                background_count += 1
                continue

            poly_data = sample[label_field]
            if poly_data is None or not poly_data.polylines:
                continue

            img_w, img_h = _get_image_dims(sample)
            if img_w <= 0 or img_h <= 0:
                logger.warning("无法获取图片尺寸: %s", sample.filepath)
                continue

            fname = _copy_image(sample.filepath, img_dir)
            stem = Path(fname).stem
            lines: list[str] = []

            for poly in poly_data.polylines:
                if classes and poly.label not in classes:
                    continue
                if poly.label not in class_map:
                    continue

                pts = poly.points[0] if poly.points else []
                if len(pts) != 4:
                    continue

                cls_id = class_map[poly.label]
                ordered = _order_quad_tl_tr_br_bl(pts)
                occu_flags = _get_occlusion_flags(poly)
                vis = _compute_kpt_visibility(
                    ordered, img_w, img_h, edge_threshold,
                    occlusion_flags=occu_flags if occu_flags else None,
                )
                total_edge_invisible += sum(1 for v in vis if v == 0)
                total_occluded += sum(1 for v in vis if v == 1)

                xs = [p[0] for p in ordered]
                ys = [p[1] for p in ordered]
                x_min = max(0.0, min(xs) - bbox_margin)
                y_min = max(0.0, min(ys) - bbox_margin)
                x_max = min(1.0, max(xs) + bbox_margin)
                y_max = min(1.0, max(ys) + bbox_margin)
                bw = x_max - x_min
                bh = y_max - y_min
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2

                parts = [f"{cls_id}", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
                for (kx, ky), kv in zip(ordered, vis):
                    if kv == 0:
                        parts.extend(["0.000000", "0.000000", "0"])
                    else:
                        parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{kv}"])
                lines.append(" ".join(parts))

            if lines:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count
        if background_count:
            logger.info("YOLO Pose 背景负样本导出到 %s: %d 张", split_name, background_count)

    logger.info(
        "YOLO Pose (from polylines) 导出: %d 张图片, 边界不可见关键点: %d, 遮挡关键点: %d",
        total, total_edge_invisible, total_occluded,
    )
    return {
        "exported": total,
        "per_split": per_split,
        "classes": list(class_map.keys()),
        "kpt_shape": [4, 3],
        "edge_invisible_keypoints": total_edge_invisible,
        "occluded_keypoints": total_occluded,
        "background_images": len(bg_ids),
        "background_tag": background_tag if include_background else None,
        "output_dir": str(output_dir),
    }


# ===================================================================
# YOLO Pose 混合导出（四边形 Pose + Detection-only）
# ===================================================================

def export_yolo_mixed_pose(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    pose_field: str = "ground_truth_polylines",
    det_field: str = "ground_truth",
    detection_classes: Optional[list[str]] = None,
    classes: Optional[list[str]] = None,
    splits: SplitsType = "train",
    edge_threshold: float = 5.0,
    bbox_margin: float = 0.0,
    class_names: Optional[list[str]] = None,
    include_background: bool = True,
    background_tag: str = "background",
) -> dict:
    """
    导出 YOLO Pose 混合训练格式：四边形转 4 个关键点 + 检测框全 0 关键点。

    用途：
      - pose_field 中的 chuan/tian 等四边形 Polylines 正常转为 tl/tr/br/bl 关键点
      - det_field 中 detection_classes 指定的类别（默认 pallet）写为 bbox + 4 个全 0 关键点

    输出仍是 YOLO task=pose 可训练的数据集，kpt_shape 固定为 [4, 3]。
    """
    output_dir = Path(output_dir).resolve()
    schema = ds.get_field_schema()
    has_pose = pose_field in schema
    has_det = det_field in schema
    if not has_pose and not has_det:
        logger.warning("四边形字段 '%s' 和检测框字段 '%s' 均不存在", pose_field, det_field)
        return {"exported": 0}

    detection_class_set = set(detection_classes or ["pallet"])
    export_filter = set(classes or [])
    if export_filter:
        detection_class_set &= export_filter

    pose_classes = _distinct_label_classes(ds, pose_field) if has_pose else []
    det_classes = _distinct_label_classes(ds, det_field) if has_det else []
    mixed_classes = pose_classes + [c for c in det_classes if c in detection_class_set]
    class_map = _build_ordered_class_map(mixed_classes, classes=classes, class_names=class_names)
    if not class_map:
        logger.warning("无可导出的混合训练类别")
        return {"exported": 0}

    eligible_ids: list[str] = []
    for sample in ds.iter_samples(progress=True):
        has_lines = False
        if has_pose:
            poly_data = sample[pose_field]
            polylines = getattr(poly_data, "polylines", None)
            if polylines:
                for poly in polylines:
                    if poly.label in class_map and (not export_filter or poly.label in export_filter):
                        pts = poly.points[0] if poly.points else []
                        if len(pts) == 4:
                            has_lines = True
                            break
        if not has_lines and has_det and detection_class_set:
            det_data = sample[det_field]
            detections = getattr(det_data, "detections", None)
            if detections:
                for det in detections:
                    if det.label in detection_class_set and det.label in class_map:
                        has_lines = True
                        break
        if has_lines:
            eligible_ids.append(sample.id)

    bg_ids = _background_ids(ds, background_tag) if include_background else []
    bg_set = set(bg_ids)
    if not eligible_ids and not bg_ids:
        logger.warning("没有可导出的混合训练样本")
        return {"exported": 0}

    export_ids = list(dict.fromkeys(eligible_ids + bg_ids))
    view = ds.select(export_ids)
    try:
        view.compute_metadata(overwrite=False)
    except Exception:
        logger.info("compute_metadata 跳过，将从图片文件读取尺寸")

    positive_view = ds.select(eligible_ids) if eligible_ids else ds.select([])
    split_ids = _normalize_split_ids(_resolve_splits(positive_view, splits))
    split_ids = _add_background_to_train(split_ids, bg_ids)
    all_split_names = list(split_ids.keys())

    _ensure_dir(output_dir)
    data_yaml = {
        "path": str(output_dir),
        "names": {v: k for k, v in class_map.items()},
        "nc": len(class_map),
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    for sn in all_split_names:
        data_yaml[sn] = f"{sn}/images"
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    total = 0
    pose_objects = 0
    detection_only_objects = 0
    total_edge_invisible = 0
    total_occluded = 0
    per_split: dict[str, int] = {}
    zero_kpts = "0.000000 0.000000 0 " * 4
    zero_kpts = zero_kpts.strip()

    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = view.select(ids)

        count = 0
        background_count = 0
        for sample in sub_view.iter_samples(progress=True):
            if sample.id in bg_set:
                fname = _copy_image(sample.filepath, img_dir)
                stem = Path(fname).stem
                (lbl_dir / f"{stem}.txt").write_text("")
                count += 1
                background_count += 1
                continue

            lines: list[str] = []
            img_w, img_h = _get_image_dims(sample)

            if has_pose:
                poly_data = sample[pose_field]
                polylines = getattr(poly_data, "polylines", None)
                if polylines:
                    if img_w <= 0 or img_h <= 0:
                        logger.warning("无法获取图片尺寸，跳过四边形关键点: %s", sample.filepath)
                    else:
                        for poly in polylines:
                            if poly.label not in class_map:
                                continue
                            if export_filter and poly.label not in export_filter:
                                continue

                            pts = poly.points[0] if poly.points else []
                            if len(pts) != 4:
                                continue

                            ordered = _order_quad_tl_tr_br_bl(pts)
                            occu_flags = _get_occlusion_flags(poly)
                            vis = _compute_kpt_visibility(
                                ordered, img_w, img_h, edge_threshold,
                                occlusion_flags=occu_flags if occu_flags else None,
                            )
                            total_edge_invisible += sum(1 for v in vis if v == 0)
                            total_occluded += sum(1 for v in vis if v == 1)

                            xs = [p[0] for p in ordered]
                            ys = [p[1] for p in ordered]
                            x_min = max(0.0, min(xs) - bbox_margin)
                            y_min = max(0.0, min(ys) - bbox_margin)
                            x_max = min(1.0, max(xs) + bbox_margin)
                            y_max = min(1.0, max(ys) + bbox_margin)
                            bw = x_max - x_min
                            bh = y_max - y_min
                            cx = (x_min + x_max) / 2
                            cy = (y_min + y_max) / 2

                            parts = [
                                f"{class_map[poly.label]}",
                                f"{cx:.6f}",
                                f"{cy:.6f}",
                                f"{bw:.6f}",
                                f"{bh:.6f}",
                            ]
                            for (kx, ky), kv in zip(ordered, vis):
                                if kv == 0:
                                    parts.extend(["0.000000", "0.000000", "0"])
                                else:
                                    parts.extend([f"{kx:.6f}", f"{ky:.6f}", f"{kv}"])
                            lines.append(" ".join(parts))
                            pose_objects += 1

            if has_det and detection_class_set:
                det_data = sample[det_field]
                detections = getattr(det_data, "detections", None)
                if detections:
                    for det in detections:
                        if det.label not in detection_class_set or det.label not in class_map:
                            continue
                        x, y, w, h = det.bounding_box
                        cx = x + w / 2
                        cy = y + h / 2
                        lines.append(
                            f"{class_map[det.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {zero_kpts}"
                        )
                        detection_only_objects += 1

            if lines:
                fname = _copy_image(sample.filepath, img_dir)
                stem = Path(fname).stem
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count
        if background_count:
            logger.info("YOLO Mixed Pose 背景负样本导出到 %s: %d 张", split_name, background_count)

    logger.info(
        "YOLO Mixed Pose 导出: %d 张图片, pose=%d, detection-only=%d",
        total, pose_objects, detection_only_objects,
    )
    return {
        "exported": total,
        "per_split": per_split,
        "classes": list(class_map.keys()),
        "kpt_shape": [4, 3],
        "detection_classes": sorted(detection_class_set),
        "pose_objects": pose_objects,
        "detection_only_objects": detection_only_objects,
        "edge_invisible_keypoints": total_edge_invisible,
        "occluded_keypoints": total_occluded,
        "background_images": len(bg_ids),
        "background_tag": background_tag if include_background else None,
        "output_dir": str(output_dir),
    }


# ===================================================================
# YOLO OBB 导出
# ===================================================================

def export_yolo_obb(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
    label_field: str = "ground_truth",
    obb_field: Optional[str] = None,
    classes: Optional[list[str]] = None,
    splits: SplitsType = "train",
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    导出 YOLO OBB 格式（旋转框）。

    每行格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
    顶点为归一化坐标。

    如果提供 obb_field（Polylines 字段），从中提取四边形顶点。
    否则从 label_field（Detections 字段）中构造，使用 rotation 属性。
    """
    output_dir = Path(output_dir).resolve()

    if obb_field and obb_field in ds.get_field_schema():
        return _export_obb_from_polylines(ds, output_dir, obb_field, classes, splits, class_names)
    else:
        return _export_obb_from_detections(ds, output_dir, label_field, classes, splits, class_names)


export_yolov8_obb = export_yolo_obb


def _export_obb_from_polylines(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: Path,
    obb_field: str,
    classes: Optional[list[str]],
    splits: SplitsType,
    class_names: Optional[list[str]] = None,
) -> dict:
    from fiftyone import ViewField as F
    view = ds.match(F(f"{obb_field}.polylines").length() > 0)
    if classes:
        view = view.filter_labels(obb_field, F("label").is_in(classes))
        view = view.match(F(f"{obb_field}.polylines").length() > 0)

    all_classes = _distinct_label_classes(ds, obb_field)
    class_map = _build_ordered_class_map(all_classes, classes=classes, class_names=class_names)

    if not class_map:
        return {"exported": 0}

    split_ids = _normalize_split_ids(_resolve_splits(view, splits))
    all_split_names = list(split_ids.keys())
    _write_data_yaml(output_dir, class_map, all_split_names)

    total = 0
    per_split = {}
    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = view.select(ids)

        count = 0
        for sample in sub_view.iter_samples(progress=True):
            poly_data = sample[obb_field]
            if poly_data is None or not poly_data.polylines:
                continue

            fname = _copy_image(sample.filepath, img_dir)
            stem = Path(fname).stem
            lines = []

            for poly in poly_data.polylines:
                if classes and poly.label not in classes:
                    continue
                if poly.label not in class_map:
                    continue

                pts = poly.points[0] if poly.points else []
                if len(pts) != 4:
                    continue

                cls_id = class_map[poly.label]
                coords = []
                for pt in pts:
                    coords.extend([f"{pt[0]:.6f}", f"{pt[1]:.6f}"])
                lines.append(f"{cls_id} " + " ".join(coords))

            if lines:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count

    return {"exported": total, "per_split": per_split, "classes": list(class_map.keys())}


def _export_obb_from_detections(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: Path,
    label_field: str,
    classes: Optional[list[str]],
    splits: SplitsType,
    class_names: Optional[list[str]] = None,
) -> dict:
    """从 Detections 字段（带 rotation 属性）导出 OBB 格式。"""
    view = _get_labeled_view(ds, label_field, classes)
    class_map = _build_class_map(ds, label_field, classes, class_names=class_names)

    if not class_map:
        return {"exported": 0}

    split_ids = _normalize_split_ids(_resolve_splits(view, splits))
    all_split_names = list(split_ids.keys())
    _write_data_yaml(output_dir, class_map, all_split_names)

    total = 0
    per_split = {}
    for split_name, ids in split_ids.items():
        if not ids:
            per_split[split_name] = 0
            continue
        img_dir = _ensure_dir(output_dir / split_name / "images")
        lbl_dir = _ensure_dir(output_dir / split_name / "labels")
        sub_view = view.select(ids)

        count = 0
        for sample in sub_view.iter_samples(progress=True):
            label_data = sample[label_field]
            if label_data is None or not label_data.detections:
                continue

            fname = _copy_image(sample.filepath, img_dir)
            stem = Path(fname).stem
            lines = []

            for det in label_data.detections:
                if classes and det.label not in classes:
                    continue
                if det.label not in class_map:
                    continue

                cls_id = class_map[det.label]
                x, y, w, h = det.bounding_box
                rotation = getattr(det, "rotation", None) or det.get_attribute_value("rotation", 0)
                if rotation is None:
                    rotation = 0

                cx = x + w / 2
                cy = y + h / 2
                angle = float(rotation) * np.pi / 180

                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                hw, hh = w / 2, h / 2

                corners = [
                    (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)
                ]
                rotated = []
                for dx, dy in corners:
                    rx = cx + dx * cos_a - dy * sin_a
                    ry = cy + dx * sin_a + dy * cos_a
                    rotated.extend([f"{rx:.6f}", f"{ry:.6f}"])

                lines.append(f"{cls_id} " + " ".join(rotated))

            if lines:
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                count += 1

        per_split[split_name] = count
        total += count

    return {"exported": total, "per_split": per_split, "classes": list(class_map.keys())}


# ===================================================================
# 纯图片导出
# ===================================================================

def export_images_only(
    ds: fo.Dataset | fo.DatasetView,
    output_dir: str | Path,
) -> dict:
    """将样本的图片文件复制到输出目录，不导出标签和 data.yaml。"""
    output_dir = Path(output_dir).resolve()
    img_dir = _ensure_dir(output_dir)

    total = 0
    for sample in ds.iter_samples(progress=True):
        fp = Path(sample.filepath)
        if fp.exists():
            _copy_image(fp, img_dir)
            total += 1

    logger.info("纯图片导出: %d 张图片 -> %s", total, output_dir)
    return {"exported": total, "output_dir": str(output_dir)}
