"""
自动化处理与清洗模块。

包括：
  - 清除无标注图像
  - 检测并清除尺寸异常 / 损坏图像
  - 重复图像检测与清理（精确哈希 + 近似嵌入）
  - YOLO 模型自动预标注（detect / pose / obb）
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2
import fiftyone as fo
import numpy as np

from .config import CONFIG

logger = logging.getLogger(__name__)


# ===================================================================
# 1. 无标注图像清理
# ===================================================================

def find_unlabeled_samples(
    ds: fo.Dataset,
    label_fields: Optional[list[str]] = None,
) -> fo.DatasetView:
    """
    查找所有标签字段均为空的样本。
    如果 label_fields 未指定，自动扫描数据集中的所有标签字段。
    """
    if label_fields is None:
        import fiftyone.core.fields as fof
        label_fields = []
        for name, field in ds.get_field_schema().items():
            if isinstance(field, fof.EmbeddedDocumentField):
                doc_type = field.document_type
                if doc_type and issubclass(doc_type, (
                    fo.Detections, fo.Keypoints, fo.Polylines,
                    fo.Classifications, fo.Classification,
                )):
                    label_fields.append(name)

    if not label_fields:
        return ds.view()

    from fiftyone import ViewField as F
    expr = None
    for field_name in label_fields:
        schema = ds.get_field_schema()
        if field_name not in schema:
            continue

        field = schema[field_name]
        sub_field = None
        if hasattr(field, 'document_type') and field.document_type:
            if issubclass(field.document_type, fo.Detections):
                sub_field = f"{field_name}.detections"
            elif issubclass(field.document_type, fo.Keypoints):
                sub_field = f"{field_name}.keypoints"
            elif issubclass(field.document_type, fo.Polylines):
                sub_field = f"{field_name}.polylines"

        if sub_field:
            cond = F(field_name).exists(False) | (F(sub_field).length() == 0)
        else:
            cond = F(field_name).exists(False)

        expr = cond if expr is None else (expr & cond)

    if expr is None:
        return ds.view()
    return ds.match(expr)


def delete_unlabeled_samples(
    ds: fo.Dataset,
    label_fields: Optional[list[str]] = None,
    physical: bool = False,
) -> int:
    """删除无标注样本。physical=True 时同时删除磁盘文件。"""
    view = find_unlabeled_samples(ds, label_fields)
    count = len(view)
    if count == 0:
        return 0

    if physical:
        for sample in view:
            p = Path(sample.filepath)
            if p.exists():
                p.unlink()

    ds.delete_samples(view)
    logger.info("删除 %d 个无标注样本 (physical=%s)", count, physical)
    return count


# ===================================================================
# 2. 尺寸异常 / 损坏图像检测
# ===================================================================

def find_corrupt_or_abnormal(
    ds: fo.Dataset,
    min_size: int = 32,
    max_size: int = 20000,
) -> list[str]:
    """
    检测损坏或尺寸异常的图像，返回需删除的 sample IDs。

    判定条件：
      - 无法用 OpenCV 读取
      - 宽或高 < min_size
      - 宽或高 > max_size
    """
    bad_ids = []
    for sample in ds.iter_samples(progress=True):
        fp = sample.filepath
        try:
            img = cv2.imread(fp)
            if img is None:
                bad_ids.append(sample.id)
                sample.tags.append("corrupt")
                sample.save()
                continue
            h, w = img.shape[:2]
            if w < min_size or h < min_size or w > max_size or h > max_size:
                bad_ids.append(sample.id)
                sample.tags.append("abnormal_size")
                sample.save()
        except Exception:
            bad_ids.append(sample.id)
            sample.tags.append("corrupt")
            sample.save()

    logger.info("检测到 %d 个损坏/异常样本", len(bad_ids))
    return bad_ids


# ===================================================================
# 3. 重复图像检测与清理
# ===================================================================

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            result[self.find(i)].append(i)
        return {k: v for k, v in result.items() if len(v) > 1}


def _file_md5(filepath: str | Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(ds: fo.Dataset) -> list[list[str]]:
    """通过 MD5 哈希查找精确重复。返回重复组（每组为 sample ID 列表）。"""
    hash_map: dict[str, list[str]] = defaultdict(list)
    for sample in ds.iter_samples(progress=True):
        fp = sample.filepath
        if Path(fp).exists():
            h = _file_md5(fp)
            hash_map[h].append(sample.id)

    groups = [ids for ids in hash_map.values() if len(ids) > 1]
    logger.info("发现 %d 组精确重复", len(groups))
    return groups


def find_near_duplicates(
    ds: fo.Dataset,
    threshold: float = 0.03,
    model_name: str = "clip-vit-base32-torch",
) -> list[list[str]]:
    """
    基于图像嵌入的近似去重。

    Args:
        threshold: 余弦距离阈值 (0~1, 越小越严格)
        model_name: FiftyOne zoo 模型名
    """
    import fiftyone.zoo as foz

    model = foz.load_zoo_model(model_name)
    embeddings = ds.compute_embeddings(model)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)
    sim = embeddings_norm @ embeddings_norm.T
    dist = 1.0 - sim

    n = len(ds)
    uf = _UnionFind(n)
    sample_ids = ds.values("id")

    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] < threshold:
                uf.union(i, j)

    groups = []
    for indices in uf.groups().values():
        groups.append([sample_ids[i] for i in indices])

    logger.info("发现 %d 组近似重复 (threshold=%.3f)", len(groups), threshold)
    return groups


def tag_duplicates(
    ds: fo.Dataset,
    groups: list[list[str]],
    tag: str = "duplicate",
) -> int:
    """给重复组中除第一个之外的样本打标签。"""
    count = 0
    for group in groups:
        for sid in group[1:]:
            sample = ds[sid]
            if tag not in sample.tags:
                sample.tags.append(tag)
                sample.save()
                count += 1
    return count


def delete_duplicates(
    ds: fo.Dataset,
    groups: list[list[str]],
    physical: bool = False,
) -> int:
    """删除重复组中除第一个之外的所有样本。"""
    to_delete = []
    for group in groups:
        to_delete.extend(group[1:])

    if not to_delete:
        return 0

    if physical:
        view = ds.select(to_delete)
        for sample in view:
            p = Path(sample.filepath)
            if p.exists():
                p.unlink()

    ds.delete_samples(to_delete)
    logger.info("删除 %d 个重复样本 (physical=%s)", len(to_delete), physical)
    return len(to_delete)


# ===================================================================
# 4. YOLO 模型自动预标注
# ===================================================================

def auto_predict_yolo(
    ds: fo.Dataset,
    model_path: str,
    pred_field: str = "predictions",
    conf_threshold: float = 0.25,
    task: str = "detect",
    view: Optional[fo.DatasetView] = None,
) -> dict:
    """
    使用 YOLO 模型对数据集进行自动预标注。

    Args:
        model_path: YOLO 模型权重路径 (.pt)
        pred_field: 预测结果存储字段名
        conf_threshold: 置信度阈值
        task: 任务类型 - "detect", "pose", "obb"
        view: 可选的数据集视图（仅对视图中的样本预标注）
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    target = view if view is not None else ds
    stats = defaultdict(int)

    for sample in target.iter_samples(progress=True, autosave=True):
        fp = sample.filepath
        results = model(fp, conf=conf_threshold, verbose=False)
        if not results:
            continue

        result = results[0]

        if task == "detect":
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    h_img, w_img = result.orig_shape
                    bbox = [x1 / w_img, y1 / h_img, (x2 - x1) / w_img, (y2 - y1) / h_img]
                    detections.append(fo.Detection(
                        label=label, bounding_box=bbox, confidence=conf,
                    ))
                    stats["detections"] += 1
            if detections:
                sample[pred_field] = fo.Detections(detections=detections)

        elif task == "pose":
            detections = []
            keypoints_list = []
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    h_img, w_img = result.orig_shape
                    bbox = [x1 / w_img, y1 / h_img, (x2 - x1) / w_img, (y2 - y1) / h_img]
                    detections.append(fo.Detection(
                        label=label, bounding_box=bbox, confidence=conf,
                    ))
                    stats["detections"] += 1

                    if result.keypoints is not None and i < len(result.keypoints):
                        kp_data = result.keypoints[i]
                        xy = kp_data.xy[0] if hasattr(kp_data, 'xy') else kp_data.data[0, :, :2]
                        kp_conf = kp_data.conf[0] if hasattr(kp_data, 'conf') and kp_data.conf is not None else None

                        points = []
                        confs = []
                        for ki in range(xy.shape[0]):
                            px = float(xy[ki, 0]) / w_img
                            py = float(xy[ki, 1]) / h_img
                            points.append((px, py))
                            if kp_conf is not None:
                                confs.append(float(kp_conf[ki]))
                            else:
                                confs.append(conf)

                        keypoints_list.append(fo.Keypoint(
                            label=label, points=points, confidence=confs,
                        ))
                        stats["keypoints"] += 1

            if detections:
                sample[pred_field] = fo.Detections(detections=detections)
            if keypoints_list:
                sample[f"{pred_field}_keypoints"] = fo.Keypoints(keypoints=keypoints_list)

        elif task == "obb":
            detections = []
            polylines = []
            if result.obb is not None:
                for obb in result.obb:
                    cls_id = int(obb.cls[0])
                    label = model.names[cls_id]
                    conf = float(obb.conf[0])
                    h_img, w_img = result.orig_shape

                    xyxyxyxy = obb.xyxyxyxy[0].tolist()
                    norm_pts = []
                    for pt in xyxyxyxy:
                        norm_pts.append((pt[0] / w_img, pt[1] / h_img))

                    polylines.append(fo.Polyline(
                        label=label,
                        points=[norm_pts],
                        closed=True,
                        filled=True,
                        confidence=conf,
                    ))
                    stats["obb"] += 1

                    xywhr = obb.xywhr[0].tolist()
                    cx, cy, bw, bh = xywhr[0], xywhr[1], xywhr[2], xywhr[3]
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    bbox = [x1 / w_img, y1 / h_img, bw / w_img, bh / h_img]
                    det = fo.Detection(
                        label=label, bounding_box=bbox, confidence=conf,
                    )
                    if len(xywhr) > 4:
                        det["rotation"] = xywhr[4]
                    detections.append(det)
                    stats["detections"] += 1

            if detections:
                sample[pred_field] = fo.Detections(detections=detections)
            if polylines:
                sample[f"{pred_field}_obb"] = fo.Polylines(polylines=polylines)

        stats["images_processed"] += 1

    logger.info("YOLO 预标注完成 (%s): %s", task, dict(stats))
    return dict(stats)
