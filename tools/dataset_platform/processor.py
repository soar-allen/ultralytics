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
                if "corrupt" not in sample.tags:
                    sample.tags.append("corrupt")
                sample.save()
                continue
            h, w = img.shape[:2]
            if w < min_size or h < min_size or w > max_size or h > max_size:
                bad_ids.append(sample.id)
                if "abnormal_size" not in sample.tags:
                    sample.tags.append("abnormal_size")
                sample.save()
        except Exception:
            bad_ids.append(sample.id)
            if "corrupt" not in sample.tags:
                sample.tags.append("corrupt")
            sample.save()

    logger.info("检测到 %d 个损坏/异常样本", len(bad_ids))
    return bad_ids


# ===================================================================
# 2b. 多边形处理（转四角 / 边界检测）
# ===================================================================

def _dedup_points(arr: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """去除距离 < tol 的重复顶点。"""
    if len(arr) == 0:
        return arr
    keep = [0]
    for i in range(1, len(arr)):
        if all(np.linalg.norm(arr[i] - arr[j]) > tol for j in keep):
            keep.append(i)
    return arr[keep]


def _order_quad_points(arr: np.ndarray) -> list[tuple[float, float]]:
    """
    将 4 个点排序为 tl→tr→br→bl（顺时针），保证不会形成交叉（倒 8 字形）。

    算法：以质心为原点，按 atan2 角度升序排列。在图像坐标系（y 向下）中，
    atan2 升序天然给出顺时针序列，再旋转使 x+y 最小的点（最靠近左上角）
    排在首位，即得 tl→tr→br→bl。
    """
    cx, cy = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
    order = np.argsort(angles)
    sorted_pts = arr[order]

    start = int(np.argmin(sorted_pts.sum(axis=1)))
    sorted_pts = np.roll(sorted_pts, -start, axis=0)

    return [(float(p[0]), float(p[1])) for p in sorted_pts]


def _best_quad_from_hull(hull_pts: np.ndarray) -> np.ndarray:
    """
    从凸包 (≥4 个有序顶点) 中选出面积最大的 4 点四边形。

    凸包顶点已按顺序排列，所以任意 4 个子集按原序构成凸四边形，
    Shoelace 面积计算正确。典型标注多边形凸包 ≤ 20 点，C(20,4)=4845 完全可行。
    """
    from itertools import combinations

    n = len(hull_pts)
    best_area = -1.0
    best_quad = hull_pts[:4]
    for idx in combinations(range(n), 4):
        quad = hull_pts[list(idx)]
        x, y = quad[:, 0], quad[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        if area > best_area:
            best_area = area
            best_quad = quad
    return best_quad


def _extract_quad(pts: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """
    从 N 点多边形中提取最优 4 点四边形 (tl→tr→br→bl)。

    算法流程：
      1. 去除重复顶点
      2. 若 < 4 个独立点 → 返回 None (无法构成四边形)
      3. 若恰好 4 个 → 直接排序
      4. 若 > 4 个 → 取凸包，从凸包中选面积最大的 4 点组合
      5. 对结果做退化检验，不合格则返回 None

    Returns:
        4 元素 list 或 None（退化时）
    """
    arr = np.array(pts, dtype=np.float64)
    arr = _dedup_points(arr)

    if len(arr) < 4:
        return None

    if len(arr) == 4:
        quad = arr
    else:
        hull = cv2.convexHull(arr.astype(np.float32).reshape(-1, 1, 2))
        hull_pts = hull.reshape(-1, 2).astype(np.float64)
        hull_pts = _dedup_points(hull_pts)
        if len(hull_pts) < 4:
            return None
        if len(hull_pts) == 4:
            quad = hull_pts
        else:
            quad = _best_quad_from_hull(hull_pts)

    ordered = _order_quad_points(quad)

    # 退化检验：任意两点距离 / 最长距离 < 2%
    oarr = np.array(ordered)
    dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            dists.append(float(np.linalg.norm(oarr[j] - oarr[i])))
    if max(dists) < 1e-9 or min(dists) / max(dists) < 0.02:
        return None

    return ordered


def convert_polylines_to_quads(
    ds: fo.Dataset | fo.DatasetView,
    label_field: str,
) -> dict:
    """
    将多边形标注转换为四角多边形。

    算法：凸包 → 最大面积 4 点子集 → 排序为 tl→tr→br→bl。
    对退化结果（三角形 / 线段）保留原始多边形并计入 degenerate。

    Returns:
        {"converted": int, "reordered": int, "skipped_few_pts": int,
         "degenerate": int, "total_polys": int}
    """
    stats = defaultdict(int)

    for sample in ds.iter_samples(progress=True, autosave=True):
        container = sample[label_field]
        if container is None:
            continue
        polylines = getattr(container, "polylines", None)
        if not polylines:
            continue

        for poly in polylines:
            new_rings = []
            for ring in poly.points:
                stats["total_polys"] += 1
                if len(ring) < 4:
                    stats["skipped_few_pts"] += 1
                    new_rings.append(ring)
                    continue

                result = _extract_quad(ring)
                if result is None:
                    stats["degenerate"] += 1
                    new_rings.append(ring)
                elif len(ring) == 4:
                    stats["reordered"] += 1
                    new_rings.append(result)
                else:
                    stats["converted"] += 1
                    new_rings.append(result)
            poly.points = new_rings

    logger.info("多边形转四角: %s (field=%s)", dict(stats), label_field)
    return dict(stats)


def _quad_quality(pts: list[tuple[float, float]]) -> dict:
    """
    评估一个四角多边形的几何质量。

    核心思路：透视变形下的正常托盘面内角可能很小（15°）或很大（165°），
    这些都是正常的。只有当两点几乎重合导致四边形退化为三角形、面积趋零、
    或形状完全塌缩时才判定为不合格。
    """
    import math

    arr = np.array(pts, dtype=np.float64)
    n = len(arr)

    # -- 所有顶点间的成对距离 (4C2 = 6 对) --
    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(float(np.linalg.norm(arr[j] - arr[i])))

    # -- 边长 (4条) --
    sides = [float(np.linalg.norm(arr[(i + 1) % n] - arr[i])) for i in range(n)]
    perimeter = sum(sides)
    min_side = min(sides)
    max_side = max(sides)
    side_ratio = min_side / max_side if max_side > 1e-9 else 0.0

    min_pair = min(pairwise)
    max_pair = max(pairwise)
    pair_ratio = min_pair / max_pair if max_pair > 1e-9 else 0.0

    # -- 内角 --
    angles = []
    for i in range(n):
        v1 = arr[(i - 1) % n] - arr[i]
        v2 = arr[(i + 1) % n] - arr[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-12:
            angles.append(0.0)
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cos_a)))

    # -- 面积 (Shoelace) --
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    # -- 紧凑度 = 面积 / (周长/4)² 完美正方形=1, 退化条状→0 --
    compactness = area / ((perimeter / 4) ** 2) if perimeter > 1e-9 else 0.0

    return {
        "min_side": min_side,
        "max_side": max_side,
        "side_ratio": side_ratio,
        "pair_ratio": pair_ratio,
        "min_angle_deg": min(angles) if angles else 0.0,
        "max_angle_deg": max(angles) if angles else 0.0,
        "area": area,
        "compactness": compactness,
        "perimeter": perimeter,
    }


def find_bad_pallet_polylines(
    ds: fo.Dataset | fo.DatasetView,
    label_field: str,
    tag: str = "bad_polygon",
    classes: list[str] | None = None,
    pair_ratio_min: float = 0.02,
    area_min: float = 5e-5,
    compactness_min: float = 0.01,
    angle_min: float = 5.0,
    angle_max: float = 178.0,
) -> dict:
    """
    检测不合格的托盘多边形并标记。

    设计理念：透视变形下的正常托盘面内角范围很宽（可达 10°~170°），
    边长差异也很大，这些都是正常的。此函数仅捕获真正退化的形状：

      1. 不是 4 个顶点
      2. 任意两点距离 / 最长距离 < pair_ratio_min → 点重合，退化三角形
      3. 面积 < area_min → 形状塌缩
      4. 紧凑度 < compactness_min → 极端细条
      5. 内角 < angle_min 或 > angle_max → 几乎完全折叠

    Args:
        classes: 仅检测指定类别，None 则检测全部
        pair_ratio_min: 最近点对距离/最远点对距离 (默认 0.02，极其宽松)
        area_min: 最小面积 (归一化坐标，默认 5e-5)
        compactness_min: 面积 / (周长/4)² 的最小值 (默认 0.01)
        angle_min / angle_max: 内角范围 (默认 5°~178°，仅捕获几乎折叠)

    Returns:
        {"bad_ids": list, "total_checked": int, "not_quad": int,
         "bad_geometry": int, "reasons": dict[str, int]}
    """
    bad_ids: list[str] = []
    reasons: dict[str, int] = {}
    total_checked = 0
    not_quad = 0
    bad_geometry = 0
    class_set = set(classes) if classes else None

    for sample in ds.iter_samples(progress=True):
        container = sample[label_field] if sample.has_field(label_field) else None
        if container is None:
            continue
        polylines = getattr(container, "polylines", None)
        if not polylines:
            continue

        sample_bad = False
        for poly in polylines:
            if class_set and poly.label not in class_set:
                continue
            total_checked += 1
            ring = poly.points[0] if poly.points else []

            if len(ring) != 4:
                not_quad += 1
                sample_bad = True
                r = f"非四角多边形 ({len(ring)}点)"
                reasons[r] = reasons.get(r, 0) + 1
                continue

            qr = _quad_quality(ring)
            reason = ""

            if qr["pair_ratio"] < pair_ratio_min:
                reason = f"两点几乎重合 (距离比 {qr['pair_ratio']:.4f})，退化三角形"
            elif qr["area"] < area_min:
                reason = f"面积过小 ({qr['area']:.6f})，形状塌缩"
            elif qr["compactness"] < compactness_min:
                reason = f"紧凑度过低 ({qr['compactness']:.4f})，极端细条"
            elif qr["min_angle_deg"] < angle_min:
                reason = f"内角近乎折叠 ({qr['min_angle_deg']:.1f}°)"
            elif qr["max_angle_deg"] > angle_max:
                reason = f"内角近乎展平 ({qr['max_angle_deg']:.1f}°)"

            if reason:
                bad_geometry += 1
                sample_bad = True
                reasons[reason] = reasons.get(reason, 0) + 1

        if sample_bad:
            if tag not in sample.tags:
                sample.tags.append(tag)
                sample.save()
            bad_ids.append(sample.id)

    logger.info(
        "托盘多边形检测: %d bad / %d checked (field=%s)",
        len(bad_ids), total_checked, label_field,
    )
    return {
        "bad_ids": bad_ids,
        "total_checked": total_checked,
        "not_quad": not_quad,
        "bad_geometry": bad_geometry,
        "reasons": reasons,
    }


def clear_tag(ds: fo.Dataset, tag: str) -> int:
    """移除数据集中所有样本的指定标签，返回受影响的样本数。"""
    count = 0
    for sample in ds.match_tags([tag]).iter_samples(autosave=True):
        if tag in sample.tags:
            sample.tags.remove(tag)
            count += 1
    return count


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


# ===================================================================
# Polyline NMS 去重工具
# ===================================================================

def _polyline_pts_to_contour(pts: list[tuple[float, float]], scale: int = 1000) -> np.ndarray:
    """将归一化 polyline 点列表转为 cv2 contour (整数像素坐标)。"""
    return np.array(
        [[int(x * scale), int(y * scale)] for x, y in pts], dtype=np.int32,
    )


def _polygon_iou(pts_a: list[tuple[float, float]], pts_b: list[tuple[float, float]],
                 scale: int = 1000) -> float:
    """计算两个多边形的 IoU（基于栅格化）。"""
    ca = _polyline_pts_to_contour(pts_a, scale)
    cb = _polyline_pts_to_contour(pts_b, scale)

    canvas_a = np.zeros((scale, scale), dtype=np.uint8)
    canvas_b = np.zeros((scale, scale), dtype=np.uint8)
    cv2.fillPoly(canvas_a, [ca], 1)
    cv2.fillPoly(canvas_b, [cb], 1)

    inter = int(np.sum(canvas_a & canvas_b))
    union = int(np.sum(canvas_a | canvas_b))
    return inter / union if union > 0 else 0.0


def _merge_polylines_nms(
    existing_polylines: list,
    new_polylines: list,
    iou_threshold: float = 0.5,
) -> tuple[list, int]:
    """
    将新预测的 polyline 合并到已有列表中，跳过与已有同类别标注 IoU 超过阈值的重复项。

    Returns:
        (合并后的列表, 被抑制的数量)
    """
    merged = list(existing_polylines)
    suppressed = 0

    for new_pl in new_polylines:
        new_pts = new_pl.points[0] if new_pl.points else []
        if len(new_pts) < 3:
            merged.append(new_pl)
            continue

        is_dup = False
        for old_pl in merged:
            if old_pl.label != new_pl.label:
                continue
            old_pts = old_pl.points[0] if old_pl.points else []
            if len(old_pts) < 3:
                continue
            iou = _polygon_iou(new_pts, old_pts)
            if iou >= iou_threshold:
                is_dup = True
                break

        if is_dup:
            suppressed += 1
        else:
            merged.append(new_pl)

    return merged, suppressed


# ===================================================================
# YOLO Pose → 四角多边形 (Polyline) 预标注
# ===================================================================

def auto_predict_pose_to_polyline(
    ds: fo.Dataset,
    model_path: str,
    pred_field: str = "predictions",
    conf_threshold: float = 0.25,
    filter_classes: Optional[list[str]] = None,
    skip_labeled: bool = False,
    nms_iou: float = 0.5,
    view: Optional[fo.DatasetView] = None,
) -> dict:
    """
    使用 YOLO Pose 模型检测四个关键点，将其连接成闭合四边形 Polyline。

    关键点顺序应为 tl → tr → br → bl。
    自动使用模型输出的类别名称（支持多类别）。

    Args:
        filter_classes: 仅保留这些类别的检测结果（None = 全部保留）
        skip_labeled: 完全跳过已有标注的样本
        nms_iou: 标注级 NMS 的 IoU 阈值
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    target = view if view is not None else ds
    stats = defaultdict(int)

    for sample in target.iter_samples(progress=True, autosave=True):
        existing_polylines = []
        if sample.has_field(pred_field):
            existing_field = sample[pred_field]
            if existing_field is not None and hasattr(existing_field, "polylines"):
                existing_polylines = list(existing_field.polylines or [])

        if skip_labeled and len(existing_polylines) > 0:
            stats["skipped_labeled"] += 1
            continue

        results = model(sample.filepath, conf=conf_threshold, verbose=False)
        if not results:
            stats["images_processed"] += 1
            continue

        result = results[0]
        h_img, w_img = result.orig_shape
        new_polylines = []

        if result.keypoints is not None and result.boxes is not None:
            for i, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                det_label = model.names.get(cls_id, f"class_{cls_id}")

                if filter_classes and det_label not in filter_classes:
                    stats["filtered_class"] += 1
                    continue

                if i >= len(result.keypoints):
                    break
                kp_data = result.keypoints[i]
                xy = kp_data.xy[0] if hasattr(kp_data, "xy") else kp_data.data[0, :, :2]

                if xy.shape[0] < 4:
                    stats["skipped_few_kp"] += 1
                    continue

                pts = []
                for ki in range(4):
                    px = float(xy[ki, 0]) / w_img
                    py = float(xy[ki, 1]) / h_img
                    px = max(0.0, min(1.0, px))
                    py = max(0.0, min(1.0, py))
                    pts.append((px, py))

                new_polylines.append(fo.Polyline(
                    label=det_label,
                    points=[pts],
                    closed=True,
                    filled=True,
                    confidence=conf,
                ))
                stats["detected"] += 1

        if new_polylines:
            merged, suppressed = _merge_polylines_nms(
                existing_polylines, new_polylines, iou_threshold=nms_iou,
            )
            stats["suppressed_nms"] += suppressed
            stats["added"] += len(new_polylines) - suppressed
            sample[pred_field] = fo.Polylines(polylines=merged)

        stats["images_processed"] += 1

    logger.info("Pose→Polyline 预标注完成: %s", dict(stats))
    return dict(stats)


# ===================================================================
# SAM3 Text Prompt → 四角多边形 (Polyline) 预标注
# ===================================================================

def _mask_to_quadrilateral(mask_np: np.ndarray) -> list[tuple[float, float]] | None:
    """
    从二值 mask 提取最小面积旋转矩形的四个顶点，按 tl→tr→br→bl 排序。

    Returns:
        归一化坐标 [(x,y), ...] 共 4 个点，或 None（mask 面积过小）。
    """
    contours, _ = cv2.findContours(
        mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)

    h, w = mask_np.shape[:2]
    pts = [(float(box[i, 0]) / w, float(box[i, 1]) / h) for i in range(4)]

    # 排序为 tl, tr, br, bl（先按 y 分上下，再按 x 分左右）
    pts.sort(key=lambda p: p[1])
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    ordered = [top[0], top[1], bottom[1], bottom[0]]

    return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in ordered]


def _mask_to_bbox(mask_np: np.ndarray) -> list[float] | None:
    """从二值 mask 提取轴对齐 bounding box，返回 [x, y, w, h] 归一化坐标。"""
    contours, _ = cv2.findContours(
        mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10:
        return None
    x, y, w, h_box = cv2.boundingRect(largest)
    h_img, w_img = mask_np.shape[:2]
    return [x / w_img, y / h_img, w / w_img, h_box / h_img]


def _polyline_to_xyxy(polyline, img_w: int, img_h: int) -> list[float]:
    """从 Polyline 的归一化点计算像素级 xyxy bbox。"""
    pts = polyline.points[0] if polyline.points else []
    if not pts:
        return [0, 0, 1, 1]
    xs = [p[0] * img_w for p in pts]
    ys = [p[1] * img_h for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox_iou_xyxy(a: list[float], b: list[float]) -> float:
    """计算两个 xyxy bbox 的 IoU。"""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_detections_nms(
    existing: list,
    new_dets: list,
    iou_threshold: float = 0.5,
) -> tuple[list, int]:
    """对 fo.Detection 列表进行同类别 NMS 合并。"""
    merged = list(existing)
    suppressed = 0
    for det in new_dets:
        bb_new = det.bounding_box
        if not bb_new:
            merged.append(det)
            continue
        new_xyxy = [bb_new[0], bb_new[1], bb_new[0] + bb_new[2], bb_new[1] + bb_new[3]]
        is_dup = False
        for old in merged:
            if old.label != det.label:
                continue
            bb_old = old.bounding_box
            if not bb_old:
                continue
            old_xyxy = [bb_old[0], bb_old[1], bb_old[0] + bb_old[2], bb_old[1] + bb_old[3]]
            if _bbox_iou_xyxy(new_xyxy, old_xyxy) >= iou_threshold:
                is_dup = True
                break
        if is_dup:
            suppressed += 1
        else:
            merged.append(det)
    return merged, suppressed


def _select_best_mask(
    masks: np.ndarray,
    polygon_points_px: list[tuple[float, float]],
    strategy: str = "smallest_covering",
) -> int | None:
    """从 SAM 返回的多个 mask 中选择最佳的一个。

    策略:
    - "largest": 面积最大的 mask
    - "smallest_covering": 包含 polygon 区域且面积最小的 mask（推荐）
    """
    n = masks.shape[0]
    if n == 0:
        return None
    if n == 1:
        return 0

    areas = masks.reshape(n, -1).sum(axis=1)

    if strategy == "largest":
        return int(np.argmax(areas))

    if strategy == "smallest_covering":
        h, w = masks.shape[1], masks.shape[2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        pts_arr = np.array(polygon_points_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(poly_mask, [pts_arr], 1)
        poly_area = float(poly_mask.sum())

        if poly_area == 0:
            return int(np.argmax(areas))

        best_idx, best_area = -1, float("inf")
        for i in range(n):
            m = (masks[i] > 0.5).astype(np.uint8)
            coverage = float((m & poly_mask).sum()) / poly_area
            if coverage < 0.3:
                continue
            a = float(m.sum())
            if a < best_area:
                best_area = a
                best_idx = i

        if best_idx >= 0:
            return best_idx
        coverages = [
            float(((masks[i] > 0.5).astype(np.uint8) & poly_mask).sum()) / poly_area
            for i in range(n)
        ]
        return int(np.argmax(coverages))

    return int(np.argmax(areas))


def _mask_to_output_item(
    mask_np: np.ndarray, label: str, conf: float, output_mode: str,
) -> fo.Polyline | fo.Detection | None:
    """将 mask 转为 Polyline 或 Detection，统一输出逻辑。"""
    if output_mode == "polyline":
        quad = _mask_to_quadrilateral(mask_np)
        if quad is None:
            return None
        return fo.Polyline(
            label=label, points=[quad], closed=True, filled=True, confidence=conf,
        )
    else:
        bbox = _mask_to_bbox(mask_np)
        if bbox is None:
            return None
        return fo.Detection(label=label, bounding_box=bbox, confidence=conf)


def _save_items_with_nms(
    sample, pred_field: str, existing_items: list, new_items: list,
    output_mode: str, nms_iou: float, stats: dict,
) -> bool:
    """合并 NMS 并保存到样本，返回是否有输出。"""
    if not new_items:
        return False
    if output_mode == "polyline":
        merged, suppressed = _merge_polylines_nms(
            existing_items, new_items, iou_threshold=nms_iou,
        )
        sample[pred_field] = fo.Polylines(polylines=merged)
    else:
        merged, suppressed = _merge_detections_nms(
            existing_items, new_items, iou_threshold=nms_iou,
        )
        sample[pred_field] = fo.Detections(detections=merged)
    stats["suppressed_nms"] += suppressed
    stats["added"] += len(new_items) - suppressed
    return True


def _read_existing_items(sample, pred_field: str, output_mode: str) -> list:
    """安全读取样本已有标注列表。"""
    if not sample.has_field(pred_field):
        return []
    existing_field = sample[pred_field]
    if existing_field is None:
        return []
    if output_mode == "polyline" and hasattr(existing_field, "polylines"):
        return list(existing_field.polylines or [])
    if output_mode != "polyline" and hasattr(existing_field, "detections"):
        return list(existing_field.detections or [])
    return []


def _tag_fail(sample, fail_tag: str, stats: dict):
    """给样本打失败标签。"""
    if fail_tag and fail_tag not in sample.tags:
        sample.tags.append(fail_tag)
        stats["fail_tagged"] += 1


def auto_predict_sam3(
    ds: fo.Dataset,
    model_path: str,
    pred_field: str = "predictions",
    conf_threshold: float = 0.25,
    output_mode: str = "polyline",
    prompt_mode: str = "text",
    text_prompts: Optional[list[str]] = None,
    box_source_field: Optional[str] = None,
    box_source_labels: Optional[list[str]] = None,
    box_source_tags: Optional[list[str]] = None,
    box_expand_ratio: float = 0.5,
    box_mask_strategy: str = "smallest_covering",
    label_name: str = "object",
    skip_labeled: bool = False,
    nms_iou: float = 0.5,
    fail_tag: str = "",
    view: Optional[fo.DatasetView] = None,
    half: bool = True,
) -> dict:
    """
    SAM3 辅助标注，四种提示策略：

    - text:        SAM3SemanticPredictor + text prompt → 概念分割
    - box_example: SAM3SemanticPredictor + bboxes (图像范例) → 概念匹配
    - combined:    SAM3SemanticPredictor + text + bboxes 联合提示 → 最强匹配
    - box_visual:  ultralytics.SAM + 扩展框 → SAM2 兼容视觉分割 (逐多边形)

    box_source_tags: 仅从带有这些标签的样本中提取来源 bbox（空列表=不过滤）。
    """
    target = view if view is not None else ds
    stats = defaultdict(int)

    # 将 box_source_tags 转为集合，空则 None 表示不过滤
    _src_tag_set = set(box_source_tags) if box_source_tags else None

    if prompt_mode == "text":
        _sam3_text_predict(
            target, model_path, pred_field, conf_threshold, output_mode,
            text_prompts, label_name, skip_labeled, nms_iou, fail_tag, half, stats,
        )
    elif prompt_mode == "box_example":
        _sam3_box_example_predict(
            target, model_path, pred_field, conf_threshold, output_mode,
            box_source_field, box_source_labels, box_expand_ratio,
            label_name, skip_labeled, nms_iou, fail_tag, half, stats,
            src_tag_set=_src_tag_set,
        )
    elif prompt_mode == "combined":
        _sam3_combined_predict(
            target, model_path, pred_field, conf_threshold, output_mode,
            text_prompts, box_source_field, box_source_labels, box_expand_ratio,
            label_name, skip_labeled, nms_iou, fail_tag, half, stats,
            src_tag_set=_src_tag_set,
        )
    elif prompt_mode == "box_visual":
        _sam_box_visual_predict(
            target, model_path, pred_field, conf_threshold, output_mode,
            box_source_field, box_source_labels, box_expand_ratio, box_mask_strategy,
            label_name, skip_labeled, nms_iou, fail_tag, stats,
            src_tag_set=_src_tag_set,
        )
    else:
        raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

    logger.info("SAM 预标注完成: %s", dict(stats))
    return dict(stats)


# ── Text prompt ──────────────────────────────────────────────

def _sam3_text_predict(
    target, model_path, pred_field, conf_threshold, output_mode,
    text_prompts, label_name, skip_labeled, nms_iou, fail_tag, half, stats,
):
    """Text prompt：SAM3SemanticPredictor 概念分割。"""
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf_threshold, task="segment", mode="predict",
        model=model_path, half=half, save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    for sample in target.iter_samples(progress=True, autosave=True):
        existing_items = _read_existing_items(sample, pred_field, output_mode)
        if skip_labeled and existing_items:
            stats["skipped_labeled"] += 1
            continue

        try:
            predictor.set_image(sample.filepath)
            if not text_prompts:
                stats["errors"] += 1
                continue
            results = predictor(text=text_prompts)
        except Exception as e:
            logger.warning("SAM3 text 预测失败 (%s): %s", sample.filepath, e)
            stats["errors"] += 1
            _tag_fail(sample, fail_tag, stats)
            continue

        has_output = _process_sam_results(
            results, sample, pred_field, existing_items, output_mode,
            label_name, nms_iou, stats,
        )
        if not has_output:
            _tag_fail(sample, fail_tag, stats)
        stats["images_processed"] += 1

    predictor.reset_prompts()


# ── Box example prompt (概念匹配，推荐) ──────────────────────

def _extract_source_bboxes_px(
    sample, box_source_field: str, box_source_labels: Optional[list[str]],
    w_img: int, h_img: int,
) -> list[list[float]]:
    """从样本的来源字段提取像素级 xyxy bboxes，支持多类别过滤。

    Args:
        box_source_labels: 要匹配的类别列表；None 或空列表表示不过滤（全部取）。
    """
    if not sample.has_field(box_source_field):
        return []
    source = sample[box_source_field]
    if source is None:
        return []
    items = getattr(source, "polylines", None) or getattr(source, "detections", None) or []
    if box_source_labels:
        label_set = set(box_source_labels)
        items = [it for it in items if getattr(it, "label", None) in label_set]

    bboxes = []
    for it in items:
        if hasattr(it, "points") and it.points:
            pts = it.points[0]
            xs = [p[0] * w_img for p in pts]
            ys = [p[1] * h_img for p in pts]
            bboxes.append([min(xs), min(ys), max(xs), max(ys)])
        elif hasattr(it, "bounding_box") and it.bounding_box:
            bb = it.bounding_box
            bboxes.append([bb[0] * w_img, bb[1] * h_img,
                           (bb[0] + bb[2]) * w_img, (bb[1] + bb[3]) * h_img])
    return bboxes


def _expand_bbox(bbox: list[float], ratio: float,
                 w_img: int, h_img: int) -> list[float]:
    """对 xyxy bbox 按比例四周扩展并 clamp 到图像边界。"""
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return [
        max(0.0, x1 - bw * ratio),
        max(0.0, y1 - bh * ratio),
        min(float(w_img), x2 + bw * ratio),
        min(float(h_img), y2 + bh * ratio),
    ]


def _sam3_box_example_predict(
    target, model_path, pred_field, conf_threshold, output_mode,
    box_source_field, box_source_labels, expand_ratio,
    label_name, skip_labeled, nms_iou, fail_tag, half, stats,
    *, src_tag_set: set | None = None,
):
    """Box example prompt：SAM3SemanticPredictor + bboxes 图像范例。

    SAM3 会把 bbox 内的内容当作"概念范例"，找出图像中所有相似实例。
    bbox 会按 expand_ratio 扩展，确保 SAM3 能看到超出来源多边形的完整目标。
    支持多类别：box_source_labels 为 None/[] 时取字段内全部标注。
    src_tag_set: 仅从带有这些标签的样本中提取来源 bbox。
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf_threshold, task="segment", mode="predict",
        model=model_path, half=half, save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    for sample in target.iter_samples(progress=True, autosave=True):
        existing_items = _read_existing_items(sample, pred_field, output_mode)
        if skip_labeled and existing_items:
            stats["skipped_labeled"] += 1
            continue

        img = cv2.imread(sample.filepath)
        if img is None:
            stats["errors"] += 1
            continue
        h_img, w_img = img.shape[:2]

        # 如果设置了来源标签，跳过不匹配的样本
        if src_tag_set and not (set(sample.tags or []) & src_tag_set):
            raw_bboxes = []
        else:
            raw_bboxes = _extract_source_bboxes_px(
                sample, box_source_field, box_source_labels, w_img, h_img,
            )
        if not raw_bboxes:
            stats["no_source_items"] += 1
            continue

        expanded = [_expand_bbox(bb, expand_ratio, w_img, h_img) for bb in raw_bboxes]

        try:
            predictor.set_image(sample.filepath)
            results = predictor(bboxes=expanded)
        except Exception as e:
            logger.warning("SAM3 box example 预测失败 (%s): %s", sample.filepath, e)
            stats["errors"] += 1
            _tag_fail(sample, fail_tag, stats)
            continue

        has_output = _process_sam_results(
            results, sample, pred_field, existing_items, output_mode,
            label_name, nms_iou, stats,
        )
        if not has_output:
            _tag_fail(sample, fail_tag, stats)
        stats["images_processed"] += 1

    predictor.reset_prompts()


# ── Combined prompt (text + box example) ─────────────────────

def _sam3_combined_predict(
    target, model_path, pred_field, conf_threshold, output_mode,
    text_prompts, box_source_field, box_source_labels, expand_ratio,
    label_name, skip_labeled, nms_iou, fail_tag, half, stats,
    *, src_tag_set: set | None = None,
):
    """联合提示：SAM3SemanticPredictor 同时接收 text 和 bboxes。

    SAM3 会将文字概念与图像范例结合理解目标，准确率高于单独使用任一方式。
    - text_prompts: 文字描述列表（如 ['pallet', 'wooden pallet']）
    - box_source_labels: 来源字段中作为视觉范例的类别列表（空=全部）
    - expand_ratio: 范例 bbox 扩展比例
    - src_tag_set: 仅从带有这些标签的样本中提取来源 bbox
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    overrides = dict(
        conf=conf_threshold, task="segment", mode="predict",
        model=model_path, half=half, save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    for sample in target.iter_samples(progress=True, autosave=True):
        existing_items = _read_existing_items(sample, pred_field, output_mode)
        if skip_labeled and existing_items:
            stats["skipped_labeled"] += 1
            continue

        img = cv2.imread(sample.filepath)
        if img is None:
            stats["errors"] += 1
            continue
        h_img, w_img = img.shape[:2]

        # 提取范例 bboxes（来源标签不匹配时跳过，可为空则退化为纯文本提示）
        if src_tag_set and not (set(sample.tags or []) & src_tag_set):
            raw_bboxes = []
        elif box_source_field:
            raw_bboxes = _extract_source_bboxes_px(
                sample, box_source_field, box_source_labels, w_img, h_img,
            )
        else:
            raw_bboxes = []
        expanded = [_expand_bbox(bb, expand_ratio, w_img, h_img) for bb in raw_bboxes]

        if not text_prompts and not expanded:
            stats["no_source_items"] += 1
            continue

        try:
            predictor.set_image(sample.filepath)
            kwargs: dict = {}
            if text_prompts:
                kwargs["text"] = text_prompts
            if expanded:
                kwargs["bboxes"] = expanded
            results = predictor(**kwargs)
        except Exception as e:
            logger.warning("SAM3 combined 预测失败 (%s): %s", sample.filepath, e)
            stats["errors"] += 1
            _tag_fail(sample, fail_tag, stats)
            continue

        has_output = _process_sam_results(
            results, sample, pred_field, existing_items, output_mode,
            label_name, nms_iou, stats,
        )
        if not has_output:
            _tag_fail(sample, fail_tag, stats)
        stats["images_processed"] += 1

    predictor.reset_prompts()


# ── Box visual prompt (SAM2 兼容，逐多边形) ───────────────────

def _sam_box_visual_predict(
    target, model_path, pred_field, conf_threshold, output_mode,
    box_source_field, box_source_labels, expand_ratio, mask_strategy,
    label_name, skip_labeled, nms_iou, fail_tag, stats,
    *, src_tag_set: set | None = None,
):
    """Box visual prompt：ultralytics.SAM (SAM2 兼容) 逐多边形视觉分割。

    对每个来源多边形单独调用 SAM，分割扩展框内的特定目标。
    支持多类别：box_source_labels 为 None/[] 时取字段内全部标注。
    src_tag_set: 仅从带有这些标签的样本中提取来源标注。
    """
    from ultralytics import SAM

    model = SAM(model_path)

    for sample in target.iter_samples(progress=True, autosave=True):
        existing_items = _read_existing_items(sample, pred_field, output_mode)
        if skip_labeled and existing_items:
            stats["skipped_labeled"] += 1
            continue

        if src_tag_set and not (set(sample.tags or []) & src_tag_set):
            stats["no_source_items"] += 1
            continue

        if not sample.has_field(box_source_field):
            stats["no_source_field"] += 1
            continue
        source = sample[box_source_field]
        if source is None:
            stats["no_source_field"] += 1
            continue

        items = getattr(source, "polylines", None) or getattr(source, "detections", None) or []
        if box_source_labels:
            label_set = set(box_source_labels)
            items = [it for it in items if getattr(it, "label", None) in label_set]
        if not items:
            stats["no_source_items"] += 1
            continue

        img = cv2.imread(sample.filepath)
        if img is None:
            stats["errors"] += 1
            continue
        h_img, w_img = img.shape[:2]

        new_items = []
        for it in items:
            if hasattr(it, "points") and it.points:
                pts_px = [(p[0] * w_img, p[1] * h_img) for p in it.points[0]]
            elif hasattr(it, "bounding_box") and it.bounding_box:
                bb = it.bounding_box
                x1, y1 = bb[0] * w_img, bb[1] * h_img
                x2, y2 = (bb[0] + bb[2]) * w_img, (bb[1] + bb[3]) * h_img
                pts_px = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                continue

            xs = [p[0] for p in pts_px]
            ys = [p[1] for p in pts_px]
            raw_bb = [min(xs), min(ys), max(xs), max(ys)]
            ex1, ey1, ex2, ey2 = _expand_bbox(raw_bb, expand_ratio, w_img, h_img)

            try:
                results = model(
                    sample.filepath,
                    bboxes=[[ex1, ey1, ex2, ey2]],
                    verbose=False,
                )
            except Exception as e:
                logger.warning("SAM box visual 预测失败 (%s): %s", sample.filepath, e)
                stats["errors"] += 1
                continue

            if not results or results[0].masks is None:
                stats["errors"] += 1
                continue

            masks = results[0].masks.data.cpu().numpy()
            if masks.shape[0] == 0:
                stats["errors"] += 1
                continue

            best_idx = _select_best_mask(masks, pts_px, strategy=mask_strategy)
            if best_idx is None:
                stats["errors"] += 1
                continue

            mask_np = (masks[best_idx] > 0.5).astype(np.uint8)
            conf_val = 1.0
            if results[0].boxes is not None and best_idx < len(results[0].boxes):
                box_r = results[0].boxes[best_idx]
                if hasattr(box_r, "conf") and box_r.conf is not None:
                    conf_val = float(box_r.conf[0])

            item = _mask_to_output_item(mask_np, label_name, conf_val, output_mode)
            if item is None:
                stats["skipped_small_mask"] += 1
                continue
            new_items.append(item)
            stats["detected"] += 1

        has_output = _save_items_with_nms(
            sample, pred_field, existing_items, new_items,
            output_mode, nms_iou, stats,
        )
        if not has_output:
            _tag_fail(sample, fail_tag, stats)
        stats["images_processed"] += 1


# ── 共用结果处理 ──────────────────────────────────────────────

def _process_sam_results(
    results, sample, pred_field, existing_items, output_mode,
    label_name, nms_iou, stats,
) -> bool:
    """处理 SAM3 predictor 返回的 results，统一用于 text 和 box_example 两种模式。"""
    if not results:
        return False

    result = results[0]
    if result.masks is None or result.masks.data is None:
        return False

    masks_tensor = result.masks.data
    boxes_result = result.boxes

    new_items = []
    for mi in range(masks_tensor.shape[0]):
        mask_np = masks_tensor[mi].cpu().numpy().astype(np.uint8)
        conf_val = 1.0
        if boxes_result is not None and mi < len(boxes_result):
            box = boxes_result[mi]
            if hasattr(box, "conf") and box.conf is not None:
                conf_val = float(box.conf[0])

        item = _mask_to_output_item(mask_np, label_name, conf_val, output_mode)
        if item is None:
            stats["skipped_small_mask"] += 1
            continue
        new_items.append(item)
        stats["detected"] += 1

    return _save_items_with_nms(
        sample, pred_field, existing_items, new_items,
        output_mode, nms_iou, stats,
    )


# ===================================================================
# SAM3 标注标签（仅识别并打标签，不生成标注）
# ===================================================================

def auto_tag_sam3(
    ds: fo.Dataset,
    model_path: str,
    text_prompts: Optional[list[str]] = None,
    box_source_field: Optional[str] = None,
    box_source_labels: Optional[list[str]] = None,
    box_source_tags: Optional[list[str]] = None,
    box_expand_ratio: float = 0.5,
    conf_threshold: float = 0.25,
    found_tag: str = "sam3_found",
    not_found_tag: str = "",
    fail_tag: str = "sam3_failed",
    view: Optional[fo.DatasetView] = None,
    half: bool = True,
) -> dict:
    """使用 SAM3 识别图像中是否存在目标，仅打标签不生成标注。

    对每张图片用 SAM3SemanticPredictor 检测，若输出任何 mask
    则为样本打上 found_tag；否则打上 not_found_tag（可选）。

    box_source_tags: 仅从带有这些标签的样本中提取来源 bbox（空=不过滤）。

    Returns:
        {"images_processed", "found", "not_found", "errors", "fail_tagged"}
    """
    from ultralytics.models.sam import SAM3SemanticPredictor

    target = view if view is not None else ds
    stats = defaultdict(int)
    _src_tag_set = set(box_source_tags) if box_source_tags else None

    overrides = dict(
        conf=conf_threshold, task="segment", mode="predict",
        model=model_path, half=half, save=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)

    for sample in target.iter_samples(progress=True, autosave=True):
        img = cv2.imread(sample.filepath)
        if img is None:
            stats["errors"] += 1
            _tag_fail(sample, fail_tag, stats)
            continue
        h_img, w_img = img.shape[:2]

        # 构建 SAM3 提示参数（来源标签不匹配时不提取 bbox）
        if _src_tag_set and not (set(sample.tags or []) & _src_tag_set):
            raw_bboxes = []
        elif box_source_field:
            raw_bboxes = _extract_source_bboxes_px(
                sample, box_source_field, box_source_labels, w_img, h_img,
            )
        else:
            raw_bboxes = []
        expanded = [_expand_bbox(bb, box_expand_ratio, w_img, h_img) for bb in raw_bboxes]

        if not text_prompts and not expanded:
            stats["no_prompt"] += 1
            continue

        try:
            predictor.set_image(sample.filepath)
            kwargs: dict = {}
            if text_prompts:
                kwargs["text"] = text_prompts
            if expanded:
                kwargs["bboxes"] = expanded
            results = predictor(**kwargs)
        except Exception as e:
            logger.warning("SAM3 tag 预测失败 (%s): %s", sample.filepath, e)
            stats["errors"] += 1
            _tag_fail(sample, fail_tag, stats)
            continue

        has_mask = (
            results
            and results[0].masks is not None
            and results[0].masks.data is not None
            and results[0].masks.data.shape[0] > 0
        )

        if has_mask:
            if found_tag and found_tag not in sample.tags:
                sample.tags.append(found_tag)
            stats["found"] += 1
        else:
            if not_found_tag and not_found_tag not in sample.tags:
                sample.tags.append(not_found_tag)
            stats["not_found"] += 1

        stats["images_processed"] += 1

    predictor.reset_prompts()
    logger.info("SAM3 标签识别完成: %s", dict(stats))
    return dict(stats)


def clear_tag_from_dataset(
    ds: fo.Dataset,
    tag: str,
    view: Optional[fo.DatasetView] = None,
) -> int:
    """从数据集（或视图）中清除指定标签，返回受影响样本数。"""
    target = view if view is not None else ds
    tagged = target.match_tags(tag)
    count = len(tagged)
    if count > 0:
        for sample in tagged.iter_samples(progress=True, autosave=True):
            if tag in sample.tags:
                sample.tags.remove(tag)
    logger.info("已从 %d 个样本中清除标签 '%s'", count, tag)
    return count
