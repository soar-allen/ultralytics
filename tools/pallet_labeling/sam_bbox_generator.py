#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CVAT XML 中读取托盘前表面 polygon 标注，使用 SAM 获取整个托盘的
bounding box，生成新的 CVAT XML（bbox + polygon 配对），供导入 CVAT 二次确认。

流程：
  1. 解析输入的 CVAT XML → 提取每张图片的前表面 polygon（4 点）
  2. 使用 SAM 对每个 polygon 进行 prompt → 获取整个托盘的 segmentation mask
  3. 从 mask 中提取 bounding box
  4. 生成新的 CVAT XML，包含：
     - pallet（bbox，来自 SAM）
     - pallet_front（polygon，来自原始标注）
     两者通过 group_id 关联为一组

用法：
  # 基本用法：只需指定根目录，xml/images/output 使用默认路径
  python tools/pallet_labeling/sam_bbox_generator.py /path/to/root_dir
  # 等价于：
  #   --cvat_xml  /path/to/root_dir/annotations.xml
  #   --images_dir /path/to/root_dir/images
  #   --output     /path/to/root_dir/output_annotations.xml

  # 也可以手动覆盖任意默认路径
  python tools/pallet_labeling/sam_bbox_generator.py /path/to/root_dir \
    --cvat_xml custom.xml --output result.xml

  # 使用 SAM3 文本语义 prompt（推荐：避免把货物框进来）
  python tools/pallet_labeling/sam_bbox_generator.py /path/to/root_dir \
    --prompt_mode text --text_prompt pallet

  # box prompt + smallest_covering 策略 + 高度约束 + 负向提示
  python tools/pallet_labeling/sam_bbox_generator.py /path/to/root_dir \
    --prompt_mode box --expand_ratio 0.5 \
    --mask_strategy smallest_covering \
    --max_height_ratio 2.5 --negative_above

输出 XML 可直接导入 CVAT（需先创建 pallet + pallet_front 两个标签）进行二次确认。
确认后导出 XML，用 cvat_paired_to_yolo_pose.py 转换为 YOLO-Pose 训练数据。
"""

from __future__ import annotations

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolygonAnn:
    label: str
    points: list[tuple[float, float]]
    source: str = "file"
    occluded: int = 0
    z_order: int = 0


@dataclass
class ImageAnn:
    id: int
    name: str
    width: int
    height: int
    polygons: list[PolygonAnn] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CVAT XML parsing
# ---------------------------------------------------------------------------

def parse_cvat_xml(xml_path: Path) -> tuple[ET.Element, list[ImageAnn]]:
    """解析 CVAT XML，返回 (原始 root, 图片标注列表)。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    images: list[ImageAnn] = []
    for img_el in root.findall("image"):
        ia = ImageAnn(
            id=int(img_el.get("id", "0")),
            name=img_el.get("name", ""),
            width=int(img_el.get("width", "0")),
            height=int(img_el.get("height", "0")),
        )
        for poly_el in img_el.findall("polygon"):
            pts_str = poly_el.get("points", "")
            if not pts_str:
                continue
            pts: list[tuple[float, float]] = []
            for seg in pts_str.split(";"):
                parts = seg.strip().split(",")
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
            if len(pts) == 4:
                ia.polygons.append(PolygonAnn(
                    label=poly_el.get("label", "pallet"),
                    points=pts,
                    source=poly_el.get("source", "file"),
                    occluded=int(poly_el.get("occluded", "0")),
                    z_order=int(poly_el.get("z_order", "0")),
                ))
        images.append(ia)
    return root, images


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
    )


def _bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _clamp(xtl: float, ytl: float, xbr: float, ybr: float,
           w: int, h: int) -> tuple[float, float, float, float]:
    return max(0.0, xtl), max(0.0, ytl), min(float(w), xbr), min(float(h), ybr)


def _union_bbox(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _bbox_containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    """inner bbox 被 outer bbox 包含的面积比例 (intersection / inner_area)。"""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter_area / inner_area if inner_area > 0 else 0.0


def _oriented_bbox_dims(pts: list[tuple[float, float]]) -> tuple[float, float]:
    """返回多边形最小外接矩形的 (宽, 高)，宽 >= 高。"""
    import cv2
    pts_arr = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts_arr)
    w, h = rect[1]
    return (max(w, h), min(w, h))


def _select_best_mask(
    masks: np.ndarray,
    polygon_points: list[tuple[float, float]],
    strategy: str = "smallest_covering",
) -> int | None:
    """从 SAM 返回的多个 mask 中选择最佳的一个。

    策略:
    - "largest": 面积最大的 mask（原始行为，易包含货物）
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
        import cv2
        h, w = masks.shape[1], masks.shape[2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        pts_arr = np.array(polygon_points, dtype=np.int32).reshape(-1, 1, 2)
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


# ---------------------------------------------------------------------------
# SAM inference
# ---------------------------------------------------------------------------

def sam_predict_bbox(
    model,
    image_path: str,
    polygon_points: list[tuple[float, float]],
    *,
    prompt_mode: str = "centroid",
    expand_ratio: float = 0.5,
    img_w: int = 0,
    img_h: int = 0,
    mask_strategy: str = "smallest_covering",
    max_height_ratio: float = 0.0,
    negative_above: bool = False,
) -> tuple[float, float, float, float] | None:
    """使用 SAM 获取 bounding box，返回 (xtl, ytl, xbr, ybr) 或 None。

    Args:
        mask_strategy: mask 选择策略。"smallest_covering"（推荐）选择包含
            polygon 且面积最小的 mask，避免把货物框进来；"largest" 为旧行为。
        max_height_ratio: bbox 最大高度 = polygon 高度 × 此倍数。0 表示不限制。
            设 2.0~3.0 可有效防止 bbox 向上扩展到货物区域。
        negative_above: 为 centroid/points 模式添加负向提示点（polygon 上方），
            帮助 SAM 排除货物区域。
    """
    try:
        if prompt_mode == "centroid":
            cx, cy = _centroid(polygon_points)
            if negative_above:
                min_y = min(p[1] for p in polygon_points)
                poly_h = max(p[1] for p in polygon_points) - min_y
                neg_y = max(0.0, min_y - poly_h * 0.5)
                results = model(
                    image_path,
                    points=[[[cx, cy], [cx, neg_y]]],
                    labels=[[1, 0]],
                    verbose=False,
                )
            else:
                results = model(image_path, points=[[cx, cy]], labels=[1], verbose=False)
        elif prompt_mode == "box":
            x1, y1, x2, y2 = _bbox(polygon_points)
            bw, bh = x2 - x1, y2 - y1
            ex1 = x1 - bw * expand_ratio
            ey1 = y1 - bh * expand_ratio
            ex2 = x2 + bw * expand_ratio
            ey2 = y2 + bh * expand_ratio
            if img_w > 0 and img_h > 0:
                ex1, ey1, ex2, ey2 = _clamp(ex1, ey1, ex2, ey2, img_w, img_h)
            results = model(image_path, bboxes=[[ex1, ey1, ex2, ey2]], verbose=False)
        elif prompt_mode == "points":
            pts = [list(p) for p in polygon_points]
            lbls = [1] * len(polygon_points)
            if negative_above:
                cx = sum(p[0] for p in polygon_points) / len(polygon_points)
                min_y = min(p[1] for p in polygon_points)
                poly_h = max(p[1] for p in polygon_points) - min_y
                neg_y = max(0.0, min_y - poly_h * 0.5)
                pts.append([cx, neg_y])
                lbls.append(0)
            results = model(
                image_path,
                points=[pts],
                labels=[lbls],
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown prompt_mode: {prompt_mode}")
    except Exception as e:
        print(f"    [SAM ERROR] {e}")
        return None

    if not results or results[0].masks is None:
        return None

    masks = results[0].masks.data.cpu().numpy()
    if masks.shape[0] == 0:
        return None

    best_idx = _select_best_mask(masks, polygon_points, strategy=mask_strategy)
    if best_idx is None:
        return None
    mask = masks[best_idx]

    ys, xs = np.where(mask > 0.5)
    if len(ys) == 0:
        return None

    sam_bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))

    poly_bbox = _bbox(polygon_points)
    final_bbox = _union_bbox(sam_bbox, poly_bbox)

    if max_height_ratio > 0:
        poly_h = poly_bbox[3] - poly_bbox[1]
        max_h = poly_h * max_height_ratio
        current_h = final_bbox[3] - final_bbox[1]
        if current_h > max_h:
            center_y = (poly_bbox[1] + poly_bbox[3]) / 2.0
            final_bbox = (final_bbox[0], center_y - max_h / 2, final_bbox[2], center_y + max_h / 2)
            final_bbox = _union_bbox(final_bbox, poly_bbox)

    if img_w > 0 and img_h > 0:
        final_bbox = _clamp(*final_bbox, img_w, img_h)

    return final_bbox


def sam_text_predict_bboxes(
    predictor,
    image_path: str,
    text_prompt: str = "pallet",
    *,
    conf: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """使用 SAM3 文本 prompt 检测图片中所有匹配目标，返回 bbox 列表。

    需要 SAM3SemanticPredictor 实例（而非普通 SAM 模型）。
    """
    predictor.set_image(image_path)
    results = predictor(text=[text_prompt])

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for b, c in zip(boxes_xyxy, confs)
        if c >= conf
    ]


def _match_polygon_to_bbox(
    polygon_points: list[tuple[float, float]],
    detected_bboxes: list[tuple[float, float, float, float]],
    min_containment: float = 0.5,
) -> tuple[float, float, float, float] | None:
    """将 polygon 匹配到最佳的检测 bbox（polygon 被 bbox 包含比例最高的）。"""
    if not detected_bboxes:
        return None
    poly_bb = _bbox(polygon_points)
    best_score, best_idx = 0.0, -1
    for i, bb in enumerate(detected_bboxes):
        score = _bbox_containment(poly_bb, bb)
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx >= 0 and best_score >= min_containment:
        return detected_bboxes[best_idx]
    return None


def _text_fallback_bbox(
    sam_model,
    image_path: str,
    polygon_points: list[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """text 模式回退: 高度 = polygon OBB 短边，宽度由 SAM centroid 检测。

    当 text prompt 未匹配到托盘时，用 centroid prompt 让 SAM 检测整个物体
    （含货物无妨），取其水平宽度；高度直接使用 polygon 最小外接矩形的短边
    （即前表面高度），从而避免 bbox 向上延伸到货物区域。
    """
    poly_bbox = _bbox(polygon_points)
    _, obb_h = _oriented_bbox_dims(polygon_points)

    sam_bbox = sam_predict_bbox(
        sam_model, image_path, polygon_points,
        prompt_mode="centroid",
        img_w=img_w, img_h=img_h,
        mask_strategy="largest",
    )

    if sam_bbox is not None:
        xtl, xbr = sam_bbox[0], sam_bbox[2]
    else:
        xtl, xbr = poly_bbox[0], poly_bbox[2]

    xtl = min(xtl, poly_bbox[0])
    xbr = max(xbr, poly_bbox[2])

    poly_cy = (poly_bbox[1] + poly_bbox[3]) / 2.0
    ytl = poly_cy - obb_h / 2.0
    ybr = poly_cy + obb_h / 2.0

    if img_w > 0 and img_h > 0:
        xtl, ytl, xbr, ybr = _clamp(xtl, ytl, xbr, ybr, img_w, img_h)

    return (xtl, ytl, xbr, ybr)


# ---------------------------------------------------------------------------
# Output CVAT XML generation
# ---------------------------------------------------------------------------

def generate_cvat_xml(
    orig_root: ET.Element,
    images: list[ImageAnn],
    bboxes: dict[tuple[int, int], tuple[float, float, float, float]],
    *,
    bbox_label: str = "pallet",
    polygon_label: str = "pallet_front",
    fallback_expand: float = 50.0,
) -> str:
    """生成包含 bbox + polygon 配对的 CVAT XML。"""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    orig_meta = orig_root.find("meta")
    if orig_meta is not None:
        orig_job = orig_meta.find("job")
        if orig_job is not None:
            job = copy.deepcopy(orig_job)
            old_labels = job.find("labels")
            if old_labels is not None:
                job.remove(old_labels)
            labels_el = ET.SubElement(job, "labels")
            for lname, lcolor in [(bbox_label, "#2a7dd1"), (polygon_label, "#d12a7d")]:
                lbl = ET.SubElement(labels_el, "label")
                ET.SubElement(lbl, "name").text = lname
                ET.SubElement(lbl, "color").text = lcolor
                ET.SubElement(lbl, "type").text = "any"
                ET.SubElement(lbl, "attributes")
            meta.append(job)
    dumped_el = ET.SubElement(meta, "dumped")
    dumped_el.text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    for img in images:
        img_el = ET.SubElement(root, "image")
        img_el.set("id", str(img.id))
        img_el.set("name", img.name)
        img_el.set("width", str(img.width))
        img_el.set("height", str(img.height))

        group_id = 0
        for pi, poly in enumerate(img.polygons):
            group_id += 1
            bbox = bboxes.get((img.id, pi))

            if bbox is not None:
                xtl, ytl, xbr, ybr = bbox
            else:
                x1, y1, x2, y2 = _bbox(poly.points)
                xtl = max(0.0, x1 - fallback_expand)
                ytl = max(0.0, y1 - fallback_expand)
                xbr = min(float(img.width), x2 + fallback_expand)
                ybr = min(float(img.height), y2 + fallback_expand)

            box_el = ET.SubElement(img_el, "box")
            box_el.set("label", bbox_label)
            box_el.set("source", "sam")
            box_el.set("occluded", "0")
            box_el.set("xtl", f"{xtl:.2f}")
            box_el.set("ytl", f"{ytl:.2f}")
            box_el.set("xbr", f"{xbr:.2f}")
            box_el.set("ybr", f"{ybr:.2f}")
            box_el.set("z_order", "0")
            box_el.set("group_id", str(group_id))

            poly_el = ET.SubElement(img_el, "polygon")
            poly_el.set("label", polygon_label)
            poly_el.set("source", poly.source)
            poly_el.set("occluded", str(poly.occluded))
            pts_str = ";".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly.points)
            poly_el.set("points", pts_str)
            poly_el.set("z_order", "0")
            poly_el.set("group_id", str(group_id))

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def resolve_image_path(images_dir: Path, image_name: str) -> Path | None:
    """根据 XML 中的 name 字段查找图片文件。"""
    direct = images_dir / image_name
    if direct.exists():
        return direct
    basename = Path(image_name).name
    for p in images_dir.rglob(basename):
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 CVAT polygon 标注 + SAM 生成 bbox，输出配对的 CVAT XML",
    )
    ap.add_argument("root_dir", type=str,
                    help="根目录，默认在其下查找 annotations.xml、images/ 并输出到 output_annotations.xml")
    ap.add_argument("--cvat_xml", type=str, default=None,
                    help="输入的 CVAT XML 文件（默认 <root_dir>/annotations.xml）")
    ap.add_argument("--images_dir", type=str, default=None,
                    help="图片根目录（默认 <root_dir>/images）")
    ap.add_argument("--output", type=str, default=None,
                    help="输出的 CVAT XML 文件路径（默认 <root_dir>/output_annotations.xml）")
    ap.add_argument("--sam_model", type=str, default="/home/cotek/ws_cotek/ultralytics/sam3.pt",
                    help="SAM 模型权重路径（默认 sam3.pt）")
    ap.add_argument("--prompt_mode", type=str, default="text",
                    choices=["centroid", "box", "points", "text"],
                    help="SAM prompt 模式：centroid（质心点）, box（扩展框）, "
                         "points（全部顶点）, text（文本语义，需 SAM3）")
    ap.add_argument("--expand_ratio", type=float, default=0.5,
                    help="box prompt 模式下的扩展比例（默认 0.5）")
    ap.add_argument("--bbox_label", type=str, default="pallet_body",
                    help="输出 XML 中 bounding box 的标签名（默认 pallet_body）")
    ap.add_argument("--polygon_label", type=str, default="pallet",
                    help="输出 XML 中前表面 polygon 的标签名（默认 pallet）")
    ap.add_argument("--fallback_expand", type=float, default=50.0,
                    help="SAM 失败时对 polygon bbox 四周扩展的像素值（默认 50）")
    ap.add_argument("--mask_strategy", type=str, default="smallest_covering",
                    choices=["smallest_covering", "largest"],
                    help="mask 选择策略：smallest_covering（包含 polygon 的最小 mask，"
                         "推荐）, largest（面积最大，旧行为）")
    ap.add_argument("--max_height_ratio", type=float, default=1.1,
                    help="bbox 最大高度 = polygon 高度 × 此倍数，0 不限制（默认 1.5）。"
                         "设 2.0~3.0 可防止 bbox 扩展到货物区域")
    ap.add_argument("--negative_above", action="store_true",
                    help="centroid/points 模式下在 polygon 上方添加负向提示点，"
                         "帮助 SAM 排除货物区域")
    ap.add_argument("--text_prompt", type=str, default="pallet",
                    help="text 模式下使用的文本提示词（默认 'pallet'）")
    ap.add_argument("--text_conf", type=float, default=0.1,
                    help="text 模式下的检测置信度阈值（默认 0.25）")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        print(f"[ERROR] 根目录不存在: {root_dir}")
        return 1

    cvat_xml_path = Path(args.cvat_xml).expanduser().resolve() if args.cvat_xml else root_dir / "annotations.xml"
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else root_dir / "images"
    output_path = Path(args.output).expanduser().resolve() if args.output else root_dir / "output_annotations.xml"

    if not cvat_xml_path.exists():
        print(f"[ERROR] CVAT XML 不存在: {cvat_xml_path}")
        return 1
    if not images_dir.exists():
        print(f"[ERROR] 图片目录不存在: {images_dir}")
        return 1

    # --- Parse XML ---
    print(f"解析 CVAT XML: {cvat_xml_path}")
    orig_root, images = parse_cvat_xml(cvat_xml_path)

    annotated = [img for img in images if img.polygons]
    total_polygons = sum(len(img.polygons) for img in annotated)
    print(f"  总图片数: {len(images)}")
    print(f"  有标注的图片: {len(annotated)}")
    print(f"  总 polygon 数: {total_polygons}")

    if total_polygons == 0:
        print("[WARN] 没有找到 polygon 标注")

    # --- Load SAM model ---
    use_text_mode = args.prompt_mode == "text"

    if use_text_mode:
        from ultralytics.models.sam import SAM3SemanticPredictor
        from ultralytics import SAM

        print(f"\n加载 SAM3 语义模型: {args.sam_model}")
        overrides = dict(
            conf=args.text_conf,
            task="segment",
            mode="predict",
            model=args.sam_model,
            half=True,
            save=False,
            verbose=False,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        print(f"SAM3 语义模型加载完成（text_prompt='{args.text_prompt}'）")

        print("加载 SAM 模型（用于 text fallback）...")
        fallback_model = SAM(args.sam_model)
        print("SAM fallback 模型加载完成")
    else:
        from ultralytics import SAM

        print(f"\n加载 SAM 模型: {args.sam_model}")
        model = SAM(args.sam_model)
        print("SAM 模型加载完成")

    # --- Run SAM ---
    mode_desc = (f"text='{args.text_prompt}'" if use_text_mode
                 else f"prompt_mode={args.prompt_mode}")
    extra_desc = []
    if not use_text_mode:
        extra_desc.append(f"mask_strategy={args.mask_strategy}")
        if args.negative_above:
            extra_desc.append("negative_above=True")
    if args.max_height_ratio > 0:
        extra_desc.append(f"max_height_ratio={args.max_height_ratio}")
    print(f"\n开始推理（{mode_desc}"
          + (f", {', '.join(extra_desc)}" if extra_desc else "")
          + "）...")
    print("-" * 60)

    bboxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    success_count = 0
    fallback_count = 0
    fail_count = 0

    for img in annotated:
        img_path = resolve_image_path(images_dir, img.name)
        if img_path is None:
            print(f"  [SKIP] 图片未找到: {img.name}")
            fail_count += len(img.polygons)
            continue

        if use_text_mode:
            detected = sam_text_predict_bboxes(
                predictor, str(img_path), args.text_prompt, conf=args.text_conf,
            )
            for pi, poly in enumerate(img.polygons):
                bbox = _match_polygon_to_bbox(poly.points, detected)
                is_fallback = False
                if bbox is not None:
                    poly_bbox = _bbox(poly.points)
                    bbox = _union_bbox(bbox, poly_bbox)
                    if args.max_height_ratio > 0:
                        poly_h = poly_bbox[3] - poly_bbox[1]
                        max_h = poly_h * args.max_height_ratio
                        bh = bbox[3] - bbox[1]
                        if bh > max_h:
                            cy = (poly_bbox[1] + poly_bbox[3]) / 2.0
                            bbox = (bbox[0], cy - max_h / 2, bbox[2], cy + max_h / 2)
                            bbox = _union_bbox(bbox, poly_bbox)
                    if img.width > 0 and img.height > 0:
                        bbox = _clamp(*bbox, img.width, img.height)
                else:
                    bbox = _text_fallback_bbox(
                        fallback_model, str(img_path), poly.points,
                        img.width, img.height,
                    )
                    is_fallback = True

                bboxes[(img.id, pi)] = bbox
                xtl, ytl, xbr, ybr = bbox
                bw, bh = xbr - xtl, ybr - ytl
                if is_fallback:
                    fallback_count += 1
                else:
                    success_count += 1
                done = success_count + fallback_count + fail_count
                tag = "SAM fallback" if is_fallback else "text"
                print(f"  [{done}/{total_polygons}] "
                      f"{img.name} poly#{pi} [{tag}] -> "
                      f"bbox=({xtl:.0f},{ytl:.0f},{xbr:.0f},{ybr:.0f}) "
                      f"size={bw:.0f}x{bh:.0f}")
        else:
            for pi, poly in enumerate(img.polygons):
                bbox = sam_predict_bbox(
                    model,
                    str(img_path),
                    poly.points,
                    prompt_mode=args.prompt_mode,
                    expand_ratio=args.expand_ratio,
                    img_w=img.width,
                    img_h=img.height,
                    mask_strategy=args.mask_strategy,
                    max_height_ratio=args.max_height_ratio,
                    negative_above=args.negative_above,
                )

                if bbox is not None:
                    bboxes[(img.id, pi)] = bbox
                    success_count += 1
                    xtl, ytl, xbr, ybr = bbox
                    bw, bh = xbr - xtl, ybr - ytl
                    print(f"  [{success_count + fail_count}/{total_polygons}] "
                          f"{img.name} poly#{pi} -> bbox=({xtl:.0f},{ytl:.0f},"
                          f"{xbr:.0f},{ybr:.0f}) size={bw:.0f}x{bh:.0f}")
                else:
                    fail_count += 1
                    print(f"  [{success_count + fail_count}/{total_polygons}] "
                          f"{img.name} poly#{pi} -> SAM 失败, 使用 fallback")

    print("-" * 60)
    parts = [f"{success_count} 成功"]
    if fallback_count > 0:
        parts.append(f"{fallback_count} SAM fallback")
    if fail_count > 0:
        parts.append(f"{fail_count} 失败(expand fallback)")
    print(f"SAM 推理完成: {', '.join(parts)}")

    # --- Generate output XML ---
    xml_str = generate_cvat_xml(
        orig_root,
        images,
        bboxes,
        bbox_label=args.bbox_label,
        polygon_label=args.polygon_label,
        fallback_expand=args.fallback_expand,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str, encoding="utf-8")

    print(f"\n输出文件: {output_path}")
    ann_count = sum(len(img.polygons) for img in images)
    print(f"包含 {ann_count} 组配对标注（{args.bbox_label} bbox + {args.polygon_label} polygon）")
    print(f"\n导入 CVAT 前，需要在任务中创建以下标签：")
    print(f"  1. {args.bbox_label}  (shape type: rectangle)")
    print(f"  2. {args.polygon_label}  (shape type: polygon)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
