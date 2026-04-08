#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CVAT XML 中读取托盘角点 polygon 标注（4 点），生成 CVAT skeleton pose 标注。

流程：
  1. 解析输入的 CVAT XML → 提取每张图片的 polygon 标注（4 个角点）
  2. 对每个 polygon：
     - 4 点排序为 tl, tr, br, bl
     - 检测贴近图像边界的关键点 → 标记为 occluded（不可见）
  3. 生成 CVAT skeleton XML（可在 CVAT 中直接编辑 pose 标注）

关键点可见性规则：
  如果某个关键点的坐标贴近图像边界（距离 < edge_threshold 像素），
  则标记为 occluded。转为 YOLO-Pose 后对应 visibility=1（有坐标但不可见）。

用法：
  python tools/pallet_labeling/polygon_to_pose_generator.py /path/to/root_dir

  # 自定义边界阈值（像素）
  python tools/pallet_labeling/polygon_to_pose_generator.py /path/to/root_dir \
    --edge_threshold 10

  # 自定义输入 polygon 标签名
  python tools/pallet_labeling/polygon_to_pose_generator.py /path/to/root_dir \
    --input_polygon_label pallet_corners

输出 XML 可直接导入 CVAT 进行编辑确认。
确认后用 convert_cvat_pose_to_yolo.py 转换为 YOLO-Pose 训练数据。
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEYPOINT_NAMES = ["tl", "tr", "br", "bl"]
KEYPOINT_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#ffff00"]

# SVG icon coordinates for skeleton preview (in a 100x100 space)
_SVG_NODES = [
    ("tl", "#ff0000", 25.0, 10.0),
    ("tr", "#00ff00", 75.0, 10.0),
    ("br", "#0000ff", 75.0, 90.0),
    ("bl", "#ffff00", 25.0, 90.0),
]
_SVG_EDGES = [(1, 2), (2, 3), (3, 4), (4, 1)]  # tl-tr, tr-br, br-bl, bl-tl


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


@dataclass
class KeypointAnn:
    """一个关键点：坐标 + 是否 occluded。"""
    x: float
    y: float
    occluded: bool = False


@dataclass
class SkeletonAnn:
    """一个 skeleton 实例：4 个排好序的关键点。"""
    keypoints: list[KeypointAnn]  # tl, tr, br, bl


# ---------------------------------------------------------------------------
# CVAT XML parsing
# ---------------------------------------------------------------------------

def parse_cvat_xml(
    xml_path: Path,
    polygon_label: str = "pallet",
) -> tuple[ET.Element, list[ImageAnn]]:
    """解析 CVAT XML，返回 (原始 root, 图片标注列表)。只提取指定 label 的 4 点 polygon。"""
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
            label = poly_el.get("label", "")
            if polygon_label and label != polygon_label:
                continue
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
                    label=label,
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

def _order_quad_tl_tr_br_bl(
    pts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """将 4 点排序为 [tl, tr, br, bl]。"""
    arr = np.array(pts, dtype=np.float64)
    s = arr.sum(axis=1)
    d = arr[:, 0] - arr[:, 1]
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


# ---------------------------------------------------------------------------
# Edge proximity → visibility
# ---------------------------------------------------------------------------

def compute_keypoint_occluded(
    ordered_pts: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    edge_threshold: float = 5.0,
) -> list[bool]:
    """判断每个关键点是否贴近图像边界 → occluded。

    对每个点检查到四条边界的最小距离，若 < edge_threshold 则 occluded=True。
    """
    result: list[bool] = []
    for x, y in ordered_pts:
        dist_left = x
        dist_right = img_w - x
        dist_top = y
        dist_bottom = img_h - y
        min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        result.append(min_dist < edge_threshold)
    return result


def polygon_to_skeleton(
    poly: PolygonAnn,
    img_w: int,
    img_h: int,
    edge_threshold: float = 5.0,
) -> SkeletonAnn:
    """将一个 4 点 polygon 转为带可见性信息的 skeleton。"""
    ordered = _order_quad_tl_tr_br_bl(poly.points)
    occluded_flags = compute_keypoint_occluded(ordered, img_w, img_h, edge_threshold)
    keypoints = [
        KeypointAnn(x=x, y=y, occluded=occ)
        for (x, y), occ in zip(ordered, occluded_flags)
    ]
    return SkeletonAnn(keypoints=keypoints)


# ---------------------------------------------------------------------------
# CVAT skeleton XML generation
# ---------------------------------------------------------------------------

def _build_skeleton_svg() -> ET.Element:
    """构造 skeleton 拓扑的 SVG 元素（用于 CVAT 标签定义中的可视化）。"""
    svg = ET.Element("svg")
    for node_id, (name, color, cx, cy) in enumerate(_SVG_NODES, 1):
        c = ET.SubElement(svg, "circle")
        c.set("r", "1.5")
        c.set("stroke", "black")
        c.set("fill", color)
        c.set("cx", f"{cx:.2f}")
        c.set("cy", f"{cy:.2f}")
        c.set("stroke-width", "0.1")
        c.set("data-type", "element node")
        c.set("data-element-id", str(node_id))
        c.set("data-node-id", str(node_id))
        c.set("data-label-name", name)
    node_coords = {i + 1: (cx, cy) for i, (_, _, cx, cy) in enumerate(_SVG_NODES)}
    for nfrom, nto in _SVG_EDGES:
        line = ET.SubElement(svg, "line")
        line.set("x1", f"{node_coords[nfrom][0]:.2f}")
        line.set("y1", f"{node_coords[nfrom][1]:.2f}")
        line.set("x2", f"{node_coords[nto][0]:.2f}")
        line.set("y2", f"{node_coords[nto][1]:.2f}")
        line.set("stroke", "black")
        line.set("stroke-width", "0.5")
        line.set("data-type", "edge")
        line.set("data-node-from", str(nfrom))
        line.set("data-node-to", str(nto))
    return svg


def _build_skeleton_label_meta(
    skeleton_label: str = "pallet",
    skeleton_color: str = "#2a7dd1",
) -> ET.Element:
    """构建 CVAT skeleton 标签的元数据定义。"""
    lbl = ET.Element("label")
    ET.SubElement(lbl, "name").text = skeleton_label
    ET.SubElement(lbl, "color").text = skeleton_color
    ET.SubElement(lbl, "type").text = "skeleton"

    sublabels_el = ET.SubElement(lbl, "sublabels")
    for name, color in zip(KEYPOINT_NAMES, KEYPOINT_COLORS):
        sub = ET.SubElement(sublabels_el, "sublabel")
        ET.SubElement(sub, "name").text = name
        ET.SubElement(sub, "color").text = color
        ET.SubElement(sub, "type").text = "points"
        ET.SubElement(sub, "attributes")

    lbl.append(_build_skeleton_svg())
    ET.SubElement(lbl, "attributes")
    return lbl


def generate_cvat_skeleton_xml(
    orig_root: ET.Element,
    images: list[ImageAnn],
    skeleton_data: dict[int, list[SkeletonAnn]],
    *,
    skeleton_label: str = "pallet",
) -> str:
    """生成包含 skeleton pose 标注的 CVAT XML。"""
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # --- meta ---
    meta = ET.SubElement(root, "meta")

    # 尝试复制原始 XML 的 task/job 元素结构，替换 labels 部分
    orig_meta = orig_root.find("meta")
    task_el = None
    if orig_meta is not None:
        for tag in ("task", "job"):
            src = orig_meta.find(tag)
            if src is not None:
                task_el = copy.deepcopy(src)
                old_labels = task_el.find("labels")
                if old_labels is not None:
                    task_el.remove(old_labels)
                break

    if task_el is None:
        task_el = ET.Element("task")
        size_el = ET.SubElement(task_el, "size")
        size_el.text = str(len(images))

    labels_el = ET.SubElement(task_el, "labels")
    labels_el.append(_build_skeleton_label_meta(skeleton_label))
    meta.append(task_el)

    dumped_el = ET.SubElement(meta, "dumped")
    dumped_el.text = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")

    # --- image annotations ---
    for img in images:
        img_el = ET.SubElement(root, "image")
        img_el.set("id", str(img.id))
        img_el.set("name", img.name)
        img_el.set("width", str(img.width))
        img_el.set("height", str(img.height))

        for skel in skeleton_data.get(img.id, []):
            skel_el = ET.SubElement(img_el, "skeleton")
            skel_el.set("label", skeleton_label)
            skel_el.set("source", "auto")
            skel_el.set("occluded", "0")
            skel_el.set("z_order", "0")

            for kp, name in zip(skel.keypoints, KEYPOINT_NAMES):
                pt_el = ET.SubElement(skel_el, "points")
                pt_el.set("label", name)
                pt_el.set("source", "auto")
                pt_el.set("outside", "0")
                pt_el.set("occluded", "1" if kp.occluded else "0")
                pt_el.set("points", f"{kp.x:.2f},{kp.y:.2f}")
                pt_el.set("z_order", "0")

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 CVAT polygon 角点标注生成 CVAT skeleton pose 标注"
                    "（自动检测边界关键点并标记为 occluded）",
    )
    ap.add_argument(
        "root_dir", type=str,
        help="根目录，默认在其下查找 annotations.xml 并输出到 output_annotations.xml",
    )
    ap.add_argument("--cvat_xml", type=str, default=None,
                    help="输入 CVAT XML（默认 <root_dir>/annotations.xml）")
    ap.add_argument("--output", type=str, default=None,
                    help="输出 CVAT XML（默认 <root_dir>/output_annotations.xml）")
    ap.add_argument("--input_polygon_label", type=str, default="pallet",
                    help="输入 XML 中 polygon 标注的标签名（默认 pallet）")
    ap.add_argument("--skeleton_label", type=str, default="pallet",
                    help="输出 XML 中 skeleton 标签名（默认 pallet）")
    ap.add_argument("--edge_threshold", type=float, default=5.0,
                    help="关键点距图像边界小于此像素值时标记为 occluded（默认 5.0）")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    if not root_dir.is_dir():
        print(f"[ERROR] 根目录不存在: {root_dir}")
        return 1

    cvat_xml_path = (Path(args.cvat_xml).expanduser().resolve()
                     if args.cvat_xml else root_dir / "annotations.xml")
    output_path = (Path(args.output).expanduser().resolve()
                   if args.output else root_dir / "output_annotations.xml")

    if not cvat_xml_path.exists():
        print(f"[ERROR] CVAT XML 不存在: {cvat_xml_path}")
        return 1

    # --- Parse XML ---
    print(f"解析 CVAT XML: {cvat_xml_path}")
    orig_root, images = parse_cvat_xml(cvat_xml_path, args.input_polygon_label)

    total_polygons = sum(len(img.polygons) for img in images)
    annotated_count = sum(1 for img in images if img.polygons)
    print(f"  总图片数: {len(images)}")
    print(f"  有标注的图片: {annotated_count}")
    print(f"  总 polygon 数: {total_polygons}")

    if total_polygons == 0:
        print("[WARN] 没有找到符合条件的 polygon 标注")

    # --- Convert polygons → skeletons with edge visibility ---
    print(f"\n转换 polygon → skeleton pose（edge_threshold={args.edge_threshold}px）...")
    print("-" * 60)

    skeleton_data: dict[int, list[SkeletonAnn]] = {}
    total_occluded_pts = 0
    total_visible_pts = 0

    for img in images:
        if not img.polygons:
            continue
        skeletons: list[SkeletonAnn] = []
        for poly in img.polygons:
            skel = polygon_to_skeleton(poly, img.width, img.height, args.edge_threshold)
            skeletons.append(skel)

            occ_names = [
                KEYPOINT_NAMES[i]
                for i, kp in enumerate(skel.keypoints)
                if kp.occluded
            ]
            n_occ = len(occ_names)
            total_occluded_pts += n_occ
            total_visible_pts += 4 - n_occ

            if occ_names:
                print(f"  {img.name}: {', '.join(occ_names)} → occluded")
            else:
                print(f"  {img.name}: 全部可见")

        skeleton_data[img.id] = skeletons

    print("-" * 60)
    print(f"转换完成: {total_polygons} 个 skeleton")
    print(f"  可见关键点: {total_visible_pts}")
    print(f"  occluded 关键点: {total_occluded_pts}")

    # --- Generate output XML ---
    xml_str = generate_cvat_skeleton_xml(
        orig_root,
        images,
        skeleton_data,
        skeleton_label=args.skeleton_label,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(xml_str, encoding="utf-8")

    print(f"\n输出文件: {output_path}")
    print(f"\n导入 CVAT 前，需要在任务中创建 skeleton 标签：")
    print(f"  标签名: {args.skeleton_label}  (type: skeleton)")
    print(f"  关键点: {', '.join(KEYPOINT_NAMES)}")
    print(f"  连接: tl-tr, tr-br, br-bl, bl-tl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
