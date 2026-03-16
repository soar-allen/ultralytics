#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过滤 CVAT XML 中的托盘前表面 polygon 标注。

基于多边形的宽高比过滤非前表面标注（如侧面）。
正常的托盘前表面呈扁平矩形（约 5:1），侧面标注则接近正方形（约 1:1）。

过滤策略：
  1. 宽高比过滤（主策略）：计算 polygon 对边均值的长短比，保留在目标范围内的标注
  2. 最小面积过滤：过滤面积过小的标注（噪声/远距离标注）
  3. 最小宽度过滤：过滤宽度过窄的标注

用法：
  # 使用默认参数（target_ratio=5, tolerance=0.5 → 比例范围 [2.5, 7.5]）
  python tools/pallet_labeling/filter_polygon.py \
    --cvat_xml annotations.xml \
    --output filtered.xml

  # 放宽容忍度（适合近距离拍摄）
  python tools/pallet_labeling/filter_polygon.py \
    --cvat_xml annotations.xml \
    --output filtered.xml \
    --target_ratio 5.0 --tolerance 0.7

  # 直接指定比例范围
  python tools/pallet_labeling/filter_polygon.py \
    --cvat_xml annotations.xml \
    --output filtered.xml \
    --min_ratio 2.0 --max_ratio 10.0

  # 仅报告不输出（用于调参）
  python tools/pallet_labeling/filter_polygon.py \
    --cvat_xml annotations.xml \
    --action report

  # 标记模式：不删除，改为重命名被拒绝的标签（在 CVAT 中可视化审查）
  python tools/pallet_labeling/filter_polygon.py \
    --cvat_xml annotations.xml \
    --output marked.xml \
    --action mark
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PolygonInfo:
    elem: ET.Element
    label: str
    points: list[tuple[float, float]]
    # computed metrics
    aspect_ratio: float = 0.0
    area: float = 0.0
    width: float = 0.0
    height: float = 0.0
    passed: bool = True
    reject_reason: str = ""


@dataclass
class ImageInfo:
    id: int
    name: str
    width: int
    height: int
    elem: ET.Element
    polygons: list[PolygonInfo] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _shoelace_area(pts: list[tuple[float, float]]) -> float:
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


def compute_quad_metrics(pts: list[tuple[float, float]]) -> dict[str, float]:
    """
    计算 4 点多边形的几何度量。

    点序约定：p0=top-left, p1=top-right, p2=bottom-right, p3=bottom-left。
    width 由 top/bottom 边（e1=p0→p1, e3=p2→p3）的均值决定，
    height 由 left/right 边（e2=p1→p2, e4=p3→p0）的均值决定。
    不使用 max/min，以确保旋转 90° 的标注（width < height）不会被误保留。
    """
    e1 = _dist(pts[0], pts[1])
    e2 = _dist(pts[1], pts[2])
    e3 = _dist(pts[2], pts[3])
    e4 = _dist(pts[3], pts[0])

    width = (e1 + e3) / 2.0   # top + bottom edges
    height = (e2 + e4) / 2.0  # left + right edges

    aspect_ratio = width / height if height > 1e-6 else float("inf")
    area = _shoelace_area(pts)

    return {
        "aspect_ratio": aspect_ratio,
        "area": area,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# CVAT XML parsing
# ---------------------------------------------------------------------------

def parse_cvat_xml(xml_path: Path, polygon_label: str | None) -> tuple[ET.Element, list[ImageInfo]]:
    """解析 CVAT XML，返回 (root, 图片列表)。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    images: list[ImageInfo] = []

    for img_el in root.findall("image"):
        img = ImageInfo(
            id=int(img_el.get("id", "0")),
            name=img_el.get("name", ""),
            width=int(img_el.get("width", "0")),
            height=int(img_el.get("height", "0")),
            elem=img_el,
        )

        for poly_el in img_el.findall("polygon"):
            label = poly_el.get("label", "")
            if polygon_label is not None and label != polygon_label:
                continue

            pts_str = poly_el.get("points", "")
            if not pts_str:
                continue
            pts: list[tuple[float, float]] = []
            for seg in pts_str.split(";"):
                parts = seg.strip().split(",")
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))

            if len(pts) != 4:
                continue

            metrics = compute_quad_metrics(pts)
            img.polygons.append(PolygonInfo(
                elem=poly_el,
                label=label,
                points=pts,
                aspect_ratio=metrics["aspect_ratio"],
                area=metrics["area"],
                width=metrics["width"],
                height=metrics["height"],
            ))

        images.append(img)

    return root, images


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filters(
    images: list[ImageInfo],
    *,
    min_ratio: float,
    max_ratio: float,
    min_area: float,
    min_width: float,
) -> tuple[int, int]:
    """
    对所有 polygon 应用过滤条件，设置 passed/reject_reason。
    返回 (passed_count, rejected_count)。
    """
    passed = 0
    rejected = 0

    for img in images:
        for poly in img.polygons:
            reasons: list[str] = []

            if poly.aspect_ratio < min_ratio:
                reasons.append(f"ratio {poly.aspect_ratio:.2f} < {min_ratio:.1f}")
            if poly.aspect_ratio > max_ratio:
                reasons.append(f"ratio {poly.aspect_ratio:.2f} > {max_ratio:.1f}")
            if poly.area < min_area:
                reasons.append(f"area {poly.area:.0f} < {min_area:.0f}")
            if poly.width < min_width:
                reasons.append(f"width {poly.width:.1f} < {min_width:.1f}")

            if reasons:
                poly.passed = False
                poly.reject_reason = "; ".join(reasons)
                rejected += 1
            else:
                poly.passed = True
                passed += 1

    return passed, rejected


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    images: list[ImageInfo],
    *,
    min_ratio: float,
    max_ratio: float,
    verbose: bool = False,
) -> None:
    """打印过滤报告和宽高比分布直方图。"""
    all_polys: list[PolygonInfo] = []
    for img in images:
        all_polys.extend(img.polygons)

    if not all_polys:
        print("  无 polygon 标注")
        return

    ratios = [p.aspect_ratio for p in all_polys]
    passed = [p for p in all_polys if p.passed]
    rejected = [p for p in all_polys if not p.passed]

    print(f"\n  总 polygon 数: {len(all_polys)}")
    print(f"  通过过滤:     {len(passed)}")
    print(f"  被过滤:       {len(rejected)}")

    # ratio statistics
    import statistics
    print(f"\n  宽高比统计:")
    print(f"    最小值:  {min(ratios):.2f}")
    print(f"    最大值:  {max(ratios):.2f}")
    print(f"    中位数:  {statistics.median(ratios):.2f}")
    print(f"    均值:    {statistics.mean(ratios):.2f}")
    if len(ratios) > 1:
        print(f"    标准差:  {statistics.stdev(ratios):.2f}")

    # ASCII histogram of aspect ratios
    bins = [0, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, float("inf")]
    bin_labels = ["<1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0",
                  "3.0-4.0", "4.0-5.0", "5.0-6.0", "6.0-8.0", "8.0-10", ">10"]
    counts = [0] * (len(bins) - 1)
    for r in ratios:
        for bi in range(len(bins) - 1):
            if bins[bi] <= r < bins[bi + 1]:
                counts[bi] += 1
                break

    max_count = max(counts) if counts else 1
    bar_width = 30

    print(f"\n  宽高比分布直方图（过滤范围: [{min_ratio:.1f}, {max_ratio:.1f}]）:")
    for i, (label, count) in enumerate(zip(bin_labels, counts)):
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        in_range = min_ratio <= (bins[i] + bins[i + 1]) / 2 <= max_ratio if bins[i + 1] != float("inf") else False
        marker = " ✓" if in_range else " ✗"
        print(f"    {label:>8s} | {bar:<{bar_width}s} {count:>4d}{marker}")

    # Rejected details
    # if verbose and rejected:
    #     print(f"\n  被过滤的标注详情:")
    #     for img in images:
    #         for poly in img.polygons:
    #             if not poly.passed:
    #                 print(f"    {img.name}  ratio={poly.aspect_ratio:.2f}  "
    #                       f"area={poly.area:.0f}  w={poly.width:.1f}  h={poly.height:.1f}  "
    #                       f"原因: {poly.reject_reason}")


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def generate_output_xml(
    orig_root: ET.Element,
    images: list[ImageInfo],
    *,
    action: str,
    reject_suffix: str = "_rejected",
) -> str:
    """
    生成过滤后的 CVAT XML。

    action='filter': 移除被拒绝的 polygon
    action='mark':   将被拒绝的 polygon 标签加后缀
    """
    root = copy.deepcopy(orig_root)

    # If mark mode, add rejected labels to meta
    if action == "mark":
        meta = root.find("meta")
        labels_parent = None
        if meta is not None:
            for tag in ["task", "job"]:
                parent = meta.find(tag)
                if parent is not None:
                    labels_el = parent.find("labels")
                    if labels_el is not None:
                        labels_parent = labels_el
                        break

        if labels_parent is not None:
            existing_names = {lbl.findtext("name", "") for lbl in labels_parent.findall("label")}
            for img in images:
                for poly in img.polygons:
                    if not poly.passed:
                        rej_name = poly.label + reject_suffix
                        if rej_name not in existing_names:
                            existing_names.add(rej_name)
                            lbl = ET.SubElement(labels_parent, "label")
                            ET.SubElement(lbl, "name").text = rej_name
                            ET.SubElement(lbl, "color").text = "#ff0000"
                            ET.SubElement(lbl, "type").text = "polygon"
                            ET.SubElement(lbl, "attributes")

    # Build lookup: polygon elem → PolygonInfo
    poly_lookup: dict[int, PolygonInfo] = {}
    for img in images:
        for poly in img.polygons:
            poly_lookup[id(poly.elem)] = poly

    for img_el in root.findall("image"):
        to_remove: list[ET.Element] = []
        for poly_el in img_el.findall("polygon"):
            orig_id = None
            for img in images:
                for poly in img.polygons:
                    pts_str = poly_el.get("points", "")
                    orig_pts_str = ";".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly.points)
                    # Match by points + label + image name
                    if (pts_str == poly_el.get("points", "")
                            and poly.label == poly_el.get("label", "")
                            and img.name == img_el.get("name", "")):
                        orig_id = id(poly.elem)
                        break
                if orig_id is not None:
                    break

            if orig_id is not None and orig_id in poly_lookup:
                info = poly_lookup[orig_id]
                if not info.passed:
                    if action == "filter":
                        to_remove.append(poly_el)
                    elif action == "mark":
                        poly_el.set("label", info.label + reject_suffix)

        for el in to_remove:
            img_el.remove(el)

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


def generate_output_xml_simple(
    xml_path: Path,
    images: list[ImageInfo],
    *,
    action: str,
    reject_suffix: str = "_rejected",
) -> str:
    """
    更简洁的输出方式：直接在原始 XML 上操作。
    利用 polygon 的 points 字符串精确匹配。
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # Build reject set: (image_name, points_str) → PolygonInfo
    reject_map: dict[tuple[str, str, str], PolygonInfo] = {}
    for img in images:
        for poly in img.polygons:
            if not poly.passed:
                pts_str = ";".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly.points)
                reject_map[(img.name, poly.label, pts_str)] = poly

    # Add rejected label to meta if mark mode
    if action == "mark" and reject_map:
        meta = root.find("meta")
        if meta is not None:
            for tag in ["task", "job"]:
                parent = meta.find(tag)
                if parent is not None:
                    labels_el = parent.find("labels")
                    if labels_el is not None:
                        existing = {lbl.findtext("name", "") for lbl in labels_el.findall("label")}
                        for (_, lbl_name, _) in reject_map:
                            rej_name = lbl_name + reject_suffix
                            if rej_name not in existing:
                                existing.add(rej_name)
                                lbl = ET.SubElement(labels_el, "label")
                                ET.SubElement(lbl, "name").text = rej_name
                                ET.SubElement(lbl, "color").text = "#ff0000"
                                ET.SubElement(lbl, "type").text = "polygon"
                                ET.SubElement(lbl, "attributes")
                        break

    for img_el in root.findall("image"):
        img_name = img_el.get("name", "")
        to_remove: list[ET.Element] = []

        for poly_el in img_el.findall("polygon"):
            label = poly_el.get("label", "")
            pts_raw = poly_el.get("points", "")

            # Try matching with raw points first, then with reformatted points
            key_raw = (img_name, label, pts_raw)
            # Reformat points to match our stored format
            pts_reformatted = []
            for seg in pts_raw.split(";"):
                parts = seg.strip().split(",")
                if len(parts) >= 2:
                    pts_reformatted.append(f"{float(parts[0]):.2f},{float(parts[1]):.2f}")
            key_fmt = (img_name, label, ";".join(pts_reformatted))

            matched = reject_map.get(key_raw) or reject_map.get(key_fmt)
            if matched is not None:
                if action == "filter":
                    to_remove.append(poly_el)
                elif action == "mark":
                    poly_el.set("label", label + reject_suffix)

        for el in to_remove:
            img_el.remove(el)

    ET.indent(root, space="  ")
    xml_body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml_body + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="基于宽高比过滤 CVAT XML 中的 polygon 标注（去除非前表面标注）",
    )
    ap.add_argument("--cvat_xml", type=str, required=True,
                    help="输入的 CVAT XML 文件")
    ap.add_argument("--output", type=str, default=None,
                    help="输出的 CVAT XML 文件路径（action=report 时不需要）")
    ap.add_argument("--polygon_label", type=str, default=None,
                    help="只过滤此标签的 polygon（默认处理所有 polygon）")

    # Ratio filtering
    ratio_grp = ap.add_argument_group("宽高比过滤参数")
    ratio_grp.add_argument("--target_ratio", type=float, default=5.0,
                           help="目标宽高比（width/height，默认 5.0）")
    ratio_grp.add_argument("--tolerance", type=float, default=0.5,
                           help="容忍度（0~1 的分数，默认 0.5 即 ±50%%）"
                                "，实际范围 = target_ratio * [1-tol, 1+tol]")
    ratio_grp.add_argument("--min_ratio", type=float, default=None,
                           help="直接指定最小宽高比（覆盖 target_ratio/tolerance）")
    ratio_grp.add_argument("--max_ratio", type=float, default=None,
                           help="直接指定最大宽高比（覆盖 target_ratio/tolerance）")

    # Additional filters
    extra_grp = ap.add_argument_group("附加过滤参数")
    extra_grp.add_argument("--min_area", type=float, default=0,
                           help="最小面积（像素²，默认 0 不限制）")
    extra_grp.add_argument("--min_width", type=float, default=0,
                           help="最小宽度（像素，默认 0 不限制）")

    # Action
    ap.add_argument("--action", type=str, default="filter",
                    choices=["filter", "mark", "report"],
                    help="操作: filter（移除）, mark（重命名标签）, report（仅报告）")
    ap.add_argument("--reject_suffix", type=str, default="_rejected",
                    help="mark 模式下被拒绝标签的后缀（默认 _rejected）")
    ap.add_argument("--verbose", action="store_true",
                    help="打印每个被过滤标注的详情")
    args = ap.parse_args()

    cvat_path = Path(args.cvat_xml).expanduser().resolve()
    if not cvat_path.exists():
        print(f"[ERROR] 文件不存在: {cvat_path}")
        return 1

    if args.action != "report" and not args.output:
        print("[ERROR] filter/mark 模式需要指定 --output")
        return 1

    # Compute ratio range
    if args.min_ratio is not None:
        min_ratio = args.min_ratio
    else:
        min_ratio = args.target_ratio * (1.0 - args.tolerance)

    if args.max_ratio is not None:
        max_ratio = args.max_ratio
    else:
        max_ratio = args.target_ratio * (1.0 + args.tolerance)

    print(f"过滤参数:")
    print(f"  目标宽高比: {args.target_ratio:.1f}")
    print(f"  容忍度:     {args.tolerance:.0%}")
    print(f"  有效范围:   [{min_ratio:.2f}, {max_ratio:.2f}]")
    if args.min_area > 0:
        print(f"  最小面积:   {args.min_area:.0f} px²")
    if args.min_width > 0:
        print(f"  最小宽度:   {args.min_width:.1f} px")

    # Parse XML
    print(f"\n解析 CVAT XML: {cvat_path}")
    _root, images = parse_cvat_xml(cvat_path, args.polygon_label)

    total_images = len(images)
    total_polys = sum(len(img.polygons) for img in images)
    print(f"  图片数: {total_images}")
    print(f"  polygon 数: {total_polys}")
    if args.polygon_label:
        print(f"  过滤标签: {args.polygon_label}")

    if total_polys == 0:
        print("无 polygon 标注，退出")
        return 0

    # Apply filters
    passed, rejected = apply_filters(
        images,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        min_area=args.min_area,
        min_width=args.min_width,
    )

    # Report
    print(f"\n{'=' * 60}")
    print("过滤结果")
    print("=" * 60)
    print_report(images, min_ratio=min_ratio, max_ratio=max_ratio, verbose=args.verbose)

    # Generate output
    if args.action != "report":
        output_path = Path(args.output).expanduser().resolve()
        xml_str = generate_output_xml_simple(
            cvat_path,
            images,
            action=args.action,
            reject_suffix=args.reject_suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(xml_str, encoding="utf-8")

        if args.action == "filter":
            print(f"\n已输出过滤后的 XML: {output_path}")
            print(f"  保留 {passed} 个 polygon，移除 {rejected} 个")
        elif args.action == "mark":
            print(f"\n已输出标记后的 XML: {output_path}")
            print(f"  {rejected} 个 polygon 标签已加后缀 '{args.reject_suffix}'")
            print(f"  可导入 CVAT 查看哪些被标记为 rejected")
    else:
        print(f"\n[report 模式] 未生成输出文件")
        print(f"  提示: 添加 --output filtered.xml --action filter 来实际过滤")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
