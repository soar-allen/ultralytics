#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 CVAT XML 中配对的 bbox + polygon 标注转换为 YOLO-Pose 训练数据。

输入：经过 CVAT 二次确认的 XML 文件，包含：
  - pallet (box): 整个托盘的 bounding box
  - pallet_front (polygon): 前表面 4 点多边形
  两者通过 group_id 关联为一组

输出：
  out_dir/
    images/{train,val,test}/...
    labels/{train,val,test}/...   # YOLO-Pose txt
    data.yaml

YOLO-Pose 标签行格式（kpt_shape=[4,3]）：
  cls xc yc w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
  - bbox (xc yc w h) 来自 pallet box 标注
  - 关键点 (x y v) 来自 pallet_front polygon 的 4 个顶点（tl, tr, br, bl）
  - 所有坐标 0~1 归一化，v=2 表示可见

用法：
  python tools/pallet_labeling/cvat_paired_to_yolo_pose.py \
    --cvat_xml verified_annotations.xml \
    --images_dir /path/to/images \
    --out_dir /path/to/output

  # 自定义划分比例
  python tools/pallet_labeling/cvat_paired_to_yolo_pose.py \
    --cvat_xml verified.xml \
    --images_dir /path/to/images \
    --out_dir output \
    --split 0.8,0.1,0.1 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedAnnotation:
    """一个托盘的配对标注：bbox + front polygon。"""
    bbox_xtl: float
    bbox_ytl: float
    bbox_xbr: float
    bbox_ybr: float
    polygon_points: list[tuple[float, float]]


@dataclass
class ImageData:
    name: str
    width: int
    height: int
    pairs: list[PairedAnnotation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _order_quad_tl_tr_br_bl(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """将 4 点排序为 [tl, tr, br, bl]。"""
    arr = np.array(pts, dtype=np.float64)
    s = arr.sum(axis=1)
    d = arr[:, 0] - arr[:, 1]
    tl = arr[int(np.argmin(s))]
    br = arr[int(np.argmax(s))]
    tr = arr[int(np.argmax(d))]
    bl = arr[int(np.argmin(d))]
    return [(float(tl[0]), float(tl[1])),
            (float(tr[0]), float(tr[1])),
            (float(br[0]), float(br[1])),
            (float(bl[0]), float(bl[1]))]


def _polygon_bbox(pts: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# CVAT XML parsing (paired format)
# ---------------------------------------------------------------------------

def parse_paired_cvat_xml(
    xml_path: Path,
    bbox_label: str = "pallet",
    polygon_label: str = "pallet_front",
) -> list[ImageData]:
    """解析含有配对标注的 CVAT XML，按 group_id 关联 bbox 与 polygon。"""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    result: list[ImageData] = []

    for img_el in root.findall("image"):
        name = img_el.get("name", "")
        width = int(img_el.get("width", "0"))
        height = int(img_el.get("height", "0"))

        boxes_by_group: dict[int, dict[str, float]] = {}
        polys_by_group: dict[int, list[tuple[float, float]]] = {}

        for box_el in img_el.findall("box"):
            if box_el.get("label") != bbox_label:
                continue
            gid = int(box_el.get("group_id", "0"))
            if gid == 0:
                continue
            boxes_by_group[gid] = {
                "xtl": float(box_el.get("xtl", "0")),
                "ytl": float(box_el.get("ytl", "0")),
                "xbr": float(box_el.get("xbr", "0")),
                "ybr": float(box_el.get("ybr", "0")),
            }

        for poly_el in img_el.findall("polygon"):
            if poly_el.get("label") != polygon_label:
                continue
            gid = int(poly_el.get("group_id", "0"))
            if gid == 0:
                continue
            pts_str = poly_el.get("points", "")
            pts: list[tuple[float, float]] = []
            for seg in pts_str.split(";"):
                parts = seg.strip().split(",")
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
            if len(pts) == 4:
                polys_by_group[gid] = pts

        # group_id=0 无法匹配的，尝试按空间关系 fallback
        unmatched_boxes = {gid: b for gid, b in boxes_by_group.items()
                          if gid not in polys_by_group}
        unmatched_polys = {gid: p for gid, p in polys_by_group.items()
                          if gid not in boxes_by_group}

        # 先收集 group_id 匹配的
        pairs: list[PairedAnnotation] = []
        for gid in sorted(set(boxes_by_group.keys()) & set(polys_by_group.keys())):
            box = boxes_by_group[gid]
            pts = polys_by_group[gid]
            ordered = _order_quad_tl_tr_br_bl(pts)
            pairs.append(PairedAnnotation(
                bbox_xtl=box["xtl"], bbox_ytl=box["ytl"],
                bbox_xbr=box["xbr"], bbox_ybr=box["ybr"],
                polygon_points=ordered,
            ))

        # fallback: 如果有未匹配的 box 和 polygon，按 polygon 中心是否在 box 内匹配
        if unmatched_boxes and unmatched_polys:
            used_poly_gids: set[int] = set()
            for _bgid, box in unmatched_boxes.items():
                bx1, by1 = box["xtl"], box["ytl"]
                bx2, by2 = box["xbr"], box["ybr"]
                best_pgid = None
                best_dist = float("inf")
                for pgid, pts in unmatched_polys.items():
                    if pgid in used_poly_gids:
                        continue
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                        dist = ((cx - (bx1 + bx2) / 2) ** 2 + (cy - (by1 + by2) / 2) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_pgid = pgid
                if best_pgid is not None:
                    used_poly_gids.add(best_pgid)
                    ordered = _order_quad_tl_tr_br_bl(unmatched_polys[best_pgid])
                    pairs.append(PairedAnnotation(
                        bbox_xtl=box["xtl"], bbox_ytl=box["ytl"],
                        bbox_xbr=box["xbr"], bbox_ybr=box["ybr"],
                        polygon_points=ordered,
                    ))

        if pairs:
            result.append(ImageData(name=name, width=width, height=height, pairs=pairs))

    return result


# ---------------------------------------------------------------------------
# YOLO-Pose export
# ---------------------------------------------------------------------------

def _pose_line(
    cls_id: int,
    bbox: tuple[float, float, float, float],
    kpts: list[tuple[float, float]],
    img_w: int,
    img_h: int,
    visibility: int = 2,
) -> str:
    """生成一行 YOLO-Pose 标签。"""
    xtl, ytl, xbr, ybr = bbox
    bw = max(xbr - xtl, 1.0)
    bh = max(ybr - ytl, 1.0)
    xc = (xtl + xbr) / 2.0
    yc = (ytl + ybr) / 2.0

    cols: list[float] = [
        float(cls_id),
        xc / img_w, yc / img_h,
        bw / img_w, bh / img_h,
    ]
    for x, y in kpts:
        cols.extend([x / img_w, y / img_h, float(visibility)])
    return " ".join(f"{c:.6f}" for c in cols)


def _link_or_copy(src: Path, dst: Path, *, copy_mode: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_mode:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_image_path(images_dir: Path, image_name: str) -> Path | None:
    direct = images_dir / image_name
    if direct.exists():
        return direct
    basename = Path(image_name).name
    for p in images_dir.rglob(basename):
        if p.is_file():
            return p
    return None


def export_yolo_pose(
    data_list: list[ImageData],
    out_dir: Path,
    images_dir: Path | None,
    *,
    cls_id: int = 0,
    class_name: str = "pallet",
    split_ratio: tuple[float, float, float] = (0.9, 0.1, 0.0),
    seed: int = 42,
    copy_images: bool = False,
    skip_images: bool = False,
) -> None:
    """将配对标注导出为 YOLO-Pose 数据集。"""
    tr, va, te = split_ratio
    rng = random.Random(seed)
    indices = list(range(len(data_list)))
    rng.shuffle(indices)
    n = len(data_list)
    n_tr = int(round(tr * n))
    n_va = int(round(va * n))

    split_map: dict[str, list[ImageData]] = {
        "train": [data_list[i] for i in indices[:n_tr]],
        "val": [data_list[i] for i in indices[n_tr:n_tr + n_va]],
        "test": [data_list[i] for i in indices[n_tr + n_va:]],
    }

    for sp, items in split_map.items():
        if not items:
            continue
        labels_dir = out_dir / "labels" / sp
        images_out_dir = out_dir / "images" / sp
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out_dir.mkdir(parents=True, exist_ok=True)

        for item in items:
            basename = Path(item.name).name
            label_path = labels_dir / Path(basename).with_suffix(".txt")

            lines: list[str] = []
            for pair in item.pairs:
                bbox = (pair.bbox_xtl, pair.bbox_ytl, pair.bbox_xbr, pair.bbox_ybr)
                line = _pose_line(cls_id, bbox, pair.polygon_points, item.width, item.height)
                lines.append(line)

            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            if not skip_images and images_dir is not None:
                img_src = resolve_image_path(images_dir, item.name)
                if img_src is not None:
                    img_dst = images_out_dir / basename
                    _link_or_copy(img_src, img_dst, copy_mode=copy_images)

    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": [class_name],
        "kpt_shape": [4, 3],
        "flip_idx": [1, 0, 3, 2],
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    print(f"\nYOLO-Pose 数据集已生成: {out_dir}")
    print(f"  data.yaml: {yaml_path}")
    total_pairs = 0
    for sp, items in split_map.items():
        if items:
            sp_pairs = sum(len(it.pairs) for it in items)
            total_pairs += sp_pairs
            print(f"  {sp}: {len(items)} 张图片, {sp_pairs} 组标注")
    print(f"  总计: {len(data_list)} 张图片, {total_pairs} 组标注")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="CVAT 配对标注（bbox + polygon）→ YOLO-Pose 训练数据",
    )
    ap.add_argument("--cvat_xml", type=str, required=True,
                    help="CVAT XML 文件路径（含 bbox + polygon 配对标注）")
    ap.add_argument("--images_dir", type=str, default=None,
                    help="图片根目录（用于链接/复制图片到输出目录）")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="输出目录（生成 images/labels/data.yaml）")
    ap.add_argument("--bbox_label", type=str, default="pallet_body",
                    help="CVAT 中 bbox 的标签名（默认 pallet_body）")
    ap.add_argument("--polygon_label", type=str, default="pallet",
                    help="CVAT 中 polygon 的标签名（默认 pallet）")
    ap.add_argument("--class_name", type=str, default="pallet",
                    help="YOLO 输出类别名（默认 pallet）")
    ap.add_argument("--cls_id", type=int, default=0,
                    help="YOLO 类别 ID（默认 0）")
    ap.add_argument("--split", type=str, default="0.9,0.1,0.0",
                    help="train,val,test 划分比例（默认 0.9,0.1,0.0）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机划分种子（默认 42）")
    ap.add_argument("--copy_images", action="store_true",
                    help="复制图片到输出目录（默认用软链接）")
    ap.add_argument("--skip_images", action="store_true",
                    help="不链接/复制图片（仅生成标签）")
    args = ap.parse_args()

    cvat_path = Path(args.cvat_xml).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else None

    if not cvat_path.exists():
        print(f"[ERROR] CVAT XML 不存在: {cvat_path}")
        return 1

    print(f"解析 CVAT XML: {cvat_path}")
    data = parse_paired_cvat_xml(cvat_path, args.bbox_label, args.polygon_label)
    total_pairs = sum(len(d.pairs) for d in data)
    print(f"  有效图片数: {len(data)}, 配对标注数: {total_pairs}")

    if not data:
        print("[ERROR] 未找到有效的配对标注（检查标签名和 group_id 是否正确）")
        return 1

    tr, va, te = [float(x) for x in args.split.split(",")]
    if abs((tr + va + te) - 1.0) > 1e-6:
        print(f"[ERROR] split 比例之和应为 1.0，当前为 {tr + va + te:.4f}")
        return 1

    export_yolo_pose(
        data,
        out_dir,
        images_dir,
        cls_id=args.cls_id,
        class_name=args.class_name,
        split_ratio=(tr, va, te),
        seed=args.seed,
        copy_images=args.copy_images,
        skip_images=args.skip_images,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
