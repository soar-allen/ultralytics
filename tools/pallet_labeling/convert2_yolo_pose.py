#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将现有标注导出为 Ultralytics YOLO-Pose 训练数据（4 点：tl,tr,br,bl）。

支持输入：
  1) Roboflow 导出的 COCO detection JSON（bbox=[x,y,w,h]）
  2) Roboflow/YOLO detection 格式（*.txt，每行: cls xc yc w h），支持 Roboflow 标准目录（train/valid/test）
  3) Pascal VOC XML（可选）

输出结构（out_dir）：
  out_dir/
    images/{train,val,test}/...
    labels/{train,val,test}/...  # YOLO-Pose txt
    data.yaml

YOLO-Pose 标签行格式（kpt_shape=[4,3]）：
  cls xc yc w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
其中坐标均为 0~1 归一化，v 为可见性（这里写 2 表示可见）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import yaml
from PIL import Image


def _order_quad_tl_tr_br_bl(pts4: np.ndarray) -> np.ndarray:
    """将 4 点排序为 tl,tr,br,bl（输入/输出均为 (4,2)）。"""
    pts = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return np.stack([tl, tr, br, bl], axis=0)


def _xyxy_from_quad(quad: np.ndarray) -> tuple[float, float, float, float]:
    x1 = float(np.min(quad[:, 0]))
    y1 = float(np.min(quad[:, 1]))
    x2 = float(np.max(quad[:, 0]))
    y2 = float(np.max(quad[:, 1]))
    return x1, y1, x2, y2


def _xywh_norm_from_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def _kpts_norm(quad: np.ndarray, w: int, h: int, v: int = 2) -> list[float]:
    out: list[float] = []
    for x, y in quad.tolist():
        out.extend([float(x) / w, float(y) / h, float(v)])
    return out


def _safe_relpath(p: Path) -> str:
    return p.as_posix().lstrip("./")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path, *, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy_images:
        shutil.copy2(src, dst)
        return
    # symlink
    try:
        os.symlink(src, dst)
    except FileExistsError:
        return
    except OSError:
        # fallback copy
        shutil.copy2(src, dst)


@dataclass(frozen=True)
class Item:
    rel_img: Path  # output relative path under images/{split}/
    img_src: Optional[Path]
    img_w: int
    img_h: int
    instances: list[tuple[int, np.ndarray]]  # [(cls, quad)], quad is (4,2) tl,tr,br,bl


def _pose_line(cls_id: int, quad: np.ndarray, w: int, h: int) -> str:
    x1, y1, x2, y2 = _xyxy_from_quad(quad)
    xc, yc, bw, bh = _xywh_norm_from_xyxy(x1, y1, x2, y2, w, h)
    kpts = _kpts_norm(quad, w, h, v=2)
    cols = [float(cls_id), float(xc), float(yc), float(bw), float(bh)] + kpts
    return " ".join([f"{c:.6f}" for c in cols])


def _image_size(p: Path) -> tuple[int, int]:
    with Image.open(p) as im:
        w, h = im.size
    return int(w), int(h)


def _build_basename_index(image_roots: list[Path]) -> dict[str, Path]:
    """
    为快速查找图片构建 basename->Path 索引。
    - 若出现重名，后者覆盖前者（一般你数据集里 basename 应该唯一）
    """
    idx: dict[str, Path] = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for root in image_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                idx[p.name] = p.resolve()
    return idx


def _load_coco_json(
    coco_json: Path,
    images_dir: Path,
    label_filter: set[str] | None,
    *,
    prefix: str,
    default_cls: int,
    allow_missing_images: bool,
    basename_index: dict[str, Path] | None,
) -> list[Item]:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    imgs = {int(im["id"]): im for im in (data.get("images", []) or [])}
    cats = {int(c["id"]): str(c.get("name", c["id"])) for c in (data.get("categories", []) or [])}
    ann_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for a in (data.get("annotations", []) or []):
        if a.get("iscrowd", 0):
            continue
        img_id = int(a["image_id"])
        ann_by_img[img_id].append(a)

    items: list[Item] = []
    for img_id, im in imgs.items():
        file_name = str(im.get("file_name", ""))
        if not file_name:
            continue
        img_rel = Path(file_name)
        img_src: Optional[Path] = (images_dir / img_rel).resolve()
        if not img_src.exists():
            if basename_index is not None:
                cand = basename_index.get(img_rel.name)
                if cand is not None and cand.exists():
                    img_src = cand
        if not img_src.exists():
            if not allow_missing_images:
                continue
            img_src = None
        w = int(im.get("width", 0) or 0)
        h = int(im.get("height", 0) or 0)
        if w <= 0 or h <= 0:
            if img_src is None:
                continue
            w, h = _image_size(img_src)

        instances: list[tuple[int, np.ndarray]] = []
        for a in ann_by_img.get(img_id, []):
            cat_name = cats.get(int(a.get("category_id", -1)), "")
            if label_filter is not None and cat_name not in label_filter:
                continue
            bbox = a.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = [float(v) for v in bbox]
            # axis-aligned -> quad corners
            quad = np.array(
                [[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]],
                dtype=np.float32,
            )
            quad = _order_quad_tl_tr_br_bl(quad)
            instances.append((int(default_cls), quad))

        if instances:
            out_rel = Path(prefix) / img_rel
            items.append(Item(rel_img=out_rel, img_src=img_src.resolve() if img_src else None, img_w=w, img_h=h, instances=instances))
    return items


def _scan_images_by_stem(images_dir: Path) -> dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out: dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p.resolve()
    return out


def _load_yolo_det_dir(
    labels_dir: Path,
    images_dir: Path,
    label_filter: set[str] | None,
    *,
    prefix: str,
    default_cls: int,
    allow_missing_images: bool,
    basename_index: dict[str, Path] | None,
) -> list[Item]:
    """
    读取 YOLO detection txt（cls xc yc w h），输出 axis-aligned 四角。
    注意：该格式没有类别名，label_filter 无法生效（保留所有 cls）。
    """
    items: list[Item] = []
    img_by_stem = _scan_images_by_stem(images_dir) if images_dir.exists() else {}
    for lp in sorted(labels_dir.rglob("*.txt")):
        stem = lp.stem
        img_src: Optional[Path] = img_by_stem.get(stem)
        if img_src is None and basename_index is not None:
            # try basename lookup
            for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                cand = basename_index.get(stem + ext)
                if cand is not None and cand.exists():
                    img_src = cand.resolve()
                    break
        if img_src is None or (not img_src.exists()):
            if not allow_missing_images:
                continue
            img_w = img_h = 0
        else:
            img_w, img_h = _image_size(img_src)

        lines = [x.strip() for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not lines:
            continue
        instances: list[tuple[int, np.ndarray]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls, xc, yc, bw, bh = [float(v) for v in parts[:5]]
            cls_i = int(default_cls) if default_cls >= 0 else int(cls)
            if img_src is None or (img_w <= 0 or img_h <= 0):
                # 没有图片尺寸无法反归一化
                continue
            x1 = (xc - bw / 2.0) * img_w
            y1 = (yc - bh / 2.0) * img_h
            x2 = (xc + bw / 2.0) * img_w
            y2 = (yc + bh / 2.0) * img_h
            quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            quad = _order_quad_tl_tr_br_bl(quad)
            instances.append((cls_i, quad))
        if instances:
            # label 文件名与图片同 stem；输出路径用 prefix + 相对路径（以 labels_dir 为根近似保持结构）
            rel_img = Path(prefix) / (stem + (img_src.suffix if img_src else ".jpg"))
            items.append(Item(rel_img=rel_img, img_src=img_src.resolve() if img_src else None, img_w=img_w, img_h=img_h, instances=instances))
    return items


def _load_voc_dir(
    voc_dir: Path,
    images_dir: Path,
    label_filter: set[str] | None,
    *,
    prefix: str,
    default_cls: int,
    allow_missing_images: bool,
    basename_index: dict[str, Path] | None,
) -> list[Item]:
    items: list[Item] = []
    for xp in sorted(voc_dir.rglob("*.xml")):
        try:
            root = ET.fromstring(xp.read_text(encoding="utf-8"))
        except Exception:
            continue
        filename = root.findtext("filename", default="").strip()
        if not filename:
            filename = xp.stem + ".jpg"
        img_rel = Path(filename)
        img_src: Optional[Path] = (images_dir / img_rel).resolve()
        if not img_src.exists():
            if basename_index is not None:
                cand = basename_index.get(img_rel.name)
                if cand is not None and cand.exists():
                    img_src = cand
        if not img_src.exists():
            if not allow_missing_images:
                continue
            img_src = None
            w = h = 0
        else:
            w, h = _image_size(img_src)

        instances: list[tuple[int, np.ndarray]] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name", default="") or "").strip()
            if label_filter is not None and name not in label_filter:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
            except Exception:
                continue
            quad = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
            quad = _order_quad_tl_tr_br_bl(quad)
            instances.append((int(default_cls), quad))
        if instances and w > 0 and h > 0:
            out_rel = Path(prefix) / img_rel
            items.append(Item(rel_img=out_rel, img_src=img_src.resolve() if img_src else None, img_w=w, img_h=h, instances=instances))
    return items


def _load_roboflow_yolo_dataset(
    root_dir: Path,
    *,
    prefix: str,
    default_cls: int,
    allow_missing_images: bool,
    basename_index: dict[str, Path] | None,
) -> dict[str, list[Item]]:
    """
    读取 Roboflow 导出的 YOLO detection 数据集目录：
      root/train/images, root/train/labels
      root/valid/images, root/valid/labels
      root/test/images,  root/test/labels

    返回已按 split 划分的 items（valid 会映射为 val）。
    """
    mapping = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    out: dict[str, list[Item]] = {"train": [], "val": [], "test": []}
    for sp_in, sp_out in mapping.items():
        sp_dir = root_dir / sp_in
        if not sp_dir.exists():
            continue
        images_dir = sp_dir / "images"
        labels_dir = sp_dir / "labels"
        if not labels_dir.exists():
            continue

        # YOLO det loader expects single labels_dir + images_dir; it will build rel paths with prefix already.
        items = _load_yolo_det_dir(
            labels_dir=labels_dir,
            images_dir=images_dir,
            label_filter=None,  # YOLO det 无类别名过滤
            prefix=str(Path(prefix) / sp_in),  # 保留 split，避免同名覆盖
            default_cls=int(default_cls),
            allow_missing_images=allow_missing_images,
            basename_index=basename_index,
        )
        out[sp_out].extend(items)
    return out


def _export(
    items_by_split: dict[str, list[Item]],
    out_dir: Path,
    *,
    cls_id: int,
    copy_images: bool,
    names: list[str],
    skip_images: bool,
    force_single_cls: bool,
) -> None:
    # 聚合写入，避免多次运行时 append 重复
    label_lines: dict[Path, list[str]] = defaultdict(list)

    for sp, items in items_by_split.items():
        if not items:
            continue
        for it in items:
            rel_img = it.rel_img
            img_dst = out_dir / "images" / sp / rel_img
            lbl_dst = out_dir / "labels" / sp / (rel_img.with_suffix(".txt"))

            if not skip_images and it.img_src is not None and it.img_src.exists():
                _link_or_copy(it.img_src, img_dst, copy_images=copy_images)

            for cls_i, quad in it.instances:
                out_cls = int(cls_id) if force_single_cls else int(cls_i)
                label_lines[lbl_dst].append(_pose_line(out_cls, quad, it.img_w, it.img_h))

    for p, lines in label_lines.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # data.yaml
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
        "kpt_shape": [4, 3],
        # tl,tr,br,bl -> 水平翻转映射：tl<->tr, bl<->br
        "flip_idx": [1, 0, 3, 2],
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert COCO/Roboflow/YOLO/VOC annotations to YOLO-Pose (4 keypoints).")
    ap.add_argument("--out_dir", type=str, required=True, help="输出目录（生成 images/labels/data.yaml）")
    ap.add_argument("--class_name", type=str, default="pallet", help="输出 data.yaml 的类名（单类）")
    ap.add_argument("--cls_id", type=int, default=0, help="类别 id（单类默认 0）")
    ap.add_argument("--copy_images", action="store_true", help="复制图片到 out_dir（默认用软链接，节省空间）")
    ap.add_argument("--skip_images", action="store_true", help="仅写 labels/data.yaml，不链接/复制图片（调试/受限环境用）")
    ap.add_argument("--force_single_cls", action="store_true", help="强制所有实例类别写为 --cls_id（单类训练推荐）")
    ap.add_argument("--seed", type=int, default=0, help="随机划分 seed")
    ap.add_argument("--split", type=str, default="0.9,0.1,0.0", help="train,val,test 比例")
    ap.add_argument("--label_filter", type=str, default=None, help="只保留这些 label（逗号分隔），为空则不过滤")
    ap.add_argument("--image_roots", type=str, default=None, help="额外图片根目录列表（逗号分隔），用于按 basename 查找图片")

    # COCO
    ap.add_argument("--coco_json", type=str, default=None, help="COCO annotations.json 路径（detection bbox）")
    ap.add_argument("--coco_images_dir", type=str, default=None, help="COCO 图片根目录（与 file_name 拼接）")
    ap.add_argument("--coco_prefix", type=str, default="coco", help="COCO 输入写入输出集时的路径前缀")

    # YOLO det
    ap.add_argument("--yolo_labels_dir", type=str, default=None, help="YOLO det 标签目录（*.txt: cls xc yc w h）")
    ap.add_argument("--yolo_images_dir", type=str, default=None, help="YOLO det 图片目录（与 txt 同 stem）")
    ap.add_argument("--yolo_prefix", type=str, default="yolo_det", help="YOLO det 输入写入输出集时的路径前缀")

    # Roboflow YOLO dataset (train/valid/test)
    ap.add_argument("--roboflow_yolo_dir", type=str, default=None, help="Roboflow YOLO detection 数据集根目录（含 train/valid/test）")
    ap.add_argument("--roboflow_prefix", type=str, default="roboflow", help="Roboflow 输入写入输出集时的路径前缀")

    # VOC
    ap.add_argument("--voc_dir", type=str, default=None, help="Pascal VOC xml 目录（*.xml）")
    ap.add_argument("--voc_images_dir", type=str, default=None, help="Pascal VOC 图片目录")
    ap.add_argument("--voc_prefix", type=str, default="voc", help="VOC 输入写入输出集时的路径前缀")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    _ensure_dir(out_dir / "images")
    _ensure_dir(out_dir / "labels")

    label_filter = None
    if args.label_filter:
        label_filter = {x.strip() for x in args.label_filter.split(",") if x.strip()}

    names = [str(args.class_name)]

    items_by_split: dict[str, list[Item]] = {"train": [], "val": [], "test": []}

    extra_roots: list[Path] = []
    if args.image_roots:
        extra_roots = [Path(x.strip()).expanduser().resolve() for x in str(args.image_roots).split(",") if x.strip()]
    basename_index = _build_basename_index(extra_roots) if extra_roots else None

    # COCO: usually already split outside; here we dump all into train by default
    if args.coco_json:
        if not args.coco_images_dir:
            raise ValueError("--coco_images_dir is required when using --coco_json")
        coco_json = Path(args.coco_json).expanduser().resolve()
        coco_images_dir = Path(args.coco_images_dir).expanduser().resolve()
        cs = _load_coco_json(
            coco_json,
            coco_images_dir,
            label_filter,
            prefix=str(args.coco_prefix),
            default_cls=int(args.cls_id),
            allow_missing_images=bool(args.skip_images),
            basename_index=basename_index,
        )
        items_by_split["train"].extend(cs)

    # YOLO det
    if args.yolo_labels_dir:
        if not args.yolo_images_dir:
            raise ValueError("--yolo_images_dir is required when using --yolo_labels_dir")
        yolo_labels_dir = Path(args.yolo_labels_dir).expanduser().resolve()
        yolo_images_dir = Path(args.yolo_images_dir).expanduser().resolve()
        ys = _load_yolo_det_dir(
            yolo_labels_dir,
            yolo_images_dir,
            label_filter,
            prefix=str(args.yolo_prefix),
            default_cls=-1,  # keep original cls from txt
            allow_missing_images=bool(args.skip_images),
            basename_index=basename_index,
        )
        items_by_split["train"].extend(ys)

    # Roboflow YOLO dataset (keep its split)
    if args.roboflow_yolo_dir:
        roboflow_root = Path(args.roboflow_yolo_dir).expanduser().resolve()
        split_items = _load_roboflow_yolo_dataset(
            roboflow_root,
            prefix=str(args.roboflow_prefix),
            default_cls=-1,  # keep original cls from txt
            allow_missing_images=bool(args.skip_images),
            basename_index=basename_index,
        )
        for k in ["train", "val", "test"]:
            items_by_split[k].extend(split_items[k])

    # VOC
    if args.voc_dir:
        if not args.voc_images_dir:
            raise ValueError("--voc_images_dir is required when using --voc_dir")
        voc_dir = Path(args.voc_dir).expanduser().resolve()
        voc_images_dir = Path(args.voc_images_dir).expanduser().resolve()
        vs = _load_voc_dir(
            voc_dir,
            voc_images_dir,
            label_filter,
            prefix=str(args.voc_prefix),
            default_cls=int(args.cls_id),
            allow_missing_images=bool(args.skip_images),
            basename_index=basename_index,
        )
        items_by_split["train"].extend(vs)

    if not any(items_by_split.values()):
        raise SystemExit("No items loaded. Please provide at least one input source.")

    _export(
        items_by_split,
        out_dir,
        cls_id=int(args.cls_id),
        copy_images=bool(args.copy_images),
        names=names,
        skip_images=bool(args.skip_images),
        force_single_cls=bool(args.force_single_cls),
    )
    print(f"Wrote YOLO-Pose dataset -> {out_dir}")
    print(f"data.yaml -> {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

