#!/usr/bin/env python3
"""
将文件夹形式的 COCO 数据集完整转换为 CVAT 标注格式 (CVAT for images 1.1)。

支持的标注类型转换:
  - keypoints   → CVAT <skeleton> 元素（含 <points> 子元素，保留 visibility）
  - segmentation → CVAT <polygon> 元素
  - bbox         → CVAT <box> 元素（仅在无 keypoints 且无 segmentation 时输出）
  - attributes   → CVAT <attribute> 子元素

使用说明
========

数据集文件夹结构要求:
    <dataset_root>/
    ├── coco_files/          # 存放 *.coco.json 标注文件
    │   ├── xxxx.coco.json
    │   └── ...
    └── image_groups/        # 存放图片，路径与 coco json 中的 image_title 对应
        └── ...

基本用法 (保留原始标签名，不进行标签转换):
    python convert_coco_folder_to_cvat_polygon.py /path/to/dataset_root

指定关键点名称 (COCO categories 中缺少 keypoints 字段时需要):
    python convert_coco_folder_to_cvat_polygon.py /path/to/dataset_root \\
        --keypoint-names '{"Pallet Face": ["top_left_corner", "top_right_corner",
            "bottom_right_corner", "bottom_left_corner", "centre_point"]}'

标签名映射 (可选，需要转换标签名时使用):
    python convert_coco_folder_to_cvat_polygon.py /path/to/dataset_root \\
        --label-map '{"Pallet Face": "pallet"}'

同时输出 bbox (默认在有 keypoints/segmentation 时省略 bbox 以避免冗余):
    python convert_coco_folder_to_cvat_polygon.py /path/to/dataset_root --include-bbox

将 bbox 输出为 polygon 而非 box:
    python convert_coco_folder_to_cvat_polygon.py /path/to/dataset_root --bbox-as-polygon

输出目录结构:
    <output_dir>/
    ├── annotations.xml      # CVAT 标注 XML
    └── images/              # 图片的符号链接（默认）或复制
        └── ...

选项:
    --copy             复制图片而非创建符号链接（默认使用符号链接以节省空间）
    --label-map        标签名映射 (JSON 字符串或 JSON 文件路径)
    --keypoint-names   关键点名称 (JSON 字符串或 JSON 文件路径)
    --include-bbox     即使有 keypoints/segmentation 也额外输出 bbox
    --bbox-as-polygon  将 bbox 输出为 polygon 而非 box
"""

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

VISIBILITY_MAP = {
    0: ("1", "0"),  # not labeled  → outside=1, occluded=0
    1: ("0", "1"),  # occluded     → outside=0, occluded=1
    2: ("0", "0"),  # visible      → outside=0, occluded=0
}


def indent_xml(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_json_arg(value):
    if value is None:
        return {}
    p = Path(value)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return json.loads(value)


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

def resolve_image_path(image_obj, image_groups_dir, basename_index):
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
        matches = basename_index.get(os.path.basename(val), [])
        if len(matches) == 1:
            return matches[0]

    return None


def build_basename_index(image_groups_dir):
    index = defaultdict(list)
    for root, _dirs, files in os.walk(image_groups_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                index[fname].append(Path(root) / fname)
    return index


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bbox_to_box_attrs(bbox, w, h):
    if not bbox or len(bbox) < 4:
        return None
    x, y, bw, bh = (float(v) for v in bbox[:4])
    wmax, hmax = float(max(w - 1, 0)), float(max(h - 1, 0))
    return {
        "xtl": f"{clamp(x, 0, wmax):.2f}",
        "ytl": f"{clamp(y, 0, hmax):.2f}",
        "xbr": f"{clamp(x + bw, 0, wmax):.2f}",
        "ybr": f"{clamp(y + bh, 0, hmax):.2f}",
    }


def bbox_to_polygon_pts(bbox, w, h):
    if not bbox or len(bbox) < 4:
        return None
    x, y, bw, bh = (float(v) for v in bbox[:4])
    wmax, hmax = float(max(w - 1, 0)), float(max(h - 1, 0))
    corners = [
        (clamp(x, 0, wmax), clamp(y, 0, hmax)),
        (clamp(x + bw, 0, wmax), clamp(y, 0, hmax)),
        (clamp(x + bw, 0, wmax), clamp(y + bh, 0, hmax)),
        (clamp(x, 0, wmax), clamp(y + bh, 0, hmax)),
    ]
    return ";".join(f"{px:.2f},{py:.2f}" for px, py in corners)


def seg_to_polygon_pts(poly, w, h):
    if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
        return None
    wmax, hmax = float(max(w - 1, 0)), float(max(h - 1, 0))
    points = []
    for i in range(0, len(poly), 2):
        points.append((clamp(float(poly[i]), 0, wmax), clamp(float(poly[i + 1]), 0, hmax)))
    if len(points) < 3:
        return None
    return ";".join(f"{px:.2f},{py:.2f}" for px, py in points)


# ---------------------------------------------------------------------------
# COCO attributes → CVAT attributes
# ---------------------------------------------------------------------------

def extract_cvat_attributes(ann):
    attrs = {}
    raw = ann.get("attributes")
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "classifications" and not v:
                continue
            attrs[k] = str(v)
    if ann.get("iscrowd"):
        attrs["iscrowd"] = "1"
    return attrs


def append_cvat_attrs(parent_el, attrs):
    for name in sorted(attrs):
        el = ET.SubElement(parent_el, "attribute", {"name": name})
        el.text = attrs[name]


# ---------------------------------------------------------------------------
# Keypoint name resolution
# ---------------------------------------------------------------------------

def get_keypoint_names(category, num_kps, user_kp_names):
    cat_name = category.get("name", "")
    if cat_name in user_kp_names:
        names = user_kp_names[cat_name]
        if len(names) >= num_kps:
            return names[:num_kps]
    if "keypoints" in category and len(category["keypoints"]) >= num_kps:
        return category["keypoints"][:num_kps]
    return [f"kp_{i}" for i in range(num_kps)]


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build_cvat_xml_and_image_map(
    input_dir,
    label_map=None,
    kp_names_map=None,
    bbox_as_polygon=False,
    include_bbox=False,
):
    label_map = label_map or {}
    kp_names_map = kp_names_map or {}

    coco_dir = input_dir / "coco_files"
    image_groups_dir = input_dir / "image_groups"
    if not coco_dir.is_dir():
        raise RuntimeError(f"coco_files 目录不存在: {coco_dir}")
    if not image_groups_dir.is_dir():
        raise RuntimeError(f"image_groups 目录不存在: {image_groups_dir}")

    coco_jsons = sorted(coco_dir.glob("*.coco.json"))
    if not coco_jsons:
        raise RuntimeError(f"在 {coco_dir} 中未找到 .coco.json 文件")

    basename_index = build_basename_index(image_groups_dir)

    xml_root = ET.Element("annotations")
    ET.SubElement(xml_root, "version").text = "1.1"
    meta = ET.SubElement(xml_root, "meta")
    task = ET.SubElement(meta, "task")
    labels_el = ET.SubElement(task, "labels")

    # label_name -> {type, sublabels, attributes}
    label_registry = OrderedDict()
    image_id_counter = 0
    source_to_output = {}
    output_used = set()

    stats = defaultdict(int)

    # Pre-scan: determine keypoint counts / names per category
    cat_kp_cache = {}
    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)
        cats = {c["id"]: c for c in data.get("categories", [])}
        for ann in data.get("annotations", []):
            kps = ann.get("keypoints", [])
            cid = ann.get("category_id")
            if kps and cid not in cat_kp_cache and cid in cats:
                num = len(kps) // 3
                cat_kp_cache[cid] = get_keypoint_names(cats[cid], num, kp_names_map)

    def register_label(name, ltype, sublabels=None):
        if name not in label_registry:
            label_registry[name] = {
                "type": ltype,
                "sublabels": sublabels or [],
                "attributes": set(),
            }

    # Main pass
    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)

        cat_by_id = {}
        for c in data.get("categories", []):
            cat_by_id[c["id"]] = c.get("name", f"category_{c['id']}")

        images = data.get("images", [])
        annotations = data.get("annotations", [])
        stats["images_total"] += len(images)
        stats["annotations_total"] += len(annotations)

        ann_by_image = defaultdict(list)
        for ann in annotations:
            ann_by_image[ann.get("image_id")].append(ann)

        for img in images:
            w = int(img.get("width", 0) or 0)
            h = int(img.get("height", 0) or 0)
            if w <= 0 or h <= 0:
                continue

            src_path = resolve_image_path(img, image_groups_dir, basename_index)
            if src_path is None:
                stats["images_missing"] += 1
                continue

            src_key = str(src_path)
            if src_key not in source_to_output:
                base = src_path.name
                out_name = f"images/{base}"
                if out_name in output_used:
                    stem, ext = src_path.stem, src_path.suffix
                    n = 2
                    while f"images/{stem}_{n}{ext}" in output_used:
                        n += 1
                    out_name = f"images/{stem}_{n}{ext}"
                source_to_output[src_key] = out_name
                output_used.add(out_name)

            image_el = ET.SubElement(
                xml_root, "image",
                {"id": str(image_id_counter), "name": source_to_output[src_key],
                 "width": str(w), "height": str(h)},
            )
            image_id_counter += 1
            stats["images_exported"] += 1

            for ann in ann_by_image.get(img.get("id"), []):
                cat_id = ann.get("category_id")
                raw_name = cat_by_id.get(cat_id, f"category_{cat_id}")
                label = label_map.get(raw_name, raw_name)
                cvat_attrs = extract_cvat_attributes(ann)

                kps = ann.get("keypoints", [])
                has_kps = bool(kps) and len(kps) >= 3
                seg = ann.get("segmentation")
                has_seg = isinstance(seg, list) and any(
                    isinstance(p, list) and len(p) >= 6 for p in seg
                )
                has_bbox = bool(ann.get("bbox")) and len(ann["bbox"]) >= 4

                emitted_rich = False

                # --- Keypoints → Skeleton ---
                if has_kps:
                    num_kps = len(kps) // 3
                    kp_names = cat_kp_cache.get(cat_id, [f"kp_{i}" for i in range(num_kps)])
                    register_label(label, "skeleton", list(kp_names))
                    label_registry[label]["attributes"].update(cvat_attrs)

                    skel_el = ET.SubElement(
                        image_el, "skeleton",
                        {"label": label, "source": "manual", "occluded": "0", "z_order": "0"},
                    )
                    append_cvat_attrs(skel_el, cvat_attrs)

                    for ki in range(num_kps):
                        kx = clamp(float(kps[ki * 3]), 0, float(max(w - 1, 0)))
                        ky = clamp(float(kps[ki * 3 + 1]), 0, float(max(h - 1, 0)))
                        kv = int(float(kps[ki * 3 + 2]))
                        outside, occluded = VISIBILITY_MAP.get(kv, ("0", "0"))
                        kp_label = kp_names[ki] if ki < len(kp_names) else f"kp_{ki}"
                        ET.SubElement(
                            skel_el, "points",
                            {"label": kp_label, "source": "manual",
                             "outside": outside, "occluded": occluded,
                             "points": f"{kx:.2f},{ky:.2f}", "z_order": "0"},
                        )
                        stats[f"kp_{['outside','occluded','visible'][kv]}"] += 1
                        stats["kp_total"] += 1

                    stats["skeletons"] += 1
                    emitted_rich = True

                # --- Segmentation → Polygon ---
                if has_seg:
                    seg_label = label
                    register_label(seg_label, "polygon")
                    label_registry[seg_label]["attributes"].update(cvat_attrs)
                    for poly in seg:
                        pts = seg_to_polygon_pts(poly, w, h)
                        if pts:
                            pel = ET.SubElement(
                                image_el, "polygon",
                                {"label": seg_label, "source": "manual", "occluded": "0",
                                 "points": pts, "z_order": "0"},
                            )
                            append_cvat_attrs(pel, cvat_attrs)
                            stats["polygons_seg"] += 1
                    emitted_rich = True

                # --- BBox → Box / Polygon ---
                if has_bbox and (include_bbox or not emitted_rich):
                    bbox_label = label
                    if bbox_as_polygon:
                        register_label(bbox_label, "polygon")
                        label_registry[bbox_label]["attributes"].update(cvat_attrs)
                        pts = bbox_to_polygon_pts(ann["bbox"], w, h)
                        if pts:
                            pel = ET.SubElement(
                                image_el, "polygon",
                                {"label": bbox_label, "source": "manual", "occluded": "0",
                                 "points": pts, "z_order": "0"},
                            )
                            append_cvat_attrs(pel, cvat_attrs)
                            stats["polygons_bbox"] += 1
                    else:
                        register_label(bbox_label, "rectangle")
                        label_registry[bbox_label]["attributes"].update(cvat_attrs)
                        ba = bbox_to_box_attrs(ann["bbox"], w, h)
                        if ba:
                            bel = ET.SubElement(
                                image_el, "box",
                                {"label": bbox_label, "source": "manual", "occluded": "0",
                                 "z_order": "0", **ba},
                            )
                            append_cvat_attrs(bel, cvat_attrs)
                            stats["boxes"] += 1

    # Write label definitions into <meta>
    for lname, linfo in label_registry.items():
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = lname
        ET.SubElement(lbl, "type").text = linfo["type"]
        if linfo["sublabels"]:
            subs = ET.SubElement(lbl, "sublabels")
            for sl in linfo["sublabels"]:
                s = ET.SubElement(subs, "sublabel")
                ET.SubElement(s, "name").text = sl
                ET.SubElement(s, "type").text = "points"
        if linfo["attributes"]:
            attrs_el = ET.SubElement(lbl, "attributes")
            for aname in sorted(linfo["attributes"]):
                a = ET.SubElement(attrs_el, "attribute")
                ET.SubElement(a, "name").text = aname
                ET.SubElement(a, "input_type").text = "text"

    indent_xml(xml_root)
    xml_bytes = ET.tostring(xml_root, encoding="utf-8", xml_declaration=True)
    return xml_bytes, source_to_output, dict(stats), label_registry


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_cvat_folder(output_dir, xml_bytes, source_to_output, use_copy=False):
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations.xml").write_bytes(xml_bytes)

    for src_str, out_rel in source_to_output.items():
        src, dst = Path(src_str), output_dir / out_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if use_copy:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.resolve())


# ---------------------------------------------------------------------------
# Label summary
# ---------------------------------------------------------------------------

def print_label_guide(label_registry):
    print()
    print("=" * 64)
    print("  需要在 CVAT 中创建的标签（创建任务/项目时配置）")
    print("=" * 64)
    for i, (name, info) in enumerate(label_registry.items(), 1):
        print(f"\n  [{i}] 标签名: {name}")
        print(f"      类型:   {info['type']}")
        if info["sublabels"]:
            print(f"      子标签 (关键点):")
            for j, sl in enumerate(info["sublabels"]):
                print(f"        {j}. {sl}")
        if info["attributes"]:
            print(f"      属性:")
            for a in sorted(info["attributes"]):
                print(f"        - {a}  (input_type: text)")
    print()
    print("=" * 64)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="将文件夹形式的 COCO 数据集完整转换为 CVAT 标注格式 (CVAT for images 1.1)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="数据集根目录（包含 coco_files/ 和 image_groups/）")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="输出目录（默认: <input_dir>_cvat/）")
    parser.add_argument("--copy", action="store_true",
                        help="复制图片而非符号链接")
    parser.add_argument("--label-map", default=None,
                        help="标签名映射 JSON（字符串或文件路径），格式: {\"旧名\": \"新名\"}")
    parser.add_argument("--keypoint-names", default=None,
                        help="关键点名称 JSON（字符串或文件路径），格式: {\"类别名\": [\"kp0\", ...]}")
    parser.add_argument("--include-bbox", action="store_true",
                        help="即使有 keypoints/segmentation 也额外输出 bbox")
    parser.add_argument("--bbox-as-polygon", action="store_true",
                        help="将 bbox 输出为 polygon 而非 box")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    output_dir = (Path(args.output_dir).resolve() if args.output_dir
                  else input_dir.parent / f"{input_dir.name}_cvat")

    label_map = parse_json_arg(args.label_map)
    kp_names_map = parse_json_arg(args.keypoint_names)

    xml_bytes, src_map, stats, label_reg = build_cvat_xml_and_image_map(
        input_dir,
        label_map=label_map,
        kp_names_map=kp_names_map,
        bbox_as_polygon=args.bbox_as_polygon,
        include_bbox=args.include_bbox,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_cvat_folder(output_dir, xml_bytes, src_map, use_copy=args.copy)

    n_coco = len(list((input_dir / "coco_files").glob("*.coco.json")))
    print(f"input_dir:            {input_dir}")
    print(f"output_dir:           {output_dir}")
    print(f"coco_json_files:      {n_coco}")
    print(f"images_total:         {stats.get('images_total', 0)}")
    print(f"images_exported:      {stats.get('images_exported', 0)}")
    print(f"images_missing:       {stats.get('images_missing', 0)}")
    print(f"annotations_total:    {stats.get('annotations_total', 0)}")
    print(f"skeletons:            {stats.get('skeletons', 0)}")
    print(f"  keypoints_total:    {stats.get('kp_total', 0)}")
    print(f"  kp_visible:         {stats.get('kp_visible', 0)}")
    print(f"  kp_occluded:        {stats.get('kp_occluded', 0)}")
    print(f"  kp_outside:         {stats.get('kp_outside', 0)}")
    print(f"boxes:                {stats.get('boxes', 0)}")
    print(f"polygons_from_seg:    {stats.get('polygons_seg', 0)}")
    print(f"polygons_from_bbox:   {stats.get('polygons_bbox', 0)}")
    print(f"format:               CVAT for images 1.1")
    print(f"image_mode:           {'copy' if args.copy else 'symlink'}")

    print_label_guide(label_reg)


if __name__ == "__main__":
    main()
