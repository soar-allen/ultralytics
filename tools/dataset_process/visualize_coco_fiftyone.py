#!/usr/bin/env python3
"""
使用 FiftyOne 可视化文件夹形式的 COCO 数据集（含 bbox、keypoints、attributes）。

使用说明
========

数据集文件夹结构要求（与 convert_coco_folder_to_cvat_polygon.py 一致）:
    <dataset_root>/
    ├── coco_files/          # 存放 *.coco.json 标注文件
    └── image_groups/        # 存放图片

依赖安装:
    pip install fiftyone

基本用法:
    python visualize_coco_fiftyone.py /path/to/dataset_root

指定关键点名称和骨架连接:
    python visualize_coco_fiftyone.py /path/to/dataset_root \\
        --keypoint-names '{"Pallet Face": ["top_left_corner", "top_right_corner",
            "bottom_right_corner", "bottom_left_corner", "centre_point"]}' \\
        --skeleton-edges '{"Pallet Face": [[0,1],[1,2],[2,3],[3,0]]}'

指定端口:
    python visualize_coco_fiftyone.py /path/to/dataset_root --port 5152

不自动打开浏览器 (适用于远程服务器):
    python visualize_coco_fiftyone.py /path/to/dataset_root --remote

示例:
    python visualize_coco_fiftyone.py \\
        /home/cotek/datasets/Thoro_pallet_dataset_v0.3 \\
        --keypoint-names '{"Pallet Face": ["top_left_corner", "top_right_corner",
            "bottom_right_corner", "bottom_left_corner", "centre_point"]}' \\
        --skeleton-edges '{"Pallet Face": [[0,1],[1,2],[2,3],[3,0]]}'
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import fiftyone as fo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_json_arg(value):
    if value is None:
        return {}
    p = Path(value)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return json.loads(value)


def build_basename_index(image_groups_dir):
    index = defaultdict(list)
    for root, _dirs, files in os.walk(image_groups_dir):
        for fname in files:
            if Path(fname).suffix.lower() in IMAGE_EXTS:
                index[fname].append(Path(root) / fname)
    return index


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


def get_keypoint_names(category, num_kps, user_kp_names):
    cat_name = category.get("name", "")
    if cat_name in user_kp_names:
        names = user_kp_names[cat_name]
        if len(names) >= num_kps:
            return names[:num_kps]
    if "keypoints" in category and len(category["keypoints"]) >= num_kps:
        return category["keypoints"][:num_kps]
    return [f"kp_{i}" for i in range(num_kps)]


def load_dataset(input_dir, dataset_name, kp_names_map, skeleton_edges_map):
    input_dir = Path(input_dir)
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

    # Pre-scan: keypoint names per category
    cat_kp_cache = {}
    all_cats = {}
    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)
        cats = {c["id"]: c for c in data.get("categories", [])}
        all_cats.update(cats)
        for ann in data.get("annotations", []):
            kps = ann.get("keypoints", [])
            cid = ann.get("category_id")
            if kps and cid not in cat_kp_cache and cid in cats:
                cat_kp_cache[cid] = get_keypoint_names(
                    cats[cid], len(kps) // 3, kp_names_map
                )

    # Build per-image aggregated data: {resolved_path -> {width, height, annotations}}
    image_data = {}
    stats = defaultdict(int)

    for coco_path in coco_jsons:
        with open(coco_path) as f:
            data = json.load(f)

        cat_by_id = {c["id"]: c.get("name", f"category_{c['id']}") for c in data.get("categories", [])}

        ann_by_image = defaultdict(list)
        for ann in data.get("annotations", []):
            ann_by_image[ann.get("image_id")].append(ann)

        for img in data.get("images", []):
            w = int(img.get("width", 0) or 0)
            h = int(img.get("height", 0) or 0)
            if w <= 0 or h <= 0:
                continue

            src_path = resolve_image_path(img, image_groups_dir, basename_index)
            if src_path is None:
                stats["missing"] += 1
                continue

            key = str(src_path.resolve())
            if key not in image_data:
                image_data[key] = {"width": w, "height": h, "annotations": []}
                stats["images"] += 1

            for ann in ann_by_image.get(img.get("id"), []):
                image_data[key]["annotations"].append((ann, cat_by_id))
                stats["annotations"] += 1

    # Create FiftyOne dataset
    if fo.dataset_exists(dataset_name):
        fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name)
    dataset.persistent = False

    # Register skeleton definitions for the keypoints field
    kp_skeleton_map = {}
    for cid, kp_names in cat_kp_cache.items():
        cat_name = all_cats[cid].get("name", f"category_{cid}")
        edges = skeleton_edges_map.get(cat_name, [])
        kp_skeleton_map[cat_name] = fo.KeypointSkeleton(labels=kp_names, edges=edges)

    if kp_skeleton_map:
        dataset.default_skeleton = list(kp_skeleton_map.values())[0]
        dataset.skeletons = {
            "keypoints": list(kp_skeleton_map.values())[0],
        }

    samples = []
    for filepath, info in image_data.items():
        sample = fo.Sample(filepath=filepath)
        w, h = info["width"], info["height"]

        detections = []
        keypoints = []

        for ann, cat_by_id in info["annotations"]:
            cat_id = ann.get("category_id")
            label = cat_by_id.get(cat_id, f"category_{cat_id}")

            # BBox → Detection (COCO [x,y,w,h] absolute → FiftyOne [x,y,w,h] relative)
            bbox = ann.get("bbox")
            det_kwargs = {}
            if bbox and len(bbox) >= 4:
                bx, by, bw, bh = (float(v) for v in bbox[:4])
                det_kwargs["bounding_box"] = [bx / w, by / h, bw / w, bh / h]

            # Attributes
            coco_attrs = ann.get("attributes", {})
            if isinstance(coco_attrs, dict):
                for ak, av in coco_attrs.items():
                    if ak == "classifications" and not av:
                        continue
                    det_kwargs[ak] = av

            if det_kwargs.get("bounding_box"):
                detections.append(fo.Detection(label=label, **det_kwargs))

            # Keypoints → fo.Keypoint (normalized [0,1] coords)
            kps = ann.get("keypoints", [])
            if kps and len(kps) >= 3:
                num_kps = len(kps) // 3
                points = []
                confidences = []
                for ki in range(num_kps):
                    kx = float(kps[ki * 3]) / w
                    ky = float(kps[ki * 3 + 1]) / h
                    kv = int(float(kps[ki * 3 + 2]))
                    # visibility 0 → not labeled, put (0,0) with confidence 0
                    if kv == 0:
                        points.append((0.0, 0.0))
                        confidences.append(0.0)
                    else:
                        points.append((kx, ky))
                        confidences.append(1.0 if kv == 2 else 0.5)

                keypoints.append(
                    fo.Keypoint(label=label, points=points, confidence=confidences)
                )

        sample["detections"] = fo.Detections(detections=detections)
        sample["keypoints"] = fo.Keypoints(keypoints=keypoints)
        samples.append(sample)

    dataset.add_samples(samples)

    print(f"images:       {stats['images']}")
    print(f"missing:      {stats['missing']}")
    print(f"annotations:  {stats['annotations']}")
    print(f"detections:   {len([d for s in samples for d in s['detections'].detections])}")
    print(f"keypoints:    {len([k for s in samples for k in s['keypoints'].keypoints])}")

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="使用 FiftyOne 可视化文件夹形式的 COCO 数据集。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", help="数据集根目录（包含 coco_files/ 和 image_groups/）")
    parser.add_argument("--dataset-name", default=None,
                        help="FiftyOne 数据集名称（默认: 目录名）")
    parser.add_argument("--keypoint-names", default=None,
                        help='关键点名称 JSON, 格式: {"类别名": ["kp0", ...]}')
    parser.add_argument("--skeleton-edges", default=None,
                        help='骨架连接 JSON, 格式: {"类别名": [[0,1],[1,2],...]}')
    parser.add_argument("--port", type=int, default=5151,
                        help="FiftyOne App 端口（默认: 5151）")
    parser.add_argument("--remote", action="store_true",
                        help="远程模式，不自动打开浏览器")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    dataset_name = args.dataset_name or input_dir.name
    kp_names_map = parse_json_arg(args.keypoint_names)
    skeleton_edges_map = parse_json_arg(args.skeleton_edges)

    dataset = load_dataset(input_dir, dataset_name, kp_names_map, skeleton_edges_map)

    print(f"\ndataset:      {dataset.name}")
    print(f"samples:      {len(dataset)}")
    print(f"app url:      http://localhost:{args.port}")
    print()

    session = fo.launch_app(dataset, port=args.port, remote=args.remote)
    session.wait()


if __name__ == "__main__":
    main()
