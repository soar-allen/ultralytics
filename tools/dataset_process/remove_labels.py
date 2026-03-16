#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CVAT XML 标注文件中删除指定的 label 及其所有标注对象。

清理目标：
  1. XML 中所有属于指定 label 的标注子元素（<polygon>/<box>/<polyline> 等）
  2. meta 中对应的 <label> 定义
  3. 删除标注后变为空标注的 <image> 条目及磁盘图片（可选，--remove_empty）

用法：
  # 仅报告（dry-run，不做修改）
  python tools/dataset_process/remove_labels.py \
    /path/to/dataset_root --labels person car

  # 执行删除
  python tools/dataset_process/remove_labels.py \
    /path/to/dataset_root --labels person car --fix

  # 同时清理删除后变空的图片
  python tools/dataset_process/remove_labels.py \
    /path/to/dataset_root --labels person car --fix --remove_empty

  # 移动变空的图片到指定目录（方便复查）
  python tools/dataset_process/remove_labels.py \
    /path/to/dataset_root --labels person car --fix --remove_empty \
    --move_dir /path/to/removed
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ANNOTATION_TAGS = {"polygon", "polyline", "box", "points", "cuboid", "skeleton", "tag"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _collect_images(root_dir: Path) -> list[Path]:
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _build_name_mapping(
    dataset_dir: Path, xml_names: list[str],
) -> tuple[dict[str, Path], Path]:
    """通过 basename 匹配建立 XML name → 磁盘路径的映射。"""
    images_dir = dataset_dir / "images"
    if not images_dir.is_dir():
        images_dir = dataset_dir

    disk_files = _collect_images(images_dir)

    basename_to_paths: dict[str, list[Path]] = defaultdict(list)
    for fp in disk_files:
        basename_to_paths[fp.name].append(fp)

    xml_to_disk: dict[str, Path] = {}
    for name in xml_names:
        basename = Path(name).name
        candidates = basename_to_paths.get(basename, [])
        if len(candidates) == 1:
            xml_to_disk[name] = candidates[0]
        elif len(candidates) > 1:
            for fp in candidates:
                try:
                    rel = str(fp.relative_to(dataset_dir)).replace("\\", "/")
                except ValueError:
                    continue
                if name == rel or name.endswith(rel) or rel.endswith(name):
                    xml_to_disk[name] = fp
                    break
            else:
                xml_to_disk[name] = candidates[0]

    return xml_to_disk, images_dir


def get_all_labels(root: ET.Element) -> set[str]:
    """从 XML 中收集所有实际使用的 label 名称。"""
    labels = set()
    for img in root.findall("image"):
        for child in img:
            if child.tag in ANNOTATION_TAGS:
                lbl = child.get("label", "")
                if lbl:
                    labels.add(lbl)
    return labels


def get_meta_labels(root: ET.Element) -> list[str]:
    """从 meta/task/labels 或 meta/job/labels 中读取 label 定义名称。"""
    labels = []
    meta = root.find("meta")
    if meta is None:
        return labels
    for tag in ("task", "job", "project"):
        parent = meta.find(tag)
        if parent is None:
            continue
        labels_elem = parent.find("labels")
        if labels_elem is None:
            continue
        for label_elem in labels_elem.findall("label"):
            name_elem = label_elem.find("name")
            if name_elem is not None and name_elem.text:
                labels.append(name_elem.text)
    return labels


def remove_label_annotations(
    root: ET.Element, labels_to_remove: set[str],
) -> dict[str, int]:
    """
    从所有 <image> 节点中移除属于指定 label 的标注子元素。

    Returns:
        每个被移除的 label 对应的标注数量。
    """
    removed_counts: dict[str, int] = defaultdict(int)
    for img in root.findall("image"):
        to_remove = []
        for child in img:
            if child.tag in ANNOTATION_TAGS:
                lbl = child.get("label", "")
                if lbl in labels_to_remove:
                    to_remove.append(child)
                    removed_counts[lbl] += 1
        for child in to_remove:
            img.remove(child)
    return dict(removed_counts)


def remove_meta_labels(root: ET.Element, labels_to_remove: set[str]) -> int:
    """从 meta 的 label 定义中移除指定 label，返回移除数量。"""
    removed = 0
    meta = root.find("meta")
    if meta is None:
        return removed
    for tag in ("task", "job", "project"):
        parent = meta.find(tag)
        if parent is None:
            continue
        labels_elem = parent.find("labels")
        if labels_elem is None:
            continue
        to_remove = []
        for label_elem in labels_elem.findall("label"):
            name_elem = label_elem.find("name")
            if name_elem is not None and name_elem.text in labels_to_remove:
                to_remove.append(label_elem)
        for elem in to_remove:
            labels_elem.remove(elem)
            removed += 1
    return removed


def find_empty_images(root: ET.Element) -> list[ET.Element]:
    """返回删除 label 后不含任何标注子元素的 <image> 节点。"""
    return [
        img for img in root.findall("image")
        if not any(child.tag in ANNOTATION_TAGS for child in img)
    ]


def reindex_and_update_meta(root: ET.Element, total_disk_count: int) -> int:
    """按 name 字母序重新索引 <image> id，物理重排 XML 树，更新 meta 中的统计信息。"""
    remaining = root.findall("image")
    remaining.sort(key=lambda e: e.get("name", ""))
    for new_id, img_elem in enumerate(remaining):
        img_elem.set("id", str(new_id))

    first_image_pos = None
    for i, elem in enumerate(root):
        if elem.tag == "image":
            first_image_pos = i
            break
    for img in root.findall("image"):
        root.remove(img)
    insert_pos = first_image_pos if first_image_pos is not None else len(list(root))
    for i, img_elem in enumerate(remaining):
        root.insert(insert_pos + i, img_elem)

    meta = root.find("meta")
    if meta is not None:
        for tag in ("task", "job", "project"):
            parent = meta.find(tag)
            if parent is None:
                continue
            size_elem = parent.find("size")
            if size_elem is not None:
                size_elem.text = str(total_disk_count)
            stop_elem = parent.find("stop_frame")
            if stop_elem is not None:
                stop_elem.text = str(max(0, total_disk_count - 1))
            for seg in parent.iter("segment"):
                seg_start = seg.find("start")
                if seg_start is not None:
                    seg_start.text = "0"
                seg_stop = seg.find("stop")
                if seg_stop is not None:
                    seg_stop.text = str(max(0, total_disk_count - 1))

    return len(remaining)


def handle_files(filepaths: list[Path], move_dir: Path | None) -> int:
    """删除或移动磁盘上的图片文件。"""
    processed = 0
    for filepath in filepaths:
        if not filepath.exists():
            continue
        if move_dir is not None:
            dst = move_dir / filepath.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            counter = 1
            base_dst = dst
            while dst.exists():
                dst = base_dst.parent / f"{base_dst.stem}_{counter}{base_dst.suffix}"
                counter += 1
            shutil.move(str(filepath), str(dst))
        else:
            filepath.unlink()
        processed += 1
    return processed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 CVAT XML 中删除指定的 label 及其所有标注对象",
    )
    ap.add_argument("dataset_dir", type=str,
                    help="数据集根目录（包含 annotations.xml 和 images/）")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="要删除的 label 名称列表")
    ap.add_argument("--fix", action="store_true",
                    help="执行删除（默认 dry-run 仅报告）")
    ap.add_argument("--remove_empty", action="store_true",
                    help="同时移除删除 label 后变为空标注的 <image> 及磁盘图片")
    ap.add_argument("--move_dir", type=str, default=None,
                    help="将变空的图片移动到此目录而非删除（需配合 --remove_empty）")
    ap.add_argument("--xml_name", type=str, default="annotations.xml",
                    help="XML 文件名（默认: annotations.xml）")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"[ERROR] 目录不存在: {dataset_dir}")
        return 1

    xml_path = dataset_dir / args.xml_name
    if not xml_path.exists():
        print(f"[ERROR] 找不到 XML: {xml_path}")
        return 1

    labels_to_remove = set(args.labels)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ---- 扫描 ----
    used_labels = get_all_labels(root)
    meta_labels = get_meta_labels(root)
    all_xml_images = root.findall("image")
    total_xml = len(all_xml_images)

    print("=" * 60)
    print("扫描数据集")
    print("=" * 60)
    print(f"  数据集目录:     {dataset_dir}")
    print(f"  XML 文件:       {xml_path.name}")
    print(f"  XML 条目总数:   {total_xml}")
    print(f"  meta 定义 label: {', '.join(meta_labels) if meta_labels else '(无)'}")
    print(f"  实际使用 label:  {', '.join(sorted(used_labels)) if used_labels else '(无)'}")
    print(f"  要删除的 label:  {', '.join(sorted(labels_to_remove))}")

    found = labels_to_remove & used_labels
    not_found = labels_to_remove - used_labels
    if not_found:
        print(f"\n  [WARN] 以下 label 在标注中未找到: {', '.join(sorted(not_found))}")
    if not found:
        print("\n  指定的 label 均不存在于标注中，无需操作")
        return 0

    # 统计将被移除的标注数量
    anno_counts: dict[str, int] = defaultdict(int)
    affected_images: set[str] = set()
    for img in root.findall("image"):
        img_name = img.get("name", "")
        for child in img:
            if child.tag in ANNOTATION_TAGS:
                lbl = child.get("label", "")
                if lbl in labels_to_remove:
                    anno_counts[lbl] += 1
                    affected_images.add(img_name)

    total_annos = sum(anno_counts.values())
    print(f"\n  将移除标注统计:")
    for lbl in sorted(anno_counts):
        print(f"    {lbl}: {anno_counts[lbl]} 个")
    print(f"    合计: {total_annos} 个标注，涉及 {len(affected_images)} 张图片")

    # 预判删除后变空的图片
    would_be_empty = []
    for img in root.findall("image"):
        remaining_annos = [
            child for child in img
            if child.tag in ANNOTATION_TAGS and child.get("label", "") not in labels_to_remove
        ]
        if not remaining_annos and any(
            child.tag in ANNOTATION_TAGS and child.get("label", "") in labels_to_remove
            for child in img
        ):
            would_be_empty.append(img.get("name", ""))

    if would_be_empty:
        print(f"\n  删除后将变为空标注的图片: {len(would_be_empty)} 张")
        shown = min(10, len(would_be_empty))
        for name in would_be_empty[:shown]:
            print(f"    - {name}")
        if len(would_be_empty) > shown:
            print(f"    ... 还有 {len(would_be_empty) - shown} 张")
        if not args.remove_empty:
            print("    提示: 添加 --remove_empty 可同时清理这些空标注图片")

    if not args.fix:
        print(f"\n[dry-run 模式] 未做任何修改")
        print(f"  提示: 添加 --fix 参数来执行删除")
        return 0

    # ---- 执行 ----
    print(f"\n{'=' * 60}")
    print("执行删除")
    print("=" * 60)

    backup_path = xml_path.with_suffix(xml_path.suffix + ".bak")
    shutil.copy2(xml_path, backup_path)
    print(f"  已备份 XML → {backup_path}")

    removed_counts = remove_label_annotations(root, labels_to_remove)
    total_removed = sum(removed_counts.values())
    for lbl in sorted(removed_counts):
        print(f"  已移除 label '{lbl}': {removed_counts[lbl]} 个标注")
    print(f"  共移除 {total_removed} 个标注")

    meta_removed = remove_meta_labels(root, labels_to_remove)
    if meta_removed:
        print(f"  已从 meta 中移除 {meta_removed} 个 label 定义")

    if args.remove_empty:
        empty_images = find_empty_images(root)
        if empty_images:
            all_xml_names = [img.get("name", "") for img in root.findall("image")]
            xml_to_disk, images_dir = _build_name_mapping(dataset_dir, all_xml_names)
            move_dir = Path(args.move_dir).expanduser().resolve() if args.move_dir else None
            action_label = f"移动到 {move_dir}" if move_dir else "删除"

            empty_filepaths = [
                xml_to_disk[elem.get("name", "")]
                for elem in empty_images
                if elem.get("name", "") in xml_to_disk
            ]
            processed = handle_files(empty_filepaths, move_dir)
            print(f"  已{action_label} {processed} 张变空的图片")

            for elem in empty_images:
                root.remove(elem)
            print(f"  已从 XML 移除 {len(empty_images)} 个空标注 <image> 条目")

            remaining_disk = _collect_images(images_dir)
            new_xml_count = reindex_and_update_meta(root, len(remaining_disk))
            print(f"\n  XML 剩余条目:   {new_xml_count}")
            print(f"  磁盘剩余图片:   {len(remaining_disk)}")

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    remaining_labels = get_all_labels(root)
    print(f"\n  剩余 label: {', '.join(sorted(remaining_labels)) if remaining_labels else '(无)'}")
    print(f"\n删除完成 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
