#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CVAT 数据集中移除没有任何标注对象的图片，以及不在 XML 中的孤儿图片。

清理目标：
  1. XML 中无标注的 <image> 条目（节点内不含 <polygon>/<box> 等子元素）
     → 从 XML 中移除条目，同时删除/移动磁盘图片
  2. 磁盘上存在但 XML 中完全没有对应条目的"孤儿"图片
     → 删除/移动磁盘图片

用法：
  # 仅报告（dry-run，不做修改）
  python tools/dataset_process/remove_unannotated.py \\
    /path/to/dataset_root

  # 从 XML 中移除无标注条目 + 删除磁盘图片 + 清理孤儿图片
  python tools/dataset_process/remove_unannotated.py \\
    /path/to/dataset_root --fix

  # 移动图片到指定目录（方便复查）
  python tools/dataset_process/remove_unannotated.py \\
    /path/to/dataset_root --fix --move_dir /path/to/unannotated
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
    """
    通过 basename 匹配建立 XML name → 磁盘路径的映射。
    同时自动检测图片根目录（用于 _collect_images 扫描）。

    Returns:
        (xml_name→disk_path 映射, 图片扫描根目录)
    """
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


def find_unannotated(root: ET.Element) -> list[ET.Element]:
    """返回 XML 中所有没有标注子元素的 <image> 节点。"""
    return [
        img for img in root.findall("image")
        if not any(child.tag in ANNOTATION_TAGS for child in img)
    ]


def reindex_and_update_meta(root: ET.Element, total_disk_count: int) -> int:
    """
    按 name 字母序重新索引 <image> id，物理重排 XML 树中的元素顺序，
    更新 meta 中的统计信息。
    total_disk_count 为磁盘上剩余图片总数，确保 CVAT 导入时 frame 号正确。
    """
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


def handle_files(
    filepaths: list[Path],
    move_dir: Path | None,
) -> int:
    """删除或移动磁盘上的图片文件。move_dir 为 None 时直接删除。"""
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
        description="从 CVAT 数据集中移除没有标注的图片和孤儿图片",
    )
    ap.add_argument("dataset_dir", type=str,
                    help="数据集根目录（包含 annotations.xml 和 images/）")
    ap.add_argument("--fix", action="store_true",
                    help="执行清理（默认 dry-run 仅报告）")
    ap.add_argument("--move_dir", type=str, default=None,
                    help="将图片移动到此目录而非删除（方便复查）")
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

    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_xml_images = root.findall("image")
    total_xml = len(all_xml_images)

    all_xml_names = [img.get("name", "") for img in all_xml_images]
    xml_to_disk, images_dir = _build_name_mapping(dataset_dir, all_xml_names)
    disk_files = _collect_images(images_dir)

    # ---- 扫描 ----
    print(f"{'=' * 60}")
    print("扫描数据集")
    print("=" * 60)
    print(f"  数据集目录:   {dataset_dir}")
    print(f"  图片目录:     {images_dir}")
    print(f"  XML 文件:     {xml_path.name}")
    print(f"  磁盘图片总数: {len(disk_files)}")
    print(f"  XML 条目总数: {total_xml}")

    mapped = len(xml_to_disk)
    unmapped = total_xml - mapped
    if unmapped:
        print(f"  [WARN] {unmapped} 个 XML 条目未在磁盘上找到对应图片")

    # 1) XML 中无标注的条目
    unannotated = find_unannotated(root)
    unannotated_names = [elem.get("name", "") for elem in unannotated]

    # 2) 磁盘上不在 XML 中的孤儿图片
    xml_mapped_disk_paths = set(xml_to_disk.values())
    orphans = [fp for fp in disk_files if fp not in xml_mapped_disk_paths]

    annotated_count = total_xml - len(unannotated)

    print(f"\n  XML 有标注:         {annotated_count} 张")
    print(f"  XML 无标注:         {len(unannotated)} 张")
    print(f"  孤儿图片(不在XML):  {len(orphans)} 张")

    if unannotated:
        print(f"\n  无标注图片:")
        shown = min(15, len(unannotated_names))
        for name in unannotated_names[:shown]:
            print(f"    - {name}")
        if len(unannotated_names) > shown:
            print(f"    ... 还有 {len(unannotated_names) - shown} 张")

    if orphans:
        print(f"\n  孤儿图片:")
        shown = min(15, len(orphans))
        for fp in orphans[:shown]:
            try:
                rel = str(fp.relative_to(images_dir)).replace("\\", "/")
            except ValueError:
                rel = fp.name
            print(f"    - {rel}")
        if len(orphans) > shown:
            print(f"    ... 还有 {len(orphans) - shown} 张")

    if not unannotated and not orphans:
        print("\n  所有图片均有标注，无孤儿图片 ✓")
        return 0

    if not args.fix:
        print(f"\n[dry-run 模式] 未做任何修改")
        print(f"  提示: 添加 --fix 参数来执行清理")
        return 0

    # ---- 修复 ----
    print(f"\n{'=' * 60}")
    print("执行清理")
    print("=" * 60)

    move_dir = Path(args.move_dir).expanduser().resolve() if args.move_dir else None
    action_label = f"移动到 {move_dir}" if move_dir else "删除"

    backup_path = xml_path.with_suffix(xml_path.suffix + ".bak")
    shutil.copy2(xml_path, backup_path)
    print(f"  已备份 XML → {backup_path}")

    # 1) 删除/移动无标注图片文件
    if unannotated:
        unannotated_filepaths = [
            xml_to_disk[name] for name in unannotated_names if name in xml_to_disk
        ]
        processed = handle_files(unannotated_filepaths, move_dir)
        print(f"  已{action_label} {processed} 张无标注图片")

        for elem in unannotated:
            root.remove(elem)
        print(f"  已从 XML 移除 {len(unannotated)} 个无标注 <image> 条目")

    # 2) 删除/移动孤儿图片
    if orphans:
        processed = handle_files(orphans, move_dir)
        print(f"  已{action_label} {processed} 张孤儿图片")

    # 3) 清理磁盘文件已不存在的残留 XML 条目
    remaining_disk = _collect_images(images_dir)
    remaining_basenames = {fp.name for fp in remaining_disk}
    dangling = [
        img for img in root.findall("image")
        if Path(img.get("name", "")).name not in remaining_basenames
    ]
    if dangling:
        for elem in dangling:
            root.remove(elem)
        print(f"  已移除 {len(dangling)} 个磁盘文件不存在的 XML 条目")

    # 4) 重新索引并更新 meta
    new_xml_count = reindex_and_update_meta(root, len(remaining_disk))

    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    print(f"\n  XML 剩余标注:   {new_xml_count}")
    print(f"  磁盘剩余图片:   {len(remaining_disk)}")
    if new_xml_count != len(remaining_disk):
        print(f"  [WARN] XML 条目数与磁盘图片数不一致！")
    print(f"\n清理完成 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
