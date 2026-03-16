#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 CVAT 数据集中图片路径与 XML 标注不一致的问题。

典型问题场景：
  - 反复导入导出导致嵌套: images/images/images/xx.png
  - XML name 带有数据集名前缀: dataset-name/images/test/images/xx.png
  - 磁盘移动后 XML 没同步: 磁盘 images/xx.png vs XML test/images/xx.png

本脚本的处理逻辑：
  1. 扫描磁盘和 XML，通过 basename 匹配建立文件对应关系
  2. 检测两侧的路径不一致
  3. 可选将磁盘文件平铺到 images/ 一级目录
  4. 将 XML name 更新为实际磁盘相对路径，确保导入 CVAT 时路径匹配

用法：
  # 仅报告（dry-run，不做修改）
  python tools/dataset_process/flatten_image_paths.py \\
    /path/to/dataset_root

  # 执行修复（平铺磁盘文件 + 同步 XML 路径）
  python tools/dataset_process/flatten_image_paths.py \\
    /path/to/dataset_root --fix

  # 仅同步 XML 路径（不移动磁盘文件）
  python tools/dataset_process/flatten_image_paths.py \\
    /path/to/dataset_root --fix --xml_only
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _collect_images(root_dir: Path) -> list[Path]:
    return sorted(
        p for p in root_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def find_nested_files(images_dir: Path) -> dict[Path, Path]:
    """返回需要移动到 images_dir 一级目录的 {当前路径: 目标路径} 映射。"""
    moves: dict[Path, Path] = {}
    for f in images_dir.rglob("*"):
        if not f.is_file() or f.suffix.lower() not in IMG_EXTS:
            continue
        target = images_dir / f.name
        if f != target:
            moves[f] = target
    return moves


def fix_disk_files(moves: dict[Path, Path]) -> int:
    moved = 0
    for src, dst in moves.items():
        if not src.exists():
            continue
        if dst.exists():
            if src.stat().st_size == dst.stat().st_size:
                src.unlink()
                moved += 1
                continue
            print(f"  [SKIP] 目标已存在且大小不同: {dst.name}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def cleanup_empty_dirs(root_dir: Path) -> int:
    removed = 0
    for d in sorted(root_dir.rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Name mapping: XML name <-> disk path
# ---------------------------------------------------------------------------

def _build_basename_index(
    images_dir: Path,
) -> tuple[dict[str, Path], list[Path]]:
    """
    扫描磁盘图片，返回 (basename→disk_path 映射, 全部文件列表)。
    """
    disk_files = _collect_images(images_dir)
    index: dict[str, Path] = {}
    for fp in disk_files:
        index[fp.name] = fp
    return index, disk_files


def compute_xml_fixes(
    xml_path: Path,
    images_dir: Path,
    dataset_dir: Path,
) -> tuple[list[tuple[ET.Element, str, str]], set[str], set[str]]:
    """
    计算 XML 中需要修正的路径。

    Returns:
        ([(elem, old_name, new_name), ...],
         xml_only_basenames,
         disk_only_basenames)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    basename_to_disk, disk_files = _build_basename_index(images_dir)
    disk_basenames = set(basename_to_disk.keys())

    fixes: list[tuple[ET.Element, str, str]] = []
    xml_basenames: set[str] = set()

    for img_elem in root.findall("image"):
        old_name = img_elem.get("name", "")
        basename = Path(old_name).name
        xml_basenames.add(basename)

        if basename not in basename_to_disk:
            continue

        disk_path = basename_to_disk[basename]
        expected_name = str(disk_path.relative_to(dataset_dir)).replace("\\", "/")

        if old_name != expected_name:
            fixes.append((img_elem, old_name, expected_name))

    only_in_xml = xml_basenames - disk_basenames
    only_on_disk = disk_basenames - xml_basenames

    return fixes, only_in_xml, only_on_disk


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="修复 CVAT 数据集中图片路径与 XML 不一致的问题",
    )
    ap.add_argument("dataset_dir", type=str,
                    help="数据集根目录（包含 annotations.xml 和 images/）")
    ap.add_argument("--fix", action="store_true",
                    help="执行修复（默认 dry-run 仅报告）")
    ap.add_argument("--xml_name", type=str, default="annotations.xml",
                    help="XML 文件名（默认: annotations.xml）")
    ap.add_argument("--prefix", type=str, default="images",
                    help="图片子目录名（默认: images）")
    ap.add_argument("--xml_only", action="store_true",
                    help="仅同步 XML 路径，不移动磁盘文件")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"[ERROR] 目录不存在: {dataset_dir}")
        return 1

    xml_path = dataset_dir / args.xml_name
    images_dir = dataset_dir / args.prefix

    if not xml_path.exists():
        print(f"[ERROR] 找不到 XML: {xml_path}")
        return 1
    if not images_dir.is_dir():
        print(f"[ERROR] 找不到图片目录: {images_dir}")
        return 1

    dry_run = not args.fix

    # ---- 1. 扫描磁盘 ----
    print(f"{'=' * 60}")
    print("1. 扫描磁盘文件结构")
    print("=" * 60)

    nested_dirs = sorted(d for d in images_dir.rglob("*") if d.is_dir())
    image_files = _collect_images(images_dir)
    moves = find_nested_files(images_dir)

    nonempty_dirs = []
    empty_dirs = []
    for d in nested_dirs:
        count = sum(1 for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS)
        if count:
            nonempty_dirs.append((d, count))
        elif not any(d.iterdir()):
            empty_dirs.append(d)

    print(f"  图片目录:     {images_dir}")
    print(f"  总图片数:     {len(image_files)}")
    if nonempty_dirs:
        print(f"  含图片的子目录: {len(nonempty_dirs)} 个")
        for d, count in nonempty_dirs:
            rel = d.relative_to(images_dir)
            print(f"    {args.prefix}/{rel}/  ({count} 张)")
        print(f"  需要平铺:     {len(moves)} 个文件")
    elif empty_dirs:
        print(f"  空子目录:     {len(empty_dirs)} 个（可清理）")
    else:
        print(f"  磁盘结构已平铺 ✓")

    # ---- 2. 扫描 XML 路径一致性 ----
    print(f"\n{'=' * 60}")
    print("2. 检查 XML 路径一致性")
    print("=" * 60)

    fixes, only_in_xml, only_on_disk = compute_xml_fixes(
        xml_path, images_dir, dataset_dir
    )

    tree = ET.parse(xml_path)
    root = tree.getroot()
    total_xml = len(root.findall("image"))

    print(f"  XML 文件:     {xml_path.name}")
    print(f"  <image> 数量: {total_xml}")
    print(f"  路径不一致:   {len(fixes)} 个")

    if fixes:
        shown = min(5, len(fixes))
        for _, old, new in fixes[:shown]:
            print(f"    {old}")
            print(f"      → {new}")
        if len(fixes) > shown:
            print(f"    ... 还有 {len(fixes) - shown} 条")

    if only_in_xml:
        print(f"  [WARN] XML 有但磁盘无: {len(only_in_xml)} 张")
        for name in sorted(only_in_xml)[:5]:
            print(f"    - {name}")
        if len(only_in_xml) > 5:
            print(f"    ... 还有 {len(only_in_xml) - 5} 张")

    if only_on_disk:
        print(f"  [INFO] 磁盘有但 XML 无: {len(only_on_disk)} 张")
        for name in sorted(only_on_disk)[:5]:
            print(f"    - {name}")
        if len(only_on_disk) > 5:
            print(f"    ... 还有 {len(only_on_disk) - 5} 张")

    # ---- 3. 汇总 ----
    disk_actions = 0 if args.xml_only else len(moves)
    total_actions = disk_actions + len(fixes) + len(empty_dirs)

    print(f"\n{'=' * 60}")
    print("汇总")
    print("=" * 60)
    if not args.xml_only:
        print(f"  磁盘文件需平铺: {len(moves)}")
        if empty_dirs:
            print(f"  空目录需清理:    {len(empty_dirs)}")
    print(f"  XML 路径需修正:  {len(fixes)}")

    if total_actions == 0 and not only_in_xml:
        print("\n  数据集路径完全一致，无需修复 ✓")
        return 0

    if dry_run:
        print(f"\n[dry-run 模式] 未做任何修改")
        print(f"  提示: 添加 --fix 参数来执行修复")
        return 0

    # ---- 4. 执行修复 ----
    print(f"\n{'=' * 60}")
    print("执行修复")
    print("=" * 60)

    # 备份 XML
    backup_path = xml_path.with_suffix(xml_path.suffix + ".bak")
    shutil.copy2(xml_path, backup_path)
    print(f"  已备份 XML → {backup_path}")

    # 4a. 平铺磁盘文件 + 清理空目录
    if not args.xml_only:
        if moves:
            moved = fix_disk_files(moves)
            print(f"  已平铺 {moved} 个文件到 {args.prefix}/")
        cleaned = cleanup_empty_dirs(images_dir)
        if cleaned:
            print(f"  已清理 {cleaned} 个空目录")

    # 4b. 修正 XML 路径（磁盘可能已变化，重新计算）
    tree = ET.parse(xml_path)
    root = tree.getroot()

    basename_to_disk, _ = _build_basename_index(images_dir)
    fixed_count = 0
    skipped_names: list[str] = []
    for img_elem in root.findall("image"):
        old_name = img_elem.get("name", "")
        basename = Path(old_name).name
        if basename not in basename_to_disk:
            skipped_names.append(old_name)
            continue
        disk_path = basename_to_disk[basename]
        new_name = str(disk_path.relative_to(dataset_dir)).replace("\\", "/")
        if old_name != new_name:
            img_elem.set("name", new_name)
            fixed_count += 1

    if skipped_names:
        print(f"  [WARN] {len(skipped_names)} 个 <image> 未找到对应磁盘文件，已跳过:")
        for name in skipped_names[:10]:
            print(f"    - {name}")
        if len(skipped_names) > 10:
            print(f"    ... 还有 {len(skipped_names) - 10} 个")

    # 4c. 按新 name 重新排序并分配 id（CVAT 要求 id 与按 name 排序一致）
    all_images = root.findall("image")
    all_images.sort(key=lambda e: e.get("name", ""))
    for new_id, img_elem in enumerate(all_images):
        img_elem.set("id", str(new_id))

    first_image_pos = None
    for i, elem in enumerate(root):
        if elem.tag == "image":
            first_image_pos = i
            break
    for img in root.findall("image"):
        root.remove(img)
    insert_pos = first_image_pos if first_image_pos is not None else len(list(root))
    for i, img_elem in enumerate(all_images):
        root.insert(insert_pos + i, img_elem)

    total_images = len(all_images)
    meta = root.find("meta")
    if meta is not None:
        for tag in ("task", "job", "project"):
            parent = meta.find(tag)
            if parent is None:
                continue
            size_elem = parent.find("size")
            if size_elem is not None:
                size_elem.text = str(total_images)
            stop_elem = parent.find("stop_frame")
            if stop_elem is not None:
                stop_elem.text = str(max(0, total_images - 1))
            for seg in parent.iter("segment"):
                seg_start = seg.find("start")
                if seg_start is not None:
                    seg_start.text = "0"
                seg_stop = seg.find("stop")
                if seg_stop is not None:
                    seg_stop.text = str(max(0, total_images - 1))

    if fixed_count:
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"  已修正 XML 中 {fixed_count} 个路径")
        print(f"  已按 name 重新排序并分配 id（0~{total_images - 1}）")
    else:
        ET.indent(tree, space="  ")
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        print(f"  XML 路径已一致，已重新排序 id")

    # 4c. 验证
    post_fixes, post_xml_only, post_disk_only = compute_xml_fixes(
        xml_path, images_dir, dataset_dir
    )
    if post_fixes:
        print(f"\n  [WARN] 修复后仍有 {len(post_fixes)} 个路径不一致")
    else:
        print(f"\n  修复后验证: 磁盘与 XML 路径完全一致 ✓")

    print(f"\n修复完成 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
