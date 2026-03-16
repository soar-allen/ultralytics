#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 FiftyOne 去除数据集中的重复和相似图片。

功能：
  1. 精确去重：检测文件哈希完全相同的重复图片
  2. 近似去重：基于深度学习图像嵌入，检测视觉上高度相似的图片
  3. 可选同步清理 CVAT XML 标注文件，根据磁盘实际剩余图片正确重建 frame id
  4. 支持二段式操作：先扫描保存结果，再决定是否执行删除

操作模式：
  - report: 仅报告，不做任何修改（默认）
  - move:   将重复图片移动到指定目录
  - delete: 直接删除重复图片

二段式操作：
  # 阶段1: 扫描 + 可视化（自动保存缓存，不删除任何文件）
  python tools/dataset_process/dedup_images.py \\
    --images_dir /path/to/images --visualize

  # 阶段2: 从缓存加载结果执行删除（无需重新计算嵌入）
  python tools/dataset_process/dedup_images.py \\
    --from_cache /path/to/.dedup_cache.json \\
    --action delete --cvat_xml annotations.xml

一次性操作（向后兼容）：
  # 仅检测报告
  python tools/dataset_process/dedup_images.py \\
    --images_dir /path/to/images

  # 检测并删除 + 同步清理 CVAT XML
  python tools/dataset_process/dedup_images.py \\
    --images_dir /path/to/images \\
    --action delete --cvat_xml annotations.xml

注意：
  --images_dir 必须指向 CVAT 导出时的图片根目录，使得图片相对路径与
  CVAT XML 中的 name 属性一致（例如 test/images/xxx.jpg）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CACHE_FILENAME = ".dedup_cache.json"


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def _collect_images(images_dir: Path) -> list[Path]:
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def _rel_path(fp: Path, base: Path) -> str:
    """Compute forward-slash relative path from base."""
    try:
        return str(fp.relative_to(base)).replace("\\", "/")
    except ValueError:
        return fp.name


# ---------------------------------------------------------------------------
# Union-Find for grouping duplicates
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            result[self.find(i)].append(i)
        return {k: v for k, v in result.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Exact duplicate detection (file hash)
# ---------------------------------------------------------------------------

def _compute_filehash(filepath: Path) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(filepaths: list[Path]) -> list[list[int]]:
    """通过 MD5 文件哈希检测完全重复的图片，返回重复组（索引列表）。"""
    print("  计算文件哈希...")
    hashes: list[str] = []
    for i, fp in enumerate(filepaths):
        hashes.append(_compute_filehash(fp))
        if (i + 1) % 100 == 0 or i == len(filepaths) - 1:
            print(f"    [{i + 1}/{len(filepaths)}]", end="\r")
    print()

    hash_counts = Counter(hashes)
    dup_hashes = {h for h, c in hash_counts.items() if c > 1}
    if not dup_hashes:
        return []

    groups_map: dict[str, list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        if h in dup_hashes:
            groups_map[h].append(i)

    return list(groups_map.values())


# ---------------------------------------------------------------------------
# Near-duplicate detection (embedding similarity)
# ---------------------------------------------------------------------------

def find_near_duplicates(
    filepaths: list[Path],
    *,
    thresh: float = 0.03,
    model_name: str = "dinov2-vitb14-reg-torch",
    skip_indices: set[int] | None = None,
) -> list[list[int]]:
    """
    基于图像嵌入的近似去重。

    Args:
        filepaths: 图片路径列表
        thresh: 余弦距离阈值（0~1，越小越严格，推荐 0.01~0.05）
        model_name: FiftyOne model zoo 中的嵌入模型名
        skip_indices: 已标记为精确重复的索引（避免重复计算）

    Returns:
        近似重复组的列表，每组为索引列表
    """
    import fiftyone as fo
    import fiftyone.zoo as foz

    print(f"  加载嵌入模型: {model_name}")
    model = foz.load_zoo_model(model_name)

    ds_name = "__dedup_near_temp__"
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)

    dataset = fo.Dataset(name=ds_name)
    samples = [fo.Sample(filepath=str(fp)) for fp in filepaths]
    dataset.add_samples(samples)

    print(f"  计算图像嵌入（{len(filepaths)} 张）...")
    embeddings = dataset.compute_embeddings(model)

    fo.delete_dataset(ds_name)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-10)
    sim_matrix = embeddings_norm @ embeddings_norm.T
    dist_matrix = 1.0 - sim_matrix

    n = len(filepaths)
    uf = UnionFind(n)
    skip = skip_indices or set()
    pair_count = 0

    for i in range(n):
        if i in skip:
            continue
        for j in range(i + 1, n):
            if j in skip:
                continue
            if dist_matrix[i, j] < thresh:
                uf.union(i, j)
                pair_count += 1

    print(f"  发现 {pair_count} 对近似重复（阈值 {thresh}）")
    return list(uf.groups().values())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_groups(
    filepaths: list[Path],
    groups: list[list[int]],
    label: str,
    images_dir: Path,
) -> set[int]:
    """打印去重报告，返回要移除的索引（每组保留第一个）。"""
    to_remove: set[int] = set()

    if not groups:
        print(f"\n  [{label}] 未发现重复图片 ✓")
        return to_remove

    total_dups = sum(len(g) - 1 for g in groups)
    print(f"\n  [{label}] 发现 {len(groups)} 组重复，共 {total_dups} 张冗余图片:")

    for gi, group in enumerate(groups):
        sorted_group = sorted(group, key=lambda i: str(filepaths[i]))
        keep_idx = sorted_group[0]
        remove_idxs = sorted_group[1:]
        to_remove.update(remove_idxs)

        keep_rel = _rel_path(filepaths[keep_idx], images_dir)
        print(f"    组 {gi + 1}: 保留  {keep_rel}")
        for ri in remove_idxs:
            rm_rel = _rel_path(filepaths[ri], images_dir)
            print(f"           移除  {rm_rel}")

    return to_remove


# ---------------------------------------------------------------------------
# Cache (for two-stage operation)
# ---------------------------------------------------------------------------

def save_cache(
    cache_path: Path,
    images_dir: Path,
    filepaths: list[Path],
    to_remove: set[int],
    mode: str,
    model: str,
    thresh: float,
) -> None:
    data = {
        "images_dir": str(images_dir),
        "filepaths": [str(fp) for fp in filepaths],
        "to_remove": sorted(to_remove),
        "mode": mode,
        "model": model,
        "thresh": thresh,
        "timestamp": datetime.now().isoformat(),
    }
    cache_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n已保存去重结果缓存: {cache_path}")
    print(f"  可使用以下命令直接执行删除（无需重新计算）:")
    print(f"    python {__file__} --from_cache {cache_path} --action delete")


def load_cache(cache_path: Path) -> tuple[Path, list[Path], set[int], dict]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    images_dir = Path(data["images_dir"])
    filepaths = [Path(fp) for fp in data["filepaths"]]
    to_remove = set(data["to_remove"])
    meta = {
        "mode": data.get("mode", "unknown"),
        "model": data.get("model", "unknown"),
        "thresh": data.get("thresh", 0),
        "timestamp": data.get("timestamp", "unknown"),
    }
    return images_dir, filepaths, to_remove, meta


# ---------------------------------------------------------------------------
# Actions (image deletion / moving)
# ---------------------------------------------------------------------------

def apply_action(
    filepaths: list[Path],
    to_remove: set[int],
    images_dir: Path,
    *,
    action: str,
    dup_dir: Path | None = None,
) -> set[str]:
    """
    执行删除/移动操作，返回被移除图片的相对路径集合（用于 XML 清理）。
    """
    removed_rel_paths: set[str] = set()

    if not to_remove:
        print("\n无需执行任何操作")
        return removed_rel_paths

    if action == "report":
        print(f"\n[report 模式] {len(to_remove)} 张重复图片已标记，未做实际修改")
        print("  提示: 使用 --action move 或 --action delete 来实际处理")
        return removed_rel_paths

    if action == "move":
        if dup_dir is None:
            raise ValueError("move 模式需要指定 --dup_dir")
        dup_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx in sorted(to_remove):
        fp = filepaths[idx]
        if not fp.exists():
            continue

        rel = _rel_path(fp, images_dir)

        if action == "move":
            dst = dup_dir / fp.name
            counter = 1
            while dst.exists():
                dst = dup_dir / f"{fp.stem}_{counter}{fp.suffix}"
                counter += 1
            shutil.move(str(fp), str(dst))
        elif action == "delete":
            fp.unlink()

        removed_rel_paths.add(rel)
        count += 1

    verb = "移动" if action == "move" else "删除"
    print(f"\n已{verb} {count} 张重复图片")
    return removed_rel_paths


# ---------------------------------------------------------------------------
# CVAT XML cleanup
# ---------------------------------------------------------------------------

def _detect_xml_root(
    images_dir: Path,
    sample_xml_names: list[str],
    disk_files: list[Path],
) -> Path:
    """
    自动检测 CVAT XML name 属性对应的磁盘根目录。
    尝试 images_dir 及其上级目录，找到使相对路径与 XML name 一致的根。
    """
    if not sample_xml_names or not disk_files:
        return images_dir

    basename_to_paths: dict[str, list[Path]] = defaultdict(list)
    for fp in disk_files:
        basename_to_paths[fp.name].append(fp)

    target_name = sample_xml_names[0]
    target_basename = Path(target_name).name

    candidates = basename_to_paths.get(target_basename, [])
    for fp in candidates:
        for ancestor in [images_dir] + list(images_dir.parents):
            try:
                rel = str(fp.relative_to(ancestor)).replace("\\", "/")
                if rel == target_name:
                    return ancestor
            except ValueError:
                continue
            if ancestor == images_dir.parent.parent:
                break

    return images_dir


def remove_images_from_cvat_xml(
    cvat_xml: Path,
    removed_rel_paths: set[str],
    images_dir: Path,
) -> int:
    """
    从 CVAT XML 中删除已移除图片对应的 <image> 节点，
    并根据磁盘上剩余的所有图片（含无标注的）重新计算 frame id。

    这确保了重新导入 CVAT 时标注与图片正确对应。
    """
    backup_path = cvat_xml.with_suffix(cvat_xml.suffix + ".bak")
    shutil.copy2(cvat_xml, backup_path)
    print(f"  已备份原始 XML → {backup_path}")

    tree = ET.parse(cvat_xml)
    root = tree.getroot()

    all_xml_images = root.findall("image")
    if not all_xml_images:
        print(f"  [CVAT XML] 未找到 <image> 节点")
        return 0

    sample_xml_names = [
        img.get("name", "") for img in all_xml_images[:5]
    ]

    remaining_disk_files = _collect_images(images_dir)
    xml_root = _detect_xml_root(images_dir, sample_xml_names, remaining_disk_files)

    if xml_root != images_dir:
        print(f"  检测到 XML 根目录: {xml_root}（与 --images_dir 不同）")

    removed_basenames = {Path(r).name for r in removed_rel_paths}
    removed_names_normalized: set[str] = set()
    for rp in removed_rel_paths:
        removed_names_normalized.add(rp.replace("\\", "/"))

    to_delete = []
    for img_elem in all_xml_images:
        name_attr = img_elem.get("name", "")
        name_norm = name_attr.replace("\\", "/")
        if name_norm in removed_names_normalized:
            to_delete.append(img_elem)
        elif Path(name_attr).name in removed_basenames:
            to_delete.append(img_elem)

    if not to_delete:
        print(f"  [CVAT XML] 未找到需要移除的 <image> 节点")
        backup_path.unlink(missing_ok=True)
        return 0

    for elem in to_delete:
        root.remove(elem)

    remaining_disk_rels = sorted(
        _rel_path(fp, xml_root) for fp in remaining_disk_files
    )
    name_to_frame: dict[str, int] = {
        name: idx for idx, name in enumerate(remaining_disk_rels)
    }

    remaining_xml_images = root.findall("image")
    unmatched = []
    for img_elem in remaining_xml_images:
        name_attr = img_elem.get("name", "").replace("\\", "/")
        if name_attr in name_to_frame:
            img_elem.set("id", str(name_to_frame[name_attr]))
        else:
            basename = Path(name_attr).name
            matched = False
            for disk_rel, frame_id in name_to_frame.items():
                if Path(disk_rel).name == basename:
                    img_elem.set("id", str(frame_id))
                    matched = True
                    break
            if not matched:
                unmatched.append(name_attr)

    if unmatched:
        print(f"  [WARN] {len(unmatched)} 个 XML 条目未在磁盘上找到对应图片:")
        for name in unmatched[:5]:
            print(f"    - {name}")
        if len(unmatched) > 5:
            print(f"    ... 及其他 {len(unmatched) - 5} 个")

    first_image_pos = None
    for i, elem in enumerate(root):
        if elem.tag == "image":
            first_image_pos = i
            break

    for img_elem in remaining_xml_images:
        root.remove(img_elem)
    remaining_xml_images.sort(key=lambda e: int(e.get("id", "0")))
    insert_pos = first_image_pos if first_image_pos is not None else len(list(root))
    for i, img_elem in enumerate(remaining_xml_images):
        root.insert(insert_pos + i, img_elem)

    total_disk_count = len(remaining_disk_files)
    meta = root.find("meta")
    if meta is not None:
        task = meta.find("task")
        if task is not None:
            size_elem = task.find("size")
            if size_elem is not None:
                size_elem.text = str(total_disk_count)
            stop_elem = task.find("stop_frame")
            if stop_elem is not None:
                stop_elem.text = str(max(0, total_disk_count - 1))
            for seg in task.iter("segment"):
                seg_start = seg.find("start")
                if seg_start is not None:
                    seg_start.text = "0"
                seg_stop = seg.find("stop")
                if seg_stop is not None:
                    seg_stop.text = str(max(0, total_disk_count - 1))

    ET.indent(tree, space="  ")
    tree.write(cvat_xml, encoding="utf-8", xml_declaration=True)

    print(
        f"  [CVAT XML] 已从 {cvat_xml.name} 中移除 {len(to_delete)} 个 <image> 节点\n"
        f"  磁盘总图片: {total_disk_count}，XML 剩余标注: {len(remaining_xml_images)}"
    )
    return len(to_delete)


# ---------------------------------------------------------------------------
# FiftyOne visualization
# ---------------------------------------------------------------------------

def visualize_results(
    filepaths: list[Path],
    to_remove: set[int],
) -> None:
    """启动 FiftyOne App 可视化去重结果。"""
    import fiftyone as fo

    ds_name = "dedup_results"
    if fo.dataset_exists(ds_name):
        fo.delete_dataset(ds_name)

    dataset = fo.Dataset(name=ds_name)
    samples = []
    for i, fp in enumerate(filepaths):
        if not fp.exists():
            continue
        s = fo.Sample(filepath=str(fp))
        if i in to_remove:
            s["status"] = "duplicate"
            s.tags.append("duplicate")
        else:
            s["status"] = "unique"
            s.tags.append("unique")
        samples.append(s)

    dataset.add_samples(samples)
    print(f"\n启动 FiftyOne App（{len(samples)} 张图片）...")
    print("  在 App 中可按 tags 筛选 duplicate / unique")
    session = fo.launch_app(dataset)
    input("按回车键关闭 FiftyOne App...")
    session.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="使用 FiftyOne 去除数据集中的重复和相似图片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
二段式操作示例：
  # 阶段1: 扫描 + 可视化（自动保存缓存）
  python %(prog)s --images_dir /path/to/images --visualize

  # 阶段2: 从缓存加载结果执行删除
  python %(prog)s --from_cache .dedup_cache.json --action delete --cvat_xml annotations.xml
""",
    )
    ap.add_argument("--images_dir", type=str, default=None,
                    help="图片目录（递归扫描）。使用 --from_cache 时可省略")
    ap.add_argument("--mode", type=str, default="both",
                    choices=["exact", "near", "both"],
                    help="去重模式: exact（精确哈希）, near（近似嵌入）, both（默认）")
    ap.add_argument("--thresh", type=float, default=0.03,
                    help="近似去重的余弦距离阈值（0~1，越小越严格，默认 0.03）")
    ap.add_argument("--model", type=str, default="clip-vit-base32-torch",  # clip-vit-base32-torch, dinov2-vitb14-reg-torch
                    help="近似去重的嵌入模型（FiftyOne model zoo）")
    ap.add_argument("--action", type=str, default="report",
                    choices=["report", "move", "delete"],
                    help="操作: report（仅报告）, move（移动）, delete（删除）")
    ap.add_argument("--dup_dir", type=str, default=None,
                    help="move 模式下重复图片的目标目录")
    ap.add_argument("--cvat_xml", type=str, default=None,
                    help="CVAT 标注 XML 文件路径（可选，指定后同步清理标注）")
    ap.add_argument("--visualize", action="store_true",
                    help="启动 FiftyOne App 可视化结果")
    ap.add_argument("--from_cache", type=str, default=None,
                    help="从缓存文件加载去重结果（跳过耗时的计算）")
    ap.add_argument("--cache_file", type=str, default=None,
                    help="缓存文件保存路径（默认: images_dir 的父目录下 .dedup_cache.json）")
    ap.add_argument("--yes", "-y", action="store_true",
                    help="跳过交互确认，直接执行操作")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Stage 2: Load from cache
    # ------------------------------------------------------------------
    if args.from_cache:
        cache_path = Path(args.from_cache).expanduser().resolve()
        if not cache_path.exists():
            print(f"[ERROR] 缓存文件不存在: {cache_path}")
            return 1

        print(f"从缓存加载去重结果: {cache_path}")
        images_dir, filepaths, all_to_remove, meta = load_cache(cache_path)

        if args.images_dir:
            images_dir = Path(args.images_dir).expanduser().resolve()

        print(f"  图片目录:   {images_dir}")
        print(f"  缓存时间:   {meta['timestamp']}")
        print(f"  去重模式:   {meta['mode']}（模型: {meta['model']}，阈值: {meta['thresh']}）")
        print(f"  总图片数:   {len(filepaths)}")
        print(f"  重复图片:   {len(all_to_remove)}")
        print(f"  去重后保留: {len(filepaths) - len(all_to_remove)}")

        missing = sum(1 for idx in all_to_remove if not filepaths[idx].exists())
        if missing:
            print(f"  [WARN] {missing} 张待删除图片已不存在（可能已被处理过）")

    # ------------------------------------------------------------------
    # Stage 1: Compute dedup
    # ------------------------------------------------------------------
    else:
        if not args.images_dir:
            print("[ERROR] 必须指定 --images_dir 或 --from_cache")
            return 1

        images_dir = Path(args.images_dir).expanduser().resolve()
        if not images_dir.exists():
            print(f"[ERROR] 图片目录不存在: {images_dir}")
            return 1

        filepaths = _collect_images(images_dir)
        print(f"扫描到 {len(filepaths)} 张图片: {images_dir}")

        if len(filepaths) < 2:
            print("图片数量不足，无需去重")
            return 0

        all_to_remove: set[int] = set()
        exact_dup_all_indices: set[int] = set()

        if args.mode in ("exact", "both"):
            print(f"\n{'=' * 60}")
            print("阶段 1: 精确去重（文件哈希）")
            print("=" * 60)
            exact_groups = find_exact_duplicates(filepaths)
            exact_remove = report_groups(filepaths, exact_groups, "精确去重", images_dir)
            all_to_remove.update(exact_remove)
            for g in exact_groups:
                exact_dup_all_indices.update(g)

        if args.mode in ("near", "both"):
            print(f"\n{'=' * 60}")
            print(f"阶段 2: 近似去重（嵌入相似度，阈值={args.thresh}）")
            print("=" * 60)
            near_groups = find_near_duplicates(
                filepaths,
                thresh=args.thresh,
                model_name=args.model,
                skip_indices=all_to_remove if args.mode == "both" else None,
            )
            filtered = []
            for g in near_groups:
                clean = [i for i in g if i not in all_to_remove]
                if len(clean) > 1:
                    filtered.append(clean)
            near_remove = report_groups(filepaths, filtered, "近似去重", images_dir)
            all_to_remove.update(near_remove)

        # --- Summary ---
        print(f"\n{'=' * 60}")
        print("汇总")
        print("=" * 60)
        print(f"  总图片数:   {len(filepaths)}")
        print(f"  重复图片:   {len(all_to_remove)}")
        print(f"  去重后保留: {len(filepaths) - len(all_to_remove)}")

        # --- Save cache ---
        if all_to_remove:
            cache_path = (
                Path(args.cache_file).expanduser().resolve()
                if args.cache_file
                else images_dir.parent / CACHE_FILENAME
            )
            save_cache(
                cache_path, images_dir, filepaths, all_to_remove,
                args.mode, args.model, args.thresh,
            )

    # ------------------------------------------------------------------
    # Visualize (before action, so user can review)
    # ------------------------------------------------------------------
    if args.visualize and all_to_remove:
        try:
            visualize_results(filepaths, all_to_remove)
        except Exception as e:
            print(f"[WARN] FiftyOne 可视化失败: {e}")

    # ------------------------------------------------------------------
    # Apply action
    # ------------------------------------------------------------------
    if args.action == "report":
        if all_to_remove and not args.from_cache:
            print("\n提示: 使用 --action delete 或 --action move 来实际处理")
        return 0

    if not all_to_remove:
        print("\n无需执行任何操作")
        return 0

    if not args.yes:
        verb = "删除" if args.action == "delete" else "移动"
        answer = input(f"\n确认{verb} {len(all_to_remove)} 张重复图片？[y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("已取消操作")
            return 0

    dup_dir = Path(args.dup_dir).expanduser().resolve() if args.dup_dir else None
    removed_rel_paths = apply_action(
        filepaths, all_to_remove, images_dir,
        action=args.action, dup_dir=dup_dir,
    )

    # --- CVAT XML cleanup ---
    cvat_xml = Path(args.cvat_xml).expanduser().resolve() if args.cvat_xml else None
    if cvat_xml is not None:
        if not cvat_xml.exists():
            print(f"[WARN] CVAT XML 文件不存在: {cvat_xml}")
        elif removed_rel_paths:
            print(f"\n{'=' * 60}")
            print("清理 CVAT XML 标注")
            print("=" * 60)
            remove_images_from_cvat_xml(cvat_xml, removed_rel_paths, images_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
