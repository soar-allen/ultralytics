#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描图片目录并生成可复现的索引文件。

示例：
    python tools/pallet_labeling/index_images.py \
      --images /abs/path/to/images \
      --out /abs/path/to/output/images_index.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageInfo:
    name: str  # relative to images root
    path: str  # absolute
    width: int
    height: int


def iter_image_files(images_dir: Path, recursive: bool) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images must be a directory: {images_dir}")

    it = images_dir.rglob("*") if recursive else images_dir.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files, key=lambda p: str(p).lower())


def build_index(images_dir: Path, recursive: bool) -> list[ImageInfo]:
    files = iter_image_files(images_dir, recursive=recursive)
    index: list[ImageInfo] = []
    for p in files:
        with Image.open(p) as im:
            w, h = im.size
        rel = p.relative_to(images_dir).as_posix()
        index.append(ImageInfo(name=rel, path=str(p.resolve()), width=int(w), height=int(h)))
    return index


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="图片目录（绝对路径或相对路径）")
    ap.add_argument("--out", type=str, required=True, help="输出 index JSON 文件路径")
    ap.add_argument("--recursive", action="store_true", help="是否递归扫描子目录")
    args = ap.parse_args()

    images_dir = Path(args.images).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    index = build_index(images_dir, recursive=bool(args.recursive))
    payload = {
        "images_dir": str(images_dir),
        "count": len(index),
        "images": [asdict(x) for x in index],
    }
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")
    tmp.replace(out_path)

    print(f"Wrote {len(index)} images -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

