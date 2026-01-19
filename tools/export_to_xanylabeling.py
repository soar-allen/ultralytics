"""
导出预标注到 X-AnyLabeling 可导入格式

支持格式:
1. YOLO OBB 格式 (X-AnyLabeling 原生支持)
2. X-AnyLabeling JSON 格式

使用方法:
    python tools/export_to_xanylabeling.py \
        --images datasets/fork_entry/images/raw \
        --labels datasets/fork_entry/labels/pre_annotations/auto_accept \
        --output datasets/fork_entry/export/x_anylabeling

Author: Ultralytics Team
Date: 2026-01-18
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


def export_yolo_obb_format(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    class_names: list = None,
    copy_images: bool = True
) -> dict:
    """
    导出为 YOLO OBB 格式 (X-AnyLabeling 原生支持)
    
    目录结构:
    output_dir/
    ├── images/
    │   └── *.jpg
    ├── labels/
    │   └── *.txt
    └── classes.txt
    
    Args:
        image_dir: 原始图片目录
        label_dir: YOLO OBB 标注目录
        output_dir: 输出目录
        class_names: 类别名称列表
        copy_images: 是否复制图片
        
    Returns:
        dict: 导出统计信息
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    
    # 创建目录结构
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    class_names = class_names or ["fork_entry_surface"]
    
    stats = {"images": 0, "labels": 0, "skipped": 0}
    
    # 支持的图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        label_path = label_dir / f"{img_path.stem}.txt"
        
        # 复制/链接图片
        if copy_images:
            shutil.copy(img_path, output_dir / "images" / img_path.name)
        else:
            # 创建符号链接
            target = output_dir / "images" / img_path.name
            if not target.exists():
                target.symlink_to(img_path.absolute())
        stats["images"] += 1
        
        # 复制标注
        if label_path.exists():
            shutil.copy(label_path, output_dir / "labels" / label_path.name)
            stats["labels"] += 1
        else:
            stats["skipped"] += 1
    
    # 创建 classes.txt
    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # 创建说明文件
    readme_content = f"""# X-AnyLabeling 导入指南

## 目录结构
```
{output_dir.name}/
├── images/     # 图片文件
├── labels/     # YOLO OBB 格式标注
└── classes.txt # 类别定义
```

## 导入步骤

1. 打开 X-AnyLabeling
2. File → Open Dir → 选择 `images` 目录
3. 在弹出的对话框中:
   - 选择 "YOLO-OBB" 作为导入格式
   - 选择 `labels` 目录作为标注目录
4. 开始审核/编辑标注

## 快捷键 (OBB 编辑)

| 快捷键 | 功能 |
|--------|------|
| R | 切换到旋转框模式 |
| Ctrl+滚轮 | 调整旋转角度 |
| Q/E | 微调角度 (-1°/+1°) |
| Delete | 删除选中的标注 |
| Ctrl+S | 保存 |

## 类别定义

{chr(10).join(f'- {i}: {name}' for i, name in enumerate(class_names))}

## 统计

- 图片数量: {stats['images']}
- 已标注: {stats['labels']}
- 未标注: {stats['skipped']}
"""
    
    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print(f"✓ YOLO OBB 格式导出完成: {output_dir}")
    print(f"  图片: {stats['images']}")
    print(f"  标注: {stats['labels']}")
    print(f"\n📖 导入说明: {output_dir / 'README.md'}")
    
    return stats


def export_xanylabeling_json(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    class_names: list = None
) -> dict:
    """
    导出为 X-AnyLabeling 原生 JSON 格式
    
    每张图片对应一个 JSON 文件，包含标注信息
    
    Args:
        image_dir: 原始图片目录
        label_dir: YOLO OBB 标注目录
        output_dir: 输出目录
        class_names: 类别名称列表
        
    Returns:
        dict: 导出统计信息
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = class_names or ["fork_entry_surface"]
    
    stats = {"images": 0, "labels": 0, "annotations": 0}
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for img_path in image_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue
        
        # 读取图片尺寸
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 复制图片
        shutil.copy(img_path, output_dir / img_path.name)
        stats["images"] += 1
        
        # 解析 YOLO OBB 标注并转换为 JSON
        shapes = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                
                class_id = int(parts[0])
                coords = list(map(float, parts[1:9]))
                
                # 反归一化坐标
                points = []
                for i in range(0, 8, 2):
                    x = coords[i] * w
                    y = coords[i + 1] * h
                    points.append([x, y])
                
                shapes.append({
                    "label": class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                    "points": points,
                    "group_id": None,
                    "shape_type": "rotation",
                    "flags": {},
                    "description": ""
                })
                stats["annotations"] += 1
        
        # 保存 JSON
        json_data = {
            "version": "0.4.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        
        json_path = output_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        stats["labels"] += 1
    
    print(f"✓ X-AnyLabeling JSON 格式导出完成: {output_dir}")
    print(f"  图片: {stats['images']}")
    print(f"  标注文件: {stats['labels']}")
    print(f"  标注框数: {stats['annotations']}")
    
    return stats


def merge_annotation_dirs(
    output_dir: str,
    *input_dirs: str,
    class_names: list = None
) -> dict:
    """
    合并多个标注目录 (auto_accept + manual_review)
    
    Args:
        output_dir: 输出目录
        *input_dirs: 输入标注目录列表
        class_names: 类别名称列表
        
    Returns:
        dict: 合并统计信息
    """
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    
    class_names = class_names or ["fork_entry_surface"]
    
    stats = {"total": 0, "from_dirs": {}}
    
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            print(f"⚠ 目录不存在: {input_dir}")
            continue
        
        dir_count = 0
        
        for label_path in input_dir.glob("*.txt"):
            # 复制标注
            shutil.copy(label_path, output_dir / "labels" / label_path.name)
            dir_count += 1
            stats["total"] += 1
        
        stats["from_dirs"][str(input_dir)] = dir_count
        print(f"  从 {input_dir.name} 合并: {dir_count} 个标注")
    
    # 创建 classes.txt
    with open(output_dir / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"✓ 合并完成: {stats['total']} 个标注 → {output_dir}")
    
    return stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="导出预标注到 X-AnyLabeling 格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出为 YOLO OBB 格式 (推荐)
  python export_to_xanylabeling.py --images ./images --labels ./labels --output ./export
  
  # 导出为 X-AnyLabeling JSON 格式
  python export_to_xanylabeling.py --images ./images --labels ./labels --output ./export --format json
  
  # 合并 auto_accept 和 manual_review
  python export_to_xanylabeling.py --merge --inputs ./auto_accept ./manual_review --output ./merged
        """
    )
    
    parser.add_argument(
        "--images", "-i",
        type=str,
        help="原始图片目录"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        help="YOLO OBB 标注目录"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["yolo_obb", "json"],
        default="yolo_obb",
        help="导出格式 (默认: yolo_obb)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["fork_entry_surface"],
        help="类别名称列表"
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="不复制图片，创建符号链接"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="合并多个标注目录"
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        help="合并模式: 输入标注目录列表"
    )
    
    args = parser.parse_args()
    
    if args.merge:
        if not args.inputs:
            parser.error("合并模式需要 --inputs 参数")
        merge_annotation_dirs(
            args.output,
            *args.inputs,
            class_names=args.classes
        )
    else:
        if not args.images or not args.labels:
            parser.error("需要 --images 和 --labels 参数")
        
        if args.format == "yolo_obb":
            export_yolo_obb_format(
                args.images,
                args.labels,
                args.output,
                class_names=args.classes,
                copy_images=not args.no_copy
            )
        else:
            export_xanylabeling_json(
                args.images,
                args.labels,
                args.output,
                class_names=args.classes
            )


if __name__ == "__main__":
    main()
