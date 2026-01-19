"""
SAM3 质检模块

用于验证 OBB 标注质量的独立模块，可配合 fork_entry_annotator 使用

功能:
1. 检查 OBB 框内 mask 覆盖率
2. 检查 mask 是否超出 OBB 范围
3. 检测叉孔特征
4. 综合评分

Author: Ultralytics Team
Date: 2026-01-18
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class QualityCheckResult:
    """质检结果数据类"""
    valid: bool
    score: float
    coverage: float        # 框内 mask 覆盖率
    overflow: float        # mask 超出框的比例
    has_fork_holes: bool   # 是否检测到叉孔
    mask: Optional[np.ndarray] = None
    reason: str = ""
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "valid": self.valid,
            "score": self.score,
            "coverage": self.coverage,
            "overflow": self.overflow,
            "has_fork_holes": self.has_fork_holes,
            "reason": self.reason
        }


class QualityChecker:
    """OBB 标注质检器
    
    使用 SAM3 分割结果验证 OBB 标注质量
    """
    
    def __init__(
        self,
        sam_model: str = "sam3.pt",
        min_coverage: float = 0.5,
        max_overflow: float = 0.3,
        fork_hole_weight: float = 0.5
    ):
        """初始化质检器
        
        Args:
            sam_model: SAM3 模型路径
            min_coverage: 最小覆盖率阈值
            max_overflow: 最大溢出比例
            fork_hole_weight: 叉孔特征权重
        """
        from ultralytics import SAM
        
        self.sam = SAM(sam_model)
        self.min_coverage = min_coverage
        self.max_overflow = max_overflow
        self.fork_hole_weight = fork_hole_weight
    
    def check_obb(
        self, 
        img: np.ndarray, 
        obb_points: np.ndarray,
        return_mask: bool = False
    ) -> QualityCheckResult:
        """检查单个 OBB 标注质量
        
        Args:
            img: BGR 图像
            obb_points: OBB 四个角点 [4, 2]
            return_mask: 是否返回 mask
            
        Returns:
            QualityCheckResult: 质检结果
        """
        h, w = img.shape[:2]
        
        # 将 OBB 转换为 bbox 作为 SAM prompt
        x_min, y_min = obb_points.min(axis=0)
        x_max, y_max = obb_points.max(axis=0)
        
        # 扩展 bbox
        padding = 0.1
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        bbox = [[
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            min(w, x_max + pad_x),
            min(h, y_max + pad_y)
        ]]
        
        # SAM3 分割
        try:
            sam_results = self.sam(img, bboxes=bbox, verbose=False)
            if sam_results and sam_results[0].masks is not None:
                mask = sam_results[0].masks.data[0].cpu().numpy()
            else:
                return QualityCheckResult(
                    valid=False, score=0.0, coverage=0.0, overflow=0.0,
                    has_fork_holes=False, reason="SAM3 无输出"
                )
        except Exception as e:
            return QualityCheckResult(
                valid=False, score=0.0, coverage=0.0, overflow=0.0,
                has_fork_holes=False, reason=f"SAM3 错误: {e}"
            )
        
        # 计算指标
        obb_mask = self._create_obb_mask((h, w), obb_points)
        
        # 覆盖率
        intersection = np.sum(mask.astype(bool) & obb_mask.astype(bool))
        coverage = intersection / (np.sum(obb_mask) + 1e-6)
        
        # 溢出率
        overflow = np.sum(mask.astype(bool) & ~obb_mask.astype(bool)) / (np.sum(mask) + 1e-6)
        
        # 叉孔检测
        has_fork_holes = self._detect_fork_holes(img, mask)
        
        # 综合评分
        score = 1.0
        valid = True
        reason = ""
        
        if coverage < self.min_coverage:
            valid = False
            score = coverage
            reason = f"覆盖率过低: {coverage:.2f}"
        else:
            if overflow > self.max_overflow:
                score *= (1 - overflow)
            if not has_fork_holes:
                score *= self.fork_hole_weight
        
        score = max(0, min(1, score))
        
        return QualityCheckResult(
            valid=valid,
            score=score,
            coverage=coverage,
            overflow=overflow,
            has_fork_holes=has_fork_holes,
            mask=mask if return_mask else None,
            reason=reason
        )
    
    def batch_check(
        self,
        image_dir: str,
        label_dir: str,
        output_report: str = None
    ) -> dict:
        """批量质检
        
        Args:
            image_dir: 图片目录
            label_dir: 标注目录
            output_report: 输出报告路径 (可选)
            
        Returns:
            dict: 质检统计
        """
        from pathlib import Path
        import json
        
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        
        stats = {
            "total_images": 0,
            "total_annotations": 0,
            "valid": 0,
            "invalid": 0,
            "with_fork_holes": 0,
            "avg_coverage": 0.0,
            "avg_score": 0.0,
            "details": []
        }
        
        coverages = []
        scores = []
        
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        for img_path in image_dir.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            h, w = img.shape[:2]
            stats["total_images"] += 1
            
            # 读取标注
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue
                    
                    coords = list(map(float, parts[1:9]))
                    # 反归一化
                    obb_points = np.array([
                        [coords[0] * w, coords[1] * h],
                        [coords[2] * w, coords[3] * h],
                        [coords[4] * w, coords[5] * h],
                        [coords[6] * w, coords[7] * h]
                    ])
                    
                    result = self.check_obb(img, obb_points)
                    stats["total_annotations"] += 1
                    
                    if result.valid:
                        stats["valid"] += 1
                    else:
                        stats["invalid"] += 1
                    
                    if result.has_fork_holes:
                        stats["with_fork_holes"] += 1
                    
                    coverages.append(result.coverage)
                    scores.append(result.score)
                    
                    stats["details"].append({
                        "image": img_path.name,
                        **result.to_dict()
                    })
        
        if coverages:
            stats["avg_coverage"] = float(np.mean(coverages))
            stats["avg_score"] = float(np.mean(scores))
        
        # 输出报告
        if output_report:
            with open(output_report, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✓ 质检报告已保存: {output_report}")
        
        return stats
    
    def _create_obb_mask(
        self, 
        shape: tuple, 
        obb_points: np.ndarray
    ) -> np.ndarray:
        """从 OBB 角点创建 mask"""
        mask = np.zeros(shape, dtype=np.uint8)
        pts = obb_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask
    
    def _detect_fork_holes(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> bool:
        """检测叉孔特征"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_uint8 = mask.astype(np.uint8)
        masked_gray = gray * mask_uint8
        
        mask_pixels = gray[mask_uint8 > 0]
        if len(mask_pixels) == 0:
            return False
        
        threshold = max(50, np.mean(mask_pixels) * 0.5)
        _, dark_regions = cv2.threshold(masked_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        dark_regions = dark_regions * mask_uint8
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_holes = 0
        min_area = (img.shape[0] * img.shape[1]) * 0.001
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            if box_area > 0 and area / box_area > 0.6:
                rectangular_holes += 1
        
        return rectangular_holes >= 2


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OBB 标注质检工具")
    parser.add_argument("--images", "-i", required=True, help="图片目录")
    parser.add_argument("--labels", "-l", required=True, help="标注目录")
    parser.add_argument("--output", "-o", help="输出报告路径")
    parser.add_argument("--sam-model", default="sam3.pt", help="SAM3 模型")
    
    args = parser.parse_args()
    
    checker = QualityChecker(sam_model=args.sam_model)
    stats = checker.batch_check(
        args.images, 
        args.labels,
        output_report=args.output
    )
    
    print("\n质检统计:")
    print(f"  总图片: {stats['total_images']}")
    print(f"  总标注: {stats['total_annotations']}")
    print(f"  有效: {stats['valid']} ({stats['valid']/max(stats['total_annotations'],1)*100:.1f}%)")
    print(f"  无效: {stats['invalid']}")
    print(f"  含叉孔: {stats['with_fork_holes']}")
    print(f"  平均覆盖率: {stats['avg_coverage']:.2f}")
    print(f"  平均质量分: {stats['avg_score']:.2f}")


if __name__ == "__main__":
    main()
