"""
入叉面 OBB 半自动标注器

工作流程:
1. YOLO-OBB 模型给出入叉面粗框 (如有模型)
2. SAM3 用粗框作为 box prompt 分割托盘
3. 在托盘 mask 内定位入叉面区域
4. 拟合入叉面 OBB
5. 质检 → 通过/人工修/丢弃

使用方法:
    python tools/fork_entry_annotator.py \
        --images datasets/fork_entry/images/raw \
        --output datasets/fork_entry/labels/pre_annotations \
        --obb-model path/to/obb_model.pt  # 可选

Author: Ultralytics Team
Date: 2026-01-18
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForkEntryAnnotator:
    """入叉面 OBB 标注器
    
    核心流程: YOLO-OBB 预标 → SAM3 质检 → 几何优化 → OBB 输出
    
    Attributes:
        sam: SAM3 模型实例
        obb_model: 可选的 YOLO OBB 模型
        config: 标注规则配置
        thresholds: 置信度分桶阈值
    """
    
    def __init__(
        self,
        obb_model: Optional[str] = None,
        sam_model: str = "sam3.pt",
        config_path: Optional[str] = None,
        skip_sam3_qc: bool = False,
        bbox_padding: float = 0.5
    ):
        """初始化标注器
        
        Args:
            obb_model: 已训练的 OBB 模型路径 (可选)
            sam_model: SAM3 模型路径
            config_path: 标注规则配置文件路径
            skip_sam3_qc: 是否跳过 SAM3 质检，直接使用 OBB 预测结果
            bbox_padding: SAM3 输入 bbox 的扩展比例 (默认 0.5 = 50%)
        """
        # 延迟导入以避免循环依赖
        from ultralytics import SAM, YOLO
        
        self.skip_sam3_qc = skip_sam3_qc
        self.bbox_padding = bbox_padding
        
        if not skip_sam3_qc:
            logger.info(f"加载 SAM3 模型: {sam_model}")
            self.sam = SAM(sam_model)
        else:
            self.sam = None
            logger.warning("跳过 SAM3 质检，直接使用 OBB 预测结果")
        
        if obb_model:
            logger.info(f"加载 OBB 模型: {obb_model}")
            self.obb_model = YOLO(obb_model)
        else:
            self.obb_model = None
            logger.warning("未提供 OBB 模型，将使用交互式模式")
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 置信度分桶阈值 - 放宽默认值
        self.thresholds = {
            "auto_accept": self.config.get("confidence_buckets", {}).get("auto_accept", {}).get("threshold", 0.7),
            "manual_review": self.config.get("confidence_buckets", {}).get("manual_review", {}).get("threshold", 0.3),
            "reject": self.config.get("confidence_buckets", {}).get("reject", {}).get("threshold", 0.1),
        }
        
        # SAM3 质检阈值 - 放宽默认值
        sam3_config = self.config.get("quality_check", {}).get("sam3", {})
        self.qc_thresholds = {
            "min_coverage": sam3_config.get("min_coverage", 0.3),  # 从 0.5 降低到 0.3
            "max_overflow": sam3_config.get("max_overflow", 0.5),  # 从 0.3 提高到 0.5
            "fork_hole_weight": sam3_config.get("fork_hole_weight", 0.8),  # 从 0.5 提高到 0.8
        }
        
        logger.info(f"置信度阈值: auto_accept={self.thresholds['auto_accept']}, manual_review={self.thresholds['manual_review']}")
        logger.info(f"SAM3 bbox padding: {self.bbox_padding*100:.0f}%")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """加载标注规则配置"""
        if config_path is None:
            # 默认配置路径
            default_path = Path(__file__).parent / "annotation_rules.yaml"
            if default_path.exists():
                config_path = str(default_path)
            else:
                logger.warning("未找到配置文件，使用默认配置")
                return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"已加载配置: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return {}
    
    def process_image(self, image_path: str) -> dict:
        """处理单张图片，返回标注结果
        
        Args:
            image_path: 图片路径
            
        Returns:
            dict: 包含标注结果、状态和质检信息
        """
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图片: {image_path}")
            return {"image_path": image_path, "annotations": [], "status": "error"}
        
        h, w = img.shape[:2]
        
        result = {
            "image_path": image_path,
            "image_size": (w, h),
            "annotations": [],
            "status": "pending",
            "qc_details": []
        }
        
        # Step 1: YOLO-OBB 预标注 (如有模型)
        if self.obb_model:
            obb_preds = self._yolo_obb_predict(img)
            logger.info(f"YOLO-OBB 检测到 {len(obb_preds)} 个目标")
            
            for pred in obb_preds:
                if self.skip_sam3_qc:
                    # 跳过 SAM3 质检，直接使用 OBB 预测
                    obb = self._normalize_obb(pred["obb_points"], w, h)
                    result["annotations"].append({
                        "obb": obb,
                        "confidence": pred["confidence"],
                        "qc_score": 1.0,  # 跳过质检时默认满分
                        "class_id": 0
                    })
                    result["qc_details"].append({"valid": True, "score": 1.0, "skipped": True})
                else:
                    # Step 2: SAM3 质检
                    qc_result = self._sam3_quality_check(img, pred)
                    result["qc_details"].append(qc_result)
                    
                    # Step 3: 几何优化入叉面 OBB
                    if qc_result["valid"]:
                        refined_obb = self._refine_fork_entry_obb(
                            img, pred, qc_result["mask"]
                        )
                        result["annotations"].append({
                            "obb": refined_obb,
                            "confidence": pred["confidence"],
                            "qc_score": qc_result["score"],
                            "class_id": 0  # fork_entry_surface
                        })
                    else:
                        # 质检失败时也记录原因，方便调试
                        logger.debug(f"质检失败: {qc_result.get('reason', 'unknown')}")
        else:
            # 无 OBB 模型时，尝试使用 SAM3 自动分割
            sam_results = self.sam(img, verbose=False)
            if sam_results and sam_results[0].masks is not None:
                for mask_data in sam_results[0].masks.data:
                    mask = mask_data.cpu().numpy()
                    obb = self._extract_obb_from_mask(mask, w, h)
                    if obb is not None:
                        result["annotations"].append({
                            "obb": obb,
                            "confidence": 0.5,  # 默认置信度
                            "qc_score": 0.5,
                            "class_id": 0
                        })
        
        # 置信度分桶
        result["status"] = self._classify_result(result)
        return result
    
    def _yolo_obb_predict(self, img: np.ndarray) -> list:
        """YOLO-OBB 预测
        
        Args:
            img: BGR 图像数组
            
        Returns:
            list: 预测结果列表，每项包含 obb_points, confidence, class_id
        """
        results = self.obb_model(img, verbose=False)
        predictions = []
        
        for r in results:
            if r.obb is not None and len(r.obb) > 0:
                for i in range(len(r.obb)):
                    obb_xyxyxyxy = r.obb.xyxyxyxy[i].cpu().numpy()
                    predictions.append({
                        "obb_points": obb_xyxyxyxy,  # [4, 2] 四个角点
                        "confidence": float(r.obb.conf[i]),
                        "class_id": int(r.obb.cls[i])
                    })
        return predictions
    
    def _sam3_quality_check(self, img: np.ndarray, pred: dict) -> dict:
        """SAM3 质检: 用 OBB 作为 box prompt 分割
        
        Args:
            img: BGR 图像数组
            pred: YOLO-OBB 预测结果
            
        Returns:
            dict: 质检结果，包含 valid, mask, score
        """
        h, w = img.shape[:2]
        
        # 将 OBB 转换为轴对齐 bbox 作为 prompt
        obb_points = pred["obb_points"]
        x_min, y_min = obb_points.min(axis=0)
        x_max, y_max = obb_points.max(axis=0)
        
        # 扩展 bbox 以包含更多上下文 (使用配置的 padding)
        # 注意：入叉面框需要较大的扩展才能让 SAM3 分割出完整托盘
        padding = self.bbox_padding
        pad_x = (x_max - x_min) * padding
        pad_y = (y_max - y_min) * padding
        bbox = [[
            max(0, x_min - pad_x),
            max(0, y_min - pad_y),
            min(w, x_max + pad_x),
            min(h, y_max + pad_y)
        ]]
        
        logger.debug(f"SAM3 bbox: 原始=[{x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}], 扩展后=[{bbox[0][0]:.0f},{bbox[0][1]:.0f},{bbox[0][2]:.0f},{bbox[0][3]:.0f}]")
        
        # SAM3 分割
        try:
            sam_results = self.sam(img, bboxes=bbox, verbose=False)
            if sam_results and sam_results[0].masks is not None:
                mask = sam_results[0].masks.data[0].cpu().numpy()
            else:
                return {"valid": False, "mask": None, "score": 0.0, "reason": "SAM3 无输出"}
        except Exception as e:
            logger.error(f"SAM3 分割失败: {e}")
            return {"valid": False, "mask": None, "score": 0.0, "reason": str(e)}
        
        # 质检指标
        qc_result = {
            "valid": True,
            "mask": mask,
            "score": 1.0,
            "details": {}
        }
        
        # 检查1: 框内 mask 覆盖率
        # 注意：OBB 检测的是入叉面，SAM3 分割的可能是整个托盘
        # 所以我们检查的是 OBB 区域内有多少被 mask 覆盖
        obb_mask = self._create_obb_mask((h, w), obb_points)
        obb_area = np.sum(obb_mask)
        intersection = np.sum(mask.astype(bool) & obb_mask.astype(bool))
        coverage = intersection / (obb_area + 1e-6)
        qc_result["details"]["coverage"] = coverage
        qc_result["details"]["obb_area"] = int(obb_area)
        qc_result["details"]["intersection"] = int(intersection)
        
        logger.debug(f"覆盖率: {coverage:.2f} (intersection={intersection}, obb_area={obb_area})")
        
        if coverage < self.qc_thresholds["min_coverage"]:
            qc_result["valid"] = False
            qc_result["score"] = coverage
            qc_result["reason"] = f"覆盖率过低: {coverage:.2f} < {self.qc_thresholds['min_coverage']}"
            logger.warning(f"质检失败 - {qc_result['reason']}")
            return qc_result
        
        # 检查2: 是否包含非入叉面区域 (mask 远超出 OBB)
        overflow = np.sum(mask.astype(bool) & ~obb_mask.astype(bool)) / (np.sum(mask) + 1e-6)
        qc_result["details"]["overflow"] = overflow
        
        if overflow > self.qc_thresholds["max_overflow"]:
            qc_result["score"] *= (1 - overflow)
        
        # 检查3: 叉孔特征检测
        has_fork_holes = self._detect_fork_holes(img, mask)
        qc_result["details"]["has_fork_holes"] = has_fork_holes
        
        if not has_fork_holes:
            qc_result["score"] *= self.qc_thresholds["fork_hole_weight"]
        
        qc_result["score"] = min(max(qc_result["score"], 0), 1)
        return qc_result
    
    def _refine_fork_entry_obb(
        self, 
        img: np.ndarray, 
        pred: dict, 
        pallet_mask: np.ndarray
    ) -> list:
        """在托盘 mask 内定位入叉面并拟合 OBB
        
        核心逻辑:
        - 入叉面通常是 mask 中最靠近相机的边界 (图像下方)
        - 通过直线检测找水平/垂直边界
        - 或通过叉孔位置反推入叉面
        
        Args:
            img: BGR 图像数组
            pred: YOLO-OBB 预测结果
            pallet_mask: SAM3 分割的托盘 mask
            
        Returns:
            list: 归一化的 OBB 坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        """
        h, w = img.shape[:2]
        
        # 确保 mask 是正确的格式
        if pallet_mask is None:
            return self._normalize_obb(pred["obb_points"], w, h)
        
        mask_uint8 = (pallet_mask * 255).astype(np.uint8)
        
        # 方法1: 叉孔检测反推
        fork_holes = self._detect_fork_hole_regions(img, pallet_mask)
        if len(fork_holes) >= 2:
            logger.debug("使用叉孔检测定位入叉面")
            # 用叉孔外接区域推断入叉面
            all_points = np.vstack(fork_holes)
            
            # 扩展区域以包含整个入叉面
            rect = cv2.minAreaRect(all_points)
            center, size, angle = rect
            
            # 入叉面通常比叉孔区域大，按比例扩展
            size = (size[0] * 1.5, size[1] * 1.3)
            rect = (center, size, angle)
            
            box = cv2.boxPoints(rect)
            return self._normalize_obb(box, w, h)
        
        # 方法2: 边缘直线检测
        edges = cv2.Canny(mask_uint8, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                                minLineLength=50, maxLineGap=10)
        
        if lines is not None and len(lines) > 0:
            # 找近水平线 (入叉面上下边)
            horizontal_lines = self._filter_lines_by_angle(lines, angle_range=(-15, 15))
            
            if len(horizontal_lines) >= 2:
                logger.debug("使用直线检测定位入叉面")
                # 选择最长的两条水平线作为上下边界
                horizontal_lines = sorted(horizontal_lines, 
                                          key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2),
                                          reverse=True)[:2]
                
                # 用这两条线构建 OBB
                all_points = np.array(horizontal_lines).reshape(-1, 2)
                rect = cv2.minAreaRect(all_points)
                box = cv2.boxPoints(rect)
                return self._normalize_obb(box, w, h)
        
        # 方法3: 对 mask 下半部分拟合
        # 入叉面通常在托盘 mask 的下部 (更靠近相机)
        logger.debug("使用 mask 下半部分定位入叉面")
        mask_lower = pallet_mask.copy()
        
        # 找 mask 的垂直范围
        rows = np.any(pallet_mask, axis=1)
        if np.any(rows):
            row_indices = np.where(rows)[0]
            mask_top = row_indices[0]
            mask_bottom = row_indices[-1]
            mask_height = mask_bottom - mask_top
            
            # 只保留下半部分
            cutoff = mask_top + int(mask_height * 0.5)
            mask_lower[:cutoff, :] = 0
        
        mask_lower_uint8 = (mask_lower * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_lower_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            if cv2.contourArea(cnt) > 100:  # 最小面积阈值
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                return self._normalize_obb(box, w, h)
        
        # 最后回退: 使用原始 OBB 预测
        logger.debug("回退到原始 OBB 预测")
        return self._normalize_obb(pred["obb_points"], w, h)
    
    def _detect_fork_holes(self, img: np.ndarray, mask: np.ndarray) -> bool:
        """检测入叉面内的叉孔特征 (暗区域矩形孔洞)
        
        Args:
            img: BGR 图像数组
            mask: 分割 mask
            
        Returns:
            bool: 是否检测到叉孔特征
        """
        holes = self._detect_fork_hole_regions(img, mask)
        return len(holes) >= 2
    
    def _detect_fork_hole_regions(
        self, 
        img: np.ndarray, 
        mask: np.ndarray
    ) -> list:
        """检测叉孔区域，返回轮廓点集列表
        
        Args:
            img: BGR 图像数组
            mask: 分割 mask
            
        Returns:
            list: 叉孔轮廓列表
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_uint8 = mask.astype(np.uint8)
        masked_gray = gray * mask_uint8
        
        # 计算 mask 区域的平均亮度作为自适应阈值
        mask_pixels = gray[mask_uint8 > 0]
        if len(mask_pixels) == 0:
            return []
        
        threshold = max(50, np.mean(mask_pixels) * 0.5)
        
        # 阈值分割找暗区域
        _, dark_regions = cv2.threshold(masked_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        dark_regions = dark_regions * mask_uint8
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_OPEN, kernel)
        dark_regions = cv2.morphologyEx(dark_regions, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选近似矩形的暗区域 (叉孔特征)
        holes = []
        min_area = (img.shape[0] * img.shape[1]) * 0.001  # 最小面积：图像的 0.1%
        max_area = (img.shape[0] * img.shape[1]) * 0.1    # 最大面积：图像的 10%
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box_area = cv2.contourArea(box)
            
            # 近似矩形判断：轮廓面积 / 外接矩形面积 > 0.6
            if box_area > 0 and area / box_area > 0.6:
                # 检查宽高比，叉孔通常是扁矩形
                width, height = rect[1]
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    if aspect_ratio < 5:  # 宽高比不能太极端
                        holes.append(cnt)
        
        return holes
    
    def _filter_lines_by_angle(
        self, 
        lines: np.ndarray, 
        angle_range: tuple
    ) -> list:
        """按角度过滤直线
        
        Args:
            lines: Hough 直线检测结果
            angle_range: (min_angle, max_angle) 角度范围 (度)
            
        Returns:
            list: 过滤后的直线列表
        """
        filtered = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # 归一化角度到 [-90, 90)
            while angle >= 90:
                angle -= 180
            while angle < -90:
                angle += 180
            
            if angle_range[0] <= angle <= angle_range[1]:
                filtered.append([x1, y1, x2, y2])
        
        return filtered
    
    def _create_obb_mask(
        self, 
        shape: tuple, 
        obb_points: np.ndarray
    ) -> np.ndarray:
        """从 OBB 角点创建 mask
        
        Args:
            shape: (height, width)
            obb_points: [4, 2] 四个角点坐标
            
        Returns:
            np.ndarray: 二值 mask
        """
        mask = np.zeros(shape, dtype=np.uint8)
        pts = obb_points.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 1)
        return mask
    
    def _normalize_obb(
        self, 
        box: np.ndarray, 
        w: int, 
        h: int
    ) -> list:
        """归一化 OBB 坐标为 YOLO 格式
        
        Args:
            box: [4, 2] 四个角点坐标 (像素)
            w: 图像宽度
            h: 图像高度
            
        Returns:
            list: 归一化坐标 [x1,y1,x2,y2,x3,y3,x4,y4]
        """
        box = np.array(box).reshape(-1, 2)
        box[:, 0] = np.clip(box[:, 0] / w, 0, 1)
        box[:, 1] = np.clip(box[:, 1] / h, 0, 1)
        return box.flatten().tolist()
    
    def _extract_obb_from_mask(
        self, 
        mask: np.ndarray, 
        w: int, 
        h: int
    ) -> Optional[list]:
        """从 mask 提取 OBB
        
        Args:
            mask: 分割 mask
            w: 图像宽度
            h: 图像高度
            
        Returns:
            Optional[list]: 归一化的 OBB 坐标，或 None
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 100:
            return None
        
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        return self._normalize_obb(box, w, h)
    
    def _classify_result(self, result: dict) -> str:
        """根据置信度分桶
        
        Args:
            result: 处理结果
            
        Returns:
            str: 分类标签 (auto_accept/manual_review/reject)
        """
        if not result["annotations"]:
            return "reject"
        
        # 计算综合置信度
        scores = []
        for ann in result["annotations"]:
            combined = ann["confidence"] * ann["qc_score"]
            scores.append(combined)
        
        avg_score = np.mean(scores)
        
        if avg_score >= self.thresholds["auto_accept"]:
            return "auto_accept"
        elif avg_score >= self.thresholds["manual_review"]:
            return "manual_review"
        else:
            return "reject"
    
    def batch_process(
        self, 
        image_dir: str,
        output_dir: str,
        export_format: str = "yolo_obb"
    ) -> dict:
        """批量处理并按置信度分类输出
        
        Args:
            image_dir: 输入图片目录
            output_dir: 输出目录
            export_format: 输出格式 (yolo_obb)
            
        Returns:
            dict: 处理统计信息
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        
        # 创建分类目录
        for subdir in ["auto_accept", "manual_review", "reject_samples"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        stats = {
            "total": 0,
            "auto_accept": 0, 
            "manual_review": 0, 
            "reject": 0,
            "error": 0
        }
        
        # 支持的图片格式
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [f for f in image_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        logger.info(f"找到 {len(image_files)} 张图片待处理")
        
        for i, img_path in enumerate(image_files):
            logger.info(f"处理 [{i+1}/{len(image_files)}]: {img_path.name}")
            stats["total"] += 1
            
            try:
                result = self.process_image(str(img_path))
                status = result["status"]
                
                if status == "error":
                    stats["error"] += 1
                    continue
                
                stats[status] += 1
                
                # 保存标注
                if status == "auto_accept" or status == "manual_review":
                    self._save_annotation(result, output_dir / status, export_format)
                elif status == "reject":
                    # reject 抽样保存
                    reject_rate = self.config.get("confidence_buckets", {}).get(
                        "reject", {}).get("sample_rate", 0.1)
                    if np.random.random() < reject_rate:
                        self._save_annotation(result, output_dir / "reject_samples", export_format)
                        
            except Exception as e:
                logger.error(f"处理 {img_path.name} 失败: {e}")
                stats["error"] += 1
        
        # 打印统计
        logger.info("=" * 50)
        logger.info("处理完成统计:")
        logger.info(f"  总计: {stats['total']}")
        logger.info(f"  自动接受: {stats['auto_accept']} ({stats['auto_accept']/max(stats['total'],1)*100:.1f}%)")
        logger.info(f"  人工审核: {stats['manual_review']} ({stats['manual_review']/max(stats['total'],1)*100:.1f}%)")
        logger.info(f"  拒绝: {stats['reject']} ({stats['reject']/max(stats['total'],1)*100:.1f}%)")
        logger.info(f"  错误: {stats['error']}")
        logger.info("=" * 50)
        
        return stats
    
    def _save_annotation(
        self, 
        result: dict, 
        output_dir: Path, 
        format: str
    ):
        """保存标注文件
        
        Args:
            result: 处理结果
            output_dir: 输出目录
            format: 输出格式
        """
        img_name = Path(result["image_path"]).stem
        
        if format == "yolo_obb":
            # YOLO OBB 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            label_path = output_dir / f"{img_name}.txt"
            
            with open(label_path, "w", encoding="utf-8") as f:
                for ann in result["annotations"]:
                    obb = ann["obb"]
                    class_id = ann.get("class_id", 0)
                    
                    # 格式化坐标，保留6位小数
                    coords = " ".join(f"{v:.6f}" for v in obb)
                    f.write(f"{class_id} {coords}\n")
            
            logger.debug(f"保存标注: {label_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="入叉面 OBB 半自动标注器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用已有 OBB 模型进行批量预标注
  python fork_entry_annotator.py --images ./images --output ./labels --obb-model best.pt
  
  # 跳过 SAM3 质检，直接使用 OBB 预测结果 (推荐先试这个)
  python fork_entry_annotator.py --images ./images --output ./labels --obb-model best.pt --skip-sam3-qc
  
  # 调整 SAM3 bbox 扩展比例 (默认 50%)
  python fork_entry_annotator.py --images ./images --output ./labels --obb-model best.pt --bbox-padding 0.8
  
  # 仅使用 SAM3 进行标注 (无 OBB 模型)
  python fork_entry_annotator.py --images ./images --output ./labels
        """
    )
    
    parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="输入图片目录"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="输出标注目录"
    )
    parser.add_argument(
        "--obb-model",
        type=str,
        default=None,
        help="YOLO OBB 模型路径 (可选)"
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="sam3.pt",
        help="SAM3 模型路径 (默认: sam3.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="标注规则配置文件路径"
    )
    parser.add_argument(
        "--skip-sam3-qc",
        action="store_true",
        help="跳过 SAM3 质检，直接使用 OBB 预测结果 (高 reject 率时推荐使用)"
    )
    parser.add_argument(
        "--bbox-padding",
        type=float,
        default=0.5,
        help="SAM3 输入 bbox 的扩展比例 (默认: 0.5 = 50%%)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建标注器实例
    annotator = ForkEntryAnnotator(
        obb_model=args.obb_model,
        sam_model=args.sam_model,
        config_path=args.config,
        skip_sam3_qc=args.skip_sam3_qc,
        bbox_padding=args.bbox_padding
    )
    
    # 批量处理
    stats = annotator.batch_process(
        image_dir=args.images,
        output_dir=args.output
    )
    
    print(f"\n✓ 处理完成！标注已保存到: {args.output}")
    print(f"  - auto_accept/: {stats['auto_accept']} 个 (可直接使用)")
    print(f"  - manual_review/: {stats['manual_review']} 个 (需人工审核)")
    print(f"  - reject_samples/: 抽样保存")
    
    # 如果 reject 率过高，给出建议
    if stats['total'] > 0:
        reject_rate = stats['reject'] / stats['total']
        if reject_rate > 0.5:
            print(f"\n⚠ 警告: reject 率较高 ({reject_rate*100:.1f}%)")
            print("  建议尝试:")
            print("    1. 使用 --skip-sam3-qc 跳过 SAM3 质检")
            print("    2. 使用 --bbox-padding 0.8 增大 bbox 扩展")
            print("    3. 使用 -v 查看详细日志定位问题")


if __name__ == "__main__":
    main()

