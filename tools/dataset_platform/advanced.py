"""
高级数据管理与 AI 辅助功能。

- 难例挖掘：比较 ground_truth 与 predictions，找出高错误率样本
- 数据集版本控制：利用 FiftyOne clone() 进行快照备份
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import fiftyone as fo
import fiftyone.brain as fob

from .data_manager import clone_dataset, load_dataset

logger = logging.getLogger(__name__)


# ===================================================================
# 难例挖掘
# ===================================================================

def evaluate_detections(
    ds: fo.Dataset,
    pred_field: str = "predictions",
    gt_field: str = "ground_truth",
    eval_key: str = "eval",
    iou_threshold: float = 0.5,
) -> fo.Dataset:
    """
    评估模型预测，计算 TP/FP/FN 指标。
    返回评估后的数据集（每个 sample 上会新增 eval 相关字段）。
    """
    results = ds.evaluate_detections(
        pred_field,
        gt_field=gt_field,
        eval_key=eval_key,
        iou=iou_threshold,
        compute_mAP=True,
    )
    logger.info("评估完成: mAP=%.4f", results.mAP())
    return ds


def find_hard_samples(
    ds: fo.Dataset,
    eval_key: str = "eval",
    pred_field: str = "predictions",
    min_false_positives: int = 1,
    min_false_negatives: int = 1,
    sort_by: str = "total_errors",
    limit: int = 100,
) -> fo.DatasetView:
    """
    挖掘难例：找出模型预测错误率高的图像。

    策略：
      1. 统计每个样本的 FP（误检）和 FN（漏检）数量
      2. 按错误总数降序排列
    """
    from fiftyone import ViewField as F

    fp_field = f"{eval_key}_fp"
    fn_field = f"{eval_key}_fn"

    if fp_field not in ds.get_field_schema():
        ds.compute_metadata()

        for sample in ds.iter_samples(progress=True, autosave=True):
            preds = sample[pred_field]
            fp_count = 0
            fn_count = 0

            if preds and preds.detections:
                for det in preds.detections:
                    ev = det.get_attribute_value(eval_key, None)
                    if ev == "fp":
                        fp_count += 1

            gt = sample.get_field("ground_truth")
            if gt and hasattr(gt, "detections") and gt.detections:
                for det in gt.detections:
                    ev = det.get_attribute_value(eval_key, None)
                    if ev == "fn":
                        fn_count += 1

            sample[fp_field] = fp_count
            sample[fn_field] = fn_count
            sample[f"{eval_key}_total_errors"] = fp_count + fn_count

    total_field = f"{eval_key}_total_errors"
    view = ds.match(F(total_field) > 0)
    view = view.sort_by(total_field, reverse=True)
    if limit:
        view = view.limit(limit)

    logger.info("找到 %d 个难例样本", len(view))
    return view


def find_misclassified(
    ds: fo.Dataset,
    pred_field: str = "predictions",
    gt_field: str = "ground_truth",
) -> fo.DatasetView:
    """找出存在类别不匹配的样本（pred 和 gt 的 label 不一致）。"""
    from fiftyone import ViewField as F

    pred_classes_expr = F(f"{pred_field}.detections.label").unique()
    gt_classes_expr = F(f"{gt_field}.detections.label").unique()

    return ds.match(pred_classes_expr != gt_classes_expr)


# ===================================================================
# 数据集版本控制 / 快照
# ===================================================================

def create_snapshot(
    ds_name: str,
    snapshot_suffix: Optional[str] = None,
    note: str = "",
) -> str:
    """
    对数据集创建快照（clone），用于版本控制。

    Args:
        ds_name: 源数据集名称
        snapshot_suffix: 快照后缀（默认用时间戳）
        note: 备注说明

    Returns:
        快照数据集名称
    """
    if snapshot_suffix is None:
        snapshot_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    snapshot_name = f"{ds_name}__snapshot__{snapshot_suffix}"
    cloned = clone_dataset(ds_name, snapshot_name)

    cloned.info["snapshot_source"] = ds_name
    cloned.info["snapshot_time"] = datetime.now().isoformat()
    cloned.info["snapshot_note"] = note
    cloned.save()

    logger.info("创建快照: %s -> %s (note: %s)", ds_name, snapshot_name, note)
    return snapshot_name


def list_snapshots(ds_name: str) -> list[dict]:
    """列出指定数据集的所有快照。"""
    prefix = f"{ds_name}__snapshot__"
    snapshots = []
    for name in fo.list_datasets():
        if name.startswith(prefix):
            ds = fo.load_dataset(name)
            info = ds.info or {}
            snapshots.append({
                "name": name,
                "suffix": name[len(prefix):],
                "time": info.get("snapshot_time", "unknown"),
                "note": info.get("snapshot_note", ""),
                "num_samples": len(ds),
            })
    return sorted(snapshots, key=lambda x: x["time"], reverse=True)


def restore_snapshot(snapshot_name: str, target_name: Optional[str] = None) -> str:
    """
    从快照恢复数据集。

    如果 target_name 与原数据集同名，会先删除原数据集再恢复。
    """
    snap_ds = load_dataset(snapshot_name)
    source_name = snap_ds.info.get("snapshot_source", "")

    if target_name is None:
        target_name = source_name

    if not target_name:
        raise ValueError("无法确定恢复目标名称")

    if fo.dataset_exists(target_name):
        fo.delete_dataset(target_name)
        logger.info("删除已有数据集: %s", target_name)

    restored = snap_ds.clone(name=target_name)
    restored.persistent = True
    restored.info["restored_from"] = snapshot_name
    restored.info["restored_time"] = datetime.now().isoformat()
    restored.save()

    logger.info("从快照恢复: %s -> %s", snapshot_name, target_name)
    return target_name


def delete_snapshot(snapshot_name: str) -> None:
    """删除快照数据集。"""
    if fo.dataset_exists(snapshot_name):
        fo.delete_dataset(snapshot_name)
        logger.info("删除快照: %s", snapshot_name)


# ===================================================================
# 数据质量分析（扩展入口）
# ===================================================================

def compute_uniqueness(ds: fo.Dataset) -> fo.Dataset:
    """计算样本独特性分数（使用 FiftyOne Brain）。"""
    fob.compute_uniqueness(ds)
    logger.info("独特性计算完成")
    return ds


def compute_hardness(ds: fo.Dataset, label_field: str = "ground_truth") -> fo.Dataset:
    """计算标注难度分数。"""
    fob.compute_hardness(ds, label_field)
    logger.info("难度分数计算完成")
    return ds
