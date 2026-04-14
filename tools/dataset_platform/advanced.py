"""
高级数据管理与 AI 辅助功能。

- 难例挖掘：比较 ground_truth 与 predictions，找出高错误率样本
- 数据集版本控制：利用 FiftyOne clone() 进行快照备份
- 完整数据集备份：复制图像文件 + 导出 FiftyOne 元数据与标注
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
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
    gt_field: str = "ground_truth",
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

            gt = sample.get_field(gt_field)
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
# 完整数据集备份（图像 + 标注 + 元数据）
# ===================================================================

def backup_dataset(
    ds: fo.Dataset,
    backup_dir: str | Path,
    note: str = "",
) -> dict:
    """
    完整备份数据集，包含图像文件和所有标注/元数据。

    备份目录结构::

        backup_dir/
        ├── images/          # 原始图像文件的副本
        ├── metadata/        # FiftyOne JSON 格式导出（含全部标注与元数据）
        ├── backup_info.json # 备份摘要信息
        └── labels_summary/  # 每个标签字段的统计 JSON

    Returns:
        {"backup_dir": str, "images_copied": int, "images_failed": int,
         "metadata_exported": bool, "note": str}
    """
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    images_dir = backup_dir / "images"
    images_dir.mkdir(exist_ok=True)
    metadata_dir = backup_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    labels_dir = backup_dir / "labels_summary"
    labels_dir.mkdir(exist_ok=True)

    copied = 0
    failed = 0
    failed_paths: list[str] = []
    filepaths = ds.values("filepath")
    for fp in filepaths:
        src = Path(fp)
        dst = images_dir / src.name
        if dst.exists():
            stem = src.stem
            suffix = src.suffix
            counter = 1
            while dst.exists():
                dst = images_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(str(src), str(dst))
            copied += 1
        except Exception as e:
            failed += 1
            failed_paths.append(f"{fp}: {e}")
            logger.warning("备份图像失败 %s: %s", fp, e)

    metadata_exported = False
    try:
        ds.export(
            export_dir=str(metadata_dir),
            dataset_type=fo.types.FiftyOneDataset,
            export_media=False,
        )
        metadata_exported = True
    except Exception as e:
        logger.error("导出 FiftyOne 元数据失败: %s", e)

    from .data_manager import get_dataset_info, get_label_stats
    info = get_dataset_info(ds)
    for lf in info.get("label_fields", []):
        stats = get_label_stats(ds, lf)
        if stats:
            (labels_dir / f"{lf}.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8",
            )

    backup_info = {
        "dataset_name": ds.name,
        "backup_time": datetime.now().isoformat(),
        "note": note,
        "num_samples": len(ds),
        "images_copied": copied,
        "images_failed": failed,
        "failed_paths": failed_paths[:20],
        "metadata_exported": metadata_exported,
        "tags": ds.distinct("tags"),
        "label_fields": info.get("label_fields", []),
    }
    (backup_dir / "backup_info.json").write_text(
        json.dumps(backup_info, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    logger.info(
        "完整备份完成: %s -> %s (图像: %d/%d, 元数据: %s)",
        ds.name, backup_dir, copied, len(filepaths), metadata_exported,
    )
    return {
        "backup_dir": str(backup_dir),
        "images_copied": copied,
        "images_failed": failed,
        "metadata_exported": metadata_exported,
        "note": note,
    }


def list_backups(backup_root: str | Path) -> list[dict]:
    """扫描备份根目录下的所有备份。"""
    backup_root = Path(backup_root)
    backups = []
    if not backup_root.is_dir():
        return backups
    try:
        entries = sorted(backup_root.iterdir(), reverse=True)
    except PermissionError:
        logger.warning("无权限访问备份目录: %s", backup_root)
        return backups
    for d in entries:
        if not d.is_dir():
            continue
        info_file = d / "backup_info.json"
        try:
            if info_file.exists():
                info = json.loads(info_file.read_text(encoding="utf-8"))
                info["path"] = str(d)
                backups.append(info)
        except PermissionError:
            continue
        except Exception:
            pass
    return backups


def restore_from_backup(
    backup_dir: str | Path,
    target_name: Optional[str] = None,
) -> str:
    """
    从完整备份恢复数据集。

    读取备份目录中的 FiftyOne 元数据重建数据集，
    并将备份的图像路径映射到 backup_dir/images/。
    """
    backup_dir = Path(backup_dir)
    info_file = backup_dir / "backup_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"备份信息文件不存在: {info_file}")

    info = json.loads(info_file.read_text(encoding="utf-8"))
    if target_name is None:
        target_name = info.get("dataset_name", "restored_dataset")

    metadata_dir = backup_dir / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"元数据目录不存在: {metadata_dir}")

    if fo.dataset_exists(target_name):
        fo.delete_dataset(target_name)

    ds = fo.Dataset.from_dir(
        dataset_dir=str(metadata_dir),
        dataset_type=fo.types.FiftyOneDataset,
        name=target_name,
    )
    ds.persistent = True
    ds.info["restored_from_backup"] = str(backup_dir)
    ds.info["restored_time"] = datetime.now().isoformat()
    ds.save()

    logger.info("从备份恢复数据集: %s -> %s", backup_dir, target_name)
    return target_name


def import_from_backup(
    backup_dir: str | Path,
    target_ds: fo.Dataset,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    从备份导入数据到已有数据集（跨实例迁移的最佳方式）。

    与 ``restore_from_backup`` 不同，此函数不会覆盖目标数据集，
    而是将备份中的样本（图像 + 标注 + 原始 Tags）**追加** 到目标数据集。
    图像路径直接指向备份目录中的 ``images/``。

    工作流::

        1. 读取 backup_info.json 确认备份有效
        2. 加载备份的 FiftyOne 元数据为临时数据集
        3. 将临时数据集中每个样本的 filepath 重映射到 backup_dir/images/
        4. 合并到目标数据集（跳过已存在的文件路径）
        5. 可选追加额外 Tags

    Args:
        backup_dir: 备份目录路径
        target_ds: 要导入到的目标 FiftyOne 数据集
        tags: 额外的 Tags（导入批次标记等），会追加到每个导入样本

    Returns:
        {"imported": int, "skipped_dup": int, "tags_added": list, "label_fields": list}
    """
    backup_dir = Path(backup_dir)
    info_file = backup_dir / "backup_info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"备份信息文件不存在: {info_file}")

    images_dir = backup_dir / "images"
    metadata_dir = backup_dir / "metadata"

    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"元数据目录不存在: {metadata_dir}")

    info = json.loads(info_file.read_text(encoding="utf-8"))

    tmp_name = f"_import_tmp_{target_ds.name}_{datetime.now().strftime('%H%M%S')}"
    try:
        tmp_ds = fo.Dataset.from_dir(
            dataset_dir=str(metadata_dir),
            dataset_type=fo.types.FiftyOneDataset,
            name=tmp_name,
        )
    except Exception as e:
        raise RuntimeError(f"加载备份元数据失败: {e}") from e

    try:
        # 重映射 filepath → backup_dir/images/
        if images_dir.is_dir():
            image_files = {p.name: str(p) for p in images_dir.iterdir() if p.is_file()}
        else:
            image_files = {}

        existing_fps = set(target_ds.values("filepath"))

        imported = 0
        skipped_dup = 0
        samples_to_add = []

        for sample in tmp_ds.iter_samples():
            orig_name = Path(sample.filepath).name
            if orig_name in image_files:
                new_fp = image_files[orig_name]
            else:
                new_fp = sample.filepath

            if new_fp in existing_fps:
                skipped_dup += 1
                continue

            sample_dict = sample.to_dict()
            sample_dict.pop("id", None)
            sample_dict.pop("_id", None)
            sample_dict["filepath"] = new_fp

            new_sample = fo.Sample.from_dict(sample_dict)
            if tags:
                for t in tags:
                    if t not in new_sample.tags:
                        new_sample.tags.append(t)
            samples_to_add.append(new_sample)
            imported += 1

        if samples_to_add:
            target_ds.add_samples(samples_to_add)

    finally:
        if fo.dataset_exists(tmp_name):
            fo.delete_dataset(tmp_name)

    result = {
        "imported": imported,
        "skipped_dup": skipped_dup,
        "tags_added": tags or [],
        "label_fields": info.get("label_fields", []),
        "source_dataset": info.get("dataset_name", "unknown"),
        "backup_note": info.get("note", ""),
    }
    logger.info(
        "从备份导入: %s -> %s (imported=%d, skipped=%d)",
        backup_dir, target_ds.name, imported, skipped_dup,
    )
    return result


def delete_backup(backup_dir: str | Path) -> None:
    """删除备份目录。"""
    backup_dir = Path(backup_dir)
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)
        logger.info("删除备份: %s", backup_dir)


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
