"""
FiftyOne 底层数据操作引擎。

封装所有与 FiftyOne 数据库的交互操作，包括数据集的 CRUD、
样本管理、标签统计、视图过滤等核心功能。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import fiftyone as fo
import fiftyone.core.fields as fof

from .config import CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

def list_datasets() -> list[str]:
    return fo.list_datasets()


def load_dataset(name: str) -> fo.Dataset:
    if not fo.dataset_exists(name):
        raise ValueError(f"数据集 '{name}' 不存在")
    return fo.load_dataset(name)


def create_dataset(name: str, persistent: bool = True) -> fo.Dataset:
    if fo.dataset_exists(name):
        raise ValueError(f"数据集 '{name}' 已存在")
    ds = fo.Dataset(name=name, persistent=persistent)
    logger.info("创建数据集: %s", name)
    return ds


def rename_dataset(old_name: str, new_name: str) -> fo.Dataset:
    ds = load_dataset(old_name)
    ds.name = new_name
    logger.info("重命名数据集: %s -> %s", old_name, new_name)
    return ds


def delete_dataset(name: str) -> None:
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
        logger.info("删除数据集: %s", name)


def clone_dataset(src_name: str, dst_name: str) -> fo.Dataset:
    ds = load_dataset(src_name)
    cloned = ds.clone(name=dst_name)
    cloned.persistent = True
    logger.info("克隆数据集: %s -> %s", src_name, dst_name)
    return cloned


# ---------------------------------------------------------------------------
# Dataset info & stats
# ---------------------------------------------------------------------------

def get_dataset_info(ds: fo.Dataset) -> dict:
    """返回数据集的基础统计信息。"""
    info = {
        "name": ds.name,
        "num_samples": len(ds),
        "media_type": ds.media_type,
        "persistent": ds.persistent,
        "tags": ds.distinct("tags"),
        "sample_fields": list(ds.get_field_schema().keys()),
    }

    label_fields = []
    for field_name, field_obj in ds.get_field_schema().items():
        if isinstance(field_obj, fof.EmbeddedDocumentField):
            doc_type = field_obj.document_type
            if doc_type and issubclass(doc_type, (fo.Detections, fo.Keypoints, fo.Polylines, fo.Classifications)):
                label_fields.append(field_name)
    info["label_fields"] = label_fields
    return info


def _resolve_label_path(ds: fo.Dataset, field_name: str) -> str | None:
    """根据字段类型返回正确的 label 子路径，如 'ground_truth.detections.label'。"""
    schema = ds.get_field_schema()
    if field_name not in schema:
        return None
    field = schema[field_name]
    if not hasattr(field, "document_type") or not field.document_type:
        return None
    doc = field.document_type
    if issubclass(doc, fo.Detections):
        return f"{field_name}.detections.label"
    if issubclass(doc, fo.Keypoints):
        return f"{field_name}.keypoints.label"
    if issubclass(doc, fo.Polylines):
        return f"{field_name}.polylines.label"
    if issubclass(doc, fo.Classifications):
        return f"{field_name}.classifications.label"
    return None


def get_field_label_type(ds: fo.Dataset, field_name: str) -> str | None:
    """返回标签字段对应的 CVAT 标注类型名，如 'detections', 'polylines', 'keypoints' 等。"""
    schema = ds.get_field_schema()
    if field_name not in schema:
        return None
    field = schema[field_name]
    if not hasattr(field, "document_type") or not field.document_type:
        return None
    doc = field.document_type
    _TYPE_MAP = {
        fo.Detections: "detections",
        fo.Keypoints: "keypoints",
        fo.Polylines: "polylines",
        fo.Classifications: "classifications",
    }
    for cls, name in _TYPE_MAP.items():
        if issubclass(doc, cls):
            return name
    return None


def get_label_classes(ds: fo.Dataset, field_name: str = "ground_truth") -> list[str]:
    """获取指定标签字段的所有类别。"""
    path = _resolve_label_path(ds, field_name)
    if path is None:
        return []
    return sorted(c for c in ds.distinct(path) if c is not None)


def get_label_stats(ds: fo.Dataset, field_name: str = "ground_truth") -> dict:
    """获取标签统计：各类的实例数。"""
    path = _resolve_label_path(ds, field_name)
    if path is None:
        return {}
    return {k: v for k, v in ds.count_values(path).items() if k is not None}


# ---------------------------------------------------------------------------
# Sample management
# ---------------------------------------------------------------------------

def add_samples_from_dir(
    ds: fo.Dataset,
    image_dir: str | Path,
    tags: Optional[list[str]] = None,
    recursive: bool = True,
) -> int:
    """从目录批量导入纯图片样本。"""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {image_dir}")

    exts = CONFIG.image_extensions
    if recursive:
        paths = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
    else:
        paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        )

    if not paths:
        logger.warning("目录 %s 中未找到图片", image_dir)
        return 0

    existing_fps = set(ds.values("filepath"))
    new_paths = [p for p in paths if str(p.resolve()) not in existing_fps]

    if not new_paths:
        logger.info("所有图片已存在于数据集中，跳过")
        return 0

    samples = []
    for p in new_paths:
        s = fo.Sample(filepath=str(p.resolve()))
        if tags:
            s.tags.extend(tags)
        samples.append(s)

    ds.add_samples(samples)
    logger.info("导入 %d 张图片到数据集 '%s'", len(samples), ds.name)
    return len(samples)


def remove_samples_by_ids(ds: fo.Dataset, sample_ids: list[str]) -> int:
    """通过 ID 列表删除样本。"""
    if not sample_ids:
        return 0
    view = ds.select(sample_ids)
    count = len(view)
    ds.delete_samples(view)
    logger.info("从数据集 '%s' 中删除 %d 个样本", ds.name, count)
    return count


def delete_samples_physically(ds: fo.Dataset, sample_ids: list[str]) -> int:
    """同时删除数据库记录和磁盘文件。"""
    if not sample_ids:
        return 0
    view = ds.select(sample_ids)
    filepaths = view.values("filepath")
    count = len(filepaths)

    for fp in filepaths:
        p = Path(fp)
        if p.exists():
            p.unlink()

    ds.delete_samples(view)
    logger.info("物理删除 %d 个样本", count)
    return count


# ---------------------------------------------------------------------------
# View / filter helpers
# ---------------------------------------------------------------------------

def _get_container_attr(ds: fo.Dataset, label_field: str) -> str | None:
    """根据字段类型返回容器属性名（detections / polylines / keypoints / classifications）。"""
    schema = ds.get_field_schema()
    if label_field not in schema:
        return None
    field = schema[label_field]
    if not hasattr(field, "document_type") or not field.document_type:
        return None
    doc = field.document_type
    _MAP = {
        fo.Detections: "detections",
        fo.Keypoints: "keypoints",
        fo.Polylines: "polylines",
        fo.Classifications: "classifications",
    }
    for cls, attr in _MAP.items():
        if issubclass(doc, cls):
            return attr
    return None


def get_unlabeled_view(ds: fo.Dataset, label_field: str = "ground_truth") -> fo.DatasetView:
    """获取无标注的样本视图。"""
    container = _get_container_attr(ds, label_field)
    if container:
        return ds.match(
            fo.ViewExpression.is_null(fo.ViewField(label_field))
            | (fo.ViewField(f"{label_field}.{container}").length() == 0)
        )
    if label_field in ds.get_field_schema():
        return ds.match(fo.ViewExpression.is_null(fo.ViewField(label_field)))
    return ds.view()


def get_labeled_view(ds: fo.Dataset, label_field: str = "ground_truth") -> fo.DatasetView:
    """获取有标注的样本视图。"""
    container = _get_container_attr(ds, label_field)
    if container:
        return ds.match(
            fo.ViewField(f"{label_field}.{container}").length() > 0
        )
    if label_field in ds.get_field_schema():
        return ds.match(fo.ViewField(label_field).exists())
    return ds.limit(0)


def filter_by_classes(
    ds: fo.Dataset,
    classes: list[str],
    label_field: str = "ground_truth",
) -> fo.DatasetView:
    """按指定类别过滤样本（包含任一指定类别即保留）。"""
    from fiftyone import ViewField as F
    return ds.filter_labels(
        label_field,
        F("label").is_in(classes),
    )


def filter_by_tags(ds: fo.Dataset, tags: list[str]) -> fo.DatasetView:
    return ds.match_tags(tags)


def rename_label(
    ds: fo.Dataset,
    field_name: str,
    old_label: str,
    new_label: str,
) -> int:
    """将指定标签字段中的某个类别名重命名，返回修改的标注实例数。"""
    from fiftyone import ViewField as F

    schema = ds.get_field_schema()
    if field_name not in schema:
        raise ValueError(f"字段 '{field_name}' 不存在")

    path = _resolve_label_path(ds, field_name)
    if path is None:
        raise ValueError(f"字段 '{field_name}' 不是标签类型字段")

    sub_path = path.rsplit(".label", 1)[0]  # e.g. "ground_truth.detections"
    view = ds.filter_labels(field_name, F("label") == old_label)

    count = 0
    for sample in view.iter_samples(autosave=True):
        container = sample[field_name]
        if container is None:
            continue
        items_attr = sub_path.split(".")[-1]  # "detections", "keypoints", etc.
        items = getattr(container, items_attr, [])
        for item in items:
            if item.label == old_label:
                item.label = new_label
                count += 1

    logger.info("标签重命名: %s -> %s (共 %d 个实例, 字段=%s)", old_label, new_label, count, field_name)
    return count


def delete_labels_by_class(
    ds: fo.Dataset,
    field_name: str,
    class_names: list[str],
    view: fo.DatasetView | None = None,
) -> int:
    """批量删除指定字段中特定类别的所有标注实例，返回删除数。"""
    from fiftyone import ViewField as F

    path = _resolve_label_path(ds, field_name)
    if path is None:
        raise ValueError(f"字段 '{field_name}' 不是标签类型字段")

    sub_path = path.rsplit(".label", 1)[0]
    items_attr = sub_path.split(".")[-1]

    target = view if view is not None else ds
    class_set = set(class_names)
    count = 0
    for sample in target.iter_samples(autosave=True):
        container = sample[field_name]
        if container is None:
            continue
        items = getattr(container, items_attr, [])
        before = len(items)
        kept = [it for it in items if it.label not in class_set]
        removed = before - len(kept)
        if removed > 0:
            setattr(container, items_attr, kept)
            count += removed

    logger.info("批量删除标签: 字段=%s, 类别=%s, 删除=%d", field_name, class_names, count)
    return count


def delete_sample_field(ds: fo.Dataset, field_name: str):
    """删除数据集中的指定样本字段。"""
    ds.delete_sample_field(field_name)
    logger.info("已删除字段: %s", field_name)


# ---------------------------------------------------------------------------
# Label field merge
# ---------------------------------------------------------------------------

def merge_label_fields(
    ds: fo.Dataset,
    source_field: str,
    target_field: str,
    delete_source: bool = False,
) -> dict:
    """
    将 source_field 的标注合并到 target_field 中。

    适用场景：分批次导入时使用了不同字段名（如 batch2_gt），
    标注完成后需要合并到统一字段（如 ground_truth）。

    对每个样本：
      - 若 source 为空 → 跳过
      - 若 target 为空 → 直接复制
      - 若两者都有 → 将 source 的标注实例追加到 target

    Returns:
        {"merged": int, "skipped": int, "source_deleted": bool}
    """
    schema = ds.get_field_schema()
    if source_field not in schema:
        raise ValueError(f"源字段 '{source_field}' 不存在")

    src_field = schema[source_field]
    if not hasattr(src_field, "document_type") or not src_field.document_type:
        raise ValueError(f"源字段 '{source_field}' 不是标签类型字段")

    doc_type = src_field.document_type
    _CONTAINER_ATTR = {
        fo.Detections: "detections",
        fo.Keypoints: "keypoints",
        fo.Polylines: "polylines",
        fo.Classifications: "classifications",
    }
    container_attr = None
    for cls, attr in _CONTAINER_ATTR.items():
        if issubclass(doc_type, cls):
            container_attr = attr
            break

    if container_attr is None:
        raise ValueError(f"不支持合并的字段类型: {doc_type}")

    merged = 0
    skipped = 0
    for sample in ds.iter_samples(autosave=True):
        src_val = sample[source_field]
        if src_val is None:
            skipped += 1
            continue

        src_items = getattr(src_val, container_attr, [])
        if not src_items:
            skipped += 1
            continue

        tgt_val = sample[target_field] if target_field in sample.field_names else None
        if tgt_val is None:
            sample[target_field] = src_val
        else:
            tgt_items = getattr(tgt_val, container_attr, [])
            tgt_items.extend(src_items)
        merged += 1

    if delete_source:
        ds.delete_sample_field(source_field)

    logger.info(
        "字段合并: %s -> %s (merged=%d, skipped=%d, deleted=%s)",
        source_field, target_field, merged, skipped, delete_source,
    )
    return {"merged": merged, "skipped": skipped, "source_deleted": delete_source}


# ---------------------------------------------------------------------------
# FiftyOne App session management
# ---------------------------------------------------------------------------

_session: Optional[fo.Session] = None


def _is_session_alive() -> bool:
    """安全检测 FiftyOne Session 是否仍在运行。"""
    if _session is None:
        return False
    try:
        # 不同版本的 FiftyOne 使用不同的属性
        if hasattr(_session, "_is_open"):
            return _session._is_open
        if hasattr(_session, "is_open"):
            return _session.is_open
        # 尝试访问 dataset 属性来判断 session 是否有效
        _ = _session.dataset
        return True
    except Exception:
        return False


def launch_app(ds: fo.Dataset, port: Optional[int] = None, address: str = "localhost") -> fo.Session:
    """启动或更新 FiftyOne App 会话。"""
    global _session
    port = port or CONFIG.fiftyone_port
    if not _is_session_alive():
        _session = fo.launch_app(ds, port=port, address=address, auto=False)
    else:
        _session.dataset = ds
    return _session


def close_app() -> None:
    """关闭 FiftyOne App 会话。"""
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None


def switch_session_dataset(ds: Optional[fo.Dataset]) -> None:
    """切换当前 session 指向的数据集，ds=None 时关闭 session。"""
    global _session
    if not _is_session_alive():
        return
    if ds is None:
        close_app()
    else:
        try:
            _session.dataset = ds
        except Exception:
            close_app()


def get_session() -> Optional[fo.Session]:
    return _session


def ensure_app(ds: fo.Dataset, port: Optional[int] = None) -> fo.Session:
    """确保 FiftyOne App 已启动，未启动则自动启动。返回 session。"""
    if _is_session_alive():
        return _session
    return launch_app(ds, port=port)


def set_session_view(view: fo.DatasetView) -> None:
    """设置当前 session 的视图。"""
    global _session
    if _session is not None:
        _session.view = view


# ---------------------------------------------------------------------------
# Filepath diagnostics & repair
# ---------------------------------------------------------------------------

def diagnose_filepaths(ds: fo.Dataset, check_limit: int = 0) -> dict:
    """
    诊断数据集中图片路径的健康状况。

    Returns:
        {
            "total": int,
            "missing": int,
            "missing_samples": [{id, filepath}],
            "symlinks": int,
            "relative": int,
        }
    """
    filepaths = ds.values(["id", "filepath"])
    total = len(filepaths)
    missing_samples = []
    symlinks = 0
    relative = 0

    items = filepaths if check_limit <= 0 else filepaths[:check_limit]
    for sid, fp in items:
        p = Path(fp)
        if not p.is_absolute():
            relative += 1
        if p.is_symlink():
            symlinks += 1
        if not p.exists():
            missing_samples.append({"id": sid, "filepath": fp})

    return {
        "total": total,
        "checked": len(items),
        "missing": len(missing_samples),
        "missing_samples": missing_samples[:50],
        "symlinks": symlinks,
        "relative": relative,
    }


def fix_filepaths(
    ds: fo.Dataset,
    old_prefix: str,
    new_prefix: str,
) -> int:
    """
    批量替换数据集中文件路径的前缀。

    用于图片目录迁移后修复路径。例如：
        old_prefix="/old/path/images" → new_prefix="/new/path/images"

    Returns:
        修改的样本数
    """
    count = 0
    for sample in ds.iter_samples(autosave=True):
        fp = sample.filepath
        if fp.startswith(old_prefix):
            sample.filepath = fp.replace(old_prefix, new_prefix, 1)
            count += 1
    logger.info("路径修复: 替换前缀 '%s' -> '%s', 共 %d 个样本", old_prefix, new_prefix, count)
    return count
