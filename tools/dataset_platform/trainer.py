"""
训练管理模块。

提供 YOLO 模型训练的后端逻辑，包括：
- 训练参数配置与启动
- 训练历史记录
- 训练结果回灌（预标注 + 评估）
"""

from __future__ import annotations

import csv
import logging
import math
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import fiftyone as fo
import yaml

logger = logging.getLogger(__name__)

_training_thread: Optional[threading.Thread] = None
_training_status: dict = {
    "running": False,
    "progress": "",
    "result": None,
    "error": None,
    "epoch": 0,
    "total_epochs": 0,
    "start_time": None,
    "end_time": None,
    "duration_seconds": 0,
    "save_dir": None,
    "current_metrics": {},
    "epoch_history": [],
    "error_trace": None,
}


def _reset_status() -> dict:
    return {
        "running": False,
        "progress": "",
        "result": None,
        "error": None,
        "epoch": 0,
        "total_epochs": 0,
        "start_time": None,
        "end_time": None,
        "duration_seconds": 0,
        "save_dir": None,
        "current_metrics": {},
        "epoch_history": [],
        "error_trace": None,
    }


def get_training_status() -> dict:
    status = dict(_training_status)
    # 训练中实时计算已用时间
    if status["running"] and status["start_time"]:
        status["duration_seconds"] = time.time() - status["start_time"]
    return status


def read_results_csv(save_dir: str | Path | None) -> list[dict]:
    """从训练输出目录读取 results.csv，返回每 epoch 的指标列表。"""
    if not save_dir:
        return []
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                cleaned = {}
                for k, v in row.items():
                    k = k.strip()
                    try:
                        cleaned[k] = float(v)
                    except (ValueError, TypeError):
                        cleaned[k] = v
                rows.append(cleaned)
            return rows
    except Exception:
        return []


def find_data_yaml(directory: str | Path) -> Optional[str]:
    """在目录中查找 data.yaml 文件。"""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    for candidate in [directory / "data.yaml", directory / "dataset.yaml"]:
        if candidate.exists():
            return str(candidate)
    for f in directory.rglob("data.yaml"):
        return str(f)
    return None


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _load_data_yaml(data_yaml: str | Path) -> dict:
    with open(data_yaml, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _names_to_id_map(names) -> dict:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _expand_train_sources(train, base_dir: Path) -> list[str]:
    paths: list[str] = []
    if isinstance(train, (list, tuple)):
        for t in train:
            paths.extend(_expand_train_sources(t, base_dir))
        return paths

    train_path = Path(train)
    if not train_path.is_absolute():
        train_path = (base_dir / train_path).resolve()

    if train_path.is_file() and train_path.suffix.lower() == ".txt":
        try:
            with open(train_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip()
                    if p:
                        paths.append(p)
        except OSError:
            return []
        return paths

    if train_path.is_dir():
        for ext in _IMAGE_EXTS:
            paths.extend(str(p) for p in train_path.rglob(f"*{ext}"))
        return paths

    if train_path.is_file():
        return [str(train_path)]

    return []


def _label_path_from_image(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = list(p.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return p.with_suffix(".txt")


def _build_balanced_data_yaml(
    data_yaml: str | Path,
    balance_classes: list[str],
    target_ratio: float,
    max_repeat: int,
    reference_classes: Optional[list[str]] = None,
    pure_minority_only: bool = False,
) -> tuple[str, dict | None]:
    data_yaml = str(data_yaml)
    data = _load_data_yaml(data_yaml)
    names_map = _names_to_id_map(data.get("names", {}))
    name_to_id = {v: k for k, v in names_map.items()}
    class_ids = [name_to_id[n] for n in balance_classes if n in name_to_id]
    if not class_ids:
        return data_yaml, None
    reference_ids = [name_to_id[n] for n in (reference_classes or []) if n in name_to_id]

    base_dir = Path(data_yaml).parent
    train_sources = data.get("train")
    if not train_sources:
        return data_yaml, None

    image_paths = _expand_train_sources(train_sources, base_dir)
    if not image_paths:
        return data_yaml, None

    class_counts = {cid: 0 for cid in class_ids}
    image_repeatable: dict[str, bool] = {}
    image_class_ids: dict[str, set[int]] = {}
    for img in image_paths:
        label_path = _label_path_from_image(img)
        has_minority = False
        img_class_ids = set()
        if label_path.exists():
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(float(parts[0]))
                        except ValueError:
                            continue
                        img_class_ids.add(cls_id)
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
                            has_minority = True
            except OSError:
                pass
        image_class_ids[img] = img_class_ids
        image_repeatable[img] = has_minority and (
            not pure_minority_only or img_class_ids.issubset(set(class_ids))
        )

    minority_total = sum(class_counts.values())
    if minority_total <= 0:
        return data_yaml, None

    majority_total = 0
    for img in image_paths:
        label_path = _label_path_from_image(img)
        if not label_path.exists():
            continue
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if reference_ids:
                        if cls_id in reference_ids:
                            majority_total += 1
                    elif cls_id not in class_counts:
                        majority_total += 1
        except OSError:
            pass

    if majority_total <= 0:
        return data_yaml, None

    repeatable_minority_total = 0
    for img, can_repeat in image_repeatable.items():
        if not can_repeat:
            continue
        label_path = _label_path_from_image(img)
        if not label_path.exists():
            continue
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    if cls_id in class_counts:
                        repeatable_minority_total += 1
        except OSError:
            pass

    if repeatable_minority_total <= 0:
        return data_yaml, None

    desired_minor = max(1, int(majority_total * float(target_ratio)))
    extra_needed = max(0, desired_minor - minority_total)
    repeat = min(max_repeat, max(1, math.ceil(extra_needed / repeatable_minority_total) + 1))
    if repeat <= 1:
        return data_yaml, None

    balanced_list: list[str] = []
    for img in image_paths:
        balanced_list.append(img)
        if image_repeatable.get(img):
            balanced_list.extend([img] * (repeat - 1))

    tmp_dir = Path.home() / ".dataset_platform" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    list_path = tmp_dir / f"train_balanced_{ts}.txt"
    yaml_path = tmp_dir / f"data_balanced_{ts}.yaml"

    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(balanced_list))

    data["train"] = str(list_path)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    info = {
        "minority_total": minority_total,
        "majority_total": majority_total,
        "repeatable_minority_total": repeatable_minority_total,
        "desired_minority_total": desired_minor,
        "repeat": repeat,
        "train_list": str(list_path),
        "reference_classes": reference_classes or [],
        "pure_minority_only": pure_minority_only,
    }
    return str(yaml_path), info


def start_training(
    data_yaml: str,
    model_path: str = "yolo11n.pt",
    task: str = "detect",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs",
    name: str = "",
    extra_args: Optional[dict] = None,
    ds: Optional[fo.Dataset] = None,
) -> dict:
    """
    在后台线程中启动 YOLO 训练。

    Returns:
        {"started": bool, "message": str}
    """
    global _training_thread, _training_status

    if _training_status["running"]:
        return {"started": False, "message": "已有训练任务正在运行"}

    if not Path(data_yaml).exists():
        return {"started": False, "message": f"data.yaml 不存在: {data_yaml}"}

    _training_status = _reset_status()
    _training_status["running"] = True
    _training_status["progress"] = "初始化中..."
    _training_status["total_epochs"] = epochs
    _training_status["start_time"] = time.time()

    def _epoch_end_callback(trainer_obj):
        """YOLO 每 epoch 验证完成后回调，实时更新训练状态。"""
        global _training_status
        cur = trainer_obj.epoch + 1
        total = trainer_obj.epochs
        _training_status["epoch"] = cur
        _training_status["total_epochs"] = total
        _training_status["progress"] = f"训练中 ({cur}/{total})..."
        if hasattr(trainer_obj, "save_dir"):
            _training_status["save_dir"] = str(trainer_obj.save_dir)
        # 从 trainer 的 metrics 中提取数值指标
        if hasattr(trainer_obj, "metrics") and trainer_obj.metrics:
            cur_m = {}
            src = trainer_obj.metrics
            if isinstance(src, dict):
                for k, v in src.items():
                    if isinstance(v, (int, float)):
                        cur_m[k.strip()] = round(float(v), 6)
            _training_status["current_metrics"] = cur_m
        # 从 results.csv 读取完整历史
        sd = _training_status.get("save_dir")
        if sd:
            _training_status["epoch_history"] = read_results_csv(sd)

    def _train_start_callback(trainer_obj):
        """训练开始时捕获 save_dir。"""
        global _training_status
        if hasattr(trainer_obj, "save_dir"):
            _training_status["save_dir"] = str(trainer_obj.save_dir)

    def _run():
        global _training_status
        t_start = _training_status["start_time"]
        try:
            from ultralytics import YOLO
            _training_status["progress"] = "加载模型..."
            model = YOLO(model_path, task=task if task else None)

            model.add_callback("on_train_start", _train_start_callback)
            model.add_callback("on_fit_epoch_end", _epoch_end_callback)

            balance_classes = []
            balance_reference_classes = []
            balance_target_ratio = 0.5
            balance_max_repeat = 3
            balance_pure_minority_only = False
            if extra_args:
                balance_classes = extra_args.pop("balance_classes", []) or []
                balance_reference_classes = extra_args.pop("balance_reference_classes", []) or []
                balance_target_ratio = extra_args.pop("balance_target_ratio", 0.5)
                balance_max_repeat = extra_args.pop("balance_max_repeat", 3)
                balance_pure_minority_only = extra_args.pop("balance_pure_minority_only", False)

            data_yaml_path = data_yaml
            balance_info = None
            if balance_classes:
                data_yaml_path, balance_info = _build_balanced_data_yaml(
                    data_yaml_path,
                    balance_classes=balance_classes,
                    target_ratio=float(balance_target_ratio),
                    max_repeat=int(balance_max_repeat),
                    reference_classes=balance_reference_classes,
                    pure_minority_only=bool(balance_pure_minority_only),
                )
                if balance_info:
                    _training_status["progress"] = (
                        "已启用少数类重采样: repeat="
                        f"{balance_info['repeat']}"
                    )

            train_kwargs = dict(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=int(device) if device.isdigit() else device,
                project=project,
                name=name or None,
            )
            if extra_args:
                train_kwargs.update(extra_args)

            _training_status["progress"] = f"训练中 (0/{epochs})..."
            results = model.train(**train_kwargs)

            t_end = time.time()
            _training_status["end_time"] = t_end
            _training_status["duration_seconds"] = t_end - t_start

            result_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else None
            _training_status["save_dir"] = str(result_dir) if result_dir else None
            best_pt = str(result_dir / "weights" / "best.pt") if result_dir else None
            last_pt = str(result_dir / "weights" / "last.pt") if result_dir else None

            metrics = {}
            if hasattr(results, "results_dict"):
                metrics = {k: float(v) for k, v in results.results_dict.items()
                           if isinstance(v, (int, float))}
            elif result_dir:
                rows = read_results_csv(result_dir)
                if rows:
                    metrics = {k: v for k, v in rows[-1].items() if isinstance(v, (int, float))}

            duration = t_end - t_start
            train_record = {
                "timestamp": datetime.fromtimestamp(t_start).isoformat(),
                "end_time": datetime.fromtimestamp(t_end).isoformat(),
                "duration_seconds": round(duration, 1),
                "data_yaml": data_yaml,
                "train_data_yaml": data_yaml_path,
                "balance_info": balance_info,
                "model": model_path,
                "task": task,
                "epochs": epochs,
                "imgsz": imgsz,
                "batch": batch,
                "best_pt": best_pt,
                "last_pt": last_pt,
                "result_dir": str(result_dir) if result_dir else None,
                "metrics": metrics,
            }

            if ds is not None:
                history = ds.info.get("training_history", [])
                history.append(train_record)
                ds.info["training_history"] = history
                ds.info["last_best_pt"] = best_pt
                ds.save()

            _training_status["result"] = train_record
            _training_status["progress"] = "训练完成"
            _training_status["epoch"] = epochs
            # 确保最终的 epoch_history 完整
            if result_dir:
                _training_status["epoch_history"] = read_results_csv(result_dir)
            logger.info("训练完成: %s", train_record)

        except Exception as e:
            t_end = time.time()
            _training_status["end_time"] = t_end
            _training_status["duration_seconds"] = t_end - t_start
            _training_status["error"] = str(e)
            _training_status["error_trace"] = traceback.format_exc()
            _training_status["progress"] = f"训练失败: {e}"
            logger.error("训练失败: %s", e)
        finally:
            _training_status["running"] = False

    _training_thread = threading.Thread(target=_run, daemon=True)
    _training_thread.start()
    return {"started": True, "message": "训练已在后台启动"}


def get_training_history(ds: fo.Dataset) -> list[dict]:
    """获取数据集的训练历史记录。"""
    return ds.info.get("training_history", [])


def run_post_training_eval(
    ds: fo.Dataset,
    best_pt: str,
    pred_field: str = "predictions",
    gt_field: str = "ground_truth",
    task: str = "detect",
    conf: float = 0.25,
    eval_key: str = "eval",
) -> dict:
    """
    训练后自动回灌：用 best.pt 预标注 + 评估。

    Returns:
        {"predicted": int, "mAP": float | None}
    """
    from . import processor
    from . import advanced as adv

    stats = processor.auto_predict_yolo(
        ds, best_pt, pred_field=pred_field,
        conf_threshold=conf, task=task,
    )
    predicted = stats.get("predicted", 0)

    mAP = None
    try:
        adv.evaluate_detections(
            ds, pred_field=pred_field, gt_field=gt_field,
            eval_key=eval_key,
        )
        results = ds.load_evaluation_results(eval_key)
        mAP = results.mAP()
    except Exception as e:
        logger.warning("评估失败: %s", e)

    return {"predicted": predicted, "mAP": mAP}
