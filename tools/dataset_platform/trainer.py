"""
训练管理模块。

提供 YOLO 模型训练的后端逻辑，包括：
- 训练参数配置与启动
- 训练历史记录
- 训练结果回灌（预标注 + 评估）
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import fiftyone as fo

logger = logging.getLogger(__name__)

_training_thread: Optional[threading.Thread] = None
_training_status: dict = {"running": False, "progress": "", "result": None, "error": None}


def get_training_status() -> dict:
    return dict(_training_status)


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

    _training_status = {"running": True, "progress": "初始化中...", "result": None, "error": None}

    def _run():
        global _training_status
        try:
            from ultralytics import YOLO
            _training_status["progress"] = "加载模型..."
            model = YOLO(model_path, task=task if task else None)

            train_kwargs = dict(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=int(device) if device.isdigit() else device,
                project=project,
                name=name or None,
            )
            if extra_args:
                train_kwargs.update(extra_args)

            _training_status["progress"] = f"训练中 (epochs={epochs})..."
            results = model.train(**train_kwargs)

            result_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else None
            best_pt = str(result_dir / "weights" / "best.pt") if result_dir else None
            last_pt = str(result_dir / "weights" / "last.pt") if result_dir else None

            metrics = {}
            if hasattr(results, "results_dict"):
                metrics = {k: float(v) for k, v in results.results_dict.items()
                           if isinstance(v, (int, float))}
            elif result_dir:
                csv_file = result_dir / "results.csv"
                if csv_file.exists():
                    import csv
                    with open(csv_file) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            for k, v in last_row.items():
                                k = k.strip()
                                try:
                                    metrics[k] = float(v)
                                except (ValueError, TypeError):
                                    pass

            train_record = {
                "timestamp": datetime.now().isoformat(),
                "data_yaml": data_yaml,
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
            logger.info("训练完成: %s", train_record)

        except Exception as e:
            _training_status["error"] = str(e)
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
