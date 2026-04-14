"""训练管理页面：配置面板、后台训练、训练历史、权重管理、训练后回灌。"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import trainer
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _path_browser


def _render_training_page():
    st.header("🏋️ 训练管理")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_train, tab_history, tab_feedback = st.tabs([
        "🚀 训练配置与启动", "📜 训练历史", "🔄 训练后回灌",
    ])

    with tab_train:
        _render_train_config(ds)
    with tab_history:
        _render_train_history(ds)
    with tab_feedback:
        _render_train_feedback(ds)


def _render_train_config(ds):
    st.subheader("训练配置")

    status = trainer.get_training_status()
    if status["running"]:
        st.warning(f"⏳ 训练进行中: {status['progress']}")
        if st.button("🔄 刷新状态", key="btn_refresh_status"):
            st.rerun()
        return

    if status["result"]:
        st.success("✅ 上一次训练已完成")
        with st.expander("上次训练结果", expanded=True):
            st.json(status["result"])
    if status["error"]:
        st.error(f"上次训练失败: {status['error']}")

    st.markdown("---")

    # 自动检测来自导出页面的 data.yaml
    auto_yaml = st.session_state.get("last_export_data_yaml", "")

    col_data, col_model = st.columns(2)

    with col_data:
        st.markdown("**数据配置**")
        data_yaml = _path_browser(
            "data.yaml 路径", "train_data_yaml", mode="file",
            file_extensions=(".yaml", ".yml"),
            start_dir=st.session_state.get("last_export_dir", CONFIG.default_export_dir),
        )
        if not data_yaml and auto_yaml:
            data_yaml = auto_yaml
            st.info(f"已自动填入导出页面的 data.yaml: `{auto_yaml}`")

    with col_model:
        st.markdown("**模型配置**")
        last_best = ds.info.get("last_best_pt", "")
        model_path = _path_browser(
            "模型权重路径 (.pt)", "train_model_path", mode="file",
            file_extensions=(".pt", ".pth", ".yaml"),
            start_dir=str(Path(last_best).parent) if last_best and Path(last_best).exists() else "",
        )
        if not model_path:
            model_path = st.text_input("或输入模型名称", value="yolo11n.pt", key="train_model_name",
                                       help="输入预训练模型名称如 yolo11n.pt 或本地路径")

        task = st.selectbox("任务类型", ["detect", "pose", "obb", "classify", "segment"], key="train_task")

    st.markdown("---")

    st.markdown("**训练超参数**")
    col_e, col_i, col_b, col_d = st.columns(4)
    with col_e:
        epochs = st.number_input("Epochs", 1, 1000, 100, key="train_epochs")
    with col_i:
        imgsz = st.number_input("Image Size", 32, 1920, 640, step=32, key="train_imgsz")
    with col_b:
        batch = st.number_input("Batch Size", 1, 256, 16, key="train_batch")
    with col_d:
        device = st.text_input("Device", value="0", key="train_device",
                               help="GPU 编号如 0, 或 cpu")

    with st.expander("📋 高级参数", expanded=False):
        col_lr, col_wd = st.columns(2)
        with col_lr:
            lr0 = st.number_input("初始学习率 (lr0)", 0.0001, 1.0, 0.01, 0.001, format="%.4f", key="train_lr0")
        with col_wd:
            weight_decay = st.number_input("权重衰减", 0.0, 0.1, 0.0005, 0.0001, format="%.4f", key="train_wd")
        optimizer = st.selectbox("优化器", ["auto", "SGD", "Adam", "AdamW"], key="train_optimizer")
        augment = st.checkbox("启用数据增强", value=True, key="train_augment")
        project_dir = st.text_input("训练输出目录 (project)", value="runs", key="train_project")
        run_name = st.text_input("运行名称 (name)", value="", key="train_name", placeholder="留空则自动命名")

    st.markdown("---")

    if st.button("🚀 开始训练", key="btn_start_train", type="primary"):
        if not data_yaml:
            st.error("请指定 data.yaml 路径")
            return
        if not model_path:
            st.error("请指定模型路径")
            return

        extra_args = {"lr0": lr0, "weight_decay": weight_decay, "optimizer": optimizer}
        if not augment:
            extra_args["augment"] = False

        result = trainer.start_training(
            data_yaml=data_yaml,
            model_path=model_path,
            task=task,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project_dir,
            name=run_name,
            extra_args=extra_args,
            ds=ds,
        )

        if result["started"]:
            st.success(result["message"])
            st.info("训练在后台运行中，可定期刷新此页面查看进度。训练完成后结果会自动记录到「训练历史」。")
        else:
            st.error(result["message"])


def _render_train_history(ds):
    st.subheader("训练历史")

    history = trainer.get_training_history(ds)
    if not history:
        st.info("当前数据集暂无训练记录。完成训练后会自动记录在此。")
        return

    import pandas as pd

    rows = []
    for i, h in enumerate(reversed(history)):
        metrics = h.get("metrics", {})
        row = {
            "#": len(history) - i,
            "时间": h.get("timestamp", ""),
            "模型": h.get("model", ""),
            "任务": h.get("task", ""),
            "Epochs": h.get("epochs", ""),
            "ImgSz": h.get("imgsz", ""),
        }
        mAP50 = metrics.get("metrics/mAP50(B)") or metrics.get("mAP50(B)")
        mAP50_95 = metrics.get("metrics/mAP50-95(B)") or metrics.get("mAP50-95(B)")
        if mAP50 is not None:
            row["mAP50"] = f"{mAP50:.4f}"
        if mAP50_95 is not None:
            row["mAP50-95"] = f"{mAP50_95:.4f}"
        row["best.pt"] = h.get("best_pt", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 权重管理
    st.markdown("### 🏆 最优权重管理")
    last_best = ds.info.get("last_best_pt", "")
    if last_best:
        st.info(f"当前最优权重: `{last_best}`")
        if st.button("📋 设为预标注模型", key="btn_set_predict_model"):
            st.session_state["pred_model_input"] = last_best
            st.success(f"已将 `{last_best}` 设为预标注模型，可前往「自动预标注」页面使用")

    st.markdown("---")
    st.markdown("**选择历史权重**")
    weight_options = [h.get("best_pt", "") for h in history if h.get("best_pt")]
    if weight_options:
        selected_weight = st.selectbox("选择权重", weight_options, key="hist_weight_select")
        if st.button("设为当前最优权重", key="btn_set_best"):
            ds.info["last_best_pt"] = selected_weight
            ds.save()
            st.session_state["_toast_msg"] = f"最优权重已更新: {selected_weight}"
            st.rerun()


def _render_train_feedback(ds):
    st.subheader("训练后自动回灌")
    st.caption("用训练好的模型对数据集进行预标注，然后自动评估，为难例挖掘做准备")

    best_pt = ds.info.get("last_best_pt", "")
    if not best_pt:
        st.info("暂无训练权重。请先在「训练配置与启动」中完成训练。")
        return

    st.info(f"当前最优权重: `{best_pt}`")

    col1, col2 = st.columns(2)
    with col1:
        pred_field = st.text_input("预测结果字段", value="predictions", key="fb_pred_field")
        gt_field = st.text_input("真值字段", value="ground_truth", key="fb_gt_field")
    with col2:
        task = st.selectbox("任务类型", ["detect", "pose", "obb"], key="fb_task")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="fb_conf")
        eval_key = st.text_input("评估键", value="eval", key="fb_eval_key")

    if st.button("🔄 执行回灌（预标注 + 评估）", key="btn_feedback", type="primary"):
        if not Path(best_pt).exists():
            st.error(f"权重文件不存在: {best_pt}")
            return

        with st.spinner("正在执行预标注与评估..."):
            try:
                result = trainer.run_post_training_eval(
                    ds, best_pt, pred_field=pred_field, gt_field=gt_field,
                    task=task, conf=conf, eval_key=eval_key,
                )
                st.success("✅ 回灌完成")
                mc1, mc2 = st.columns(2)
                mc1.metric("预标注样本数", result["predicted"])
                if result["mAP"] is not None:
                    mc2.metric("mAP", f"{result['mAP']:.4f}")
                else:
                    mc2.metric("mAP", "N/A")
                st.info(
                    "💡 回灌完成后可前往「高级功能 → 难例挖掘」查找模型表现差的样本，"
                    "然后推送到 CVAT 进行重标注。"
                )
            except Exception as e:
                st.error(f"回灌失败: {e}")

    # 展示训练可视化
    st.markdown("---")
    st.markdown("### 📊 训练结果可视化")
    history = trainer.get_training_history(ds)
    if history:
        latest = history[-1]
        result_dir = latest.get("result_dir")
        if result_dir and Path(result_dir).is_dir():
            vis_files = {
                "混淆矩阵": Path(result_dir) / "confusion_matrix.png",
                "混淆矩阵 (归一化)": Path(result_dir) / "confusion_matrix_normalized.png",
                "PR 曲线": Path(result_dir) / "PR_curve.png",
                "F1 曲线": Path(result_dir) / "F1_curve.png",
                "训练结果": Path(result_dir) / "results.png",
            }
            found_any = False
            for name, fpath in vis_files.items():
                if fpath.exists():
                    found_any = True
                    with st.expander(name, expanded=name == "训练结果"):
                        st.image(str(fpath), use_container_width=True)
            if not found_any:
                st.caption("训练结果目录中未找到可视化图表")
        else:
            st.caption("训练结果目录不存在或未记录")
    else:
        st.caption("暂无训练历史")
