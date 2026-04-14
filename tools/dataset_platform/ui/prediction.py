"""自动预标注独立页面。"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import processor
from tools.dataset_platform.ui.components import _get_ds, _path_browser


def _render_auto_predict_page():
    """独立的自动预标注页面。"""
    st.header("🤖 自动预标注")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    st.caption(
        "使用 YOLO 模型对数据集样本进行自动预标注，生成的标注可作为 CVAT 标注员的预标注参考，"
        "或用于难例挖掘中与人工标注对比评估。"
    )

    info = dm.get_dataset_info(ds)
    col_overview, col_unlabeled = st.columns(2)
    with col_overview:
        st.metric("📷 数据集样本数", info["num_samples"])
    with col_unlabeled:
        unlabeled = processor.find_unlabeled_samples(ds)
        st.metric("🔲 无标注样本数", len(unlabeled))

    st.markdown("---")

    col_model, col_config = st.columns(2)

    with col_model:
        st.subheader("模型配置")
        model_path = _path_browser(
            "模型权重路径 (.pt)", "pred_model", mode="file",
            file_extensions=(".pt", ".pth", ".onnx", ".engine"),
        )
        task = st.selectbox("任务类型", ["detect", "pose", "obb"], key="pred_task",
                            help="detect=矩形框检测, pose=关键点检测, obb=旋转框检测")

    with col_config:
        st.subheader("预标注配置")
        pred_field = st.text_input("预测结果字段名", value="predictions", key="pred_field",
                                   help="预标注结果会存入此字段，后续可在 FiftyOne 中查看或用于模型评估")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="pred_conf",
                         help="低于此值的预测将被过滤")

    st.markdown("---")

    st.subheader("预标注范围")
    pred_scope = st.radio(
        "选择范围", ["仅无标注样本", "整个数据集", "按 Tags 筛选"],
        key="pred_scope", horizontal=True,
    )

    pred_view = None
    if pred_scope == "仅无标注样本":
        pred_view = unlabeled
        st.info(f"将对 **{len(pred_view)}** 个无标注样本进行预标注")
    elif pred_scope == "按 Tags 筛选":
        available_tags = ds.distinct("tags")
        if available_tags:
            pred_tags = st.multiselect("选择 Tags", available_tags, key="pred_tags")
            if pred_tags:
                pred_view = ds.match_tags(pred_tags)
                st.info(f"将对 **{len(pred_view)}** 个匹配样本进行预标注")
        else:
            st.info("当前数据集没有 Tags")
    else:
        st.info(f"将对整个数据集的 **{len(ds)}** 个样本进行预标注")

    st.markdown("---")
    if st.button("🚀 开始预标注", key="btn_predict", type="primary"):
        if model_path and Path(model_path).exists():
            target = pred_view if pred_view is not None else None
            target_count = len(target) if target is not None else len(ds)
            with st.spinner(f"使用 {task} 模型对 {target_count} 个样本进行预标注..."):
                stats = processor.auto_predict_yolo(
                    ds, model_path, pred_field=pred_field,
                    conf_threshold=conf, task=task, view=target,
                )
            st.success("✅ 预标注完成")
            st.json(stats)
            st.info(
                "💡 预标注结果已写入字段 `" + pred_field + "`。\n\n"
                "- 可在 FiftyOne 中查看预标注效果\n"
                "- 可在「CVAT 同步」中推送预标注到 CVAT 辅助人工标注\n"
                "- 可在「高级功能 → 难例挖掘」中与人工标注对比评估"
            )
        else:
            st.error("请输入有效的模型路径")
