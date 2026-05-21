"""Workflow-level pages that group existing feature pages without changing behavior."""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform.ui.components import _get_ds, _get_info


def _render_import_and_processing():
    """Render ingestion and cleaning as one workflow page."""
    section = st.radio(
        "数据管理模块",
        ["📥 导入向导", "🧹 清洗工具箱", "🏷️ 标签管理", "🩺 健康检查"],
        key="prepare_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("📥"):
        from tools.dataset_platform.ui.ingestion import _render_ingestion

        _render_ingestion()
    elif section.startswith("🧹"):
        from tools.dataset_platform.ui.processing import _render_processing

        _render_processing()
    else:
        ds = _get_ds()
        if ds is None:
            st.info("请先选择数据集，或在「工作区设置」中新建数据集")
            return

        if section.startswith("🏷️"):
            from tools.dataset_platform.ui.hub import _render_label_management

            _render_label_management(ds)
        else:
            from tools.dataset_platform.ui.hub import _render_hub_health

            _render_hub_health(ds, _get_info(ds))


def _render_dataset_annotation():
    """Render automatic prediction and CVAT sync as one annotation workflow page."""
    section = st.radio(
        "数据集标注模块",
        ["🤖 自动预标注", "🔄 CVAT 标注同步"],
        key="annotation_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("🤖"):
        from tools.dataset_platform.ui.prediction import _render_auto_predict_page

        _render_auto_predict_page()
    else:
        from tools.dataset_platform.ui.cvat_page import _render_cvat_sync

        _render_cvat_sync()


def _render_export_and_training():
    """Render export and training as one workflow page."""
    section = st.radio(
        "模型训练与导出模块",
        ["📤 数据交付", "🏋️ 训练与模型"],
        key="export_train_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("📤"):
        from tools.dataset_platform.ui.export_page import _render_export

        _render_export()
    else:
        from tools.dataset_platform.ui.training_page import _render_training_page

        _render_training_page()
