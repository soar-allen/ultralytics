"""Workflow-level pages that group existing feature pages without changing behavior."""
from __future__ import annotations

import streamlit as st


def _render_import_and_processing():
    """Render ingestion and cleaning as one workflow page."""
    section = st.radio(
        "数据准备模块",
        ["📥 导入向导", "🧹 清洗工具箱"],
        key="prepare_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("📥"):
        from tools.dataset_platform.ui.ingestion import _render_ingestion

        _render_ingestion()
    else:
        from tools.dataset_platform.ui.processing import _render_processing

        _render_processing()


def _render_export_and_training():
    """Render export and training as one workflow page."""
    section = st.radio(
        "导出训练模块",
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
