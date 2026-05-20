"""Workflow-level pages that group existing feature pages without changing behavior."""
from __future__ import annotations

import streamlit as st


def _render_import_and_processing():
    """Render ingestion and cleaning as one workflow page."""
    tab_ingest, tab_process = st.tabs(["📥 导入向导", "🧹 清洗工具箱"])

    with tab_ingest:
        from tools.dataset_platform.ui.ingestion import _render_ingestion

        _render_ingestion()
    with tab_process:
        from tools.dataset_platform.ui.processing import _render_processing

        _render_processing()


def _render_export_and_training():
    """Render export and training as one workflow page."""
    tab_export, tab_train = st.tabs(["📤 数据交付", "🏋️ 训练与模型"])

    with tab_export:
        from tools.dataset_platform.ui.export_page import _render_export

        _render_export()
    with tab_train:
        from tools.dataset_platform.ui.training_page import _render_training_page

        _render_training_page()
