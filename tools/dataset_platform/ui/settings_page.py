"""设置与管理：数据集操作、CVAT 配置、备份恢复。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform.ui.advanced_page import _render_backup
from tools.dataset_platform.ui.components import _get_ds
from tools.dataset_platform.ui.sidebar import _render_cvat_settings, _render_dataset_actions


def _render_settings_page():
    st.header("⚙️ 设置与管理")

    tab_dataset, tab_cvat, tab_backup = st.tabs([
        "📦 数据集操作", "🔗 CVAT 配置", "💾 备份恢复",
    ])

    with tab_dataset:
        _render_dataset_actions(st)

    with tab_cvat:
        _render_cvat_settings(st)

    with tab_backup:
        ds = _get_ds()
        if ds is None:
            st.info("请先选择数据集")
        else:
            _render_backup(ds)
