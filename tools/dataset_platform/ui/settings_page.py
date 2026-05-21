"""工作区设置：数据集操作、CVAT 配置、备份恢复。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform.ui.advanced_page import _render_backup
from tools.dataset_platform.ui.components import _get_ds
from tools.dataset_platform.ui.sidebar import _render_cvat_settings, _render_dataset_actions


def _render_settings_page():
    st.header("⚙️ 工作区设置")

    section = st.radio(
        "工作区设置模块",
        ["📦 数据集操作", "🔗 CVAT 配置", "💾 备份恢复"],
        key="settings_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("📦"):
        _render_dataset_actions(st)
    elif section.startswith("🔗"):
        _render_cvat_settings(st)
    else:
        ds = _get_ds()
        if ds is None:
            st.info("请先选择数据集")
        else:
            _render_backup(ds)
