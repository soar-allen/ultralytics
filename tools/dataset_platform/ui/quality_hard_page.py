"""数据集分析：标注质量检查、难例挖掘、FiftyOne Brain。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform.ui.advanced_page import _render_brain, _render_hard_mining
from tools.dataset_platform.ui.components import _get_ds
from tools.dataset_platform.ui.quality_page import _render_quality_checks


def _render_quality_and_hard_samples():
    st.header("🔬 数据集分析")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    section = st.radio(
        "数据集分析模块",
        ["🔬 标注质量", "🎯 难例挖掘", "🧬 FiftyOne Brain"],
        key="quality_hard_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("🔬"):
        _render_quality_checks(ds)
    elif section.startswith("🎯"):
        _render_hard_mining(ds)
    else:
        _render_brain(ds)
