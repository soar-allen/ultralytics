"""质量与难例：标注质量检查、难例挖掘、FiftyOne Brain。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform.ui.advanced_page import _render_brain, _render_hard_mining
from tools.dataset_platform.ui.components import _get_ds
from tools.dataset_platform.ui.quality_page import _render_quality_checks


def _render_quality_and_hard_samples():
    st.header("🔬 质量与难例")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_quality, tab_hard, tab_brain = st.tabs([
        "🔬 标注质量", "🎯 难例挖掘", "🧬 FiftyOne Brain",
    ])

    with tab_quality:
        _render_quality_checks(ds)
    with tab_hard:
        _render_hard_mining(ds)
    with tab_brain:
        _render_brain(ds)
