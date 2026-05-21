"""
Streamlit 前端交互界面 - 主入口。

功能区：
  1. 侧边栏：数据集切换、工作流导航、设置入口
  2. 工作区摘要：当前数据集、样本数、标签字段、Tags、FiftyOne 快捷入口
  3. 数据总览：数据集统计、Tags 筛选
  4. 数据管理：多格式导入、清洗、去重、多边形处理、字段管理、标签管理、健康检查
  5. 数据集标注：自动预标注、CVAT 推送拉取、运行管理、任务状态与审核
  6. 模型训练与导出：多格式导出、训练配置、训练进度、模型转换、回灌
  7. 数据集分析：类别平衡、面积分析、空标注检测、难例挖掘、FiftyOne Brain
  8. 工作区设置：数据集操作、CVAT 配置、完整备份

启动方式：
    streamlit run tools/dataset_platform/ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(
    page_title="CV 数据集管理平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from tools.dataset_platform.ui.components import _init_state, _render_workspace_header  # noqa: E402
from tools.dataset_platform.ui.navigation import DEFAULT_PAGE_KEY, load_renderer, page_by_key, resolve_page_key  # noqa: E402
from tools.dataset_platform.ui.sidebar import _render_sidebar  # noqa: E402

_init_state()

def _render_active_page(page_key: str):
    """按需导入并渲染当前激活的页面，避免加载全部模块。"""
    renderer = load_renderer(page_by_key(page_key))
    renderer()


def _auto_start_fiftyone():
    """平台启动时自动启动 FiftyOne App（仅首次执行）。"""
    if st.session_state.get("_fo_auto_started"):
        return
    ds_name = st.session_state.get("current_dataset")
    if not ds_name:
        return
    try:
        from tools.dataset_platform import data_manager as dm
        ds = dm.load_dataset(ds_name)
        port = st.session_state.get("fo_port", 5151)
        session = dm.ensure_app(ds, port=port)
        st.session_state.fo_session = session
        st.session_state["_fo_auto_started"] = True
    except Exception:
        pass


def main():
    if "_toast_msg" in st.session_state:
        st.toast(st.session_state.pop("_toast_msg"), icon="✅")

    _render_sidebar()
    _auto_start_fiftyone()
    _render_workspace_header()

    active = resolve_page_key(st.session_state.get("_active_page", DEFAULT_PAGE_KEY))
    st.session_state["_active_page"] = active
    _render_active_page(active)


if __name__ == "__main__":
    main()
