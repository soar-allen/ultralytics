"""
Streamlit 前端交互界面 - 主入口。

功能区：
  1. 侧边栏：数据集切换、新建/重命名/删除、页面导航
  2. Data Hub：数据集统计、标签管理、Tags 筛选、FiftyOne/CVAT 查看
  3. 数据导入：多格式支持
  4. 数据清洗与处理
  5. 自动预标注（独立大页面）
  6. CVAT 双向同步
  7. 多格式导出（含训练衔接）
  8. 训练管理：配置/启动/历史/回灌
  9. 标注质量检查
  10. 高级功能：难例挖掘、FiftyOne Brain、完整备份

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

from tools.dataset_platform.ui.components import _init_state  # noqa: E402
from tools.dataset_platform.ui.sidebar import _render_sidebar  # noqa: E402

_init_state()

_PAGES = (
    "📊 Data Hub",
    "📥 数据导入",
    "🧹 处理清洗",
    "🤖 自动预标注",
    "🔄 CVAT 同步",
    "📤 数据导出",
    "🏋️ 训练管理",
    "🔬 质量检查",
    "🧠 高级功能",
)


def _render_active_page(page: str):
    """按需导入并渲染当前激活的页面，避免加载全部模块。"""
    if page == _PAGES[0]:
        from tools.dataset_platform.ui.hub import _render_data_hub
        _render_data_hub()
    elif page == _PAGES[1]:
        from tools.dataset_platform.ui.ingestion import _render_ingestion
        _render_ingestion()
    elif page == _PAGES[2]:
        from tools.dataset_platform.ui.processing import _render_processing
        _render_processing()
    elif page == _PAGES[3]:
        from tools.dataset_platform.ui.prediction import _render_auto_predict_page
        _render_auto_predict_page()
    elif page == _PAGES[4]:
        from tools.dataset_platform.ui.cvat_page import _render_cvat_sync
        _render_cvat_sync()
    elif page == _PAGES[5]:
        from tools.dataset_platform.ui.export_page import _render_export
        _render_export()
    elif page == _PAGES[6]:
        from tools.dataset_platform.ui.training_page import _render_training_page
        _render_training_page()
    elif page == _PAGES[7]:
        from tools.dataset_platform.ui.quality_page import _render_quality_page
        _render_quality_page()
    elif page == _PAGES[8]:
        from tools.dataset_platform.ui.advanced_page import _render_advanced
        _render_advanced()


def main():
    if "_toast_msg" in st.session_state:
        st.toast(st.session_state.pop("_toast_msg"), icon="✅")

    _render_sidebar()

    active = st.session_state.get("_active_page", _PAGES[0])
    _render_active_page(active)


if __name__ == "__main__":
    main()
