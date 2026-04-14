"""
Streamlit 前端交互界面 - 主入口。

功能区：
  1. 侧边栏：数据集切换、新建/重命名/删除
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

# 延迟导入子模块，确保 set_page_config 先执行
from tools.dataset_platform.ui.components import _init_state  # noqa: E402
from tools.dataset_platform.ui.sidebar import _render_sidebar  # noqa: E402
from tools.dataset_platform.ui.hub import _render_data_hub  # noqa: E402
from tools.dataset_platform.ui.ingestion import _render_ingestion  # noqa: E402
from tools.dataset_platform.ui.processing import _render_processing  # noqa: E402
from tools.dataset_platform.ui.prediction import _render_auto_predict_page  # noqa: E402
from tools.dataset_platform.ui.cvat_page import _render_cvat_sync  # noqa: E402
from tools.dataset_platform.ui.export_page import _render_export  # noqa: E402
from tools.dataset_platform.ui.training_page import _render_training_page  # noqa: E402
from tools.dataset_platform.ui.quality_page import _render_quality_page  # noqa: E402
from tools.dataset_platform.ui.advanced_page import _render_advanced  # noqa: E402

_init_state()


def main():
    if "_toast_msg" in st.session_state:
        st.toast(st.session_state.pop("_toast_msg"), icon="✅")

    _render_sidebar()

    tabs = st.tabs([
        "📊 Data Hub",
        "📥 数据导入",
        "🧹 处理清洗",
        "🤖 自动预标注",
        "🔄 CVAT 同步",
        "📤 数据导出",
        "🏋️ 训练管理",
        "🔬 质量检查",
        "🧠 高级功能",
    ])

    with tabs[0]:
        _render_data_hub()
    with tabs[1]:
        _render_ingestion()
    with tabs[2]:
        _render_processing()
    with tabs[3]:
        _render_auto_predict_page()
    with tabs[4]:
        _render_cvat_sync()
    with tabs[5]:
        _render_export()
    with tabs[6]:
        _render_training_page()
    with tabs[7]:
        _render_quality_page()
    with tabs[8]:
        _render_advanced()


if __name__ == "__main__":
    main()
