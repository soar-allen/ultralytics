"""公共 UI 组件：路径浏览器、状态管理等。"""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform.config import CONFIG


# ===================================================================
# Session State 初始化
# ===================================================================

def _init_state():
    defaults = {
        "current_dataset": None,
        "fo_session": None,
        "fo_port": CONFIG.fiftyone_port,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_ds():
    """获取当前选中的 FiftyOne Dataset 对象。"""
    name = st.session_state.current_dataset
    if name and name in dm.list_datasets():
        return dm.load_dataset(name)
    return None


# ===================================================================
# 通用 UI 组件：路径浏览器
# ===================================================================

def _path_browser(
    label: str,
    key: str,
    mode: str = "dir",
    start_dir: str = "",
    file_extensions: tuple | None = None,
) -> str:
    """
    可视化路径浏览器组件，支持手动输入和点击浏览。

    Args:
        label: 显示标签
        key: Streamlit widget key 前缀
        mode: "dir" 选择目录, "file" 选择文件
        start_dir: 浏览的起始目录
        file_extensions: 文件模式下的扩展名过滤 (如 (".pt", ".pth"))

    Returns:
        选中的路径字符串
    """
    input_key = f"{key}_input"
    browse_key = f"{key}_browse_dir"
    pending_key = f"{key}_pending"

    # 初始化浏览目录
    if browse_key not in st.session_state:
        if start_dir and Path(start_dir).is_dir():
            st.session_state[browse_key] = start_dir
        else:
            st.session_state[browse_key] = str(Path.home())

    # 把上一轮按钮写入的 pending 值应用到 input_key（必须在 widget 渲染前）
    if pending_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_key)

    # 手动输入框（顶层，用户可直接粘贴路径）
    path_val = st.text_input(label, key=input_key)

    with st.expander("📂 浏览文件系统", expanded=False):
        current_dir = st.session_state[browse_key]

        # 导航栏
        nav_col1, nav_col2 = st.columns([5, 1])
        with nav_col1:
            new_dir = st.text_input(
                "当前目录（可直接输入路径跳转）",
                value=current_dir,
                key=f"{key}_nav_{hash(current_dir) % 100000}",
            )
        with nav_col2:
            st.write("")
            if st.button("⬆️ 上级", key=f"{key}_parent"):
                st.session_state[browse_key] = str(Path(current_dir).parent)
                st.rerun()

        if new_dir != current_dir and Path(new_dir).is_dir():
            st.session_state[browse_key] = new_dir
            st.rerun()

        if not Path(current_dir).is_dir():
            st.warning(f"目录不存在: {current_dir}")
            return path_val

        try:
            entries = sorted(os.listdir(current_dir))
        except PermissionError:
            st.error("无权限访问该目录")
            return path_val

        dirs = []
        files = []
        for entry in entries:
            if entry.startswith("."):
                continue
            full = os.path.join(current_dir, entry)
            if os.path.isdir(full):
                dirs.append(entry)
            elif os.path.isfile(full):
                if file_extensions:
                    if Path(entry).suffix.lower() in file_extensions:
                        files.append(entry)
                else:
                    files.append(entry)

        dir_hash = hash(current_dir) % 100000

        if dirs:
            st.caption(f"📁 子目录 ({len(dirs)})")
            dir_container = st.container(height=250)
            with dir_container:
                for i, d in enumerate(dirs):
                    if st.button(
                        f"📁 {d}",
                        key=f"{key}_d{i}_{dir_hash}",
                        use_container_width=True,
                    ):
                        st.session_state[browse_key] = os.path.join(current_dir, d)
                        st.rerun()

        if mode == "file":
            if files:
                st.caption(f"📄 文件 ({len(files)})")
                file_container = st.container(height=250)
                with file_container:
                    for i, f in enumerate(files):
                        if st.button(
                            f"📄 {f}",
                            key=f"{key}_f{i}_{dir_hash}",
                            use_container_width=True,
                        ):
                            st.session_state[pending_key] = os.path.join(current_dir, f)
                            st.rerun()
            else:
                ext_hint = ", ".join(file_extensions) if file_extensions else "所有文件"
                st.caption(f"当前目录无匹配文件 ({ext_hint})")

        if mode == "dir":
            if st.button("✅ 选择当前目录", key=f"{key}_select_dir"):
                st.session_state[pending_key] = current_dir
                st.rerun()

        st.caption(f"📍 {current_dir}")

    return path_val
