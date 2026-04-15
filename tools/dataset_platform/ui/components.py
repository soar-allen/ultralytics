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
    轻量级路径浏览器组件。使用 selectbox 替代逐条 button 避免性能问题。

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

    if browse_key not in st.session_state:
        if start_dir and Path(start_dir).is_dir():
            st.session_state[browse_key] = start_dir
        else:
            st.session_state[browse_key] = str(Path.home())

    if pending_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_key)

    path_val = st.text_input(label, key=input_key)

    with st.expander("📂 浏览文件系统", expanded=False):
        current_dir = st.session_state[browse_key]

        nav_col, parent_col = st.columns([5, 1])
        with nav_col:
            new_dir = st.text_input(
                "当前目录", value=current_dir, key=f"{key}_nav",
            )
        with parent_col:
            st.write("")
            if st.button("⬆️", key=f"{key}_parent", help="上级目录"):
                st.session_state[browse_key] = str(Path(current_dir).parent)
                st.rerun()

        if new_dir != current_dir and Path(new_dir).is_dir():
            st.session_state[browse_key] = new_dir
            st.rerun()

        if not Path(current_dir).is_dir():
            st.warning(f"目录不存在: {current_dir}")
            return path_val

        try:
            entries = sorted(os.scandir(current_dir), key=lambda e: e.name)
        except PermissionError:
            st.error("无权限访问该目录")
            return path_val

        dirs = []
        files = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False):
                if file_extensions:
                    if Path(entry.name).suffix.lower() in file_extensions:
                        files.append(entry.name)
                elif mode == "file":
                    files.append(entry.name)

        if dirs:
            dir_options = ["（选择子目录进入）"] + [f"📁 {d}" for d in dirs]
            sel_dir = st.selectbox(
                f"子目录 ({len(dirs)})", dir_options, key=f"{key}_dir_sel",
            )
            if sel_dir != dir_options[0]:
                dir_name = sel_dir.replace("📁 ", "", 1)
                col_enter, col_select = st.columns(2)
                with col_enter:
                    if st.button("📂 进入此目录", key=f"{key}_enter_dir"):
                        st.session_state[browse_key] = os.path.join(current_dir, dir_name)
                        st.rerun()
                with col_select:
                    if mode == "dir":
                        if st.button("✅ 选择此目录", key=f"{key}_pick_subdir"):
                            st.session_state[pending_key] = os.path.join(current_dir, dir_name)
                            st.rerun()

        if mode == "file":
            if files:
                file_options = ["（选择文件）"] + files
                sel_file = st.selectbox(
                    f"文件 ({len(files)})", file_options, key=f"{key}_file_sel",
                )
                if sel_file != file_options[0]:
                    if st.button("✅ 选择此文件", key=f"{key}_pick_file"):
                        st.session_state[pending_key] = os.path.join(current_dir, sel_file)
                        st.rerun()
            else:
                ext_hint = ", ".join(file_extensions) if file_extensions else "所有文件"
                st.caption(f"当前目录无匹配文件 ({ext_hint})")

        if mode == "dir":
            if st.button("✅ 选择当前目录", key=f"{key}_select_cur"):
                st.session_state[pending_key] = current_dir
                st.rerun()

    return path_val
