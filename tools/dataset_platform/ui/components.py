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
    路径浏览器组件。

    通过导航计数器使 selectbox 的 key 随目录切换而变化，
    确保进入新目录时 selectbox 总是重置为默认值，解决导航卡死问题。

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
    browse_key = f"{key}_cwd"
    pending_key = f"{key}_pending"
    nav_key = f"{key}_nav_seq"

    if browse_key not in st.session_state:
        st.session_state[browse_key] = (
            start_dir if start_dir and Path(start_dir).is_dir()
            else str(Path.home())
        )
    if nav_key not in st.session_state:
        st.session_state[nav_key] = 0

    if pending_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_key)

    path_val = st.text_input(label, key=input_key)

    def _navigate(new_dir: str):
        st.session_state[browse_key] = new_dir
        st.session_state[nav_key] += 1
        st.rerun()

    with st.expander("📂 浏览文件系统", expanded=False):
        current_dir = st.session_state[browse_key]
        seq = st.session_state[nav_key]

        # --- 路径栏 + 上级按钮 ---
        col_path, col_up = st.columns([6, 1])
        with col_path:
            typed_dir = st.text_input(
                "当前目录", value=current_dir, key=f"{key}_path_{seq}",
            )
        with col_up:
            st.write("")
            if st.button("⬆️", key=f"{key}_up_{seq}", help="上级目录"):
                _navigate(str(Path(current_dir).parent))

        if typed_dir != current_dir and Path(typed_dir).is_dir():
            _navigate(typed_dir)

        if not Path(current_dir).is_dir():
            st.warning(f"目录不存在: {current_dir}")
            return path_val

        # --- 列出目录内容 ---
        try:
            entries = sorted(
                os.scandir(current_dir),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            st.error("无权限访问该目录")
            return path_val

        dirs: list[str] = []
        files: list[str] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                dirs.append(entry.name)
            elif mode == "file" and entry.is_file(follow_symlinks=False):
                if file_extensions:
                    if Path(entry.name).suffix.lower() in file_extensions:
                        files.append(entry.name)
                else:
                    files.append(entry.name)

        # --- 目录列表 ---
        if dirs:
            placeholder = "— 选择子目录 —"
            options = [placeholder] + dirs
            chosen = st.selectbox(
                f"📁 子目录 ({len(dirs)})", options,
                key=f"{key}_dsel_{seq}",
            )
            if chosen != placeholder:
                target = os.path.join(current_dir, chosen)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("📂 进入", key=f"{key}_go_{seq}"):
                        _navigate(target)
                if mode == "dir":
                    with c2:
                        if st.button("✅ 选择此目录", key=f"{key}_pickd_{seq}"):
                            st.session_state[pending_key] = target
                            st.rerun()

        # --- 文件列表 ---
        if mode == "file":
            if files:
                placeholder_f = "— 选择文件 —"
                options_f = [placeholder_f] + files
                chosen_f = st.selectbox(
                    f"📄 文件 ({len(files)})", options_f,
                    key=f"{key}_fsel_{seq}",
                )
                if chosen_f != placeholder_f:
                    if st.button("✅ 选择此文件", key=f"{key}_pickf_{seq}"):
                        st.session_state[pending_key] = os.path.join(
                            current_dir, chosen_f
                        )
                        st.rerun()
            else:
                ext_hint = (
                    ", ".join(file_extensions) if file_extensions else "所有文件"
                )
                st.caption(f"当前目录无匹配文件 ({ext_hint})")

        # --- 选择当前目录 ---
        if mode == "dir":
            st.markdown("---")
            if st.button("✅ 选择当前目录", key=f"{key}_pickc_{seq}"):
                st.session_state[pending_key] = current_dir
                st.rerun()

    return path_val
