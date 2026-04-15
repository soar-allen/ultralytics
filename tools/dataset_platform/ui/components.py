"""公共 UI 组件：路径浏览器、状态管理、per-rerun 缓存。"""
from __future__ import annotations
import os
from pathlib import Path
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform.config import CONFIG


# ===================================================================
# Per-rerun 缓存
# ===================================================================
# Streamlit 每次交互重跑整个脚本，同一次 rerun 内多处调用
# _get_ds() / _get_info() 会重复触发 FiftyOne 查询。
# 用 session_state 中的序号 + 模块级 dict 实现单次 rerun 内的结果复用。

_rerun_store: dict = {}
_RERUN_SEQ_KEY = "__rerun_seq__"


def _rerun_seq() -> int:
    if _RERUN_SEQ_KEY not in st.session_state:
        st.session_state[_RERUN_SEQ_KEY] = 0
    return st.session_state[_RERUN_SEQ_KEY]


def _bump_rerun_seq():
    """在每次 rerun 开始时递增序号，使旧缓存失效。"""
    st.session_state[_RERUN_SEQ_KEY] = st.session_state.get(_RERUN_SEQ_KEY, 0) + 1
    _rerun_store.clear()


def _rc_get(key: str):
    """读取 per-rerun 缓存，返回 (hit, value)。"""
    seq = _rerun_seq()
    full = (seq, key)
    if full in _rerun_store:
        return True, _rerun_store[full]
    return False, None


def _rc_set(key: str, value):
    seq = _rerun_seq()
    # 清理上一次 rerun 的残留
    stale = [k for k in _rerun_store if k[0] != seq]
    for k in stale:
        del _rerun_store[k]
    _rerun_store[(seq, key)] = value
    return value


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
    _bump_rerun_seq()


def _get_ds():
    """获取当前选中的 FiftyOne Dataset 对象（per-rerun 缓存）。"""
    hit, val = _rc_get("ds")
    if hit:
        return val
    name = st.session_state.current_dataset
    if name and name in _list_datasets_cached():
        ds = dm.load_dataset(name)
        return _rc_set("ds", ds)
    return _rc_set("ds", None)


def _list_datasets_cached() -> list[str]:
    """list_datasets 的 per-rerun 缓存。"""
    hit, val = _rc_get("ds_list")
    if hit:
        return val
    return _rc_set("ds_list", dm.list_datasets())


def _get_info(ds):
    """get_dataset_info 的 per-rerun 缓存。"""
    if ds is None:
        return None
    hit, val = _rc_get(f"info_{ds.name}")
    if hit:
        return val
    return _rc_set(f"info_{ds.name}", dm.get_dataset_info(ds))


# ===================================================================
# 通用 UI 组件：路径浏览器
# ===================================================================

import shutil
import subprocess
import sys


def _open_native_dialog(
    mode: str = "dir",
    title: str = "选择路径",
    start_dir: str = "",
    file_extensions: tuple | None = None,
) -> str | None:
    """
    打开系统原生文件对话框。
    优先级: tkinter (跨平台) → zenity (Linux GTK) → None
    """
    # --- 1) tkinter: Windows / macOS / Linux 均可用 ---
    result = _tkinter_dialog(mode, title, start_dir, file_extensions)
    if result is not None:
        return result

    # --- 2) zenity: Linux GTK 桌面环境 ---
    if sys.platform != "win32" and shutil.which("zenity"):
        return _zenity_dialog(mode, title, start_dir, file_extensions)

    return None


def _tkinter_dialog(
    mode: str, title: str, start_dir: str, file_extensions: tuple | None,
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", True)
        root.focus_force()
        if mode == "dir":
            path = filedialog.askdirectory(
                parent=root, initialdir=start_dir or None, title=title,
            )
        else:
            filetypes = []
            if file_extensions:
                exts = " ".join(f"*{e}" for e in file_extensions)
                filetypes.append(("匹配文件", exts))
            filetypes.append(("所有文件", "*.*"))
            path = filedialog.askopenfilename(
                parent=root, initialdir=start_dir or None,
                title=title, filetypes=filetypes,
            )
        root.destroy()
        return path if path else None
    except Exception:
        return None


def _zenity_dialog(
    mode: str, title: str, start_dir: str, file_extensions: tuple | None,
) -> str | None:
    cmd = ["zenity", "--file-selection", "--title", title]
    if mode == "dir":
        cmd.append("--directory")
    if start_dir:
        trail = "/" if mode == "dir" else ""
        cmd.extend(["--filename", start_dir + trail])
    if file_extensions and mode == "file":
        exts = " ".join(f"*{e}" for e in file_extensions)
        cmd.extend(["--file-filter", f"匹配文件 | {exts}"])
        cmd.extend(["--file-filter", "所有文件 | *"])
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _path_browser(
    label: str,
    key: str,
    mode: str = "dir",
    start_dir: str = "",
    file_extensions: tuple | None = None,
) -> str:
    """
    路径浏览器组件。

    优先使用系统原生文件对话框（tkinter/zenity），
    同时保留手动输入和 Web 端备用浏览器。
    Web 浏览器中 selectbox 选择即触发（无需额外点击按钮）。

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
    pending_key = f"{key}_pending"

    if pending_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_key)

    # --- 手动输入 + 浏览按钮 ---
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        path_val = st.text_input(label, key=input_key)
    with col_btn:
        st.write("")
        browse_clicked = st.button("📂 浏览", key=f"{key}_browse")

    if browse_clicked:
        initial = path_val or start_dir or str(Path.home())
        if mode == "dir" and Path(initial).is_file():
            initial = str(Path(initial).parent)

        chosen = _open_native_dialog(
            mode=mode, title=label,
            start_dir=initial, file_extensions=file_extensions,
        )
        if chosen:
            st.session_state[pending_key] = chosen
            st.rerun()
        else:
            # 原生对话框不可用或用户取消 → 展开 Web 浏览器
            st.session_state[f"{key}_show_fallback"] = True
            st.rerun()

    # --- Web 备用浏览器 ---
    if st.session_state.get(f"{key}_show_fallback", False):
        _web_path_browser(key, mode, start_dir, file_extensions, pending_key)

    return path_val


def _web_path_browser(
    key: str,
    mode: str,
    start_dir: str,
    file_extensions: tuple | None,
    pending_key: str,
):
    """
    纯 Web selectbox 备用路径浏览器。
    选择即触发：选中目录自动进入，选中文件自动确认。
    """
    browse_key = f"{key}_cwd"
    nav_key = f"{key}_nav_seq"

    if browse_key not in st.session_state:
        st.session_state[browse_key] = (
            start_dir if start_dir and Path(start_dir).is_dir()
            else str(Path.home())
        )
    if nav_key not in st.session_state:
        st.session_state[nav_key] = 0

    def _navigate(new_dir: str):
        st.session_state[browse_key] = new_dir
        st.session_state[nav_key] += 1
        st.rerun()

    with st.expander("📂 Web 文件浏览器", expanded=True):
        current_dir = st.session_state[browse_key]
        seq = st.session_state[nav_key]

        # --- 路径栏 + 上级 ---
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
            return

        # --- 选择当前目录（dir 模式，置顶显示） ---
        if mode == "dir":
            if st.button("✅ 选择当前目录", key=f"{key}_pickc_{seq}",
                         type="primary", use_container_width=True):
                st.session_state[pending_key] = current_dir
                st.rerun()

        # --- 列出内容 ---
        try:
            entries = sorted(
                os.scandir(current_dir),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except PermissionError:
            st.error("无权限访问该目录")
            return

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

        # --- 目录列表：选择即进入 ---
        if dirs:
            placeholder = "— 点击选择子目录（自动进入）—"
            options = [placeholder] + dirs
            chosen = st.selectbox(
                f"📁 子目录 ({len(dirs)})", options,
                key=f"{key}_dsel_{seq}",
            )
            if chosen != placeholder:
                _navigate(os.path.join(current_dir, chosen))

        # --- 文件列表：选择即确认 ---
        if mode == "file":
            if files:
                placeholder_f = "— 点击选择文件（自动确认）—"
                options_f = [placeholder_f] + files
                chosen_f = st.selectbox(
                    f"📄 文件 ({len(files)})", options_f,
                    key=f"{key}_fsel_{seq}",
                )
                if chosen_f != placeholder_f:
                    st.session_state[pending_key] = os.path.join(
                        current_dir, chosen_f,
                    )
                    st.rerun()
            else:
                ext_hint = (
                    ", ".join(file_extensions) if file_extensions else "所有文件"
                )
                st.caption(f"当前目录无匹配文件 ({ext_hint})")
