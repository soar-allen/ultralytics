"""公共 UI 组件：路径浏览器、状态管理、per-rerun 缓存。"""
from __future__ import annotations
from pathlib import Path
from typing import Callable
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
# Session 级 UI 统计缓存
# ===================================================================

_UI_CACHE_EPOCH_KEY = "__ui_stats_cache_epoch__"
_UI_STATS_CACHE_KEY = "__ui_stats_cache__"


def _ui_cache_epoch() -> int:
    if _UI_CACHE_EPOCH_KEY not in st.session_state:
        st.session_state[_UI_CACHE_EPOCH_KEY] = 0
    return st.session_state[_UI_CACHE_EPOCH_KEY]


def _invalidate_ui_stats_cache():
    """Invalidate cached UI stats after data-changing operations."""
    st.session_state[_UI_CACHE_EPOCH_KEY] = st.session_state.get(_UI_CACHE_EPOCH_KEY, 0) + 1
    st.session_state[_UI_STATS_CACHE_KEY] = {}


def _ui_cached(ds, key: str, compute: Callable[[], object]):
    """Cache heavier UI-only stats across reruns for the current dataset."""
    cache = st.session_state.setdefault(_UI_STATS_CACHE_KEY, {})
    full_key = (ds.name if ds is not None else "", _ui_cache_epoch(), key)
    if full_key not in cache:
        cache[full_key] = compute()
    return cache[full_key]


def _cached_tags(ds) -> list[str]:
    """Return sorted sample tags using session-level cache."""
    if ds is None:
        return []
    return _ui_cached(ds, "tags", lambda: sorted(ds.distinct("tags")))


def _cached_label_classes(ds, field_name: str) -> list[str]:
    """Return classes for a label field using session-level cache."""
    if ds is None or not field_name:
        return []
    return _ui_cached(ds, f"classes:{field_name}", lambda: dm.get_label_classes(ds, field_name))


def _cached_label_stats(ds, field_name: str) -> dict:
    """Return per-class counts for a label field using session-level cache."""
    if ds is None or not field_name:
        return {}
    return _ui_cached(ds, f"stats:{field_name}", lambda: dm.get_label_stats(ds, field_name))


def _default_label_field(ds, label_fields: list[str] | tuple[str, ...], preferred: str = "ground_truth", label_type: str | None = None) -> str:
    """Choose a stable default label field with optional type filtering."""
    fields = list(label_fields)
    if label_type:
        typed = [f for f in fields if dm.get_field_label_type(ds, f) == label_type]
        fields = typed or fields
    if preferred in fields:
        return preferred
    return fields[0] if fields else preferred


# ===================================================================
# 通用 UI 组件：工作区摘要、操作摘要、危险操作确认
# ===================================================================

def _format_short_list(values: list[str] | tuple[str, ...], limit: int = 6) -> str:
    """Format a short comma-separated preview for dense UI captions."""
    items = [str(v) for v in values if v]
    if not items:
        return "无"
    shown = items[:limit]
    suffix = f" 等 {len(items)} 项" if len(items) > limit else ""
    return ", ".join(shown) + suffix


def _render_workspace_header():
    """Render the persistent dataset context at the top of every page."""
    ds = _get_ds()

    st.title("CV 数据集管理平台")
    if ds is None:
        st.info("请先选择数据集，或在「工作区设置」中新建数据集")
        return
    st.markdown("---")


def _action_summary(title: str, items: dict[str, object], expanded: bool = True):
    """Render a compact pre-action summary from label/value pairs."""
    with st.expander(title, expanded=expanded):
        for label, value in items.items():
            if isinstance(value, (list, tuple, set)):
                value = _format_short_list([str(v) for v in value])
            elif value is None or value == "":
                value = "未设置"
            st.markdown(f"**{label}**: {value}")


def _confirm_danger_action(label: str, expected: str, key: str) -> bool:
    """Render a typed confirmation field for irreversible actions."""
    st.error(label)
    confirm = st.text_input(f"输入 `{expected}` 确认", key=key)
    return confirm == expected


def _result_panel(result: dict | list | str, title: str = "执行结果"):
    """Render a consistent result panel for operation outputs."""
    with st.expander(title, expanded=True):
        if isinstance(result, (dict, list)):
            st.json(result)
        else:
            st.write(result)


def _primary_action_section(
    button_label: str,
    key: str,
    summary: dict[str, object] | None = None,
    disabled: bool = False,
    disabled_reason: str | None = None,
    preview_view=None,
    preview_key: str | None = None,
    preview_label: str = "👁️ 展示筛选结果",
) -> bool:
    """Render a consistent pre-action summary and primary button."""
    if summary:
        _action_summary("执行前确认", summary, expanded=True)
    if disabled and disabled_reason:
        st.warning(disabled_reason)
    if preview_view is not None:
        col_run, col_preview = st.columns([2, 1])
        with col_run:
            clicked = st.button(button_label, key=key, type="primary", disabled=disabled, use_container_width=True)
        with col_preview:
            _show_view_in_fiftyone_button(
                preview_view,
                key=preview_key or f"{key}_preview_fo",
                label=preview_label,
            )
        return clicked
    return st.button(button_label, key=key, type="primary", disabled=disabled, use_container_width=True)


def _show_view_in_fiftyone_button(
    view,
    *,
    key: str,
    label: str = "👁️ 展示筛选结果",
    disabled: bool = False,
    use_container_width: bool = True,
) -> bool:
    """Render a button that sends the provided dataset/view to the FiftyOne App."""
    clicked = st.button(label, key=key, disabled=disabled, use_container_width=use_container_width)
    if not clicked:
        return False

    try:
        count = len(view)
        if count == 0:
            st.warning("当前筛选结果为空，无法在 FiftyOne 中展示")
            return True

        dataset = getattr(view, "_dataset", view)
        display_view = view.view() if dataset is view and hasattr(view, "view") else view
        port = st.session_state.get("fo_port", CONFIG.fiftyone_port)
        dm.ensure_app(dataset, port=port)
        dm.set_session_view(display_view)
        st.success(f"已在 FiftyOne 中展示 {count} 个样本")
        st.link_button("打开 FiftyOne", f"http://localhost:{port}")
    except Exception as e:
        st.error(f"展示筛选结果失败: {e}")
    return True


def _render_scope_selector(
    ds,
    key_prefix: str,
    *,
    label: str = "操作范围",
    include_unlabeled: bool = False,
    unlabeled_view=None,
    label_field: str | None = None,
    default: str = "整个数据集",
    show_count: bool = True,
):
    """Render a reusable dataset scope selector and return (view, mode, selected_tags)."""
    options = ["整个数据集"]
    if include_unlabeled:
        options.append("仅无标注样本")
    options.append("按 Tags 筛选")
    index = options.index(default) if default in options else 0

    mode = st.radio(label, options, index=index, key=f"{key_prefix}_scope", horizontal=True)
    selected_tags: list[str] = []
    view = ds

    if mode == "仅无标注样本":
        if unlabeled_view is not None:
            view = unlabeled_view
        elif label_field:
            view = dm.get_unlabeled_view(ds, label_field)
        else:
            view = ds
    elif mode == "按 Tags 筛选":
        tags = _cached_tags(ds)
        if tags:
            selected_tags = st.multiselect("选择 Tags", tags, key=f"{key_prefix}_tags")
            if selected_tags:
                view = ds.match_tags(selected_tags)
        else:
            st.info("当前数据集没有 Tags")

    if show_count:
        st.caption(f"当前范围: **{len(view)}** 个样本")
    else:
        st.caption("当前范围会在执行时统计样本数")
    return view, mode, selected_tags


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
    Linux 优先使用 zenity，否则 tkinter。用户取消选择时不再 fallback 到另一个对话框。
    """
    # normalize alias values
    if mode == "directory":
        mode = "dir"

    if sys.platform == "linux":
        if shutil.which("zenity"):
            return _zenity_dialog(mode, title, start_dir, file_extensions)
        return _tkinter_dialog(mode, title, start_dir, file_extensions)

    # Windows / macOS: tkinter only. A cancel should simply return None.
    return _tkinter_dialog(mode, title, start_dir, file_extensions)


def _tkinter_dialog(
    mode: str, title: str, start_dir: str, file_extensions: tuple | None,
) -> str | None:
    # normalize alias values
    if mode == "directory":
        mode = "dir"

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
    # normalize alias values
    if mode == "directory":
        mode = "dir"

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
    default_value: str | None = None,
) -> str:
    """
    路径浏览器组件。

    使用系统原生文件对话框（tkinter/zenity），并保留手动输入。

    Args:
        label: 显示标签
        key: Streamlit widget key 前缀
        mode: "dir" 选择目录, "file" 选择文件
        start_dir: 浏览的起始目录
        file_extensions: 文件模式下的扩展名过滤 (如 (".pt", ".pth"))
        default_value: 输入框的默认路径

    Returns:
        选中的路径字符串
    """
    # normalize alias values (accept 'directory' as synonym for 'dir')
    if mode == "directory":
        mode = "dir"

    input_key = f"{key}_input"
    pending_key = f"{key}_pending_path"

    if input_key not in st.session_state and default_value:
        st.session_state[input_key] = default_value
    if pending_key in st.session_state:
        st.session_state[input_key] = st.session_state.pop(pending_key)

    current_val = st.session_state.get(input_key, "")

    # --- 手动输入 + 文件夹图标按钮 ---
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        path_val = st.text_input(label, key=input_key)
    with col_btn:
        st.write("")
        browse_clicked = st.button(
            "📂",
            key=f"{key}_browse",
            help="选择文件夹" if mode == "dir" else "选择文件",
            use_container_width=True,
        )

    if browse_clicked:
        initial = path_val or current_val or start_dir or str(Path.home())
        if mode == "dir" and Path(initial).is_file():
            initial = str(Path(initial).parent)

        chosen = _open_native_dialog(
            mode=mode, title=label,
            start_dir=initial, file_extensions=file_extensions,
        )
        if chosen:
            st.session_state[pending_key] = chosen
            st.rerun()

    return path_val
