"""
Streamlit 前端交互界面。

功能区：
  1. 侧边栏：数据集切换、新建/重命名/删除
  2. Data Hub：FiftyOne App 嵌入查看
  3. 数据导入：多格式支持
  4. 数据清洗与处理
  5. CVAT 双向同步
  6. 多格式导出
  7. 高级功能：难例挖掘、版本控制

启动方式：
    streamlit run tools/dataset_platform/ui.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import ingestors
from tools.dataset_platform import processor
from tools.dataset_platform import cvat_sync
from tools.dataset_platform import exporter
from tools.dataset_platform import advanced
from tools.dataset_platform.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="CV 数据集管理平台",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


_init_state()


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


# ===================================================================
# 侧边栏：数据集管理
# ===================================================================

def _render_sidebar():
    st.sidebar.title("📦 数据集管理")

    datasets = dm.list_datasets()

    if datasets:
        idx = 0
        if st.session_state.current_dataset in datasets:
            idx = datasets.index(st.session_state.current_dataset)
        selected = st.sidebar.selectbox(
            "选择数据集", datasets, index=idx, key="ds_selector",
        )
        if selected != st.session_state.current_dataset:
            st.session_state.current_dataset = selected
            # 切换数据集时同步更新 FiftyOne Session
            ds = _get_ds()
            if ds:
                dm.switch_session_dataset(ds)
    else:
        st.sidebar.info("暂无数据集，请先创建")

    st.sidebar.markdown("---")
    st.sidebar.subheader("数据集操作")

    with st.sidebar.expander("➕ 新建数据集"):
        new_name = st.text_input("数据集名称", key="new_ds_name")
        if st.button("创建", key="btn_create_ds"):
            if new_name:
                try:
                    dm.create_dataset(new_name)
                    st.session_state.current_dataset = new_name
                    st.success(f"✅ 已创建: {new_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"创建失败: {e}")
            else:
                st.warning("请输入名称")

    with st.sidebar.expander("✏️ 重命名数据集"):
        if st.session_state.current_dataset:
            rename_to = st.text_input(
                "新名称", value=st.session_state.current_dataset, key="rename_input"
            )
            if st.button("确认重命名", key="btn_rename"):
                if rename_to and rename_to != st.session_state.current_dataset:
                    try:
                        dm.rename_dataset(st.session_state.current_dataset, rename_to)
                        st.session_state.current_dataset = rename_to
                        st.success(f"✅ 已重命名为: {rename_to}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"重命名失败: {e}")

    with st.sidebar.expander("🗑️ 删除数据集"):
        if st.session_state.current_dataset:
            st.warning(f"即将删除: **{st.session_state.current_dataset}**")
            confirm = st.text_input("输入数据集名称确认删除", key="del_confirm")
            if st.button("确认删除", key="btn_delete", type="primary"):
                if confirm == st.session_state.current_dataset:
                    deleted_name = st.session_state.current_dataset

                    remaining = [n for n in dm.list_datasets() if n != deleted_name]
                    if remaining:
                        fallback_ds = dm.load_dataset(remaining[0])
                        dm.switch_session_dataset(fallback_ds)
                    else:
                        dm.close_app()
                        st.session_state.fo_session = None

                    dm.delete_dataset(deleted_name)
                    st.session_state.current_dataset = remaining[0] if remaining else None
                    st.session_state["_toast_msg"] = f"数据集 '{deleted_name}' 已成功删除"
                    st.rerun()
                else:
                    st.error("名称不匹配，取消删除")

    st.sidebar.markdown("---")
    st.sidebar.subheader("CVAT 配置")
    CONFIG.cvat.url = st.sidebar.text_input("CVAT URL", value=CONFIG.cvat.url)
    CONFIG.cvat.username = st.sidebar.text_input("用户名", value=CONFIG.cvat.username)
    CONFIG.cvat.password = st.sidebar.text_input("密码", value=CONFIG.cvat.password, type="password")
    CONFIG.cvat.organization = st.sidebar.text_input(
        "Organization（可选）", value=CONFIG.cvat.organization,
        help="CVAT 组织 slug（如 my-team），留空则使用个人空间。推送和拉取都会指向该组织。",
    )

    if st.sidebar.button("🔗 测试 CVAT 连接", key="btn_test_cvat"):
        with st.sidebar.status("测试连接中...", expanded=True) as status:
            result = cvat_sync.test_connection()
            if result["connected"]:
                status.update(label="连接成功", state="complete")
                st.sidebar.success(f"✅ CVAT 已连接")
                info_lines = [f"- 服务器版本: {result['server_version'] or '未知'}"]
                if result["projects_count"] is not None:
                    info_lines.append(f"- 项目数: {result['projects_count']}")
                if result["tasks_count"] is not None:
                    info_lines.append(f"- 任务数: {result['tasks_count']}")
                st.sidebar.markdown("\n".join(info_lines))
            else:
                status.update(label="连接失败", state="error")
                st.sidebar.error(f"❌ {result['error']}")
                if result["solution"]:
                    st.sidebar.warning(f"💡 {result['solution']}")

    st.sidebar.markdown("---")
    ds = _get_ds()
    if ds:
        info = dm.get_dataset_info(ds)
        st.sidebar.metric("样本数", info["num_samples"])
        st.sidebar.caption(f"标签字段: {', '.join(info['label_fields']) or '无'}")

        all_classes = []
        for lf in info["label_fields"]:
            all_classes.extend(dm.get_label_classes(ds, lf))
        unique_classes = sorted(set(all_classes))
        st.sidebar.caption(f"标注类别: {', '.join(unique_classes) or '无'}")
        if info["tags"]:
            st.sidebar.caption(f"样本标签: {', '.join(info['tags'][:10])}")


# ===================================================================
# Tab 1: Data Hub - 查看
# ===================================================================

def _render_data_hub():
    st.header("🔍 Data Hub - 数据集查看")
    ds = _get_ds()
    if ds is None:
        st.info("请先在侧边栏选择或创建数据集")
        return

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("数据集信息")
        info = dm.get_dataset_info(ds)
        st.json(info)

        if info["label_fields"]:
            selected_field = st.selectbox("选择标签字段查看统计", info["label_fields"])
            if selected_field:
                stats = dm.get_label_stats(ds, selected_field)
                if stats:
                    import pandas as pd
                    df = pd.DataFrame(
                        list(stats.items()), columns=["类别", "数量"]
                    ).sort_values("数量", ascending=False)
                    st.dataframe(df, use_container_width=True)
                    st.bar_chart(df.set_index("类别"))

    with col1:
        st.subheader("FiftyOne 可视化")
        port = st.session_state.fo_port
        if st.button("🚀 启动 / 刷新 FiftyOne App", key="btn_launch_fo"):
            try:
                session = dm.launch_app(ds, port=port)
                st.session_state.fo_session = session
                st.success(f"FiftyOne App 已启动 (端口 {port})")
            except Exception as e:
                st.error(f"启动失败: {e}")

        st.components.v1.iframe(
            f"http://localhost:{port}",
            height=700,
            scrolling=True,
        )

        with st.expander("🔧 图片路径诊断与修复", expanded=False):
            st.caption("如果 FiftyOne 中图片无法显示，可能是数据库中的文件路径与磁盘不一致")
            if st.button("🔍 诊断路径", key="btn_diag_path"):
                with st.spinner("扫描中..."):
                    diag = dm.diagnose_filepaths(ds)
                st.session_state["_path_diag"] = diag

            diag = st.session_state.get("_path_diag")
            if diag:
                c1, c2, c3 = st.columns(3)
                c1.metric("总样本数", diag["total"])
                c2.metric("路径缺失", diag["missing"], delta=None if diag["missing"] == 0 else f"-{diag['missing']}", delta_color="inverse")
                c3.metric("符号链接", diag["symlinks"])

                if diag["missing"] > 0:
                    st.error(f"发现 {diag['missing']} 个文件不存在！图片将无法在 FiftyOne 中显示。")
                    st.caption("缺失路径示例：")
                    for item in diag["missing_samples"][:5]:
                        st.code(item["filepath"], language=None)

                    st.markdown("**批量修复路径前缀**")
                    st.caption("如果图片目录发生了迁移，可以将旧路径前缀替换为新路径")
                    old_pfx = st.text_input("旧路径前缀", key="fix_old_pfx",
                                            placeholder="/old/path/to/images")
                    new_pfx = st.text_input("新路径前缀", key="fix_new_pfx",
                                            placeholder="/new/path/to/images")
                    if st.button("🔧 执行路径修复", key="btn_fix_path"):
                        if old_pfx and new_pfx:
                            with st.spinner("修复中..."):
                                fixed = dm.fix_filepaths(ds, old_pfx, new_pfx)
                            st.session_state["_toast_msg"] = f"已修复 {fixed} 个样本的路径"
                            if "_path_diag" in st.session_state:
                                del st.session_state["_path_diag"]
                            st.rerun()
                        else:
                            st.error("请填写新旧路径前缀")
                elif diag["total"] > 0:
                    st.success("所有文件路径均有效")


# ===================================================================
# Tab 2: 数据导入
# ===================================================================

def _render_ingestion():
    st.header("📥 数据导入")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    format_choice = st.selectbox(
        "选择导入格式",
        ["A. 纯图片目录", "B. Thoro COCO 格式", "C. Roboflow COCO 格式",
         "D. Roboflow YOLO 格式", "E. CVAT 1.1 XML 格式"],
    )

    batch_tags_str = st.text_input(
        "批次标签（逗号分隔，用于区分不同批次数据）",
        key="ingest_batch_tags",
        placeholder="例如: batch_02, factory_A",
        help="为本次导入的所有样本打上标签，后续可在 FiftyOne 中按标签筛选，也可按标签分批推送到 CVAT",
    )
    batch_tags = [t.strip() for t in batch_tags_str.split(",") if t.strip()] if batch_tags_str else None

    if format_choice.startswith("A"):
        _render_ingest_images(ds, batch_tags)
    elif format_choice.startswith("B"):
        _render_ingest_thoro_coco(ds, batch_tags)
    elif format_choice.startswith("C"):
        _render_ingest_roboflow_coco(ds, batch_tags)
    elif format_choice.startswith("D"):
        _render_ingest_roboflow_yolo(ds, batch_tags)
    elif format_choice.startswith("E"):
        _render_ingest_cvat(ds, batch_tags)


def _render_ingest_images(ds, tags):
    st.subheader("A. 导入纯图片目录")
    image_dir = _path_browser("图片目录路径", "img_dir", mode="dir")
    recursive = st.checkbox("递归扫描子目录", value=True, key="img_recursive")

    if st.button("开始导入", key="btn_ingest_images"):
        if image_dir and Path(image_dir).is_dir():
            with st.spinner("正在导入图片..."):
                count = ingestors.ingest_images(ds, image_dir, tags=tags, recursive=recursive)
            st.success(f"✅ 成功导入 {count} 张图片")
        else:
            st.error("请输入有效的目录路径")


def _render_ingest_thoro_coco(ds, tags):
    st.subheader("B. 导入 Thoro COCO 格式")
    st.caption("目录需包含 `coco_files/` 和 `image_groups/` 子目录")
    input_dir = _path_browser("数据集根目录", "thoro_dir", mode="dir")
    label_field = st.text_input("标签字段名", value="ground_truth", key="thoro_field")
    label_map_str = st.text_area("标签映射 JSON（可选）", key="thoro_lmap", placeholder='{"旧名": "新名"}')
    kp_names_str = st.text_area("关键点名称 JSON（可选）", key="thoro_kpnames",
                                placeholder='{"类别名": ["kp0", "kp1", ...]}')
    include_bbox = st.checkbox("同时导入 BBox（即使有关键点/多边形）", key="thoro_bbox")

    if st.button("开始导入", key="btn_ingest_thoro"):
        if input_dir and Path(input_dir).is_dir():
            import json
            lmap = json.loads(label_map_str) if label_map_str.strip() else None
            kpmap = json.loads(kp_names_str) if kp_names_str.strip() else None
            with st.spinner("正在解析 Thoro COCO 格式..."):
                stats = ingestors.ingest_thoro_coco(
                    ds, input_dir, label_field=label_field,
                    label_map=lmap, kp_names_map=kpmap, include_bbox=include_bbox,
                    tags=tags,
                )
            st.success("✅ 导入完成")
            st.json(stats)
        else:
            st.error("请输入有效的目录路径")


def _render_ingest_roboflow_coco(ds, tags):
    st.subheader("C. 导入 Roboflow COCO 格式")
    dataset_dir = _path_browser("数据集根目录", "rf_coco_dir", mode="dir")
    label_field = st.text_input("标签字段名", value="ground_truth", key="rf_coco_field")
    splits = st.multiselect("选择 Split", ["train", "valid", "test"], default=["train", "valid", "test"],
                            key="rf_coco_splits")

    if st.button("开始导入", key="btn_ingest_rf_coco"):
        if dataset_dir and Path(dataset_dir).is_dir():
            with st.spinner("正在导入 Roboflow COCO..."):
                stats = ingestors.ingest_roboflow_coco(
                    ds, dataset_dir, label_field=label_field, splits=splits,
                    tags=tags,
                )
            st.success("✅ 导入完成")
            st.json(stats)
        else:
            st.error("请输入有效的目录路径")


def _render_ingest_roboflow_yolo(ds, tags):
    st.subheader("D. 导入 Roboflow YOLO 格式")
    dataset_dir = _path_browser("数据集根目录（含 data.yaml）", "rf_yolo_dir", mode="dir")
    label_field = st.text_input("标签字段名", value="ground_truth", key="rf_yolo_field")
    splits = st.multiselect("选择 Split", ["train", "valid", "test"], default=["train", "valid", "test"],
                            key="rf_yolo_splits")

    if st.button("开始导入", key="btn_ingest_rf_yolo"):
        if dataset_dir and Path(dataset_dir).is_dir():
            with st.spinner("正在导入 Roboflow YOLO..."):
                stats = ingestors.ingest_roboflow_yolo(
                    ds, dataset_dir, label_field=label_field, splits=splits,
                    tags=tags,
                )
            st.success("✅ 导入完成")
            st.json(stats)
        else:
            st.error("请输入有效的目录路径")


def _render_ingest_cvat(ds, tags):
    st.subheader("E. 导入 CVAT 1.1 XML 格式")
    xml_path = _path_browser(
        "annotations.xml 路径", "cvat_xml_path", mode="file",
        file_extensions=(".xml",),
    )
    image_dir = _path_browser(
        "图片目录（可选，默认为 XML 同级 images/）", "cvat_img_dir", mode="dir",
    )
    label_field = st.text_input("标签字段名", value="ground_truth", key="cvat_field")

    if st.button("开始导入", key="btn_ingest_cvat"):
        if xml_path and Path(xml_path).exists():
            img_dir = image_dir if image_dir else None
            with st.spinner("正在解析 CVAT XML..."):
                stats = ingestors.ingest_cvat_xml(
                    ds, xml_path, image_dir=img_dir, label_field=label_field,
                    tags=tags,
                )
            st.success("✅ 导入完成")
            st.json(stats)
        else:
            st.error("请输入有效的 XML 文件路径")


# ===================================================================
# Tab 3: 数据处理与清洗
# ===================================================================

def _render_processing():
    st.header("🧹 数据处理与清洗")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_clean, tab_label, tab_merge, tab_predict = st.tabs(
        ["🧹 图像清理", "🏷️ 标签管理", "🔀 字段合并", "🤖 自动预标注"],
    )

    with tab_clean:
        _render_cleaning(ds)
    with tab_label:
        _render_label_management(ds)
    with tab_merge:
        _render_field_merge(ds)
    with tab_predict:
        _render_auto_predict(ds)


def _render_cleaning(ds):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("无标注图像")
        unlabeled = processor.find_unlabeled_samples(ds)
        st.metric("无标注样本数", len(unlabeled))
        physical = st.checkbox("同时删除磁盘文件", key="clean_unlabeled_physical")
        if st.button("🗑️ 删除无标注样本", key="btn_del_unlabeled"):
            if len(unlabeled) > 0:
                with st.spinner("删除中..."):
                    count = processor.delete_unlabeled_samples(ds, physical=physical)
                st.success(f"✅ 删除 {count} 个样本")
                st.rerun()
            else:
                st.info("没有无标注样本")

    with col2:
        st.subheader("损坏/异常图像")
        min_size = st.number_input("最小尺寸 (px)", value=32, key="min_size")
        max_size = st.number_input("最大尺寸 (px)", value=20000, key="max_size")
        if st.button("🔍 扫描异常图像", key="btn_scan_corrupt"):
            with st.spinner("扫描中（需遍历所有图片）..."):
                bad_ids = processor.find_corrupt_or_abnormal(ds, min_size=min_size, max_size=max_size)
            st.session_state["corrupt_ids"] = bad_ids
            st.metric("发现异常", len(bad_ids))

        if st.session_state.get("corrupt_ids"):
            if st.button("🗑️ 删除异常样本（含磁盘文件）", key="btn_del_corrupt"):
                from tools.dataset_platform.data_manager import delete_samples_physically
                count = delete_samples_physically(ds, st.session_state["corrupt_ids"])
                st.success(f"✅ 物理删除 {count} 个样本")
                del st.session_state["corrupt_ids"]
                st.rerun()

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("按标签删除样本")
        available_tags = ds.distinct("tags")
        if available_tags:
            del_tags = st.multiselect(
                "选择标签（包含这些标签的样本将被删除）",
                available_tags,
                key="clean_del_tags",
            )
            if del_tags:
                tag_view = ds.match_tags(del_tags)
                st.metric("匹配样本数", len(tag_view))
            del_physical = st.checkbox("同时删除磁盘文件", key="clean_tag_physical")
            if st.button("🗑️ 删除匹配样本", key="btn_del_by_tag"):
                if del_tags:
                    tag_view = ds.match_tags(del_tags)
                    count = len(tag_view)
                    if count > 0:
                        if del_physical:
                            ids = tag_view.values("id")
                            dm.delete_samples_physically(ds, ids)
                        else:
                            ds.delete_samples(tag_view)
                        st.session_state["_toast_msg"] = f"已删除 {count} 个带标签 {del_tags} 的样本"
                        st.rerun()
                    else:
                        st.info("没有匹配的样本")
                else:
                    st.error("请先选择标签")
        else:
            st.info("当前数据集没有任何标签")

    with col4:
        st.subheader("重复图像清理")

        dedup_scope = st.radio(
            "扫描范围",
            ["整个数据集", "按标签筛选"],
            key="dedup_scope",
            horizontal=True,
        )
        dedup_view = ds
        if dedup_scope == "按标签筛选":
            avail = ds.distinct("tags")
            if avail:
                dedup_tags = st.multiselect(
                    "选择标签（仅扫描包含这些标签的样本）",
                    avail, key="dedup_scope_tags",
                )
                if dedup_tags:
                    dedup_view = ds.match_tags(dedup_tags)
            else:
                st.info("当前数据集没有任何标签")
        st.caption(f"扫描范围: **{len(dedup_view)}** 个样本")

        dedup_mode = st.selectbox("去重模式", ["精确哈希", "近似嵌入", "两者兼有"], key="dedup_mode")

        near_threshold = 0.03
        near_model = "clip-vit-base32-torch"
        if dedup_mode in ("近似嵌入", "两者兼有"):
            near_model = st.selectbox(
                "嵌入模型",
                [
                    "clip-vit-base32-torch",
                    "clip-vit-large14-torch",
                    "dinov2-vits14-torch",
                    "dinov2-vitb14-torch",
                    "dinov2-vitb14-reg-torch",
                    "dinov2-vitl14-torch",
                ],
                index=0,
                key="dedup_model",
                help="CLIP 更通用，DINOv2 对视觉相似度更敏感",
            )
            near_threshold = st.slider(
                "余弦距离阈值",
                min_value=0.001,
                max_value=0.20,
                value=0.03,
                step=0.005,
                key="dedup_threshold",
                help="越小越严格（仅非常相似才算重复），推荐 0.01~0.05",
            )

        dup_count = len(ds.match_tags(["duplicate"]))
        if dup_count > 0:
            st.warning(f"当前有 **{dup_count}** 个样本标记为 `duplicate`")
            if st.button("🧹 清除所有 duplicate 标记", key="btn_clear_dup_tag",
                         help="移除所有样本的 duplicate 标签，以便用新参数重新扫描去重"):
                with st.spinner("清除中..."):
                    cleared = 0
                    for sample in ds.match_tags(["duplicate"]).iter_samples(autosave=True):
                        if "duplicate" in sample.tags:
                            sample.tags.remove("duplicate")
                            cleared += 1
                st.session_state["_toast_msg"] = f"已清除 {cleared} 个样本的 duplicate 标记"
                if "dup_groups" in st.session_state:
                    del st.session_state["dup_groups"]
                st.rerun()

        if st.button("🔍 扫描重复图像", key="btn_scan_dup"):
            all_groups = []
            with st.spinner(f"扫描 {len(dedup_view)} 个样本中..."):
                if dedup_mode in ("精确哈希", "两者兼有"):
                    exact = processor.find_exact_duplicates(dedup_view)
                    all_groups.extend(exact)
                    st.info(f"精确重复: {len(exact)} 组")
                if dedup_mode in ("近似嵌入", "两者兼有"):
                    near = processor.find_near_duplicates(
                        dedup_view, threshold=near_threshold, model_name=near_model,
                    )
                    all_groups.extend(near)
                    st.info(f"近似重复: {len(near)} 组 (模型={near_model}, 阈值={near_threshold})")
            st.session_state["dup_groups"] = all_groups
            total_dups = sum(len(g) - 1 for g in all_groups)
            st.metric("可删除重复", total_dups)

        if st.session_state.get("dup_groups"):
            groups = st.session_state["dup_groups"]
            tag_only = st.checkbox("仅打标签不删除", value=True, key="dup_tag_only")
            physical = st.checkbox("同时删除磁盘文件", key="dup_physical")
            if st.button("执行去重", key="btn_exec_dedup"):
                if tag_only:
                    count = processor.tag_duplicates(ds, groups)
                    st.success(f"✅ 标记 {count} 个重复样本")
                else:
                    count = processor.delete_duplicates(ds, groups, physical=physical)
                    st.success(f"✅ 删除 {count} 个重复样本")
                del st.session_state["dup_groups"]
                st.rerun()


def _render_label_management(ds):
    # ---- 样本标签（Tags）管理 ----
    st.subheader("样本标签 (Tags) 管理")
    st.caption("给已有样本批量添加或移除 Tags（如批次标记、审核状态等）")

    available_tags = ds.distinct("tags")
    info = dm.get_dataset_info(ds)
    label_fields = info.get("label_fields", [])

    tag_scope = st.radio(
        "操作范围",
        ["整个数据集", "按已有标签筛选", "按标注字段筛选"],
        key="tag_mgmt_scope",
        horizontal=True,
    )

    scope_view = None
    if tag_scope == "按已有标签筛选":
        if available_tags:
            scope_tags = st.multiselect(
                "选择要操作的样本（包含以下标签的样本）",
                available_tags,
                key="tag_mgmt_scope_tags",
            )
            if scope_tags:
                scope_view = ds.match_tags(scope_tags)
        else:
            st.info("当前数据集没有任何标签")

    elif tag_scope == "按标注字段筛选":
        if label_fields:
            scope_field = st.selectbox("选择标注字段", label_fields, key="tag_mgmt_scope_field")
            field_filter = st.radio(
                "筛选条件",
                ["有标注的样本", "无标注的样本"],
                key="tag_mgmt_field_filter",
                horizontal=True,
            )
            if field_filter == "有标注的样本":
                scope_view = dm.get_labeled_view(ds, scope_field)
            else:
                scope_view = dm.get_unlabeled_view(ds, scope_field)
        else:
            st.info("当前数据集没有标注字段")

    target = scope_view if scope_view is not None else ds
    st.caption(f"当前范围: **{len(target)}** 个样本")

    tag_action = st.radio("操作", ["添加标签", "移除标签"], key="tag_mgmt_action", horizontal=True)

    if tag_action == "添加标签":
        new_tags_str = st.text_input(
            "要添加的标签（逗号分隔）",
            key="tag_mgmt_add",
            placeholder="例如: batch_02, reviewed",
        )
        new_tags = [t.strip() for t in new_tags_str.split(",") if t.strip()] if new_tags_str else []
    else:
        if available_tags:
            new_tags = st.multiselect("选择要移除的标签", available_tags, key="tag_mgmt_remove")
        else:
            st.info("当前数据集没有任何标签可移除")
            new_tags = []

    if st.button(f"{'🏷️ 添加' if tag_action == '添加标签' else '🗑️ 移除'}标签", key="btn_tag_mgmt"):
        if not new_tags:
            st.error("请指定要操作的标签")
        else:
            count = 0
            with st.spinner("处理中..."):
                for sample in target.iter_samples(autosave=True):
                    if tag_action == "添加标签":
                        for t in new_tags:
                            if t not in sample.tags:
                                sample.tags.append(t)
                        count += 1
                    else:
                        for t in new_tags:
                            if t in sample.tags:
                                sample.tags.remove(t)
                        count += 1
            st.session_state["_toast_msg"] = (
                f"已{'添加' if tag_action == '添加标签' else '移除'}标签，影响 {count} 个样本"
            )
            st.rerun()

    if available_tags:
        with st.expander("📊 当前标签分布", expanded=False):
            import pandas as pd
            tag_counts = ds.count_values("tags")
            if tag_counts:
                df = pd.DataFrame(
                    list(tag_counts.items()), columns=["标签", "样本数"]
                ).sort_values("样本数", ascending=False)
                st.dataframe(df, use_container_width=True)

    # ---- 标注类别重命名 ----
    st.markdown("---")
    st.subheader("标注类别重命名")
    st.caption("可将指定标签字段中的某个类别批量重命名为新名称")

    info = dm.get_dataset_info(ds)
    label_fields = info.get("label_fields", [])
    if not label_fields:
        st.info("当前数据集没有标签字段，请先导入标注数据")
        return

    rename_field = st.selectbox("选择标签字段", label_fields, key="rename_field")
    current_classes = dm.get_label_classes(ds, rename_field)

    if not current_classes:
        st.info(f"字段 `{rename_field}` 中暂无标签类别")
        return

    st.markdown(f"**当前类别** ({len(current_classes)}): `{'`, `'.join(current_classes)}`")

    col1, col2 = st.columns(2)
    with col1:
        old_label = st.selectbox("选择要重命名的类别", current_classes, key="rename_old")
    with col2:
        new_label = st.text_input("新类别名称", key="rename_new")

    if st.button("✏️ 执行重命名", key="btn_rename_label"):
        if not new_label or not new_label.strip():
            st.error("请输入新类别名称")
        elif new_label.strip() == old_label:
            st.warning("新名称与旧名称相同，无需操作")
        else:
            with st.spinner(f"正在将 `{old_label}` 重命名为 `{new_label.strip()}`..."):
                try:
                    count = dm.rename_label(ds, rename_field, old_label, new_label.strip())
                    st.session_state["_toast_msg"] = f"已将 '{old_label}' 重命名为 '{new_label.strip()}'（共 {count} 个实例）"
                    st.rerun()
                except Exception as e:
                    st.error(f"重命名失败: {e}")

    st.markdown("---")
    st.subheader("标注统计")
    stats = dm.get_label_stats(ds, rename_field)
    if stats:
        import pandas as pd
        df = pd.DataFrame(
            list(stats.items()), columns=["类别", "数量"]
        ).sort_values("数量", ascending=False)
        st.dataframe(df, use_container_width=True)


def _render_field_merge(ds):
    st.subheader("标签字段合并")
    st.caption(
        "将一个标签字段的标注合并到另一个字段中。"
        "适用于分批导入时使用了不同字段名（如 `batch2_gt`），"
        "标注完成后需要汇总到统一字段（如 `ground_truth`）。"
    )

    info = dm.get_dataset_info(ds)
    label_fields = info.get("label_fields", [])
    all_fields = info.get("sample_fields", [])

    if len(label_fields) < 2:
        st.info("当前数据集只有 0~1 个标签字段，无需合并。需要至少 2 个标签字段才能进行合并操作。")
        return

    col1, col2 = st.columns(2)
    with col1:
        source_field = st.selectbox("源字段（将被合并）", label_fields, key="merge_src")
        src_stats = dm.get_label_stats(ds, source_field)
        if src_stats:
            st.caption(f"包含 {sum(src_stats.values())} 个标注实例 ({len(src_stats)} 类别)")
        else:
            st.caption("该字段暂无标注数据")

    with col2:
        target_options = [f for f in label_fields if f != source_field]
        target_field = st.selectbox("目标字段（合并到此）", target_options, key="merge_tgt")
        tgt_stats = dm.get_label_stats(ds, target_field)
        if tgt_stats:
            st.caption(f"包含 {sum(tgt_stats.values())} 个标注实例 ({len(tgt_stats)} 类别)")
        else:
            st.caption("该字段暂无标注数据")

    src_type = dm.get_field_label_type(ds, source_field)
    tgt_type = dm.get_field_label_type(ds, target_field)
    if src_type and tgt_type and src_type != tgt_type:
        st.error(f"类型不匹配：源字段为 `{src_type}`，目标字段为 `{tgt_type}`，无法合并。")
        return

    delete_source = st.checkbox(
        "合并后删除源字段", value=True, key="merge_del_src",
        help="勾选后，合并完成会从数据集中移除源字段",
    )

    if st.button("🔀 执行合并", key="btn_merge_fields"):
        with st.spinner(f"正在将 `{source_field}` 合并到 `{target_field}`..."):
            try:
                result = dm.merge_label_fields(
                    ds, source_field, target_field, delete_source=delete_source,
                )
                st.session_state["_toast_msg"] = (
                    f"合并完成：{result['merged']} 个样本已合并"
                    + (f"，源字段 `{source_field}` 已删除" if result["source_deleted"] else "")
                )
                st.rerun()
            except Exception as e:
                st.error(f"合并失败: {e}")


def _render_auto_predict(ds):
    st.subheader("YOLO 模型自动预标注")
    model_path = _path_browser(
        "模型权重路径 (.pt)", "pred_model", mode="file",
        file_extensions=(".pt", ".pth", ".onnx", ".engine"),
    )
    task = st.selectbox("任务类型", ["detect", "pose", "obb"], key="pred_task")
    pred_field = st.text_input("预测结果字段名", value="predictions", key="pred_field")
    conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="pred_conf")

    only_unlabeled = st.checkbox("仅对无标注样本预标注", value=True, key="pred_unlabeled_only")

    if st.button("🚀 开始预标注", key="btn_predict"):
        if model_path and Path(model_path).exists():
            view = None
            if only_unlabeled:
                view = processor.find_unlabeled_samples(ds)
                st.info(f"将对 {len(view)} 个无标注样本进行预标注")

            with st.spinner(f"使用 {task} 模型预标注中..."):
                stats = processor.auto_predict_yolo(
                    ds, model_path, pred_field=pred_field,
                    conf_threshold=conf, task=task, view=view,
                )
            st.success("✅ 预标注完成")
            st.json(stats)
        else:
            st.error("请输入有效的模型路径")


# ===================================================================
# Tab 4: CVAT 同步
# ===================================================================

def _render_cvat_sync():
    st.header("🔄 CVAT 双向同步")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    with st.expander("ℹ️ CVAT 同步概念说明", expanded=False):
        st.markdown("""
**映射关系**: FiftyOne Dataset ↔ CVAT Project | FiftyOne View → CVAT Task | CVAT 自动切分 → Jobs

**核心概念：**

| 概念 | 说明 |
|------|------|
| **anno_key (标注键)** | 一次标注任务的唯一标识符。推送数据到 CVAT 时设置，拉取时通过同一 key 匹配回原数据集。建议命名格式：`round1_detect`、`v2_polygon` 等，便于区分不同轮次的标注。 |
| **FiftyOne View (视图)** | 数据集的一个子集/筛选视图。推送时可以选择"整个数据集"、"仅无标注样本"或"仅有标注样本"，这些选项决定了哪些图片会被上传到 CVAT。 |
| **label_field (标签字段)** | FiftyOne 中存储标注的字段名，如 `ground_truth`。推送时会将该字段的已有标注作为预标注上传到 CVAT；拉取时 CVAT 的标注结果会写回此字段。 |
| **label_type (标注类型)** | 对应 CVAT 中的标注方式：`detections`=矩形框, `polylines`=多边形, `keypoints`=关键点/骨架, `classifications`=图片分类。 |
| **segment_size** | CVAT 中每个 Job 包含的图片数。影响标注员的工作量划分。|

**典型工作流：**
1. **推送** → 选择数据 + 设置 anno_key → 数据上传到 CVAT → 标注员在 CVAT 中标注
2. **拉取** → 选择同一 anno_key → 标注结果自动同步回 FiftyOne 数据集
        """)


    tab_push, tab_pull, tab_manage = st.tabs(["📤 推送到 CVAT", "📥 从 CVAT 拉取", "📋 管理标注运行"])

    with tab_push:
        _render_cvat_push(ds)
    with tab_pull:
        _render_cvat_pull(ds)
    with tab_manage:
        _render_cvat_manage(ds)


def _render_cvat_push(ds):
    st.subheader("推送数据到 CVAT")

    _TYPE_LABELS = {
        "detections": "矩形框 (detections)",
        "polylines": "多边形 (polylines)",
        "keypoints": "关键点 (keypoints)",
        "classifications": "分类 (classifications)",
    }

    # -- 已有标注运行提示 --
    existing_runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
    if existing_runs:
        st.info(f"当前数据集已有标注运行: **{', '.join(existing_runs)}**（可在「管理标注运行」中查看）")

    anno_key = st.text_input(
        "标注键 (anno_key)", key="push_anno_key",
        placeholder="例如: round1_all",
        help="标注任务的唯一 ID，拉取标注时需要用同一个 key 匹配。每次推送必须使用不同的 key。",
    )

    # -- 标签字段配置 --
    info = dm.get_dataset_info(ds)
    label_fields = info.get("label_fields", [])

    st.markdown("**标签字段**")
    field_mode = st.radio(
        "字段来源",
        ["选择已有字段（推送已有标注作为预标注）", "输入新字段名（在 CVAT 中从零开始标注）"],
        key="push_field_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    # 这些变量根据模式分支赋值，用于最终推送
    label_schema = None       # 多字段模式
    single_label_field = None # 单字段模式
    single_label_type = None
    single_push_attrs = None
    polyline_fields = []

    if field_mode.startswith("选择已有"):
        if label_fields:
            selected_fields = st.multiselect(
                "选择标签字段（可多选，同时推送多种标注类型到同一 CVAT 任务）",
                label_fields,
                default=label_fields,
                key="push_fields_sel",
                help="选择多个字段可将不同类型的标注（如矩形框 + 多边形）一次性推送",
            )

            if selected_fields:
                field_type_map = {}
                for f in selected_fields:
                    ftype = dm.get_field_label_type(ds, f)
                    all_classes = dm.get_label_classes(ds, f)
                    type_display = _TYPE_LABELS.get(ftype, ftype or "未知")
                    field_type_map[f] = {"type": ftype, "classes": all_classes, "display": type_display}
                    if ftype == "polylines":
                        polyline_fields.append(f)

                summary = "\n".join(
                    f"- **`{f}`** — {v['display']} ({len(v['classes'])} 类别)"
                    for f, v in field_type_map.items()
                )
                st.markdown("**已选字段：**\n" + summary)

                include_occu_attrs = False
                if polyline_fields:
                    include_occu_attrs = st.checkbox(
                        "为多边形字段添加四角遮挡属性 (1_occu ~ 4_occu)",
                        value=False,
                        key="push_occu_attrs",
                        help="为每个多边形标注添加四个 checkbox 属性，标记托盘四个角是否被遮挡（默认 false）",
                    )

                field_classes_filter = {}
                with st.expander("🏷️ 按字段筛选类别（默认推送全部类别）", expanded=False):
                    for f in selected_fields:
                        all_cls = field_type_map[f]["classes"]
                        if all_cls:
                            sel = st.multiselect(
                                f"`{f}` — 选择类别（留空=全部 {len(all_cls)} 个）",
                                all_cls,
                                key=f"push_cls_{f}",
                            )
                            if sel:
                                field_classes_filter[f] = sel

                label_schema = {}
                for f in selected_fields:
                    entry = {}
                    if f in field_classes_filter:
                        entry["classes"] = field_classes_filter[f]
                    if f in polyline_fields and include_occu_attrs:
                        entry["attributes"] = cvat_sync.OCCLUSION_ATTRS
                    label_schema[f] = entry
            else:
                st.warning("请至少选择一个字段")
        else:
            st.warning("当前数据集没有标签字段，请切换到「输入新字段名」模式")
    else:
        single_label_field = st.text_input(
            "新字段名", value="ground_truth", key="push_field_new",
            help="CVAT 标注结果将写入此字段",
        )
        single_label_type = st.selectbox(
            "标注类型（新字段必选）",
            ["detections", "polylines", "keypoints", "classifications"],
            key="push_ltype",
            help="detections=矩形框, polylines=多边形, keypoints=关键点/骨架, classifications=图片分类",
        )
        if single_label_type == "polylines":
            if st.checkbox("添加四角遮挡属性 (1_occu ~ 4_occu)", value=False, key="push_occu_attrs_new"):
                single_push_attrs = cvat_sync.OCCLUSION_ATTRS

    st.markdown("---")

    # -- 推送范围 --
    push_mode = st.radio(
        "推送范围", ["整个数据集", "仅无标注样本", "仅有标注样本"], key="push_mode",
        help="决定哪些图片上传到 CVAT（多字段模式下基于首个选中字段判断）",
        horizontal=True,
    )

    # -- 标签筛选 --
    available_tags = ds.distinct("tags")
    push_filter_tags = []
    if available_tags:
        push_filter_tags = st.multiselect(
            "按批次标签筛选（留空 = 不过滤，推送上方范围内的所有样本）",
            available_tags,
            key="push_filter_tags",
            help="选择一个或多个标签，仅推送包含这些标签的样本。配合「推送范围」可实现如「仅推送 batch_02 中的无标注样本」",
        )

    # -- 类别过滤（仅单字段模式下显示） --
    selected_classes = []
    if label_schema is None and label_fields:
        filter_field = single_label_field if (single_label_field and single_label_field in label_fields) else None
        if filter_field:
            all_classes = dm.get_label_classes(ds, filter_field)
            if all_classes:
                selected_classes = st.multiselect(
                    "仅推送以下类别（留空 = 推送全部类别）",
                    all_classes, key="push_classes",
                    help="勾选的类别会被推送到 CVAT；不勾选任何则推送所有类别",
                )

    col_seg, col_quality = st.columns(2)
    with col_seg:
        segment_size = st.number_input(
            "每个 Job 图片数", value=200, min_value=10, key="push_seg",
            help="CVAT 会将任务切分为多个 Job，每个 Job 包含此数量图片",
        )
    with col_quality:
        image_quality = st.number_input(
            "图像质量", value=100, min_value=1, max_value=100, key="push_img_quality",
            help="上传到 CVAT 的图像压缩质量 (1~100)，100 为无损。FiftyOne 默认 75，此处默认 100",
        )

    # -- 推送按钮 --
    can_push = label_schema is not None or single_label_field
    if st.button("📤 推送到 CVAT", key="btn_push", disabled=not can_push):
        if not anno_key:
            st.error("请输入标注键 (anno_key)")
            return
        if not anno_key.replace("_", "").replace("-", "").isalnum():
            st.error("标注键只能包含字母、数字、下划线和短横线")
            return
        if anno_key in existing_runs:
            st.error(f"标注键 `{anno_key}` 已存在！请使用不同的 key，或先在「管理标注运行」中删除旧记录。")
            return

        ref_field = list(label_schema.keys())[0] if label_schema else single_label_field
        samples = ds
        if push_mode == "仅无标注样本":
            samples = dm.get_unlabeled_view(ds, ref_field)
        elif push_mode == "仅有标注样本":
            samples = dm.get_labeled_view(ds, ref_field)

        if push_filter_tags:
            samples = samples.match_tags(push_filter_tags)

        if len(samples) == 0:
            st.error("所选范围内没有样本可推送")
            return

        with st.spinner(f"正在推送 {len(samples)} 个样本到 CVAT..."):
            try:
                if label_schema:
                    result = cvat_sync.push_to_cvat(
                        samples, anno_key,
                        label_schema=label_schema,
                        segment_size=segment_size,
                        image_quality=image_quality,
                    )
                else:
                    result = cvat_sync.push_to_cvat(
                        samples, anno_key,
                        label_field=single_label_field,
                        label_type=single_label_type,
                        segment_size=segment_size,
                        image_quality=image_quality,
                        classes=selected_classes or None,
                        attributes=single_push_attrs,
                    )
                st.success("✅ 推送成功")
                st.json(result)
                st.info("💡 现在可以在 CVAT 中进行标注，完成后回到「从 CVAT 拉取」页面同步结果。")
            except Exception as e:
                st.error(f"推送失败: {e}")


def _render_cvat_pull(ds):
    st.subheader("从 CVAT 拉取标注")

    runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
    if runs:
        st.success(f"检测到 **{len(runs)}** 个标注运行（数据持久化在 FiftyOne 数据库中，重启不会丢失）")
        anno_key = st.selectbox(
            "选择标注运行", runs, key="pull_anno_key",
            help="选择之前推送时使用的 anno_key",
        )

        try:
            run_info = ds.get_annotation_info(anno_key)
            with st.expander("运行详情", expanded=False):
                st.caption(f"创建时间: {run_info.timestamp if hasattr(run_info, 'timestamp') else 'N/A'}")
                st.caption(f"配置: {run_info.config if hasattr(run_info, 'config') else 'N/A'}")
        except Exception:
            pass
    else:
        st.warning(
            "当前数据集没有标注运行记录。\n\n"
            "**可能原因：**\n"
            "- 尚未推送任何数据到 CVAT（请先使用「推送到 CVAT」）\n"
            "- 选错了数据集（请在侧边栏确认当前数据集）"
        )
        anno_key = ""

    cleanup = st.checkbox(
        "拉取后删除 CVAT 端任务", key="pull_cleanup",
        help="勾选后，拉取完成会同时删除 CVAT 服务器上的对应任务和项目",
    )

    if st.button("📥 拉取标注", key="btn_pull", disabled=not anno_key):
        if not anno_key:
            st.error("请选择标注运行")
            return
        with st.spinner("拉取中..."):
            try:
                result = cvat_sync.pull_from_cvat(ds, anno_key, cleanup=cleanup)
                st.success("✅ 拉取成功，标注已同步到数据集")
                st.json(result)
            except Exception as e:
                st.error(f"拉取失败: {e}")


def _render_cvat_manage(ds):
    st.subheader("标注运行管理")

    runs = cvat_sync.list_annotation_runs(ds)
    if runs:
        st.caption(f"以下标注运行记录持久存储在 FiftyOne 数据库中（共 {len(runs)} 条），重启应用不会丢失。")
        import pandas as pd
        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True)

        del_key = st.selectbox("选择要删除的运行", [r["anno_key"] for r in runs], key="del_run_key")
        col1, col2 = st.columns(2)
        with col1:
            cleanup_cvat = st.checkbox("同时清理 CVAT 端", key="del_cleanup_cvat",
                                       help="勾选后，删除运行记录的同时会删除 CVAT 服务器上的任务")
        with col2:
            if st.button("🗑️ 删除标注运行", key="btn_del_run"):
                try:
                    cvat_sync.delete_annotation_run(ds, del_key, cleanup=cleanup_cvat)
                    st.session_state["_toast_msg"] = f"已删除标注运行: {del_key}"
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
    else:
        st.info("暂无标注运行记录。推送数据到 CVAT 后会在此显示。")


# ===================================================================
# Tab 5: 数据导出
# ===================================================================

def _render_export():
    st.header("📤 多格式导出")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    format_choice = st.selectbox(
        "导出格式",
        [
            "YOLO Detect (纯框)",
            "YOLO Pose (关键点)",
            "YOLO Pose (四边形转关键点)",
            "YOLO OBB (旋转框)",
        ],
        key="export_format",
    )

    if format_choice == "YOLO Pose (四边形转关键点)":
        st.info("将 4 点多边形 (Polylines) 自动转换为 YOLO Pose 格式：\n"
                "- 四角排序为 tl→tr→br→bl\n"
                "- 贴近图像边界的关键点标记为不可见 (v=0, 坐标归零)\n"
                "- 其余关键点标记为可见 (v=2)")

    output_dir = _path_browser("输出目录", "export_dir", mode="dir", start_dir=CONFIG.default_export_dir)

    split_mode = st.radio(
        "数据划分方式",
        ["按比例自动划分", "指定单个 Split"],
        key="export_split_mode",
        horizontal=True,
    )

    if split_mode == "按比例自动划分":
        st.caption("设置比例，三者之和应为 100%")
        col_t, col_v, col_te = st.columns(3)
        with col_t:
            train_pct = st.number_input("Train %", 0, 100, 80, key="split_train_pct")
        with col_v:
            valid_pct = st.number_input("Valid %", 0, 100, 10, key="split_valid_pct")
        with col_te:
            test_pct = st.number_input("Test %", 0, 100, 10, key="split_test_pct")
        total_pct = train_pct + valid_pct + test_pct
        if total_pct != 100:
            st.warning(f"当前总比例为 {total_pct}%，请调整为 100%")
        splits: str | dict[str, float] = {}
        if train_pct > 0:
            splits["train"] = train_pct / 100.0
        if valid_pct > 0:
            splits["valid"] = valid_pct / 100.0
        if test_pct > 0:
            splits["test"] = test_pct / 100.0
    else:
        single_split = st.selectbox("Split 名称", ["train", "valid", "test"], key="export_split")
        splits = single_split

    info = dm.get_dataset_info(ds)
    label_field = st.selectbox("标签字段", info.get("label_fields", ["ground_truth"]),
                               key="export_label_field")

    all_classes = dm.get_label_classes(ds, label_field)
    if all_classes:
        selected_classes = st.multiselect(
            "选择导出类别（留空导出全部）", all_classes, key="export_classes"
        )
    else:
        selected_classes = []

    if format_choice == "YOLO Pose (关键点)":
        kp_field = st.text_input("关键点字段名", value=f"{label_field}_keypoints", key="export_kp_field")

    if format_choice == "YOLO Pose (四边形转关键点)":
        st.markdown("**可见性参数**")
        col_e, col_m = st.columns(2)
        with col_e:
            edge_threshold = st.number_input(
                "边界阈值 (像素)",
                min_value=0.0, max_value=100.0, value=5.0, step=1.0,
                key="export_edge_threshold",
                help="关键点距图像边界小于此值 → v=0 且坐标归零（不可见）",
            )
        with col_m:
            bbox_margin = st.number_input(
                "BBox 外扩边距 (归一化)",
                min_value=0.0, max_value=0.2, value=0.0, step=0.005,
                key="export_bbox_margin",
                help="从 polygon 顶点向外扩展此比例作为包围框（0=紧贴顶点）",
            )

    if format_choice == "YOLO OBB (旋转框)":
        obb_field = st.text_input("OBB 字段名（Polylines, 可选）", key="export_obb_field",
                                  placeholder="留空则从 Detections + rotation 构造")

    can_export = True
    if isinstance(splits, dict) and total_pct != 100:
        can_export = False

    if st.button("📦 开始导出", key="btn_export", disabled=not can_export):
        if not output_dir:
            st.error("请指定输出目录")
            return

        classes = selected_classes or None
        with st.spinner("导出中..."):
            try:
                if format_choice == "YOLO Detect (纯框)":
                    result = exporter.export_yolo_detect(
                        ds, output_dir, label_field=label_field,
                        classes=classes, splits=splits,
                    )
                elif format_choice == "YOLO Pose (关键点)":
                    result = exporter.export_yolo_pose(
                        ds, output_dir, det_field=label_field,
                        kp_field=kp_field, classes=classes, splits=splits,
                    )
                elif format_choice == "YOLO Pose (四边形转关键点)":
                    result = exporter.export_yolo_pose_from_polylines(
                        ds, output_dir, label_field=label_field,
                        classes=classes, splits=splits,
                        edge_threshold=edge_threshold,
                        bbox_margin=bbox_margin,
                    )
                elif format_choice == "YOLO OBB (旋转框)":
                    result = exporter.export_yolo_obb(
                        ds, output_dir, label_field=label_field,
                        obb_field=obb_field if obb_field else None,
                        classes=classes, splits=splits,
                    )
                else:
                    result = {}

                st.success("✅ 导出完成")
                st.json(result)
            except Exception as e:
                st.error(f"导出失败: {e}")


# ===================================================================
# Tab 6: 高级功能
# ===================================================================

def _render_advanced():
    st.header("🧠 高级功能")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_hard, tab_snapshot = st.tabs(["🎯 难例挖掘", "📸 版本控制"])

    with tab_hard:
        _render_hard_mining(ds)
    with tab_snapshot:
        _render_snapshots(ds)


def _render_hard_mining(ds):
    st.subheader("难例挖掘")
    st.caption("先运行自动预标注，再通过对比 ground_truth 和 predictions 找出高错误率图像")

    pred_field = st.text_input("预测字段", value="predictions", key="hard_pred")
    gt_field = st.text_input("真值字段", value="ground_truth", key="hard_gt")
    iou_thresh = st.slider("IoU 阈值", 0.1, 0.95, 0.5, 0.05, key="hard_iou")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 评估模型表现", key="btn_eval"):
            with st.spinner("评估中..."):
                try:
                    advanced.evaluate_detections(
                        ds, pred_field=pred_field, gt_field=gt_field,
                        iou_threshold=iou_thresh,
                    )
                    st.success("✅ 评估完成")
                except Exception as e:
                    st.error(f"评估失败: {e}")

    with col2:
        limit = st.number_input("最多显示难例数", value=50, key="hard_limit")
        if st.button("🎯 挖掘难例", key="btn_hard"):
            with st.spinner("挖掘中..."):
                try:
                    hard_view = advanced.find_hard_samples(
                        ds, pred_field=pred_field, limit=limit,
                    )
                    st.info(f"找到 {len(hard_view)} 个难例样本")
                    session = dm.get_session()
                    if session:
                        dm.set_session_view(hard_view)
                        st.success("✅ 已在 FiftyOne App 中展示难例")
                    else:
                        st.warning("请先启动 FiftyOne App")
                except Exception as e:
                    st.error(f"挖掘失败: {e}")


def _render_snapshots(ds):
    st.subheader("数据集版本控制（快照）")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**创建快照**")
        note = st.text_input("快照备注", key="snap_note", placeholder="例如: v1.0 发版前备份")
        suffix = st.text_input("快照后缀（可选，默认时间戳）", key="snap_suffix")
        if st.button("📸 创建快照", key="btn_snap"):
            with st.spinner("创建中..."):
                name = advanced.create_snapshot(
                    ds.name,
                    snapshot_suffix=suffix if suffix else None,
                    note=note,
                )
            st.success(f"✅ 快照已创建: {name}")

    with col2:
        st.markdown("**已有快照**")
        snapshots = advanced.list_snapshots(ds.name)
        if snapshots:
            import pandas as pd
            df = pd.DataFrame(snapshots)
            st.dataframe(df[["suffix", "time", "note", "num_samples"]], use_container_width=True)

            snap_name = st.selectbox("选择快照", [s["name"] for s in snapshots], key="snap_select")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("♻️ 恢复此快照", key="btn_restore"):
                    st.warning(f"⚠️ 将覆盖当前数据集 '{ds.name}'！")
                    if st.button("确认恢复", key="btn_confirm_restore"):
                        advanced.restore_snapshot(snap_name, ds.name)
                        st.success("✅ 已恢复")
                        st.rerun()
            with col_b:
                if st.button("🗑️ 删除快照", key="btn_del_snap"):
                    advanced.delete_snapshot(snap_name)
                    st.success("✅ 已删除")
                    st.rerun()
        else:
            st.info("暂无快照")


# ===================================================================
# 主入口
# ===================================================================

def main():
    if "_toast_msg" in st.session_state:
        st.toast(st.session_state.pop("_toast_msg"), icon="✅")

    _render_sidebar()

    tab_hub, tab_import, tab_process, tab_cvat, tab_export, tab_adv = st.tabs([
        "🔍 Data Hub",
        "📥 数据导入",
        "🧹 处理清洗",
        "🔄 CVAT 同步",
        "📤 数据导出",
        "🧠 高级功能",
    ])

    with tab_hub:
        _render_data_hub()
    with tab_import:
        _render_ingestion()
    with tab_process:
        _render_processing()
    with tab_cvat:
        _render_cvat_sync()
    with tab_export:
        _render_export()
    with tab_adv:
        _render_advanced()


if __name__ == "__main__":
    main()
