"""数据导入页面：多格式导入支持。"""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import ingestors
from tools.dataset_platform import advanced
from tools.dataset_platform.ui.components import _get_ds, _path_browser


def _render_ingestion():
    st.header("📥 数据导入")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    format_choice = st.selectbox(
        "选择导入格式",
        ["A. 纯图片目录", "B. Thoro COCO 格式", "C. Roboflow COCO 格式",
         "D. Roboflow YOLO 格式", "E. CVAT 1.1 XML 格式", "F. 从备份导入"],
    )

    if format_choice.startswith("F"):
        _render_ingest_from_backup(ds)
        return

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


def _render_ingest_from_backup(ds):
    st.subheader("F. 从备份导入")
    st.caption(
        "从本平台生成的完整备份中导入数据。备份中的图像和标注（包括原始 Tags、标签字段）"
        "会被追加到当前数据集，已存在的同名文件会自动跳过。\n\n"
        "**适用场景**：从另一台机器/另一个项目的备份中迁移数据到当前数据集。"
    )

    backup_dir = _path_browser("备份目录路径", "ingest_backup_dir", mode="dir")

    if backup_dir and Path(backup_dir).is_dir():
        info_file = Path(backup_dir) / "backup_info.json"
        if info_file.exists():
            import json
            backup_info = json.loads(info_file.read_text(encoding="utf-8"))
            st.success("✅ 检测到有效备份")
            col1, col2, col3 = st.columns(3)
            col1.metric("源数据集", backup_info.get("dataset_name", "未知"))
            col2.metric("样本数", backup_info.get("num_samples", "?"))
            col3.metric("图像文件", backup_info.get("images_copied", "?"))

            if backup_info.get("note"):
                st.info(f"备份备注: {backup_info['note']}")
            if backup_info.get("label_fields"):
                st.caption(f"标签字段: {', '.join(backup_info['label_fields'])}")
            if backup_info.get("tags"):
                st.caption(f"原始 Tags: {', '.join(backup_info['tags'][:20])}")
        else:
            st.warning("所选目录不是有效的备份目录（缺少 backup_info.json）")
            return
    else:
        st.info("请选择备份目录")
        return

    extra_tags_str = st.text_input(
        "额外标签（逗号分隔，可选）",
        key="ingest_backup_tags",
        placeholder="例如: imported_from_machine2, batch_03",
        help="除了保留备份中的原始 Tags 外，还可以额外添加标签用于区分来源",
    )
    extra_tags = [t.strip() for t in extra_tags_str.split(",") if t.strip()] if extra_tags_str else None

    if st.button("📦 开始导入", key="btn_ingest_backup", type="primary"):
        with st.spinner("正在从备份导入数据..."):
            try:
                result = advanced.import_from_backup(
                    backup_dir, ds, tags=extra_tags,
                )
                st.success("✅ 导入完成")
                st.json(result)
                if result["skipped_dup"] > 0:
                    st.info(f"跳过了 {result['skipped_dup']} 个重复文件（已存在于数据集中）")
            except Exception as e:
                st.error(f"导入失败: {e}")
