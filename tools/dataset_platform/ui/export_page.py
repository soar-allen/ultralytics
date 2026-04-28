"""数据导出页面：多格式 YOLO 导出 + 自动衔接训练。"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import exporter
from tools.dataset_platform import trainer
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _get_info, _path_browser


def _render_export():
    st.header("📤 多格式导出")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    format_choice = st.selectbox(
        "导出格式",
        ["YOLO Detect (纯框)", "YOLO Pose (关键点)", "YOLO Pose (四边形转关键点)", "YOLO OBB (旋转框)", "📷 纯图片"],
        key="export_format",
    )

    is_images_only = format_choice == "📷 纯图片"

    if format_choice == "YOLO Pose (四边形转关键点)":
        st.info("将 4 点多边形 (Polylines) 自动转换为 YOLO Pose 格式")
    if is_images_only:
        st.info("仅导出图片文件，不包含标签和 data.yaml，不进行数据划分")

    output_dir = _path_browser("输出目录", "export_dir", mode="dir", start_dir=CONFIG.default_export_dir)

    # ── Tag 筛选（所有格式通用） ──
    all_tags = sorted(ds.distinct("tags"))
    selected_tags = []
    if all_tags:
        selected_tags = st.multiselect("按 Tag 筛选样本（留空导出全部）", all_tags, key="export_tags")

    if selected_tags:
        export_view = ds.match_tags(selected_tags)
        st.info(f"已按 Tag 筛选：{', '.join(selected_tags)}，共 {len(export_view)} 个样本")
    else:
        export_view = ds

    # ── 以下配置仅 YOLO 格式需要 ──
    splits: str | dict[str, float] = "train"
    total_pct = 100
    label_field = "ground_truth"
    selected_classes: list[str] = []
    kp_field = ""
    edge_threshold = 5.0
    bbox_margin = 0.0
    obb_field = ""
    can_export = True

    if not is_images_only:
        split_mode = st.radio("数据划分方式", ["按比例自动划分", "指定单个 Split"], key="export_split_mode", horizontal=True)

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
            splits = {}
            if train_pct > 0:
                splits["train"] = train_pct / 100.0
            if valid_pct > 0:
                splits["valid"] = valid_pct / 100.0
            if test_pct > 0:
                splits["test"] = test_pct / 100.0
        else:
            single_split = st.selectbox("Split 名称", ["train", "valid", "test"], key="export_split")
            splits = single_split

        info = _get_info(ds)
        label_field = st.selectbox("标签字段", info.get("label_fields", ["ground_truth"]), key="export_label_field")

        all_classes = dm.get_label_classes(ds, label_field)
        if all_classes:
            selected_classes = st.multiselect("选择导出类别（留空导出全部）", all_classes, key="export_classes")

        if format_choice == "YOLO Pose (关键点)":
            kp_field = st.text_input("关键点字段名", value=f"{label_field}_keypoints", key="export_kp_field")

        if format_choice == "YOLO Pose (四边形转关键点)":
            st.markdown("**可见性参数**")
            col_e, col_m = st.columns(2)
            with col_e:
                edge_threshold = st.number_input("边界阈值 (像素)", 0.0, 100.0, 5.0, 1.0, key="export_edge_threshold")
            with col_m:
                bbox_margin = st.number_input("BBox 外扩边距 (归一化)", 0.0, 0.2, 0.0, 0.005, key="export_bbox_margin")

        if format_choice == "YOLO OBB (旋转框)":
            obb_field = st.text_input("OBB 字段名（Polylines, 可选）", key="export_obb_field")

        if isinstance(splits, dict) and split_mode == "按比例自动划分" and total_pct != 100:
            can_export = False

    if st.button("📦 开始导出", key="btn_export", disabled=not can_export):
        if not output_dir:
            st.error("请指定输出目录")
            return

        with st.spinner("导出中..."):
            try:
                if is_images_only:
                    result = exporter.export_images_only(export_view, output_dir)
                    st.success("✅ 导出完成")
                    st.json(result)

                    export_record = {
                        "format": format_choice,
                        "output_dir": output_dir,
                        "tags_filter": selected_tags or None,
                        "num_samples": result.get("exported", len(export_view)),
                    }
                    history = ds.info.get("export_history", [])
                    history.append(export_record)
                    ds.info["export_history"] = history
                    ds.save()
                    return

                classes = selected_classes or None

                if format_choice == "YOLO Detect (纯框)":
                    result = exporter.export_yolo_detect(
                        export_view, output_dir, label_field=label_field, classes=classes, splits=splits,
                    )
                elif format_choice == "YOLO Pose (关键点)":
                    result = exporter.export_yolo_pose(
                        export_view, output_dir, det_field=label_field, kp_field=kp_field, classes=classes, splits=splits,
                    )
                elif format_choice == "YOLO Pose (四边形转关键点)":
                    result = exporter.export_yolo_pose_from_polylines(
                        export_view, output_dir, label_field=label_field, classes=classes, splits=splits,
                        edge_threshold=edge_threshold, bbox_margin=bbox_margin,
                    )
                elif format_choice == "YOLO OBB (旋转框)":
                    result = exporter.export_yolo_obb(
                        export_view, output_dir, label_field=label_field,
                        obb_field=obb_field if obb_field else None, classes=classes, splits=splits,
                    )
                else:
                    result = {}

                st.success("✅ 导出完成")
                st.json(result)

                data_yaml = result.get("data_yaml") or trainer.find_data_yaml(output_dir)
                if data_yaml:
                    st.session_state["last_export_data_yaml"] = data_yaml
                    st.session_state["last_export_dir"] = output_dir
                    st.info(f"📄 data.yaml 路径: `{data_yaml}`")
                    st.success("💡 可直接前往「训练管理」Tab 一键开始训练，导出路径已自动填入。")

                export_record = {
                    "format": format_choice,
                    "output_dir": output_dir,
                    "data_yaml": data_yaml,
                    "label_field": label_field,
                    "classes": classes,
                    "splits": str(splits),
                    "tags_filter": selected_tags or None,
                    "num_samples": result.get("total_samples", len(export_view)),
                }
                history = ds.info.get("export_history", [])
                history.append(export_record)
                ds.info["export_history"] = history
                ds.save()
            except Exception as e:
                st.error(f"导出失败: {e}")
