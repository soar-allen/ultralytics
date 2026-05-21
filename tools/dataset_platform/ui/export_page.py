"""数据导出页面：数据集导出、模型导出。"""
from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import exporter
from tools.dataset_platform import trainer
from tools.dataset_platform.ui.components import (
    _cached_label_classes,
    _cached_tags,
    _default_label_field,
    _get_ds,
    _get_info,
    _path_browser,
    _primary_action_section,
)

_DATA_TRAIN_DIR = Path(__file__).resolve().parents[3] / "data_train"
_EXPORT_FORMAT_SETTINGS_KEY = "dataset_export_format_settings"
_EXPORT_ACTIVE_FORMAT_KEY = "_export_active_format"
_EXPORT_PENDING_APPLY_KEY = "_export_pending_apply_format"


def _get_export_format_settings(ds, format_choice: str) -> dict:
    """Return saved dataset export settings for one export format."""
    settings = ds.info.get(_EXPORT_FORMAT_SETTINGS_KEY, {})
    if not isinstance(settings, dict):
        return {}
    value = settings.get(format_choice, {})
    return value if isinstance(value, dict) else {}


def _save_export_format_settings(ds, format_choice: str, settings: dict) -> None:
    """Persist the latest successful non-image export settings on the dataset."""
    if format_choice == "📷 纯图片":
        return
    all_settings = ds.info.get(_EXPORT_FORMAT_SETTINGS_KEY, {})
    if not isinstance(all_settings, dict):
        all_settings = {}
    all_settings[format_choice] = settings
    ds.info[_EXPORT_FORMAT_SETTINGS_KEY] = all_settings
    ds.save()


def _apply_export_format_settings(format_choice: str, settings: dict, all_tags: list[str]) -> None:
    """Reset shared widgets when switching export formats and apply saved defaults."""
    if st.session_state.get(_EXPORT_ACTIVE_FORMAT_KEY) == format_choice:
        return
    st.session_state[_EXPORT_ACTIVE_FORMAT_KEY] = format_choice
    st.session_state[_EXPORT_PENDING_APPLY_KEY] = format_choice

    if format_choice == "📷 纯图片":
        st.session_state["export_tags"] = []
        return

    st.session_state["export_tags"] = _filter_existing(settings.get("tags_filter", []), all_tags)
    st.session_state["export_include_background_train"] = bool(settings.get("include_background_train", True))
    split_mode = settings.get("split_mode", "按比例自动划分")
    if split_mode not in ("按比例自动划分", "指定单个 Split"):
        split_mode = "按比例自动划分"
    st.session_state["export_split_mode"] = split_mode
    st.session_state["split_train_pct"] = int(settings.get("train_pct", 80))
    st.session_state["split_valid_pct"] = int(settings.get("valid_pct", 10))
    st.session_state["split_test_pct"] = int(settings.get("test_pct", 10))
    single_split = settings.get("single_split", "train")
    st.session_state["export_split"] = single_split if single_split in ("train", "valid", "test") else "train"
    if settings.get("class_order_yaml"):
        st.session_state["export_class_order_yaml_input"] = settings["class_order_yaml"]
    else:
        st.session_state.pop("export_class_order_yaml_input", None)
    if settings.get("kp_field"):
        st.session_state["export_kp_field"] = settings["kp_field"]
    else:
        st.session_state.pop("export_kp_field", None)
    st.session_state["export_mixed_detection_classes"] = ", ".join(settings.get("mixed_detection_classes") or ["pallet"])
    st.session_state["export_mixed_edge_threshold"] = float(settings.get("edge_threshold", 5.0))
    st.session_state["export_mixed_bbox_margin"] = float(settings.get("bbox_margin", 0.0))
    st.session_state["export_edge_threshold"] = float(settings.get("edge_threshold", 5.0))
    st.session_state["export_bbox_margin"] = float(settings.get("bbox_margin", 0.0))
    if settings.get("obb_field"):
        st.session_state["export_obb_field"] = settings["obb_field"]
    else:
        st.session_state.pop("export_obb_field", None)


def _filter_existing(values, options: list[str]) -> list[str]:
    """Keep saved multiselect values that still exist in current options."""
    if not values:
        return []
    option_set = set(options)
    return [value for value in values if value in option_set]


def _select_index(options: list[str], value: str | None, default: int = 0) -> int:
    """Return a safe selectbox index."""
    if value in options:
        return options.index(value)
    return default if options else 0


def _is_applying_export_settings(format_choice: str) -> bool:
    return st.session_state.get(_EXPORT_PENDING_APPLY_KEY) == format_choice


def _render_export():
    st.header("📤 数据导出")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    export_sections = ["📦 数据集导出", "🔁 模型导出"]
    if st.session_state.get("data_export_section") not in (None, *export_sections):
        del st.session_state["data_export_section"]

    section = st.radio(
        "数据导出模块",
        export_sections,
        key="data_export_section",
        horizontal=True,
        label_visibility="collapsed",
    )

    if section.startswith("🔁"):
        from tools.dataset_platform.ui.training_page import _render_model_export

        _render_model_export(ds)
        return

    _render_dataset_export(ds)


def _render_dataset_export(ds):
    st.subheader("数据集导出")
    _DATA_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    default_output_dir = str(_DATA_TRAIN_DIR)
    if st.session_state.get("export_dir_input") in (None, "", str(Path.home() / "dataset_exports")):
        st.session_state["export_dir_input"] = default_output_dir

    format_choice = st.selectbox(
        "导出格式",
        [
            "YOLO Detect (纯框)",
            "YOLO Pose (关键点)",
            "YOLO 混合训练（pose）",
            "YOLO Pose (四边形转关键点)",
            "YOLO OBB (旋转框)",
            "📷 纯图片",
        ],
        key="export_format",
    )

    is_images_only = format_choice == "📷 纯图片"
    all_tags = _cached_tags(ds)
    saved_settings = _get_export_format_settings(ds, format_choice)
    _apply_export_format_settings(format_choice, saved_settings, all_tags)

    if format_choice == "YOLO Pose (四边形转关键点)":
        st.info("将 4 点多边形 (Polylines) 自动转换为 YOLO Pose 格式")
    if format_choice == "YOLO 混合训练（pose）":
        st.info("在四边形转关键点的基础上，将指定检测框类别写成 bbox + 全 0 关键点，用于 YOLO task=pose 训练")
    if is_images_only:
        st.info("仅导出图片文件，不包含标签和 data.yaml，不进行数据划分")

    output_dir = _path_browser(
        "输出目录",
        "export_dir",
        mode="dir",
        start_dir=default_output_dir,
    )

    # ── Tag 筛选（所有格式通用） ──
    selected_tags = []
    if all_tags:
        if "export_tags" in st.session_state:
            st.session_state["export_tags"] = _filter_existing(st.session_state["export_tags"], all_tags)
        selected_tags = st.multiselect("按 Tag 筛选样本（留空导出全部）", all_tags, key="export_tags")

    if selected_tags:
        export_view = ds.match_tags(selected_tags)
        st.info(f"已按 Tag 筛选：{', '.join(selected_tags)}，共 {len(export_view)} 个样本")
    else:
        export_view = ds

    background_formats = {"YOLO 混合训练（pose）", "YOLO Pose (四边形转关键点)"}
    include_background_train = False
    background_tag = "background"
    if format_choice in background_formats and background_tag in all_tags:
        include_background_train = st.checkbox(
            "将 background 标签图片全部加入 train 作为负样本",
            value=True,
            key="export_include_background_train",
            help="这些图片会强制进入 train/images，并生成空 label 文件，不参与 valid/test 划分。",
        )
        if include_background_train:
            bg_view = ds.match_tags(background_tag)
            export_ids = list(dict.fromkeys(list(export_view.values("id")) + list(bg_view.values("id"))))
            export_view = ds.select(export_ids)
            st.info(f"已加入 `{background_tag}` 背景图 {len(bg_view)} 张，导出时会全部放入 train。")

    # ── 以下配置仅 YOLO 格式需要 ──
    splits: str | dict[str, float] = "train"
    total_pct = 100
    label_field = "ground_truth"
    selected_classes: list[str] = []
    kp_field = ""
    mixed_det_field = "ground_truth"
    mixed_detection_classes: list[str] = []
    edge_threshold = 5.0
    bbox_margin = 0.0
    obb_field = ""
    class_order_yaml = ""
    locked_class_names: list[str] = []
    can_export = True

    if not is_images_only:
        split_mode = st.radio("数据划分方式", ["按比例自动划分", "指定单个 Split"], key="export_split_mode", horizontal=True)

        if split_mode == "按比例自动划分":
            st.caption("设置比例，三者之和应为 100%")
            col_t, col_v, col_te = st.columns(3)
            with col_t:
                train_pct = st.number_input("Train %", 0, 100, int(saved_settings.get("train_pct", 80)), key="split_train_pct")
            with col_v:
                valid_pct = st.number_input("Valid %", 0, 100, int(saved_settings.get("valid_pct", 10)), key="split_valid_pct")
            with col_te:
                test_pct = st.number_input("Test %", 0, 100, int(saved_settings.get("test_pct", 10)), key="split_test_pct")
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
        label_fields = info.get("label_fields", ["ground_truth"])
        label_field_label = "四边形字段" if format_choice == "YOLO 混合训练（pose）" else "标签字段"
        default_field = _default_label_field(
            ds,
            label_fields,
            label_type="polylines" if format_choice in {"YOLO 混合训练（pose）", "YOLO Pose (四边形转关键点)"} else None,
        )
        saved_label_field = saved_settings.get("label_field")
        if saved_label_field in label_fields:
            default_field = saved_label_field
        if _is_applying_export_settings(format_choice):
            if saved_label_field in label_fields:
                st.session_state["export_label_field"] = saved_label_field
            else:
                st.session_state.pop("export_label_field", None)
        elif st.session_state.get("export_label_field") not in (None, *label_fields):
            st.session_state.pop("export_label_field", None)
        default_idx = label_fields.index(default_field) if default_field in label_fields else 0
        label_field = st.selectbox(label_field_label, label_fields, index=default_idx, key="export_label_field")

        all_classes = _cached_label_classes(ds, label_field)
        if all_classes and format_choice != "YOLO 混合训练（pose）":
            default_classes = _filter_existing(saved_settings.get("classes", []), all_classes)
            if _is_applying_export_settings(format_choice):
                st.session_state["export_classes"] = default_classes
            elif "export_classes" in st.session_state:
                st.session_state["export_classes"] = _filter_existing(st.session_state["export_classes"], all_classes)
            selected_classes = st.multiselect(
                "选择导出类别（留空导出全部）",
                all_classes,
                default=default_classes,
                key="export_classes",
            )

        with st.expander("🔒 类别顺序锁定（增量训练推荐）", expanded=False):
            st.caption("选择上一版 data.yaml 后，旧类别 class id 保持不变，新类别追加到末尾。")
            class_order_yaml = _path_browser(
                "上一版 data.yaml", "export_class_order_yaml", mode="file",
                file_extensions=(".yaml", ".yml"),
                start_dir=default_output_dir,
            )
            if class_order_yaml:
                locked_class_names = exporter.load_class_names_from_yaml(class_order_yaml)
                if locked_class_names:
                    st.success(f"已读取 {len(locked_class_names)} 个旧类别: {', '.join(locked_class_names)}")
                else:
                    st.warning("未能从该 YAML 读取类别顺序，将使用当前导出类别顺序")

        if format_choice == "YOLO Pose (关键点)":
            kp_field = st.text_input(
                "关键点字段名",
                value=saved_settings.get("kp_field") or f"{label_field}_keypoints",
                key="export_kp_field",
            )

        if format_choice == "YOLO 混合训练（pose）":
            st.markdown("**混合训练参数**")
            detection_fields = [f for f in label_fields if dm.get_field_label_type(ds, f) == "detections"]
            mixed_det_options = detection_fields or label_fields
            saved_mixed_det_field = saved_settings.get("mixed_det_field") or "ground_truth"
            if _is_applying_export_settings(format_choice):
                if saved_mixed_det_field in mixed_det_options:
                    st.session_state["export_mixed_det_field"] = saved_mixed_det_field
                else:
                    st.session_state.pop("export_mixed_det_field", None)
            elif st.session_state.get("export_mixed_det_field") not in (None, *mixed_det_options):
                st.session_state.pop("export_mixed_det_field", None)
            mixed_det_field = st.selectbox(
                "检测框字段",
                mixed_det_options,
                index=_select_index(
                    mixed_det_options,
                    saved_mixed_det_field,
                ),
                key="export_mixed_det_field",
            )
            det_text = st.text_input(
                "只检测类别（逗号分隔）",
                value=", ".join(saved_settings.get("mixed_detection_classes") or ["pallet"]),
                key="export_mixed_detection_classes",
                help="这些类别会使用检测框导出，关键点全部写为 0。例如 pallet。",
            )
            mixed_detection_classes = [c.strip() for c in det_text.split(",") if c.strip()]
            st.markdown("**可见性参数**")
            col_e, col_m = st.columns(2)
            with col_e:
                edge_threshold = st.number_input(
                    "边界阈值 (像素)", 0.0, 100.0, float(saved_settings.get("edge_threshold", 5.0)), 1.0,
                    key="export_mixed_edge_threshold",
                )
            with col_m:
                bbox_margin = st.number_input(
                    "BBox 外扩边距 (归一化)", 0.0, 0.2, float(saved_settings.get("bbox_margin", 0.0)), 0.005,
                    key="export_mixed_bbox_margin",
                )
            pose_classes = all_classes
            det_all_classes = _cached_label_classes(ds, mixed_det_field)
            det_classes = [c for c in det_all_classes if not mixed_detection_classes or c in mixed_detection_classes]
            combined_classes = sorted(set(pose_classes + det_classes))
            if combined_classes:
                default_mixed_classes = _filter_existing(saved_settings.get("classes", []), combined_classes)
                if _is_applying_export_settings(format_choice):
                    st.session_state["export_mixed_classes"] = default_mixed_classes
                elif "export_mixed_classes" in st.session_state:
                    st.session_state["export_mixed_classes"] = _filter_existing(
                        st.session_state["export_mixed_classes"], combined_classes,
                    )
                selected_classes = st.multiselect(
                    "选择导出类别（留空导出全部）",
                    combined_classes,
                    default=default_mixed_classes,
                    key="export_mixed_classes",
                )

        if format_choice == "YOLO Pose (四边形转关键点)":
            st.markdown("**可见性参数**")
            col_e, col_m = st.columns(2)
            with col_e:
                edge_threshold = st.number_input(
                    "边界阈值 (像素)", 0.0, 100.0, float(saved_settings.get("edge_threshold", 5.0)), 1.0,
                    key="export_edge_threshold",
                )
            with col_m:
                bbox_margin = st.number_input(
                    "BBox 外扩边距 (归一化)", 0.0, 0.2, float(saved_settings.get("bbox_margin", 0.0)), 0.005,
                    key="export_bbox_margin",
                )

        if format_choice == "YOLO OBB (旋转框)":
            obb_field = st.text_input(
                "OBB 字段名（Polylines, 可选）",
                value=saved_settings.get("obb_field", ""),
                key="export_obb_field",
            )

        if isinstance(splits, dict) and split_mode == "按比例自动划分" and total_pct != 100:
            can_export = False

    st.session_state.pop(_EXPORT_PENDING_APPLY_KEY, None)

    disabled_reason = ""
    if not output_dir:
        disabled_reason = "请指定输出目录"
    elif not can_export:
        disabled_reason = "请调整数据划分比例，总和需要为 100%"

    summary = {
        "导出格式": format_choice,
        "输出目录": output_dir,
        "样本范围": f"{len(export_view)} 个样本",
        "Tags 筛选": selected_tags or "全部",
    }
    if not is_images_only:
        summary.update({
            "标签字段": label_field,
            "导出类别": selected_classes or "全部",
            "数据划分": splits,
            "类别顺序锁定": class_order_yaml or "未启用",
        })

    if _primary_action_section(
        "📦 开始导出",
        "btn_export",
        summary,
        disabled=bool(disabled_reason),
        disabled_reason=disabled_reason,
        preview_view=export_view,
        preview_key="btn_export_preview",
        preview_label="👁️ 展示导出样本",
    ):

        with st.spinner("导出中..."):
            try:
                _prepare_dataset_export_dir(output_dir)
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
                        class_names=locked_class_names or None,
                    )
                elif format_choice == "YOLO Pose (关键点)":
                    result = exporter.export_yolo_pose(
                        export_view, output_dir, det_field=label_field, kp_field=kp_field, classes=classes, splits=splits,
                        class_names=locked_class_names or None,
                    )
                elif format_choice == "YOLO 混合训练（pose）":
                    result = exporter.export_yolo_mixed_pose(
                        export_view,
                        output_dir,
                        pose_field=label_field,
                        det_field=mixed_det_field,
                        detection_classes=mixed_detection_classes or ["pallet"],
                        classes=classes,
                        splits=splits,
                        edge_threshold=edge_threshold,
                        bbox_margin=bbox_margin,
                        class_names=locked_class_names or None,
                        include_background=include_background_train,
                        background_tag=background_tag,
                    )
                elif format_choice == "YOLO Pose (四边形转关键点)":
                    result = exporter.export_yolo_pose_from_polylines(
                        export_view, output_dir, label_field=label_field, classes=classes, splits=splits,
                        edge_threshold=edge_threshold, bbox_margin=bbox_margin,
                        class_names=locked_class_names or None,
                        include_background=include_background_train,
                        background_tag=background_tag,
                    )
                elif format_choice == "YOLO OBB (旋转框)":
                    result = exporter.export_yolo_obb(
                        export_view, output_dir, label_field=label_field,
                        obb_field=obb_field if obb_field else None, classes=classes, splits=splits,
                        class_names=locked_class_names or None,
                    )
                else:
                    result = {}

                _save_export_format_settings(
                    ds,
                    format_choice,
                    {
                        "tags_filter": selected_tags,
                        "include_background_train": include_background_train,
                        "split_mode": split_mode,
                        "train_pct": st.session_state.get("split_train_pct", 80),
                        "valid_pct": st.session_state.get("split_valid_pct", 10),
                        "test_pct": st.session_state.get("split_test_pct", 10),
                        "single_split": st.session_state.get("export_split", "train"),
                        "label_field": label_field,
                        "classes": selected_classes,
                        "class_order_yaml": class_order_yaml,
                        "kp_field": kp_field,
                        "mixed_det_field": mixed_det_field,
                        "mixed_detection_classes": mixed_detection_classes,
                        "edge_threshold": edge_threshold,
                        "bbox_margin": bbox_margin,
                        "obb_field": obb_field,
                    },
                )

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
                    "mixed_det_field": mixed_det_field if format_choice == "YOLO 混合训练（pose）" else None,
                    "kp_field": kp_field or None,
                    "classes": classes,
                    "mixed_detection_classes": mixed_detection_classes or None,
                    "include_background_train": include_background_train,
                    "background_tag": background_tag if include_background_train else None,
                    "class_order_yaml": class_order_yaml or None,
                    "locked_class_names": locked_class_names or None,
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


def _prepare_dataset_export_dir(output_dir: str | Path) -> None:
    """Clear data_train before dataset export so different export formats replace old files."""
    target = Path(output_dir).expanduser().resolve()
    data_train = _DATA_TRAIN_DIR.resolve()
    if target != data_train:
        return

    target.mkdir(parents=True, exist_ok=True)
    for child in target.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
