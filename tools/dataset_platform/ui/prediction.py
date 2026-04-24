"""
自动预标注独立页面。

支持多种预标注模式：
  1. YOLO Pose → 四角多边形：用关键点检测模型识别四个角点，连接为闭合 Polyline（多类别）
  2. SAM3 辅助标注：支持 text prompt / box prompt，输出四角多边形或 bbox
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import processor
from tools.dataset_platform.ui.components import _get_ds, _get_info, _path_browser

_MODEL_DIR = Path.home() / ".dataset_platform" / "model"
_MODEL_EXTENSIONS = (".pt", ".pth", ".onnx", ".engine")


def _list_preset_models() -> list[str]:
    """扫描 ~/.dataset_platform/model/ 下的权重文件。"""
    if not _MODEL_DIR.is_dir():
        return []
    models = []
    for f in sorted(_MODEL_DIR.iterdir()):
        if f.suffix.lower() in _MODEL_EXTENSIONS and f.is_file():
            models.append(str(f))
    return models


def _model_selector(label: str, key_prefix: str) -> str:
    """模型选择器：预设目录下拉 + 手动路径浏览。"""
    presets = _list_preset_models()

    source = st.radio(
        "模型来源", ["预设模型目录", "手动选择路径"],
        key=f"{key_prefix}_source", horizontal=True,
        help=f"预设目录: `{_MODEL_DIR}`",
    )

    if source == "预设模型目录":
        if presets:
            selected = st.selectbox(
                label, presets,
                format_func=lambda p: Path(p).name,
                key=f"{key_prefix}_preset",
            )
            return selected
        else:
            st.info(f"预设目录 `{_MODEL_DIR}` 下无模型文件，请先放入 .pt 权重或切换到手动选择")
            return ""
    else:
        return _path_browser(
            label, f"{key_prefix}_browser", mode="file",
            file_extensions=_MODEL_EXTENSIONS,
        )


def _field_selector(ds, key_prefix: str, default: str = "ground_truth",
                    field_type_filter: str | None = None) -> str:
    """字段选择器：下拉选择已有字段 / 手动输入新字段。

    Args:
        field_type_filter: "polylines" / "detections" / None (不过滤)
    """
    import fiftyone as fo

    _FILTER_MAP = {
        "polylines": fo.Polylines,
        "detections": fo.Detections,
    }

    schema = ds.get_field_schema()
    label_fields = []
    for fn, field in schema.items():
        if fn.startswith("_"):
            continue
        if not hasattr(field, "document_type") or not field.document_type:
            continue
        if field_type_filter and field_type_filter in _FILTER_MAP:
            if not issubclass(field.document_type, _FILTER_MAP[field_type_filter]):
                continue
        label_fields.append(fn)

    mode = st.radio(
        "字段来源", ["输入新字段名", "选择已有字段"],
        key=f"{key_prefix}_field_mode", horizontal=True,
    )

    if mode == "选择已有字段":
        if label_fields:
            idx = label_fields.index(default) if default in label_fields else 0
            return st.selectbox(
                "输出字段", label_fields, index=idx,
                key=f"{key_prefix}_field_select",
            )
        else:
            st.warning("当前无匹配的标签字段，请输入新字段名")
            return st.text_input("输出字段", value=default, key=f"{key_prefix}_field_input")
    else:
        return st.text_input("输出字段", value=default, key=f"{key_prefix}_field_input")


def _render_scope_selector(ds, unlabeled_view, pred_field: str, skip_labeled: bool):
    """预标注范围选择，返回 (view_or_none, display_count)。"""
    pred_scope = st.radio(
        "选择范围", ["仅无标注样本", "整个数据集", "按 Tags 筛选"],
        key="pred_scope", horizontal=True,
    )

    pred_view = None
    if pred_scope == "仅无标注样本":
        pred_view = unlabeled_view
        st.info(f"将对 **{len(pred_view)}** 个无标注样本进行预标注")
    elif pred_scope == "按 Tags 筛选":
        available_tags = ds.distinct("tags")
        if available_tags:
            pred_tags = st.multiselect("选择 Tags", available_tags, key="pred_tags")
            if pred_tags:
                pred_view = ds.match_tags(pred_tags)
                st.info(f"将对 **{len(pred_view)}** 个匹配样本进行预标注")
        else:
            st.info("当前数据集没有 Tags")
    else:
        st.info(f"将对整个数据集的 **{len(ds)}** 个样本进行预标注")

    if skip_labeled and pred_scope != "仅无标注样本":
        st.caption("已启用去重：目标字段中已有标注的样本将被自动跳过")

    return pred_view


def _render_auto_predict_page():
    """独立的自动预标注页面。"""
    st.header("🤖 自动预标注")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    st.caption(
        "使用模型自动检测并标注，生成的标注可推送到 CVAT 辅助人工标注。"
    )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    info = _get_info(ds)
    col_overview, col_unlabeled = st.columns(2)
    with col_overview:
        st.metric("📷 数据集样本数", info["num_samples"])
    with col_unlabeled:
        unlabeled = processor.find_unlabeled_samples(ds)
        st.metric("🔲 无标注样本数", len(unlabeled))

    st.markdown("---")

    mode = st.radio(
        "**预标注模式**",
        ["🦴 YOLO Pose → 四角多边形", "🎯 SAM3 辅助标注", "🏷️ SAM3 标注标签"],
        key="pred_mode", horizontal=True,
    )

    if mode.startswith("🦴"):
        _render_pose_mode(ds, info, unlabeled)
    elif mode.startswith("🎯"):
        _render_sam3_mode(ds, info, unlabeled)
    else:
        _render_sam3_tag_mode(ds, info, unlabeled)


# -----------------------------------------------------------------------
# YOLO Pose
# -----------------------------------------------------------------------

def _render_pose_mode(ds, info: dict, unlabeled):
    """YOLO Pose 模式：关键点 → 四角多边形（多类别）。"""
    st.markdown("### 🦴 YOLO Pose → 四角多边形")
    st.caption(
        "使用 YOLO Pose 模型检测四个关键点（左上→右上→右下→左下），自动连接成闭合四边形 Polyline。\n"
        "模型输出的类别名自动使用，支持多类别检测。"
    )

    col_model, col_config = st.columns(2)

    with col_model:
        st.subheader("模型配置")
        model_path = _model_selector("YOLO Pose 权重", "pose_model")

    with col_config:
        st.subheader("标注配置")
        pred_field = _field_selector(ds, "pose", default="ground_truth")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="pose_conf")

    # 类别过滤（可选）
    with st.expander("类别过滤（可选）"):
        st.caption("留空表示保留模型输出的所有类别")
        filter_text = st.text_input(
            "仅保留类别（逗号分隔）", value="", key="pose_filter_cls",
            help="例如 'pallet,box'。留空则保留所有类别。",
        )
        filter_classes = [c.strip() for c in filter_text.split(",") if c.strip()] or None

    st.subheader("去重设置")
    col_dedup1, col_dedup2 = st.columns(2)
    with col_dedup1:
        skip_labeled = st.checkbox(
            "完全跳过已有标注的样本", value=False, key="pose_skip",
            help="勾选后，目标字段中已有任何 Polyline 的样本将被整体跳过",
        )
    with col_dedup2:
        nms_iou = st.slider(
            "标注 NMS IoU 阈值", 0.0, 1.0, 0.5, 0.05, key="pose_nms",
            help="同一样本中，新预测与已有同类别标注的 IoU 超过此值时，新标注将被抑制。\n"
                 "设为 0 = 不做去重，设为 1.0 = 仅完全重合时去重",
        )

    st.markdown("---")
    st.subheader("预标注范围")
    pred_view = _render_scope_selector(ds, unlabeled, pred_field, skip_labeled)

    st.markdown("---")
    if st.button("🚀 开始 Pose 预标注", key="btn_pose_predict", type="primary"):
        if not model_path or not Path(model_path).exists():
            st.error("请选择有效的模型权重文件")
            return

        target = pred_view if pred_view is not None else None
        target_count = len(target) if target is not None else len(ds)
        with st.spinner(f"使用 YOLO Pose 模型对 {target_count} 个样本检测四角多边形..."):
            stats = processor.auto_predict_pose_to_polyline(
                ds, model_path, pred_field=pred_field,
                conf_threshold=conf, filter_classes=filter_classes,
                skip_labeled=skip_labeled, nms_iou=nms_iou, view=target,
            )
        st.success("Pose 预标注完成")
        _show_predict_stats(stats)


# -----------------------------------------------------------------------
# SAM3
# -----------------------------------------------------------------------

def _render_sam3_mode(ds, info: dict, unlabeled):
    """SAM3 辅助标注：三种提示策略，两种输出格式。"""
    st.markdown("### 🎯 SAM3 辅助标注")
    st.caption(
        "支持三种提示策略和两种输出格式，根据场景选择最佳方案。"
    )

    col_model, col_output = st.columns(2)

    with col_model:
        st.subheader("模型配置")
        model_path = _model_selector("SAM3 权重", "sam3_model")
        half = st.checkbox("FP16 半精度加速", value=True, key="sam3_half")

    with col_output:
        st.subheader("输出配置")
        output_mode = st.radio(
            "输出格式", ["四角多边形 (Polyline)", "矩形框 (BBox)"],
            key="sam3_output_mode", horizontal=True,
        )
        is_polyline = output_mode.startswith("四角")
        pred_field = _field_selector(ds, "sam3", default="predict", field_type_filter=None)
        label_name = st.text_input(
            "输出类别名", value="pallet", key="sam3_label_name",
            help="所有输出标注统一使用此类别名",
        )
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="sam3_conf")

    st.markdown("---")
    st.subheader("提示策略")

    prompt_choice = st.radio(
        "选择提示策略",
        [
            "🔗 联合提示 (文本+范例) — 推荐",
            "📝 文本概念 (Text Prompt)",
            "📦 图像范例 (Box Example)",
            "🔍 视觉分割 (Box Visual, SAM2 兼容)",
        ],
        key="sam3_prompt_mode",
        help="联合提示：同时用文字描述和已有标注范例，准确率最高\n"
             "文本概念：仅用文字描述目标\n"
             "图像范例：仅用已有标注作为视觉范例\n"
             "视觉分割：SAM2 兼容模式，对每个标注逐一分割",
    )

    text_prompts = None
    box_source_field = None
    box_source_labels = None
    box_source_tags: list[str] = []
    box_expand_ratio = 0.5
    box_mask_strategy = "smallest_covering"

    if prompt_choice.startswith("🔗"):
        p_mode = "combined"
        st.info(
            "**联合提示 (推荐)**：同时向 SAM3SemanticPredictor 传入文字描述和已有标注范例，\n"
            "SAM3 将文字概念与视觉范例结合理解目标，识别准确率高于单独使用任一方式。\n"
            "当某张图没有范例时，自动退化为纯文本提示。"
        )
        text_input = st.text_area(
            "文本提示 (每行一个提示词)",
            value="pallet",
            key="sam3_text",
            help="目标的文字描述，如 'pallet', 'wooden pallet' 等",
        )
        text_prompts = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        if text_prompts:
            st.caption(f"文本提示词: {text_prompts}")
        box_source_field, box_source_labels, box_source_tags = _render_box_source_selector(ds)
        box_expand_ratio = st.slider(
            "范例 BBox 扩展比例", 0.0, 2.0, 0.5, 0.1, key="sam3_box_expand",
            help="扩展已有标注的 bbox，使 SAM3 能看到更大的上下文。",
        )

    elif prompt_choice.startswith("📝"):
        p_mode = "text"
        st.info(
            "**文本概念分割**：使用 SAM3SemanticPredictor 查找图像中所有匹配文字描述的目标。\n"
            "适合：知道目标名称，一次性检测所有实例。"
        )
        text_input = st.text_area(
            "文本提示 (每行一个提示词)",
            value="pallet",
            key="sam3_text",
            help="SAM3 概念分割文本提示，如 'pallet', 'wooden pallet' 等",
        )
        text_prompts = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        if text_prompts:
            st.caption(f"提示词: {text_prompts}")

    elif prompt_choice.startswith("📦"):
        p_mode = "box_example"
        st.info(
            "**图像范例匹配**：将已有标注的 bbox 作为视觉范例传入 SAM3SemanticPredictor，\n"
            "SAM3 会理解『这个框里的东西长什么样』，然后找出图像中**所有相似实例**。\n"
            "适合：已有局部标注（如托盘前表面），需要找到完整目标。"
        )
        box_source_field, box_source_labels, box_source_tags = _render_box_source_selector(ds)
        box_expand_ratio = st.slider(
            "范例 BBox 扩展比例", 0.0, 2.0, 0.5, 0.1, key="sam3_box_expand",
            help="扩展已有标注的 bbox，使 SAM3 能看到更大的上下文。\n"
                 "例如 0.5 = 宽高各扩展 50%。对于需要识别超出已标注区域的目标很有效。",
        )

    else:
        p_mode = "box_visual"
        st.info(
            "**SAM2 兼容视觉分割**：对每个已有标注逐一调用 SAM (SAM2 模式)，\n"
            "在扩展的 bbox 区域内进行分割。输出与来源标注 1:1 对应。\n"
            "适合：需要精确的逐标注分割。"
        )
        box_source_field, box_source_labels, box_source_tags = _render_box_source_selector(ds)

        col_expand, col_strategy = st.columns(2)
        with col_expand:
            box_expand_ratio = st.slider(
                "BBox 扩展比例", 0.0, 2.0, 0.5, 0.1, key="sam3_box_expand_v",
                help="扩展来源标注的 bbox。目标超出原标注时需要加大此值。",
            )
        with col_strategy:
            box_mask_strategy = st.selectbox(
                "Mask 选择策略", ["smallest_covering", "largest"],
                key="sam3_mask_strategy",
                help="smallest_covering: 包含原标注且面积最小的 mask（推荐）\n"
                     "largest: 面积最大的 mask",
            )

    st.markdown("---")
    st.subheader("去重设置")
    col_dedup1, col_dedup2 = st.columns(2)
    with col_dedup1:
        skip_labeled = st.checkbox(
            "完全跳过已有标注的样本", value=False, key="sam3_skip",
            help="勾选后，目标字段中已有标注的样本将被整体跳过",
        )
    with col_dedup2:
        nms_iou = st.slider(
            "标注 NMS IoU 阈值", 0.0, 1.0, 0.5, 0.05, key="sam3_nms",
            help="同一样本中，新预测与已有同类别标注的 IoU 超过此值时，新标注将被抑制",
        )

    st.markdown("---")
    st.subheader("失败处理")
    col_tag1, col_tag2 = st.columns(2)
    with col_tag1:
        fail_tag = st.text_input(
            "预测失败标签", value="sam3_failed", key="sam3_fail_tag",
            help="SAM3 对某个样本预测失败或无输出时，自动打上此标签（留空则不打）",
        )
    with col_tag2:
        st.markdown("&nbsp;")
        _render_clear_tag_section(ds)

    st.markdown("---")
    st.subheader("预标注范围")
    pred_view = _render_scope_selector(ds, unlabeled, pred_field, skip_labeled)

    st.markdown("---")
    if st.button("🚀 开始 SAM3 预标注", key="btn_sam3_predict", type="primary"):
        if not model_path or not Path(model_path).exists():
            st.error("请选择有效的模型权重文件")
            return
        if p_mode == "text" and not text_prompts:
            st.error("请输入至少一个文本提示")
            return
        if p_mode == "combined" and not text_prompts and not box_source_field:
            st.error("联合提示模式需要至少提供文本提示或来源字段之一")
            return
        if p_mode in ("box_example", "box_visual") and not box_source_field:
            st.error("请选择来源字段")
            return

        o_mode = "polyline" if is_polyline else "bbox"
        target = pred_view if pred_view is not None else None
        target_count = len(target) if target is not None else len(ds)

        strategy_names = {
            "text": "文本概念", "box_example": "图像范例",
            "combined": "联合提示", "box_visual": "视觉分割",
        }
        with st.spinner(f"使用 SAM3 ({strategy_names[p_mode]} → {o_mode}) 对 {target_count} 个样本标注..."):
            stats = processor.auto_predict_sam3(
                ds, model_path,
                pred_field=pred_field,
                conf_threshold=conf,
                output_mode=o_mode,
                prompt_mode=p_mode,
                text_prompts=text_prompts,
                box_source_field=box_source_field,
                box_source_labels=box_source_labels,
                box_source_tags=box_source_tags,
                box_expand_ratio=box_expand_ratio,
                box_mask_strategy=box_mask_strategy,
                label_name=label_name,
                skip_labeled=skip_labeled,
                nms_iou=nms_iou,
                fail_tag=fail_tag.strip(),
                view=target,
                half=half,
            )
        st.success("SAM3 预标注完成")
        _show_predict_stats(stats)


# -----------------------------------------------------------------------
# SAM3 标注标签
# -----------------------------------------------------------------------

def _render_sam3_tag_mode(ds, info: dict, unlabeled):
    """SAM3 标注标签：仅识别是否存在目标并为图片打标签，不生成标注。"""
    st.markdown("### 🏷️ SAM3 标注标签")
    st.caption(
        "使用 SAM3 识别图片中是否存在你指定的目标，**不生成标注**，仅为图片打上相应标签。\n"
        "适合场景：快速分类筛选、给数据集分批、标记含特定目标的样本。"
    )

    col_model, col_tag = st.columns(2)

    with col_model:
        st.subheader("模型配置")
        model_path = _model_selector("SAM3 权重", "sam3_tag_model")
        half = st.checkbox("FP16 半精度加速", value=True, key="sam3_tag_half")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="sam3_tag_conf")

    with col_tag:
        st.subheader("标签配置")
        found_tag = st.text_input(
            "识别成功标签", value="sam3_found", key="sam3_tag_found",
            help="SAM3 检测到目标时为图片打上此标签",
        )
        not_found_tag = st.text_input(
            "未识别标签（可选）", value="", key="sam3_tag_not_found",
            help="SAM3 未检测到目标时打上此标签。留空则不打",
        )
        fail_tag = st.text_input(
            "预测失败标签", value="sam3_tag_failed", key="sam3_tag_fail",
            help="SAM3 预测出错时打上此标签。留空则不打",
        )

    st.markdown("---")
    st.subheader("提示策略")

    tag_prompt_choice = st.radio(
        "选择提示策略",
        [
            "🔗 联合提示 (文本+范例) — 推荐",
            "📝 文本概念 (Text Prompt)",
            "📦 图像范例 (Box Example)",
        ],
        key="sam3_tag_prompt_mode", horizontal=True,
    )

    text_prompts = None
    box_source_field = None
    box_source_labels = None
    box_source_tags: list[str] = []
    box_expand_ratio = 0.5

    if tag_prompt_choice.startswith("🔗"):
        st.info("同时使用文字描述和已有标注范例作为提示，当无范例时退化为纯文本提示。")
        text_input = st.text_area(
            "文本提示 (每行一个提示词)", value="pallet",
            key="sam3_tag_text_combined",
        )
        text_prompts = [line.strip() for line in text_input.strip().split("\n") if line.strip()]
        box_source_field, box_source_labels, box_source_tags = _render_box_source_selector(
            ds, key_prefix="sam3_tag_box",
        )
        box_expand_ratio = st.slider(
            "范例 BBox 扩展比例", 0.0, 2.0, 0.5, 0.1, key="sam3_tag_expand",
        )

    elif tag_prompt_choice.startswith("📝"):
        st.info("仅使用文字描述来检测目标是否存在。")
        text_input = st.text_area(
            "文本提示 (每行一个提示词)", value="pallet",
            key="sam3_tag_text",
        )
        text_prompts = [line.strip() for line in text_input.strip().split("\n") if line.strip()]

    else:
        st.info("使用已有标注作为视觉范例来检测目标。")
        box_source_field, box_source_labels, box_source_tags = _render_box_source_selector(
            ds, key_prefix="sam3_tag_box",
        )
        box_expand_ratio = st.slider(
            "范例 BBox 扩展比例", 0.0, 2.0, 0.5, 0.1, key="sam3_tag_expand",
        )

    st.markdown("---")
    st.subheader("识别范围")

    tag_scope = st.radio(
        "选择范围", ["整个数据集", "按 Tags 筛选"],
        key="sam3_tag_scope", horizontal=True,
    )
    tag_view = None
    if tag_scope == "按 Tags 筛选":
        available_tags = ds.distinct("tags")
        if available_tags:
            sel_tags = st.multiselect("选择 Tags", available_tags, key="sam3_tag_filter_tags")
            if sel_tags:
                tag_view = ds.match_tags(sel_tags)
                st.info(f"将对 **{len(tag_view)}** 个匹配样本进行识别")
        else:
            st.info("当前数据集没有 Tags")
    else:
        st.info(f"将对整个数据集的 **{len(ds)}** 个样本进行识别")

    st.markdown("---")
    if st.button("🚀 开始 SAM3 标签识别", key="btn_sam3_tag_start", type="primary"):
        if not model_path or not Path(model_path).exists():
            st.error("请选择有效的模型权重文件")
            return
        if not text_prompts and not box_source_field:
            st.error("请至少提供文本提示或来源字段之一")
            return

        target = tag_view if tag_view is not None else None
        target_count = len(target) if target is not None else len(ds)
        with st.spinner(f"使用 SAM3 对 {target_count} 个样本进行目标识别..."):
            stats = processor.auto_tag_sam3(
                ds, model_path,
                text_prompts=text_prompts,
                box_source_field=box_source_field,
                box_source_labels=box_source_labels,
                box_source_tags=box_source_tags,
                box_expand_ratio=box_expand_ratio,
                conf_threshold=conf,
                found_tag=found_tag.strip(),
                not_found_tag=not_found_tag.strip(),
                fail_tag=fail_tag.strip(),
                view=target,
                half=half,
            )
        st.success("SAM3 标签识别完成")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("处理图像", stats.get("images_processed", 0))
        col_s2.metric("识别到目标", stats.get("found", 0))
        col_s3.metric("未识别到", stats.get("not_found", 0))
        col_s4.metric("失败/报错", stats.get("errors", 0))

        with st.expander("详细统计 JSON"):
            st.json(stats)

    # 清除标签区域
    st.markdown("---")
    _render_clear_tag_section(ds)


def _render_box_source_selector(ds, key_prefix: str = "sam3") -> tuple:
    """渲染来源标签、来源字段和来源类别多选器。

    Returns:
        (box_source_field, box_source_labels, box_source_tags)
        - box_source_labels: None 表示不过滤（取全部类别）
        - box_source_tags: 选中的标签列表，空列表表示不按标签过滤
    """
    import fiftyone as fo

    # 来源标签（按 tags 筛选哪些样本的标注作为范例来源）
    available_tags = ds.distinct("tags")
    box_source_tags: list[str] = []
    if available_tags:
        box_source_tags = st.multiselect(
            "来源标签（筛选哪些样本的标注作为范例，不选则不过滤）",
            available_tags,
            key=f"{key_prefix}_box_tags",
            help="可多选。仅使用带有所选标签的样本中的标注作为视觉范例来源。留空则使用所有样本。",
        )

    schema = ds.get_field_schema()
    source_fields = []
    for fn, field in schema.items():
        if fn.startswith("_"):
            continue
        if not hasattr(field, "document_type") or not field.document_type:
            continue
        if issubclass(field.document_type, (fo.Polylines, fo.Detections)):
            source_fields.append(fn)

    box_source_field = None
    box_source_labels = None

    if source_fields:
        box_source_field = st.selectbox(
            "来源字段（包含已有 Polyline/Detection）", source_fields,
            key=f"{key_prefix}_box_field",
        )
        classes = dm.get_label_classes(ds, box_source_field) if box_source_field else []
        if classes:
            selected = st.multiselect(
                "来源类别（选择用作范例的类别，不选则使用全部类别）",
                classes,
                key=f"{key_prefix}_box_label",
                help="可多选。为空时取字段内所有类别的标注作为视觉范例。",
            )
            box_source_labels = selected if selected else None
        else:
            st.info("该字段中暂无标注类别")
    else:
        st.warning("当前数据集无可用的 Polyline/Detection 字段，请先添加标注或使用文本概念模式")

    return box_source_field, box_source_labels, box_source_tags


# -----------------------------------------------------------------------
# 共用组件
# -----------------------------------------------------------------------

def _show_predict_stats(stats: dict):
    """展示预标注结果统计。"""
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("检测到", stats.get("detected", 0))
    col_s2.metric("实际添加", stats.get("added", 0))
    col_s3.metric("NMS 抑制", stats.get("suppressed_nms", 0))

    extra_cols = st.columns(4)
    extra_cols[0].metric("处理图像", stats.get("images_processed", 0))
    extra_cols[1].metric("跳过已标注", stats.get("skipped_labeled", 0))
    extra_cols[2].metric("失败/报错", stats.get("errors", 0))
    extra_cols[3].metric("失败打标签", stats.get("fail_tagged", 0))

    if stats.get("filtered_class", 0) > 0:
        st.info(f"类别过滤: 跳过了 {stats['filtered_class']} 个不在过滤列表中的检测")
    if stats.get("skipped_small_mask", 0) > 0:
        st.warning(f"有 {stats['skipped_small_mask']} 个 mask 面积过小被跳过")

    with st.expander("详细统计 JSON"):
        st.json(stats)


def _render_clear_tag_section(ds):
    """清除失败标签的功能区。"""
    available_tags = ds.distinct("tags")
    if not available_tags:
        st.caption("当前数据集无任何标签")
        return

    with st.expander("🗑️ 清除标签"):
        tag_to_clear = st.selectbox(
            "选择要清除的标签", available_tags,
            key="clear_tag_select",
        )
        tagged_count = len(ds.match_tags(tag_to_clear))
        st.caption(f"有 **{tagged_count}** 个样本带有此标签")

        if tagged_count > 0 and st.button(
            f"清除标签 「{tag_to_clear}」", key="btn_clear_tag",
        ):
            with st.spinner(f"正在从 {tagged_count} 个样本中清除标签..."):
                cleared = processor.clear_tag_from_dataset(ds, tag_to_clear)
            st.success(f"已从 {cleared} 个样本中清除标签 「{tag_to_clear}」")
            st.rerun()
