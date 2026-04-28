"""CVAT 双向同步页面。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import cvat_sync
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _get_info


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

    tab_push, tab_pull, tab_manage, tab_review = st.tabs([
        "📤 推送到 CVAT", "📥 从 CVAT 拉取", "📋 管理标注运行", "🔍 任务状态与审核",
    ])

    with tab_push:
        _render_cvat_push(ds)
    with tab_pull:
        _render_cvat_pull(ds)
    with tab_manage:
        _render_cvat_manage(ds)
    with tab_review:
        _render_cvat_review(ds)


def _render_cvat_push(ds):
    st.subheader("推送数据到 CVAT")

    _TYPE_LABELS = {
        "detections": "矩形框 (detections)",
        "polylines": "多边形 (polylines)",
        "keypoints": "关键点 (keypoints)",
        "classifications": "分类 (classifications)",
    }

    existing_runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
    if existing_runs:
        st.info(f"当前数据集已有标注运行: **{', '.join(existing_runs)}**（可在「管理标注运行」中查看）")

    anno_key = st.text_input(
        "标注键 (anno_key)", key="push_anno_key",
        placeholder="例如: round1_all",
        help="标注任务的唯一 ID，拉取标注时需要用同一个 key 匹配。每次推送必须使用不同的 key。",
    )

    custom_project = st.text_input(
        "CVAT 项目名（可选）", key="push_project_name",
        placeholder=f"留空则使用数据集名称: {ds.name}",
        help="CVAT 中的 Project 名称。",
    )

    info = _get_info(ds)
    label_fields = info.get("label_fields", [])

    st.markdown("**标签字段**")
    field_mode = st.radio(
        "字段来源",
        ["选择已有字段（推送已有标注作为预标注）", "输入新字段名（在 CVAT 中从零开始标注）"],
        key="push_field_mode", horizontal=True, label_visibility="collapsed",
    )

    label_schema = None
    single_label_field = None
    single_label_type = None
    single_push_attrs = None
    polyline_fields = []

    if field_mode.startswith("选择已有"):
        if label_fields:
            selected_fields = st.multiselect(
                "选择标签字段（可多选）", label_fields, default=label_fields,
                key="push_fields_sel",
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
                        "为多边形字段添加四角遮挡属性 (1_occu ~ 4_occu)", value=False, key="push_occu_attrs",
                    )

                field_classes_filter = {}
                with st.expander("🏷️ 按字段筛选类别（默认推送全部类别）", expanded=False):
                    for f in selected_fields:
                        all_cls = field_type_map[f]["classes"]
                        if all_cls:
                            sel = st.multiselect(f"`{f}` — 选择类别（留空=全部 {len(all_cls)} 个）", all_cls, key=f"push_cls_{f}")
                            if sel:
                                field_classes_filter[f] = sel

                label_schema = {}
                for f in selected_fields:
                    all_cls = field_type_map[f]["classes"]
                    entry = {}
                    if f in field_classes_filter:
                        entry["classes"] = field_classes_filter[f]
                    elif all_cls:
                        entry["classes"] = all_cls
                    if f in polyline_fields and include_occu_attrs:
                        entry["attributes"] = cvat_sync.OCCLUSION_ATTRS
                    label_schema[f] = entry
            else:
                st.warning("请至少选择一个字段")
        else:
            st.warning("当前数据集没有标签字段，请切换到「输入新字段名」模式")
    else:
        single_label_field = st.text_input("新字段名", value="ground_truth", key="push_field_new")
        single_label_type = st.selectbox(
            "标注类型（新字段必选）",
            ["detections", "polylines", "keypoints", "classifications"], key="push_ltype",
        )
        if single_label_type == "polylines":
            if st.checkbox("添加四角遮挡属性 (1_occu ~ 4_occu)", value=False, key="push_occu_attrs_new"):
                single_push_attrs = cvat_sync.OCCLUSION_ATTRS

    st.markdown("---")

    push_mode = st.radio(
        "推送范围", ["整个数据集", "仅无标注样本", "仅有标注样本"], key="push_mode", horizontal=True,
    )

    available_tags = ds.distinct("tags")
    push_filter_tags = []
    if available_tags:
        push_filter_tags = st.multiselect("按批次标签筛选（留空 = 不过滤）", available_tags, key="push_filter_tags")

    selected_classes = []
    if label_schema is None and label_fields:
        filter_field = single_label_field if (single_label_field and single_label_field in label_fields) else None
        if filter_field:
            all_classes = dm.get_label_classes(ds, filter_field)
            if all_classes:
                selected_classes = st.multiselect("仅推送以下类别（留空 = 推送全部类别）", all_classes, key="push_classes")

    col_seg, col_quality = st.columns(2)
    with col_seg:
        segment_size = st.number_input("每个 Job 图片数", value=200, min_value=10, key="push_seg")
    with col_quality:
        image_quality = st.number_input("图像质量", value=100, min_value=1, max_value=100, key="push_img_quality")

    include_review = st.checkbox("启用废弃标记（允许标注员在 CVAT 中标记需废弃的图片）", value=False, key="push_include_review")
    if include_review:
        with st.expander("📖 标注员如何在 CVAT 中标记废弃图片", expanded=True):
            st.markdown("""
**推送后，标注员在 CVAT 中的操作步骤：**

1. 打开对应的标注任务，进入标注界面
2. 在顶部工具栏找到标注模式切换区（左上角），点击 **「Tag」** 图标
3. 在左侧标签列表中选择 **`discard`**
4. 浏览图片，遇到需要废弃的图片时直接点击画面即可为该帧添加 `discard` 标签
5. 切回 **「Shape」** 模式继续正常标注其他图片
            """)

    is_multi_field = label_schema is not None and len(label_schema) > 1
    if is_multi_field:
        st.info(f"已选 **{len(label_schema)}** 个字段，将逐字段推送到独立 CVAT 任务。")

    can_push = label_schema is not None or single_label_field
    if st.button("📤 推送到 CVAT", key="btn_push", disabled=not can_push):
        if not anno_key:
            st.error("请输入标注键 (anno_key)")
            return
        if not anno_key.replace("_", "").replace("-", "").isalnum():
            st.error("标注键只能包含字母、数字、下划线和短横线")
            return

        if is_multi_field:
            derived_keys = {f: f"{anno_key}_{f}" for f in label_schema}
            conflicts = [k for k in derived_keys.values() if k in existing_runs]
            if conflicts:
                st.error(f"以下标注键已存在: **{', '.join(conflicts)}**")
                return
        else:
            if anno_key in existing_runs:
                st.error(f"标注键 `{anno_key}` 已存在！")
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

        review_schema = (
            {cvat_sync.REVIEW_FIELD: {"type": "classifications", "classes": cvat_sync.REVIEW_CLASSES}}
            if include_review else {}
        )
        push_project_name = custom_project.strip() or None

        if is_multi_field:
            all_results = []
            review_key = None
            for idx, (field_name, field_config) in enumerate(label_schema.items()):
                field_key = f"{anno_key}_{field_name}"
                push_schema = {field_name: field_config}
                if idx == 0 and review_schema:
                    push_schema.update(review_schema)
                    review_key = field_key
                with st.spinner(f"正在推送字段 `{field_name}` ({len(samples)} 样本)..."):
                    try:
                        r = cvat_sync.push_to_cvat(
                            samples, field_key, label_schema=push_schema,
                            project_name=push_project_name, segment_size=segment_size, image_quality=image_quality,
                        )
                        all_results.append(r)
                        st.success(f"✅ 字段 `{field_name}` 推送成功 (anno_key: `{field_key}`)")
                    except Exception as field_e:
                        st.error(f"❌ 字段 `{field_name}` 推送失败: {field_e}")
            if all_results:
                st.json(all_results)
        else:
            with st.spinner(f"正在推送 {len(samples)} 个样本到 CVAT..."):
                try:
                    if label_schema or review_schema:
                        push_schema = dict(label_schema) if label_schema else {}
                        if not push_schema:
                            entry: dict = {}
                            if single_label_type:
                                entry["type"] = single_label_type
                            if selected_classes:
                                entry["classes"] = selected_classes
                            if single_push_attrs:
                                entry["attributes"] = single_push_attrs
                            push_schema[single_label_field] = entry
                        if review_schema:
                            push_schema.update(review_schema)
                        result = cvat_sync.push_to_cvat(
                            samples, anno_key, label_schema=push_schema,
                            project_name=push_project_name, segment_size=segment_size, image_quality=image_quality,
                        )
                    else:
                        result = cvat_sync.push_to_cvat(
                            samples, anno_key, label_field=single_label_field, label_type=single_label_type,
                            project_name=push_project_name, segment_size=segment_size, image_quality=image_quality,
                            classes=selected_classes or None, attributes=single_push_attrs,
                        )
                    st.success("✅ 推送成功")
                    st.json(result)
                except Exception as e:
                    st.error(f"推送失败: {e}")


def _render_cvat_pull(ds):
    st.subheader("从 CVAT 拉取标注")

    runs = cvat_sync.list_annotation_runs_keys(ds) if hasattr(ds, "list_annotation_runs") else []
    if runs:
        st.success(f"检测到 **{len(runs)}** 个标注运行")
        anno_key = st.selectbox("选择标注运行（最新在前）", runs, key="pull_anno_key")
        try:
            run_info = ds.get_annotation_info(anno_key)
            with st.expander("运行详情", expanded=False):
                st.caption(f"创建时间: {run_info.timestamp if hasattr(run_info, 'timestamp') else 'N/A'}")
                st.caption(f"配置: {run_info.config if hasattr(run_info, 'config') else 'N/A'}")
        except Exception:
            pass

        anno_types = cvat_sync._get_anno_type(ds, anno_key)
        if anno_types:
            _TYPE_LABELS = {"detections": "矩形框", "polylines": "多边形", "keypoints": "关键点", "classifications": "分类"}
            type_display = ", ".join(_TYPE_LABELS.get(t, t) for t in anno_types)
            prefix = cvat_sync._tag_prefix_for_types(anno_types)
            st.info(f"📋 任务标注类型: **{type_display}** — 拉取后完成的 Job 将标记为 `{prefix}_completed`")
    else:
        st.warning("当前数据集没有标注运行记录。")
        anno_key = ""

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        cleanup = st.checkbox("拉取后删除 CVAT 端任务", key="pull_cleanup")
        if cleanup:
            st.warning("⚠️ 拉取完成后将**永久删除** CVAT 服务器上的对应任务和项目，此操作不可逆！")
            confirm_cleanup = st.checkbox("我确认要在拉取后删除 CVAT 端任务", key="pull_cleanup_confirm")
        else:
            confirm_cleanup = True
    with col_opt2:
        skip_tagging = st.checkbox(
            "跳过自动 Job 状态标签", key="pull_skip_tagging",
            help="大数据集打标签可能耗时较长，勾选后仅拉取标注不打标签。可稍后在 DataHub 手动刷新。",
        )

    pull_disabled = not anno_key or (cleanup and not confirm_cleanup)
    if st.button("📥 拉取标注", key="btn_pull", disabled=pull_disabled):
        if not anno_key:
            st.error("请选择标注运行")
            return
        spinner_msg = "拉取中..." if skip_tagging else "拉取中（完成后会自动标记 Job 状态标签）..."
        with st.spinner(spinner_msg):
            try:
                result = cvat_sync.pull_from_cvat(
                    ds, anno_key, cleanup=cleanup, skip_tagging=skip_tagging,
                )
                st.success("✅ 拉取成功，标注已同步到数据集")
                job_tags = result.get("job_tags", {})
                if job_tags:
                    st.markdown("**自动标记的 Job 状态标签：**")
                    for tag, count in job_tags.items():
                        st.text(f"  {tag}: {count} 个样本")
                elif skip_tagging:
                    st.info("已跳过自动标签，可前往 DataHub 手动刷新 Job 状态标签")
                st.json(result)
            except Exception as e:
                st.error(f"拉取失败: {e}")


def _render_cvat_manage(ds):
    st.subheader("标注运行管理")
    runs = cvat_sync.list_annotation_runs(ds)
    if runs:
        import pandas as pd
        df = pd.DataFrame(runs)
        st.dataframe(df, use_container_width=True)

        del_key = st.selectbox("选择要删除的运行（最新在前）", [r["anno_key"] for r in runs], key="del_run_key")
        cleanup_cvat = st.checkbox("同时清理 CVAT 端", key="del_cleanup_cvat")
        st.warning(f"⚠️ 即将删除标注运行: **{del_key}**"
                   + ("（同时删除 CVAT 端任务）" if cleanup_cvat else "（仅删除本地记录）"))
        confirm_del = st.text_input(f"请输入标注键 `{del_key}` 确认删除", key="del_run_confirm")
        if st.button("🗑️ 确认删除标注运行", key="btn_del_run", type="primary"):
            if confirm_del != del_key:
                st.error(f"输入不匹配")
            else:
                try:
                    cvat_sync.delete_annotation_run(ds, del_key, cleanup=cleanup_cvat)
                    st.session_state["_toast_msg"] = f"已删除标注运行: {del_key}"
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
    else:
        st.info("暂无标注运行记录。")


def _render_cvat_review(ds):
    st.subheader("任务状态与审核")
    runs = cvat_sync.list_annotation_runs_keys(ds) if hasattr(ds, "list_annotation_runs") else []
    if not runs:
        st.info("暂无标注运行记录。请先推送数据到 CVAT。")
        return

    # --- CVAT 标注进度看板 ---
    st.markdown("### 📊 标注进度看板")

    selected_keys = st.multiselect(
        "选择标注运行（可多选）", runs, default=runs, key="review_anno_keys",
    )

    if st.button("🔍 查询 Job 状态", key="btn_query_jobs"):
        if not selected_keys:
            st.warning("请至少选择一个标注运行")
        else:
            with st.spinner("正在查询 CVAT Job 状态..."):
                try:
                    all_jobs: list[dict] = []
                    for key in selected_keys:
                        all_jobs.extend(cvat_sync.get_job_details(ds, key))
                    if all_jobs:
                        import pandas as pd
                        df = pd.DataFrame([{
                            "anno_key": j.get("_anno_key", ""),
                            "Job ID": j["job_id"],
                            "Task ID": j["task_id"],
                            "状态 (state)": j["state"],
                            "阶段 (stage)": j["stage"],
                            "标注员": j["assignee"],
                            "帧范围": f"{j['start_frame']}–{j['stop_frame']}",
                            "样本数": j["num_samples"],
                        } for j in all_jobs])
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        state_counts: dict[str, int] = {}
                        total_jobs = len(all_jobs)
                        for j in all_jobs:
                            state_counts[j["state"]] = state_counts.get(j["state"], 0) + 1

                        completed = state_counts.get("completed", 0)
                        st.progress(completed / total_jobs if total_jobs else 0,
                                    text=f"完成进度: {completed}/{total_jobs} Jobs")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**按 State 统计：**")
                            for s, c in state_counts.items():
                                st.text(f"  {s}: {c} 个 Job")
                        with col2:
                            stage_counts: dict[str, int] = {}
                            for j in all_jobs:
                                stage_counts[j["stage"]] = stage_counts.get(j["stage"], 0) + 1
                            st.markdown("**按 Stage 统计：**")
                            for s, c in stage_counts.items():
                                st.text(f"  {s}: {c} 个 Job")

                        # CVAT Task 深链
                        cvat_url = CONFIG.cvat.url.rstrip("/")
                        task_ids = sorted(set(j["task_id"] for j in all_jobs))
                        st.markdown("### 🔗 CVAT Task 快速链接")
                        for tid in task_ids:
                            st.link_button(f"Task {tid}", f"{cvat_url}/tasks/{tid}")

                        st.session_state["_review_jobs"] = all_jobs
                    else:
                        st.warning("未找到任何 Job")
                except Exception as e:
                    st.error(f"查询失败: {e}")

    st.markdown("---")

    # --- 废弃图片管理 ---
    st.markdown("### 🗑️ 废弃图片管理")
    review_field = cvat_sync.REVIEW_FIELD
    schema = ds.get_field_schema()

    if review_field not in schema:
        st.info(f"当前数据集没有 `{review_field}` 字段。在「推送到 CVAT」时勾选「启用废弃标记」即可使用。")
    else:
        try:
            discarded = cvat_sync.find_discarded_samples(ds, review_field)
            discard_count = len(discarded)
        except Exception as e:
            st.error(f"检测废弃样本失败: {e}")
            return

        if discard_count == 0:
            st.success("✅ 没有被标记为废弃的样本")
        else:
            st.warning(f"发现 **{discard_count}** 个被标记为废弃的样本")

            if st.button("👁️ 在 FiftyOne App 中查看废弃样本", key="btn_view_discarded"):
                session = dm.get_session()
                if session:
                    dm.set_session_view(discarded)
                    st.success("✅ 已在 FiftyOne App 中展示废弃样本")

            discard_action = st.radio(
                "处理方式",
                ["仅打标签", "从数据集移除（保留文件）", "从数据集移除并删除文件"],
                key="discard_action", horizontal=True,
            )
            action_map = {"仅打标签": "tag", "从数据集移除（保留文件）": "remove", "从数据集移除并删除文件": "remove_and_delete"}
            tag_name = "discarded"
            if discard_action == "仅打标签":
                tag_name = st.text_input("标签名", value="discarded", key="discard_tag_name")

            action = action_map[discard_action]
            can_proceed = True
            if action in ("remove", "remove_and_delete"):
                st.error(f"⚠️ 即将从数据集移除 **{discard_count}** 个样本，此操作不可逆！")
                confirm = st.text_input(f"请输入数字 `{discard_count}` 确认操作", key="discard_confirm")
                can_proceed = confirm == str(discard_count)

            if st.button("✅ 执行处理", key="btn_handle_discard", disabled=not can_proceed):
                with st.spinner("处理中..."):
                    try:
                        result = cvat_sync.handle_discarded_samples(ds, review_field, action=action, tag_name=tag_name)
                        st.success(f"✅ 处理完成：{result['action']} 了 {result['count']} 个样本")
                        if action != "tag":
                            st.session_state["_toast_msg"] = f"已处理 {result['count']} 个废弃样本"
                            st.rerun()
                    except Exception as e:
                        st.error(f"处理失败: {e}")
