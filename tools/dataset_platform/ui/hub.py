"""Data Hub：数据集统计仪表盘、标签筛选、标签管理。"""
from __future__ import annotations
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import cvat_sync
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _get_info


def _render_data_hub():
    st.header("📊 Data Hub - 数据集管理中心")
    ds = _get_ds()
    if ds is None:
        st.info("请先在侧边栏选择或创建数据集")
        return

    info = _get_info(ds)

    tab_stats, tab_tags_filter, tab_label_mgmt = st.tabs([
        "📊 数据集统计", "🔎 标签筛选与查看", "🏷️ 标签管理",
    ])

    with tab_stats:
        _render_hub_statistics(ds, info)
    with tab_tags_filter:
        _render_hub_tag_filter(ds, info)
    with tab_label_mgmt:
        _render_label_management(ds)


def _render_hub_statistics(ds, info: dict):
    """数据集统计仪表盘。"""
    import pandas as pd

    # --- 概览指标 ---
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("📷 样本总数", info["num_samples"])
    col_m2.metric("🏷️ 标签字段数", len(info["label_fields"]))
    col_m3.metric("🔖 Tags 种类", len(info["tags"]))
    total_labels = 0
    for lf in info["label_fields"]:
        stats = dm.get_label_stats(ds, lf)
        total_labels += sum(stats.values()) if stats else 0
    col_m4.metric("📝 标注实例总数", total_labels)

    # --- FiftyOne 查看入口 ---
    st.markdown("---")
    port = st.session_state.fo_port
    fo_col1, fo_col2 = st.columns([3, 1])
    with fo_col1:
        st.subheader("🔍 在 FiftyOne 中查看完整数据集")
        st.caption("点击下方按钮启动 FiftyOne App，将在新浏览器标签页中打开可视化界面")
    with fo_col2:
        if st.button("🚀 启动 / 刷新 FiftyOne App", key="btn_launch_fo"):
            try:
                session = dm.launch_app(ds, port=port)
                st.session_state.fo_session = session
                st.success(f"FiftyOne App 已启动 (端口 {port})")
            except Exception as e:
                st.error(f"启动失败: {e}")
        fo_url = f"http://localhost:{port}"
        st.link_button("🌐 打开 FiftyOne 查看", fo_url)

    # --- Label 统计 ---
    st.markdown("---")
    st.subheader("📊 标注类别统计")

    if info["label_fields"]:
        for lf in info["label_fields"]:
            field_type = dm.get_field_label_type(ds, lf)
            type_display = {"detections": "矩形框", "polylines": "多边形",
                            "keypoints": "关键点", "classifications": "分类"}.get(field_type, field_type or "未知")
            with st.expander(f"**`{lf}`** — {type_display}", expanded=len(info["label_fields"]) <= 3):
                stats = dm.get_label_stats(ds, lf)
                if stats:
                    df = pd.DataFrame(
                        list(stats.items()), columns=["类别", "数量"]
                    ).sort_values("数量", ascending=False)

                    chart_col, table_col = st.columns([2, 1])
                    with chart_col:
                        st.bar_chart(df.set_index("类别"))
                    with table_col:
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.caption(f"共 **{len(stats)}** 个类别，**{sum(stats.values())}** 个实例")
                else:
                    st.info("该字段暂无标注数据")
    else:
        st.info("当前数据集没有标签字段，请先导入标注数据")

    # --- Tags 统计 ---
    st.markdown("---")
    st.subheader("🔖 样本 Tags 统计")

    available_tags = info["tags"]
    if available_tags:
        raw_counts = ds.count_values("tags")
        raw_counts.pop(None, None)
        num_samples = info["num_samples"]

        has_anomaly = any(v > num_samples for v in raw_counts.values())
        if has_anomaly:
            tag_counts = {t: len(ds.match_tags(t)) for t in available_tags}
        else:
            tag_counts = raw_counts

        if tag_counts:
            df_tags = pd.DataFrame(
                list(tag_counts.items()), columns=["标签", "样本数"]
            ).sort_values("样本数", ascending=False)

            chart_col, table_col = st.columns([2, 1])
            with chart_col:
                st.bar_chart(df_tags.set_index("标签"))
            with table_col:
                st.dataframe(df_tags, use_container_width=True, hide_index=True)
                st.caption(
                    f"共 **{len(tag_counts)}** 种标签，"
                    f"一个样本可拥有多个标签"
                )
            if has_anomaly:
                st.warning(
                    "⚠️ 检测到数据库中存在**重复 Tags**（同一样本内相同标签出现多次），"
                    "请点击下方『修复重复 Tags』清理。上方统计已自动修正显示。"
                )
    else:
        st.info("当前数据集没有任何样本 Tags")

    # --- 修复重复 Tags ---
    st.markdown("---")
    with st.expander("🔧 修复重复 Tags"):
        st.caption("扫描所有样本，将 tags 列表中的重复项去重（如 `['a','a','b']` → `['a','b']`）。")
        if st.button("🔍 扫描并修复", key="btn_fix_dup_tags"):
            fixed = 0
            with st.spinner("扫描样本 Tags..."):
                for sample in ds.iter_samples(progress=True):
                    unique = list(dict.fromkeys(sample.tags))
                    if len(unique) != len(sample.tags):
                        sample.tags = unique
                        sample.save()
                        fixed += 1
            if fixed:
                st.success(f"已修复 {fixed} 个样本的重复 Tags")
                st.rerun()
            else:
                st.success("所有样本的 Tags 均无重复，无需修复")

    # --- 数据完整性检查 ---
    st.markdown("---")
    with st.expander("🔍 数据集完整性检查"):
        if st.button("运行检查", key="btn_integrity_check"):
            with st.spinner("检查文件完整性..."):
                import os as _os
                total = info["num_samples"]
                missing = []
                for fp in ds.values("filepath"):
                    if not _os.path.isfile(fp):
                        missing.append(fp)
                ok_count = total - len(missing)

            col_ok, col_miss = st.columns(2)
            col_ok.metric("文件存在", f"{ok_count}/{total}")
            col_miss.metric("文件缺失", len(missing))

            if missing:
                st.warning(
                    f"有 {len(missing)} 个样本指向不存在的文件。"
                    "这些样本的记录仍在 FiftyOne 中，但图片文件已丢失。"
                )
                with st.expander(f"查看缺失文件列表 ({len(missing)})"):
                    for fp in missing[:100]:
                        st.text(fp)
                    if len(missing) > 100:
                        st.caption(f"...及其他 {len(missing) - 100} 个")

                if st.checkbox("确认从数据集中移除这些缺失样本", key="confirm_remove_missing"):
                    if st.button("🗑️ 移除缺失样本", key="btn_remove_missing"):
                        view = ds.match(
                            __import__("fiftyone").ViewField("filepath").is_in(missing)
                        )
                        ds.delete_samples(view)
                        st.session_state["_toast_msg"] = f"已移除 {len(missing)} 个缺失样本"
                        st.rerun()
            else:
                st.success("所有样本的图片文件均存在，数据集完整")

    # --- 数据集元信息 ---
    with st.expander("📋 数据集完整元信息", expanded=False):
        st.json(info)


def _render_hub_tag_filter(ds, info: dict):
    """通用 Tags 筛选，并在 FiftyOne 或 CVAT 中查看。"""
    st.subheader("🔎 按 Tags 筛选样本")
    st.caption(
        "选择一个或多个 Tags 筛选样本，然后可以在 FiftyOne 中查看或直接在 CVAT 中打开对应任务。\n\n"
        "Tags 来源包括：导入时的批次标签、CVAT Job 状态标记、手动添加的标签等。"
    )

    available_tags = info["tags"]
    if not available_tags:
        st.info("当前数据集没有任何 Tags。可以在「标签管理」中添加，或在导入数据时指定批次标签。")
        return

    selected_tags = st.multiselect(
        "选择 Tags（多选为 OR 关系，匹配任一即保留）",
        available_tags,
        key="hub_filter_tags",
    )

    if selected_tags:
        filtered_view = ds.match_tags(selected_tags)
        st.info(f"筛选结果: **{len(filtered_view)}** 个样本（共 {len(ds)} 个）")

        if len(filtered_view) == 0:
            st.warning("没有匹配的样本")
            return

        col_fo, col_cvat = st.columns(2)
        with col_fo:
            st.markdown("**在 FiftyOne 中查看**")
            if st.button("👁️ 在 FiftyOne App 中展示筛选结果", key="hub_fo_view"):
                session = dm.get_session()
                if session:
                    dm.set_session_view(filtered_view)
                    st.success("✅ 已更新 FiftyOne App 视图为筛选结果")
                    port = st.session_state.fo_port
                    st.link_button("🌐 打开 FiftyOne 查看", f"http://localhost:{port}")
                else:
                    st.warning("请先在「数据集统计」Tab 中启动 FiftyOne App")

        with col_cvat:
            st.markdown("**在 CVAT 中查看**")
            st.caption("如果这些样本已推送到 CVAT，可以直接跳转到 CVAT 对应任务查看。")
            runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
            if runs:
                cvat_url = CONFIG.cvat.url.rstrip("/")
                st.link_button("🔗 打开 CVAT 面板", f"{cvat_url}/tasks")
            else:
                st.caption("当前数据集暂无 CVAT 标注运行，需先在「CVAT 同步」中推送数据。")

        # 筛选结果的标注统计
        with st.expander("📊 筛选结果统计", expanded=False):
            import pandas as pd
            for lf in info.get("label_fields", []):
                stats = dm.get_label_stats(filtered_view, lf)
                if stats:
                    st.markdown(f"**`{lf}`**")
                    df = pd.DataFrame(
                        list(stats.items()), columns=["类别", "数量"]
                    ).sort_values("数量", ascending=False)
                    st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("请在上方选择 Tags 进行筛选")

    # CVAT Job 状态快捷操作
    st.markdown("---")
    st.subheader("🔄 CVAT Job 状态标签刷新")
    st.caption(
        "如果数据已推送到 CVAT，可以一键从 CVAT 同步最新 Job 状态标签到样本 Tags 中，"
        "然后通过上方的 Tags 筛选来查看特定状态的样本。"
    )

    runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
    if runs:
        refresh_keys = st.multiselect(
            "选择标注运行", runs, default=runs, key="hub_refresh_keys",
            help="选择要刷新状态标签的标注运行",
        )
        if st.button("🔄 刷新 Job 状态标签", key="hub_refresh_tags"):
            if refresh_keys:
                with st.spinner("正在从 CVAT 查询最新 Job 状态并更新标签..."):
                    try:
                        tagged = cvat_sync.tag_samples_by_job_status(ds, refresh_keys)
                        if tagged:
                            st.success("✅ 标签已更新为最新状态")
                            for tag, count in tagged.items():
                                st.text(f"  {tag}: {count} 个样本")
                        else:
                            st.warning("无法标记：可能缺少 frame_id_map 映射信息")
                    except Exception as e:
                        st.error(f"刷新失败: {e}")
            else:
                st.warning("请先选择标注运行")
    else:
        st.info("暂无 CVAT 标注运行。推送数据到 CVAT 后可在此刷新 Job 状态标签。")


def _render_label_management(ds):
    # ---- 样本标签（Tags）管理 ----
    st.subheader("样本标签 (Tags) 管理")
    st.caption("给已有样本批量添加或移除 Tags（如批次标记、审核状态等）")

    info = _get_info(ds)
    available_tags = info["tags"]
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

    # ---- 标注类别操作 ----
    st.markdown("---")
    if not label_fields:
        st.info("当前数据集没有标签字段，请先导入标注数据")
        return

    op_tab_rename, op_tab_del_label, op_tab_del_field = st.tabs([
        "✏️ 类别重命名", "🗑️ 批量删除标注", "⚠️ 删除字段",
    ])

    # ── 类别重命名 ──
    with op_tab_rename:
        st.subheader("标注类别重命名")
        st.caption("将指定字段中的某个类别批量重命名为新名称")

        rename_field = st.selectbox("选择标签字段", label_fields, key="rename_field")
        current_classes = dm.get_label_classes(ds, rename_field)

        if current_classes:
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
                    st.warning("新名称与旧名称相同")
                else:
                    with st.spinner(f"正在将 `{old_label}` 重命名为 `{new_label.strip()}`..."):
                        try:
                            count = dm.rename_label(ds, rename_field, old_label, new_label.strip())
                            st.session_state["_toast_msg"] = f"已将 '{old_label}' 重命名为 '{new_label.strip()}'（共 {count} 个实例）"
                            st.rerun()
                        except Exception as e:
                            st.error(f"重命名失败: {e}")
        else:
            st.info(f"字段 `{rename_field}` 中暂无标签类别")

    # ── 批量删除标注 ──
    with op_tab_del_label:
        st.subheader("批量删除指定类别的标注")
        st.caption("从指定字段中删除选中类别的所有标注实例（不删除样本本身）")

        del_field = st.selectbox("选择标签字段", label_fields, key="del_label_field")
        del_classes = dm.get_label_classes(ds, del_field)

        if del_classes:
            import pandas as pd
            stats_del = dm.get_label_stats(ds, del_field)
            if stats_del:
                df_stats = pd.DataFrame(
                    list(stats_del.items()), columns=["类别", "数量"]
                ).sort_values("数量", ascending=False)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

            selected_classes = st.multiselect(
                "选择要删除的类别（可多选）", del_classes, key="del_label_classes",
            )

            if selected_classes:
                total_to_delete = sum(stats_del.get(c, 0) for c in selected_classes)
                st.warning(f"将从字段 `{del_field}` 中删除 **{len(selected_classes)}** 个类别共 **{total_to_delete}** 个标注实例")

                confirm_del = st.checkbox(
                    f"确认删除以上 {total_to_delete} 个标注实例（不可撤销）",
                    key="confirm_del_labels",
                )
                if st.button("🗑️ 执行批量删除", key="btn_del_labels", disabled=not confirm_del):
                    with st.spinner("正在删除标注..."):
                        try:
                            count = dm.delete_labels_by_class(ds, del_field, selected_classes)
                            st.session_state["_toast_msg"] = f"已从字段 '{del_field}' 中删除 {count} 个标注实例"
                            st.rerun()
                        except Exception as e:
                            st.error(f"删除失败: {e}")
        else:
            st.info(f"字段 `{del_field}` 中暂无标签类别")

    # ── 删除字段 ──
    with op_tab_del_field:
        st.subheader("删除整个标签字段")
        st.caption("从数据集中永久删除一个标签字段及其所有标注数据")

        schema = ds.get_field_schema()
        deletable_fields = []
        for fn, field in schema.items():
            if fn.startswith("_") or fn in ("id", "filepath", "tags", "metadata"):
                continue
            deletable_fields.append(fn)

        if deletable_fields:
            field_to_del = st.selectbox(
                "选择要删除的字段", deletable_fields, key="del_field_select",
            )

            field_type = dm.get_field_label_type(ds, field_to_del)
            if field_type:
                field_stats = dm.get_label_stats(ds, field_to_del)
                total_instances = sum(field_stats.values()) if field_stats else 0
                st.info(f"字段类型: **{field_type}**，包含 **{len(field_stats)}** 个类别共 **{total_instances}** 个标注实例")
            else:
                st.info(f"字段 `{field_to_del}` 是非标签类型字段")

            st.error("此操作不可撤销！删除后该字段的所有数据将永久丢失。")
            confirm_text = st.text_input(
                f"输入字段名 `{field_to_del}` 以确认删除",
                key="confirm_del_field_text",
            )
            if st.button(
                f"⚠️ 永久删除字段 {field_to_del}",
                key="btn_del_field",
                disabled=(confirm_text != field_to_del),
            ):
                with st.spinner(f"正在删除字段 `{field_to_del}`..."):
                    try:
                        dm.delete_sample_field(ds, field_to_del)
                        st.session_state["_toast_msg"] = f"已删除字段 '{field_to_del}'"
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
        else:
            st.info("当前数据集没有可删除的字段")

    # ── 标注统计 ──
    st.markdown("---")
    st.subheader("标注统计")
    stat_field = st.selectbox("查看字段", label_fields, key="stat_field_view")
    stats = dm.get_label_stats(ds, stat_field)
    if stats:
        import pandas as pd
        df = pd.DataFrame(
            list(stats.items()), columns=["类别", "数量"]
        ).sort_values("数量", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

