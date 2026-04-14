"""数据处理与清洗页面。"""
from __future__ import annotations
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import processor
from tools.dataset_platform.ui.components import _get_ds


# ===================================================================
# Tab 3: 数据处理与清洗
# ===================================================================

def _render_processing():
    st.header("🧹 数据处理与清洗")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_clean, tab_merge = st.tabs(
        ["🧹 图像清理", "🔀 字段合并"],
    )

    with tab_clean:
        _render_cleaning(ds)
    with tab_merge:
        _render_field_merge(ds)


def _render_bad_polylines(ds):
    st.subheader("多边形处理")

    info = dm.get_dataset_info(ds)
    poly_fields = [f for f in info.get("label_fields", []) if dm.get_field_label_type(ds, f) == "polylines"]

    if not poly_fields:
        st.info("当前数据集没有多边形 (Polylines) 类型的标签字段")
        return

    bp_field = st.selectbox("多边形字段", poly_fields, key="bp_field")

    bp_scope = st.radio("操作范围", ["整个数据集", "按标签筛选"], key="bp_scope", horizontal=True)
    bp_view = ds
    if bp_scope == "按标签筛选":
        avail = ds.distinct("tags")
        if avail:
            bp_tags = st.multiselect("选择标签", avail, key="bp_scope_tags")
            if bp_tags:
                bp_view = ds.match_tags(bp_tags)
        else:
            st.info("当前数据集没有任何标签")
    st.caption(f"操作范围: **{len(bp_view)}** 个样本")

    tab_convert, tab_boundary = st.tabs(["🔄 多边形转四角", "⚠️ 边界多边形检测"])

    # ---- Tab 1: 多边形转四角 ----
    with tab_convert:
        st.caption(
            "将 N 点多边形转换为 4 点四边形：从所有顶点中提取左上 (tl)、右上 (tr)、"
            "右下 (br)、左下 (bl) 四个极角点。已经是 4 点的多边形仅重新排序为 tl→tr→br→bl。"
        )
        if st.button("🔄 执行转换", key="btn_convert_quads"):
            with st.spinner(f"转换 {len(bp_view)} 个样本中..."):
                stats = processor.convert_polylines_to_quads(bp_view, label_field=bp_field)
            st.success("转换完成")
            st.json(stats)

    # ---- Tab 2: 边界多边形检测 ----
    with tab_boundary:
        st.caption(
            "检测「单多边形 + 角点贴近图像边界」的样本：如果一张图片仅有 1 个四角多边形，"
            "且任一顶点的归一化坐标贴近图像边缘（≈ 被截断），则标记为 `bad_polygon`。"
        )
        edge_th = st.slider(
            "边界阈值（归一化）", min_value=0.001, max_value=0.05, value=0.005, step=0.001,
            key="bp_edge_threshold",
            help="顶点坐标距边界小于此值即判定为贴近边界。0.005 ≈ 1000px 图像的 5px",
        )

        bad_tag = "bad_polygon"
        bad_count = len(ds.match_tags([bad_tag]))
        if bad_count > 0:
            st.warning(f"当前有 **{bad_count}** 个样本标记为 `{bad_tag}`")
            bp_act1, bp_act2 = st.columns(2)
            with bp_act1:
                bp_del_physical = st.checkbox("同时删除磁盘文件", key="bp_del_physical")
                confirm_bp_del = st.checkbox(
                    f"⚠️ 确认删除 {bad_count} 个异常样本", key="confirm_del_bad_poly",
                )
                if st.button("🗑️ 删除已标记的异常样本", key="btn_del_bad_poly",
                             type="primary", disabled=not confirm_bp_del):
                    with st.spinner("删除中..."):
                        bad_view = ds.match_tags([bad_tag])
                        ids = bad_view.values("id")
                        if bp_del_physical:
                            dm.delete_samples_physically(ds, ids)
                        else:
                            ds.delete_samples(ids)
                    st.session_state["_toast_msg"] = f"已删除 {len(ids)} 个边界多边形样本"
                    st.rerun()
            with bp_act2:
                if st.button("🧹 清除 bad_polygon 标记", key="btn_clear_bad_poly",
                             help="仅移除标签，不删除样本"):
                    with st.spinner("清除中..."):
                        cleared = processor.clear_tag(ds, bad_tag)
                    st.session_state["_toast_msg"] = f"已清除 {cleared} 个样本的 {bad_tag} 标记"
                    st.rerun()

        if st.button("🔍 扫描边界多边形", key="btn_scan_bad_poly"):
            with st.spinner(f"扫描 {len(bp_view)} 个样本中..."):
                bad_ids = processor.find_boundary_polylines(
                    bp_view, label_field=bp_field, edge_threshold=edge_th, tag=bad_tag,
                )
            if bad_ids:
                st.error(f"发现 **{len(bad_ids)}** 个样本的多边形角点贴近图像边界")
            else:
                st.success("未发现边界多边形")


def _render_cleaning(ds):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("无标注图像")
        unlabeled = processor.find_unlabeled_samples(ds)
        unlabeled_count = len(unlabeled)
        st.metric("无标注样本数", unlabeled_count)
        physical = st.checkbox("同时删除磁盘文件", key="clean_unlabeled_physical")
        if unlabeled_count > 0:
            confirm_del_unlabeled = st.checkbox(
                f"⚠️ 确认删除 {unlabeled_count} 个无标注样本", key="confirm_del_unlabeled",
            )
        else:
            confirm_del_unlabeled = False
        if st.button("🗑️ 删除无标注样本", key="btn_del_unlabeled",
                     disabled=not confirm_del_unlabeled):
            with st.spinner("删除中..."):
                count = processor.delete_unlabeled_samples(ds, physical=physical)
            st.success(f"✅ 删除 {count} 个样本")
            st.rerun()

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
            corrupt_count = len(st.session_state["corrupt_ids"])
            confirm_del_corrupt = st.checkbox(
                f"⚠️ 确认删除 {corrupt_count} 个异常样本（含磁盘文件）",
                key="confirm_del_corrupt",
            )
            if st.button("🗑️ 删除异常样本（含磁盘文件）", key="btn_del_corrupt",
                         disabled=not confirm_del_corrupt):
                from tools.dataset_platform.data_manager import delete_samples_physically
                count = delete_samples_physically(ds, st.session_state["corrupt_ids"])
                st.success(f"✅ 物理删除 {count} 个样本")
                del st.session_state["corrupt_ids"]
                st.rerun()

    st.markdown("---")
    _render_bad_polylines(ds)

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
                tag_match_count = len(tag_view)
                st.metric("匹配样本数", tag_match_count)
            else:
                tag_match_count = 0
            del_physical = st.checkbox("同时删除磁盘文件", key="clean_tag_physical")
            confirm_del_by_tag = st.checkbox(
                f"⚠️ 确认删除匹配的 {tag_match_count} 个样本", key="confirm_del_by_tag",
            ) if del_tags and tag_match_count > 0 else False
            if st.button("🗑️ 删除匹配样本", key="btn_del_by_tag",
                         disabled=not confirm_del_by_tag):
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
            dup_col1, dup_col2 = st.columns(2)
            with dup_col1:
                dup_del_physical = st.checkbox("同时删除磁盘文件", key="dup_del_existing_physical")
                confirm_del_dup = st.checkbox(
                    f"⚠️ 确认删除 {dup_count} 个重复样本", key="confirm_del_dup_tagged",
                )
                if st.button("🗑️ 删除已标记的重复样本", key="btn_del_dup_tagged",
                             type="primary", disabled=not confirm_del_dup):
                    with st.spinner("删除中..."):
                        dup_view = ds.match_tags(["duplicate"])
                        ids = dup_view.values("id")
                        if dup_del_physical:
                            dm.delete_samples_physically(ds, ids)
                        else:
                            ds.delete_samples(ids)
                    st.session_state["_toast_msg"] = f"已删除 {len(ids)} 个重复样本"
                    if "dup_groups" in st.session_state:
                        del st.session_state["dup_groups"]
                    st.rerun()
            with dup_col2:
                if st.button("🧹 清除 duplicate 标记", key="btn_clear_dup_tag",
                             help="仅移除标签，不删除样本。用于调整参数重新扫描"):
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
