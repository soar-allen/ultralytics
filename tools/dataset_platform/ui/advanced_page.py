"""高级功能页面：难例挖掘、FiftyOne Brain、数据集备份。"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import advanced
from tools.dataset_platform import cvat_sync
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _path_browser


def _render_advanced():
    st.header("🧠 高级功能")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_hard, tab_brain, tab_backup = st.tabs([
        "🎯 难例挖掘", "🧬 FiftyOne Brain", "💾 数据集备份",
    ])

    with tab_hard:
        _render_hard_mining(ds)
    with tab_brain:
        _render_brain(ds)
    with tab_backup:
        _render_backup(ds)


def _render_hard_mining(ds):
    st.subheader("难例挖掘")
    st.caption("对比 ground_truth 和 predictions 找出高错误率图像，可一键推送到 CVAT 重标注")

    pred_field = st.text_input("预测字段", value="predictions", key="hard_pred")
    gt_field = st.text_input("真值字段", value="ground_truth", key="hard_gt")
    iou_thresh = st.slider("IoU 阈值", 0.1, 0.95, 0.5, 0.05, key="hard_iou")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 评估模型表现", key="btn_eval"):
            with st.spinner("评估中..."):
                try:
                    advanced.evaluate_detections(
                        ds, pred_field=pred_field, gt_field=gt_field, iou_threshold=iou_thresh,
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
                        ds, pred_field=pred_field, gt_field=gt_field, limit=limit,
                    )
                    st.info(f"找到 {len(hard_view)} 个难例样本")
                    st.session_state["_hard_view"] = hard_view
                    session = dm.get_session()
                    if session:
                        dm.set_session_view(hard_view)
                        st.success("✅ 已在 FiftyOne App 中展示难例")
                except Exception as e:
                    st.error(f"挖掘失败: {e}")

    # 难例推送 CVAT 重标注
    st.markdown("---")
    st.markdown("### 🔄 难例推送 CVAT 重标注")
    hard_view = st.session_state.get("_hard_view")
    if hard_view is not None and len(hard_view) > 0:
        st.info(f"当前难例视图: **{len(hard_view)}** 个样本")

        tag_hard = st.checkbox("为难例打上 `hard_sample` 标签", value=True, key="hard_tag_check")
        hard_anno_key = st.text_input("CVAT 标注键", value="hard_relabel", key="hard_anno_key",
                                      help="推送难例到 CVAT 的 anno_key")
        hard_label_field = st.text_input("标注字段", value=gt_field, key="hard_label_field")

        if st.button("📤 推送难例到 CVAT", key="btn_push_hard", type="primary"):
            if not hard_anno_key:
                st.error("请输入标注键")
                return
            try:
                if tag_hard:
                    for sample in hard_view.iter_samples(autosave=True):
                        if "hard_sample" not in sample.tags:
                            sample.tags.append("hard_sample")
                    st.info(f"已为 {len(hard_view)} 个难例添加 `hard_sample` 标签")

                with st.spinner("推送难例到 CVAT..."):
                    result = cvat_sync.push_to_cvat(
                        hard_view, hard_anno_key, label_field=hard_label_field,
                    )
                    st.success("✅ 难例已推送到 CVAT，等待重标注")
                    st.json(result)
            except Exception as e:
                st.error(f"推送失败: {e}")
    else:
        st.caption("请先执行上方的「挖掘难例」生成难例视图")


def _render_brain(ds):
    """FiftyOne Brain 分析：uniqueness、hardness、representativeness。"""
    st.subheader("FiftyOne Brain 智能分析")
    st.caption("利用 FiftyOne Brain 对数据集进行高级分析")

    tab_uniq, tab_hardness, tab_repr = st.tabs([
        "🔍 Uniqueness (唯一性)", "🎯 Hardness (难度)", "📊 Representativeness (代表性)",
    ])

    with tab_uniq:
        st.markdown("""
**Uniqueness** 衡量每个样本在数据集中的独特程度。
- 低唯一性 = 与其他样本高度相似 → 可考虑去重
- 高唯一性 = 独特样本 → 更有价值
        """)
        if st.button("🔍 计算 Uniqueness", key="btn_brain_uniq"):
            with st.spinner("计算中（需要提取嵌入向量，首次较慢）..."):
                try:
                    import fiftyone.brain as fob
                    fob.compute_uniqueness(ds)
                    ds.save()
                    st.success("✅ 计算完成，结果已写入 `uniqueness` 字段")
                except Exception as e:
                    st.error(f"计算失败: {e}")

        if "uniqueness" in ds.get_field_schema():
            from fiftyone import ViewField as F
            col1, col2 = st.columns(2)
            with col1:
                n_dup = st.number_input("显示最相似（最不唯一）的 N 个", value=20, key="uniq_n_dup")
                if st.button("查看最不唯一样本", key="btn_uniq_low"):
                    view = ds.sort_by("uniqueness")[:n_dup]
                    session = dm.get_session()
                    if session:
                        dm.set_session_view(view)
                        st.success(f"✅ 已展示 uniqueness 最低的 {n_dup} 个样本")
            with col2:
                n_unique = st.number_input("显示最独特的 N 个", value=20, key="uniq_n_unique")
                if st.button("查看最独特样本", key="btn_uniq_high"):
                    view = ds.sort_by("uniqueness", reverse=True)[:n_unique]
                    session = dm.get_session()
                    if session:
                        dm.set_session_view(view)
                        st.success(f"✅ 已展示 uniqueness 最高的 {n_unique} 个样本")

            import fiftyone as fo_agg
            try:
                mean_val = ds.mean("uniqueness")
                bounds = ds.bounds("uniqueness")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("平均值", f"{mean_val:.3f}" if mean_val is not None else "N/A")
                mc2.metric("最小值", f"{bounds[0]:.3f}" if bounds and bounds[0] is not None else "N/A")
                mc3.metric("最大值", f"{bounds[1]:.3f}" if bounds and bounds[1] is not None else "N/A")
            except Exception:
                pass

    with tab_hardness:
        st.markdown("""
**Hardness** 基于模型预测的不确定性衡量每个样本的标注难度。
- 高难度样本 = 模型最不确定 → 优先标注（主动学习核心）
        """)
        pred_field_h = st.text_input("预测字段", value="predictions", key="brain_hard_pred")
        if st.button("🎯 计算 Hardness", key="btn_brain_hard"):
            with st.spinner("计算中..."):
                try:
                    import fiftyone.brain as fob
                    fob.compute_hardness(ds, pred_field_h)
                    ds.save()
                    st.success("✅ 计算完成，结果已写入 `hardness` 字段")
                except Exception as e:
                    st.error(f"计算失败: {e}")

        if "hardness" in ds.get_field_schema():
            n_hard = st.number_input("显示最难的 N 个", value=30, key="hard_brain_n")
            if st.button("查看最难样本", key="btn_hard_brain_view"):
                view = ds.sort_by("hardness", reverse=True)[:n_hard]
                session = dm.get_session()
                if session:
                    dm.set_session_view(view)
                    st.success(f"✅ 已展示 hardness 最高的 {n_hard} 个样本")

    with tab_repr:
        st.markdown("""
**Representativeness** 衡量每个样本对数据集整体分布的代表程度。
- 高代表性 = 分布核心样本
- 低代表性 = 分布边缘/异常样本
        """)
        if st.button("📊 计算 Representativeness", key="btn_brain_repr"):
            with st.spinner("计算中..."):
                try:
                    import fiftyone.brain as fob
                    fob.compute_representativeness(ds)
                    ds.save()
                    st.success("✅ 计算完成，结果已写入 `representativeness` 字段")
                except Exception as e:
                    st.error(f"计算失败: {e}")

        if "representativeness" in ds.get_field_schema():
            n_repr = st.number_input("显示代表性最低的 N 个", value=20, key="repr_n")
            if st.button("查看最不具代表性的样本", key="btn_repr_low"):
                view = ds.sort_by("representativeness")[:n_repr]
                session = dm.get_session()
                if session:
                    dm.set_session_view(view)
                    st.success(f"✅ 已展示代表性最低的 {n_repr} 个样本")


def _render_backup(ds):
    st.subheader("数据集完整备份")
    st.caption(
        "创建数据集的完整备份，包括所有图像文件和标注数据。\n"
        "与快照（仅克隆 FiftyOne 元数据）不同，完整备份会复制实际的图像文件到备份目录。"
    )

    from datetime import datetime as _dt

    st.markdown("### 📦 创建新备份")
    default_backup_root = str(Path(CONFIG.default_export_dir).parent / "dataset_backups")
    backup_root = _path_browser("备份根目录", "backup_root", mode="dir", start_dir=default_backup_root)
    note = st.text_input("备份备注", key="backup_note", placeholder="例如: v1.0 标注完成后备份")
    backup_name = st.text_input(
        "备份子目录名（可选）", key="backup_name",
        placeholder=f"{ds.name}_{_dt.now().strftime('%Y%m%d_%H%M%S')}",
    )

    if st.button("💾 创建完整备份", key="btn_backup", type="primary"):
        if not backup_root:
            st.error("请指定备份根目录")
        else:
            sub = backup_name.strip() if backup_name.strip() else f"{ds.name}_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir = str(Path(backup_root) / sub)
            with st.spinner(f"正在备份 {len(ds)} 个样本（包含图像复制）..."):
                try:
                    result = advanced.backup_dataset(ds, backup_dir, note=note)
                    st.success("✅ 完整备份已创建")
                    st.json(result)
                    if result["images_failed"] > 0:
                        st.warning(f"⚠️ 有 {result['images_failed']} 个图像文件复制失败")
                except Exception as e:
                    st.error(f"备份失败: {e}")

    st.markdown("---")

    st.markdown("### 📋 已有备份")
    if backup_root and Path(backup_root).is_dir():
        backups = advanced.list_backups(backup_root)
        if backups:
            import pandas as pd
            df = pd.DataFrame(backups)
            display_cols = [c for c in ["dataset_name", "backup_time", "note", "num_samples",
                                        "images_copied", "metadata_exported"] if c in df.columns]
            st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)

            selected_backup = st.selectbox(
                "选择备份",
                [b["path"] for b in backups],
                format_func=lambda p: f"{Path(p).name} — {next((b.get('note', '') for b in backups if b['path'] == p), '')}",
                key="backup_select",
            )

            col_restore, col_delete = st.columns(2)
            with col_restore:
                restore_name = st.text_input("恢复为数据集名称", value=ds.name, key="backup_restore_name")
                if st.button("♻️ 从备份恢复", key="btn_backup_restore"):
                    st.warning(f"⚠️ 将覆盖数据集 '{restore_name}'！")
                confirm_restore = st.text_input(f"输入 `{restore_name}` 确认恢复", key="backup_restore_confirm")
                if st.button("✅ 确认恢复", key="btn_confirm_backup_restore"):
                    if confirm_restore == restore_name:
                        with st.spinner("恢复中..."):
                            try:
                                advanced.restore_from_backup(selected_backup, restore_name)
                                st.session_state.current_dataset = restore_name
                                st.session_state["_toast_msg"] = f"已从备份恢复: {restore_name}"
                                st.rerun()
                            except Exception as e:
                                st.error(f"恢复失败: {e}")
                    else:
                        st.error("名称不匹配")

            with col_delete:
                st.markdown("**删除备份**")
                confirm_del_backup = st.checkbox("⚠️ 确认删除此备份", key="confirm_del_backup")
                if st.button("🗑️ 删除此备份", key="btn_del_backup", disabled=not confirm_del_backup):
                    try:
                        advanced.delete_backup(selected_backup)
                        st.session_state["_toast_msg"] = "备份已删除"
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {e}")
        else:
            st.info("指定目录下暂无备份")
    else:
        st.info("请先指定备份根目录以查看已有备份")
