"""标注质量检查页面：一致性、面积异常、类别平衡分析。"""
from __future__ import annotations

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform.ui.components import _get_ds


def _render_quality_page():
    st.header("🔬 标注质量检查")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    info = dm.get_dataset_info(ds)
    label_fields = info.get("label_fields", [])
    if not label_fields:
        st.warning("当前数据集没有标签字段")
        return

    label_field = st.selectbox("选择标签字段", label_fields, key="qa_label_field")

    tab_balance, tab_area, tab_empty, tab_consistency = st.tabs([
        "📊 类别平衡", "📐 面积分析", "🔲 空标注检测", "🔍 标注一致性",
    ])

    with tab_balance:
        _render_class_balance(ds, label_field)
    with tab_area:
        _render_area_analysis(ds, label_field)
    with tab_empty:
        _render_empty_annotations(ds, label_field)
    with tab_consistency:
        _render_annotation_consistency(ds, label_field)


def _render_class_balance(ds, label_field: str):
    st.subheader("类别平衡分析")
    st.caption("可视化类别不平衡程度，给出过采样/欠采样建议")

    all_classes = dm.get_label_classes(ds, label_field)
    if not all_classes:
        st.info("该字段未检测到类别")
        return

    class_counts = {}
    ftype = dm.get_field_label_type(ds, label_field)

    for sample in ds.iter_samples():
        labels = sample.get_field(label_field)
        if labels is None:
            continue
        items = getattr(labels, "detections", None) or getattr(labels, "polylines", None) or []
        for item in items:
            cls = item.label if hasattr(item, "label") else str(item)
            class_counts[cls] = class_counts.get(cls, 0) + 1

    if not class_counts:
        st.info("未找到任何标注实例")
        return

    import pandas as pd
    df = pd.DataFrame([
        {"类别": cls, "数量": cnt} for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1])
    ])
    st.bar_chart(df.set_index("类别"))
    st.dataframe(df, use_container_width=True, hide_index=True)

    total = sum(class_counts.values())
    ideal = total / len(class_counts) if class_counts else 0
    st.markdown("### 💡 平衡建议")

    imbalanced = []
    for cls, cnt in class_counts.items():
        ratio = cnt / ideal if ideal > 0 else 0
        if ratio < 0.5:
            imbalanced.append(f"- **{cls}**: {cnt} 实例 (仅为理想值的 {ratio:.0%}) → 建议**补充采集或过采样**")
        elif ratio > 2.0:
            imbalanced.append(f"- **{cls}**: {cnt} 实例 (理想值的 {ratio:.0%}) → 建议**欠采样或加权训练**")

    if imbalanced:
        st.warning("检测到类别不平衡：\n" + "\n".join(imbalanced))
    else:
        st.success("✅ 类别分布相对均衡")

    imbalance_ratio = max(class_counts.values()) / min(class_counts.values()) if min(class_counts.values()) > 0 else float("inf")
    st.metric("最大/最小类别比", f"{imbalance_ratio:.1f}x")


def _render_area_analysis(ds, label_field: str):
    st.subheader("标注面积分析")
    st.caption("检测面积异常的标注（过小或过大）")

    min_area_pct = st.number_input("最小面积阈值 (%图像面积)", 0.0, 100.0, 0.01, 0.01, key="qa_min_area",
                                   help="bbox面积小于此百分比的标注视为异常小标注")
    max_area_pct = st.number_input("最大面积阈值 (%图像面积)", 0.0, 100.0, 90.0, 1.0, key="qa_max_area",
                                   help="bbox面积大于此百分比的标注视为异常大标注")

    if st.button("🔍 分析面积分布", key="btn_area_analysis"):
        areas = []
        small_samples = set()
        large_samples = set()

        for sample in ds.iter_samples():
            labels = sample.get_field(label_field)
            if labels is None:
                continue
            items = getattr(labels, "detections", None) or []
            for det in items:
                if hasattr(det, "bounding_box") and det.bounding_box:
                    bbox = det.bounding_box
                    area_pct = bbox[2] * bbox[3] * 100
                    areas.append({"sample_id": sample.id, "class": det.label, "area_%": area_pct})
                    if area_pct < min_area_pct:
                        small_samples.add(sample.id)
                    if area_pct > max_area_pct:
                        large_samples.add(sample.id)

        if not areas:
            st.info("该字段中未找到带 bounding_box 的标注")
            return

        import pandas as pd
        df = pd.DataFrame(areas)

        col1, col2, col3 = st.columns(3)
        col1.metric("总标注数", len(areas))
        col2.metric("异常小标注样本", len(small_samples))
        col3.metric("异常大标注样本", len(large_samples))

        st.markdown("**面积分布直方图**")
        st.bar_chart(df["area_%"].value_counts(bins=50).sort_index())

        if small_samples:
            st.warning(f"⚠️ {len(small_samples)} 个样本包含异常小标注 (< {min_area_pct}% 图像面积)")
            if st.button("👁️ 查看异常小标注样本", key="btn_view_small"):
                from fiftyone import ViewField as F
                view = ds.select(list(small_samples))
                session = dm.get_session()
                if session:
                    dm.set_session_view(view)
                    st.success("已在 FiftyOne 中展示")

        if large_samples:
            st.warning(f"⚠️ {len(large_samples)} 个样本包含异常大标注 (> {max_area_pct}% 图像面积)")
            if st.button("👁️ 查看异常大标注样本", key="btn_view_large"):
                view = ds.select(list(large_samples))
                session = dm.get_session()
                if session:
                    dm.set_session_view(view)
                    st.success("已在 FiftyOne 中展示")


def _render_empty_annotations(ds, label_field: str):
    st.subheader("空标注检测")
    st.caption("检测有标注字段但 detections 为空列表的样本")

    if st.button("🔍 检测空标注", key="btn_empty_check"):
        empty_ids = []
        for sample in ds.iter_samples():
            labels = sample.get_field(label_field)
            if labels is not None:
                items = getattr(labels, "detections", None) or getattr(labels, "polylines", None)
                if items is not None and len(items) == 0:
                    empty_ids.append(sample.id)

        if empty_ids:
            st.warning(f"发现 **{len(empty_ids)}** 个样本有标注字段但内容为空")
            if st.button("👁️ 在 FiftyOne 中查看", key="btn_view_empty"):
                view = ds.select(empty_ids)
                session = dm.get_session()
                if session:
                    dm.set_session_view(view)
                    st.success("已在 FiftyOne 中展示")
            st.caption("这些样本可能是标注过程中被跳过的图像，建议重新标注或标记为无目标。")
        else:
            st.success("✅ 未发现空标注样本")


def _render_annotation_consistency(ds, label_field: str):
    st.subheader("标注一致性分析")
    st.caption("分析不同 anno_key 之间的标注分布差异，检测标注员偏差")

    runs = ds.list_annotation_runs() if hasattr(ds, "list_annotation_runs") else []
    if not runs:
        st.info("当前数据集没有标注运行记录，无法分析标注一致性。")
        return

    selected_runs = st.multiselect("选择标注运行对比", runs, default=runs[:2] if len(runs) >= 2 else runs,
                                   key="qa_runs")

    if len(selected_runs) < 2:
        st.info("请至少选择 2 个标注运行进行对比")
        return

    if st.button("🔍 分析一致性", key="btn_consistency"):
        all_classes = dm.get_label_classes(ds, label_field)
        if not all_classes:
            st.info("未找到类别信息")
            return

        run_class_counts = {}
        for run_key in selected_runs:
            try:
                run_view = ds.load_annotation_view(run_key)
            except Exception:
                run_view = ds
            counts = {}
            for sample in run_view.iter_samples():
                labels = sample.get_field(label_field)
                if labels is None:
                    continue
                items = getattr(labels, "detections", None) or getattr(labels, "polylines", None) or []
                for item in items:
                    cls = item.label if hasattr(item, "label") else str(item)
                    counts[cls] = counts.get(cls, 0) + 1
            run_class_counts[run_key] = counts

        import pandas as pd
        comparison = []
        for cls in sorted(set().union(*[set(c.keys()) for c in run_class_counts.values()])):
            row = {"类别": cls}
            for run_key in selected_runs:
                row[run_key] = run_class_counts[run_key].get(cls, 0)
            comparison.append(row)

        df = pd.DataFrame(comparison)
        st.dataframe(df, use_container_width=True, hide_index=True)

        numeric_cols = [c for c in df.columns if c != "类别"]
        if len(numeric_cols) >= 2:
            st.bar_chart(df.set_index("类别")[numeric_cols])
            st.caption("如果不同标注运行之间的类别分布差异很大，可能存在标注员偏差，建议进行交叉审核。")
