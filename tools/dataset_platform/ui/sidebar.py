"""侧边栏：数据集管理与 CVAT 配置。"""
from __future__ import annotations
import streamlit as st
from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import cvat_sync
from tools.dataset_platform.config import CONFIG, save_config
from tools.dataset_platform.ui.components import _get_ds


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

    if st.sidebar.button("💾 保存配置", key="btn_save_config"):
        save_config(CONFIG)
        st.sidebar.success("✅ 配置已保存到本地")

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
                orgs = result.get("organizations", [])
                if orgs:
                    org_names = [f"`{o['slug']}`" + (f" ({o['name']})" if o['name'] else "") for o in orgs]
                    info_lines.append(f"- 组织: {', '.join(org_names)}")
                    if not CONFIG.cvat.organization:
                        st.sidebar.warning(
                            f"检测到你的账号属于组织 {', '.join(org_names)}，"
                            "但「Organization」字段为空。如果推送失败，请填入对应的组织 slug。"
                        )
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

