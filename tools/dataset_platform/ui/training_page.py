"""训练管理页面：配置面板、后台训练、训练进度与历史、权重管理、训练后回灌。"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from tools.dataset_platform import data_manager as dm
from tools.dataset_platform import trainer
from tools.dataset_platform.config import CONFIG
from tools.dataset_platform.ui.components import _get_ds, _path_browser

_BASE_MODEL_DIR = Path.home() / ".dataset_platform" / "base_model"
_DEFAULT_DATA_YAML = "/home/cotek/datasets/temp/data.yaml"


def _list_base_models() -> list[str]:
    """扫描 ~/.dataset_platform/base_model/ 下的权重文件。"""
    if not _BASE_MODEL_DIR.is_dir():
        _BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = []
    for f in sorted(_BASE_MODEL_DIR.iterdir()):
        if f.suffix.lower() in (".pt", ".pth", ".yaml") and f.is_file():
            models.append(str(f))
    return models


def _render_training_page():
    st.header("🏋️ 训练管理")
    ds = _get_ds()
    if ds is None:
        st.info("请先选择数据集")
        return

    tab_train, tab_progress, tab_predict, tab_feedback = st.tabs([
        "🚀 训练配置与启动", "📊 训练进度与历史", "🔮 通用预测", "🔄 训练后回灌",
    ])

    with tab_train:
        _render_train_config(ds)
    with tab_progress:
        _render_progress_and_history(ds)
    with tab_predict:
        _render_predict(ds)
    with tab_feedback:
        _render_train_feedback(ds)


def _render_train_config(ds):
    st.subheader("训练配置")

    status = trainer.get_training_status()
    if status["running"]:
        st.warning(f"⏳ 训练进行中: {status['progress']} — 请前往「训练进度与历史」查看实时进度")
        return

    st.markdown("---")

    auto_yaml = st.session_state.get("last_export_data_yaml", "")

    # ═══════════════════ 模型配置 & 数据配置 并排 ═══════════════════
    col_model, col_data = st.columns(2)

    # ── 模型配置（左侧） ──
    with col_model:
        st.markdown("#### 🧠 模型配置")

        base_models = _list_base_models()
        model_source = st.radio(
            "基础模型来源",
            ["基础模型目录", "自定义路径", "输入模型名称"],
            key="train_model_source", horizontal=True,
            help=f"基础模型目录: `{_BASE_MODEL_DIR}`",
        )

        model_path = ""
        if model_source == "基础模型目录":
            if base_models:
                model_path = st.selectbox(
                    "选择基础模型", base_models,
                    format_func=lambda p: Path(p).name,
                    key="train_base_model",
                )
            else:
                st.info(f"目录 `{_BASE_MODEL_DIR}` 为空，请先放入 .pt 权重文件")
        elif model_source == "自定义路径":
            last_best = ds.info.get("last_best_pt", "")
            model_path = _path_browser(
                "模型权重 (.pt)", "train_model_path", mode="file",
                file_extensions=(".pt", ".pth", ".yaml"),
                start_dir=str(Path(last_best).parent) if last_best and Path(last_best).exists() else "",
            )
        else:
            model_path = st.text_input(
                "模型名称", value="yolo11n.pt", key="train_model_name",
                help="Ultralytics 预训练模型名称，首次使用会自动下载",
            )

        task = st.selectbox(
            "任务类型",
            ["detect", "pose", "obb", "classify", "segment"],
            key="train_task",
        )

        _default_project = f"/home/cotek/ws_cotek/ultralytics/runs/{task}"
        project_source = st.radio(
            "训练输出目录", ["默认路径", "自定义路径"],
            key="train_project_source", horizontal=True,
        )
        if project_source == "默认路径":
            project_dir = st.text_input(
                "输出目录 (project)", value=_default_project, key=f"train_project_{task}",
                help=f"默认: `{_default_project}`，训练产出保存在此目录下",
            )
        else:
            project_dir = _path_browser(
                "输出目录 (project)", "train_project_custom", mode="directory",
            )

    # ── 数据配置（右侧） ──
    with col_data:
        st.markdown("#### 📂 数据配置")

        yaml_source = st.radio(
            "data.yaml 来源",
            ["默认路径", "自定义路径"],
            key="train_yaml_source", horizontal=True,
        )

        if yaml_source == "默认路径":
            default_path = auto_yaml if auto_yaml else _DEFAULT_DATA_YAML
            data_yaml = st.text_input(
                "data.yaml 路径", value=default_path, key="train_data_yaml_default",
                help="来自 train.py 的默认路径，可直接修改",
            )
        else:
            data_yaml = _path_browser(
                "data.yaml 路径", "train_data_yaml", mode="file",
                file_extensions=(".yaml", ".yml"),
                start_dir=st.session_state.get("last_export_dir", CONFIG.default_export_dir),
            )

        if data_yaml and Path(data_yaml).exists():
            st.caption(f"✅ `{Path(data_yaml).name}`")
        elif data_yaml:
            st.warning("⚠️ 文件不存在")

        run_name = st.text_input(
            "运行名称 (name)", value="", key="train_name",
            placeholder="留空则自动命名",
            help="本次训练的子目录名，方便区分不同实验",
        )

    st.markdown("---")

    # ═══════════════════ 训练超参数 ═══════════════════
    st.markdown("#### ⚙️ 训练超参数")
    col_e, col_i, col_b, col_d = st.columns(4)
    with col_e:
        epochs = st.number_input("Epochs", 1, 1000, 150, key="train_epochs")
    with col_i:
        imgsz = st.number_input("Image Size", 32, 1920, 640, step=32, key="train_imgsz")
    with col_b:
        batch = st.number_input("Batch Size", 1, 256, 16, key="train_batch")
    with col_d:
        device = st.text_input("Device", value="0", key="train_device",
                               help="GPU 编号如 0, 或 cpu, 或 0,1 多卡")

    # ═══════════════════ 色彩空间增强 ═══════════════════
    with st.expander("🎨 色彩空间增强", expanded=False):
        col_h, col_s, col_v = st.columns(3)
        with col_h:
            hsv_h = st.slider("hsv_h (色调)", 0.0, 1.0, 0.015, 0.005, key="train_hsv_h",
                              help="色调微调，适应不同色温光源")
        with col_s:
            hsv_s = st.slider("hsv_s (饱和度)", 0.0, 1.0, 0.7, 0.05, key="train_hsv_s",
                              help="饱和度变化，模拟光照对颜色的影响")
        with col_v:
            hsv_v = st.slider("hsv_v (亮度)", 0.0, 1.0, 0.5, 0.05, key="train_hsv_v",
                              help="亮度变化，仓库明暗差异大")

    # ═══════════════════ 几何变换 ═══════════════════
    with st.expander("📐 几何变换", expanded=False):
        st.caption("保守策略：大目标场景需避免边界裁剪导致标注偏移")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            degrees = st.slider("degrees (旋转)", 0.0, 45.0, 3.0, 1.0, key="train_degrees")
            translate = st.slider("translate (平移)", 0.0, 0.9, 0.0, 0.05, key="train_translate")
        with col_g2:
            scale = st.slider("scale (缩放)", 0.0, 0.9, 0.0, 0.05, key="train_scale")
            shear = st.slider("shear (剪切)", 0.0, 10.0, 2.0, 0.5, key="train_shear")
        with col_g3:
            perspective = st.number_input("perspective (透视)", 0.0, 0.01, 0.0003, 0.0001,
                                          format="%.4f", key="train_perspective")
            fliplr = st.slider("fliplr (左右翻转)", 0.0, 1.0, 0.5, 0.1, key="train_fliplr")
            flipud = st.slider("flipud (上下翻转)", 0.0, 1.0, 0.0, 0.1, key="train_flipud")

    # ═══════════════════ 组合增强 ═══════════════════
    with st.expander("🧩 组合增强", expanded=False):
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            mosaic = st.slider("mosaic", 0.0, 1.0, 1.0, 0.1, key="train_mosaic")
        with col_m2:
            close_mosaic = st.number_input("close_mosaic (最后N个epoch关闭)", 0, 100, 15,
                                           key="train_close_mosaic")
        with col_m3:
            mixup = st.slider("mixup", 0.0, 1.0, 0.0, 0.1, key="train_mixup")
        with col_m4:
            cutmix = st.slider(
                "cutmix", 0.0, 1.0, 0.0, 0.05, key="train_cutmix",
                help="检测任务可用的混合增强。类别不均衡时可小幅开启，如 0.05~0.15。",
            )
        multi_scale = st.slider(
            "multi_scale (多尺度训练概率)", 0.0, 1.0, 0.0, 0.1, key="train_multi_scale",
            help="随机改变训练尺寸提升泛化。小目标/尺度变化大时可尝试 0.3~0.5。",
        )

    # ═══════════════════ 损失函数与类别平衡 ═══════════════════
    with st.expander("⚖️ 损失函数与类别平衡", expanded=False):
        st.caption("类别不平衡时，优先启用少数类重采样；再适度提高 cls 权重")
        col_bal, col_cls, col_box, col_dfl = st.columns(4)
        with col_bal:
            balance_enable = st.checkbox("少数类重采样", value=False, key="train_balance_enable")
            balance_names = st.text_input(
                "少数类名称(逗号分隔)", value="tian", key="train_balance_names",
                help="仅用于训练前构造重采样列表，不影响标注",
            )
            balance_reference_names = st.text_input(
                "参考多数类(逗号分隔)", value="chuan", key="train_balance_reference_names",
                help="例如 chuan。留空时使用所有非少数类作为参考。",
            )
            balance_ratio = st.number_input(
                "目标比例(少/多)", 0.1, 1.0, 0.5, 0.1, key="train_balance_ratio",
                help="重采样后少数类实例数 / 参考多数类实例数，建议从 0.5~0.8 开始",
            )
            balance_max_repeat = st.number_input(
                "最大重复倍数", 1, 10, 3, 1, key="train_balance_max_repeat",
                help="少数类样本最多重复次数",
            )
            balance_pure_minority_only = st.checkbox(
                "只重采样纯少数类图片", value=True, key="train_balance_pure_minority_only",
                help="启用后只重复标注文件中不含参考多数类/其他类别的少数类图片，避免重复 tian 时同时增加 chuan。",
            )
        with col_cls:
            cls_weight = st.number_input(
                "cls (分类损失权重)", 0.0, 10.0, 0.8, 0.1, key="train_cls",
                help="默认 0.5。chuan/tian 易混淆或 tian 召回低时可提高到 0.8~1.5。",
            )
        with col_box:
            box_weight = st.number_input(
                "box (框回归损失权重)", 0.0, 20.0, 7.5, 0.5, key="train_box",
                help="默认 7.5",
            )
        with col_dfl:
            dfl_weight = st.number_input(
                "dfl (分布焦点损失权重)", 0.0, 10.0, 1.5, 0.1, key="train_dfl",
                help="默认 1.5",
            )

    # ═══════════════════ 优化器与学习率 ═══════════════════
    with st.expander("📈 优化器与学习率", expanded=False):
        col_lr, col_lrf, col_wd, col_opt = st.columns(4)
        with col_lr:
            lr0 = st.number_input("初始学习率 (lr0)", 0.0001, 1.0, 0.01, 0.001,
                                  format="%.4f", key="train_lr0")
        with col_lrf:
            lrf = st.number_input(
                "最终学习率比例 (lrf)", 0.001, 1.0, 0.01, 0.001,
                format="%.3f", key="train_lrf",
            )
        with col_wd:
            weight_decay = st.number_input("权重衰减", 0.0, 0.1, 0.0005, 0.0001,
                                           format="%.4f", key="train_wd")
        with col_opt:
            optimizer = st.selectbox("优化器", ["auto", "SGD", "Adam", "AdamW"],
                                     key="train_optimizer")
        col_cos, col_pat = st.columns(2)
        with col_cos:
            cos_lr = st.checkbox("余弦退火学习率 (cos_lr)", value=False, key="train_cos_lr",
                                 help="使用余弦退火调度，后期学习率平滑下降")
        with col_pat:
            patience = st.number_input("早停耐心值 (patience)", 0, 500, 100, key="train_patience",
                                       help="默认 100。类别不平衡时建议设为 50~100，给模型更多学习少数类的机会")
        col_mom, col_warm, col_warm_mom = st.columns(3)
        with col_mom:
            momentum = st.number_input("momentum", 0.0, 1.0, 0.937, 0.001,
                                       format="%.3f", key="train_momentum")
        with col_warm:
            warmup_epochs = st.number_input("warmup_epochs", 0.0, 10.0, 3.0, 0.5,
                                            key="train_warmup_epochs")
        with col_warm_mom:
            warmup_momentum = st.number_input("warmup_momentum", 0.0, 1.0, 0.8, 0.05,
                                              key="train_warmup_momentum")

    st.markdown("---")

    # ═══════════════════ 启动训练 ═══════════════════
    if st.button("🚀 开始训练", key="btn_start_train", type="primary"):
        if not data_yaml:
            st.error("请指定 data.yaml 路径")
            return
        if not model_path:
            st.error("请指定模型路径")
            return

        extra_args = {
            "lr0": lr0, "lrf": lrf, "momentum": momentum,
            "weight_decay": weight_decay, "optimizer": optimizer,
            "cos_lr": cos_lr, "patience": patience,
            "warmup_epochs": warmup_epochs, "warmup_momentum": warmup_momentum,
            "cls": cls_weight, "box": box_weight, "dfl": dfl_weight,
            "hsv_h": hsv_h, "hsv_s": hsv_s, "hsv_v": hsv_v,
            "degrees": degrees, "translate": translate, "scale": scale,
            "shear": shear, "perspective": perspective,
            "fliplr": fliplr, "flipud": flipud,
            "mosaic": mosaic, "close_mosaic": close_mosaic, "mixup": mixup,
            "cutmix": cutmix, "multi_scale": multi_scale,
        }
        if balance_enable:
            names = [n.strip() for n in balance_names.split(",") if n.strip()]
            reference_names = [n.strip() for n in balance_reference_names.split(",") if n.strip()]
            extra_args["balance_classes"] = names
            extra_args["balance_reference_classes"] = reference_names
            extra_args["balance_target_ratio"] = float(balance_ratio)
            extra_args["balance_max_repeat"] = int(balance_max_repeat)
            extra_args["balance_pure_minority_only"] = bool(balance_pure_minority_only)

        result = trainer.start_training(
            data_yaml=data_yaml,
            model_path=model_path,
            task=task,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project_dir,
            name=run_name,
            extra_args=extra_args,
            ds=ds,
        )

        if result["started"]:
            st.success(result["message"])
            st.info("训练在后台运行中，可定期刷新此页面查看进度。训练完成后结果会自动记录到「训练历史」。")
        else:
            st.error(result["message"])


def _fmt_duration(seconds: float) -> str:
    """将秒数格式化为可读时长。"""
    if seconds < 60:
        return f"{seconds:.0f}秒"
    if seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}分{s}秒"
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}时{m}分{s}秒"


def _render_progress_and_history(ds):
    st.subheader("训练进度与历史")

    status = trainer.get_training_status()

    # ═══════════════════ 实时训练进度 ═══════════════════
    if status["running"]:
        st.markdown("### ⏳ 训练进行中")

        epoch = status.get("epoch", 0)
        total = status.get("total_epochs", 0) or 1
        progress_pct = epoch / total

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric("当前轮数", f"{epoch} / {total}")
        col_p2.metric("已用时间", _fmt_duration(status.get("duration_seconds", 0)))
        if epoch > 0:
            eta = status.get("duration_seconds", 0) / epoch * (total - epoch)
            col_p3.metric("预计剩余", _fmt_duration(eta))
        else:
            col_p3.metric("预计剩余", "计算中...")

        st.progress(progress_pct, text=status.get("progress", ""))

        cur_m = status.get("current_metrics", {})
        if cur_m:
            st.markdown("**当前指标**")
            _display_metrics_row(cur_m)

        # 显示 epoch 历史曲线（从 results.csv）
        epoch_hist = status.get("epoch_history", [])
        if not epoch_hist:
            save_dir = status.get("save_dir")
            if save_dir:
                epoch_hist = trainer.read_results_csv(save_dir)

        if epoch_hist:
            _render_epoch_chart(epoch_hist)

        st.markdown("---")
        auto_refresh = st.checkbox("自动刷新（每 5 秒）", value=True, key="auto_refresh_progress")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 手动刷新", key="btn_manual_refresh_progress"):
                st.rerun()
        if auto_refresh:
            time.sleep(5)
            st.rerun()
        return

    # 显示上次训练结果摘要（如果刚完成）
    if status.get("result"):
        result = status["result"]
        st.markdown("### ✅ 最近一次训练已完成")
        col_r1, col_r2, col_r3 = st.columns(3)
        start_ts = result.get("timestamp", "")
        if start_ts:
            try:
                col_r1.metric("开始时间", datetime.fromisoformat(start_ts).strftime("%m-%d %H:%M"))
            except Exception:
                col_r1.metric("开始时间", start_ts[:16])
        dur = result.get("duration_seconds", 0)
        if dur:
            col_r2.metric("训练用时", _fmt_duration(dur))
        col_r3.metric("Epochs", result.get("epochs", ""))

        metrics = result.get("metrics", {})
        if metrics:
            _display_metrics_row(metrics)

        # 显示完成训练的 epoch 曲线
        rd = result.get("result_dir")
        if rd:
            epoch_hist = trainer.read_results_csv(rd)
            if epoch_hist:
                _render_epoch_chart(epoch_hist)

        with st.expander("完整训练记录", expanded=False):
            st.json(result)
        st.markdown("---")
    elif status.get("error"):
        st.error(f"上次训练失败: {status['error']}")
        dur = status.get("duration_seconds", 0)
        if dur:
            st.caption(f"运行时长: {_fmt_duration(dur)}")
        trace = status.get("error_trace")
        if trace:
            with st.expander("查看详细错误", expanded=False):
                st.code(trace)
        st.markdown("---")

    # ═══════════════════ 训练历史记录 ═══════════════════
    st.markdown("### 📜 训练历史记录")

    history = trainer.get_training_history(ds)
    if not history:
        st.info("当前数据集暂无训练记录。完成训练后会自动记录在此。")
    else:
        import pandas as pd

        rows = []
        for i, h in enumerate(reversed(history)):
            metrics = h.get("metrics", {})
            start_ts = h.get("timestamp", "")
            try:
                start_display = datetime.fromisoformat(start_ts).strftime("%Y-%m-%d %H:%M")
            except Exception:
                start_display = start_ts[:16] if start_ts else ""

            dur = h.get("duration_seconds")
            row = {
                "#": len(history) - i,
                "开始时间": start_display,
                "用时": _fmt_duration(dur) if dur else "-",
                "模型": Path(h.get("model", "")).name,
                "任务": h.get("task", ""),
                "Epochs": h.get("epochs", ""),
                "ImgSz": h.get("imgsz", ""),
            }
            mAP50 = metrics.get("metrics/mAP50(B)") or metrics.get("mAP50(B)")
            mAP50_95 = metrics.get("metrics/mAP50-95(B)") or metrics.get("mAP50-95(B)")
            if mAP50 is not None:
                row["mAP50"] = f"{mAP50:.4f}"
            if mAP50_95 is not None:
                row["mAP50-95"] = f"{mAP50_95:.4f}"
            row["best.pt"] = h.get("best_pt", "")
            rows.append(row)

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ═══════════════════ 权重管理 ═══════════════════
    st.markdown("### 🏆 最优权重管理")
    last_best = ds.info.get("last_best_pt", "")
    if last_best:
        st.info(f"当前最优权重: `{last_best}`")
        if st.button("📋 设为预标注模型", key="btn_set_predict_model"):
            st.session_state["pred_model_input"] = last_best
            st.success(f"已将 `{last_best}` 设为预标注模型，可前往「自动预标注」页面使用")

    if history:
        st.markdown("---")
        st.markdown("**选择历史权重**")
        weight_options = [h.get("best_pt", "") for h in history if h.get("best_pt")]
        if weight_options:
            selected_weight = st.selectbox("选择权重", weight_options, key="hist_weight_select")
            if st.button("设为当前最优权重", key="btn_set_best"):
                ds.info["last_best_pt"] = selected_weight
                ds.save()
                st.session_state["_toast_msg"] = f"最优权重已更新: {selected_weight}"
                st.rerun()


def _display_metrics_row(metrics: dict):
    """以 metric 卡片方式展示关键指标。"""
    _METRIC_LABELS = {
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
        "metrics/precision(B)": "Precision",
        "metrics/recall(B)": "Recall",
        "train/box_loss": "Box Loss",
        "train/cls_loss": "Cls Loss",
        "train/dfl_loss": "DFL Loss",
        "val/box_loss": "Val Box",
        "val/cls_loss": "Val Cls",
        "val/dfl_loss": "Val DFL",
    }
    display_items = []
    for key, label in _METRIC_LABELS.items():
        val = metrics.get(key)
        if val is not None and isinstance(val, (int, float)):
            display_items.append((label, val))
    # 兜底：显示所有数值指标
    if not display_items:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                display_items.append((k, v))

    if display_items:
        cols = st.columns(min(len(display_items), 5))
        for i, (label, val) in enumerate(display_items[:10]):
            cols[i % len(cols)].metric(label, f"{val:.4f}")


def _render_epoch_chart(epoch_history: list[dict]):
    """用 st.line_chart 绘制训练过程曲线。"""
    import pandas as pd

    if not epoch_history:
        return

    df = pd.DataFrame(epoch_history)

    # 尝试用 epoch 列作为索引
    epoch_col = None
    for candidate in ("epoch", "Epoch"):
        if candidate in df.columns:
            epoch_col = candidate
            break
    if epoch_col:
        df = df.set_index(epoch_col)

    # 只保留数值列
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return

    loss_cols = [c for c in num_cols if "loss" in c.lower()]
    map_cols = [c for c in num_cols if "map" in c.lower() or "precision" in c.lower() or "recall" in c.lower()]

    with st.expander("📈 训练曲线", expanded=True):
        if loss_cols:
            st.caption("Loss 曲线")
            st.line_chart(df[loss_cols])
        if map_cols:
            st.caption("精度曲线")
            st.line_chart(df[map_cols])


# ═══════════════════ 通用预测 ═══════════════════

_SUPPORTED_TASKS = ["detect", "segment", "pose", "obb", "classify"]
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def _render_predict(ds):
    st.subheader("通用预测")
    st.caption("基于 predict.py，使用训练好的模型对任意输入源执行推理预测。")

    # ── 模型选择 ──
    st.markdown("#### 🧠 模型权重")
    last_best = ds.info.get("last_best_pt", "")
    history = trainer.get_training_history(ds)
    weight_options = [h.get("best_pt", "") for h in reversed(history) if h.get("best_pt")]

    model_source = st.radio(
        "权重来源",
        ["最新 best.pt", "历史权重", "自定义路径", "输入模型名称"],
        key="pred_model_source", horizontal=True,
    )

    pred_model = ""
    if model_source == "最新 best.pt":
        if last_best and Path(last_best).exists():
            pred_model = last_best
            st.info(f"使用最新权重: `{Path(last_best).name}` — `{last_best}`")
        elif last_best:
            st.warning(f"最新权重文件不存在: `{last_best}`，请选择其他来源")
        else:
            st.warning("暂无训练权重记录。请先完成训练或选择其他来源。")
    elif model_source == "历史权重":
        if weight_options:
            pred_model = st.selectbox(
                "选择历史权重（最新在前）", weight_options,
                format_func=lambda p: Path(p).name,
                key="pred_hist_weight",
            )
        else:
            st.warning("暂无历史权重记录")
    elif model_source == "自定义路径":
        pred_model = _path_browser(
            "模型权重 (.pt)", "pred_model_browser", mode="file",
            file_extensions=(".pt", ".pth", ".onnx", ".engine"),
            start_dir=str(Path(last_best).parent) if last_best and Path(last_best).parent.exists() else "",
        )
    else:
        pred_model = st.text_input(
            "模型名称", value="yolo11n.pt", key="pred_model_name",
            help="Ultralytics 预训练模型名称",
        )

    st.markdown("---")

    # ── 输入源 ──
    st.markdown("#### 📂 输入源")
    _DEFAULT_SOURCE = "/home/cotek/ws_cotek/ultralytics/detection_images"
    source_type = st.radio(
        "输入源类型",
        ["默认文件夹", "自定义文件夹", "图片/视频文件", "URL / 流地址"],
        key="pred_source_type", horizontal=True,
    )

    source = ""
    if source_type == "默认文件夹":
        source = st.text_input(
            "输入源路径", value=_DEFAULT_SOURCE, key="pred_source_default",
            help=f"默认: `{_DEFAULT_SOURCE}`",
        )
        if source and Path(source).is_dir():
            imgs = [f for f in Path(source).iterdir() if f.suffix.lower() in _IMAGE_EXTS]
            vids = [f for f in Path(source).iterdir() if f.suffix.lower() in _VIDEO_EXTS]
            st.caption(f"包含 {len(imgs)} 张图片, {len(vids)} 个视频")
        elif source:
            st.warning("⚠️ 目录不存在")
    elif source_type == "图片/视频文件":
        source = _path_browser(
            "选择图片或视频", "pred_source_file", mode="file",
            file_extensions=tuple(_IMAGE_EXTS | _VIDEO_EXTS),
        )
    elif source_type == "自定义文件夹":
        source = _path_browser(
            "选择文件夹", "pred_source_dir", mode="directory",
        )
        if source and Path(source).is_dir():
            imgs = [f for f in Path(source).iterdir() if f.suffix.lower() in _IMAGE_EXTS]
            vids = [f for f in Path(source).iterdir() if f.suffix.lower() in _VIDEO_EXTS]
            st.caption(f"包含 {len(imgs)} 张图片, {len(vids)} 个视频")
    else:
        source = st.text_input(
            "URL / 流地址", value="", key="pred_source_url",
            placeholder="http://... 或 rtsp://...",
        )

    st.markdown("---")

    # ── 推理参数 ──
    st.markdown("#### ⚙️ 推理参数")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1:
        pred_task = st.selectbox("任务类型", [None] + _SUPPORTED_TASKS,
                                 format_func=lambda x: "自动检测" if x is None else x,
                                 key="pred_task")
    with col_p2:
        pred_conf = st.slider("置信度", 0.0, 1.0, 0.4, 0.05, key="pred_conf")
    with col_p3:
        pred_iou = st.slider("NMS IoU", 0.0, 1.0, 0.2, 0.05, key="pred_iou")
    with col_p4:
        pred_imgsz = st.number_input("图片尺寸", 32, 1920, 640, step=32, key="pred_imgsz")

    col_p5, col_p6, col_p7 = st.columns(3)
    with col_p5:
        pred_device = st.text_input("Device", value="0", key="pred_device",
                                    help="GPU 编号如 0，或 cpu")
    with col_p6:
        pred_max_det = st.number_input("最大检测数", 1, 10000, 300, key="pred_max_det")
    with col_p7:
        pred_half = st.checkbox("FP16 半精度", value=False, key="pred_half")

    # ── 输出控制 ──
    with st.expander("📁 输出控制", expanded=False):
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            pred_save = st.checkbox("保存标注图/视频", value=True, key="pred_save")
            pred_save_txt = st.checkbox("保存 txt 标签", value=False, key="pred_save_txt")
            pred_save_conf = st.checkbox("txt 中包含置信度", value=False, key="pred_save_conf")
            pred_save_crop = st.checkbox("保存裁剪目标", value=False, key="pred_save_crop")
        with col_o2:
            pred_project = st.text_input(
                "输出目录 (project)", value="", key="pred_project",
                placeholder="留空则自动 runs/{task}",
            )
            pred_name = st.text_input(
                "子目录名 (name)", value="predict", key="pred_run_name",
            )
            pred_exist_ok = st.checkbox("允许覆盖已有目录", value=False, key="pred_exist_ok")

    # ── 显示控制 ──
    with st.expander("👁️ 显示控制", expanded=False):
        col_v1, col_v2, col_v3 = st.columns(3)
        with col_v1:
            pred_show_labels = st.checkbox("显示标签", value=True, key="pred_show_labels")
        with col_v2:
            pred_show_conf = st.checkbox("显示置信度", value=True, key="pred_show_conf")
        with col_v3:
            pred_line_width = st.number_input("边框线宽 (0=自动)", 0, 20, 0, key="pred_line_width")

    st.markdown("---")

    # ── 执行预测 ──
    if st.button("🔮 开始预测", key="btn_start_predict", type="primary"):
        if not pred_model:
            st.error("请选择或输入模型权重")
            return
        if not source:
            st.error("请指定输入源")
            return

        with st.spinner("正在执行预测..."):
            try:
                from ultralytics import YOLO
                model = YOLO(pred_model, task=pred_task if pred_task else None)

                project = pred_project.strip() or None
                predict_kwargs = dict(
                    source=source,
                    conf=pred_conf,
                    iou=pred_iou,
                    imgsz=pred_imgsz,
                    device=int(pred_device) if pred_device.strip().isdigit() else pred_device.strip(),
                    max_det=pred_max_det,
                    save=pred_save,
                    save_txt=pred_save_txt,
                    save_conf=pred_save_conf,
                    save_crop=pred_save_crop,
                    project=project,
                    name=pred_name,
                    exist_ok=pred_exist_ok,
                    show_labels=pred_show_labels,
                    show_conf=pred_show_conf,
                    half=pred_half,
                )
                if pred_task:
                    predict_kwargs["task"] = pred_task
                if pred_line_width > 0:
                    predict_kwargs["line_width"] = pred_line_width

                results = list(model.predict(**predict_kwargs))

                total_det = 0
                for r in results:
                    if r.boxes is not None:
                        total_det += len(r.boxes)
                    elif r.obb is not None:
                        total_det += len(r.obb)

                save_dir = results[0].save_dir if results and hasattr(results[0], "save_dir") else None

                st.success("预测完成")
                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("处理文件数", len(results))
                col_r2.metric("总检测数", total_det)
                if save_dir:
                    col_r3.metric("输出目录", str(save_dir))

                # 展示部分预测结果图
                if pred_save and save_dir and Path(save_dir).is_dir():
                    preview_imgs = sorted(Path(save_dir).glob("*.jpg"))[:6]
                    if not preview_imgs:
                        preview_imgs = sorted(Path(save_dir).glob("*.png"))[:6]
                    if preview_imgs:
                        st.markdown("**预测结果预览**")
                        preview_cols = st.columns(min(len(preview_imgs), 3))
                        for i, img_path in enumerate(preview_imgs):
                            preview_cols[i % len(preview_cols)].image(
                                str(img_path), caption=img_path.name, use_container_width=True,
                            )

            except Exception as e:
                st.error(f"预测失败: {e}")
                import traceback
                st.code(traceback.format_exc())


def _render_train_feedback(ds):
    st.subheader("训练后自动回灌")
    st.caption("用训练好的模型对数据集进行预标注，然后自动评估，为难例挖掘做准备")

    best_pt = ds.info.get("last_best_pt", "")
    if not best_pt:
        st.info("暂无训练权重。请先在「训练配置与启动」中完成训练。")
        return

    st.info(f"当前最优权重: `{best_pt}`")

    col1, col2 = st.columns(2)
    with col1:
        pred_field = st.text_input("预测结果字段", value="predictions", key="fb_pred_field")
        gt_field = st.text_input("真值字段", value="ground_truth", key="fb_gt_field")
    with col2:
        task = st.selectbox("任务类型", ["detect", "pose", "obb"], key="fb_task")
        conf = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05, key="fb_conf")
        eval_key = st.text_input("评估键", value="eval", key="fb_eval_key")

    if st.button("🔄 执行回灌（预标注 + 评估）", key="btn_feedback", type="primary"):
        if not Path(best_pt).exists():
            st.error(f"权重文件不存在: {best_pt}")
            return

        with st.spinner("正在执行预标注与评估..."):
            try:
                result = trainer.run_post_training_eval(
                    ds, best_pt, pred_field=pred_field, gt_field=gt_field,
                    task=task, conf=conf, eval_key=eval_key,
                )
                st.success("✅ 回灌完成")
                mc1, mc2 = st.columns(2)
                mc1.metric("预标注样本数", result["predicted"])
                if result["mAP"] is not None:
                    mc2.metric("mAP", f"{result['mAP']:.4f}")
                else:
                    mc2.metric("mAP", "N/A")
                st.info(
                    "💡 回灌完成后可前往「高级功能 → 难例挖掘」查找模型表现差的样本，"
                    "然后推送到 CVAT 进行重标注。"
                )
            except Exception as e:
                st.error(f"回灌失败: {e}")

    # 展示训练可视化
    st.markdown("---")
    st.markdown("### 📊 训练结果可视化")
    history = trainer.get_training_history(ds)
    if history:
        latest = history[-1]
        result_dir = latest.get("result_dir")
        if result_dir and Path(result_dir).is_dir():
            vis_files = {
                "混淆矩阵": Path(result_dir) / "confusion_matrix.png",
                "混淆矩阵 (归一化)": Path(result_dir) / "confusion_matrix_normalized.png",
                "PR 曲线": Path(result_dir) / "PR_curve.png",
                "F1 曲线": Path(result_dir) / "F1_curve.png",
                "训练结果": Path(result_dir) / "results.png",
            }
            found_any = False
            for name, fpath in vis_files.items():
                if fpath.exists():
                    found_any = True
                    with st.expander(name, expanded=name == "训练结果"):
                        st.image(str(fpath), use_container_width=True)
            if not found_any:
                st.caption("训练结果目录中未找到可视化图表")
        else:
            st.caption("训练结果目录不存在或未记录")
    else:
        st.caption("暂无训练历史")
