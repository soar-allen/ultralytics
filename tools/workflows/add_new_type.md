# 添加新托盘/货架类型工作流

本指南介绍如何为新的托盘或货架类型扩展数据集。

## 1. 数据收集

### 样本数量要求

| 阶段 | 图片数量 | 说明 |
|------|----------|------|
| 初始标注 | 10-20 张 | 人工精标，用于首次训练 |
| 扩展标注 | 30-50 张 | 模型预标 + 人工微调 |
| 完整数据集 | 100+ 张 | 持续迭代 |

### 拍摄要求

```
相机视角: 平视略高 (叉车驾驶员视角)
距离范围: 1-5 米
```

#### 多样性覆盖

- [ ] 多角度: 正面、左右 30°、左右 45°
- [ ] 多距离: 近景 (1-2m)、中景 (2-4m)、远景 (4m+)
- [ ] 多光照: 日光、荧光灯、混合光照
- [ ] 多遮挡: 无遮挡、部分遮挡、边缘遮挡

#### 图片规格

| 项目 | 要求 |
|------|------|
| 分辨率 | ≥ 1280×720 |
| 格式 | JPG, PNG |
| 命名 | `{类型}_{场景}_{序号}.jpg` |

示例: `pallet_tian_warehouse_001.jpg`

---

## 2. 初始标注 (人工)

### 使用 X-AnyLabeling

```bash
# 1. 安装 X-AnyLabeling
pip install x-anylabeling

# 2. 启动
x-anylabeling
```

### 标注步骤

1. File → Open Dir → 选择图片目录
2. 按 `R` 切换到旋转框模式
3. 绘制入叉面 OBB:
   - 框住入叉面整个平面
   - 调整角度使长边与水平对齐
4. 保存为 YOLO-OBB 格式

### 标注规则检查清单

- [ ] 只框入叉面，不包含侧面/顶面
- [ ] 遮挡时按平面补全
- [ ] 长边沿水平方向
- [ ] 框内可见叉孔/凹槽特征

---

## 3. 首次模型训练

```python
from ultralytics import YOLO

# 加载预训练 OBB 模型
model = YOLO("yolo11n-obb.pt")

# 训练
results = model.train(
    data="datasets/fork_entry/fork_entry.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    name="fork_entry_v1"
)
```

---

## 4. 自动预标注

使用训练好的模型预标注新图片:

```bash
python tools/fork_entry_annotator.py \
    --images datasets/fork_entry/images/raw/new_batch \
    --output datasets/fork_entry/labels/pre_annotations \
    --obb-model runs/obb/fork_entry_v1/weights/best.pt
```

### 输出目录结构

```
pre_annotations/
├── auto_accept/       # 高置信度，可直接使用
├── manual_review/     # 需人工检查
└── reject_samples/    # 低质量抽样
```

---

## 5. 人工审核

### 导出到 X-AnyLabeling

```bash
python tools/export_to_xanylabeling.py \
    --images datasets/fork_entry/images/raw/new_batch \
    --labels datasets/fork_entry/labels/pre_annotations/manual_review \
    --output datasets/fork_entry/export/review_batch
```

### 审核流程

1. 在 X-AnyLabeling 中打开导出目录
2. 逐张检查并修正:
   - 调整框边界
   - 修正旋转角度
   - 删除错误标注
3. 保存修正后的标注

---

## 6. 合并数据集

```bash
# 合并 auto_accept 和审核通过的标注
python tools/export_to_xanylabeling.py \
    --merge \
    --inputs \
        datasets/fork_entry/labels/pre_annotations/auto_accept \
        datasets/fork_entry/labels/reviewed \
    --output datasets/fork_entry/labels/final
```

---

## 7. 增量训练

```python
from ultralytics import YOLO

# 加载上一版模型
model = YOLO("runs/obb/fork_entry_v1/weights/best.pt")

# 继续训练
results = model.train(
    data="datasets/fork_entry/fork_entry.yaml",
    epochs=50,
    imgsz=640,
    resume=False,  # 从头开始但使用预训练权重
    name="fork_entry_v2"
)
```

---

## 8. 版本管理

### 模型版本记录

| 版本 | 日期 | 类别 | 样本数 | mAP50 |
|------|------|------|--------|-------|
| v1.0 | 2026-01-18 | fork_entry_surface | 50 | - |
| v1.1 | - | + pallet_tian | - | - |

### 检查点保存

```bash
# 保存重要版本
cp runs/obb/fork_entry_v1/weights/best.pt models/fork_entry_v1.0.pt
```

---

## 常见问题

### Q: 如何添加新类别?

1. 更新 `annotation_rules.yaml`:
   ```yaml
   classes:
     0: fork_entry_surface
     1: pallet_tian_entry   # 新类别
   ```

2. 更新 `fork_entry.yaml`:
   ```yaml
   names:
     0: fork_entry_surface
     1: pallet_tian_entry
   ```

3. 为新类别标注 10-20 张样本

4. 重新训练模型

### Q: 标注质量不佳怎么办?

1. 运行质检:
   ```bash
   python tools/quality_checker.py \
       --images ./images \
       --labels ./labels \
       --output qc_report.json
   ```

2. 查看报告，修正低分标注

3. 重新训练
