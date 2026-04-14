# CV 视觉数据集统一管理与标注平台

基于 FiftyOne + Streamlit 构建的一站式 CV 数据集管理平台，集成 CVAT 标注同步、多格式导入导出、自动预标注等功能。

## 快速启动

```bash
# 安装依赖
pip install -r tools/dataset_platform/requirements.txt

# 启动平台
streamlit run tools/dataset_platform/ui.py
```

## 项目结构

```
tools/dataset_platform/
├── ui.py              # Streamlit 前端界面（主入口）
├── data_manager.py    # FiftyOne 底层数据操作引擎
├── ingestors.py       # 多格式数据导入模块
├── processor.py       # 自动化处理与清洗
├── cvat_sync.py       # CVAT 双向同步模块
├── exporter.py        # 多格式导出模块
├── advanced.py        # 难例挖掘与完整数据集备份
├── config.py          # 全局配置
├── requirements.txt   # 依赖列表
└── README.md
```

## 功能模块

### 1. Data Hub — 数据集管理中心

- 侧边栏数据集列表切换、新建、重命名、删除
- **数据集统计仪表盘**：样本总数、标注实例数、Tags 种类数等概览指标
- **标注类别统计**：每个标签字段的类别分布图表
- **Tags 统计**：样本标签分布可视化
- **标签筛选与查看**：按 Tags 筛选样本，支持在 FiftyOne（新页面跳转）或 CVAT 中查看
- **CVAT Job 状态刷新**：一键同步 CVAT 最新 Job 状态到样本 Tags
- **标签管理**：样本 Tags 批量增删、标注类别重命名、标注统计
- **路径诊断**：图片路径健康检查与批量修复
- FiftyOne App 跳转到新浏览器标签页查看完整数据集

### 2. 数据导入 (Ingestion)


| 格式               | 说明                                                       |
| ---------------- | -------------------------------------------------------- |
| A. 纯图片           | 导入未标注图片目录                                                |
| B. Thoro COCO    | 解析 coco_files/ + image_groups/ 结构，直接映射为 FiftyOne 对象      |
| C. Roboflow COCO | 标准 Roboflow COCO 导出格式 (train/valid/test splits)          |
| D. Roboflow YOLO | 标准 Roboflow YOLO 导出格式 (含 data.yaml)                      |
| E. CVAT 1.1 XML  | CVAT for images 1.1 标准格式 (box/polygon/polyline/skeleton) |


### 3. 数据处理与清洗 (Processing)

- **无标注清理**: 一键删除无标签图像
- **异常检测**: 扫描损坏/尺寸异常图像
- **去重**: 精确哈希 + 近似嵌入 (CLIP/DINOv2) 双模式
- **多边形处理**: 多边形转四角、边界多边形检测
- **字段合并**: 将分批导入到不同字段的标注合并到统一字段

### 4. 自动预标注 (Auto Pre-labeling)

独立大页面，使用 YOLO 模型对数据集进行自动预标注：
- 支持 **detect / pose / obb** 三种任务类型
- 灵活的预标注范围：仅无标注样本、整个数据集、按 Tags 筛选
- 预标注结果可推送到 CVAT 辅助人工标注，或用于难例挖掘评估

### 5. CVAT 双向同步 (Annotation Sync)

- **映射关系**: Dataset↔Project, View→Task, 自动切分→Jobs
- **推送**: 支持 detections / polylines / keypoints / classifications
- **拉取**: 按 anno_key 版本管理标注结果
- **管理**: 查看/删除标注运行

### 6. 数据导出 (Exporting)


| 格式                  | 输出                                  |
| ------------------- | ----------------------------------- |
| YOLO Detect         | class_id cx cy w h                  |
| YOLO Pose           | class_id cx cy w h kx1 ky1 kv1 ...  |
| YOLO Pose (四边形转关键点) | 4点polygon → tl/tr/br/bl 关键点 + 边界可见性 |
| YOLO OBB            | class_id x1 y1 x2 y2 x3 y3 x4 y4    |


支持按类别过滤，仅导出有标签图像，自动生成 data.yaml。
支持按比例自动划分 train/valid/test（如 80%/10%/10%）。

**四边形转 Pose 说明**：将 Polylines 中的 4 点多边形自动排序为 tl→tr→br→bl，  
包围框从顶点外扩计算，贴近图像边界的关键点标记为 occluded (v=0)。

### 7. 高级功能

- **难例挖掘**: 对比 ground_truth 与 predictions，找出高错误率样本
- **完整数据集备份**: 复制图像文件 + 导出 FiftyOne 元数据与标注，支持从备份恢复
- 旧版快照管理（FiftyOne clone）仍可用，作为轻量级元数据备份

## 多批次数据集管理最佳实践

在实际项目中，数据往往分多批次采集和导入。以下是推荐的管理方式。

### 核心概念：Tags vs Fields


| 概念               | 用途                | 示例                                        |
| ---------------- | ----------------- | ----------------------------------------- |
| **Tags（样本标签）**   | 标记样本的元数据：批次、来源、状态 | `batch_01`, `factory_A`, `duplicate`      |
| **Fields（标签字段）** | 存储不同类型/来源的标注数据    | `ground_truth`（人工标注）, `predictions`（模型预测） |


### 推荐方案：Tags 标记批次 + 统一字段 + CVAT 按标签筛选推送

这是最简洁的工作流，适合大多数场景：

```
1. 导入 batch_01 图片
   → 数据导入 → 批次标签填写 "batch_01"（所有导入格式都支持）
   → 所有标注写入 ground_truth 字段

2. 推送 batch_01 到 CVAT
   → CVAT 同步 → 推送 → 按批次标签筛选选择 "batch_01"
   → anno_key 设为 "round1_batch01"

3. batch_01 正在 CVAT 标注期间，导入 batch_02
   → 数据导入 → 批次标签填写 "batch_02"
   → 标注同样写入 ground_truth 字段

4. 推送 batch_02 到 CVAT（不影响正在标注的 batch_01）
   → CVAT 同步 → 推送 → 按批次标签筛选选择 "batch_02"
   → anno_key 设为 "round1_batch02"

5. 分别拉取各自的标注
   → 选择 anno_key "round1_batch01" → 拉取
   → 选择 anno_key "round1_batch02" → 拉取
```

### 备选方案：分字段导入 + 合并

如果各批次的标注需要先隔离审核再合并，可以使用分字段导入：

```
1. batch_01 标注写入 ground_truth
2. batch_02 导入时 label_field 填写 "batch02_gt"
3. 推送 batch_02 到 CVAT 时选择字段 "batch02_gt"
4. 拉取 batch_02 标注回 "batch02_gt"
5. 审核无误后 → 处理清洗 → 字段合并 → 将 "batch02_gt" 合并到 "ground_truth"
```

> 字段合并功能位于「处理清洗 → 🔀 字段合并」，支持将源字段的标注追加到目标字段，
> 合并后可选择删除源字段以保持数据集整洁。

### 在 FiftyOne App 中按批次筛选可视化

**方法一：侧边栏 Tag 过滤（推荐）**

1. 在 Data Hub 启动 FiftyOne App
2. 在 App 左侧 **SAMPLE TAGS** 面板中，点击想查看的 tag（如 `batch_02`）
3. 视图会自动筛选，只显示该批次的图片

**方法二：通过 Python SDK 设置视图**

```python
import fiftyone as fo

ds = fo.load_dataset("my_dataset")
session = fo.launch_app(ds)

# 按 tag 筛选
view = ds.match_tags(["batch_02"])
session.view = view

# 组合筛选：某批次 + 有标注
from fiftyone import ViewField as F
view = ds.match_tags(["batch_02"]).match(
    F("ground_truth.detections").length() > 0
)
session.view = view
```

### 去重后如何可视化重复图片

平台的「重复图像清理」功能运行后，如果选择 **「仅打标签不删除」**，
会给每组重复图片中除保留项之外的所有样本添加 `duplicate` 标签。

**在 FiftyOne App 中查看重复图片：**

1. 在 Data Hub 启动 FiftyOne App
2. 左侧 **SAMPLE TAGS** 面板中点击 `duplicate`，即可看到所有被标记为重复的图片
3. 如需查看"非重复"图片，点击 `duplicate` 右侧的排除按钮（`-`），即可过滤掉重复项

**通过 Python SDK 查看：**

```python
ds = fo.load_dataset("my_dataset")
session = fo.launch_app(ds)

# 查看重复样本
dup_view = ds.match_tags(["duplicate"])
session.view = dup_view
print(f"重复样本数: {len(dup_view)}")

# 查看非重复样本（即去重后的干净集合）
clean_view = ds.match_tags(["duplicate"], bool=False)
session.view = clean_view
```

**去重后的典型操作：**

- **确认无误后批量删除**：回到平台 UI → 取消勾选「仅打标签不删除」→ 执行去重
- **手动审核**：在 FiftyOne App 中逐一查看 `duplicate` 样本，手动移除误判的标签
- **导出时排除**：导出数据时先过滤掉 `duplicate` 标签的样本

### 常用 Tag 命名规范


| Tag                        | 含义                       |
| -------------------------- | ------------------------ |
| `batch_01`, `batch_02` ... | 数据采集批次                   |
| `factory_A`, `line_3`      | 数据来源（工厂、产线等）             |
| `duplicate`                | 去重标记（平台自动添加）             |
| `corrupt`, `abnormal_size` | 异常图片标记（平台自动添加）           |
| `train`, `valid`, `test`   | 数据集划分（Roboflow 格式导入自动添加） |
| `reviewed`, `approved`     | 人工审核状态                   |


## CVAT 配置

在侧边栏配置 CVAT 连接信息，或修改 `config.py`:

```python
CVATConfig(
    url="http://localhost:8080",
    username="admin",
    password="admin",
)
```

