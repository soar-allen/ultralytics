# CV 视觉数据集统一管理与标注平台

基于 FiftyOne + Streamlit 构建的一站式 CV 数据集管理平台，集成 CVAT 标注同步、多格式导入导出、自动预标注、训练管理、标注质量检查等功能。

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
├── ui.py              # Streamlit 前端主入口（薄编排层）
├── ui/                # UI 子模块（按 Tab 拆分）
│   ├── components.py  # 公共组件（路径浏览器、状态管理）
│   ├── sidebar.py     # 侧边栏（数据集管理、CVAT/配置持久化）
│   ├── hub.py         # Data Hub 仪表盘
│   ├── ingestion.py   # 数据导入
│   ├── processing.py  # 数据处理与清洗
│   ├── prediction.py  # 自动预标注
│   ├── cvat_page.py   # CVAT 双向同步
│   ├── export_page.py # 多格式导出（含训练衔接）
│   ├── training_page.py # 训练管理
│   ├── quality_page.py  # 标注质量检查
│   └── advanced_page.py # 高级功能（Brain、难例、备份）
├── data_manager.py    # FiftyOne 底层数据操作引擎
├── ingestors.py       # 多格式数据导入模块
├── processor.py       # 自动化处理与清洗
├── cvat_sync.py       # CVAT 双向同步模块
├── exporter.py        # 多格式导出模块
├── trainer.py         # YOLO 训练管理（后台训练、历史、回灌）
├── advanced.py        # 难例挖掘与完整数据集备份
├── config.py          # 全局配置（支持 YAML 持久化 + 环境变量）
├── requirements.txt   # 依赖列表
└── README.md
```

## 功能模块

### 1. Data Hub — 数据集管理中心

- 侧边栏数据集列表切换、新建、重命名、删除
- **数据集统计仪表盘**：样本总数、标注实例数、Tags 种类数等概览指标
- **标注类别统计**：每个标签字段的类别分布图表
- **Tags 统计**：样本标签分布可视化
- **标签筛选与查看**：按 Tags 筛选样本，支持在 FiftyOne 或 CVAT 中查看
- **CVAT Job 状态刷新**：一键同步 CVAT 最新 Job 状态到样本 Tags
- **标签管理**：样本 Tags 批量增删、标注类别重命名、标注统计
- FiftyOne App 跳转到新浏览器标签页查看完整数据集

### 2. 数据导入 (Ingestion)

| 格式               | 说明                                                       |
| ---------------- | -------------------------------------------------------- |
| A. 纯图片           | 导入未标注图片目录                                                |
| B. Thoro COCO    | 解析 coco_files/ + image_groups/ 结构                        |
| C. Roboflow COCO | 标准 Roboflow COCO 导出格式 (train/valid/test splits)          |
| D. Roboflow YOLO | 标准 Roboflow YOLO 导出格式 (含 data.yaml)                      |
| E. CVAT 1.1 XML  | CVAT for images 1.1 标准格式                                  |
| F. 从备份导入        | 从完整备份导入数据（含图像、标注、Tags）                                  |

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
- **标注进度看板**: Job 完成率、进度条、按状态/阶段统计
- **CVAT Task 深链**: 直接跳转到具体 Task 页面
- **废弃图片管理**: 标注员在 CVAT 中标记的废弃图片处理

### 6. 数据导出 (Exporting)

| 格式                  | 输出                                  |
| ------------------- | ----------------------------------- |
| YOLO Detect         | class_id cx cy w h                  |
| YOLO Pose           | class_id cx cy w h kx1 ky1 kv1 ...  |
| YOLO 混合训练（pose） | Pose 目标正常写关键点，检测类写 bbox + 全 0 关键点 |
| YOLO Pose (四边形转关键点) | 4点polygon → tl/tr/br/bl 关键点 + 边界可见性 |
| YOLO OBB            | class_id x1 y1 x2 y2 x3 y3 x4 y4    |

- 支持按类别/Tags 过滤，仅导出有标签图像，自动生成 data.yaml
- 支持按比例自动划分 train/valid/test
- **导出-训练衔接**: 导出完成后自动记录 data.yaml 路径，可一键跳转训练管理

### 7. 训练管理 (Training)

全新的一站式训练管理页面：
- **训练配置面板**: 选择 data.yaml（自动检测导出结果）、模型权重、epochs/imgsz/batch 等超参数
- **一键后台训练**: 调用 YOLO.train() 在后台运行，不阻塞 UI
- **训练历史**: 自动记录每次训练的参数、数据集版本、最终指标（mAP 等）到 `dataset.info`
- **最优权重管理**: 训练完成自动识别 best.pt，一键设为预标注模型
- **训练结果可视化**: 混淆矩阵、PR 曲线、F1 曲线等训练图表
- **训练后回灌**: 自动用 best.pt 预标注 + 评估，为难例挖掘做准备
- **增量训练回放**: 新增类别全量训练，同时从旧类别抽样回放，降低旧类别遗忘风险
- **类别顺序锁定**: 导出时可读取上一版 `data.yaml`，保持旧 class id 不变并将新类别追加到末尾

### 8. 标注质量检查 (Quality)

专用的标注质量分析页面：
- **类别平衡分析**: 类别分布可视化 + 不平衡检测 + 过采样/欠采样建议
- **面积分析**: 标注 bbox 面积分布直方图，检测异常小/大标注
- **空标注检测**: 发现有标注字段但 detections 为空列表的样本
- **标注一致性**: 对比不同 anno_key 的类别分布差异，检测标注员偏差

### 9. 高级功能

- **难例挖掘**: 对比 ground_truth 与 predictions，找出高错误率样本，一键推送到 CVAT 重标注
- **FiftyOne Brain**: 
  - Uniqueness (唯一性): 发现冗余/相似样本
  - Hardness (难度): 基于模型不确定性排序，主动学习核心
  - Representativeness (代表性): 评估数据集覆盖度
- **完整数据集备份**: 复制图像文件 + 导出元数据，支持从备份恢复和跨实例导入

## 配置管理

配置支持三层优先级（高 → 低）：

1. **环境变量**: `CVAT_URL`, `CVAT_USERNAME`, `CVAT_PASSWORD`, `CVAT_ORGANIZATION`
2. **本地配置文件**: `~/.dataset_platform/config.yaml`（侧边栏修改后点击「保存配置」自动写入）
3. **代码默认值**

```bash
# 使用环境变量配置
export CVAT_URL="http://192.168.1.89:8080"
export CVAT_USERNAME="admin"
export CVAT_PASSWORD="your_password"
```

## 完整工作流

```
图像采集 → 数据导入 → 自动预标注 → 推送CVAT → 人工标注 → 拉取标注
    → 质量检查 → 数据导出 → 一键训练 → 训练后回灌
    → 难例挖掘 → 推送CVAT重标注 → ... (闭环迭代)
```

## 多批次数据集管理最佳实践

### 核心概念：Tags vs Fields

| 概念               | 用途                | 示例                                        |
| ---------------- | ----------------- | ----------------------------------------- |
| **Tags（样本标签）**   | 标记样本的元数据：批次、来源、状态 | `batch_01`, `factory_A`, `duplicate`      |
| **Fields（标签字段）** | 存储不同类型/来源的标注数据    | `ground_truth`（人工标注）, `predictions`（模型预测） |

### 推荐方案：Tags 标记批次 + 统一字段 + CVAT 按标签筛选推送

```
1. 导入 batch_01 图片 → 批次标签 "batch_01" → 标注写入 ground_truth
2. 推送 batch_01 到 CVAT → anno_key "round1_batch01"
3. 导入 batch_02 → 批次标签 "batch_02" → 同样写入 ground_truth
4. 推送 batch_02 → anno_key "round1_batch02"
5. 分别拉取各批次标注
```

### 常用 Tag 命名规范

| Tag                        | 含义                       |
| -------------------------- | ------------------------ |
| `batch_01`, `batch_02` ... | 数据采集批次                   |
| `factory_A`, `line_3`      | 数据来源（工厂、产线等）             |
| `duplicate`                | 去重标记（平台自动添加）             |
| `corrupt`, `abnormal_size` | 异常图片标记（平台自动添加）           |
| `hard_sample`              | 难例标记（难例挖掘自动添加）           |
| `train`, `valid`, `test`   | 数据集划分                    |
| `reviewed`, `approved`     | 人工审核状态                   |
