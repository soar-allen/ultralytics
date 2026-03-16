# 托盘标注工具集

托盘检测与入叉面定位的标注、训练数据生成工具。

## 整体流程

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       CVAT + SAM 标注流程                                    │
│                                                                              │
│  ① CVAT 标注前表面 polygon（4 点）                                            │
│       ↓ 导出 CVAT XML                                                        │
│  ② sam_bbox_generator.py：SAM 自动生成整个托盘 bbox                           │
│       ↓ 生成配对 CVAT XML（bbox + polygon, group_id 关联）                    │
│  ③ 导入 CVAT 二次确认：检查 bbox 是否正确框选整个托盘                          │
│       ↓ 导出确认后的 CVAT XML                                                │
│  ④ cvat_paired_to_yolo_pose.py：转换为 YOLO-Pose 训练数据                    │
│       ↓                                                                      │
│  ⑤ YOLO-Pose 训练                                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 数据清洗

### dedup_images.py

使用 FiftyOne 去除数据集中的重复和相似图片，支持两种检测模式：

- **精确去重**：通过 MD5 文件哈希检测完全相同的图片
- **近似去重**：通过深度学习图像嵌入（默认 MobileNet-V2）计算余弦距离，检测视觉上高度相似的图片

可选传入 `--cvat_xml` 参数指定 CVAT 标注 XML 文件，在移动/删除图片时会同步移除 XML 中对应的 `<image>` 节点并重新索引，确保处理后的数据集能正常导入回 CVAT。不指定时仅处理图片文件。

```bash
# 仅检测报告，不做修改
python tools/pallet_labeling/dedup_images.py \
  --images_dir /path/to/images

# 检测并移动重复图片到指定目录
python tools/pallet_labeling/dedup_images.py \
  --images_dir /path/to/images \
  --action move --dup_dir /path/to/duplicates

# 仅做近似去重，调整阈值（越小越严格）
python tools/pallet_labeling/dedup_images.py \
  --images_dir /path/to/images \
  --mode near --thresh 0.05

# 删除重复 + 同步清理 CVAT XML 标注
python tools/pallet_labeling/dedup_images.py \
  --images_dir /path/to/images \
  --action delete --cvat_xml annotations.xml

# 删除重复 + 启动 FiftyOne 可视化
python tools/pallet_labeling/dedup_images.py \
  --images_dir /path/to/images \
  --action delete --visualize
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--images_dir` | 图片目录（递归扫描） | 必填 |
| `--mode` | `exact` / `near` / `both` | `both` |
| `--thresh` | 近似去重余弦距离阈值（0~1，越小越严格） | 0.03 |
| `--model` | 嵌入模型（FiftyOne model zoo） | `mobilenet-v2-imagenet-torch` |
| `--action` | `report`（仅报告）/ `move` / `delete` | `report` |
| `--dup_dir` | move 模式的目标目录 | - |
| `--cvat_xml` | CVAT 标注 XML 文件路径（可选，指定后同步移除已删除图片的标注） | - |
| `--visualize` | 启动 FiftyOne App 可视化结果 | false |

**依赖安装：**

```bash
pip install fiftyone torch torchvision
```

### 1.1（可选）：过滤非前表面标注

如果标注中混入了侧面标注（接近正方形的 polygon），可以使用 `filter_polygon.py` 过滤：

```bash
# 先看报告，了解数据分布
python tools/pallet_labeling/filter_polygon.py \
  --cvat_xml annotations.xml \
  --action report

# 确认后执行过滤（默认保留宽高比在 [2.5, 7.5] 范围内的标注）
python tools/pallet_labeling/filter_polygon.py \
  --cvat_xml annotations.xml \
  --output filtered.xml

# 或标记模式：不删除，重命名被拒绝的标签，在 CVAT 中审查
python tools/pallet_labeling/filter_polygon.py \
  --cvat_xml annotations.xml \
  --output marked.xml \
  --action mark
```

**核心参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--target_ratio` | 目标宽高比（前表面≈5:1） | 5.0 |
| `--tolerance` | 容忍度（±百分比） | 0.5（即范围 [2.5, 7.5]） |
| `--min_ratio` | 直接指定最小比例（覆盖上面两项） | - |
| `--max_ratio` | 直接指定最大比例 | - |
| `--min_area` | 最小面积 px² | 0 |
| `--action` | `filter` / `mark` / `report` | `filter` |


## CVAT + SAM 标注

### 第一步：CVAT 标注前表面

在 CVAT 中创建任务，标签名为 `pallet`，标注类型为 **polygon**。
对每个托盘标注前表面的 4 个角点，然后导出为 **CVAT for images 1.1** XML 格式。



### 第二步：SAM 自动生成 bbox

```bash
python tools/pallet_labeling/sam_bbox_generator.py \
  --cvat_xml annotations.xml \
  --images_dir /path/to/images \
  --output output_annotations.xml
```

脚本会：
1. 解析 XML 中的前表面 polygon
2. 对每个 polygon 用 SAM 自动分割出整个托盘
3. 从分割 mask 中提取 bounding box
4. 输出新的 CVAT XML，每个托盘包含一组配对标注：
   - `pallet_body` (rectangle): 整个托盘的 bbox
   - `pallet` (polygon): 原始的前表面 4 点

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cvat_xml` | 输入 CVAT XML 文件 | 必填 |
| `--images_dir` | 图片根目录 | 必填 |
| `--output` | 输出 CVAT XML 路径 | 必填 |
| `--sam_model` | SAM 模型权重路径 | `sam3.pt` |
| `--prompt_mode` | SAM prompt 模式 | `centroid` |
| `--expand_ratio` | box 模式扩展比例 | 0.5 |
| `--bbox_label` | bbox 标签名 | `pallet_body` |
| `--polygon_label` | polygon 标签名 | `pallet` |
| `--fallback_expand` | SAM 失败时回退扩展像素 | 50.0 |
| `--mask_strategy` | mask 选择策略 | `smallest_covering` |
| `--max_height_ratio` | bbox 最大高度倍数（0=不限） | 0.0 |
| `--negative_above` | 添加负向提示点排除货物 | false |
| `--text_prompt` | text 模式的文本提示词 | `pallet` |
| `--text_conf` | text 模式的检测置信度阈值 | 0.25 |

**Prompt 模式：**
- `text`（推荐）：使用 SAM3 文本语义 prompt，以 `--text_prompt` 指定目标（如 "pallet"），有语义理解能力，能区分托盘和货物
- `centroid`：使用 polygon 质心作为 SAM 点 prompt
- `box`：使用 polygon 外扩框作为 SAM box prompt
- `points`：使用全部 4 个顶点作为 SAM 点 prompt

**避免 bbox 包含货物的策略：**
- `--mask_strategy smallest_covering`（默认）：从 SAM 返回的多个 mask 中选择包含 polygon 且面积最小的，而非最大的
- `--negative_above`：在 polygon 上方添加负向提示点，引导 SAM 排除货物区域
- `--max_height_ratio 2.5`：限制 bbox 高度不超过 polygon 高度的 2.5 倍，硬性约束防止过度扩展

### 第三步：导入 CVAT 二次确认

1. 在 CVAT 中创建新任务（使用相同图片集）
2. 添加两个标签：
   - `pallet_body`（shape type: rectangle）
   - `pallet`（shape type: polygon）
3. 上传标注：Actions → Upload annotations → 选择 "CVAT 1.1" 格式 → 上传生成的 XML
4. 检查并调整 SAM 生成的 bbox 是否正确覆盖整个托盘

### 第四步：转换为 YOLO-Pose 训练数据

从 CVAT 导出确认后的 XML，然后运行：

```bash
python tools/pallet_labeling/cvat_paired_to_yolo_pose.py \
  --cvat_xml verified_annotations.xml \
  --images_dir /path/to/images \
  --out_dir /path/to/yolo_pose_dataset \
  --split 0.9,0.1,0.0
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cvat_xml` | 确认后的 CVAT XML 文件 | 必填 |
| `--images_dir` | 图片根目录 | 可选 |
| `--out_dir` | 输出目录 | 必填 |
| `--bbox_label` | bbox 标签名 | `pallet_body` |
| `--polygon_label` | polygon 标签名 | `pallet` |
| `--class_name` | YOLO 类别名 | `pallet` |
| `--split` | train,val,test 比例 | `0.9,0.1,0.0` |
| `--seed` | 随机划分种子 | 42 |
| `--copy_images` | 复制图片（默认软链接） | false |
| `--skip_images` | 仅生成标签不处理图片 | false |

**输出结构：**

```
out_dir/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/   # YOLO-Pose txt
│   └── val/
└── data.yaml
```

**YOLO-Pose 标签格式** (`kpt_shape=[4,3]`)：

```
cls xc yc w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4
```

- `xc yc w h`：整个托盘 bbox（来自 pallet rectangle）
- `x1..x4, y1..y4`：前表面 4 个关键点（tl, tr, br, bl）
- `v`：可见性标志（2 = 可见）
- 所有坐标归一化到 0~1

## 格式转换

### convert2_yolo_pose.py

将多种标注格式转换为 YOLO-Pose 训练数据，支持：
- COCO detection JSON
- YOLO detection txt
- Roboflow YOLO dataset
- Pascal VOC XML

```bash
python tools/pallet_labeling/convert2_yolo_pose.py \
  --coco_json /path/to/annotations.json \
  --coco_images_dir /path/to/images \
  --out_dir /path/to/output \
  --class_name pallet
```

## 文件结构

```
tools/pallet_labeling/
├── sam_bbox_generator.py        # CVAT polygon + SAM → 配对 CVAT XML (bbox + polygon)
├── cvat_paired_to_yolo_pose.py  # 配对 CVAT XML → YOLO-Pose 训练数据
├── filter_polygon.py            # 基于宽高比过滤非前表面 polygon 标注
├── dedup_images.py              # FiftyOne 图片去重（精确 + 近似）
├── convert2_yolo_pose.py        # 多格式 → YOLO-Pose 转换器（COCO/YOLO/Roboflow/VOC）
├── index_images.py              # 图片目录扫描工具
├── annotations.xml              # CVAT 标注示例
└── README.md
```
