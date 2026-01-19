from ultralytics import YOLO
import os

# 1. 加载你训练好的OBB模型
model = YOLO('runs/obb/train2/weights/best.pt')  # 请替换为你的模型文件路径[citation:7]

# 2. 指定包含所有图片的文件夹路径
source_folder = 'datasets/fork_entry/images/raw'  # 请替换为你的图片文件夹路径

# 3. 进行批量预测
#    结果会自动保存到 `runs/obb/predict` 目录下[citation:7]
results = model.predict(
    source=source_folder,  # 源路径：可以是单张图片、视频、文件夹或URL[citation:4]
    task='obb',            # 明确指定任务为旋转框检测[citation:4]
    save=True,             # 保存带标注的图片
    save_txt=False,        # 为True时同时保存OBB标签文件（.txt格式）
    conf=0.25,             # 置信度阈值
    imgsz=640              # 推理图片尺寸
)

print(f"处理完成！结果已保存。")