from ultralytics.models.sam import SAM3SemanticPredictor
from pathlib import Path

# 初始化predictor
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="sam3.pt",
    half=True,
    save=True,
    exist_ok=True,  # 允许覆盖现有目录
    name="predict",  # 固定保存目录名称
)

predictor = SAM3SemanticPredictor(overrides=overrides)

# 设置图片文件夹路径
image_folder = "/home/cotek/datasets/jpeg/"
text_prompts = ["blue pallet"]  # 可以修改为多个prompt

# 获取文件夹中所有图片
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_paths = [str(p) for p in Path(image_folder).iterdir() 
               if p.suffix.lower() in image_extensions]

# 批量处理所有图片
for img_path in image_paths:
    print(f"处理图片: {img_path}")
    
    # 设置当前图片
    predictor.set_image(img_path)
    
    # 进行识别
    results = predictor(text=text_prompts)
    
    # 可选：保存结果到特定文件
    # 默认会保存在 runs/segment/predict/ 目录下
    # 文件名会基于原图片名生成