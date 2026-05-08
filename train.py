import albumentations as A

from ultralytics import YOLO

model = YOLO("yolo11s-pose.pt")

# AGV 托盘识别场景的 Albumentations 自定义增强
# 模拟仓库中常见的图像质量问题：运动模糊、传感器噪声、低光照对比度差
custom_transforms = [
    A.OneOf(
        [
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ],
        p=0.3,
    ),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), shadow_dimension=5, p=0.2),
]

results = model.train(
    data="/home/cotek/datasets/temp/data.yaml",
    epochs=150,
    imgsz=640,
    # --- 色彩空间增强 ---
    hsv_h=0.015,  # 色调微调，适应仓库不同色温光源（LED/荧光灯/自然光）
    hsv_s=0.7,    # 饱和度变化，模拟光照强度对颜色的影响
    hsv_v=0.5,    # 亮度变化（略高于默认0.4），仓库明暗差异大
    # --- 几何变换（保守策略：托盘占画面比例大，必须避免边界裁剪导致标注偏移）---
    degrees=3.0,        # 3度微小旋转，避免大物体角落溢出
    translate=0.0,     # 0.05仅5%平移，防止托盘被推到边界外
    scale=0.0,         # 0.15缩放因子 [0.85, 1.15]，最多放大15%，大托盘不会溢出
    shear=2.0,          # 2.0微小剪切
    perspective=0.0003, # 极小透视变换
    fliplr=0.5,       # 左右翻转，托盘从左右两侧都可能出现
    flipud=0.0,       # 禁用上下翻转，托盘不会倒置
    # --- 组合增强 ---
    mosaic=1.0,       # 马赛克增强，提高多目标/小目标检测能力
    close_mosaic=15,  # 最后15个epoch关闭mosaic，精细调优
    mixup=0.0,        # 禁用mixup，工业场景不适用
    # --- Albumentations 自定义增强 ---
    augmentations=custom_transforms,
)