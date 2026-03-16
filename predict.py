"""
YOLO 通用预测脚本
支持多种任务类型: detect, segment, pose, obb, classify
支持多种输入源: 单张图片、文件夹、视频、摄像头、URL

使用示例:
    # 检测任务 - 单张图片
    python predict.py --model yolo11n.pt --source image.jpg --task detect

    # OBB任务 - 文件夹批量处理
    python predict.py --model runs/obb/train/weights/best.pt --source datasets/images/ --task obb

    # 分割任务 - 保存标签
    python predict.py --model yolo11n-seg.pt --source video.mp4 --task segment --save-txt

    # 姿态估计
    python predict.py --model yolo11n-pose.pt --source 0 --task pose  # 摄像头

    # 分类任务
    python predict.py --model yolo11n-cls.pt --source image.jpg --task classify
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


# 支持的任务类型
SUPPORTED_TASKS = ['detect', 'segment', 'pose', 'obb', 'classify']

# 支持的图片扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

# 支持的视频扩展名
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLO通用预测脚本 - 支持detect/segment/pose/obb/classify任务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py --model yolo11n.pt --source image.jpg
  python predict.py --model runs/obb/train/weights/best.pt --source ./images/ --task obb
  python predict.py --model yolo11n-seg.pt --source video.mp4 --task segment --save-txt
  python predict.py --model yolo11n-pose.pt --source 0 --task pose
        """
    )

    # 必需参数
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型路径，例如: yolo11n.pt, runs/detect/train/weights/best.pt'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='输入源: 图片路径、文件夹路径、视频路径、摄像头ID(0,1,...)、或URL'
    )

    # 任务类型
    parser.add_argument(
        '--task', '-t',
        type=str,
        choices=SUPPORTED_TASKS,
        default=None,
        help='任务类型: detect, segment, pose, obb, classify (默认自动检测)'
    )

    # 推理参数
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.4,
        help='置信度阈值 (默认: 0.25)'
    )
    parser.add_argument(
        '--iou',
        type=float,
        default=0.2,
        help='NMS IoU阈值 (默认: 0.7)'
    )
    parser.add_argument(
        '--imgsz', '--img-size',
        type=int,
        default=640,
        help='推理图片尺寸 (默认: 640)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='推理设备: cpu, 0, 0,1,2,3 (默认自动选择)'
    )
    parser.add_argument(
        '--classes',
        type=int,
        nargs='+',
        default=None,
        help='只检测指定类别ID，例如: --classes 0 2 3'
    )
    parser.add_argument(
        '--max-det',
        type=int,
        default=300,
        help='每张图片最大检测数量 (默认: 300)'
    )

    # 输出控制
    parser.add_argument(
        '--save',
        action='store_true',
        default=True,
        help='保存带标注的图片/视频 (默认: True)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存带标注的图片/视频'
    )
    parser.add_argument(
        '--save-txt',
        action='store_true',
        help='保存检测结果为txt文件'
    )
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='在txt文件中保存置信度'
    )
    parser.add_argument(
        '--save-crop',
        action='store_true',
        help='保存裁剪的检测目标'
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='保存结果的项目目录 (默认: runs/{task})'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='predict',
        help='保存结果的子目录名 (默认: predict)'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='允许覆盖已存在的输出目录'
    )

    # 显示控制
    parser.add_argument(
        '--show',
        action='store_true',
        help='实时显示检测结果窗口'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        default=True,
        help='显示标签 (默认: True)'
    )
    parser.add_argument(
        '--show-conf',
        action='store_true',
        default=True,
        help='显示置信度 (默认: True)'
    )
    parser.add_argument(
        '--line-width',
        type=int,
        default=None,
        help='边框线宽 (默认自动)'
    )

    # 其他
    parser.add_argument(
        '--half',
        action='store_true',
        help='使用FP16半精度推理'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    return parser.parse_args()


def validate_source(source: str) -> str:
    """验证输入源是否有效"""
    # 检查是否为摄像头ID
    if source.isdigit():
        return source

    # 检查是否为URL
    if source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        return source

    # 检查是否为有效路径
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"输入源不存在: {source}")

    return source


def get_source_info(source: str) -> dict:
    """获取输入源的详细信息"""
    info = {
        'type': 'unknown',
        'path': source,
        'count': 0
    }

    # 摄像头
    if source.isdigit():
        info['type'] = 'webcam'
        info['count'] = 1
        return info

    # URL
    if source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://')):
        info['type'] = 'url'
        info['count'] = 1
        return info

    path = Path(source)

    # 文件夹
    if path.is_dir():
        info['type'] = 'directory'
        image_files = [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
        video_files = [f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]
        info['images'] = len(image_files)
        info['videos'] = len(video_files)
        info['count'] = len(image_files) + len(video_files)
        return info

    # 单个文件
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            info['type'] = 'image'
        elif suffix in VIDEO_EXTENSIONS:
            info['type'] = 'video'
        info['count'] = 1
        return info

    return info


def main():
    """主函数"""
    args = parse_args()

    # 验证输入源
    try:
        source = validate_source(args.source)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    # 获取输入源信息
    source_info = get_source_info(source)

    # 加载模型
    print(f"正在加载模型: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"错误: 无法加载模型 - {e}")
        return 1

    # 确定任务类型
    task = args.task
    if task is None:
        # 尝试从模型自动推断任务类型
        task = getattr(model, 'task', 'detect')
        print(f"自动检测任务类型: {task}")
    
    # 设置项目目录
    project = args.project if args.project else f'runs/{task}'

    # 打印输入信息
    print("-" * 50)
    print(f"模型: {args.model}")
    print(f"任务: {task}")
    print(f"输入源: {source}")
    print(f"输入类型: {source_info['type']}")
    if source_info['type'] == 'directory':
        print(f"  - 图片数量: {source_info.get('images', 0)}")
        print(f"  - 视频数量: {source_info.get('videos', 0)}")
    print(f"置信度阈值: {args.conf}")
    print(f"图片尺寸: {args.imgsz}")
    print(f"保存目录: {project}/{args.name}")
    print("-" * 50)

    # 处理save参数
    save = not args.no_save and args.save

    # 执行预测
    try:
        results = model.predict(
            source=source,
            task=task,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            classes=args.classes,
            max_det=args.max_det,
            save=save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            project=project,
            name=args.name,
            exist_ok=args.exist_ok,
            show=args.show,
            show_labels=args.show_labels,
            show_conf=args.show_conf,
            line_width=args.line_width,
            half=args.half,
            verbose=args.verbose,
        )

        # 统计结果
        total_detections = 0
        processed_count = 0
        
        for result in results:
            processed_count += 1
            if result.boxes is not None:
                total_detections += len(result.boxes)
            elif result.obb is not None:
                total_detections += len(result.obb)

        print("-" * 50)
        print(f"处理完成!")
        print(f"  - 处理文件数: {processed_count}")
        print(f"  - 总检测数量: {total_detections}")
        if save:
            print(f"  - 结果已保存到: {project}/{args.name}")

    except Exception as e:
        print(f"错误: 预测过程出错 - {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())