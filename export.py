"""General-purpose YOLO model export script.

Usage:
    python export.py --weights runs/obb/train2/weights/best.pt --task obb --format onnx
    python export.py --weights yolo11n.pt --format engine --half --device 0
    python export.py --weights yolo11n.pt --format onnx --dynamic --simplify --opset 16
"""

import argparse
import sys

from ultralytics import YOLO

SUPPORTED_FORMATS = [
    "torchscript",
    "onnx",
    "openvino",
    "engine",
    "coreml",
    "saved_model",
    "pb",
    "tflite",
    "edgetpu",
    "tfjs",
    "paddle",
    "mnn",
    "ncnn",
    "imx",
    "rknn",
    "executorch",
    "axelera",
]

SUPPORTED_TASKS = ["detect", "segment", "classify", "pose", "obb"]


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to various formats")

    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--task", type=str, default=None, choices=SUPPORTED_TASKS, help="Model task type (auto-detected if omitted)")
    parser.add_argument("--format", type=str, default="onnx", choices=SUPPORTED_FORMATS, help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, nargs="+", default=[640], help="Inference image size, e.g. --imgsz 640 or --imgsz 640 480")
    parser.add_argument("--device", type=str, default="0", help="Export device, e.g. 0 or cpu")
    parser.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic axes for ONNX/TensorRT")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX model")
    parser.add_argument("--nms", action="store_true", help="Add NMS post-processing to model")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT max workspace size (GB)")
    parser.add_argument("--fraction", type=float, default=1.0, help="Dataset fraction for INT8 calibration (default: 1.0)")
    parser.add_argument("--keras", action="store_true", help="Use Keras for TF SavedModel export")
    parser.add_argument("--optimize", action="store_true", help="Optimize TorchScript for mobile")

    return parser.parse_args()


def main():
    args = parse_args()

    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz

    model = YOLO(args.weights, task=args.task)

    export_kwargs = dict(
        format=args.format,
        imgsz=imgsz,
        batch=args.batch,
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        simplify=args.simplify,
        nms=args.nms,
        fraction=args.fraction,
        keras=args.keras,
        optimize=args.optimize,
    )

    if args.device is not None:
        export_kwargs["device"] = int(args.device) if args.device.isdigit() else args.device
    if args.opset is not None:
        export_kwargs["opset"] = args.opset
    if args.workspace is not None:
        export_kwargs["workspace"] = args.workspace

    path = model.export(**export_kwargs)
    print(f"\nExport complete: {path}")


if __name__ == "__main__":
    main()
