"""Convert YOLO ONNX models to RKNN models."""

from __future__ import annotations

import sys
from pathlib import Path

DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = '../model/yolo11.rknn'
DEFAULT_QUANT = True
SUPPORTED_DTYPES = ("i8", "u8", "fp")

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] [dataset_txt(optional)]".format(sys.argv[0]))
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b, rv1109, rv1126, rk1808]")
        print("       dtype choose from [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    dtype = "i8" if DEFAULT_QUANT else "fp"
    if len(sys.argv) > 3:
        dtype = sys.argv[3]
        if dtype not in SUPPORTED_DTYPES:
            print("ERROR: Invalid model type: {}".format(dtype))
            exit(1)

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    if len(sys.argv) > 5:
        dataset_path = sys.argv[5]
    else:
        dataset_path = DATASET_PATH

    return model_path, platform, dtype, output_path, dataset_path


def convert_onnx_to_rknn(
    onnx_model_path: str | Path,
    platform: str,
    dtype: str = "i8",
    output_rknn_path: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> str:
    """Convert an ONNX model to RKNN and return the output RKNN path."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Invalid RKNN dtype: {dtype}")

    onnx_model_path = Path(onnx_model_path)
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
    if onnx_model_path.suffix.lower() != ".onnx":
        raise ValueError(f"RKNN conversion requires an ONNX model: {onnx_model_path}")

    output_rknn_path = Path(output_rknn_path or DEFAULT_RKNN_PATH)
    output_rknn_path.parent.mkdir(parents=True, exist_ok=True)

    do_quant = dtype in ("i8", "u8")
    if do_quant:
        if not dataset_path:
            raise ValueError("Quantized RKNN export requires a calibration dataset txt")
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Calibration dataset txt not found: {dataset_path}")
        dataset_arg = str(dataset_path)
    else:
        dataset_arg = None

    try:
        from rknn.api import RKNN
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("缺少 rknn-toolkit2，请先安装提供 rknn.api 的 RKNN Toolkit2") from exc

    rknn = RKNN(verbose=False)
    try:
        print('--> Config model')
        rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
        print('done')

        print('--> Loading model')
        ret = rknn.load_onnx(model=str(onnx_model_path))
        if ret != 0:
            raise RuntimeError(f"Load ONNX model failed: {ret}")
        print('done')

        print('--> Building model')
        if do_quant:
            ret = rknn.build(do_quantization=True, dataset=dataset_arg)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"Build RKNN model failed: {ret}")
        print('done')

        print('--> Export rknn model')
        ret = rknn.export_rknn(str(output_rknn_path))
        if ret != 0:
            raise RuntimeError(f"Export RKNN model failed: {ret}")
        print('done')
    finally:
        rknn.release()

    return str(output_rknn_path)

if __name__ == '__main__':
    model_path, platform, dtype, output_path, dataset_path = parse_arg()
    convert_onnx_to_rknn(model_path, platform, dtype, output_path, dataset_path)
