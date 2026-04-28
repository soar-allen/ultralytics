"""将当前文件夹中所有图片的上半部分涂成黑色（原地覆盖）。"""

from pathlib import Path

import cv2

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_SCRIPT_DIR = Path(__file__).resolve().parent


def blackout_upper_half():
    images = [f for f in _SCRIPT_DIR.iterdir()
              if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]

    if not images:
        print("未找到图片文件")
        return

    print(f"共找到 {len(images)} 张图片，开始处理...")

    for i, img_path in enumerate(sorted(images), 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  跳过（无法读取）: {img_path.name}")
            continue

        h = img.shape[0]
        img[: h // 2, :] = 0
        cv2.imwrite(str(img_path), img)

        if i % 100 == 0 or i == len(images):
            print(f"  已处理 {i}/{len(images)}")

    print("全部完成")


if __name__ == "__main__":
    blackout_upper_half()
