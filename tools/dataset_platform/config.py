"""
全局配置管理 - 支持持久化。

配置优先级（高 → 低）：
  1. 环境变量: CVAT_URL, CVAT_USERNAME, CVAT_PASSWORD, CVAT_ORGANIZATION
  2. 本地配置文件: ~/.dataset_platform/config.yaml
  3. 代码中的默认值
"""

import os
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path.home() / ".dataset_platform"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"


@dataclass
class CVATConfig:
    url: str = "http://localhost:8080"
    username: str = ""
    password: str = ""
    organization: str = ""


@dataclass
class PlatformConfig:
    cvat: CVATConfig = field(default_factory=CVATConfig)
    fiftyone_port: int = 5151
    default_export_dir: str = str(Path.home() / "dataset_exports")
    image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    max_samples_per_page: int = 50

    VISIBILITY_MAP: dict = field(default_factory=lambda: {
        0: ("not_labeled", 0.0),
        1: ("occluded", 1.0),
        2: ("visible", 2.0),
    })


def _load_config() -> PlatformConfig:
    """按优先级加载配置：环境变量 > 本地 YAML > 默认值。"""
    cfg = PlatformConfig()

    # 从本地 YAML 加载
    if _CONFIG_FILE.exists():
        try:
            import yaml
            with open(_CONFIG_FILE) as f:
                data = yaml.safe_load(f) or {}
            cvat_data = data.get("cvat", {})
            if cvat_data:
                for k in ("url", "username", "password", "organization"):
                    if k in cvat_data:
                        setattr(cfg.cvat, k, cvat_data[k])
            if "fiftyone_port" in data:
                cfg.fiftyone_port = int(data["fiftyone_port"])
            if "default_export_dir" in data:
                cfg.default_export_dir = data["default_export_dir"]
            logger.info("已加载本地配置: %s", _CONFIG_FILE)
        except Exception as e:
            logger.warning("读取配置文件失败: %s", e)

    # 环境变量覆盖（最高优先级）
    if os.environ.get("CVAT_URL"):
        cfg.cvat.url = os.environ["CVAT_URL"]
    if os.environ.get("CVAT_USERNAME"):
        cfg.cvat.username = os.environ["CVAT_USERNAME"]
    if os.environ.get("CVAT_PASSWORD"):
        cfg.cvat.password = os.environ["CVAT_PASSWORD"]
    if os.environ.get("CVAT_ORGANIZATION"):
        cfg.cvat.organization = os.environ["CVAT_ORGANIZATION"]

    return cfg


def save_config(cfg: PlatformConfig) -> None:
    """将配置持久化到本地 YAML 文件。"""
    try:
        import yaml
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "cvat": {
                "url": cfg.cvat.url,
                "username": cfg.cvat.username,
                "password": cfg.cvat.password,
                "organization": cfg.cvat.organization,
            },
            "fiftyone_port": cfg.fiftyone_port,
            "default_export_dir": cfg.default_export_dir,
        }
        with open(_CONFIG_FILE, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)
        logger.info("配置已保存: %s", _CONFIG_FILE)
    except Exception as e:
        logger.warning("保存配置失败: %s", e)


CONFIG = _load_config()
