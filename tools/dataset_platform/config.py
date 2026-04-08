"""全局配置管理"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CVATConfig:
    url: str = "http://192.168.1.89:8080"
    username: str = "zlm"
    password: str = "Cotek@123"
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


CONFIG = PlatformConfig()
