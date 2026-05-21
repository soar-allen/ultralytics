"""Central navigation metadata for the dataset platform UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PageSpec:
    """Definition for one top-level Streamlit page."""

    key: str
    label: str
    legacy_labels: tuple[str, ...]
    renderer: str


PAGES: tuple[PageSpec, ...] = (
    PageSpec(
        key="overview",
        label="📊 数据总览",
        legacy_labels=("📊 Data Hub",),
        renderer="tools.dataset_platform.ui.hub:_render_data_hub",
    ),
    PageSpec(
        key="prepare",
        label="🧰 数据管理",
        legacy_labels=("organize", "🧰 数据准备", "📥 导入与整理", "📥 数据导入", "🧹 处理清洗"),
        renderer="tools.dataset_platform.ui.workflow_pages:_render_import_and_processing",
    ),
    PageSpec(
        key="annotation",
        label="🏷️ 数据集标注",
        legacy_labels=("prediction", "cvat", "🤖 自动预标注", "🔄 CVAT 标注同步", "🔄 CVAT 同步"),
        renderer="tools.dataset_platform.ui.workflow_pages:_render_dataset_annotation",
    ),
    PageSpec(
        key="export_train",
        label="📤 模型训练与导出",
        legacy_labels=("📤 导出与训练", "📤 数据导出", "🏋️ 训练管理"),
        renderer="tools.dataset_platform.ui.workflow_pages:_render_export_and_training",
    ),
    PageSpec(
        key="quality_hard",
        label="🔬 数据集分析",
        legacy_labels=("quality", "advanced", "🔬 质量与难例", "🔬 质量检查", "🧠 高级功能", "🧠 高级工具"),
        renderer="tools.dataset_platform.ui.quality_hard_page:_render_quality_and_hard_samples",
    ),
    PageSpec(
        key="settings",
        label="⚙️ 工作区设置",
        legacy_labels=("⚙️ 设置与管理", "⚙️ CVAT 配置", "💾 数据集备份", "📦 数据集管理"),
        renderer="tools.dataset_platform.ui.settings_page:_render_settings_page",
    ),
)

DEFAULT_PAGE_KEY = PAGES[0].key
PAGE_LABELS = tuple(page.label for page in PAGES)


def page_by_key(key: str | None) -> PageSpec:
    """Return the page matching a key, with a stable default fallback."""
    resolved = resolve_page_key(key)
    for page in PAGES:
        if page.key == resolved:
            return page
    return PAGES[0]


def resolve_page_key(value: str | None) -> str:
    """Resolve current and legacy page values to the new stable page key."""
    if not value:
        return DEFAULT_PAGE_KEY
    for page in PAGES:
        if value == page.key or value == page.label or value in page.legacy_labels:
            return page.key
    return DEFAULT_PAGE_KEY


def load_renderer(page: PageSpec) -> Callable[[], None]:
    """Import the page renderer on demand."""
    module_name, func_name = page.renderer.split(":", 1)
    module = __import__(module_name, fromlist=[func_name])
    return getattr(module, func_name)
