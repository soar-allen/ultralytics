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
        key="organize",
        label="📥 导入与整理",
        legacy_labels=("📥 数据导入", "🧹 处理清洗"),
        renderer="tools.dataset_platform.ui.workflow_pages:_render_import_and_processing",
    ),
    PageSpec(
        key="prediction",
        label="🤖 自动预标注",
        legacy_labels=("🤖 自动预标注",),
        renderer="tools.dataset_platform.ui.prediction:_render_auto_predict_page",
    ),
    PageSpec(
        key="cvat",
        label="🔄 CVAT 标注同步",
        legacy_labels=("🔄 CVAT 同步",),
        renderer="tools.dataset_platform.ui.cvat_page:_render_cvat_sync",
    ),
    PageSpec(
        key="export_train",
        label="📤 导出与训练",
        legacy_labels=("📤 数据导出", "🏋️ 训练管理"),
        renderer="tools.dataset_platform.ui.workflow_pages:_render_export_and_training",
    ),
    PageSpec(
        key="quality",
        label="🔬 质量检查",
        legacy_labels=("🔬 质量检查",),
        renderer="tools.dataset_platform.ui.quality_page:_render_quality_page",
    ),
    PageSpec(
        key="advanced",
        label="🧠 高级工具",
        legacy_labels=("🧠 高级功能",),
        renderer="tools.dataset_platform.ui.advanced_page:_render_advanced",
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
