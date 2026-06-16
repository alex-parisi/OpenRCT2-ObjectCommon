"""
Shared bookends for a generator's ``export_*_test`` fast-iteration renders.

Every generator's test exporter opens the same way -- ensure the output
directory exists -- and most close by tiling the rendered views into one
``preview_combined.png`` so the test sprite shows every view at a glance. These
own that shared frame so each test exporter only writes its kind-specific
per-view PNGs in between.
"""

from pathlib import Path

from openrct2_x7_renderer.image import write_png
from openrct2_x7_renderer.types import IndexedImage

from .parkobj import combine_indexed_images

__all__ = ["open_test_dir", "write_combined_preview"]


def open_test_dir(test_dir: Path | str) -> Path:
    """Resolve ``test_dir`` to a ``Path`` and create it (parents included)."""
    path = Path(test_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_combined_preview(
    images: list[IndexedImage], test_dir: Path | str, *, columns: int = 2
) -> None:
    """Tile ``images`` into one ``preview_combined.png`` under ``test_dir``.

    Cells are aligned by draw offset (see :func:`combine_indexed_images`) so the
    views line up; pass the subset to show (e.g. the four rotation sprites)."""
    combined = combine_indexed_images(images, columns=columns)
    write_png(combined, Path(test_dir) / "preview_combined.png")
