"""
Assemble an object's ``.parkobj`` into a generator's output directory.

Every generator's public ``export_*`` entrypoint does the same plumbing: pick
the delivered ``<output>/<id>.parkobj`` path and the ``object`` working dir,
then render-and-zip the kind-specific object.json via :func:`assemble_parkobj`.
``output_paths`` owns the path convention; ``export_object`` owns binding the
generator's ``render_sprites`` to one object + render context.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from openrct2_x7_renderer.ray_trace import Context

from .parkobj import assemble_parkobj

__all__ = [
    "ExportTo",
    "ProgressFn",
    "RenderSprites",
    "export_object",
    "export_to_directory",
    "output_paths",
]

T = TypeVar("T")

# Called as progress(done, total) while sprites render.
ProgressFn = Callable[[int, int], None]

# Render an object's sprites into a work dir (writing images.dat) and return the
# object.json "images" list. ``export_object`` binds the leading object/context.
RenderSprites = Callable[[T, Context, Path, "ProgressFn | None"], list[str]]

# A generator's explicit-path exporter, e.g. ``export_small_scenery_to``: render
# ``obj`` and write the ``.parkobj`` to a caller-chosen parkobj/work path pair.
# Loosely typed like the dispatch table's exporters since each kind binds its own
# object type. ``export_to_directory`` adapts one to the default-directory layout.
ExportTo = Callable[..., None]


def output_paths(
    output_directory: Path | str, obj_id: str, work_dir: str = "object"
) -> tuple[Path, Path]:
    """The default delivered paths: ``(<output_directory>/<obj_id>.parkobj, <work_dir>)``."""
    return Path(output_directory) / f"{obj_id}.parkobj", Path(work_dir)


def export_object(
    obj: T,
    context: Context,
    obj_json: dict[str, Any],
    render_sprites: "RenderSprites[T]",
    parkobj_path: Path | str,
    work_dir: Path | str,
    *,
    skip_render: bool = False,
    progress: ProgressFn | None = None,
) -> None:
    """Render ``obj``'s sprites (or reuse a previous render) and zip the ``.parkobj``.

    Binds ``render_sprites`` to ``obj`` / ``context`` / ``progress`` and hands
    the result to :func:`assemble_parkobj`; callers differ only in ``obj_json``
    and ``render_sprites``.
    """
    assemble_parkobj(
        obj_json,
        Path(parkobj_path),
        Path(work_dir),
        lambda wd: render_sprites(obj, context, wd, progress),
        skip_render=skip_render,
    )


def export_to_directory(
    export_to: ExportTo,
    obj: object,
    context: Context,
    output_directory: Path | str,
    obj_id: str,
    *,
    skip_render: bool = False,
) -> None:
    """Export ``obj`` to the default ``<output_directory>/<obj_id>.parkobj`` layout.

    Resolves the delivered parkobj + working-dir paths via :func:`output_paths`
    and hands off to the generator's explicit-path ``export_to`` (e.g.
    ``export_small_scenery_to``), so every generator's directory-level
    ``export_*`` shares one path convention instead of re-deriving it.
    """
    parkobj_path, work_dir = output_paths(output_directory, obj_id)
    export_to(obj, context, parkobj_path, work_dir, skip_render=skip_render)
