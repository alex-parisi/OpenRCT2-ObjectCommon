"""
Write ``images.dat``, the object.json, and assemble the ``.parkobj`` ZIP.

Every object kind follows the same packaging path: render sprites into a single
``images.dat`` blob (or reuse a previous render), write the object.json that
references it via ``$LGX:``, and zip the two together. The per-kind work is just
*what* sprites to render and *what* metadata to emit; this module owns the rest.
"""

import contextlib
import json
import logging
import math
import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from openrct2_x7_renderer.images_dat import write_images_dat
from openrct2_x7_renderer.types import IndexedImage

__all__ = ["RenderFn", "assemble_parkobj", "combine_indexed_images", "write_images_dat_lgx"]

log = logging.getLogger(__name__)

# Render the object's sprites into ``work_dir`` (writing ``images.dat``) and
# return the object.json "images" list (the ``$LGX:`` references).
RenderFn = Callable[[Path], list[str]]


def combine_indexed_images(images: list[IndexedImage], columns: int = 2) -> IndexedImage:
    """Tile IndexedImages into a single grid image, aligned by draw offset.

    Each cell spans the union of every image's draw-offset bounding box, so a
    shared sprite anchor lands at the same spot in every cell and the rotated
    views line up. Cells fill left-to-right, top-to-bottom over a transparent
    (palette index 0) background; ``columns`` is capped at the image count so a
    single image doesn't leave a blank cell. Used to show all four rotated
    preview directions in one image.
    """
    if not images:
        return IndexedImage.blank(1, 1)
    columns = max(1, min(columns, len(images)))
    left = min(im.x_offset for im in images)
    top = min(im.y_offset for im in images)
    cell_w = max(im.x_offset + im.width for im in images) - left
    cell_h = max(im.y_offset + im.height for im in images) - top
    rows = math.ceil(len(images) / columns)
    canvas = np.zeros((rows * cell_h, columns * cell_w), dtype=np.uint8)
    for idx, im in enumerate(images):
        row, col = divmod(idx, columns)
        x = col * cell_w + (im.x_offset - left)
        y = row * cell_h + (im.y_offset - top)
        canvas[y : y + im.height, x : x + im.width] = im.pixels
    return IndexedImage(
        width=canvas.shape[1],
        height=canvas.shape[0],
        x_offset=0,
        y_offset=0,
        pixels=canvas,
    )


def write_images_dat_lgx(
    images: list[IndexedImage], work_dir: Path, *, note: str = ""
) -> list[str]:
    """Write ``images`` to ``work_dir/images.dat`` and return the object.json
    "images" value referencing it.

    ``note`` is appended to the log line (e.g. ``" for 4 tiles"``). Returns the
    single-element ``["$LGX:images.dat[0..N-1]"]`` list OpenRCT2 expects.
    """
    if not images:
        raise ValueError("Cannot write images.dat with no sprites")
    out_path = work_dir / "images.dat"
    write_images_dat(images, out_path)
    log.info(
        "wrote %s (%d sprites%s, %.1f KB)",
        out_path,
        len(images),
        note,
        out_path.stat().st_size / 1024,
    )
    return [f"$LGX:images.dat[0..{len(images) - 1}]"]


def assemble_parkobj(
    obj_json: dict[str, Any],
    parkobj_path: Path,
    work_dir: Path,
    render: RenderFn,
    *,
    skip_render: bool = False,
) -> None:
    """Render (or reuse) sprites, write the object.json, and zip the ``.parkobj``.

    Shared by every object kind; callers differ only in ``obj_json`` (the built
    metadata) and ``render`` (which renders the sprites, writes ``images.dat``,
    and returns the object.json "images" list -- typically via
    :func:`write_images_dat_lgx`).

    With ``skip_render`` the "images" list is read back from a previous run's
    object.json in ``work_dir`` and ``render`` is not called; otherwise the
    stale object.json / images.dat are removed first so a failed render can't
    leave a half-written pair behind.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    if skip_render:
        prev = json.loads((work_dir / "object.json").read_text())
        images_json = prev.get("images")
        if not isinstance(images_json, list):
            raise RuntimeError('Property "images" is not an array')
    else:
        for p in (work_dir / "object.json", work_dir / "images.dat"):
            p.unlink(missing_ok=True)
        images_json = render(work_dir)

    obj_json["images"] = images_json
    (work_dir / "object.json").write_text(json.dumps(obj_json, indent=4))

    parkobj_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parkobj", dir=parkobj_path.parent)
    os.close(tmp_fd)
    # mkstemp creates the file 0o600; widen to the umask default so the
    # delivered .parkobj has normal file permissions after os.replace.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp_path, 0o666 & ~umask)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(work_dir / "object.json", "object.json")
            zf.write(work_dir / "images.dat", "images.dat")
        os.replace(tmp_path, parkobj_path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
