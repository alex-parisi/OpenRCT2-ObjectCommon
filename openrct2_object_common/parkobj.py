"""
Write ``images.dat``, the object.json, and assemble the ``.parkobj`` ZIP.

Every object kind follows the same packaging path: render sprites into a single
``images.dat`` blob (or reuse a previous render), write the object.json that
references it via ``$LGX:``, and zip the two together. The per-kind work is just
*what* sprites to render and *what* metadata to emit; this module owns the rest.
"""

import json
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openrct2_x7_renderer.images_dat import write_images_dat
from openrct2_x7_renderer.types import IndexedImage

__all__ = ["RenderFn", "assemble_parkobj", "write_images_dat_lgx"]

log = logging.getLogger(__name__)

# Render the object's sprites into ``work_dir`` (writing ``images.dat``) and
# return the object.json "images" list (the ``$LGX:`` references).
RenderFn = Callable[[Path], list[str]]


def write_images_dat_lgx(
    images: list[IndexedImage], work_dir: Path, *, note: str = ""
) -> list[str]:
    """Write ``images`` to ``work_dir/images.dat`` and return the object.json
    "images" value referencing it.

    ``note`` is appended to the log line (e.g. ``" for 4 tiles"``). Returns the
    single-element ``["$LGX:images.dat[0..N-1]"]`` list OpenRCT2 expects.
    """
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
    with zipfile.ZipFile(parkobj_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(work_dir / "object.json", "object.json")
        zf.write(work_dir / "images.dat", "images.dat")
