"""Tests for .parkobj assembly and the images.dat helper."""

import json
import os
import zipfile

import numpy as np
import pytest
from openrct2_object_common.parkobj import (
    assemble_parkobj,
    combine_indexed_images,
    parkobj_filename,
    write_images_dat_lgx,
)
from openrct2_x7_renderer.types import IndexedImage


def test_parkobj_filename_uses_id_and_sanitises_slashes():
    assert parkobj_filename("alex.coaster/wooden", default="x") == "alex.coaster_wooden.parkobj"


def test_parkobj_filename_falls_back_to_default_when_empty():
    assert parkobj_filename("", default="vehicle") == "vehicle.parkobj"


def _render_two_pixels(work_dir):
    """Stand-in render: writes a dummy images.dat and returns its $LGX ref."""
    return write_images_dat_lgx(
        [IndexedImage.blank(1, 1), IndexedImage.blank(2, 2)], work_dir
    )


def test_write_images_dat_lgx_writes_blob_and_ref(tmp_path):
    refs = write_images_dat_lgx([IndexedImage.blank(1, 1)], tmp_path)
    assert refs == ["$LGX:images.dat[0..0]"]
    assert (tmp_path / "images.dat").exists()


def test_write_images_dat_lgx_rejects_empty_list(tmp_path):
    # An empty list would otherwise emit a malformed "$LGX:images.dat[0..-1]".
    with pytest.raises(ValueError, match="no sprites"):
        write_images_dat_lgx([], tmp_path)
    assert not (tmp_path / "images.dat").exists()


def test_assemble_renders_and_zips(tmp_path):
    work = tmp_path / "object"
    parkobj = tmp_path / "out" / "thing.parkobj"
    obj_json = {"id": "rct2.thing", "objectType": "scenery_small"}

    assemble_parkobj(obj_json, parkobj, work, _render_two_pixels)

    written = json.loads((work / "object.json").read_text())
    assert written["images"] == ["$LGX:images.dat[0..1]"]
    assert parkobj.exists()
    with zipfile.ZipFile(parkobj) as zf:
        assert set(zf.namelist()) == {"object.json", "images.dat"}


def test_assemble_cleans_up_temp_on_zip_failure(tmp_path):
    work = tmp_path / "object"
    out = tmp_path / "out"
    parkobj = out / "thing.parkobj"

    def _render_without_images_dat(work_dir):
        # Return a valid images ref but never write images.dat, so zipping it
        # raises and the temp-file cleanup path runs.
        return ["$LGX:images.dat[0..0]"]

    with pytest.raises(FileNotFoundError):
        assemble_parkobj({"id": "x"}, parkobj, work, _render_without_images_dat)

    assert not parkobj.exists()
    # The temp .parkobj must have been unlinked, not left behind.
    assert list(out.glob("*.parkobj")) == []


def test_assemble_parkobj_respects_umask(tmp_path):
    # The temp file behind the atomic replace is mkstemp'd (0o600); the final
    # .parkobj must carry normal umask-derived permissions instead.
    work = tmp_path / "object"
    parkobj = tmp_path / "thing.parkobj"
    old_umask = os.umask(0o022)
    try:
        assemble_parkobj({"id": "rct2.thing"}, parkobj, work, _render_two_pixels)
    finally:
        os.umask(old_umask)
    assert parkobj.stat().st_mode & 0o777 == 0o644


def test_skip_render_reuses_previous_images(tmp_path):
    work = tmp_path / "object"
    work.mkdir()
    (work / "object.json").write_text(json.dumps({"images": ["$LGX:images.dat[0..4]"]}))
    (work / "images.dat").write_bytes(b"stale-but-reused")
    parkobj = tmp_path / "thing.parkobj"

    def _must_not_render(_work_dir):
        raise AssertionError("render must not be called when skip_render=True")

    assemble_parkobj(
        {"id": "rct2.thing"}, parkobj, work, _must_not_render, skip_render=True
    )

    written = json.loads((work / "object.json").read_text())
    assert written["images"] == ["$LGX:images.dat[0..4]"]
    assert parkobj.exists()


def test_skip_render_rejects_missing_images_array(tmp_path):
    work = tmp_path / "object"
    work.mkdir()
    (work / "object.json").write_text(json.dumps({"id": "rct2.thing"}))  # no "images"
    with pytest.raises(RuntimeError, match="images"):
        assemble_parkobj({}, tmp_path / "x.parkobj", work, _render_two_pixels, skip_render=True)


def test_combine_indexed_images_tiles_into_grid():
    imgs = [
        IndexedImage(1, 1, 0, 0, np.full((1, 1), v, dtype=np.uint8))
        for v in (10, 20, 30, 40)
    ]
    out = combine_indexed_images(imgs, columns=2)
    assert (out.width, out.height) == (2, 2)
    assert out.pixels.tolist() == [[10, 20], [30, 40]]


def test_combine_indexed_images_empty_returns_blank():
    out = combine_indexed_images([])
    assert (out.width, out.height) == (1, 1)


def test_combine_indexed_images_single_no_blank_cell():
    one = IndexedImage(1, 1, 0, 0, np.full((1, 1), 7, dtype=np.uint8))
    out = combine_indexed_images([one], columns=2)
    assert (out.width, out.height) == (1, 1)
