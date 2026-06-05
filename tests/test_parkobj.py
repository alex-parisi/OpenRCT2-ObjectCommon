"""Tests for .parkobj assembly and the images.dat helper."""

import json
import zipfile

import pytest
from openrct2_object_common.parkobj import assemble_parkobj, write_images_dat_lgx
from openrct2_x7_renderer.types import IndexedImage


def _render_two_pixels(work_dir):
    """Stand-in render: writes a dummy images.dat and returns its $LGX ref."""
    return write_images_dat_lgx(
        [IndexedImage.blank(1, 1), IndexedImage.blank(2, 2)], work_dir
    )


def test_write_images_dat_lgx_writes_blob_and_ref(tmp_path):
    refs = write_images_dat_lgx([IndexedImage.blank(1, 1)], tmp_path)
    assert refs == ["$LGX:images.dat[0..0]"]
    assert (tmp_path / "images.dat").exists()


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
