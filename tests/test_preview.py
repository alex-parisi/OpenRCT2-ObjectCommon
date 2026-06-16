"""Tests for the shared ``export_*_test`` bookend helpers."""

from openrct2_object_common.preview import open_test_dir, write_combined_preview
from openrct2_x7_renderer.types import IndexedImage


def test_open_test_dir_creates_nested_directory(tmp_path):
    target = tmp_path / "a" / "b"
    out = open_test_dir(target)
    assert out == target
    assert out.is_dir()


def test_open_test_dir_is_idempotent(tmp_path):
    open_test_dir(tmp_path / "x")
    # A second call on the existing directory must not raise.
    assert open_test_dir(tmp_path / "x").is_dir()


def test_write_combined_preview_writes_png(tmp_path):
    images = [IndexedImage.blank(2, 2), IndexedImage.blank(2, 2)]
    write_combined_preview(images, tmp_path)
    assert (tmp_path / "preview_combined.png").is_file()
