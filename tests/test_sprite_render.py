"""Tests for the shared sprite-rendering primitives.

``corner_anchors`` and ``trim`` are pure; the scene helpers are exercised
against the fake render pipeline from ``openrct2_object_common.testing`` and a
stub builder, so no renderer/Embree scene is needed.
"""

import numpy as np
from openrct2_object_common.sprite_render import (
    IDENTITY3,
    add_split_ghost,
    center_in_box,
    corner_anchors,
    fit_in_box,
    render_corner_anchored_rotations,
    render_corner_anchored_view,
    render_scene_view,
    render_scene_views,
    trim,
)
from openrct2_object_common.testing import FakeContext
from openrct2_x7_renderer.constants import TILE_SIZE
from openrct2_x7_renderer.mesh import Material, Mesh
from openrct2_x7_renderer.types import IndexedImage


def _mesh(*, is_ghost: bool = False) -> Mesh:
    """A one-triangle mesh with a single (optionally ghost) material."""
    return Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[Material(is_ghost=is_ghost)],
    )


def _empty_mesh() -> Mesh:
    """A mesh with no faces (the blank-sprite path)."""
    return Mesh(
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        uvs=np.zeros((0, 2), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.uint32),
        face_materials=np.zeros((0,), dtype=np.uint32),
        materials=[],
    )


class _StubBuilder:
    """Records the (matrix, translation, mask) of each add_model call."""

    def __init__(self):
        self.calls = []

    def add_model(self, mesh, matrix, translation, mask):
        self.calls.append((matrix, translation, mask))


def test_corner_anchors_default_is_half_tile_pattern():
    h = TILE_SIZE / 2.0
    assert corner_anchors() == [(h, h), (-h, h), (-h, -h), (h, -h)]


def test_corner_anchors_scales_with_render_scale():
    assert corner_anchors(8.0) == [(4.0, 4.0), (-4.0, 4.0), (-4.0, -4.0), (4.0, -4.0)]


def test_add_split_ghost_adds_with_identity_matrix():
    builder = _StubBuilder()
    translation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    add_split_ghost(builder, _mesh(), translation)
    assert builder.calls
    for matrix, trans, _mask in builder.calls:
        np.testing.assert_array_equal(matrix, IDENTITY3)
        np.testing.assert_array_equal(trans, translation)


def test_render_scene_view_runs_full_lifecycle():
    ctx = FakeContext()
    out = render_scene_view(ctx, _mesh(), np.zeros(3), np.eye(3))
    assert isinstance(out, IndexedImage)
    assert "begin" in ctx.events
    assert "finalize" in ctx.events
    assert ctx.events[-1] == "end"


def test_render_scene_views_shares_one_scene():
    ctx = FakeContext()
    views = [np.eye(3), np.eye(3), np.eye(3)]
    out = render_scene_views(ctx, _mesh(), np.zeros(3), views)
    assert len(out) == len(views)
    # One scene serves every view: a single begin/finalize/end cycle.
    assert ctx.events.count("begin") == 1
    assert ctx.events.count("finalize") == 1
    assert ctx.events.count("end") == 1


def test_render_corner_anchored_view_runs_full_lifecycle():
    ctx = FakeContext()
    out = render_corner_anchored_view(ctx, _mesh(), 1, units_per_tile=8.0)
    assert isinstance(out, IndexedImage)
    assert ctx.events.count("begin") == 1
    assert ctx.events.count("finalize") == 1
    assert ctx.events[-1] == "end"


def test_render_corner_anchored_view_empty_mesh_is_blank():
    ctx = FakeContext()
    out = render_corner_anchored_view(ctx, _empty_mesh(), 0)
    assert (out.width, out.height) == (1, 1)
    # An empty mesh short-circuits before any scene is opened.
    assert ctx.events == []


def test_render_corner_anchored_rotations_per_view_scenes():
    ctx = FakeContext()
    out = render_corner_anchored_rotations(ctx, _mesh())
    assert len(out) == 4
    # The default corners differ per direction, so each view gets its own scene.
    assert ctx.events.count("begin") == 4
    assert ctx.events.count("finalize") == 4


def test_render_corner_anchored_rotations_shared_anchor_one_scene():
    ctx = FakeContext()
    corners = [(0.0, 0.0)] * 4
    out = render_corner_anchored_rotations(ctx, _mesh(), corners=corners)
    assert len(out) == 4
    # Every direction shares one anchor, so a single scene serves all 4 views.
    assert ctx.events.count("begin") == 1
    assert ctx.events.count("finalize") == 1


def test_render_corner_anchored_rotations_empty_mesh_is_four_blanks():
    ctx = FakeContext()
    out = render_corner_anchored_rotations(ctx, _empty_mesh())
    assert len(out) == 4
    assert all((img.width, img.height) == (1, 1) for img in out)
    assert ctx.events == []


def _image(pixels: np.ndarray, *, x_offset: int = 0, y_offset: int = 0) -> IndexedImage:
    pixels = np.asarray(pixels, dtype=np.uint8)
    return IndexedImage(
        width=pixels.shape[1],
        height=pixels.shape[0],
        x_offset=x_offset,
        y_offset=y_offset,
        pixels=pixels,
    )


def test_trim_crops_transparent_border_and_keeps_anchor():
    pixels = np.zeros((4, 4), dtype=np.uint8)
    pixels[1, 2] = 7  # single opaque pixel
    out = trim(_image(pixels, x_offset=10, y_offset=20))
    assert (out.width, out.height) == (1, 1)
    assert (out.x_offset, out.y_offset) == (12, 21)
    assert out.pixels[0, 0] == 7


def test_trim_fully_transparent_is_blank():
    out = trim(_image(np.zeros((3, 3), dtype=np.uint8)))
    assert (out.width, out.height) == (1, 1)
    assert not out.pixels.any()


def test_trim_fully_opaque_is_unchanged():
    pixels = np.arange(1, 7, dtype=np.uint8).reshape(2, 3)
    out = trim(_image(pixels, x_offset=5, y_offset=6))
    assert (out.width, out.height) == (3, 2)
    assert (out.x_offset, out.y_offset) == (5, 6)
    np.testing.assert_array_equal(out.pixels, pixels)


def test_center_in_box_centres_even_dimensions():
    img = IndexedImage.blank(10, 4)
    out = center_in_box(img, 100, 100)
    assert (out.x_offset, out.y_offset) == ((100 - 10) // 2, (100 - 4) // 2)
    assert (out.width, out.height) == (10, 4)


def test_center_in_box_subtracts_draw_origin():
    img = IndexedImage.blank(6, 6)
    out = center_in_box(img, 66, 80, draw=(11, 16))
    assert out.x_offset == (66 - 6) // 2 - 11
    assert out.y_offset == (80 - 6) // 2 - 16


def test_center_in_box_shares_pixels():
    img = IndexedImage.blank(2, 2)
    assert center_in_box(img, 10, 10).pixels is img.pixels


def test_fit_in_box_keeps_small_content_and_centres():
    img = IndexedImage(20, 10, 0, 0, np.ones((10, 20), dtype=np.uint8))
    out = fit_in_box(img, 100, 100)
    assert (out.width, out.height) == (20, 10)
    assert (out.x_offset, out.y_offset) == ((100 - 20) // 2, (100 - 10) // 2)
    np.testing.assert_array_equal(out.pixels, img.pixels)


def test_fit_in_box_trims_before_fitting():
    pixels = np.zeros((40, 40), dtype=np.uint8)
    pixels[5:15, 8:28] = 7  # a 20x10 opaque block in a transparent field
    out = fit_in_box(IndexedImage(40, 40, 0, 0, pixels), 100, 100)
    assert (out.width, out.height) == (20, 10)
    assert (out.x_offset, out.y_offset) == ((100 - 20) // 2, (100 - 10) // 2)


def test_fit_in_box_shrinks_oversized_content_preserving_aspect():
    img = IndexedImage(200, 100, 0, 0, np.ones((100, 200), dtype=np.uint8))
    out = fit_in_box(img, 50, 50)
    # Wider than tall: width hits the box, height follows the 2:1 aspect.
    assert (out.width, out.height) == (50, 25)
    assert (out.x_offset, out.y_offset) == (0, (50 - 25) // 2)


def test_fit_in_box_shrink_preserves_palette_indices():
    # Nearest-neighbour keeps real indices (e.g. remap colours) -- it never
    # averages two indices into a third, meaningless one.
    rng = np.random.default_rng(0)
    pixels = rng.integers(200, 214, size=(200, 200), dtype=np.uint8)
    out = fit_in_box(IndexedImage(200, 200, 0, 0, pixels), 40, 40)
    assert set(np.unique(out.pixels)).issubset(set(np.unique(pixels)))
