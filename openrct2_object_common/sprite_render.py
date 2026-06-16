"""
Shared sprite-rendering primitives for object generators.

Every generator renders a combined ``Mesh`` under OpenRCT2's cardinal views and
anchors the result at a tile's per-direction reference corner. These helpers own
the parts that don't vary by object kind: opening a one-model scene (splitting
ghost faces so the tracer sees through them), rendering one or more views from
it, the half-tile corner-anchor pattern, and trimming transparent borders. The
per-kind work -- which mesh, which views, where to anchor -- stays in each
generator's ``sprite_renderer``.
"""

import numpy as np
from numpy.typing import NDArray
from openrct2_x7_renderer.constants import TILE_SIZE
from openrct2_x7_renderer.geometry import split_mesh_by_ghost
from openrct2_x7_renderer.mesh import Mesh
from openrct2_x7_renderer.palette import TRANSPARENT_INDEX
from openrct2_x7_renderer.ray_trace import VIEWS, Context, SceneBuilder
from openrct2_x7_renderer.types import IndexedImage

__all__ = [
    "IDENTITY3",
    "add_split_ghost",
    "center_in_box",
    "corner_anchors",
    "render_corner_anchored_rotations",
    "render_corner_anchored_view",
    "render_scene_view",
    "render_scene_views",
    "trim",
]

# No-rotation orientation; placement orientation is baked into the mesh, so the
# scene helpers add models with an identity matrix and a plain translation.
IDENTITY3 = np.eye(3, dtype=np.float64)


def add_split_ghost(
    scene: SceneBuilder, mesh: Mesh, translation: NDArray[np.float64]
) -> None:
    """Add ``mesh`` to ``scene`` at ``translation``, splitting ghost faces into
    their own GHOST model so the renderer traces through them (e.g. baked-in
    ghost geometry)."""
    for sub_mesh, mask in split_mesh_by_ghost(mesh):
        scene.add_model(sub_mesh, IDENTITY3, translation, mask)


def render_scene_view(
    context: Context,
    mesh: Mesh,
    translation: NDArray[np.float64],
    view: NDArray[np.float64],
) -> IndexedImage:
    """Render a single model under a single view in its own scene."""
    with context.begin_render() as scene:
        add_split_ghost(scene, mesh, translation)
        with scene.finalize() as ready:
            return ready.render_view(view)


def render_scene_views(
    context: Context,
    mesh: Mesh,
    translation: NDArray[np.float64],
    views: list[NDArray[np.float64]],
) -> list[IndexedImage]:
    """Render a single model under several views, sharing one finalized scene."""
    with context.begin_render() as scene:
        add_split_ghost(scene, mesh, translation)
        with scene.finalize() as ready:
            return [ready.render_view(v) for v in views]


def corner_anchors(units_per_tile: float = TILE_SIZE) -> list[tuple[float, float]]:
    """Per-direction half-tile corner offsets in OBJ units, scaled to the
    authored render scale.

    OpenRCT2 anchors large-scenery and stall sprites at the tile's reference
    CORNER (paint offset ``{0,0}``), not its centre like small scenery; the
    corner rotates with the view direction.
    """
    h = units_per_tile / 2.0
    return [(h, h), (-h, h), (-h, -h), (h, -h)]


def render_corner_anchored_view(
    context: Context,
    mesh: Mesh,
    direction: int,
    *,
    units_per_tile: float = TILE_SIZE,
    center: tuple[float, float] = (0.0, 0.0),
    corners: list[tuple[float, float]] | None = None,
) -> IndexedImage:
    """Render ``mesh`` under cardinal view ``direction``, anchored at the tile's
    per-direction reference corner.

    The corner (``corner_anchors(units_per_tile)[direction]``, or
    ``corners[direction]`` when supplied) is offset by ``center`` -- the tile's
    world centre, for multi-tile objects -- and moved to the screen origin. An
    empty mesh renders a 1x1 blank. Shared by every object kind that anchors
    sprites at a tile corner (large scenery, stalls).
    """
    if mesh.faces.shape[0] == 0:
        return IndexedImage.blank(1, 1)
    by_dir = corners if corners is not None else corner_anchors(units_per_tile)
    ox, oz = by_dir[direction]
    cx, cz = center
    translation = np.array([-(cx + ox), 0.0, -(cz + oz)], dtype=np.float64)
    return render_scene_view(context, mesh, translation, VIEWS[direction])


def render_corner_anchored_rotations(
    context: Context,
    mesh: Mesh,
    *,
    units_per_tile: float = TILE_SIZE,
    center: tuple[float, float] = (0.0, 0.0),
    corners: list[tuple[float, float]] | None = None,
) -> list[IndexedImage]:
    """Render the 4 cardinal rotations of ``mesh``, each anchored at the tile's
    per-direction reference corner (see :func:`render_corner_anchored_view`).

    When every direction shares one anchor (e.g. path additions, whose four
    corners are equal) a single finalized scene serves all 4 views; otherwise
    each view is rendered in its own scene. An empty mesh renders 4 blanks.
    """
    by_dir = corners if corners is not None else corner_anchors(units_per_tile)
    if mesh.faces.shape[0] == 0:
        return [IndexedImage.blank(1, 1) for _ in range(4)]
    cx, cz = center
    if len(set(by_dir)) == 1:
        ox, oz = by_dir[0]
        translation = np.array([-(cx + ox), 0.0, -(cz + oz)], dtype=np.float64)
        return render_scene_views(context, mesh, translation, [VIEWS[d] for d in range(4)])
    out: list[IndexedImage] = []
    for d in range(4):
        ox, oz = by_dir[d]
        translation = np.array([-(cx + ox), 0.0, -(cz + oz)], dtype=np.float64)
        out.append(render_scene_view(context, mesh, translation, VIEWS[d]))
    return out


def trim(img: IndexedImage, *, transparent: int = TRANSPARENT_INDEX) -> IndexedImage:
    """Crop fully transparent borders, preserving the draw anchor.

    A fully transparent image trims to a 1x1 blank. ``transparent`` is the
    palette index treated as empty (the palette's transparent slot by default).
    """
    opaque = img.pixels != transparent
    rows = np.flatnonzero(opaque.any(axis=1))
    cols = np.flatnonzero(opaque.any(axis=0))
    if rows.size == 0:
        return IndexedImage.blank(1, 1)
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return IndexedImage(
        width=c1 - c0,
        height=r1 - r0,
        x_offset=img.x_offset + c0,
        y_offset=img.y_offset + r0,
        pixels=np.ascontiguousarray(img.pixels[r0:r1, c0:c1]),
    )


def center_in_box(
    img: IndexedImage, box_w: int, box_h: int, *, draw: tuple[int, int] = (0, 0)
) -> IndexedImage:
    """Re-anchor ``img`` so its content centres in a ``box_w`` x ``box_h`` UI box.

    A menu preview sprite is blitted at the fixed ``draw`` position inside a
    window box (the ride window's 112x112 preview box, the scenery window's
    button); offsetting by the box's half-extent less the image's half-extent
    (and ``draw``) lands the content in the box centre. Only the draw offset
    changes -- the pixels are shared with ``img``.
    """
    draw_x, draw_y = draw
    return IndexedImage(
        width=img.width,
        height=img.height,
        x_offset=(box_w - img.width) // 2 - draw_x,
        y_offset=(box_h - img.height) // 2 - draw_y,
        pixels=img.pixels,
    )
