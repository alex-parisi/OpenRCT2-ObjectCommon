"""
Place a Model's meshes into a render scene.

Both generators turn a ``Model`` (a list of placements, each carrying one
``MeshFrame`` per animation frame) into Embree scene geometry the same way:
convert the frame's degree orientation to a rotation matrix and add the
referenced mesh at its position. The only difference is how out-of-range frame
indices are handled (vehicles index exactly; scenery clamps to a placement's
last pose), which is the ``clamp_frame`` flag.
"""

import math

import numpy as np
from openrct2_x7_renderer.geometry import rotate_x, rotate_y, rotate_z
from openrct2_x7_renderer.mesh import Mesh
from openrct2_x7_renderer.ray_trace import SceneBuilder
from openrct2_x7_renderer.types import Model

__all__ = ["add_model_to_scene", "orientation_to_matrix"]


def orientation_to_matrix(orientation_deg: np.ndarray) -> np.ndarray:
    """A MeshFrame orientation ``(angle_y, angle_z, angle_x)`` in degrees as a
    ``(3, 3)`` rotation matrix, applied as ``rotate_y @ rotate_z @ rotate_x``."""
    rx, ry, rz = orientation_deg * (math.pi / 180.0)
    return rotate_y(rx) @ rotate_z(ry) @ rotate_x(rz)


def add_model_to_scene(
    builder: SceneBuilder,
    meshes: list[Mesh],
    model: Model,
    *,
    frame: int = 0,
    mask: int = 0,
    clamp_frame: bool = False,
) -> None:
    """Add ``model``'s placed meshes to an open ``SceneBuilder`` at pose ``frame``.

    Placements whose ``mesh_index`` is ``-1`` (an empty slot) are skipped. With
    ``clamp_frame=True`` a placement with fewer frames falls back to its last
    frame (scenery's animated poses); with ``False`` the frame is indexed exactly
    (vehicle frames are uniform across placements). ``mask`` is the per-model
    ``MeshFlag`` bitmask (e.g. ghost / mask geometry).
    """
    for mesh_frames in model.meshes:
        idx = min(frame, len(mesh_frames) - 1) if clamp_frame else frame
        mf = mesh_frames[idx]
        if mf.mesh_index == -1:
            continue
        builder.add_model(
            meshes[mf.mesh_index],
            orientation_to_matrix(mf.orientation),
            mf.position.astype(np.float64),
            mask,
        )
