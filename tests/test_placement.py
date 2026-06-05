"""Tests for model placement.

``orientation_to_matrix`` is pure numpy. ``add_model_to_scene`` is exercised
against a stub builder that records ``add_model`` calls, so no renderer/Embree
scene is needed.
"""

import numpy as np
from openrct2_object_common.placement import add_model_to_scene, orientation_to_matrix
from openrct2_x7_renderer.types import MeshFrame, Model


class _StubBuilder:
    """Records the (mesh, matrix, translation, mask) of each add_model call."""

    def __init__(self):
        self.calls = []

    def add_model(self, mesh, matrix, translation, mask):
        self.calls.append((mesh, matrix, translation, mask))


def test_orientation_to_matrix_zero_is_identity():
    m = orientation_to_matrix(np.zeros(3, dtype=np.float64))
    assert m.shape == (3, 3)
    np.testing.assert_allclose(m, np.eye(3), atol=1e-12)


def test_orientation_to_matrix_is_orthonormal():
    m = orientation_to_matrix(np.array([30.0, -45.0, 90.0]))
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(m), 1.0)


def test_add_model_places_each_referenced_mesh():
    meshes = ["mesh0", "mesh1"]
    model = Model(meshes=[
        [MeshFrame(mesh_index=0, position=np.array([1.0, 2.0, 3.0]))],
        [MeshFrame(mesh_index=1)],
    ])
    builder = _StubBuilder()
    add_model_to_scene(builder, meshes, model, mask=7)

    assert [c[0] for c in builder.calls] == ["mesh0", "mesh1"]
    np.testing.assert_allclose(builder.calls[0][2], [1.0, 2.0, 3.0])
    assert all(c[3] == 7 for c in builder.calls)


def test_add_model_skips_empty_slots():
    model = Model(meshes=[[MeshFrame(mesh_index=-1)], [MeshFrame(mesh_index=0)]])
    builder = _StubBuilder()
    add_model_to_scene(builder, ["only"], model)
    assert [c[0] for c in builder.calls] == ["only"]


def test_clamp_frame_falls_back_to_last_pose():
    # Placement has 2 frames; requesting frame 5 with clamp uses the last (idx 1).
    model = Model(meshes=[[MeshFrame(mesh_index=0), MeshFrame(mesh_index=1)]])
    builder = _StubBuilder()
    add_model_to_scene(builder, ["a", "b"], model, frame=5, clamp_frame=True)
    assert [c[0] for c in builder.calls] == ["b"]
