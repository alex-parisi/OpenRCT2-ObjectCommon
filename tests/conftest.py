"""Shared test scaffolding.

The ``openrct2_object_common.blender`` submodules import ``bpy`` and
``mathutils``, which only exist inside a running Blender. Minimal fakes are
installed into ``sys.modules`` here — before any test module is collected — so
the blender helpers can be imported and exercised without a Blender runtime.

The fakes are intentionally tiny: ``bpy.props`` factories return inert markers
(the property descriptors are never invoked outside Blender), and the
``mathutils`` ``Matrix``/``Vector`` types are thin numpy wrappers covering only
the operations the helpers actually use.
"""

import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# Fake mathutils — numpy-backed Matrix/Vector
# ---------------------------------------------------------------------------


class _Vector:
    def __init__(self, data):
        self.v = np.asarray(data, dtype=np.float64).reshape(-1)

    @property
    def x(self) -> float:
        return float(self.v[0])

    @property
    def y(self) -> float:
        return float(self.v[1])

    @property
    def z(self) -> float:
        return float(self.v[2])

    def __getitem__(self, i):
        return float(self.v[i])

    def normalized(self) -> "_Vector":
        norm = float(np.linalg.norm(self.v))
        if norm == 0.0:
            return _Vector(self.v.copy())
        return _Vector(self.v / norm)


class _Matrix:
    def __init__(self, rows):
        self.m = np.asarray(rows, dtype=np.float64)

    def __matmul__(self, other):
        if isinstance(other, _Vector):
            return _Vector(self.m @ other.v)
        if isinstance(other, _Matrix):
            return _Matrix(self.m @ other.m)
        return NotImplemented

    def to_3x3(self) -> "_Matrix":
        return _Matrix(self.m[:3, :3])

    def to_translation(self) -> _Vector:
        return _Vector(self.m[:3, 3])

    def inverted_safe(self) -> "_Matrix":
        try:
            return _Matrix(np.linalg.inv(self.m))
        except np.linalg.LinAlgError:
            return _Matrix(np.linalg.pinv(self.m))

    def transposed(self) -> "_Matrix":
        return _Matrix(self.m.T)


def _install_mathutils() -> None:
    mod = types.ModuleType("mathutils")
    mod.Matrix = _Matrix
    mod.Vector = _Vector
    sys.modules["mathutils"] = mod


# ---------------------------------------------------------------------------
# Fake bpy — types, props factories, path helper
# ---------------------------------------------------------------------------


class _FakeOperator:
    """Stand-in for bpy.types.Operator."""

    def report(self, type_set, message):
        pass


class _FakePropertyGroup:
    """Stand-in for bpy.types.PropertyGroup."""


def _prop_factory(*args, **kwargs):
    """Inert stand-in for a bpy.props.*Property descriptor."""
    return ("prop", args, kwargs)


def _install_bpy() -> None:
    bpy = types.ModuleType("bpy")

    bpy_types = types.ModuleType("bpy.types")
    bpy_types.Operator = _FakeOperator
    bpy_types.PropertyGroup = _FakePropertyGroup

    bpy_props = types.ModuleType("bpy.props")
    for name in (
        "BoolProperty",
        "EnumProperty",
        "FloatProperty",
        "FloatVectorProperty",
        "IntProperty",
        "StringProperty",
        "PointerProperty",
        "CollectionProperty",
    ):
        setattr(bpy_props, name, _prop_factory)

    bpy_path = types.ModuleType("bpy.path")
    bpy_path.abspath = lambda p: p

    bpy.types = bpy_types
    bpy.props = bpy_props
    bpy.path = bpy_path

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy_types
    sys.modules["bpy.props"] = bpy_props
    sys.modules["bpy.path"] = bpy_path


# Install before any test module imports the blender helpers.
_install_mathutils()
_install_bpy()
