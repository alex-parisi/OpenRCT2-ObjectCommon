"""Fake bpy + mathutils in sys.modules so blender helpers import without Blender."""

import sys
import types

import numpy as np

# Fake mathutils — numpy-backed Matrix/Vector.


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


class _Euler:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


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

    def to_euler(self, order: str) -> _Euler:
        # Only the renderer's "YZX" convention is needed: the matrix is taken to
        # be Ry(y) @ Rz(z) @ Rx(x), inverted back to (x, y, z) radians.
        assert order == "YZX", f"fake to_euler only supports YZX, got {order}"
        m = self.m[:3, :3]
        z = float(np.arcsin(np.clip(m[1, 0], -1.0, 1.0)))
        x = float(np.arctan2(-m[1, 2], m[1, 1]))
        y = float(np.arctan2(-m[2, 0], m[0, 0]))
        return _Euler(x, y, z)


def _install_mathutils() -> None:
    mod = types.ModuleType("mathutils")
    mod.Matrix = _Matrix
    mod.Vector = _Vector
    sys.modules["mathutils"] = mod


# Fake bpy — types, props factories, path helper.


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
    # ID-datablock types referenced as PointerProperty(type=...) targets at class
    # body evaluation (e.g. SharedMaterialSettings.texture -> bpy.types.Image).
    bpy_types.Image = type("Image", (), {})

    # The viewport progress overlay registers/unregisters a draw handler here;
    # inert stubs let ProgressOverlay.add()/remove() run without a GPU runtime.
    class _SpaceView3D:
        @staticmethod
        def draw_handler_add(fn, args, region, kind):
            return ("draw_handler", fn, args, region, kind)

        @staticmethod
        def draw_handler_remove(handle, region):
            pass

    bpy_types.SpaceView3D = _SpaceView3D

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

    # Inert register/unregister; tests that care about call order monkeypatch these.
    bpy_utils = types.ModuleType("bpy.utils")
    bpy_utils.register_class = lambda cls: None
    bpy_utils.unregister_class = lambda cls: None

    # bpy.data.images.remove — used when a temp image copy is freed after writing
    # it to disk (save_bpy_image_png); inert here.
    bpy_data = types.ModuleType("bpy.data")
    bpy_data.images = types.SimpleNamespace(remove=lambda img: None)

    bpy.types = bpy_types
    bpy.props = bpy_props
    bpy.path = bpy_path
    bpy.utils = bpy_utils
    bpy.data = bpy_data

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy_types
    sys.modules["bpy.props"] = bpy_props
    sys.modules["bpy.path"] = bpy_path
    sys.modules["bpy.utils"] = bpy_utils
    sys.modules["bpy.data"] = bpy_data


# Install before any test module imports the blender helpers.
_install_mathutils()
_install_bpy()
