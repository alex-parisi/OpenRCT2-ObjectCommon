"""Shared Blender PropertyGroup utilities and base classes.

Provides helpers and property definitions common to both the vehicle and scenery
add-ons. Requires ``bpy``; only import inside Blender.

NOTE: no ``from __future__ import annotations``; PEP 563 would stringify the
``prop: SomeProperty(...)`` definitions and break Blender registration.
"""

from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
)
from bpy.types import PropertyGroup
from openrct2_x7_renderer.constants import TILE_SIZE


def title(name: str) -> str:
    """``"steep_slopes"`` → ``"Steep Slopes"``."""
    return name.replace("_", " ").title()


def simple_items(names):
    """Build ``(identifier, label, description)`` tuples for a single-select enum."""
    return [(n, title(n), "") for n in names]


SCALE_PRESET_VALUES = {
    "REALISTIC": TILE_SIZE,
    "TILE": 1.0,
}

SCALE_PRESET_ITEMS = [
    ("REALISTIC", f"Realistic ({TILE_SIZE:g} m/tile)", "Match RCT2's real-world tile scale"),
    ("TILE", "1 unit = 1 tile", "Model in tiles: one OBJ unit spans one tile"),
    ("CUSTOM", "Custom", "Set the units-per-tile value manually"),
]


def scale_preset_update(self, _context):
    """Write the preset's units-per-tile into the consumed value (Custom: no-op)."""
    value = SCALE_PRESET_VALUES.get(self.scale_preset)
    if value is not None:
        self.units_per_tile = value


LIGHT_TYPE_ITEMS = [
    ("diffuse", "Diffuse", "Directional diffuse light"),
    ("specular", "Specular", "Specular highlight light"),
]


class SharedLight(PropertyGroup):
    """One entry in a custom lighting rig (shared by both add-ons)."""

    type: EnumProperty(name="Type", items=LIGHT_TYPE_ITEMS, default="diffuse")
    shadow: BoolProperty(
        name="Casts Shadow",
        description="Whether this light contributes to ambient-occlusion shadowing",
        default=False,
    )
    direction: FloatVectorProperty(
        name="Direction",
        description="Direction in OBJ space (+X forward, +Y up, +Z right); normalized at render",
        size=3,
        default=(0.0, 1.0, 0.0),
        subtype="XYZ",
    )
    strength: FloatProperty(
        name="Strength",
        description="Light intensity",
        default=0.5,
        min=0.0,
    )
