"""
Build the renderer's lighting rig from add-on UI items.

The default rig is the renderer's own ``default_lights()`` (re-exported here), so
the add-ons no longer hand-copy the nine-light list -- a single source shared
with the CLI. ``lights_from_items`` reads any sequence of objects exposing
``type`` / ``shadow`` / ``direction`` / ``strength`` (a Blender ``PropertyGroup``
collection, but nothing here imports ``bpy``), falling back to the default rig
when the collection is empty.
"""

from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np
from openrct2_x7_renderer.constants import LightType
from openrct2_x7_renderer.lights import default_lights
from openrct2_x7_renderer.types import Light

__all__ = ["LIGHT_TYPE_MAP", "default_lights", "lights_from_items", "normalize_direction"]

# The light types the add-on UI exposes (hemisphere lights are CLI/config only).
LIGHT_TYPE_MAP = {"diffuse": LightType.DIFFUSE, "specular": LightType.SPECULAR}


class _LightItem(Protocol):
    type: str
    shadow: bool
    direction: Any  # 3-element sequence (bpy float vector)
    strength: float


def normalize_direction(v: Sequence[float]) -> np.ndarray:
    """A light direction as a unit ``(3,)`` float64 vector; a zero vector is
    returned unchanged (the renderer rejects it later)."""
    arr = np.array(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    return arr / n if n > 0 else arr


def lights_from_items(items: Iterable[_LightItem]) -> list[Light]:
    """Build a light rig from UI items, or the default rig when ``items`` is empty."""
    rig = [
        Light(
            type=LIGHT_TYPE_MAP[item.type],
            shadow=bool(item.shadow),
            direction=normalize_direction(list(item.direction)),
            intensity=item.strength,
        )
        for item in items
    ]
    return rig or default_lights()
