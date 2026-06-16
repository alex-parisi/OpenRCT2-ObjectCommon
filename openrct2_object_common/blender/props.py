"""Shared Blender PropertyGroup utilities and base classes.

Provides helpers and property definitions common to both the vehicle and scenery
add-ons. Requires ``bpy``; only import inside Blender.

NOTE: no ``from __future__ import annotations``; PEP 563 would stringify the
``prop: SomeProperty(...)`` definitions and break Blender registration.
"""

from collections.abc import Iterable

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup
from openrct2_x7_renderer.constants import TILE_SIZE

from .bake import BAKE_RESOLUTION_ITEMS
from .registration import register_classes, unregister_classes


def title(name: str) -> str:
    """``"steep_slopes"`` → ``"Steep Slopes"``."""
    return name.replace("_", " ").title()


# Per-material region: how OpenRCT2 treats the material's pixels. The identifier
# is what matters -- it maps to ``mesh.Material.region`` / remap flags via
# ``mesh_extract.REGION_MAP``; the labels are shared UI text. Object kinds with a
# lift-hill chain (track, scenery) expose ``MATERIAL_REGION_ITEMS_WITH_CHAIN``;
# the rest restrict the enum to the base six. The per-kind default differs, so
# each add-on still declares its own ``region`` EnumProperty pointing at one of
# these lists.
MATERIAL_REGION_ITEMS = [
    ("NONE", "None", "Plain shaded colour"),
    ("REMAP1", "Remap 1 (primary colour)", "Recoloured by the object's primary colour"),
    ("REMAP2", "Remap 2 (secondary)", "Recoloured by the secondary colour"),
    ("REMAP3", "Remap 3 (tertiary)", "Recoloured by the tertiary colour"),
    ("GREYSCALE", "Greyscale", "Greyscale shading region"),
    ("PEEP", "Peep", "Peep region"),
]

MATERIAL_REGION_ITEMS_WITH_CHAIN = MATERIAL_REGION_ITEMS + [
    ("CHAIN", "Chain", "Lift-hill chain region"),
]


def simple_items(names):
    """Build ``(identifier, label, description)`` tuples for a single-select enum."""
    return [(n, title(n), "") for n in names]


def copy_props(src, dst) -> None:
    """Recursively deep-copy a PropertyGroup's property values from *src* to *dst*.

    Nested PropertyGroups (POINTER) recurse; collections are cleared and rebuilt
    item-by-item; ID-datablock pointers (e.g. an Image) are shared by reference;
    array properties are copied as tuples; read-only properties and ``rna_type``
    are skipped. Used to seed a new entry's settings from an existing one.
    """
    for prop in src.bl_rna.properties:
        ident = prop.identifier
        if ident == "rna_type":
            continue
        if prop.type == "POINTER":
            value = getattr(src, ident)
            if isinstance(value, PropertyGroup):
                copy_props(value, getattr(dst, ident))
            else:
                # ID datablock pointer (e.g. an Image): share the reference.
                setattr(dst, ident, value)
        elif prop.type == "COLLECTION":
            dst_coll = getattr(dst, ident)
            dst_coll.clear()
            for item in getattr(src, ident):
                copy_props(item, dst_coll.add())
        elif not prop.is_readonly:
            value = getattr(src, ident)
            if getattr(prop, "is_array", False):
                value = tuple(value)
            setattr(dst, ident, value)


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


# Palette dithering modes. Identifiers match the strings the renderer's
# ``Context`` / ``make_context`` accept. Floyd–Steinberg is the historical
# default (highest fidelity); Bayer is screen-anchored so its pattern stays
# stable across an animation's frames, avoiding the per-frame "swimming" of
# error diffusion.
DITHER_MODE_ITEMS = [
    (
        "floyd_steinberg",
        "Floyd–Steinberg",
        "Highest fidelity, but the dither pattern shifts between animation frames",
    ),
    (
        "bayer",
        "Bayer (frame-stable)",
        "Ordered dither locked to screen position; stable across animation frames",
    ),
    (
        "blue_noise",
        "Blue noise (frame-stable)",
        "Like Bayer but a blue-noise mask; less perceptible residual motion under rotation",
    ),
    ("none", "None", "No dithering; flat palette quantisation"),
]

DEFAULT_DITHER_MODE = "floyd_steinberg"


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


# SharedLight is registered cooperatively, NOT in any add-on's ``_CLASSES``:
# Blender shares the bundled OpenRCT2-ObjectCommon wheel across the add-ons, so
# ``SharedLight`` is one class object -- whichever add-on loads first registers
# it (and owns this flag), the rest must skip it (else "already registered as a
# subclass 'SharedLight'"). The flag lives here, with the class, so ownership is
# tracked once across every add-on rather than per add-on.
_shared_light_owned = False


def register_shared_light() -> None:
    """Register :class:`SharedLight` unless another OpenRCT2 add-on already did.

    Call before registering any settings class with a
    ``CollectionProperty(type=SharedLight)``; ``is_registered`` is the reliable
    cross-add-on check (the class is not exposed as ``bpy.types.SharedLight``).
    """
    global _shared_light_owned
    if not SharedLight.is_registered:
        bpy.utils.register_class(SharedLight)
        _shared_light_owned = True


def unregister_shared_light() -> None:
    """Drop :class:`SharedLight` only if this add-on was the one that registered it."""
    global _shared_light_owned
    if _shared_light_owned:
        bpy.utils.unregister_class(SharedLight)
        _shared_light_owned = False


# (owner_type, attribute_name, settings_class) for a PointerProperty binding,
# e.g. (bpy.types.Scene, "vg_ride", VGRideSettings).
PointerBinding = tuple[type, str, type]


def register_settings(
    classes: Iterable[type], pointers: Iterable[PointerBinding]
) -> None:
    """Register an add-on's settings module in one call.

    Registers :class:`SharedLight` cooperatively (it backs the inherited
    ``SharedRenderSettings.lights`` rig, so it must exist before any settings
    subclass), then ``classes`` in order, then binds each
    ``(owner, attr, settings_cls)`` in *pointers* as a ``PointerProperty`` on its
    owner type. Pairs with :func:`unregister_settings`, which reverses all three.
    """
    register_shared_light()
    register_classes(classes)
    for owner, attr, cls in pointers:
        setattr(owner, attr, PointerProperty(type=cls))


def unregister_settings(
    classes: Iterable[type], pointers: Iterable[PointerBinding]
) -> None:
    """Reverse :func:`register_settings`: delete the pointer properties (in
    reverse), unregister ``classes``, then drop :class:`SharedLight`."""
    for owner, attr, _cls in reversed(list(pointers)):
        delattr(owner, attr)
    unregister_classes(classes)
    unregister_shared_light()


class SharedMaterialSettings(PropertyGroup):
    """Per-material settings shared by every generator's add-on.

    Carries the renderer/baking/Phong controls that are identical across the
    vehicle, scenery, track and ride add-ons. Each add-on subclasses this and
    adds its own ``region`` enum (the available regions + default differ per
    object kind) plus any object-specific flags (e.g. the track's visible-mask
    flag, the wall's glass/side classification). Blender registers inherited
    PropertyGroup annotations, so the subclass is registered as usual and picks
    these up automatically.

    Not registered directly -- only the concrete subclasses are.
    """

    is_mask: BoolProperty(name="Mask", default=False)
    no_ao: BoolProperty(name="No Ambient Occlusion", default=False)
    edge: BoolProperty(name="Edge AA", default=False)
    dark_edge: BoolProperty(name="Dark Edge AA", default=False)
    no_bleed: BoolProperty(name="No Bleed", default=False)
    texture: PointerProperty(
        name="Texture",
        description="Optional image; must be saved to disk (its file is read at export)",
        type=bpy.types.Image,
    )
    bake_procedural: BoolProperty(
        name="Bake Procedural Nodes",
        description=(
            "Bake this material's procedural shader-node graph to a texture at export "
            "(albedo only). Requires a UV unwrap. Overrides the flat color"
        ),
        default=False,
    )
    bake_resolution: EnumProperty(
        name="Bake Resolution",
        description="Pixel size of the baked texture",
        items=BAKE_RESOLUTION_ITEMS,
        default="256",
    )
    # Phong shading controls. These drive the renderer's Material fields without
    # going through Blender's PBR shader: specular is always taken from here;
    # diffuse colour falls back to the shader's Base Color unless overridden.
    use_color_override: BoolProperty(
        name="Override Color",
        description="Use the color below instead of the shader's Base Color",
        default=False,
    )
    diffuse_color: FloatVectorProperty(
        name="Color",
        description="Flat diffuse color (used when Override Color is on)",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(0.8, 0.8, 0.8),
    )
    specular_intensity: FloatProperty(
        name="Specular Intensity",
        description="Brightness of the specular highlight (scales the specular color)",
        default=0.5,
        min=0.0,
        soft_max=1.0,
    )
    specular_exponent: FloatProperty(
        name="Specular Exponent",
        description=(
            "Phong specular exponent: tightness of the highlight (higher = smaller, sharper)"
        ),
        default=50.0,
        min=1.0,
        soft_max=256.0,
    )
    use_specular_tint: BoolProperty(
        name="Tint Highlight",
        description="Tint the specular highlight with the color below (off = white)",
        default=False,
    )
    specular_tint: FloatVectorProperty(
        name="Specular Tint",
        description="Specular highlight color (used when Tint Highlight is on)",
        subtype="COLOR",
        size=3,
        min=0.0,
        max=1.0,
        default=(1.0, 1.0, 1.0),
    )


class SharedRenderSettings(PropertyGroup):
    """Scene-settings fields shared by every generator's add-on.

    Carries the geometry-scale + palette-dithering controls and the
    authors/version metadata -- identical across all add-ons. Each add-on
    subclasses this and adds its per-object ``id``/``name`` (and any
    ``description``; their defaults differ per generator) plus the
    object-specific fields. Blender registers inherited PropertyGroup
    annotations, so the subclass is registered as usual.

    Not registered directly -- only the concrete subclasses are.
    """

    scale_preset: EnumProperty(
        name="Scale",
        description="How many OBJ units map to one OpenRCT2 tile",
        items=SCALE_PRESET_ITEMS,
        default="REALISTIC",
        update=scale_preset_update,
    )
    units_per_tile: FloatProperty(
        name="Units / Tile",
        description="OBJ units per OpenRCT2 tile; drives sprite size and tile anchoring",
        default=TILE_SIZE,
        min=0.01,
        soft_max=16.0,
    )
    dither: EnumProperty(
        name="Dither",
        description=(
            "Palette dithering mode. Bayer and Blue noise stay stable across "
            "animation frames; Floyd-Steinberg has higher fidelity but its pattern "
            "shifts per frame"
        ),
        items=DITHER_MODE_ITEMS,
        default=DEFAULT_DITHER_MODE,
    )
    dither_stability: FloatProperty(
        name="Dither Stability",
        description=(
            "Temporal-stability deadband in palette units. Shading changes smaller "
            "than this quantise identically between frames, reducing dither "
            "'swimming' in animations; 0 disables it"
        ),
        default=0.0,
        min=0.0,
        soft_max=16.0,
    )
    authors: StringProperty(name="Authors", description="Comma-separated", default="")
    version: StringProperty(name="Version", default="1.0")

    # Optional custom lighting rig, shared by every add-on. With no entries the
    # renderer uses its built-in default lights; one or more entries replace
    # them. ``SharedLight`` must be registered before any subclass (the add-ons'
    # ``register`` calls ``register_shared_light`` first; see that function).
    lights: CollectionProperty(type=SharedLight)
    light_index: IntProperty(default=0)
    show_lights: BoolProperty(
        name="Custom Lighting",
        description="Override the default lighting rig with a custom one",
        default=False,
    )
