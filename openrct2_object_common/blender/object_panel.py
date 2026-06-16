"""Shared "Selected Object" parent panel for the generator add-ons.

Every add-on wants to show its per-object settings in the 3D View N-panel under
a single "Selected Object" header, so that with several add-ons installed the
user sees one panel hosting each add-on's child sub-panel rather than one
"Selected Object" panel per add-on.

Blender has no built-in notion of a panel owned by several add-ons, so the
parent is registered *cooperatively*: whichever add-on loads first registers it
(guarded by ``bl_idname``), and it is only unregistered once no add-on's child
still nests under it. This module owns the single canonical copy so the add-ons
no longer each ship an identical one that must be kept byte-for-byte in sync.

Each add-on contributes a child panel with ``bl_parent_id =
SHARED_PARENT_IDNAME``; this module owns only the header (its ``draw`` is empty).

NOTE: no ``from __future__ import annotations`` -- consistent with the other
blender helpers, whose subclasses declare bpy properties as annotations that PEP
563 would stringify and break add-on registration.

This module imports ``bpy`` and is meant to run inside Blender only; install the
package's ``blender`` extra (``pip install OpenRCT2-ObjectCommon[blender]``) when
type-checking or testing it outside Blender.
"""

from collections.abc import Callable, Sequence
from typing import Any

import bpy

from .bake import draw_bake

try:
    from bpy.types import Panel
except ImportError:  # pragma: no cover
    # Outside Blender: a no-op stub so the module imports (for type-checking or
    # test collection) without a Blender runtime. Registration requires real bpy.
    class Panel:  # type: ignore[no-redef]
        pass


# The bl_idname every add-on's child panel points its bl_parent_id at. MUST stay
# constant across releases of every add-on, or the cooperative parent breaks.
SHARED_PARENT_IDNAME = "OPENRCT2_PT_selected_object"


class OPENRCT2_PT_selected_object(Panel):
    """The shared "Selected Object" header. Add-ons add child sub-panels under
    it via ``bl_parent_id``; this panel itself draws nothing."""

    bl_idname = SHARED_PARENT_IDNAME
    bl_label = "Selected Object"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRCT2"
    bl_order = 1

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == "MESH"

    def draw(self, context):
        pass


def register_shared_parent():
    """Register the shared parent unless another add-on already did.

    Call from an add-on's ``register()`` *before* registering its own child
    panels.
    """
    if not hasattr(bpy.types, SHARED_PARENT_IDNAME):
        bpy.utils.register_class(OPENRCT2_PT_selected_object)


def unregister_shared_parent():
    """Drop the shared parent only once no add-on's child still nests under it.

    Call from an add-on's ``unregister()`` *after* unregistering its own child
    panels, so the scan below sees only the other add-ons' remaining children.
    """
    cls = getattr(bpy.types, SHARED_PARENT_IDNAME, None)
    if cls is None:
        return
    for name in dir(bpy.types):
        if getattr(getattr(bpy.types, name, None), "bl_parent_id", "") == SHARED_PARENT_IDNAME:
            return
    bpy.utils.unregister_class(cls)


def make_object_view3d_panel(
    *,
    name: str,
    label: str,
    order: int,
    prop_attr: str,
    draw: Callable[[Any, Any], None],
) -> type:
    """Build an add-on's child panel under the shared "Selected Object" parent.

    Each add-on shows the active object's settings as a sub-panel of the shared
    header; the panels differ only in registration name, label, ordering, the
    per-object PropertyGroup attribute they gate on, and how they draw. This
    factory captures that boilerplate so the add-ons declare just the differences.

    Args:
        name: the panel class's ``__name__``, which Blender uses as its idname
            (must follow the ``XX_PT_yyy`` convention and be unique per add-on,
            e.g. ``"VG_PT_object_view3d"``).
        label: the tab/header label, e.g. ``"Vehicle"``.
        order: ``bl_order`` among the sibling sub-panels.
        prop_attr: the per-object PropertyGroup attribute that marks an object as
            belonging to this add-on (e.g. ``"vg_object"``); the panel only shows
            for mesh objects that have it.
        draw: called as ``draw(layout, context)`` to render the panel body.

    Returns:
        A ``Panel`` subclass ready to pass to ``bpy.utils.register_class``.
    """

    def _poll(cls: type, context: Any) -> bool:
        obj = context.object
        return obj is not None and obj.type == "MESH" and hasattr(obj, prop_attr)

    def _draw(self: Any, context: Any) -> None:
        draw(self.layout, context)

    return type(
        name,
        (Panel,),
        {
            "__doc__": (
                f'The active object\'s {label.lower()} settings, '
                'as a child of "Selected Object".'
            ),
            "bl_label": label,
            "bl_space_type": "VIEW_3D",
            "bl_region_type": "UI",
            "bl_category": "OpenRCT2",
            "bl_parent_id": SHARED_PARENT_IDNAME,
            "bl_order": order,
            "poll": classmethod(_poll),
            "draw": _draw,
        },
    )


# --- Shared scene-panel building blocks -------------------------------------
# The generators' main scene panels share the same Identity / Dither / scale /
# action-button blocks (only the field lists and operator idnames differ). These
# helpers own that boilerplate so each add-on's panel declares just its specifics.


def draw_identity_box(layout, settings, fields):
    """Draw the standard "Identity" box and its metadata props.

    ``fields`` is the ordered tuple of settings props to show; the set differs
    per generator (a stall has no ride_type, a vehicle adds original_id/capacity,
    scenery omits description, ...). Returns the box so callers can append extra
    rows inside it (e.g. the vehicle add-on tucks the scale preset in here).
    """
    box = layout.box()
    box.label(text="Identity", icon="INFO")
    for field in fields:
        box.prop(settings, field)
    return box


def draw_dither_box(layout, settings) -> None:
    """Draw the standard "Dither" box: palette dither mode + temporal stability."""
    box = layout.box()
    box.label(text="Dither", icon="MOD_NOISE")
    box.prop(settings, "dither")
    box.prop(settings, "dither_stability")


def draw_scale(layout, settings) -> None:
    """Draw the scale-preset selector, revealing units-per-tile only for Custom.

    ``layout`` is whatever container the add-on wants the rows in -- the panel
    layout for most, the Identity box for the vehicle add-on.
    """
    layout.prop(settings, "scale_preset")
    if settings.scale_preset == "CUSTOM":
        layout.prop(settings, "units_per_tile")


def draw_render_buttons(layout, test_op, export_op) -> None:
    """Draw the prominent Test Render / Export buttons at the foot of a panel.

    ``test_op`` / ``export_op`` are the add-on's operator bl_idnames (they differ
    per add-on, e.g. ``"vg.test_render"`` / ``"vg.export_parkobj"``).
    """
    col = layout.column(align=True)
    col.scale_y = 1.3
    col.operator(test_op, icon="RENDER_STILL")
    col.operator(export_op, icon="EXPORT")


# The per-material boolean flag props every add-on draws (in order) in the
# Materials box's flags column. Track inserts/extends this list.
DEFAULT_MATERIAL_FLAGS = ("is_mask", "no_ao", "edge", "dark_edge", "no_bleed")


def draw_material_settings(
    layout,
    ms,
    *,
    flags: Sequence[str] = DEFAULT_MATERIAL_FLAGS,
    preamble: Callable[[Any, Any], None] | None = None,
) -> None:
    """Draw a material's OpenRCT2 region / flags / texture / shading controls.

    Args:
        layout: the bpy UILayout to draw into.
        ms: the per-material settings PropertyGroup.
        flags: the boolean flag props listed (in order) in the flags column.
        preamble: optional ``preamble(layout, ms)`` drawing add-on-specific
            controls above the region selector (e.g. scenery's wall/banner
            classification).
    """
    if preamble is not None:
        preamble(layout, ms)
    layout.prop(ms, "region")
    col = layout.column(align=True)
    for name in flags:
        col.prop(ms, name)
    layout.prop(ms, "texture")
    draw_bake(layout.column(align=True), ms)

    col = layout.column(align=True)
    col.label(text="Shading")
    row = col.row(align=True)
    row.prop(ms, "use_color_override", text="")
    sub = row.row()
    sub.enabled = ms.use_color_override
    sub.prop(ms, "diffuse_color", text="Color")
    col.prop(ms, "specular_exponent")
    col.prop(ms, "specular_intensity")
    row = col.row(align=True)
    row.prop(ms, "use_specular_tint", text="")
    sub = row.row()
    sub.enabled = ms.use_specular_tint
    sub.prop(ms, "specular_tint", text="Specular Tint")


def draw_materials_box(
    layout,
    obj,
    material_attr: str,
    *,
    flags: Sequence[str] = DEFAULT_MATERIAL_FLAGS,
    preamble: Callable[[Any, Any], None] | None = None,
) -> None:
    """Draw the object panel's "Materials" box: the slot list (when multi-slot)
    plus the active material's settings.

    *material_attr* is the per-material PropertyGroup attribute on a bpy Material
    (e.g. ``"vg_material"``); *flags* and *preamble* are forwarded to
    :func:`draw_material_settings`.
    """
    box = layout.box()
    box.label(text="Materials", icon="MATERIAL")
    if not obj.material_slots:
        box.label(text="No materials on this object.", icon="INFO")
        return
    if len(obj.material_slots) > 1:
        box.template_list(
            "MATERIAL_UL_matslots", "", obj, "material_slots", obj, "active_material_index", rows=2
        )
    mat = obj.active_material
    if mat is None:
        box.label(text="Empty material slot.", icon="INFO")
    else:
        draw_material_settings(box, getattr(mat, material_attr), flags=flags, preamble=preamble)
