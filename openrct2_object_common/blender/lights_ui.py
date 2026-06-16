"""Shared Blender UI for the custom lighting rig: the light list + add/remove ops.

Every add-on exposes the same per-scene lighting rig -- a ``lights`` collection
with a ``light_index`` -- drawn as a UIList with Add/Remove buttons. The widgets
are identical bar the add-on's class names, operator idnames, and the scene
settings group the rig lives on, so this module builds them from those few
parameters (the same factory approach as ``object_panel``).

This is separate from ``lights`` (the bpy-free ``lights_from_items`` helper): it
imports ``bpy`` and only runs inside Blender. Install the package's ``blender``
extra (``pip install OpenRCT2-ObjectCommon[blender]``) to type-check or test it
outside Blender.

NOTE: no ``from __future__ import annotations`` -- consistent with the other
blender helpers (PEP 563 would stringify the bpy annotations subclasses declare).
"""

from .collection_ops import make_collection_ops

try:
    from bpy.types import UIList
except ImportError:  # pragma: no cover
    # Outside Blender: a no-op stub so the module imports (for type-checking or
    # test collection) without a Blender runtime. Registration requires real bpy.
    class UIList:  # type: ignore[no-redef]
        pass


def draw_lights_rig(
    layout,
    settings,
    *,
    prefix: str,
    uilist_name: str,
    info_text: str = "No lights - using the default rig.",
) -> None:
    """Draw the collapsible custom-lighting-rig box for an add-on's settings.

    Every add-on draws the same widget: a foldout header bound to
    ``settings.show_lights`` and, when open, the ``lights`` UIList with Add/Remove
    buttons plus the selected light's fields (or an info line when empty).

    Args:
        layout: the parent ``UILayout`` to draw into.
        settings: the per-scene settings group holding ``show_lights`` / ``lights``
            / ``light_index`` (e.g. ``context.scene.vg_ride``).
        prefix: the add-on's operator namespace, so the Add/Remove buttons call
            ``{prefix}.light_add`` / ``{prefix}.light_remove`` (e.g. ``"vg"``).
        uilist_name: the add-on's lights UIList idname (e.g. ``"VG_UL_lights"``).
        info_text: line shown when the rig is enabled but has no lights.
    """
    box = layout.box()
    row = box.row()
    row.prop(
        settings,
        "show_lights",
        icon="TRIA_DOWN" if settings.show_lights else "TRIA_RIGHT",
        emboss=False,
    )
    row.label(text="", icon="LIGHT_SUN")
    if not settings.show_lights:
        return
    row = box.row()
    row.template_list(uilist_name, "", settings, "lights", settings, "light_index", rows=3)
    col = row.column(align=True)
    col.operator(f"{prefix}.light_add", icon="ADD", text="")
    col.operator(f"{prefix}.light_remove", icon="REMOVE", text="")
    if settings.lights:
        light = settings.lights[settings.light_index]
        sub = box.column()
        sub.prop(light, "type")
        sub.prop(light, "shadow")
        sub.prop(light, "direction")
        sub.prop(light, "strength")
    else:
        box.label(text=info_text, icon="INFO")


def make_lights_uilist(name: str) -> type:
    """Build an add-on's UIList for the lighting rig (one row per light: a light
    icon plus its type and strength).

    ``name`` is the class ``__name__``, which Blender uses as the UIList idname
    referenced by ``template_list`` (must follow the ``XX_UL_yyy`` convention and
    match the add-on's ``template_list`` call, e.g. ``"VG_UL_lights"``).
    """

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname):
        row = layout.row(align=True)
        row.label(text="", icon="LIGHT")
        row.prop(item, "type", text="")
        row.prop(item, "strength", text="")

    return type(name, (UIList,), {"draw_item": draw_item})


def make_light_ops(
    *,
    prefix: str,
    settings_attr: str,
    remove_description: str = "Remove the selected light",
) -> tuple[type, type]:
    """Build the ``(add, remove)`` operator pair for the lighting rig.

    Args:
        prefix: the add-on's operator namespace (lowercase). The operator idnames
            are ``{prefix}.light_add`` / ``{prefix}.light_remove`` and the classes
            are named ``{PREFIX}_OT_light_add`` / ``{PREFIX}_OT_light_remove``
            (e.g. ``"vg"`` -> ``vg.light_add`` / ``VG_OT_light_add``).
        settings_attr: the per-scene PropertyGroup holding the ``lights``
            collection and its ``light_index`` (e.g. ``"vg_ride"``).
        remove_description: tooltip for the remove operator.

    Returns:
        ``(add_cls, remove_cls)``, each a ``Operator`` subclass ready to pass to
        ``bpy.utils.register_class``.
    """
    return make_collection_ops(
        prefix=prefix,
        name="light",
        settings_attr=settings_attr,
        coll_attr="lights",
        index_attr="light_index",
        add_label="Add Light",
        add_description="Add a light to the custom lighting rig",
        remove_label="Remove Light",
        remove_description=remove_description,
    )
