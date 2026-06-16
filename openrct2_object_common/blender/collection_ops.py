"""Shared factory for ``CollectionProperty`` add/remove operators.

Every add-on manages several bpy ``CollectionProperty`` lists (lights, colour
presets, car types, footprint tiles, scenery-group entries, track sections)
through the same operator pair: *add* appends an item and selects it; *remove*
drops the selected item and clamps the active index. They differ only in names,
which settings group / collection they touch, and the odd extra step on add
(seed a name, cap the count). ``make_collection_ops`` builds that pair from those
parameters; the lighting-rig case lives in ``lights_ui.make_light_ops``.

Requires ``bpy``; only import inside Blender. A no-op ``Operator`` stub lets the
module import for type-checking / test collection without a Blender runtime.

NOTE: no ``from __future__ import annotations`` -- consistent with the other
blender helpers (PEP 563 would stringify the bpy annotations operators declare).
"""

from collections.abc import Callable

try:
    from bpy.types import Operator
except ImportError:  # pragma: no cover
    # Outside Blender: a no-op stub so the module imports without a Blender
    # runtime. Registration requires real bpy.
    class Operator:  # type: ignore[no-redef]
        pass


def _scene_attr_getter(settings_attr: str) -> Callable:
    """A ``get_settings`` that reads ``context.scene.<settings_attr>``."""

    def get_settings(context):
        return getattr(context.scene, settings_attr)

    return get_settings


def make_collection_ops(
    *,
    prefix: str,
    name: str,
    coll_attr: str,
    index_attr: str,
    add_label: str,
    add_description: str,
    remove_label: str,
    remove_description: str,
    settings_attr: str | None = None,
    get_settings: Callable | None = None,
    max_items: int | None = None,
    max_items_message: str = "",
    on_add: Callable | None = None,
) -> tuple[type, type]:
    """Build the ``(add, remove)`` operator pair for one ``CollectionProperty``.

    The operator idnames are ``{prefix}.{name}_add`` / ``{prefix}.{name}_remove``
    and the classes are named ``{PREFIX}_OT_{name}_add`` /
    ``{PREFIX}_OT_{name}_remove`` (e.g. ``prefix="vg", name="car_type"`` ->
    ``vg.car_type_add`` / ``VG_OT_car_type_add``).

    The collection is reached as ``getattr(settings, coll_attr)`` and the active
    index as ``settings.<index_attr>``, where *settings* is resolved per call by
    ``get_settings(context)`` -- or, when only *settings_attr* is given, by
    ``getattr(context.scene, settings_attr)``.

    Optional hooks:
        max_items: cap the collection; add reports *max_items_message* and
            cancels when the collection is already this long.
        on_add: ``on_add(context, settings, item)`` run after a new item is
            appended (e.g. to seed its name or a default slot).
    """
    resolve = get_settings if get_settings is not None else _scene_attr_getter(settings_attr or "")
    upper = prefix.upper()

    def add_execute(self, context):
        settings = resolve(context)
        coll = getattr(settings, coll_attr)
        if max_items is not None and len(coll) >= max_items:
            self.report({"WARNING"}, max_items_message)
            return {"CANCELLED"}
        item = coll.add()
        if on_add is not None:
            on_add(context, settings, item)
        setattr(settings, index_attr, len(coll) - 1)
        return {"FINISHED"}

    def remove_execute(self, context):
        settings = resolve(context)
        coll = getattr(settings, coll_attr)
        if not coll:
            return {"CANCELLED"}
        index = getattr(settings, index_attr)
        coll.remove(index)
        setattr(settings, index_attr, max(0, min(index, len(coll) - 1)))
        return {"FINISHED"}

    add_cls = type(
        f"{upper}_OT_{name}_add",
        (Operator,),
        {
            "bl_idname": f"{prefix}.{name}_add",
            "bl_label": add_label,
            "bl_description": add_description,
            "execute": add_execute,
        },
    )
    remove_cls = type(
        f"{upper}_OT_{name}_remove",
        (Operator,),
        {
            "bl_idname": f"{prefix}.{name}_remove",
            "bl_label": remove_label,
            "bl_description": remove_description,
            "execute": remove_execute,
        },
    )
    return add_cls, remove_cls
