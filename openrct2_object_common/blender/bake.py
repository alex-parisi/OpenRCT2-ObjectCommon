"""Bake a material's procedural shader-node graph to an albedo texture.

The standalone renderer has no Cycles/``bpy`` at shade time, so it can't evaluate
Blender's procedural node graphs directly. Instead, when a material opts in
(``bake_procedural``), the add-on bakes the node graph's albedo (Cycles
``DIFFUSE``/``COLOR`` pass — colour without lighting, since the renderer does its
own shading) into an image at export and feeds it through the normal texture
path. Baked textures are UV-locked, so they are rotation-stable across sprites.

Shared by both add-ons. ``bpy``-dependent orchestration lives in
:func:`bake_materials`; :func:`_image_to_texture` is kept ``bpy``-free so it can
be unit-tested without Blender.
"""

from __future__ import annotations

import numpy as np
from openrct2_x7_renderer.mesh import Texture

from .mesh_extract import SceneError

_TARGET_NODE = "__orct2_bake_target"

# (identifier, label, description) tuples for a bpy EnumProperty; both add-ons
# expose the same bake resolutions.
BAKE_RESOLUTION_ITEMS = [
    ("64", "64×64", "Bake at 64×64"),
    ("128", "128×128", "Bake at 128×128"),
    ("256", "256×256", "Bake at 256×256"),
    ("512", "512×512", "Bake at 512×512"),
]


def _image_to_texture(image) -> Texture:
    """Convert a baked Blender image into a linear-RGB :class:`Texture`.

    Blender stores image rows bottom-up; the renderer samples top-left, so the
    rows are flipped (mirroring the UV V-flip in ``mesh_extract.extract_mesh``).
    Float images are already linear, matching what the renderer expects.
    """
    width, height = int(image.size[0]), int(image.size[1])
    flat = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    rgb = np.ascontiguousarray(flat[::-1, :, :3])
    return Texture(width=width, height=height, pixels=rgb)


def _bake_settings(mat, prop_attr):
    """The add-on's per-material settings if this material opts into baking."""
    s = getattr(mat, prop_attr, None)
    if s is not None and getattr(s, "bake_procedural", False):
        return s
    return None


def _add_target_node(mat, image):
    """Add (and activate) an Image Texture node Cycles will bake into."""
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    node = nodes.new("ShaderNodeTexImage")
    node.name = _TARGET_NODE
    node.image = image
    node.select = True
    nodes.active = node
    return node


def _select_only(view_layer, obj) -> None:
    for other in view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    view_layer.objects.active = obj


def bake_materials(context, objects, *, prop_attr) -> dict:
    """Bake every opted-in material on *objects* to a texture.

    Returns ``{bpy.types.Material: Texture}``. Main-thread only (drives Cycles via
    ``bpy.ops``). Raises :class:`SceneError` with author-facing guidance when a
    to-bake object has no UV map. Render engine, selection and the touched
    materials' ``use_nodes`` are saved and restored.

    *prop_attr* is the add-on's material settings attribute (``"vgs_material"`` /
    ``"vg_material"``); each opted-in material's ``bake_resolution`` sets its size.
    """
    import bpy

    # Plan: per object, the opted-in materials and their settings.
    plan = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        wanted = {}
        for slot in obj.material_slots:
            if slot.material is None:
                continue
            settings = _bake_settings(slot.material, prop_attr)
            if settings is not None:
                wanted[slot.material] = settings
        if not wanted:
            continue
        if not obj.data.uv_layers:
            name = next(iter(wanted)).name
            raise SceneError(
                f"{obj.name} / {name}: bake needs a UV map — "
                "unwrap the object (U ▸ Smart UV Project)."
            )
        plan.append((obj, wanted))
    if not plan:
        return {}

    scene = context.scene
    view_layer = context.view_layer
    prev_engine = scene.render.engine
    prev_active = view_layer.objects.active
    prev_selected = [o for o in view_layer.objects if o.select_get()]

    result: dict = {}
    try:
        scene.render.engine = "CYCLES"
        for obj, wanted in plan:
            # Every material slot used by the mesh needs an active image node for
            # the bake to succeed; non-target slots get a 1x1 throwaway.
            touched = []  # (mat, node, image, prev_use_nodes, keep)
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None:
                    continue
                keep = mat in wanted
                res = int(wanted[mat].bake_resolution) if keep else 1
                prev_use_nodes = mat.use_nodes
                image = bpy.data.images.new(
                    f"__orct2_bake_{mat.name}",
                    width=res,
                    height=res,
                    float_buffer=True,
                    alpha=False,
                )
                node = _add_target_node(mat, image)
                touched.append((mat, node, image, prev_use_nodes, keep))

            _select_only(view_layer, obj)
            bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=4)

            for mat, node, image, prev_use_nodes, keep in touched:
                if keep:
                    result[mat] = _image_to_texture(image)
                mat.node_tree.nodes.remove(node)
                mat.use_nodes = prev_use_nodes
                bpy.data.images.remove(image)
    finally:
        scene.render.engine = prev_engine
        for o in view_layer.objects:
            o.select_set(o in prev_selected)
        view_layer.objects.active = prev_active

    return result


def draw_bake(layout, ms) -> None:
    """Draw the bake toggle (+ resolution when enabled) into *layout*."""
    layout.prop(ms, "bake_procedural")
    if ms.bake_procedural:
        layout.prop(ms, "bake_resolution")
