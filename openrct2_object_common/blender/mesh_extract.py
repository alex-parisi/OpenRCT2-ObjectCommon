"""Shared Blender mesh extraction utilities.

Extracts geometry from Blender scene objects into the renderer's Mesh format,
applying the standard coordinate basis change (Blender → OBJ space). Used by
both the vehicle and scenery add-ons so the extraction logic has a single
source of truth.

Requires ``bpy``; only import inside Blender.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import bpy
import numpy as np
from mathutils import Matrix, Vector
from openrct2_x7_renderer.image import quantize_to_indexed, read_png
from openrct2_x7_renderer.mesh import Material, Mesh
from openrct2_x7_renderer.types import IndexedImage

# Blender (x, y, z) → OBJ (x, z, -y). Proper rotation (det = +1), so winding
# is preserved. Every contributing object bakes this into emitted vertices.
BASIS = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))


class SceneError(Exception):
    """Raised when the Blender scene can't be converted to a valid object."""


MaterialFn = Callable[[object], Material]


def base_color(bmat) -> tuple[float, float, float]:
    """The material's flat RGB colour from the Principled BSDF Base Color.

    Falls back to ``diffuse_color`` (viewport colour) when there's no Principled
    BSDF node or the Base Color input is linked (textured).
    """
    if getattr(bmat, "use_nodes", False) and bmat.node_tree is not None:
        for node in bmat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            base = node.inputs.get("Base Color")
            if base is not None and not base.is_linked:
                c = base.default_value
                return (c[0], c[1], c[2])
    col = bmat.diffuse_color
    return (col[0], col[1], col[2])


RegionMap = dict[str, tuple[int, int]]


def material_base(
    bmat, *, prop_attr: str, region_map: RegionMap
) -> tuple[Material, object]:
    """Build a Material with shared colour/specular/region/flag handling.

    Returns ``(material, settings)`` where *settings* is the add-on's property
    group (e.g. ``bmat.vg_material``) or ``None``. Callers extend *material*
    with domain-specific flags and texture loading after this returns.

    *prop_attr*: attribute name on the bpy material (e.g. ``"vg_material"``).
    *region_map*: maps region enum string → ``(flag_bits, region_id)``.
    """
    from openrct2_x7_renderer.constants import MaterialFlag

    m = Material()
    if bmat is None:
        return m, None

    s = getattr(bmat, prop_attr, None)

    if s is not None and s.use_color_override:
        m.color = np.array(tuple(s.diffuse_color), dtype=np.float64)
    else:
        m.color = np.array(base_color(bmat), dtype=np.float64)

    intensity = float(s.specular_intensity) if s is not None else 0.5
    m.specular_exponent = float(s.specular_exponent) if s is not None else 50.0
    tint = tuple(s.specular_tint) if (s is not None and s.use_specular_tint) else (1.0, 1.0, 1.0)
    m.specular_color = np.array(tint, dtype=np.float64) * intensity

    if s is None:
        return m, None

    flag, region = region_map.get(s.region, (0, 0))
    m.flags |= flag
    m.region = region
    if s.is_mask:
        m.flags |= MaterialFlag.IS_MASK
    if s.no_ao:
        m.flags |= MaterialFlag.NO_AO
    if s.edge:
        m.flags |= MaterialFlag.BACKGROUND_AA
    if s.dark_edge:
        m.flags |= MaterialFlag.BACKGROUND_AA_DARK
    if s.no_bleed:
        m.flags |= MaterialFlag.NO_BLEED

    return m, s


def extract_mesh(obj, depsgraph, material_fn: MaterialFn) -> Mesh | None:
    """Evaluate *obj*, bake its world rotation+scale + basis change, → Mesh.

    *material_fn* converts a ``bpy.types.Material`` (or ``None``) into the
    renderer's :class:`~openrct2_x7_renderer.mesh.Material`. Each add-on
    provides its own implementation to handle domain-specific flags.
    """
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        me.calc_loop_triangles()
        tris = me.loop_triangles
        if len(tris) == 0:
            return None

        slots = [s.material for s in obj.material_slots]
        materials = [material_fn(bm) for bm in slots] or [Material()]
        n_mats = len(materials)

        linear = BASIS @ obj.matrix_world.to_3x3()
        normal_mat = linear.inverted_safe().transposed()

        uv_layer = me.uv_layers.active
        verts: list[tuple[float, float, float]] = []
        norms: list[tuple[float, float, float]] = []
        uvs: list[tuple[float, float]] = []
        faces: list[tuple[int, int, int]] = []
        face_mats: list[int] = []

        for lt in tris:
            corner = []
            split_n = lt.split_normals
            for k in range(3):
                vidx = lt.vertices[k]
                loop_idx = lt.loops[k]
                co = linear @ me.vertices[vidx].co
                n = (normal_mat @ Vector(split_n[k])).normalized()
                uv = uv_layer.data[loop_idx].uv if uv_layer else (0.0, 0.0)
                verts.append((co.x, co.y, co.z))
                norms.append((n.x, n.y, n.z))
                # Blender UVs use a bottom-left origin (V=0 at the bottom of the
                # image); the renderer samples textures top-left (V=0 = row 0).
                # Flip V so the render matches Blender's viewport.
                uvs.append((uv[0], 1.0 - uv[1]))
                corner.append(len(verts) - 1)
            faces.append((corner[0], corner[1], corner[2]))
            face_mats.append(min(lt.material_index, n_mats - 1))

        return Mesh(
            vertices=np.array(verts, dtype=np.float32),
            normals=np.array(norms, dtype=np.float32),
            uvs=np.array(uvs, dtype=np.float32),
            faces=np.array(faces, dtype=np.uint32),
            face_materials=np.array(face_mats, dtype=np.uint32),
            materials=materials,
        )
    finally:
        eval_obj.to_mesh_clear()


def object_position(obj) -> list[float]:
    """World translation of *obj* converted to OBJ space."""
    p = BASIS @ obj.matrix_world.to_translation()
    return [float(p.x), float(p.y), float(p.z)]


def load_preview(filepath) -> IndexedImage | None:
    """Load a preview image from *filepath*, quantizing non-paletted sources."""
    if not filepath:
        return None
    path = bpy.path.abspath(filepath)
    if not path or not os.path.exists(path):
        return None
    try:
        return read_png(path)
    except Exception:
        pass
    try:
        return quantize_to_indexed(path)
    except Exception:
        return None
