"""Shared Blender mesh extraction utilities.

Extracts geometry from Blender scene objects into the renderer's Mesh format,
applying the standard coordinate basis change (Blender → OBJ space). Used by
both the vehicle and scenery add-ons so the extraction logic has a single
source of truth.

Requires ``bpy``; only import inside Blender.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable

import bpy
import numpy as np
from mathutils import Matrix, Vector
from openrct2_x7_renderer.constants import MaterialFlag
from openrct2_x7_renderer.image import quantize_to_indexed, read_png
from openrct2_x7_renderer.mesh import Material, Mesh, load_texture
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

# Canonical region enum string → ``(flag_bits, region_id)`` mapping, shared by
# every add-on's ``material_base`` call. ``CHAIN`` is only reachable from add-ons
# whose material enum exposes it (scenery, track); the others restrict the enum,
# so the extra row is harmless to pass through everywhere.
REGION_MAP: RegionMap = {
    "NONE": (0, 0),
    "REMAP1": (MaterialFlag.IS_REMAPPABLE, 1),
    "REMAP2": (MaterialFlag.IS_REMAPPABLE, 2),
    "REMAP3": (MaterialFlag.IS_REMAPPABLE, 3),
    "GREYSCALE": (0, 4),
    "PEEP": (0, 5),
    "CHAIN": (0, 6),
}


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
                # Blender UVs use a bottom-left origin (V=0 at the bottom of the
                # image); the renderer samples textures top-left (V=0 = row 0).
                # Flip V so the render matches Blender's viewport. With no UV
                # layer, fall back to (0, 0) (unflipped) since UVs are unused.
                if uv_layer:
                    u, v = uv_layer.data[loop_idx].uv
                    uv = (u, 1.0 - v)
                else:
                    uv = (0.0, 0.0)
                verts.append((co.x, co.y, co.z))
                norms.append((n.x, n.y, n.z))
                uvs.append((uv[0], uv[1]))
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


def rest_rotation_inverse(matrix_world):
    """The inverse of a rest-frame world rotation, for OBJ-space orientation deltas.

    Computed once at an animated rigid part's rest frame and handed to
    :func:`rigid_pose` at each sampled frame, so the part's later poses carry only
    its rotation *relative* to rest (pose 0 then emits ``[0, 0, 0]``).
    """
    return matrix_world.to_3x3().inverted_safe()


def rigid_pose(matrix_world, rest_rotation_inv) -> tuple[list[float], list[float]]:
    """OBJ-space ``(position, orientation_deg)`` for one frame of a rigid part.

    *matrix_world* is the object's evaluated world matrix at the sampled frame;
    *rest_rotation_inv* is :func:`rest_rotation_inverse` of its rest-frame matrix.
    The translation maps straight into OBJ space; the orientation is the frame's
    rotation *relative* to rest, expressed as the degree triple the renderer's
    ``rotate_y @ rotate_z @ rotate_x`` consumes. Blender's ``"YZX"`` Euler
    reconstructs as ``Ry(e.y) @ Rz(e.z) @ Rx(e.x)``, so the angles are emitted in
    that order. Shared by the vehicle (restraint) and scenery (animated-pose)
    add-ons, whose per-frame sampling differs only in how it walks the timeline.
    """
    p = BASIS @ matrix_world.to_translation()
    r_rel = matrix_world.to_3x3() @ rest_rotation_inv
    r_obj = BASIS @ r_rel @ BASIS.transposed()
    e = r_obj.to_euler("YZX")
    return (
        [float(p.x), float(p.y), float(p.z)],
        [float(math.degrees(e.y)), float(math.degrees(e.z)), float(math.degrees(e.x))],
    )


def geometry_objects(objects, role_attr: str) -> list:
    """The mesh objects in *objects* that are part of the model (role != IGNORE).

    *role_attr* is the per-object PropertyGroup carrying the OpenRCT2 role (e.g.
    ``"vgs_object"``); objects set to the IGNORE role are excluded.
    """
    return [
        obj for obj in objects if obj.type == "MESH" and getattr(obj, role_attr).role != "IGNORE"
    ]


def parse_authors(value: str) -> list[str]:
    """Split an add-on's comma-separated ``authors`` string into a clean list.

    Surrounding whitespace is stripped and blank entries are dropped, so the
    config dict every scene reader builds carries a tidy ``authors`` list.
    """
    return [a.strip() for a in value.split(",") if a.strip()]


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


def base_color_image(bmat):
    """The image feeding a material's Principled BSDF ``Base Color``, if that input
    is directly linked to an Image Texture node; otherwise ``None``.

    Lets a user texture a material with a node in the shader editor instead of also
    filling in the add-on's explicit texture pointer. Only a direct
    ``Base Color <- Image Texture`` link is followed (no chains through colour-mix
    nodes) to keep the behaviour predictable.
    """
    if not (getattr(bmat, "use_nodes", False) and bmat.node_tree is not None):
        return None
    for node in bmat.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        base = node.inputs.get("Base Color")
        if base is None or not base.is_linked:
            return None
        from_node = base.links[0].from_node
        if from_node.type == "TEX_IMAGE":
            return from_node.image
    return None


def save_bpy_image_png(img, path) -> None:
    """Write a ``bpy`` image to *path* as a PNG, without disturbing the source.

    Works on a copy so the user's image keeps its own ``file_format`` /
    ``filepath`` — the copy carries the (possibly packed or generated) pixel data,
    so an image with no usable file on disk still writes out.
    """
    copy = img.copy()
    try:
        copy.file_format = "PNG"
        copy.filepath_raw = str(path)
        copy.save()
    finally:
        bpy.data.images.remove(copy)


def _materialise_packed_image(img, tmp_prefix: str):
    """Write a packed/generated ``bpy`` image to a temp PNG and load it as a Texture.

    Returns ``None`` if the image can't be saved (e.g. it has no pixel data), so
    colour handling matches on-disk images instead of failing the whole build.
    """
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp_dir:
        tmp = os.path.join(tmp_dir, "packed.png")
        try:
            save_bpy_image_png(img, tmp)
            return load_texture(tmp)
        except (RuntimeError, OSError):
            return None


def load_bpy_image(img, *, tmp_prefix: str = "orct2_tex_"):
    """Load a ``bpy.types.Image`` into a core ``Texture``, or ``None`` if it has no
    usable pixels.

    On-disk files load directly; packed or generated images (no file on disk) are
    materialised to a temp PNG first (see :func:`save_bpy_image_png`) so their
    colour handling matches on-disk images.
    """
    if img is None:
        return None
    path = bpy.path.abspath(img.filepath_from_user() or img.filepath)
    if path and os.path.exists(path):
        return load_texture(path)
    if img.packed_file is not None or img.source == "GENERATED" or img.has_data:
        return _materialise_packed_image(img, tmp_prefix)
    return None


def apply_settings_texture(m: Material, s, bmat, baked_textures: dict) -> None:
    """Apply a material setting's texture to *m*, in the add-ons' standard order.

    An explicit image on the material settings wins; otherwise a baked
    procedural texture (from :func:`~openrct2_object_common.blender.bake`) is used
    if one was produced for *bmat*. Sets :attr:`MaterialFlag.HAS_TEXTURE` when a
    texture is applied. ``s`` is the per-material settings PropertyGroup (with a
    ``texture`` image pointer) and *baked_textures* maps ``bpy`` materials to
    pre-baked :class:`IndexedImage`-backed textures.
    """
    if s.texture is not None:
        path = bpy.path.abspath(s.texture.filepath_from_user() or s.texture.filepath)
        if path and os.path.exists(path):
            m.texture = load_texture(path)
            m.flags |= MaterialFlag.HAS_TEXTURE
    if not (m.flags & MaterialFlag.HAS_TEXTURE) and bmat in baked_textures:
        m.texture = baked_textures[bmat]
        m.flags |= MaterialFlag.HAS_TEXTURE


def make_extractor(
    material_fn: MaterialFn,
    *,
    ghost_attr: str | None = None,
    post: Callable[[Mesh], Mesh] | None = None,
) -> Callable[[object, object], Mesh | None]:
    """Build an add-on's ``_extract(obj, depsgraph)`` from :func:`extract_mesh`.

    Args:
        material_fn: the add-on's ``bpy`` material → renderer ``Material`` mapper.
        ghost_attr: when set, the per-object PropertyGroup attribute (e.g.
            ``"vgs_object"``) whose ``is_ghost`` toggle tags every face of the
            extracted mesh as ghost geometry, so the renderer traces through it.
        post: an optional final transform applied to the extracted mesh (e.g. the
            track add-on's load rotation).

    Returns:
        ``extract(obj, depsgraph) -> Mesh | None``, returning ``None`` for objects
        that yield no geometry.
    """

    def extract(obj, depsgraph) -> Mesh | None:
        mesh = extract_mesh(obj, depsgraph, material_fn)
        if mesh is None:
            return None
        if ghost_attr is not None and getattr(obj, ghost_attr).is_ghost:
            for material in mesh.materials:
                material.is_ghost = True
        if post is not None:
            mesh = post(mesh)
        return mesh

    return extract


# (material, settings) -> None: set kind-specific flags on the renderer Material.
ExtraFlagsFn = Callable[[Material, object], None]
# (material, settings, bpy_material, baked_textures) -> None.
TextureFn = Callable[[Material, object, object, dict], None]


class MaterialExtractor:
    """A build's baked-texture map plus the scene mesh extractor it feeds.

    Replaces the ``_baked_textures`` module global + ``_material_from_bpy`` +
    :func:`make_extractor` boilerplate every add-on's scene reader repeated. The
    material function runs :func:`material_base`, an optional per-generator
    ``extra`` hook (to set kind-specific flags), then a texture step
    (:func:`apply_settings_texture` by default; the vehicle add-on supplies its
    own richer resolver). :meth:`bake` refreshes the baked-texture map on the
    main thread before extraction, so :attr:`extract` sees the fresh map.
    ``ghost_attr`` / ``post`` are forwarded to :func:`make_extractor`.

    Usage (module level, then per build)::

        _extractor = MaterialExtractor("vgr_material", ghost_attr="vgr_object")
        _extract = _extractor.extract
        ...
        _extractor.bake(context, geo_objs)  # in build_*; then call _extract(obj, dg)
    """

    def __init__(
        self,
        prop_attr: str,
        *,
        extra: ExtraFlagsFn | None = None,
        texture_fn: TextureFn = apply_settings_texture,
        ghost_attr: str | None = None,
        post: Callable[[Mesh], Mesh] | None = None,
    ) -> None:
        self.prop_attr = prop_attr
        self._extra = extra
        self._texture_fn = texture_fn
        self.baked: dict = {}
        self.extract = make_extractor(self._material, ghost_attr=ghost_attr, post=post)

    def _material(self, bmat) -> Material:
        m, s = material_base(bmat, prop_attr=self.prop_attr, region_map=REGION_MAP)
        if s is None:
            return m
        if self._extra is not None:
            self._extra(m, s)
        self._texture_fn(m, s, bmat, self.baked)
        return m

    def bake(self, context, objects) -> None:
        """Bake opted-in procedural materials on *objects* into :attr:`baked`.

        Main-thread only (drives Cycles). Imported lazily to avoid a circular
        import with :mod:`openrct2_object_common.blender.bake`.
        """
        from .bake import bake_materials

        self.baked = bake_materials(context, objects, prop_attr=self.prop_attr)
