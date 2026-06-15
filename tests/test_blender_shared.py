"""Tests for the shared blender helpers: mesh_extract + props.

``bpy`` and ``mathutils`` are faked in ``conftest.py`` (installed before
collection), so these modules import and run without a Blender runtime.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from mathutils import Matrix, Vector  # provided by conftest fake
from openrct2_object_common.blender import bake, mesh_extract, props
from openrct2_object_common.blender.mesh_extract import SceneError
from openrct2_x7_renderer.constants import MaterialFlag

# ===========================================================================
# props.py
# ===========================================================================


def test_title_capitalizes_and_replaces_underscores():
    assert props.title("steep_slopes") == "Steep Slopes"


def test_simple_items_builds_enum_tuples():
    assert props.simple_items(["foo_bar", "baz"]) == [
        ("foo_bar", "Foo Bar", ""),
        ("baz", "Baz", ""),
    ]


def test_scale_preset_update_writes_preset_value():
    obj = SimpleNamespace(scale_preset="TILE", units_per_tile=None)
    props.scale_preset_update(obj, None)
    assert obj.units_per_tile == props.SCALE_PRESET_VALUES["TILE"]


def test_scale_preset_update_custom_is_noop():
    obj = SimpleNamespace(scale_preset="CUSTOM", units_per_tile=42.0)
    props.scale_preset_update(obj, None)
    assert obj.units_per_tile == 42.0


def test_shared_light_propertygroup_defined():
    # Importing the module defines the PropertyGroup subclass and the enums.
    assert issubclass(props.SharedLight, props.PropertyGroup)
    assert ("CUSTOM", "Custom", "Set the units-per-tile value manually") in (
        props.SCALE_PRESET_ITEMS
    )


def test_dither_mode_items_match_renderer_modes():
    # The add-on enum identifiers must be exactly the strings make_context /
    # Context accept, and the default must be one of them.
    from openrct2_x7_renderer.ray_trace import DITHER_MODES

    ids = {ident for ident, _label, _desc in props.DITHER_MODE_ITEMS}
    assert ids == set(DITHER_MODES)
    assert props.DEFAULT_DITHER_MODE in ids


# ===========================================================================
# mesh_extract.base_color
# ===========================================================================


def _principled_node(default_value=(0.1, 0.2, 0.3), linked=False, base_present=True):
    base = None
    if base_present:
        base = SimpleNamespace(is_linked=linked, default_value=default_value)
    inputs = SimpleNamespace(get=lambda name: base if name == "Base Color" else None)
    return SimpleNamespace(type="BSDF_PRINCIPLED", inputs=inputs)


def _bmat_with_nodes(nodes):
    return SimpleNamespace(
        use_nodes=True,
        node_tree=SimpleNamespace(nodes=nodes),
        diffuse_color=(0.9, 0.9, 0.9, 1.0),
    )


def test_base_color_from_principled_base_color():
    bmat = _bmat_with_nodes([_principled_node(default_value=(0.4, 0.5, 0.6))])
    assert mesh_extract.base_color(bmat) == (0.4, 0.5, 0.6)


def test_base_color_skips_non_principled_node():
    other = SimpleNamespace(type="TEX_IMAGE")
    bmat = _bmat_with_nodes([other])
    assert mesh_extract.base_color(bmat) == (0.9, 0.9, 0.9)


def test_base_color_falls_back_when_base_linked():
    bmat = _bmat_with_nodes([_principled_node(linked=True)])
    assert mesh_extract.base_color(bmat) == (0.9, 0.9, 0.9)


def test_base_color_without_nodes_uses_diffuse_color():
    bmat = SimpleNamespace(use_nodes=False, node_tree=None, diffuse_color=(0.2, 0.3, 0.4, 1.0))
    assert mesh_extract.base_color(bmat) == (0.2, 0.3, 0.4)


# ===========================================================================
# mesh_extract.material_base
# ===========================================================================

_REGION_MAP = {"primary": (MaterialFlag.IS_REMAPPABLE, 1)}


def test_material_base_none_bmat_returns_default():
    m, s = mesh_extract.material_base(None, prop_attr="vg_material", region_map=_REGION_MAP)
    assert s is None
    assert m.specular_exponent == 50.0  # untouched default Material


def test_material_base_without_settings_uses_defaults():
    bmat = SimpleNamespace(
        vg_material=None,
        use_nodes=False,
        node_tree=None,
        diffuse_color=(0.1, 0.1, 0.1, 1.0),
    )
    m, s = mesh_extract.material_base(bmat, prop_attr="vg_material", region_map=_REGION_MAP)
    assert s is None
    assert m.specular_exponent == 50.0
    assert np.allclose(m.specular_color, np.array([1.0, 1.0, 1.0]) * 0.5)
    assert np.allclose(m.color, [0.1, 0.1, 0.1])


def test_material_base_with_settings_color_override_and_flags():
    settings = SimpleNamespace(
        use_color_override=True,
        diffuse_color=(0.7, 0.6, 0.5),
        specular_intensity=2.0,
        specular_exponent=12.0,
        use_specular_tint=True,
        specular_tint=(0.5, 0.5, 1.0),
        region="primary",
        is_mask=True,
        no_ao=True,
        edge=True,
        dark_edge=True,
        no_bleed=True,
    )
    bmat = SimpleNamespace(vg_material=settings)
    m, s = mesh_extract.material_base(bmat, prop_attr="vg_material", region_map=_REGION_MAP)
    assert s is settings
    assert np.allclose(m.color, [0.7, 0.6, 0.5])
    assert m.specular_exponent == 12.0
    assert np.allclose(m.specular_color, np.array([0.5, 0.5, 1.0]) * 2.0)
    assert m.region == 1
    for flag in (
        MaterialFlag.IS_REMAPPABLE,
        MaterialFlag.IS_MASK,
        MaterialFlag.NO_AO,
        MaterialFlag.BACKGROUND_AA,
        MaterialFlag.BACKGROUND_AA_DARK,
        MaterialFlag.NO_BLEED,
    ):
        assert m.flags & flag


def test_material_base_settings_no_override_unknown_region_no_tint():
    settings = SimpleNamespace(
        use_color_override=False,
        specular_intensity=1.0,
        specular_exponent=8.0,
        use_specular_tint=False,
        region="unmapped",
        is_mask=False,
        no_ao=False,
        edge=False,
        dark_edge=False,
        no_bleed=False,
    )
    bmat = SimpleNamespace(
        vg_material=settings,
        use_nodes=False,
        node_tree=None,
        diffuse_color=(0.3, 0.3, 0.3, 1.0),
    )
    m, s = mesh_extract.material_base(bmat, prop_attr="vg_material", region_map=_REGION_MAP)
    assert np.allclose(m.color, [0.3, 0.3, 0.3])  # base_color fallback
    assert np.allclose(m.specular_color, np.array([1.0, 1.0, 1.0]) * 1.0)
    assert m.region == 0
    assert m.flags == 0


# ===========================================================================
# mesh_extract.extract_mesh / object_position
# ===========================================================================


def _identity_world():
    return Matrix(np.eye(4))


def _make_obj(me, slots):
    eval_obj = SimpleNamespace(
        to_mesh=lambda: me,
        to_mesh_clear=lambda: None,
    )
    return SimpleNamespace(
        evaluated_get=lambda dg: eval_obj,
        material_slots=[SimpleNamespace(material=bm) for bm in slots],
        matrix_world=_identity_world(),
    )


def _make_tri(vertices, loops, normals, material_index):
    return SimpleNamespace(
        vertices=vertices,
        loops=loops,
        split_normals=normals,
        material_index=material_index,
    )


def test_extract_mesh_returns_none_when_no_triangles():
    me = SimpleNamespace(
        calc_loop_triangles=lambda: None,
        loop_triangles=[],
    )
    obj = _make_obj(me, [None])
    assert mesh_extract.extract_mesh(obj, None, lambda bm: object()) is None


def test_extract_mesh_builds_mesh_with_uvs_and_clamps_material_index():
    verts = [SimpleNamespace(co=Vector(c)) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    uv_data = [
        SimpleNamespace(uv=(0.0, 0.0)),
        SimpleNamespace(uv=(1.0, 0.0)),
        SimpleNamespace(uv=(0.0, 1.0)),
    ]
    me = SimpleNamespace(
        calc_loop_triangles=lambda: None,
        vertices=verts,
        uv_layers=SimpleNamespace(active=SimpleNamespace(data=uv_data)),
        loop_triangles=[
            _make_tri([0, 1, 2], [0, 1, 2], [(0, 0, 1), (0, 0, 1), (0, 0, 1)], material_index=0),
            _make_tri([0, 1, 2], [0, 1, 2], [(0, 0, 1), (0, 0, 1), (0, 0, 1)], material_index=5),
        ],
    )
    from openrct2_x7_renderer.mesh import Material

    obj = _make_obj(me, [object(), object()])
    result = mesh_extract.extract_mesh(obj, None, lambda bm: Material())
    assert result is not None
    assert result.faces.shape == (2, 3)
    assert list(result.face_materials) == [0, 1]  # second clamped to n_mats-1
    assert result.uvs.shape == (6, 2)


def test_extract_mesh_without_uv_layer_uses_zero_uvs():
    verts = [SimpleNamespace(co=Vector(c)) for c in ((0, 0, 0), (1, 0, 0), (0, 1, 0))]
    me = SimpleNamespace(
        calc_loop_triangles=lambda: None,
        vertices=verts,
        uv_layers=SimpleNamespace(active=None),
        loop_triangles=[
            _make_tri([0, 1, 2], [0, 1, 2], [(0, 0, 1), (0, 0, 1), (0, 0, 1)], material_index=0),
        ],
    )
    obj = _make_obj(me, [])  # no slots -> default Material list
    result = mesh_extract.extract_mesh(obj, None, lambda bm: object())
    assert result is not None
    assert np.allclose(result.uvs, 0.0)


def test_object_position_applies_basis():
    world = Matrix([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
    obj = SimpleNamespace(matrix_world=world)
    # BASIS maps (x, y, z) -> (x, z, -y): (2,3,4) -> (2, 4, -3)
    assert mesh_extract.object_position(obj) == [2.0, 4.0, -3.0]


# ===========================================================================
# mesh_extract.load_preview
# ===========================================================================


def test_load_preview_empty_path_returns_none():
    assert mesh_extract.load_preview("") is None


def test_load_preview_missing_file_returns_none(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: False)
    assert mesh_extract.load_preview("/nope.png") is None


def test_load_preview_reads_png(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: True)
    monkeypatch.setattr(mesh_extract, "read_png", lambda p: sentinel)
    assert mesh_extract.load_preview("/img.png") is sentinel


def test_load_preview_falls_back_to_quantize(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: True)

    def _boom(p):
        raise ValueError("not a paletted png")

    monkeypatch.setattr(mesh_extract, "read_png", _boom)
    monkeypatch.setattr(mesh_extract, "quantize_to_indexed", lambda p: sentinel)
    assert mesh_extract.load_preview("/img.png") is sentinel


def test_load_preview_returns_none_when_both_fail(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: True)

    def _boom(p):
        raise ValueError("bad")

    monkeypatch.setattr(mesh_extract, "read_png", _boom)
    monkeypatch.setattr(mesh_extract, "quantize_to_indexed", _boom)
    assert mesh_extract.load_preview("/img.png") is None


# ===========================================================================
# bake.py
# ===========================================================================


def test_image_to_texture_flips_rows_and_drops_alpha():
    # 2x2 RGBA float image, Blender's bottom-up row order: bottom red, top blue.
    pixels = [
        1.0, 0.0, 0.0, 1.0,  1.0, 0.0, 0.0, 1.0,  # bottom row
        0.0, 0.0, 1.0, 1.0,  0.0, 0.0, 1.0, 1.0,  # top row
    ]
    image = SimpleNamespace(size=(2, 2), pixels=pixels)
    tex = bake._image_to_texture(image)
    assert tex.width == 2
    assert tex.height == 2
    assert tex.pixels.shape == (2, 2, 3)
    # After the V-flip, row 0 is the top (blue), row 1 the bottom (red).
    assert np.allclose(tex.pixels[0, 0], [0.0, 0.0, 1.0])
    assert np.allclose(tex.pixels[1, 0], [1.0, 0.0, 0.0])


class _BakeMat:
    # A real bpy Material is hashable (by identity); SimpleNamespace is not,
    # so use a plain class for materials used as dict keys.
    def __init__(self, name, *, on, res="256"):
        self.name = name
        self.vg_material = SimpleNamespace(bake_procedural=on, bake_resolution=res)


def _bake_mat(name, *, on, res="256"):
    return _BakeMat(name, on=on, res=res)


def _bake_obj(materials, *, has_uv):
    return SimpleNamespace(
        name="Cube",
        type="MESH",
        material_slots=[SimpleNamespace(material=m) for m in materials],
        data=SimpleNamespace(uv_layers=[object()] if has_uv else []),
    )


def test_bake_materials_no_opted_in_materials_returns_empty():
    obj = _bake_obj([_bake_mat("Plain", on=False)], has_uv=True)
    assert bake.bake_materials(SimpleNamespace(), [obj], prop_attr="vg_material") == {}


def test_bake_materials_skips_non_mesh_objects():
    obj = SimpleNamespace(type="EMPTY")
    assert bake.bake_materials(SimpleNamespace(), [obj], prop_attr="vg_material") == {}


def test_bake_materials_missing_uv_raises_guidance():
    obj = _bake_obj([_bake_mat("Wood", on=True)], has_uv=False)
    with pytest.raises(SceneError, match="UV map"):
        bake.bake_materials(SimpleNamespace(), [obj], prop_attr="vg_material")


class _FakeLayout:
    def __init__(self):
        self.props: list[str] = []

    def prop(self, _data, name):
        self.props.append(name)


def test_draw_bake_shows_resolution_only_when_enabled():
    on = _FakeLayout()
    bake.draw_bake(on, SimpleNamespace(bake_procedural=True, bake_resolution="256"))
    assert on.props == ["bake_procedural", "bake_resolution"]

    off = _FakeLayout()
    bake.draw_bake(off, SimpleNamespace(bake_procedural=False, bake_resolution="256"))
    assert off.props == ["bake_procedural"]
