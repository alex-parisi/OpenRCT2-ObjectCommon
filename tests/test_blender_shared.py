"""Tests for the shared blender helpers: mesh_extract + props.

``bpy`` and ``mathutils`` are faked in ``conftest.py`` (installed before
collection), so these modules import and run without a Blender runtime.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from mathutils import Matrix, Vector  # provided by conftest fake
from openrct2_object_common.blender import (
    bake,
    collection_ops,
    lights_ui,
    mesh_extract,
    modal,
    object_panel,
    progress_overlay,
    props,
    registration,
)
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
# object_panel.py
# ===========================================================================


def test_shared_parent_idname_is_canonical():
    # Every add-on's child panel pins bl_parent_id to this exact string; it must
    # not drift, or the cooperative parent breaks across add-ons.
    assert object_panel.SHARED_PARENT_IDNAME == "OPENRCT2_PT_selected_object"
    assert object_panel.OPENRCT2_PT_selected_object.bl_idname == "OPENRCT2_PT_selected_object"


def test_shared_parent_panel_attributes():
    cls = object_panel.OPENRCT2_PT_selected_object
    assert cls.bl_label == "Selected Object"
    assert cls.bl_space_type == "VIEW_3D"
    assert cls.bl_region_type == "UI"
    assert cls.bl_category == "OpenRCT2"


def test_shared_parent_poll_requires_mesh_object():
    cls = object_panel.OPENRCT2_PT_selected_object
    assert cls.poll(SimpleNamespace(object=SimpleNamespace(type="MESH"))) is True
    assert cls.poll(SimpleNamespace(object=SimpleNamespace(type="LIGHT"))) is False
    assert cls.poll(SimpleNamespace(object=None)) is False


def test_shared_parent_draw_is_a_noop():
    cls = object_panel.OPENRCT2_PT_selected_object
    # The parent draws nothing; the call must simply not raise.
    assert cls.draw(cls(), SimpleNamespace()) is None


def test_register_shared_parent_registers_when_absent(monkeypatch):
    import bpy

    registered = []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.delattr(bpy.types, object_panel.SHARED_PARENT_IDNAME, raising=False)
    object_panel.register_shared_parent()
    assert registered == [object_panel.OPENRCT2_PT_selected_object]


def test_register_shared_parent_skips_when_already_present(monkeypatch):
    import bpy

    registered = []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.setattr(
        bpy.types, object_panel.SHARED_PARENT_IDNAME, object(), raising=False
    )
    object_panel.register_shared_parent()
    assert registered == []  # another add-on already owns the parent


def test_unregister_shared_parent_noop_when_absent(monkeypatch):
    import bpy

    unregistered = []
    monkeypatch.setattr(bpy.utils, "unregister_class", unregistered.append)
    monkeypatch.delattr(bpy.types, object_panel.SHARED_PARENT_IDNAME, raising=False)
    object_panel.unregister_shared_parent()
    assert unregistered == []


def test_unregister_shared_parent_drops_when_no_children(monkeypatch):
    import bpy

    parent = object_panel.OPENRCT2_PT_selected_object
    unregistered = []
    monkeypatch.setattr(bpy.utils, "unregister_class", unregistered.append)
    monkeypatch.setattr(bpy.types, object_panel.SHARED_PARENT_IDNAME, parent, raising=False)
    object_panel.unregister_shared_parent()
    assert unregistered == [parent]


def test_unregister_shared_parent_keeps_while_child_remains(monkeypatch):
    import bpy

    parent = object_panel.OPENRCT2_PT_selected_object
    unregistered = []
    monkeypatch.setattr(bpy.utils, "unregister_class", unregistered.append)
    monkeypatch.setattr(bpy.types, object_panel.SHARED_PARENT_IDNAME, parent, raising=False)
    # A still-registered child panel nests under the shared parent.
    child = SimpleNamespace(bl_parent_id=object_panel.SHARED_PARENT_IDNAME)
    monkeypatch.setattr(bpy.types, "VG_PT_still_here", child, raising=False)
    object_panel.unregister_shared_parent()
    assert unregistered == []  # parent kept while a child remains


def test_make_object_view3d_panel_attributes():
    cls = object_panel.make_object_view3d_panel(
        name="VG_PT_object_view3d",
        label="Vehicle",
        order=0,
        prop_attr="vg_object",
        draw=lambda layout, context: None,
    )
    # Blender registers panels under the class __name__, so it must be preserved.
    assert cls.__name__ == "VG_PT_object_view3d"
    assert cls.bl_label == "Vehicle"
    assert cls.bl_order == 0
    assert cls.bl_parent_id == object_panel.SHARED_PARENT_IDNAME
    assert cls.bl_space_type == "VIEW_3D"
    assert cls.bl_category == "OpenRCT2"


def test_make_object_view3d_panel_poll_gates_on_prop_attr():
    cls = object_panel.make_object_view3d_panel(
        name="VGS_PT_object_view3d",
        label="Scenery",
        order=1,
        prop_attr="vgs_object",
        draw=lambda layout, context: None,
    )
    has = SimpleNamespace(type="MESH", vgs_object=object())
    missing = SimpleNamespace(type="MESH")  # no vgs_object
    not_mesh = SimpleNamespace(type="LIGHT", vgs_object=object())
    assert cls.poll(SimpleNamespace(object=has)) is True
    assert cls.poll(SimpleNamespace(object=missing)) is False
    assert cls.poll(SimpleNamespace(object=not_mesh)) is False
    assert cls.poll(SimpleNamespace(object=None)) is False


def test_make_object_view3d_panel_draw_forwards_layout_and_context():
    seen = {}

    def _draw(layout, context):
        seen["layout"] = layout
        seen["context"] = context

    cls = object_panel.make_object_view3d_panel(
        name="TG_PT_object_view3d", label="Track", order=2, prop_attr="tg_object", draw=_draw
    )
    panel = cls()
    panel.layout = "LAYOUT"
    cls.draw(panel, "CONTEXT")
    assert seen == {"layout": "LAYOUT", "context": "CONTEXT"}


# ===========================================================================
# lights_ui.py
# ===========================================================================


def test_make_lights_uilist_name_preserved():
    # template_list references the UIList by its class __name__, so it must be
    # preserved through the factory.
    cls = lights_ui.make_lights_uilist("VG_UL_lights")
    assert cls.__name__ == "VG_UL_lights"
    assert issubclass(cls, lights_ui.UIList)


def test_make_lights_uilist_draws_type_and_strength():
    cls = lights_ui.make_lights_uilist("VG_UL_lights")
    rows = []

    class _Row:
        def label(self, **kwargs):
            rows.append(("label", kwargs))

        def prop(self, item, name, **kwargs):
            rows.append(("prop", name))

    class _Layout:
        def row(self, align=False):
            return _Row()

    cls().draw_item(None, _Layout(), None, object(), None, None, None)
    assert [r for r in rows if r[0] == "prop"] == [("prop", "type"), ("prop", "strength")]


def test_make_light_ops_names_and_idnames():
    add_cls, remove_cls = lights_ui.make_light_ops(prefix="vgs", settings_attr="vgs_scenery")
    assert add_cls.__name__ == "VGS_OT_light_add"
    assert remove_cls.__name__ == "VGS_OT_light_remove"
    assert add_cls.bl_idname == "vgs.light_add"
    assert remove_cls.bl_idname == "vgs.light_remove"
    assert remove_cls.bl_description == "Remove the selected light"


def test_make_light_ops_remove_description_override():
    _, remove_cls = lights_ui.make_light_ops(
        prefix="vg", settings_attr="vg_ride", remove_description="custom tip"
    )
    assert remove_cls.bl_description == "custom tip"


def _scene_with_lights(attr, lights, index):
    settings = SimpleNamespace(lights=lights, light_index=index)
    return SimpleNamespace(scene=SimpleNamespace(**{attr: settings})), settings


def test_make_light_ops_add_appends_and_selects():
    class _Lights(list):
        def add(self):
            self.append(object())

    add_cls, _ = lights_ui.make_light_ops(prefix="vg", settings_attr="vg_ride")
    lights = _Lights([object()])
    context, settings = _scene_with_lights("vg_ride", lights, 0)
    assert add_cls.execute(add_cls(), context) == {"FINISHED"}
    assert len(lights) == 2
    assert settings.light_index == 1


def test_make_light_ops_remove_clamps_index_and_handles_empty():
    class _Lights(list):
        def remove(self, i):
            del self[i]

    _, remove_cls = lights_ui.make_light_ops(prefix="vg", settings_attr="vg_ride")

    lights = _Lights([object(), object(), object()])
    context, settings = _scene_with_lights("vg_ride", lights, 2)
    assert remove_cls.execute(remove_cls(), context) == {"FINISHED"}
    assert len(lights) == 2
    assert settings.light_index == 1  # clamped to last valid

    empty_ctx, empty_settings = _scene_with_lights("vg_ride", _Lights(), 0)
    assert remove_cls.execute(remove_cls(), empty_ctx) == {"CANCELLED"}


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
# mesh_extract.base_color_image / save_bpy_image_png / load_bpy_image
# ===========================================================================


def _principled_linked_to(from_node):
    base = SimpleNamespace(is_linked=True, links=[SimpleNamespace(from_node=from_node)])
    inputs = SimpleNamespace(get=lambda name: base if name == "Base Color" else None)
    return SimpleNamespace(type="BSDF_PRINCIPLED", inputs=inputs)


def test_base_color_image_without_nodes_returns_none():
    bmat = SimpleNamespace(use_nodes=False, node_tree=None)
    assert mesh_extract.base_color_image(bmat) is None


def test_base_color_image_returns_none_when_base_not_linked():
    bmat = _bmat_with_nodes([_principled_node(linked=False)])
    assert mesh_extract.base_color_image(bmat) is None


def test_base_color_image_follows_link_to_image_texture():
    tex = SimpleNamespace(type="TEX_IMAGE", image="IMG")
    bmat = _bmat_with_nodes([_principled_linked_to(tex)])
    assert mesh_extract.base_color_image(bmat) == "IMG"


def test_base_color_image_skips_non_principled_nodes():
    tex = SimpleNamespace(type="TEX_IMAGE", image="IMG")
    # A leading non-Principled node is skipped before the Principled one is read.
    bmat = _bmat_with_nodes([SimpleNamespace(type="TEX_IMAGE"), _principled_linked_to(tex)])
    assert mesh_extract.base_color_image(bmat) == "IMG"


def test_base_color_image_returns_none_for_non_texture_link():
    bmat = _bmat_with_nodes([_principled_linked_to(SimpleNamespace(type="RGB"))])
    assert mesh_extract.base_color_image(bmat) is None


def test_save_bpy_image_png_writes_via_a_copy(monkeypatch):
    saved = {}
    copy = SimpleNamespace(
        file_format=None,
        filepath_raw=None,
        save=lambda: saved.setdefault("saved", True),
    )
    removed = []
    monkeypatch.setattr(mesh_extract.bpy.data.images, "remove", removed.append)
    img = SimpleNamespace(copy=lambda: copy)

    mesh_extract.save_bpy_image_png(img, "/tmp/icon.png")

    assert copy.file_format == "PNG"
    assert copy.filepath_raw == "/tmp/icon.png"
    assert saved["saved"] is True
    # The copy (not the user's image) is the one freed afterwards.
    assert removed == [copy]


def test_load_bpy_image_none_returns_none():
    assert mesh_extract.load_bpy_image(None) is None


def test_load_bpy_image_on_disk_loads_directly(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: True)
    monkeypatch.setattr(mesh_extract, "load_texture", lambda p: ("TEX", p))
    img = SimpleNamespace(filepath_from_user=lambda: "/a.png", filepath="/a.png")
    assert mesh_extract.load_bpy_image(img) == ("TEX", "/a.png")


def test_load_bpy_image_packed_is_materialised(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: False)
    monkeypatch.setattr(mesh_extract, "save_bpy_image_png", lambda img, path: None)
    monkeypatch.setattr(mesh_extract, "load_texture", lambda p: "MATERIALISED")
    img = SimpleNamespace(
        filepath_from_user=lambda: "",
        filepath="",
        packed_file=object(),
        source="FILE",
        has_data=False,
    )
    assert mesh_extract.load_bpy_image(img) == "MATERIALISED"


def test_load_bpy_image_generated_is_materialised(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: False)
    monkeypatch.setattr(mesh_extract, "save_bpy_image_png", lambda img, path: None)
    monkeypatch.setattr(mesh_extract, "load_texture", lambda p: "MATERIALISED")
    img = SimpleNamespace(
        filepath_from_user=lambda: "",
        filepath="",
        packed_file=None,
        source="GENERATED",
        has_data=False,
    )
    assert mesh_extract.load_bpy_image(img) == "MATERIALISED"


def test_load_bpy_image_materialise_failure_returns_none(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: False)

    def _boom(img, path):
        raise OSError("no pixels")

    monkeypatch.setattr(mesh_extract, "save_bpy_image_png", _boom)
    img = SimpleNamespace(
        filepath_from_user=lambda: "",
        filepath="",
        packed_file=object(),
        source="FILE",
        has_data=False,
    )
    assert mesh_extract.load_bpy_image(img) is None


def test_load_bpy_image_no_pixels_returns_none(monkeypatch):
    monkeypatch.setattr(mesh_extract.os.path, "exists", lambda p: False)
    img = SimpleNamespace(
        filepath_from_user=lambda: "",
        filepath="",
        packed_file=None,
        source="FILE",
        has_data=False,
    )
    assert mesh_extract.load_bpy_image(img) is None


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
# mesh_extract.rest_rotation_inverse / rigid_pose
# ===========================================================================


def test_rest_rotation_inverse_is_matrix_inverse():
    # 90deg rotation about Z: its inverse composed with it is the identity.
    world = Matrix([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    inv = mesh_extract.rest_rotation_inverse(world)
    assert np.allclose((world.to_3x3() @ inv).m, np.eye(3))


def test_rigid_pose_maps_translation_and_rest_is_zero_orientation():
    world = Matrix([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
    rest_inv = mesh_extract.rest_rotation_inverse(world)
    position, orientation = mesh_extract.rigid_pose(world, rest_inv)
    # BASIS maps (x, y, z) -> (x, z, -y): (2, 3, 4) -> (2, 4, -3)
    assert position == [2.0, 4.0, -3.0]
    # At rest (frame rotation == rest), the orientation delta is zero.
    assert np.allclose(orientation, [0.0, 0.0, 0.0])


def test_rigid_pose_orientation_delta_relative_to_rest():
    rest = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    rest_inv = mesh_extract.rest_rotation_inverse(rest)
    # +90deg about Blender Z. BASIS maps Blender Z -> OBJ Y, so the renderer-space
    # delta is a +90deg rotation about Y, i.e. orientation [deg_y, deg_z, deg_x].
    world = Matrix([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    _position, orientation = mesh_extract.rigid_pose(world, rest_inv)
    assert np.allclose(orientation, [90.0, 0.0, 0.0])


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


class _BakeNodes:
    """Fake material node collection: records new/remove + the active node."""

    def __init__(self):
        self.active = None
        self.items: list = []

    def new(self, kind):
        assert kind == "ShaderNodeTexImage"
        node = SimpleNamespace(name="", image=None, select=False)
        self.items.append(node)
        return node

    def remove(self, node):
        self.items.remove(node)


class _BakeableMat:
    def __init__(self, name, *, on, res="2"):
        self.name = name
        self.vg_material = SimpleNamespace(bake_procedural=on, bake_resolution=res)
        self.use_nodes = False
        self.node_tree = SimpleNamespace(nodes=_BakeNodes())


class _BakeObjects(list):
    """A view_layer.objects stand-in: iterable and carries ``.active``."""

    active = None


def test_bake_materials_full_bake_round_trip(monkeypatch):
    import bpy
    from openrct2_x7_renderer.mesh import Texture

    target = _BakeableMat("Wood", on=True, res="2")
    plain = _BakeableMat("Plain", on=False)
    obj = SimpleNamespace(
        name="Cube",
        type="MESH",
        material_slots=[
            SimpleNamespace(material=target),
            SimpleNamespace(material=plain),  # not opted in -> 1x1 throwaway slot
            SimpleNamespace(material=None),  # empty slot -> skipped
        ],
        data=SimpleNamespace(uv_layers=[object()]),
        _selected=False,
    )
    obj.select_get = lambda: obj._selected
    obj.select_set = lambda value: setattr(obj, "_selected", value)

    view_layer = SimpleNamespace(objects=_BakeObjects([obj]))
    view_layer.objects.active = None
    scene = SimpleNamespace(render=SimpleNamespace(engine="EEVEE"))
    context = SimpleNamespace(scene=scene, view_layer=view_layer)

    def _new_image(name, *, width, height, float_buffer, alpha):
        return SimpleNamespace(size=(width, height), pixels=[0.0] * (width * height * 4))

    bake_calls: list = []
    monkeypatch.setattr(bpy.data.images, "new", _new_image, raising=False)
    monkeypatch.setattr(
        bpy,
        "ops",
        SimpleNamespace(object=SimpleNamespace(bake=lambda **kw: bake_calls.append(kw))),
        raising=False,
    )

    result = bake.bake_materials(context, [obj], prop_attr="vg_material")

    # Only the opted-in material yields a Texture.
    assert set(result) == {target}
    assert isinstance(result[target], Texture)
    assert (result[target].width, result[target].height) == (2, 2)
    # The bake op ran with the albedo-only DIFFUSE/COLOR pass.
    assert bake_calls and bake_calls[0]["type"] == "DIFFUSE"
    assert bake_calls[0]["pass_filter"] == {"COLOR"}
    # Render engine and each touched material's use_nodes are restored, and the
    # temporary target node is removed.
    assert scene.render.engine == "EEVEE"
    assert target.use_nodes is False
    assert target.node_tree.nodes.items == []


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


# ===========================================================================
# mesh_extract.apply_settings_texture + make_extractor
# ===========================================================================


def test_apply_settings_texture_uses_baked_when_no_explicit_image():
    bmat = object()
    m = SimpleNamespace(flags=0, texture=None)
    s = SimpleNamespace(texture=None)
    mesh_extract.apply_settings_texture(m, s, bmat, {bmat: "BAKED"})
    assert m.texture == "BAKED"
    assert m.flags & MaterialFlag.HAS_TEXTURE


def test_apply_settings_texture_loads_explicit_image(monkeypatch, tmp_path):
    img_path = tmp_path / "tex.png"
    img_path.write_bytes(b"x")
    monkeypatch.setattr(mesh_extract, "load_texture", lambda p: ("TEX", p))
    m = SimpleNamespace(flags=0, texture=None)
    s = SimpleNamespace(
        texture=SimpleNamespace(
            filepath_from_user=lambda: str(img_path),
            filepath=str(img_path),
        )
    )
    # An explicit image wins over any baked texture for the same material.
    mesh_extract.apply_settings_texture(m, s, object(), {object(): "BAKED"})
    assert m.texture == ("TEX", str(img_path))
    assert m.flags & MaterialFlag.HAS_TEXTURE


def test_apply_settings_texture_no_image_no_baked_is_noop():
    m = SimpleNamespace(flags=0, texture=None)
    s = SimpleNamespace(texture=None)
    mesh_extract.apply_settings_texture(m, s, object(), {})
    assert m.texture is None
    assert m.flags == 0


def test_apply_settings_texture_existing_texture_skips_baked():
    bmat = object()
    m = SimpleNamespace(flags=MaterialFlag.HAS_TEXTURE, texture="ALREADY")
    s = SimpleNamespace(texture=None)
    mesh_extract.apply_settings_texture(m, s, bmat, {bmat: "BAKED"})
    assert m.texture == "ALREADY"


def test_make_extractor_tags_ghost_when_toggled(monkeypatch):
    fake = SimpleNamespace(
        materials=[SimpleNamespace(is_ghost=False), SimpleNamespace(is_ghost=False)]
    )
    monkeypatch.setattr(mesh_extract, "extract_mesh", lambda obj, dg, fn: fake)
    extract = mesh_extract.make_extractor(lambda bm: None, ghost_attr="vgs_object")
    obj = SimpleNamespace(vgs_object=SimpleNamespace(is_ghost=True))
    assert extract(obj, None) is fake
    assert all(mat.is_ghost for mat in fake.materials)


def test_make_extractor_leaves_ghost_untouched_when_off(monkeypatch):
    fake = SimpleNamespace(materials=[SimpleNamespace(is_ghost=False)])
    monkeypatch.setattr(mesh_extract, "extract_mesh", lambda obj, dg, fn: fake)
    extract = mesh_extract.make_extractor(lambda bm: None, ghost_attr="vgr_object")
    obj = SimpleNamespace(vgr_object=SimpleNamespace(is_ghost=False))
    extract(obj, None)
    assert fake.materials[0].is_ghost is False


def test_make_extractor_applies_post_transform(monkeypatch):
    fake = SimpleNamespace(materials=[])
    rotated = SimpleNamespace(materials=[])
    monkeypatch.setattr(mesh_extract, "extract_mesh", lambda obj, dg, fn: fake)
    extract = mesh_extract.make_extractor(lambda bm: None, post=lambda mesh: rotated)
    assert extract(object(), None) is rotated


def test_make_extractor_returns_none_for_empty_geometry(monkeypatch):
    monkeypatch.setattr(mesh_extract, "extract_mesh", lambda obj, dg, fn: None)
    extract = mesh_extract.make_extractor(
        lambda bm: None, ghost_attr="vgs_object", post=lambda m: m
    )
    assert extract(SimpleNamespace(vgs_object=SimpleNamespace(is_ghost=True)), None) is None


# ===========================================================================
# mesh_extract.MaterialExtractor
# ===========================================================================


def test_material_extractor_runs_extra_then_texture_with_baked(monkeypatch):
    m = SimpleNamespace(flags=0)
    settings = SimpleNamespace()
    monkeypatch.setattr(
        mesh_extract, "material_base", lambda bmat, *, prop_attr, region_map: (m, settings)
    )
    calls = []
    ex = mesh_extract.MaterialExtractor(
        "vg_material",
        extra=lambda mat, s: calls.append(("extra", mat is m, s is settings)),
        texture_fn=lambda mat, s, bmat, baked: calls.append(("texture", bmat, baked)),
    )
    ex.baked = {"k": "BAKED"}
    assert ex._material("BMAT") is m
    # extra runs before the texture step; both see the live baked map.
    assert calls == [("extra", True, True), ("texture", "BMAT", {"k": "BAKED"})]


def test_material_extractor_skips_hooks_when_no_settings(monkeypatch):
    m = SimpleNamespace(flags=0)
    monkeypatch.setattr(
        mesh_extract, "material_base", lambda bmat, *, prop_attr, region_map: (m, None)
    )
    called = []
    ex = mesh_extract.MaterialExtractor(
        "vg_material",
        extra=lambda *a: called.append("extra"),
        texture_fn=lambda *a: called.append("texture"),
    )
    assert ex._material("BMAT") is m
    assert called == []


def test_material_extractor_default_texture_fn_is_apply_settings_texture():
    assert mesh_extract.MaterialExtractor("vg_material")._texture_fn is (
        mesh_extract.apply_settings_texture
    )


def test_material_extractor_bake_populates_baked(monkeypatch):
    from openrct2_object_common.blender import bake as bake_mod

    monkeypatch.setattr(
        bake_mod,
        "bake_materials",
        lambda context, objects, *, prop_attr: {"prop_attr": prop_attr, "objs": objects},
    )
    ex = mesh_extract.MaterialExtractor("tg_material")
    ex.bake("CTX", ["o1", "o2"])
    assert ex.baked == {"prop_attr": "tg_material", "objs": ["o1", "o2"]}


# ===========================================================================
# mesh_extract.geometry_objects
# ===========================================================================


def test_geometry_objects_filters_non_mesh_and_ignore_role():
    keep = SimpleNamespace(type="MESH", vgs_object=SimpleNamespace(role="GEOMETRY"))
    ignored = SimpleNamespace(type="MESH", vgs_object=SimpleNamespace(role="IGNORE"))
    lamp = SimpleNamespace(type="LIGHT", vgs_object=SimpleNamespace(role="GEOMETRY"))
    result = mesh_extract.geometry_objects([keep, ignored, lamp], "vgs_object")
    assert result == [keep]


def test_geometry_objects_honours_role_attr():
    obj = SimpleNamespace(type="MESH", vgr_object=SimpleNamespace(role="GEOMETRY"))
    assert mesh_extract.geometry_objects([obj], "vgr_object") == [obj]


# ===========================================================================
# props.copy_props
# ===========================================================================


def _prop_spec(identifier, ptype, *, is_readonly=False, is_array=False):
    return SimpleNamespace(
        identifier=identifier, type=ptype, is_readonly=is_readonly, is_array=is_array
    )


def test_copy_props_scalars_arrays_and_skips():
    specs = [
        _prop_spec("rna_type", "POINTER"),  # always skipped before the POINTER branch
        _prop_spec("name", "STRING"),
        _prop_spec("count", "INT"),
        _prop_spec("color", "FLOAT", is_array=True),
        _prop_spec("locked", "STRING", is_readonly=True),
    ]
    src = SimpleNamespace(
        bl_rna=SimpleNamespace(properties=specs),
        rna_type="SRC",
        name="hello",
        count=5,
        color=[1.0, 2.0, 3.0],
        locked="src",
    )
    dst = SimpleNamespace(rna_type="DST", name="", count=0, color=None, locked="dst")
    props.copy_props(src, dst)
    assert dst.name == "hello"
    assert dst.count == 5
    assert dst.color == (1.0, 2.0, 3.0)  # arrays copied as a tuple
    assert dst.locked == "dst"  # read-only left untouched
    assert dst.rna_type == "DST"  # rna_type skipped


def test_copy_props_pointers_and_collections():
    from bpy.types import PropertyGroup  # conftest fake

    class _PG(PropertyGroup):
        def __init__(self, specs, **values):
            self.bl_rna = SimpleNamespace(properties=specs)
            for k, v in values.items():
                setattr(self, k, v)

    class _Coll(list):
        def __init__(self, item_specs, items=()):
            super().__init__(items)
            self._item_specs = item_specs

        def add(self):
            item = _PG(self._item_specs, slot=0)
            self.append(item)
            return item

    item_specs = [_prop_spec("slot", "INT")]
    nested_specs = [_prop_spec("depth", "INT")]
    top_specs = [
        _prop_spec("nested", "POINTER"),
        _prop_spec("icon", "POINTER"),
        _prop_spec("entries", "COLLECTION"),
    ]
    image = object()  # ID datablock pointer, not a PropertyGroup

    src = _PG(
        top_specs,
        nested=_PG(nested_specs, depth=7),
        icon=image,
        entries=_Coll(item_specs, [_PG(item_specs, slot=1), _PG(item_specs, slot=2)]),
    )
    dst = _PG(
        top_specs,
        nested=_PG(nested_specs, depth=0),
        icon=None,
        entries=_Coll(item_specs, [_PG(item_specs, slot=99)]),
    )
    props.copy_props(src, dst)
    assert dst.nested.depth == 7  # recursed into nested PropertyGroup
    assert dst.icon is image  # ID datablock shared by reference
    assert [e.slot for e in dst.entries] == [1, 2]  # collection cleared then rebuilt


# ===========================================================================
# object_panel.draw_material_settings + draw_materials_box
# ===========================================================================


class _RecLayout:
    """Records every prop()/template_list() call (across the layout tree)."""

    def __init__(self, calls):
        self.calls = calls
        self.enabled = True

    def prop(self, data, name, **kw):
        self.calls.append(name)

    def label(self, **kw):
        pass

    def column(self, align=False):
        return _RecLayout(self.calls)

    def row(self, align=False):
        return _RecLayout(self.calls)

    def box(self):
        return _RecLayout(self.calls)

    def template_list(self, *a, **k):
        self.calls.append(("template_list", a[0]))


def _material_settings():
    return SimpleNamespace(
        use_color_override=False, use_specular_tint=False,
        bake_procedural=False, bake_resolution="256",
    )


def test_draw_material_settings_default_flag_order():
    calls = []
    object_panel.draw_material_settings(_RecLayout(calls), _material_settings())
    assert calls == [
        "region", "is_mask", "no_ao", "edge", "dark_edge", "no_bleed",
        "texture", "bake_procedural",
        "use_color_override", "diffuse_color", "specular_exponent",
        "specular_intensity", "use_specular_tint", "specular_tint",
    ]


def test_draw_material_settings_custom_flags():
    calls = []
    track_flags = ("is_mask", "is_visible_mask", "no_ao", "edge", "dark_edge", "no_bleed",
                   "flat_shaded")
    object_panel.draw_material_settings(_RecLayout(calls), _material_settings(), flags=track_flags)
    assert calls[1:8] == list(track_flags)


def test_draw_material_settings_preamble_runs_before_region():
    calls = []

    def preamble(layout, ms):
        calls.append("PREAMBLE")

    object_panel.draw_material_settings(_RecLayout(calls), _material_settings(), preamble=preamble)
    assert calls[0] == "PREAMBLE"
    assert calls[1] == "region"


def test_draw_materials_box_no_slots_draws_nothing():
    calls = []
    obj = SimpleNamespace(material_slots=[], active_material=None)
    object_panel.draw_materials_box(_RecLayout(calls), obj, "vg_material")
    assert calls == []


def test_draw_materials_box_empty_slot_draws_nothing():
    calls = []
    obj = SimpleNamespace(material_slots=[object()], active_material=None)
    object_panel.draw_materials_box(_RecLayout(calls), obj, "vg_material")
    assert calls == []


def test_draw_materials_box_single_slot_no_list():
    calls = []
    mat = SimpleNamespace(vg_material=_material_settings())
    obj = SimpleNamespace(material_slots=[object()], active_material=mat)
    object_panel.draw_materials_box(_RecLayout(calls), obj, "vg_material")
    assert "region" in calls and "texture" in calls
    assert all(not isinstance(c, tuple) for c in calls)  # no template_list for one slot


def test_draw_materials_box_multislot_shows_list_and_uses_attr():
    calls = []
    mat = SimpleNamespace(vgs_material=_material_settings())
    obj = SimpleNamespace(material_slots=[object(), object()], active_material=mat)
    object_panel.draw_materials_box(_RecLayout(calls), obj, "vgs_material")
    assert ("template_list", "MATERIAL_UL_matslots") in calls
    assert "region" in calls


# ===========================================================================
# object_panel scene-panel blocks: identity / dither / scale / render buttons
# ===========================================================================


def test_draw_identity_box_draws_fields_in_order():
    calls = []
    box = object_panel.draw_identity_box(_RecLayout(calls), object(), ("id", "name", "version"))
    assert calls == ["id", "name", "version"]
    assert box is not None  # returned so callers can append rows (e.g. the scale)


def test_draw_dither_box_draws_mode_and_stability():
    calls = []
    object_panel.draw_dither_box(_RecLayout(calls), object())
    assert calls == ["dither", "dither_stability"]


def test_draw_scale_hides_units_unless_custom():
    calls = []
    object_panel.draw_scale(_RecLayout(calls), SimpleNamespace(scale_preset="REALISTIC"))
    assert calls == ["scale_preset"]


def test_draw_scale_shows_units_when_custom():
    calls = []
    object_panel.draw_scale(_RecLayout(calls), SimpleNamespace(scale_preset="CUSTOM"))
    assert calls == ["scale_preset", "units_per_tile"]


class _ButtonLayout:
    """Records column()/operator() calls for the render-button footer."""

    def __init__(self, ops):
        self.ops = ops
        self.scale_y = 1.0

    def column(self, align=False):
        return self

    def operator(self, idname, **kw):
        self.ops.append((idname, kw.get("icon")))


def test_draw_render_buttons_emits_test_then_export():
    ops = []
    object_panel.draw_render_buttons(_ButtonLayout(ops), "vg.test_render", "vg.export_parkobj")
    assert ops == [
        ("vg.test_render", "RENDER_STILL"),
        ("vg.export_parkobj", "EXPORT"),
    ]


# ===========================================================================
# collection_ops.make_collection_ops
# ===========================================================================


class _Coll(list):
    """A minimal stand-in for a bpy CollectionProperty."""

    def add(self):
        item = SimpleNamespace()
        self.append(item)
        return item

    def remove(self, i):
        del self[i]


def _coll_scene(attr, items, index, *, coll_attr="items", index_attr="item_index"):
    settings = SimpleNamespace(**{coll_attr: items, index_attr: index})
    return SimpleNamespace(scene=SimpleNamespace(**{attr: settings})), settings


def _make_item_ops(**overrides):
    kwargs = dict(
        prefix="vg", name="item", settings_attr="vg_ride",
        coll_attr="items", index_attr="item_index",
        add_label="Add Item", add_description="add it",
        remove_label="Remove Item", remove_description="remove it",
    )
    kwargs.update(overrides)
    return collection_ops.make_collection_ops(**kwargs)


def test_make_collection_ops_names_and_idnames():
    add_cls, remove_cls = collection_ops.make_collection_ops(
        prefix="vg", name="car_type", settings_attr="vg_ride",
        coll_attr="car_types", index_attr="car_type_index",
        add_label="Add Car Type", add_description="add a car type",
        remove_label="Remove Car Type", remove_description="remove the car type",
    )
    assert add_cls.__name__ == "VG_OT_car_type_add"
    assert remove_cls.__name__ == "VG_OT_car_type_remove"
    assert add_cls.bl_idname == "vg.car_type_add"
    assert remove_cls.bl_idname == "vg.car_type_remove"
    assert add_cls.bl_label == "Add Car Type"
    assert remove_cls.bl_description == "remove the car type"


def test_make_collection_ops_add_appends_and_selects():
    add_cls, _ = _make_item_ops()
    items = _Coll([SimpleNamespace()])
    context, settings = _coll_scene("vg_ride", items, 0)
    assert add_cls.execute(add_cls(), context) == {"FINISHED"}
    assert len(items) == 2
    assert settings.item_index == 1


def test_make_collection_ops_remove_clamps_index_and_handles_empty():
    _, remove_cls = _make_item_ops()
    items = _Coll([SimpleNamespace(), SimpleNamespace(), SimpleNamespace()])
    context, settings = _coll_scene("vg_ride", items, 2)
    assert remove_cls.execute(remove_cls(), context) == {"FINISHED"}
    assert len(items) == 2
    assert settings.item_index == 1  # clamped to last valid

    empty_ctx, _ = _coll_scene("vg_ride", _Coll(), 0)
    assert remove_cls.execute(remove_cls(), empty_ctx) == {"CANCELLED"}


def test_make_collection_ops_uses_get_settings_over_scene_attr():
    settings = SimpleNamespace(items=_Coll([SimpleNamespace()]), item_index=0)
    add_cls, _ = _make_item_ops(settings_attr=None, get_settings=lambda ctx: settings)
    # No scene attribute is consulted: any context object works.
    assert add_cls.execute(add_cls(), object()) == {"FINISHED"}
    assert len(settings.items) == 2
    assert settings.item_index == 1


def test_make_collection_ops_max_items_caps_with_warning():
    add_cls, _ = _make_item_ops(max_items=3, max_items_message="full")
    items = _Coll([SimpleNamespace(), SimpleNamespace(), SimpleNamespace()])
    context, settings = _coll_scene("vg_ride", items, 0)
    assert add_cls.execute(add_cls(), context) == {"CANCELLED"}
    assert len(items) == 3  # unchanged


def test_make_collection_ops_on_add_seeds_new_item():
    seen = {}

    def on_add(context, settings, item):
        item.tag = "seeded"
        seen["count"] = len(settings.items)

    add_cls, _ = _make_item_ops(on_add=on_add)
    items = _Coll()
    context, _ = _coll_scene("vg_ride", items, 0)
    add_cls.execute(add_cls(), context)
    assert items[0].tag == "seeded"
    assert seen["count"] == 1  # the new item is already appended when on_add runs


# ===========================================================================
# registration.register_classes / unregister_classes
# ===========================================================================


def test_register_classes_registers_in_order(monkeypatch):
    import bpy

    order = []
    monkeypatch.setattr(bpy.utils, "register_class", order.append)
    registration.register_classes(["a", "b", "c"])
    assert order == ["a", "b", "c"]


def test_unregister_classes_reverses_order(monkeypatch):
    import bpy

    order = []
    monkeypatch.setattr(bpy.utils, "unregister_class", order.append)
    registration.unregister_classes(["a", "b", "c"])
    assert order == ["c", "b", "a"]


# ===========================================================================
# props.register_shared_light / unregister_shared_light (cooperative)
# ===========================================================================


def test_register_shared_light_registers_and_owns(monkeypatch):
    import bpy

    registered, unregistered = [], []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.setattr(bpy.utils, "unregister_class", unregistered.append)
    monkeypatch.setattr(props.SharedLight, "is_registered", False, raising=False)
    monkeypatch.setattr(props, "_shared_light_owned", False, raising=False)

    props.register_shared_light()
    assert registered == [props.SharedLight]

    props.unregister_shared_light()
    assert unregistered == [props.SharedLight]


def test_register_shared_light_skips_when_another_addon_owns_it(monkeypatch):
    import bpy

    registered, unregistered = [], []
    monkeypatch.setattr(bpy.utils, "register_class", registered.append)
    monkeypatch.setattr(bpy.utils, "unregister_class", unregistered.append)
    # Another add-on already registered the shared class object.
    monkeypatch.setattr(props.SharedLight, "is_registered", True, raising=False)
    monkeypatch.setattr(props, "_shared_light_owned", False, raising=False)

    props.register_shared_light()
    assert registered == []  # skipped: not ours to register

    props.unregister_shared_light()
    assert unregistered == []  # and not ours to drop


# ===========================================================================
# lights_ui.draw_lights_rig
# ===========================================================================


class _LightsLayout:
    """Records prop names, operators, template_list ids, and label text."""

    def __init__(self, rec):
        self.rec = rec

    def box(self):
        return self

    def row(self, align=False):
        return self

    def column(self, align=False):
        return self

    def prop(self, data, name, **kw):
        self.rec.append(("prop", name))

    def label(self, *, text="", **kw):
        self.rec.append(("label", text))

    def operator(self, idname, **kw):
        self.rec.append(("operator", idname))

    def template_list(self, uilist, *a, **k):
        self.rec.append(("template_list", uilist))


def test_draw_lights_rig_collapsed_draws_only_header():
    rec = []
    settings = SimpleNamespace(show_lights=False, lights=[], light_index=0)
    lights_ui.draw_lights_rig(_LightsLayout(rec), settings, prefix="vg", uilist_name="VG_UL_lights")
    assert ("prop", "show_lights") in rec
    assert not any(kind == "operator" for kind, _ in rec)
    assert not any(kind == "template_list" for kind, _ in rec)


def test_draw_lights_rig_expanded_with_lights_draws_fields():
    rec = []
    settings = SimpleNamespace(show_lights=True, lights=[SimpleNamespace()], light_index=0)
    lights_ui.draw_lights_rig(
        _LightsLayout(rec), settings, prefix="vgs", uilist_name="VGS_UL_lights"
    )
    drawn = {name for kind, name in rec if kind == "prop"}
    assert {"show_lights", "type", "shadow", "direction", "strength"} <= drawn
    assert ("template_list", "VGS_UL_lights") in rec
    assert ("operator", "vgs.light_add") in rec
    assert ("operator", "vgs.light_remove") in rec


def test_draw_lights_rig_expanded_without_lights_shows_info():
    rec = []
    settings = SimpleNamespace(show_lights=True, lights=[], light_index=0)
    lights_ui.draw_lights_rig(
        _LightsLayout(rec), settings, prefix="vg", uilist_name="VG_UL_lights",
        info_text="nothing here",
    )
    assert ("label", "nothing here") in rec
    assert not any(name in ("type", "shadow") for kind, name in rec if kind == "prop")


# ===========================================================================
# modal.show_test_sprite
# ===========================================================================


class _ReportingOp:
    """Captures operator.report() calls as (level, message) tuples."""

    def __init__(self):
        self.reports = []

    def report(self, level, message):
        self.reports.append((next(iter(level)), message))


def _editor_context():
    """A context with one Image Editor area (plus an unrelated one)."""
    editor = SimpleNamespace(
        type="IMAGE_EDITOR", spaces=SimpleNamespace(active=SimpleNamespace(image=None))
    )
    other = SimpleNamespace(
        type="VIEW_3D", spaces=SimpleNamespace(active=SimpleNamespace(image=None))
    )
    return SimpleNamespace(screen=SimpleNamespace(areas=[other, editor])), editor


def test_show_test_sprite_no_png_reports_warning():
    op = _ReportingOp()
    context, _ = _editor_context()
    assert modal.show_test_sprite(op, context, None) == {"CANCELLED"}
    assert op.reports == [("WARNING", "Render produced no sprite")]


def test_show_test_sprite_missing_file_reports_warning(tmp_path):
    op = _ReportingOp()
    context, _ = _editor_context()
    assert modal.show_test_sprite(op, context, str(tmp_path / "nope.png")) == {"CANCELLED"}
    assert op.reports == [("WARNING", "Render produced no sprite")]


def test_show_test_sprite_loads_and_shows_in_image_editor(tmp_path, monkeypatch):
    png = tmp_path / "sprite.png"
    png.write_bytes(b"")
    img = SimpleNamespace(name="sprite.png")
    loaded = []
    monkeypatch.setattr(
        sys.modules["bpy"],
        "data",
        SimpleNamespace(
            images=SimpleNamespace(load=lambda path, check_existing: loaded.append(path) or img)
        ),
        raising=False,
    )
    op = _ReportingOp()
    context, editor = _editor_context()
    assert modal.show_test_sprite(op, context, str(png)) == {"FINISHED"}
    assert loaded == [str(png)]
    assert editor.spaces.active.image is img
    assert op.reports == [("INFO", "Test sprite loaded: sprite.png")]


# ===========================================================================
# mesh_extract.parse_authors
# ===========================================================================


def test_parse_authors_strips_and_drops_blanks():
    assert mesh_extract.parse_authors(" Alice , ,  Bob ") == ["Alice", "Bob"]


def test_parse_authors_empty_string_is_empty_list():
    assert mesh_extract.parse_authors("") == []


# ===========================================================================
# props.register_settings / unregister_settings
# ===========================================================================


def test_register_settings_registers_light_then_classes_then_pointers(monkeypatch):
    calls = []
    monkeypatch.setattr(props, "register_shared_light", lambda: calls.append("light"))
    monkeypatch.setattr(props, "register_classes", lambda cs: calls.append(("classes", tuple(cs))))
    owner = SimpleNamespace()
    classes = (object(),)
    pointers = ((owner, "foo", int), (owner, "bar", str))
    props.register_settings(classes, pointers)
    # SharedLight first, then the classes, then the pointer bindings.
    assert calls == ["light", ("classes", classes)]
    assert hasattr(owner, "foo")
    assert hasattr(owner, "bar")


def test_unregister_settings_deletes_pointers_then_classes_then_light(monkeypatch):
    calls = []
    monkeypatch.setattr(props, "unregister_shared_light", lambda: calls.append("light"))
    monkeypatch.setattr(props, "unregister_classes", lambda cs: calls.append("classes"))
    owner = SimpleNamespace(foo=1, bar=2)
    pointers = ((owner, "foo", int), (owner, "bar", str))
    props.unregister_settings((object(),), pointers)
    assert not hasattr(owner, "foo")
    assert not hasattr(owner, "bar")
    # Classes are dropped before SharedLight (reverse of register order).
    assert calls == ["classes", "light"]


# ===========================================================================
# progress_overlay.py
# ===========================================================================


def _install_fake_gpu(monkeypatch):
    """Install headless gpu/blf/gpu_extras fakes + reset the cached shader."""
    import types as _types

    drawn: list = []

    class _Shader:
        def uniform_float(self, name, color):
            pass

    class _Batch:
        def draw(self, shader):
            drawn.append("batch")

    gpu = _types.ModuleType("gpu")
    gpu.shader = SimpleNamespace(from_builtin=lambda name: _Shader())
    gpu.state = SimpleNamespace(blend_set=lambda mode: None)

    gpu_extras = _types.ModuleType("gpu_extras")
    gpu_extras_batch = _types.ModuleType("gpu_extras.batch")
    gpu_extras_batch.batch_for_shader = lambda *a, **k: _Batch()
    gpu_extras.batch = gpu_extras_batch

    blf = _types.ModuleType("blf")
    blf.size = lambda *a: None
    blf.color = lambda *a: None
    blf.position = lambda *a: None
    blf.draw = lambda *a: None

    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setitem(sys.modules, "gpu_extras", gpu_extras)
    monkeypatch.setitem(sys.modules, "gpu_extras.batch", gpu_extras_batch)
    monkeypatch.setitem(sys.modules, "blf", blf)
    # The shader is cached on first draw; reset so it rebuilds against the fake.
    monkeypatch.setattr(progress_overlay, "_shader", None)
    return drawn


def test_progress_overlay_add_remove_is_idempotent(monkeypatch):
    overlay = progress_overlay.ProgressOverlay()
    overlay.add()
    handle = overlay._handle
    assert handle is not None
    overlay.add()  # already added: handle is unchanged
    assert overlay._handle is handle
    overlay.remove()
    assert overlay._handle is None
    overlay.remove()  # already removed: no-op


def test_progress_overlay_tag_redraw_flags_only_view3d():
    overlay = progress_overlay.ProgressOverlay()
    redrawn = []
    v3d = SimpleNamespace(type="VIEW_3D", tag_redraw=lambda: redrawn.append("v3d"))
    other = SimpleNamespace(type="IMAGE_EDITOR", tag_redraw=lambda: redrawn.append("other"))
    ctx = SimpleNamespace(screen=SimpleNamespace(areas=[v3d, other]))
    overlay.tag_redraw(ctx)
    assert redrawn == ["v3d"]


def test_progress_overlay_draw_determinate_and_indeterminate(monkeypatch):
    import bpy

    drawn = _install_fake_gpu(monkeypatch)
    monkeypatch.setattr(
        bpy, "context", SimpleNamespace(region=SimpleNamespace(width=800, height=600)),
        raising=False,
    )
    overlay = progress_overlay.ProgressOverlay()

    overlay.total, overlay.done = 10, 5  # determinate: a filled bar + percent label
    overlay._draw()
    assert drawn

    drawn.clear()
    overlay.total, overlay.done = 0, 0  # indeterminate: a sliding chunk
    overlay._draw()
    assert drawn


def test_progress_overlay_draw_skips_tiny_or_missing_region(monkeypatch):
    import bpy

    _install_fake_gpu(monkeypatch)
    overlay = progress_overlay.ProgressOverlay()

    # No region at all: bail before any drawing.
    monkeypatch.setattr(bpy, "context", SimpleNamespace(region=None), raising=False)
    overlay._draw()

    # Region too small to fit the bar: also bail.
    monkeypatch.setattr(
        bpy, "context", SimpleNamespace(region=SimpleNamespace(width=10, height=10)),
        raising=False,
    )
    overlay._draw()
