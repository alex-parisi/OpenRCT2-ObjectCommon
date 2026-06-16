"""Tests for the shared config-loading helpers.

These are pure dict/path parsers; ``apply_identity`` is exercised against a
minimal dataclass that satisfies the ``IdentityTarget`` protocol.
"""

from dataclasses import dataclass, field

import pytest
from openrct2_object_common import loading as ld
from openrct2_object_common.colours import COLOR_NAMES
from openrct2_object_common.config import LoadError
from openrct2_object_common.loading import (
    apply_identity,
    config_dir,
    enum_index,
    flag_bits,
    load_colour_presets,
    load_object,
    load_single_frame_model,
    load_units_per_tile,
    object_type_of,
    parse_single_frame_model,
    require_choice,
    validate_mesh_index,
)
from openrct2_x7_renderer.constants import TILE_SIZE


@dataclass
class _Target:
    """A stand-in object carrying the shared identity + render-scale fields."""

    id: str = ""
    original_id: str = ""
    name: str = ""
    authors: list[str] = field(default_factory=list)
    version: str = "1.0"
    units_per_tile: float = TILE_SIZE


def test_load_units_per_tile_defaults_to_tile_size():
    assert load_units_per_tile({}) == TILE_SIZE


def test_load_units_per_tile_reads_override():
    assert load_units_per_tile({"units_per_tile": 8.0}) == 8.0


def test_load_units_per_tile_rejects_non_positive():
    for bad in (0.0, -1.0):
        with pytest.raises(LoadError, match="greater than 0"):
            load_units_per_tile({"units_per_tile": bad})


def test_apply_identity_populates_shared_fields():
    obj = _Target()
    apply_identity(
        obj,
        {
            "id": "scenery.test",
            "original_id": "ORIG",
            "name": "Test",
            "authors": ["alex"],
            "version": "2.3",
            "units_per_tile": 16.0,
        },
    )
    assert obj.id == "scenery.test"
    assert obj.original_id == "ORIG"
    assert obj.name == "Test"
    assert obj.authors == ["alex"]
    assert obj.version == "2.3"
    assert obj.units_per_tile == 16.0


def test_apply_identity_keeps_target_default_version_when_absent():
    obj = _Target(version="9.9")
    apply_identity(obj, {"id": "a", "name": "n"})
    assert obj.version == "9.9"
    assert obj.original_id == ""


def test_apply_identity_requires_id_and_name():
    with pytest.raises(LoadError):
        apply_identity(_Target(), {"name": "n"})
    with pytest.raises(LoadError):
        apply_identity(_Target(), {"id": "a"})


def test_config_dir_returns_parent(tmp_path):
    cfg = tmp_path / "sub" / "object.yaml"
    assert config_dir(cfg) == tmp_path / "sub"
    assert config_dir(str(cfg)) == tmp_path / "sub"


def test_object_type_of_defaults_and_validates():
    allowed = ("scenery_small", "scenery_large")
    assert object_type_of({}, allowed, default="scenery_small") == "scenery_small"
    assert (
        object_type_of({"object_type": "scenery_large"}, allowed, default="scenery_small")
        == "scenery_large"
    )
    with pytest.raises(LoadError, match="Unrecognized object_type"):
        object_type_of({"object_type": "scenery_huge"}, allowed, default="scenery_small")


def test_validate_mesh_index_accepts_in_range_and_empty_slot():
    assert validate_mesh_index(0, 3) == 0
    assert validate_mesh_index(2, 3) == 2
    assert validate_mesh_index(-1, 3) == -1  # -1 marks an empty slot


@pytest.mark.parametrize("bad", [None, True, "0", 1.0])
def test_validate_mesh_index_rejects_non_integer(bad):
    with pytest.raises(LoadError, match="not found or is not an integer"):
        validate_mesh_index(bad, 3)


@pytest.mark.parametrize("bad", [3, 4, -2])
def test_validate_mesh_index_rejects_out_of_range(bad):
    with pytest.raises(LoadError, match="out of bounds"):
        validate_mesh_index(bad, 3)


def test_parse_single_frame_model_missing_is_load_error():
    with pytest.raises(LoadError, match='"model" not found'):
        parse_single_frame_model(None, 3)


def test_parse_single_frame_model_non_object_element_rejected():
    with pytest.raises(LoadError, match='"model" is not an object'):
        parse_single_frame_model([5], 3)


def test_parse_single_frame_model_wraps_lone_object():
    parsed = parse_single_frame_model({"mesh_index": 1}, 3)
    assert len(parsed) == 1
    (frame, elem) = parsed[0]
    assert len(frame) == 1  # one MeshFrame per single-frame placement
    assert frame[0].mesh_index == 1
    assert elem == {"mesh_index": 1}


def test_parse_single_frame_model_reads_vectors_and_defaults():
    parsed = parse_single_frame_model(
        [
            {"mesh_index": 0, "position": [1, 2, 3], "orientation": [0, 90, 0]},
            {"mesh_index": -1},
        ],
        3,
    )
    (frame0, _), (frame1, _) = parsed
    assert list(frame0[0].position) == [1, 2, 3]
    assert list(frame0[0].orientation) == [0, 90, 0]
    # Absent position/orientation fall back to the MeshFrame zero-vector default.
    assert list(frame1[0].position) == [0, 0, 0]
    assert list(frame1[0].orientation) == [0, 0, 0]


def test_parse_single_frame_model_returns_source_element():
    parsed = parse_single_frame_model([{"mesh_index": 0, "door": True}], 3)
    (_frame, elem) = parsed[0]
    assert elem["door"] is True


def test_load_single_frame_model_discards_source_elements():
    model = load_single_frame_model([{"mesh_index": 0}, {"mesh_index": 1}], 3)
    # One MeshFrame per placement; the source dict is dropped (vs parse_*).
    assert [frame[0].mesh_index for frame in model.meshes] == [0, 1]


def test_load_colour_presets_none_returns_default_copy():
    default = [[0, 0, 0]]
    out = load_colour_presets(None, "car_colours", default=default)
    assert out == [[0, 0, 0]]
    # A fresh copy, so callers can mutate without touching the default.
    assert out is not default
    assert out[0] is not default[0]


def test_load_colour_presets_none_without_default_raises():
    with pytest.raises(LoadError, match='"car_colours" not found or is not an array'):
        load_colour_presets(None, "car_colours")


def test_load_colour_presets_rejects_non_array():
    with pytest.raises(LoadError, match='"p" is not a array'):
        load_colour_presets("nope", "p")


def test_load_colour_presets_rejects_empty_when_disallowed():
    with pytest.raises(LoadError, match='"p" is not a non-empty array'):
        load_colour_presets([], "p", allow_empty=False)


def test_load_colour_presets_rejects_non_array_element():
    with pytest.raises(LoadError, match='"p" contains an element which is not an array'):
        load_colour_presets(["x"], "p")


def test_load_colour_presets_require_triple_rejects_wrong_length():
    with pytest.raises(LoadError, match="triple"):
        load_colour_presets([[COLOR_NAMES[0]]], "p", require_triple=True)


def test_load_colour_presets_maps_names_to_indices_and_pads():
    # A short preset is padded to three slots; missing slots default to index 0.
    out = load_colour_presets([[COLOR_NAMES[1]]], "p")
    assert out == [[1, 0, 0]]


def test_require_choice_returns_value_when_allowed():
    assert require_choice("a", ["a", "b"], "thing") == "a"


def test_require_choice_rejects_with_plain_message():
    with pytest.raises(LoadError, match=r'Unrecognized thing "z"$'):
        require_choice("z", ["a", "b"], "thing")


def test_require_choice_appends_expected_hint():
    with pytest.raises(LoadError, match=r'Unrecognized thing "z" \(expected one of \[1, 2\]\)'):
        require_choice("z", ["a"], "thing", expected=[1, 2])


def test_enum_index_returns_position():
    assert enum_index("b", ["a", "b", "c"], "p", "label") == 1


def test_enum_index_rejects_non_string():
    with pytest.raises(LoadError, match=r'Property "p" not found or is not a string'):
        enum_index(3, ["a"], "p", "label")


def test_enum_index_rejects_unknown_value():
    with pytest.raises(LoadError, match=r'Unrecognized label "z"'):
        enum_index("z", ["a"], "p", "label")


def test_flag_bits_ors_positions():
    assert flag_bits(["a", "c"], ["a", "b", "c"], "p", "flag") == 0b101


def test_flag_bits_rejects_non_list():
    with pytest.raises(LoadError, match=r'Property "p" not found or is not an array'):
        flag_bits("a", ["a"], "p", "flag")


def test_flag_bits_rejects_non_string_element():
    with pytest.raises(LoadError, match=r'Array "p" contains non-string value'):
        flag_bits([1], ["a"], "p", "flag")


def test_flag_bits_rejects_unknown_tag():
    with pytest.raises(LoadError, match=r'Unrecognized flag "z"'):
        flag_bits(["z"], ["a"], "p", "flag")


def test_load_object_passes_root_meshes_preview(monkeypatch, tmp_path):
    monkeypatch.setattr(ld, "parse_config", lambda _p: {"k": "v"})
    monkeypatch.setattr(ld, "load_meshes", lambda root, base: ["mesh"])
    monkeypatch.setattr(ld, "load_preview", lambda root, base: "preview")

    seen = {}

    def build(root, meshes, preview):
        seen.update(root=root, meshes=meshes, preview=preview)
        return "built"

    assert load_object(tmp_path / "x.json", build) == "built"
    assert seen == {"root": {"k": "v"}, "meshes": ["mesh"], "preview": "preview"}


def test_load_object_without_meshes_skips_mesh_load(monkeypatch, tmp_path):
    monkeypatch.setattr(ld, "parse_config", lambda _p: {"k": "v"})
    monkeypatch.setattr(ld, "load_preview", lambda root, base: "preview")

    def fail_meshes(*_a, **_k):  # pragma: no cover - must not be called
        raise AssertionError("load_meshes should not run when with_meshes=False")

    monkeypatch.setattr(ld, "load_meshes", fail_meshes)

    seen = {}

    def build(root, preview):
        seen.update(root=root, preview=preview)
        return "group"

    assert load_object(tmp_path / "g.json", build, with_meshes=False) == "group"
    assert seen == {"root": {"k": "v"}, "preview": "preview"}
