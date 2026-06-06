"""Tests for openrct2_object_common.config — shared config parsing helpers."""

import sys

import numpy as np
import pytest
from openrct2_object_common.config import (
    LoadError,
    as_array_or_wrap,
    load_meshes,
    load_preview,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    optional_string_list,
    parse_config,
    read_vector3,
    require_int,
    require_number,
    require_string,
)

# --------------------------------------------------------------------------
# parse_config
# --------------------------------------------------------------------------


def test_parse_config_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"key": "value"}')
    result = parse_config(p)
    assert result == {"key": "value"}


def test_parse_config_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("key: value\n")
    result = parse_config(p)
    assert result == {"key": "value"}


def test_parse_config_yml_extension(tmp_path):
    p = tmp_path / "cfg.yml"
    p.write_text("num: 42\n")
    result = parse_config(p)
    assert result == {"num": 42}


def test_parse_config_yaml_requires_pyyaml(tmp_path, monkeypatch):
    p = tmp_path / "cfg.yaml"
    p.write_text("key: value\n")
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(LoadError, match="PyYAML"):
        parse_config(p)


def test_parse_config_non_dict_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("[]")
    with pytest.raises(LoadError, match="not an object"):
        parse_config(p)


# --------------------------------------------------------------------------
# require_string
# --------------------------------------------------------------------------


def test_require_string_returns_value():
    assert require_string({"k": "hello"}, "k") == "hello"


def test_require_string_missing_raises():
    with pytest.raises(LoadError, match='"k"'):
        require_string({}, "k")


def test_require_string_non_string_raises():
    with pytest.raises(LoadError, match='"k"'):
        require_string({"k": 42}, "k")


# --------------------------------------------------------------------------
# optional_string
# --------------------------------------------------------------------------


def test_optional_string_absent_returns_default():
    assert optional_string({}, "k") == ""
    assert optional_string({}, "k", "def") == "def"


def test_optional_string_present_returns_value():
    assert optional_string({"k": "hi"}, "k") == "hi"


def test_optional_string_non_string_raises():
    with pytest.raises(LoadError, match='"k" is not a string'):
        optional_string({"k": 5}, "k")


# --------------------------------------------------------------------------
# optional_string_list
# --------------------------------------------------------------------------


def test_optional_string_list_absent_returns_empty():
    assert optional_string_list({}, "k") == []


def test_optional_string_list_single_string_wraps():
    assert optional_string_list({"k": "one"}, "k") == ["one"]


def test_optional_string_list_list_returned():
    assert optional_string_list({"k": ["a", "b"]}, "k") == ["a", "b"]


def test_optional_string_list_invalid_raises():
    with pytest.raises(LoadError):
        optional_string_list({"k": [1, 2]}, "k")

    with pytest.raises(LoadError):
        optional_string_list({"k": 42}, "k")


# --------------------------------------------------------------------------
# require_int
# --------------------------------------------------------------------------


def test_require_int_returns_value():
    assert require_int({"k": 7}, "k") == 7


def test_require_int_missing_raises():
    with pytest.raises(LoadError, match='"k"'):
        require_int({}, "k")


def test_require_int_bool_raises():
    with pytest.raises(LoadError):
        require_int({"k": True}, "k")


def test_require_int_float_raises():
    with pytest.raises(LoadError):
        require_int({"k": 3.5}, "k")


# --------------------------------------------------------------------------
# optional_int
# --------------------------------------------------------------------------


def test_optional_int_absent_returns_default():
    assert optional_int({}, "k", 99) == 99


def test_optional_int_present_returns_value():
    assert optional_int({"k": 5}, "k", 0) == 5


def test_optional_int_bool_raises():
    with pytest.raises(LoadError):
        optional_int({"k": False}, "k", 0)


# --------------------------------------------------------------------------
# require_number
# --------------------------------------------------------------------------


def test_require_number_int_coerced_to_float():
    assert require_number({"k": 3}, "k") == 3.0
    assert isinstance(require_number({"k": 3}, "k"), float)


def test_require_number_float_returned():
    assert require_number({"k": 1.5}, "k") == 1.5


def test_require_number_missing_raises():
    with pytest.raises(LoadError, match='"k"'):
        require_number({}, "k")


def test_require_number_bool_raises():
    with pytest.raises(LoadError):
        require_number({"k": True}, "k")


# --------------------------------------------------------------------------
# optional_number
# --------------------------------------------------------------------------


def test_optional_number_absent_returns_default():
    assert optional_number({}, "k", 1.0) == 1.0


def test_optional_number_present_returns_float():
    assert optional_number({"k": 2}, "k", 0.0) == 2.0


def test_optional_number_bool_raises():
    with pytest.raises(LoadError):
        optional_number({"k": True}, "k", 0.0)


# --------------------------------------------------------------------------
# optional_bool
# --------------------------------------------------------------------------


def test_optional_bool_absent_returns_default():
    assert optional_bool({}, "k") is False
    assert optional_bool({}, "k", True) is True


def test_optional_bool_present_returns_value():
    assert optional_bool({"k": True}, "k") is True
    assert optional_bool({"k": False}, "k") is False


def test_optional_bool_non_bool_raises():
    with pytest.raises(LoadError):
        optional_bool({"k": "yes"}, "k")


# --------------------------------------------------------------------------
# read_vector3
# --------------------------------------------------------------------------


def test_read_vector3_valid():
    v = read_vector3([1.0, 2.0, 3.0])
    assert isinstance(v, np.ndarray)
    np.testing.assert_allclose(v, [1.0, 2.0, 3.0])


def test_read_vector3_wrong_length_raises():
    with pytest.raises(LoadError, match="3 numbers"):
        read_vector3([1.0, 2.0])


def test_read_vector3_not_list_raises():
    with pytest.raises(LoadError, match="3 numbers"):
        read_vector3("bad")


def test_read_vector3_non_numeric_raises():
    with pytest.raises(LoadError, match="not a number"):
        read_vector3([1.0, "x", 3.0])


# --------------------------------------------------------------------------
# as_array_or_wrap
# --------------------------------------------------------------------------


def test_as_array_or_wrap_none_raises():
    with pytest.raises(LoadError, match="Missing"):
        as_array_or_wrap(None)


def test_as_array_or_wrap_empty_list_raises():
    with pytest.raises(LoadError, match="Empty"):
        as_array_or_wrap([])


def test_as_array_or_wrap_list_returned_as_is():
    lst = [1, 2, 3]
    assert as_array_or_wrap(lst) is lst


def test_as_array_or_wrap_scalar_wrapped():
    assert as_array_or_wrap(42) == [42]
    assert as_array_or_wrap("hello") == ["hello"]


# --------------------------------------------------------------------------
# load_meshes
# --------------------------------------------------------------------------

_TRI = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def test_load_meshes_loads_obj_files(tmp_path):
    (tmp_path / "m.obj").write_text(_TRI)
    meshes = load_meshes({"meshes": [str(tmp_path / "m.obj")]})
    assert len(meshes) == 1


def test_load_meshes_with_base_dir_resolves_relative(tmp_path):
    (tmp_path / "a.obj").write_text(_TRI)
    meshes = load_meshes({"meshes": ["a.obj"]}, base_dir=tmp_path)
    assert len(meshes) == 1


def test_load_meshes_absolute_path_ignores_base_dir(tmp_path):
    obj_path = tmp_path / "b.obj"
    obj_path.write_text(_TRI)
    meshes = load_meshes({"meshes": [str(obj_path)]}, base_dir=tmp_path / "other")
    assert len(meshes) == 1


def test_load_meshes_missing_key_raises():
    with pytest.raises(LoadError, match='"meshes"'):
        load_meshes({})


def test_load_meshes_non_string_path_raises():
    with pytest.raises(LoadError, match="not a string"):
        load_meshes({"meshes": [42]})


# --------------------------------------------------------------------------
# load_preview
# --------------------------------------------------------------------------


def test_load_preview_absent_returns_none():
    assert load_preview({}) is None


def test_load_preview_loads_png(tmp_path):
    from openrct2_x7_renderer.image import write_png
    from openrct2_x7_renderer.types import IndexedImage

    img = IndexedImage.blank(2, 2)
    png_path = tmp_path / "preview.png"
    write_png(img, png_path)

    result = load_preview({"preview": str(png_path)})
    assert result is not None
    assert result.width == 2


def test_load_preview_with_base_dir(tmp_path):
    from openrct2_x7_renderer.image import write_png
    from openrct2_x7_renderer.types import IndexedImage

    write_png(IndexedImage.blank(1, 1), tmp_path / "p.png")
    result = load_preview({"preview": "p.png"}, base_dir=tmp_path)
    assert result is not None


def test_load_preview_non_string_raises():
    with pytest.raises(LoadError, match='"preview" is not a string'):
        load_preview({"preview": 42})


def test_load_preview_missing_file_raises(tmp_path):
    with pytest.raises(LoadError, match="Unable to open"):
        load_preview({"preview": str(tmp_path / "no.png")})
