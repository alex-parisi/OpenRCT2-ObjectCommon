"""Tests for the shared object.json header."""

from openrct2_object_common.objectjson import object_json_header


def test_header_key_order_and_defaults():
    out = object_json_header("rct2.foo", object_type="scenery_small")
    # Canonical key order, no originalId when empty, authors defaults to [].
    assert list(out.keys()) == ["id", "version", "authors", "objectType"]
    assert out == {
        "id": "rct2.foo",
        "version": "1.0",
        "authors": [],
        "objectType": "scenery_small",
    }


def test_header_includes_original_id_when_set():
    out = object_json_header(
        "rct2.foo",
        object_type="ride",
        original_id="aa|BAR|bb",
        version="2.3",
        authors=("Alex", "Chris"),
    )
    assert list(out.keys()) == ["id", "originalId", "version", "authors", "objectType"]
    assert out["originalId"] == "aa|BAR|bb"
    assert out["version"] == "2.3"
    assert out["authors"] == ["Alex", "Chris"]


def test_authors_is_copied_not_aliased():
    src = ["Alex"]
    out = object_json_header("x", object_type="ride", authors=src)
    out["authors"].append("Mallory")
    assert src == ["Alex"]
