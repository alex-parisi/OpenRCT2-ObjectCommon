"""Tests for the shared object.json header."""

from dataclasses import dataclass, field

from openrct2_object_common.objectjson import (
    object_json_header,
    object_json_header_for,
    object_strings,
)


@dataclass
class _Obj:
    id: str = "rct2.foo"
    original_id: str = ""
    version: str = "1.0"
    authors: list[str] = field(default_factory=list)


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


def test_strings_name_only():
    assert object_strings("My Object") == {"name": {"en-GB": "My Object"}}


def test_strings_emits_optional_fields_in_canonical_order():
    out = object_strings("Name", description="Desc", capacity="5 guests")
    assert list(out.keys()) == ["name", "description", "capacity"]
    assert out == {
        "name": {"en-GB": "Name"},
        "description": {"en-GB": "Desc"},
        "capacity": {"en-GB": "5 guests"},
    }


def test_strings_omits_unset_optionals():
    # Description without capacity, and vice versa, each appear alone.
    assert "capacity" not in object_strings("N", description="D")
    assert "description" not in object_strings("N", capacity="C")


def test_header_for_sources_identity_fields():
    obj = _Obj(id="rct2.bar", original_id="aa|BAR|bb", version="2.0", authors=["Alex"])
    out = object_json_header_for(obj, "scenery_small")
    assert out == {
        "id": "rct2.bar",
        "originalId": "aa|BAR|bb",
        "version": "2.0",
        "authors": ["Alex"],
        "objectType": "scenery_small",
    }


def test_header_for_omits_empty_original_id():
    out = object_json_header_for(_Obj(), "ride")
    assert "originalId" not in out
    assert out["objectType"] == "ride"
