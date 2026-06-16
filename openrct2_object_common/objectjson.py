"""
The object.json fields every OpenRCT2 object shares.

`id`, `originalId`, `version`, `authors`, and `objectType` are identical across
rides and all three scenery kinds; each generator's ``build_*_json`` opens with
this header, then fills in its kind-specific ``properties`` / ``strings``.
"""

from collections.abc import Iterable
from typing import Any, Protocol

__all__ = ["object_json_header", "object_json_header_for", "object_strings"]


class HeaderSource(Protocol):
    """The identity fields :func:`object_json_header_for` reads off an object.

    Every generated object dataclass carries these (see
    :class:`openrct2_object_common.loading.IdentityTarget`)."""

    id: str
    original_id: str
    version: str
    authors: list[str]


def object_strings(
    name: str, *, description: str | None = None, capacity: str | None = None
) -> dict[str, dict[str, str]]:
    """The object.json ``strings`` block, in OpenRCT2's canonical key order.

    Every object names itself; ``description`` and ``capacity`` are emitted only
    when supplied (the en-GB locale is the single authored language). Returned as
    its own dict so the caller assigns it to ``obj_json["strings"]``.
    """
    out: dict[str, dict[str, str]] = {"name": {"en-GB": name}}
    if description is not None:
        out["description"] = {"en-GB": description}
    if capacity is not None:
        out["capacity"] = {"en-GB": capacity}
    return out


def object_json_header(
    obj_id: str,
    *,
    object_type: str,
    original_id: str = "",
    version: str = "1.0",
    authors: Iterable[str] = (),
) -> dict[str, Any]:
    """The shared leading object.json fields, in OpenRCT2's canonical key order.

    ``originalId`` is omitted when empty (OpenRCT2 treats it as absent); the
    caller adds ``properties``/``strings``/``images`` to the returned dict.
    """
    out: dict[str, Any] = {"id": obj_id}
    if original_id:
        out["originalId"] = original_id
    out["version"] = version
    out["authors"] = list(authors)
    out["objectType"] = object_type
    return out


def object_json_header_for(obj: HeaderSource, object_type: str) -> dict[str, Any]:
    """:func:`object_json_header` sourced from an object's identity fields.

    Every generator's ``build_*_json`` opens the same way -- the id / originalId /
    version / authors come straight off the loaded object -- so this binds them in
    one place; callers pass only the object and its ``objectType``.
    """
    return object_json_header(
        obj.id,
        object_type=object_type,
        original_id=obj.original_id,
        version=obj.version,
        authors=obj.authors,
    )
