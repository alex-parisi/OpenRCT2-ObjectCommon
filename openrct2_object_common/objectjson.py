"""
The object.json fields every OpenRCT2 object shares.

`id`, `originalId`, `version`, `authors`, and `objectType` are identical across
rides and all three scenery kinds; each generator's ``build_*_json`` opens with
this header, then fills in its kind-specific ``properties`` / ``strings``.
"""

from collections.abc import Iterable
from typing import Any

__all__ = ["object_json_header"]


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
