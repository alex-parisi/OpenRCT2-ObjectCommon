"""
Shared config-loading helpers for object generators.

Every generator's loader opens the same way: read the render scale
(``units_per_tile``), fill in the identity header (``id`` / ``original_id`` /
``name`` / ``authors`` / ``version``), validate the ``object_type``
discriminator, resolve relative ``meshes`` / ``preview`` paths against the
config file's directory, and parse the ``model`` placement list. These helpers
own that shared preamble so each loader only handles its kind-specific fields
(and its own preview-fallback policy).
"""

from collections.abc import Callable, Container, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from openrct2_x7_renderer.constants import TILE_SIZE
from openrct2_x7_renderer.types import MeshFrame, Model

from .colours import COLOR_NAMES
from .config import (
    LoadError,
    as_array_or_wrap,
    load_meshes,
    load_preview,
    optional_number,
    optional_string,
    optional_string_list,
    parse_config,
    read_vector3,
    require_string,
)

__all__ = [
    "IdentityTarget",
    "apply_identity",
    "config_dir",
    "enum_index",
    "flag_bits",
    "load_colour_presets",
    "load_object",
    "load_single_frame_model",
    "load_units_per_tile",
    "object_type_of",
    "parse_single_frame_model",
    "require_choice",
    "validate_mesh_index",
]

T = TypeVar("T")


def load_units_per_tile(root: dict[str, Any]) -> float:
    """Render scale: OBJ units per tile. Defaults to RCT2's real-world tile."""
    upt = optional_number(root, "units_per_tile", TILE_SIZE)
    if upt <= 0.0:
        raise LoadError('Property "units_per_tile" must be greater than 0')
    return upt


class IdentityTarget(Protocol):
    """The identity + render-scale fields every generated object carries."""

    id: str
    original_id: str
    name: str
    authors: list[str]
    version: str
    units_per_tile: float


def apply_identity(obj: IdentityTarget, root: dict[str, Any]) -> None:
    """Populate the identity + render-scale fields shared by every object kind.

    ``version`` is overwritten only when the config supplies one, so the
    target's own default stands otherwise. The preview image and any
    kind-specific fields remain the caller's responsibility.
    """
    obj.id = require_string(root, "id")
    obj.original_id = optional_string(root, "original_id")
    obj.name = require_string(root, "name")
    obj.authors = optional_string_list(root, "authors")
    version = optional_string(root, "version")
    if version:
        obj.version = version
    obj.units_per_tile = load_units_per_tile(root)


def config_dir(json_path: Path | str) -> Path:
    """The directory containing the config file; relative ``meshes`` /
    ``preview`` paths resolve against it."""
    return Path(json_path).parent


def load_object(
    json_path: Path | str, build: Callable[..., T], *, with_meshes: bool = True
) -> T:
    """Parse a config file and hand it to a kind-specific ``build`` function.

    Owns the loader preamble every generator repeats: parse the config, then
    load its meshes / preview from paths resolved against the config's directory.
    ``build`` is called as ``build(root, meshes, preview)`` for a geometry object,
    or ``build(root, preview)`` when ``with_meshes`` is false (e.g. a scenery
    group, which carries only a tab icon).
    """
    root = parse_config(json_path)
    base = config_dir(json_path)
    preview = load_preview(root, base)
    if with_meshes:
        return build(root, load_meshes(root, base), preview)
    return build(root, preview)


def object_type_of(
    config: dict[str, Any], allowed: Container[str], *, default: str
) -> str:
    """Read and validate the ``object_type`` discriminator.

    Defaults to ``default`` when absent; raises :class:`LoadError` for any value
    not in ``allowed``.
    """
    return require_choice(
        optional_string(config, "object_type", default), allowed, "object_type"
    )


def require_choice(
    value: T, allowed: Container[T], label: str, *, expected: object = None
) -> T:
    """Return ``value`` if it is in ``allowed``, else raise :class:`LoadError`.

    The message is ``Unrecognized <label> "<value>"``; pass ``expected`` (the
    permitted values, in whatever order/shape the caller wants shown) to append
    ``(expected one of <expected>)``. Centralizes the ``object_type`` / shape /
    ride_type / shop-item / colour checks scattered across the loaders.
    """
    if value not in allowed:
        hint = "" if expected is None else f" (expected one of {expected})"
        raise LoadError(f'Unrecognized {label} "{value}"{hint}')
    return value


def enum_index(value: Any, names: Sequence[str], prop: str, label: str) -> int:
    """The index of a config string within an ordered ``names`` list.

    Used where a name's position is its engine value (a sound id, a colour
    index, ...). Raises :class:`LoadError` if ``value`` is not a string
    (``Property "<prop>" not found or is not a string``) or is not in ``names``
    (``Unrecognized <label> "<value>"``).
    """
    if not isinstance(value, str):
        raise LoadError(f'Property "{prop}" not found or is not a string')
    require_choice(value, names, label)
    return names.index(value)


def flag_bits(value: Any, names: Sequence[str], prop: str, label: str) -> int:
    """OR together ``1 << names.index(tag)`` for each config string in a list.

    The bit a tag sets is its position in ``names`` (the loader's flag order).
    Raises :class:`LoadError` for a non-list ``value``, a non-string element, or
    a tag not in ``names``.
    """
    if not isinstance(value, list):
        raise LoadError(f'Property "{prop}" not found or is not an array')
    flags = 0
    for tag in value:
        if not isinstance(tag, str):
            raise LoadError(f'Array "{prop}" contains non-string value')
        require_choice(tag, names, label)
        flags |= 1 << names.index(tag)
    return flags


def validate_mesh_index(value: Any, num_meshes: int) -> int:
    """Validate a placement ``mesh_index``.

    Must be an integer (not a bool) in ``-1 .. num_meshes - 1``; ``-1`` marks an
    empty slot. Raises :class:`LoadError` for a missing/non-integer value or one
    out of range.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise LoadError('Property "mesh_index" not found or is not an integer')
    if value >= num_meshes or value < -1:
        raise LoadError(f"Mesh index {value} is out of bounds")
    return int(value)


def parse_single_frame_model(
    value: Any, num_meshes: int
) -> list[tuple[list[MeshFrame], dict[str, Any]]]:
    """Parse a single-frame ``model`` placement list into ``(frame, element)`` pairs.

    A lone placement object is wrapped into a one-element list. Each element must
    be an object with a valid ``mesh_index`` (:func:`validate_mesh_index`) and
    optional ``position`` / ``orientation`` vectors; an absent vector falls back
    to the ``MeshFrame`` zero-vector default. ``frame`` is the single-element
    ``[MeshFrame]`` list ready to drop into ``Model.meshes``; the source element
    dict is returned alongside it so callers can read kind-specific flags (e.g. a
    facility's ``door``). The same ``frame`` object can be shared between models,
    so a caller's two models can reference one placement by identity.
    """
    if value is None:
        raise LoadError('Property "model" not found')
    out: list[tuple[list[MeshFrame], dict[str, Any]]] = []
    for elem in as_array_or_wrap(value):
        if not isinstance(elem, dict):
            raise LoadError('Property "model" is not an object')
        kwargs: dict[str, Any] = {
            "mesh_index": validate_mesh_index(elem.get("mesh_index"), num_meshes)
        }
        for key in ("position", "orientation"):
            prop = elem.get(key)
            if prop is not None:
                kwargs[key] = read_vector3(prop)
        out.append(([MeshFrame(**kwargs)], elem))
    return out


def load_single_frame_model(value: Any, num_meshes: int) -> Model:
    """Parse a single-frame ``model`` placement list into a :class:`Model`.

    The common case over :func:`parse_single_frame_model`: every placement
    becomes one mesh entry, discarding the source element (callers that need a
    per-placement flag -- e.g. a stall facility's ``door`` subset -- iterate
    :func:`parse_single_frame_model` directly instead).
    """
    return Model(meshes=[frame for frame, _ in parse_single_frame_model(value, num_meshes)])


def load_colour_presets(
    value: Any,
    prop: str,
    *,
    label: str = "colour",
    default: list[list[int]] | None = None,
    allow_empty: bool = True,
    require_triple: bool = False,
) -> list[list[int]]:
    """Parse a list of ``[main, additional1, additional2]`` colour-name presets
    into index triples (each colour's position in :data:`COLOR_NAMES`).

    Every object that paints with the remap palette (vehicle ``default_colors``,
    stall ``car_colours``) names its preset colours from ``COLOR_NAMES``; this is
    the shared validation. ``value`` of ``None`` returns ``default`` (a copy) when
    one is given, else raises. Each preset must be a list of ``COLOR_NAMES``
    strings; with ``require_triple`` it must list exactly three, otherwise it is
    padded/truncated to three (missing slots default to index ``0``). With
    ``allow_empty`` cleared an empty preset list is rejected.
    """
    if value is None:
        if default is not None:
            return [list(triple) for triple in default]
        raise LoadError(f'Property "{prop}" not found or is not an array')
    if not isinstance(value, list) or (not allow_empty and not value):
        article = "" if allow_empty else " non-empty"
        raise LoadError(f'Property "{prop}" is not a{article} array')
    presets: list[list[int]] = []
    for preset in value:
        if not isinstance(preset, list):
            raise LoadError(f'Property "{prop}" contains an element which is not an array')
        if require_triple and len(preset) != 3:
            raise LoadError(
                f'Each "{prop}" preset must be a [main, additional1, additional2] triple'
            )
        triple = [0, 0, 0]
        for j, colour in enumerate(preset[:3]):
            triple[j] = enum_index(colour, COLOR_NAMES, prop, label)
        presets.append(triple)
    return presets
