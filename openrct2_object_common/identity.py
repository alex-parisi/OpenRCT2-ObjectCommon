"""
The identity + render-scale fields every generated object dataclass shares.

Every generator's object (the vehicle ``Ride``, the scenery kinds, the stall,
the track) opens with the same id / original_id / name / authors / version /
units_per_tile block. ``ObjectIdentity`` owns those as a base dataclass so the
fields and their defaults live in one place and :func:`apply_identity` can
populate any object uniformly; each kind subclasses it and declares only its own
fields. It satisfies the structural ``IdentityTarget`` protocol.
"""

from dataclasses import dataclass, field

from openrct2_x7_renderer.constants import TILE_SIZE

__all__ = ["ObjectIdentity"]


@dataclass
class ObjectIdentity:
    """Base dataclass carrying the shared identity + render-scale fields.

    Subclasses must keep all of their own fields defaulted too: these are
    defaulted, and dataclass inheritance forbids a non-default field following a
    defaulted one.
    """

    id: str = ""
    original_id: str = ""
    name: str = ""
    authors: list[str] = field(default_factory=list)
    version: str = "1.0"

    # Model units per tile: the scale mapping OBJ-space units onto one OpenRCT2
    # tile. Drives both the render projection (sprite size) and the exporter's
    # model->game-unit conversions, so they always agree. Default matches the
    # realistic ~3.3 m tile.
    units_per_tile: float = TILE_SIZE
