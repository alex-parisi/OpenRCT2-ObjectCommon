"""
Shared scaffolding for OpenRCT2 object generators: config parsing/validation,
CLI flow, model placement, object.json headers, and `.parkobj` assembly. Sits
between the renderer (`openrct2-x7-renderer`) and the generators, so the vehicle
and scenery tools share one config layer and one packaging path.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("OpenRCT2-ObjectCommon")
except PackageNotFoundError:  # pragma: no cover - source tree without an install
    __version__ = "0.0.0"
