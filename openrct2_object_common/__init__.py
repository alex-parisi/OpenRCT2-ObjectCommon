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

from .cli import make_context, run_cli
from .config import LoadError, load_meshes, load_preview, parse_config
from .objectjson import object_json_header
from .parkobj import assemble_parkobj, write_images_dat_lgx
from .placement import add_model_to_scene

__all__ = [
    "LoadError",
    "add_model_to_scene",
    "assemble_parkobj",
    "load_meshes",
    "load_preview",
    "make_context",
    "object_json_header",
    "parse_config",
    "run_cli",
    "write_images_dat_lgx",
]
