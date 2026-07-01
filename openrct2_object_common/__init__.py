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
from .colours import COLOR_NAMES
from .config import LoadError, load_meshes, load_preview, parse_config
from .dispatch import run_dispatch_cli
from .export import (
    ExportTo,
    ProgressFn,
    RenderSprites,
    export_object,
    export_to_directory,
    output_paths,
)
from .identity import ObjectIdentity
from .loading import (
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
from .objectjson import object_json_header, object_json_header_for, object_strings
from .parkobj import assemble_parkobj, parkobj_filename, write_images_dat_lgx
from .placement import add_model_to_scene
from .preview import open_test_dir, write_combined_preview
from .sprite_render import (
    add_split_ghost,
    center_in_box,
    corner_anchors,
    fit_in_box,
    render_corner_anchored_rotations,
    render_corner_anchored_view,
    render_scene_view,
    render_scene_views,
    trim,
)

__all__ = [
    "COLOR_NAMES",
    "ExportTo",
    "LoadError",
    "ObjectIdentity",
    "ProgressFn",
    "RenderSprites",
    "add_model_to_scene",
    "add_split_ghost",
    "apply_identity",
    "assemble_parkobj",
    "center_in_box",
    "config_dir",
    "corner_anchors",
    "enum_index",
    "export_object",
    "export_to_directory",
    "fit_in_box",
    "flag_bits",
    "load_colour_presets",
    "load_meshes",
    "load_object",
    "load_preview",
    "load_single_frame_model",
    "load_units_per_tile",
    "make_context",
    "object_json_header",
    "object_json_header_for",
    "object_strings",
    "object_type_of",
    "open_test_dir",
    "output_paths",
    "parkobj_filename",
    "parse_config",
    "parse_single_frame_model",
    "render_corner_anchored_rotations",
    "render_corner_anchored_view",
    "render_scene_view",
    "render_scene_views",
    "require_choice",
    "run_cli",
    "run_dispatch_cli",
    "trim",
    "validate_mesh_index",
    "write_combined_preview",
    "write_images_dat_lgx",
]
