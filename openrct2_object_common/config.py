"""Re-export config utilities from the renderer for downstream generators."""

from openrct2_x7_renderer.config import (  # noqa: F401
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
    resolve_asset_path,
)

__all__ = [
    "LoadError",
    "as_array_or_wrap",
    "load_meshes",
    "load_preview",
    "optional_bool",
    "optional_int",
    "optional_number",
    "optional_string",
    "optional_string_list",
    "parse_config",
    "read_vector3",
    "require_int",
    "require_number",
    "require_string",
    "resolve_asset_path",
]
