"""Re-export CLI scaffolding from the renderer for downstream generators."""

from openrct2_x7_renderer.cli import (  # noqa: F401
    TEST_ZOOM,
    RenderFn,
    make_context,
    output_directory_of,
    parse_cli_args,
    run_cli,
)

__all__ = [
    "TEST_ZOOM",
    "RenderFn",
    "make_context",
    "output_directory_of",
    "parse_cli_args",
    "run_cli",
]
