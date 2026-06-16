"""
Shared object-type dispatch for generator CLIs.

The scenery and ride generators both select a ``(load, export, export_test)``
triple by the config's ``object_type``, then run the same flow: load the object,
build a render context at its scale, and either write per-view test PNGs
(``--test``) or assemble the ``.parkobj`` into the configured output directory.
``run_dispatch_cli`` owns that flow; each generator supplies only its dispatch
table and ``object_type_of``.
"""

import argparse
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from openrct2_x7_renderer.types import Light

from .cli import make_context, output_directory_of, run_cli

__all__ = ["Dispatch", "dispatch_render", "run_dispatch_cli"]


class _CliObject(Protocol):
    """The surface the dispatch CLI needs from a loaded object."""

    units_per_tile: float


_Loader = Callable[[Path], _CliObject]
# export(obj, context, output_dir, *, skip_render) and export_test(obj, context)
# share no single signature, so the triple's exporters stay loosely typed.
_Exporter = Callable[..., None]
Dispatch = Mapping[str, tuple[_Loader, _Exporter, _Exporter]]


def dispatch_render(
    args: argparse.Namespace,
    root: dict[str, Any],
    lights: list[Light],
    dispatch: Dispatch,
    object_type_of: Callable[[dict[str, Any]], str],
) -> None:
    """Load the object for ``root``'s object_type and export (or test-render) it."""
    load, export, export_test = dispatch[object_type_of(root)]
    obj = load(args.input)
    context = make_context(lights, obj.units_per_tile, args.test, root)
    if args.test:
        export_test(obj, context)
    else:
        export(obj, context, output_directory_of(root), skip_render=args.skip_render)


def run_dispatch_cli(
    prog: str,
    argv: list[str] | None,
    dispatch: Dispatch,
    object_type_of: Callable[[dict[str, Any]], str],
) -> int:
    """Run the shared generator CLI for a ``(load, export, export_test)`` table."""
    return run_cli(
        prog,
        argv,
        lambda args, root, lights: dispatch_render(
            args, root, lights, dispatch, object_type_of
        ),
    )
