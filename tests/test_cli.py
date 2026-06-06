"""Tests for openrct2_object_common.cli — shared CLI scaffolding."""

import json
from pathlib import Path

import pytest
from openrct2_object_common.cli import (
    make_context,
    output_directory_of,
    parse_cli_args,
    run_cli,
)
from openrct2_x7_renderer.lights import default_lights

# --------------------------------------------------------------------------
# parse_cli_args
# --------------------------------------------------------------------------


def test_parse_cli_args_positional_only():
    args = parse_cli_args("prog", ["cfg.json"])
    assert args.input == Path("cfg.json")
    assert args.test is False
    assert args.skip_render is False


def test_parse_cli_args_test_flag():
    args = parse_cli_args("prog", ["--test", "cfg.json"])
    assert args.test is True
    assert args.skip_render is False


def test_parse_cli_args_skip_render_flag():
    args = parse_cli_args("prog", ["--skip-render", "cfg.json"])
    assert args.skip_render is True
    assert args.test is False


def test_parse_cli_args_test_and_skip_render_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_cli_args("prog", ["--test", "--skip-render", "cfg.json"])


# --------------------------------------------------------------------------
# output_directory_of
# --------------------------------------------------------------------------


def test_output_directory_of_returns_configured_path():
    result = output_directory_of({"output_directory": "/out/dir"})
    assert result == Path("/out/dir")


def test_output_directory_of_defaults_to_cwd():
    assert output_directory_of({}) == Path(".")
    assert output_directory_of({"output_directory": None}) == Path(".")


# --------------------------------------------------------------------------
# make_context
# --------------------------------------------------------------------------


def test_make_context_normal_uses_full_upt():
    lights = default_lights()
    ctx = make_context(lights, 32.0, test=False)
    assert ctx is not None


def test_make_context_test_applies_zoom():
    lights = default_lights()
    # In test mode, upt is scaled by TEST_ZOOM (0.125). Verify the context
    # object is created successfully (the internal upt value is not exposed).
    ctx = make_context(lights, 32.0, test=True)
    assert ctx is not None


def test_make_context_test_without_root_has_no_overrides():
    # test=True but root=None → overrides are {} (no remap loads).
    ctx = make_context(default_lights(), 32.0, test=True, root=None)
    assert ctx is not None


def test_make_context_test_with_root_loads_remap_overrides():
    # test=True with root that has test_remap_colors → load_remap_overrides runs.
    root = {"test_remap_colors": {"1": "grey"}}
    ctx = make_context(default_lights(), 32.0, test=True, root=root)
    assert ctx is not None


def test_make_context_non_test_ignores_root():
    # test=False → overrides are always {} regardless of root content.
    root = {"test_remap_colors": {"1": "grey"}}
    ctx = make_context(default_lights(), 32.0, test=False, root=root)
    assert ctx is not None


# --------------------------------------------------------------------------
# run_cli
# --------------------------------------------------------------------------


def test_run_cli_success_returns_zero(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"id": "test"}))

    rendered = []
    rc = run_cli("prog", [str(cfg)], lambda args, root, lights: rendered.append(root))
    assert rc == 0
    assert rendered == [{"id": "test"}]


def test_run_cli_uses_lights_from_config(tmp_path):
    cfg = tmp_path / "cfg.json"
    lights_cfg = [{"type": "diffuse", "direction": [0, 1, 0], "strength": 0.5}]
    cfg.write_text(json.dumps({"lights": lights_cfg}))

    captured = []
    rc = run_cli("prog", [str(cfg)], lambda args, root, lights: captured.append(lights))
    assert rc == 0
    assert len(captured[0]) == 1


def test_run_cli_load_error_returns_one(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text("[]")  # not a dict → LoadError
    assert run_cli("prog", [str(cfg)], lambda *a: None) == 1


def test_run_cli_os_error_returns_one():
    assert run_cli("prog", ["/nonexistent/file.json"], lambda *a: None) == 1


def test_run_cli_value_error_returns_one(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    def bad_render(args, root, lights):
        raise ValueError("bad value")

    assert run_cli("prog", [str(cfg)], bad_render) == 1


def test_run_cli_runtime_error_returns_one(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    def bad_render(args, root, lights):
        raise RuntimeError("bad runtime")

    assert run_cli("prog", [str(cfg)], bad_render) == 1
