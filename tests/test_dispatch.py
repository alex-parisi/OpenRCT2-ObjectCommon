"""Tests for the shared object-type dispatch CLI.

``dispatch_render`` is exercised with a fake (load, export, export_test) table
and stubbed context/output helpers; ``run_dispatch_cli`` is checked for its
delegation to the renderer's ``run_cli``.
"""

import argparse
import types

from openrct2_object_common import dispatch as d


def _args(input_path, test=False, skip_render=False):
    return argparse.Namespace(input=input_path, test=test, skip_render=skip_render)


def _table(calls, object_type="thing"):
    obj = types.SimpleNamespace(units_per_tile=32.0)

    def fake_load(path):
        calls["load"] = path
        return obj

    def fake_export(o, ctx, out, skip_render):
        calls["export"] = {"obj": o, "ctx": ctx, "out": out, "skip_render": skip_render}

    def fake_export_test(o, ctx):
        calls["export_test"] = {"obj": o, "ctx": ctx}

    return obj, {object_type: (fake_load, fake_export, fake_export_test)}


def _patch_context(monkeypatch):
    monkeypatch.setattr(
        d, "make_context", lambda lights, upt, test, root=None: ("ctx", upt, test)
    )
    monkeypatch.setattr(d, "output_directory_of", lambda root: "out-dir")


def test_dispatch_render_full_export_path(monkeypatch):
    _patch_context(monkeypatch)
    calls = {}
    obj, table = _table(calls)
    d.dispatch_render(_args("s.json", skip_render=True), {}, [], table, lambda root: "thing")
    assert "export_test" not in calls
    assert calls["load"] == "s.json"
    assert calls["export"]["obj"] is obj
    assert calls["export"]["out"] == "out-dir"
    assert calls["export"]["skip_render"] is True
    # make_context was told this is not a test render.
    assert calls["export"]["ctx"] == ("ctx", 32.0, False)


def test_dispatch_render_test_path(monkeypatch):
    _patch_context(monkeypatch)
    calls = {}
    _obj, table = _table(calls)
    d.dispatch_render(_args("s.json", test=True), {}, [], table, lambda root: "thing")
    assert "export" not in calls
    # make_context was told this is a test render (TEST_ZOOM preview scale).
    assert calls["export_test"]["ctx"] == ("ctx", 32.0, True)


def test_dispatch_render_selects_by_object_type(monkeypatch):
    _patch_context(monkeypatch)
    calls = {}
    _obj, table = _table(calls, object_type="other")
    d.dispatch_render(
        _args("s.json"), {"object_type": "other"}, [], table, lambda root: root["object_type"]
    )
    assert calls["export"]["out"] == "out-dir"


def test_run_dispatch_cli_delegates_to_run_cli(monkeypatch):
    captured = {}

    def fake_run_cli(prog, argv, render):
        captured.update(prog=prog, argv=argv, render=render)
        return 7

    monkeypatch.setattr(d, "run_cli", fake_run_cli)

    def otype(root):
        return "thing"

    table = {"thing": (lambda p: None, lambda *a, **k: None, lambda *a: None)}
    rc = d.run_dispatch_cli("prog-x", ["a.json"], table, otype)
    assert rc == 7
    assert captured["prog"] == "prog-x"
    assert captured["argv"] == ["a.json"]

    # The render handed to run_cli forwards its arguments to dispatch_render
    # along with this call's dispatch table and object_type_of.
    seen = {}
    monkeypatch.setattr(
        d,
        "dispatch_render",
        lambda args, root, lights, dispatch, object_type_of: seen.update(
            dispatch=dispatch, object_type_of=object_type_of
        ),
    )
    captured["render"](_args("a.json"), {}, [])
    assert seen["dispatch"] is table
    assert seen["object_type_of"] is otype
