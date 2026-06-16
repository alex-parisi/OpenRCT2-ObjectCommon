"""Tests for openrct2_object_common.blender — lights + modal.

``modal`` imports ``bpy`` which is only available inside Blender. A minimal
fake ``bpy`` module is inserted into ``sys.modules`` here, before the import,
so the module can be imported and exercised without a Blender runtime.
"""

import os
import shutil
import sys
import tempfile
import time
import types
from types import SimpleNamespace

import numpy as np

# ---------------------------------------------------------------------------
# Fake bpy — must be injected BEFORE any import of the blender submodules.
# ---------------------------------------------------------------------------

_bpy = types.ModuleType("bpy")
_bpy_types = types.ModuleType("bpy.types")


class _FakeOperator:
    """Minimal stand-in for bpy.types.Operator."""

    def report(self, type_set, message):
        pass


_bpy_types.Operator = _FakeOperator
_bpy.types = _bpy_types
sys.modules.setdefault("bpy", _bpy)
sys.modules.setdefault("bpy.types", _bpy_types)

# Now safe to import the blender submodules.
from openrct2_object_common.blender import modal as modal_mod  # noqa: E402
from openrct2_object_common.blender.lights import (  # noqa: E402
    LIGHT_TYPE_MAP,
    default_lights,
    lights_from_items,
    normalize_direction,
)
from openrct2_object_common.blender.modal import (  # noqa: E402
    ExportParkobjModalBase,
    RenderModalBase,
    TestRenderModalBase,
)

# ---------------------------------------------------------------------------
# Fake context helpers for modal tests
# ---------------------------------------------------------------------------


class _FakeWM:
    def __init__(self):
        self.calls = []

    def progress_begin(self, *a):
        self.calls.append("progress_begin")

    def progress_update(self, v):
        self.calls.append("progress_update")

    def progress_end(self):
        self.calls.append("progress_end")

    def modal_handler_add(self, op):
        self.calls.append("modal_handler_add")

    def event_timer_add(self, interval, window=None):
        self.calls.append("event_timer_add")
        return "fake_timer"

    def event_timer_remove(self, timer):
        self.calls.append("event_timer_remove")


class _FakeWindow:
    def cursor_modal_set(self, cursor):
        pass

    def cursor_modal_restore(self):
        pass


class _FakeWorkspace:
    def __init__(self):
        self.status = None

    def status_text_set(self, text):
        self.status = text


class _FakeCtx:
    def __init__(self):
        self.window_manager = _FakeWM()
        self.window = _FakeWindow()
        self.workspace = _FakeWorkspace()


class _FakeEvent:
    def __init__(self, event_type="TIMER"):
        self.type = event_type


# ---------------------------------------------------------------------------
# Concrete RenderModalBase subclass for testing
# ---------------------------------------------------------------------------


class _Op(RenderModalBase):
    """Concrete operator: build returns a payload, render does nothing,
    success returns FINISHED."""

    _status_verb = "Testing"
    _render_raises: Exception | None = None

    def __init__(self):
        self._reported: list = []

    def report(self, type_set, message):
        self._reported.append((type_set, message))

    def _build(self, context):
        return "payload"

    def _prepare(self, context, payload):
        pass

    def _render(self, payload):
        if self._render_raises is not None:
            raise self._render_raises

    def _on_success(self, context):
        return {"FINISHED"}


def _wait_finished(op, timeout=5.0):
    """Block until the worker thread sets op._finished."""
    deadline = time.monotonic() + timeout
    while not getattr(op, "_finished", False):
        if time.monotonic() > deadline:
            raise TimeoutError("worker thread did not finish in time")
        time.sleep(0.01)
    op._thread.join()


# ===========================================================================
# blender/lights.py tests
# ===========================================================================


def test_light_type_map_contains_diffuse_and_specular():
    from openrct2_x7_renderer.constants import LightType

    assert LIGHT_TYPE_MAP["diffuse"] == LightType.DIFFUSE
    assert LIGHT_TYPE_MAP["specular"] == LightType.SPECULAR


def test_normalize_direction_unit_vector_unchanged():
    v = np.array([0.0, 1.0, 0.0])
    result = normalize_direction(v)
    assert np.allclose(result, [0.0, 1.0, 0.0])


def test_normalize_direction_non_unit_normalized():
    v = np.array([0.0, 0.0, 5.0])
    result = normalize_direction(v)
    assert np.allclose(result, [0.0, 0.0, 1.0])


def test_normalize_direction_zero_vector_returned_unchanged():
    v = np.array([0.0, 0.0, 0.0])
    result = normalize_direction(v)
    assert np.allclose(result, [0.0, 0.0, 0.0])


def test_lights_from_items_empty_returns_default():
    result = lights_from_items([])
    default = default_lights()
    assert len(result) == len(default)


def test_lights_from_items_builds_from_items():
    from openrct2_x7_renderer.constants import LightType

    class FakeItem:
        type = "diffuse"
        shadow = False
        direction = [0.0, 1.0, 0.0]
        strength = 0.8

    result = lights_from_items([FakeItem()])
    assert len(result) == 1
    assert result[0].type == LightType.DIFFUSE
    assert result[0].intensity == 0.8


def test_lights_from_items_specular_type():
    from openrct2_x7_renderer.constants import LightType

    class FakeItem:
        type = "specular"
        shadow = True
        direction = [1.0, 0.0, 0.0]
        strength = 0.5

    result = lights_from_items([FakeItem()])
    assert result[0].type == LightType.SPECULAR
    assert result[0].shadow == 1


# ===========================================================================
# blender/modal.py tests
# ===========================================================================


def test_set_progress_stores_units():
    op = _Op()
    op.set_progress(3, 10)
    assert op._done_units == 3
    assert op._total_units == 10


def test_read_render_settings_stashes_dither_and_default_lights():
    op = _Op()
    settings = SimpleNamespace(lights=[], dither="bayer", dither_stability=3.0)
    op._read_render_settings(settings)
    assert op._dither == "bayer"
    assert op._dither_stability == 3.0
    # An empty rig falls back to the renderer's default lights.
    assert len(op._lights) == len(default_lights())


def test_read_render_settings_builds_lights_from_items():
    class FakeItem:
        type = "diffuse"
        shadow = False
        direction = [0.0, 1.0, 0.0]
        strength = 0.8

    op = _Op()
    settings = SimpleNamespace(lights=[FakeItem()], dither="none", dither_stability=0.0)
    op._read_render_settings(settings)
    assert len(op._lights) == 1
    assert op._lights[0].intensity == 0.8


def test_make_context_passes_stashed_settings(monkeypatch):
    from openrct2_object_common import cli

    captured = {}

    def fake_make_context(lights, upt, test, *, dither, stability):
        captured.update(lights=lights, upt=upt, test=test, dither=dither, stability=stability)
        return "CTX"

    monkeypatch.setattr(cli, "make_context", fake_make_context)
    op = _Op()
    op._lights = ["L"]
    op._dither = "bayer"
    op._dither_stability = 2.0
    assert op._make_context(8.0, test=True) == "CTX"
    assert captured == {
        "lights": ["L"],
        "upt": 8.0,
        "test": True,
        "dither": "bayer",
        "stability": 2.0,
    }


def test_elapsed_suffix_omits_build_when_zero():
    op = _Op()
    op._start_time = time.monotonic()
    op._build_secs = 0
    assert op._elapsed_suffix() == "0s"


def test_elapsed_suffix_includes_build_when_nonzero():
    op = _Op()
    op._start_time = time.monotonic()
    op._build_secs = 3
    assert op._elapsed_suffix() == "0s (build 3s)"


def test_execute_returns_running_modal():
    op = _Op()
    ctx = _FakeCtx()
    result = op.execute(ctx)
    _wait_finished(op)
    assert result == {"RUNNING_MODAL"}


def test_execute_sets_up_timer_and_progress():
    op = _Op()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    assert "progress_begin" in ctx.window_manager.calls
    assert "event_timer_add" in ctx.window_manager.calls
    assert "modal_handler_add" in ctx.window_manager.calls


def test_execute_build_clean_error_returns_cancelled():
    class OpClean(_Op):
        _clean_error_types = (ValueError,)

        def _build(self, context):
            raise ValueError("bad build")

    op = OpClean()
    result = op.execute(_FakeCtx())
    assert result == {"CANCELLED"}
    assert any("bad build" in msg for _, msg in op._reported)


def test_execute_build_other_error_returns_cancelled_with_prefix():
    class OpOther(_Op):
        def _build(self, context):
            raise RuntimeError("unexpected!")

    op = OpOther()
    result = op.execute(_FakeCtx())
    assert result == {"CANCELLED"}
    assert any("Invalid object" in msg for _, msg in op._reported)


def test_modal_non_timer_event_returns_pass_through():
    op = _Op()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    result = op.modal(ctx, _FakeEvent("MOUSEMOVE"))
    assert result == {"PASS_THROUGH"}


def test_modal_timer_not_finished_increments_spinner_and_returns_pass_through():
    op = _Op()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    op._finished = False  # reset to simulate in-progress
    before = op._spinner_step
    result = op.modal(ctx, _FakeEvent("TIMER"))
    assert result == {"PASS_THROUGH"}
    assert op._spinner_step == before + 1


def test_modal_timer_finished_calls_finish_and_returns_success():
    op = _Op()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    result = op.modal(ctx, _FakeEvent("TIMER"))
    assert result == {"FINISHED"}
    assert "event_timer_remove" in ctx.window_manager.calls


def test_set_status_spinner_mode():
    op = _Op()
    ctx = _FakeCtx()
    op._start_time = time.monotonic()
    op._build_secs = 0
    op._spinner_step = 0
    op._done_units = 0
    op._total_units = 0
    op._status_verb = "Working"
    op._set_status(ctx)
    assert ctx.workspace.status is not None
    assert "Working" in ctx.workspace.status


def test_set_status_percentage_mode():
    op = _Op()
    ctx = _FakeCtx()
    op._start_time = time.monotonic()
    op._build_secs = 0
    op._spinner_step = 0
    op._done_units = 5
    op._total_units = 10
    op._status_verb = "Working"
    op._set_status(ctx)
    assert "50%" in ctx.workspace.status


def test_set_status_shows_build_time_when_nonzero():
    op = _Op()
    ctx = _FakeCtx()
    op._start_time = time.monotonic()
    op._build_secs = 3
    op._spinner_step = 0
    op._done_units = 0
    op._total_units = 0
    op._status_verb = "Working"
    op._set_status(ctx)
    assert "(build 3s)" in ctx.workspace.status


def test_finish_no_error_returns_on_success():
    op = _Op()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    result = op._finish(ctx)
    assert result == {"FINISHED"}
    assert "progress_end" in ctx.window_manager.calls


def test_finish_with_render_error_reports_cancelled(capsys):
    class OpFails(_Op):
        _render_raises = RuntimeError("render boom")

    op = OpFails()
    ctx = _FakeCtx()
    op.execute(ctx)
    _wait_finished(op)
    result = op._finish(ctx)
    assert result == {"CANCELLED"}
    assert any("failed" in msg for _, msg in op._reported)
    # The traceback is printed to stdout.
    captured = capsys.readouterr()
    assert "render boom" in captured.out


# ===========================================================================
# blender/modal.py — TestRenderModalBase
# ===========================================================================


class _TestRenderOp(TestRenderModalBase):
    _tmp_prefix = "orct2_test_blendertest_"

    def __init__(self):
        self._reported = []

    def report(self, type_set, message):
        self._reported.append((type_set, message))


def test_test_render_modal_base_replaces_previous_dir():
    _TestRenderOp._last_test_dir = None
    op1 = _TestRenderOp()
    op1._prepare(_FakeCtx(), "payload")
    first = op1._tmp
    assert os.path.isdir(first)
    assert op1._png is None

    op2 = _TestRenderOp()
    op2._prepare(_FakeCtx(), "payload")
    # The previous run's dir is removed only when the next render replaces it.
    assert not os.path.exists(first)
    assert os.path.isdir(op2._tmp)
    shutil.rmtree(op2._tmp, ignore_errors=True)
    _TestRenderOp._last_test_dir = None


def test_test_render_modal_base_on_success_shows_png(monkeypatch):
    seen = {}

    def _fake_show(operator, context, png):
        seen["png"] = png
        return {"FINISHED"}

    monkeypatch.setattr(modal_mod, "show_test_sprite", _fake_show)
    op = _TestRenderOp()
    op._png = "/some/sprite.png"
    assert op._on_success(None) == {"FINISHED"}
    assert seen["png"] == "/some/sprite.png"


# ===========================================================================
# blender/modal.py — ExportParkobjModalBase
# ===========================================================================


class _ExportOp(ExportParkobjModalBase):
    def __init__(self):
        self._reported = []
        self.filepath = ""

    def report(self, type_set, message):
        self._reported.append((type_set, message))

    def _default_filename(self, context):
        return "thing.parkobj"


def test_export_parkobj_base_invoke_defaults_filename():
    op = _ExportOp()
    added = []

    class _WM:
        def fileselect_add(self, operator):
            added.append(operator)

    ctx = SimpleNamespace(window_manager=_WM())
    assert op.invoke(ctx, None) == {"RUNNING_MODAL"}
    assert op.filepath == "thing.parkobj"
    assert added == [op]


def test_export_parkobj_base_invoke_keeps_existing_filepath():
    op = _ExportOp()
    op.filepath = "/already/set.parkobj"

    class _WM:
        def fileselect_add(self, operator):
            pass

    op.invoke(SimpleNamespace(window_manager=_WM()), None)
    assert op.filepath == "/already/set.parkobj"


def test_export_parkobj_base_prepare_sets_paths():
    op = _ExportOp()
    op.filepath = "/tmp/out.parkobj"
    op._prepare(_FakeCtx(), "payload")
    # The conftest fake bpy.path.abspath is the identity.
    assert op._parkobj == "/tmp/out.parkobj"
    assert os.path.isdir(op._work)
    shutil.rmtree(op._work, ignore_errors=True)


def test_export_parkobj_base_finish_removes_work(monkeypatch):
    monkeypatch.setattr(RenderModalBase, "_finish", lambda self, ctx: {"FINISHED"})
    op = _ExportOp()
    op._work = tempfile.mkdtemp(prefix="orct2_exp_blendertest_")
    work = op._work
    assert op._finish(None) == {"FINISHED"}
    assert not os.path.exists(work)


def test_export_parkobj_base_on_success_reports_filename():
    op = _ExportOp()
    op._parkobj = "/a/b/cool.parkobj"
    op._start_time = time.monotonic()
    op._build_secs = 0
    assert op._on_success(None) == {"FINISHED"}
    assert op._reported
    assert "Exported cool.parkobj" in op._reported[0][1]


class _FakeOverlay:
    """Records the overlay updates the export operator drives."""

    def __init__(self):
        self.done = 0
        self.total = 0
        self.removed = False
        self.redraws = 0

    def tag_redraw(self, context):
        self.redraws += 1

    def remove(self):
        self.removed = True


def test_export_parkobj_set_progress_mirrors_to_overlay():
    op = _ExportOp()
    op._overlay = _FakeOverlay()
    op.set_progress(4, 8)
    # The shared base still stores the units for the status-bar percentage...
    assert (op._done_units, op._total_units) == (4, 8)
    # ...and the in-viewport overlay is fed the same values.
    assert (op._overlay.done, op._overlay.total) == (4, 8)


def test_export_parkobj_set_status_redraws_overlay():
    op = _ExportOp()
    op._overlay = _FakeOverlay()
    op._start_time = time.monotonic()
    op._build_secs = 0
    op._spinner_step = 0
    op._done_units = 0
    op._total_units = 0
    op._status_verb = "Working"
    op._set_status(_FakeCtx())
    assert op._overlay.redraws == 1


def test_export_parkobj_finish_removes_overlay(monkeypatch):
    monkeypatch.setattr(RenderModalBase, "_finish", lambda self, ctx: {"FINISHED"})
    op = _ExportOp()
    op._overlay = _FakeOverlay()
    op._work = tempfile.mkdtemp(prefix="orct2_exp_blendertest_")
    work = op._work
    assert op._finish(_FakeCtx()) == {"FINISHED"}
    assert op._overlay.removed
    assert op._overlay.redraws == 1
    assert not os.path.exists(work)
