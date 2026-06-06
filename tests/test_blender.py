"""Tests for openrct2_object_common.blender — lights + modal.

``modal`` imports ``bpy`` which is only available inside Blender. A minimal
fake ``bpy`` module is inserted into ``sys.modules`` here, before the import,
so the module can be imported and exercised without a Blender runtime.
"""

import sys
import time
import types

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
from openrct2_object_common.blender.lights import (  # noqa: E402
    LIGHT_TYPE_MAP,
    default_lights,
    lights_from_items,
    normalize_direction,
)
from openrct2_object_common.blender.modal import RenderModalBase  # noqa: E402

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
