"""Shared modal operator for add-ons that render off the main thread.

The render (the renderer-bound work) is an opaque, blocking call, so it runs in
a worker thread while a modal timer drives a status-bar readout. The build phase
(reading bpy data into the core object) stays on the main thread.

NOTE: no ``from __future__ import annotations`` -- subclasses declare bpy
properties as annotations and PEP 563 would stringify them and break add-on
registration.

This module imports ``bpy`` and is meant to run inside Blender only; install the
package's ``blender`` extra (``pip install OpenRCT2-ObjectCommon[blender]``) when
type-checking or testing it outside Blender.
"""

import threading
import time
import traceback
from typing import Any

try:
    from bpy.types import Operator
except ImportError:  # pragma: no cover
    # Outside Blender: provide a no-op stub so this module can be imported
    # (e.g. for type-checking or test collection) without a Blender runtime.
    # A real operator requires bpy; this stub allows the class to be defined.
    class Operator:  # type: ignore[no-redef]
        pass

_SPINNER_FRAMES = "|/-\\"


class RenderModalBase(Operator):
    """Scaffolding for operators that run a blocking render off the main thread
    while showing a status-bar readout.

    Subclasses provide the four hooks below. ``_build`` / ``_prepare`` /
    ``_on_success`` run on the main thread and may touch ``context`` and bpy
    data; ``_render`` runs in the worker thread and must touch only ``self``
    (values it needs from ``context`` are stashed in ``_prepare``).

    Progress is *indeterminate* (a cycling spinner) unless ``_render`` reports
    real units via :meth:`set_progress`, in which case the status line shows a
    percentage instead.

    Subclass hooks:
      ``_status_verb``                 label shown in the status bar.
      ``_clean_error_types``           build-error exception types whose message
                                       is already user-facing (reported as-is,
                                       not wrapped with ``_invalid_prefix``).
      ``_build(context)``              read bpy data into the core object(s);
                                       returns an opaque payload for the others.
      ``_prepare(context, payload)``   stash per-run state on ``self`` (lights,
                                       output paths) -- main thread.
      ``_render(payload)``             the blocking work -- worker thread.
      ``_on_success(context)``         post-render UI; returns an operator result.
    """

    _status_verb = "Working"
    _invalid_prefix = "Invalid object"
    # Plain assignment, NOT an annotation: Blender's register_class() scans
    # `__annotations__` across the MRO for bpy properties, and an annotated
    # non-property class attribute here can trip subclass registration. Value is
    # a tuple[type[Exception], ...] of build-error types reported verbatim.
    _clean_error_types = ()

    # -- hooks ---------------------------------------------------------------

    def _build(self, context) -> Any:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def _prepare(self, context, payload) -> None:  # pragma: no cover - subclass hook
        pass

    def _render(self, payload) -> None:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def _on_success(self, context):  # pragma: no cover - subclass hook
        raise NotImplementedError

    # -- worker-facing -------------------------------------------------------

    def set_progress(self, done: int, total: int) -> None:
        """Report determinate progress from ``_render`` (worker thread). Plain
        int writes; the modal timer reads them to repaint. No lock needed for
        single-word assignments."""
        self._done_units = done
        self._total_units = total

    # -- operator flow -------------------------------------------------------

    def execute(self, context):
        build_start = time.monotonic()
        try:
            payload = self._build(context)
        except self._clean_error_types as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"{self._invalid_prefix}: {e}")
            return {"CANCELLED"}
        self._build_secs = int(time.monotonic() - build_start)

        self._prepare(context, payload)
        self._error: str | None = None
        self._finished = False
        self._start_time = time.monotonic()
        self._spinner_step = 0
        self._done_units = 0
        self._total_units = 0

        def worker():
            try:
                self._render(payload)
            except Exception:
                self._error = traceback.format_exc()
            finally:
                self._finished = True

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

        wm = context.window_manager
        wm.progress_begin(0, 1)
        context.window.cursor_modal_set("WAIT")
        self._set_status(context)
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            if self._finished:
                return self._finish(context)
            self._spinner_step += 1
            self._set_status(context)
        return {"PASS_THROUGH"}

    def _set_status(self, context) -> None:
        elapsed = int(time.monotonic() - self._start_time)
        # Only mention the build time when it was non-trivial (animated / large
        # builds); a static build is instant and "(build 0s)" would be noise.
        build = f" (build {self._build_secs}s)" if self._build_secs else ""
        wm = context.window_manager
        if self._total_units > 0:
            pct = int(100 * self._done_units / self._total_units)
            context.workspace.status_text_set(
                f"{self._status_verb}... {pct}% {elapsed}s{build}"
            )
            wm.progress_update(self._done_units / self._total_units)
        else:
            glyph = _SPINNER_FRAMES[self._spinner_step % len(_SPINNER_FRAMES)]
            context.workspace.status_text_set(
                f"{glyph} {self._status_verb}... {elapsed}s{build}"
            )
            wm.progress_update((self._spinner_step % 20) / 20.0)

    def _finish(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        wm.progress_end()
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        self._thread.join()
        if self._error:
            print(self._error)
            self.report(
                {"ERROR"}, f"{self._status_verb} failed; see the system console for details."
            )
            return {"CANCELLED"}
        return self._on_success(context)
