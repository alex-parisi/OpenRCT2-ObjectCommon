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

import os
import shutil
import tempfile
import threading
import time
import traceback
from typing import Any

from .lights import lights_from_items
from .progress_overlay import ProgressOverlay

try:
    from bpy.types import Operator
except ImportError:  # pragma: no cover
    # Outside Blender: provide a no-op stub so this module can be imported
    # (e.g. for type-checking or test collection) without a Blender runtime.
    # A real operator requires bpy; this stub allows the class to be defined.
    class Operator:  # type: ignore[no-redef]
        pass

_SPINNER_FRAMES = "|/-\\"


def show_test_sprite(operator, context, png):
    """Show a freshly rendered test sprite in an open Image Editor.

    Shared by every add-on's test-render operator: ``png`` is the file the render
    just wrote (or ``None``). Reports a warning and returns ``{"CANCELLED"}`` when
    nothing was produced; otherwise loads the image, assigns it to the first open
    Image Editor area (if any), reports it and returns ``{"FINISHED"}``.

    ``operator`` is the calling Operator (used for ``report``); kept as a plain
    argument so both the modal test-render operators and the vehicle add-on's
    non-modal one can share this.
    """
    import bpy  # local: this helper only runs inside Blender (see module docstring)

    if not png or not os.path.exists(png):
        operator.report({"WARNING"}, "Render produced no sprite")
        return {"CANCELLED"}
    img = bpy.data.images.load(png, check_existing=False)
    for area in context.screen.areas:
        if area.type == "IMAGE_EDITOR":
            area.spaces.active.image = img
            break
    operator.report({"INFO"}, f"Test sprite loaded: {img.name}")
    return {"FINISHED"}


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

    # -- shared _prepare helper ----------------------------------------------

    def _read_render_settings(self, settings) -> None:
        """Stash the render-affecting scene settings on ``self`` (main thread).

        Reads the custom lighting rig + dither config off a settings
        PropertyGroup into ``self._lights`` / ``self._dither`` /
        ``self._dither_stability`` so the worker thread (``_render``) can build
        the render Context without touching bpy data. Subclasses call this from
        ``_prepare`` with their scene settings group.
        """
        self._lights = lights_from_items(settings.lights)
        self._dither = settings.dither
        self._dither_stability = settings.dither_stability

    def _make_context(self, units_per_tile, *, test: bool = False):
        """Build a render Context from the settings stashed by
        :meth:`_read_render_settings` (worker thread; touches no bpy data).

        ``test`` zooms in for a single-viewpoint preview; exports pass ``False``.
        Shared so every add-on's ``_render`` builds its Context the same way
        instead of repeating the lights/dither/stability wiring.
        """
        from openrct2_object_common.cli import make_context

        return make_context(
            self._lights,
            units_per_tile,
            test,
            dither=self._dither,
            stability=self._dither_stability,
        )

    def _elapsed_suffix(self) -> str:
        """``"12s"``, or ``"12s (build 3s)"`` when the build phase was non-trivial.

        Seconds since the render started, plus the build-phase time when it is
        worth mentioning (animated / large builds; a static build is instant and
        "(build 0s)" would be noise). Shared by the status readout and the
        operators' success reports.
        """
        elapsed = int(time.monotonic() - self._start_time)
        build = f" (build {self._build_secs}s)" if self._build_secs else ""
        return f"{elapsed}s{build}"

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
        suffix = self._elapsed_suffix()
        wm = context.window_manager
        if self._total_units > 0:
            pct = int(100 * self._done_units / self._total_units)
            context.workspace.status_text_set(
                f"{self._status_verb}... {pct}% {suffix}"
            )
            wm.progress_update(self._done_units / self._total_units)
        else:
            glyph = _SPINNER_FRAMES[self._spinner_step % len(_SPINNER_FRAMES)]
            context.workspace.status_text_set(
                f"{glyph} {self._status_verb}... {suffix}"
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


class TestRenderModalBase(RenderModalBase):
    """Render a quick test sprite to a temp dir and show it in the Image Editor.

    Owns the throwaway-directory lifecycle every add-on's Test Render shares: the
    PNG must outlive the operator (the Image Editor reads it from disk), so the
    previous run's directory is removed only when the next render replaces it.
    The directory is tracked per operator subclass.

    Subclasses set ``_tmp_prefix`` and implement ``_render`` -- writing into
    ``self._tmp`` and setting ``self._png`` to the sprite path to display. The
    add-on's intermediate base still supplies ``_build`` and the render-settings
    read (via ``_read_render_settings`` in its ``_prepare``), reached here through
    ``super()._prepare``.
    """

    _status_verb = "Rendering test"
    _tmp_prefix = "test_"
    # Per-subclass: the previous run's output dir, removed on the next render.
    # ``type(self)`` writes route to the concrete subclass, so add-ons don't
    # clobber each other's directory.
    _last_test_dir: str | None = None

    def _prepare(self, context, payload) -> None:
        super()._prepare(context, payload)
        cls = type(self)
        if cls._last_test_dir is not None:
            shutil.rmtree(cls._last_test_dir, ignore_errors=True)
        self._tmp = tempfile.mkdtemp(prefix=self._tmp_prefix)
        cls._last_test_dir = self._tmp
        self._png = None

    def _on_success(self, context):
        return show_test_sprite(self, context, self._png)


class ExportParkobjModalBase(RenderModalBase):
    """Render every sprite off-thread and write one ``.parkobj`` via a file dialog.

    Owns the shared single-object export flow: a file-select dialog defaulting the
    filename from the object id, a throwaway work dir, work-dir cleanup, and the
    success report. Subclasses declare the ``filepath`` / ``filename_ext`` /
    ``filter_glob`` bpy properties, set ``_tmp_prefix``, and implement
    ``_default_filename`` and ``_render`` (driving their exporter against
    ``self._parkobj`` / ``self._work``); the add-on's intermediate base supplies
    ``_build`` and the render-settings read, reached through ``super()._prepare``.

    Every parkobj export also gets the in-viewport progress bar
    (:class:`ProgressOverlay`): it is added in ``_prepare``, fed by
    :meth:`set_progress` (the same calls that drive the status-bar percentage),
    repainted from ``_set_status``, and removed in ``_finish``.
    """

    _status_verb = "Exporting .parkobj"
    _tmp_prefix = "export_"

    def _default_filename(self, context) -> str:  # pragma: no cover - subclass hook
        raise NotImplementedError

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = self._default_filename(context)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def _prepare(self, context, payload) -> None:
        super()._prepare(context, payload)
        import bpy  # local: this helper only runs inside Blender (see module docstring)

        self._parkobj = bpy.path.abspath(self.filepath)
        self._work = tempfile.mkdtemp(prefix=self._tmp_prefix)
        # In-viewport progress bar; the shared base also drives a status-bar
        # percentage from the same set_progress() calls.
        self._overlay = ProgressOverlay()
        self._overlay.add()

    def set_progress(self, done: int, total: int) -> None:
        super().set_progress(done, total)
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.done = done
            overlay.total = total

    def _set_status(self, context) -> None:
        super()._set_status(context)
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.tag_redraw(context)

    def _finish(self, context):
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.remove()
            overlay.tag_redraw(context)
        # The work dir is a render scratch space; drop it once the worker is
        # joined (super()._finish joins it), whether the export succeeded or not.
        result = super()._finish(context)
        work = getattr(self, "_work", None)
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        return result

    def _on_success(self, context):
        name = os.path.basename(self._parkobj)
        self.report({"INFO"}, f"Exported {name} in {self._elapsed_suffix()}")
        return {"FINISHED"}
