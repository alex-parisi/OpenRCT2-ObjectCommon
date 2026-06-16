"""A GPU-drawn progress bar overlaid on the 3D viewport.

Used by the threaded ``.parkobj`` export (:class:`ExportParkobjModalBase`) so
progress is shown on screen instead of only on the mouse cursor. The owning
operator updates :class:`ProgressOverlay`'s fields from its ``set_progress``
calls; the draw handler reads them when each viewport repaints.

Blender-only: ``blf`` / ``bpy`` / ``gpu`` are imported lazily (inside the
methods that use them, and the shader is built on first draw) so this module
imports without a Blender runtime -- type-checking and test collection don't
need the GPU module.
"""

# Bar geometry / colours (pixels, lower-left origin within the region).
_BAR_MAX_WIDTH = 420
_BAR_HEIGHT = 20
_MARGIN_BOTTOM = 70
_PAD = 12

_COL_BACKDROP = (0.0, 0.0, 0.0, 0.55)
_COL_TROUGH = (0.12, 0.12, 0.12, 0.9)
_COL_FILL = (0.20, 0.55, 0.95, 1.0)
_COL_TEXT = (1.0, 1.0, 1.0, 1.0)

# Built on first draw (inside Blender), not at import time.
_shader = None


def _bar_shader():
    """The shared UNIFORM_COLOR shader, built lazily so importing this module
    needs no GPU module (and so no Blender runtime)."""
    global _shader
    if _shader is None:
        import gpu

        _shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    return _shader


def _rect(x: float, y: float, w: float, h: float, color) -> None:
    from gpu_extras.batch import batch_for_shader

    shader = _bar_shader()
    verts = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
    batch = batch_for_shader(shader, "TRIS", {"pos": verts}, indices=((0, 1, 2), (2, 3, 0)))
    shader.uniform_float("color", color)
    batch.draw(shader)


class ProgressOverlay:
    """Manages a POST_PIXEL draw handler on the 3D viewport."""

    def __init__(self) -> None:
        self.label = "Exporting .parkobj..."
        self.done = 0
        self.total = 0
        self._pulse = 0.0  # animates the bar while total is still unknown
        self._handle = None

    def add(self) -> None:
        import bpy

        if self._handle is None:
            self._handle = bpy.types.SpaceView3D.draw_handler_add(
                self._draw, (), "WINDOW", "POST_PIXEL"
            )

    def remove(self) -> None:
        import bpy

        if self._handle is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self._handle, "WINDOW")
            self._handle = None

    def tag_redraw(self, context) -> None:
        self._pulse = (self._pulse + 0.04) % 1.0
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()

    def _draw(self) -> None:
        import blf
        import bpy
        import gpu

        region = bpy.context.region
        if region is None:
            return
        width, height = region.width, region.height
        if width < 80 or height < 60:
            return

        bar_w = min(_BAR_MAX_WIDTH, width - 2 * _MARGIN_BOTTOM)
        bar_x = (width - bar_w) / 2.0
        bar_y = _MARGIN_BOTTOM

        gpu.state.blend_set("ALPHA")

        # Backdrop behind both the text and the trough.
        _rect(
            bar_x - _PAD,
            bar_y - _PAD,
            bar_w + 2 * _PAD,
            _BAR_HEIGHT + 2 * _PAD + 12,
            _COL_BACKDROP,
        )
        _rect(bar_x, bar_y, bar_w, _BAR_HEIGHT, _COL_TROUGH)

        if self.total > 0:
            frac = max(0.0, min(1.0, self.done / self.total))
            _rect(bar_x, bar_y, bar_w * frac, _BAR_HEIGHT, _COL_FILL)
            label = f"{self.label}  {int(frac * 100)}%"
        else:
            # Indeterminate: a chunk slides back and forth until work is counted.
            chunk = bar_w * 0.25
            travel = bar_w - chunk
            offset = abs(self._pulse * 2.0 - 1.0) * travel
            _rect(bar_x + offset, bar_y, chunk, _BAR_HEIGHT, _COL_FILL)
            label = self.label

        gpu.state.blend_set("NONE")

        font_id = 0
        blf.size(font_id, 13)
        blf.color(font_id, *_COL_TEXT)
        blf.position(font_id, bar_x, bar_y + _BAR_HEIGHT + 6, 0)
        blf.draw(font_id, label)
