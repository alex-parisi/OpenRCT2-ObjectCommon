"""Shared renderer stubs for unit-testing object generators.

VehicleGenerator and SceneryGenerator both need a lightweight fake of the
Embree render pipeline that records lifecycle events without touching real
GPU/Embree resources.  Import these instead of reimplementing them per-suite.

Event format in ``FakeContext.events``:
  ``"begin"``        — :meth:`FakeContext.begin_render` was called.
  ``"finalize"``     — :meth:`FakeBuilder.finalize` was called.
  ``"end"``          — :meth:`FakeScene.end_render` was called (via explicit
                       call or context-manager exit).
  ``("add", mask)``  — :meth:`FakeBuilder.add_model` was called with ``mask``.
"""

from openrct2_x7_renderer.types import IndexedImage

__all__ = ["FakeContext", "FakeScene", "FakeBuilder"]

# Events appended to FakeContext.events by the stub pipeline.
Event = str | tuple[str, int]


class FakeScene:
    """Stands in for ``FinalizedScene``; every view renders a 1×1 dummy."""

    def __init__(self, events: list[Event]) -> None:
        self._events = events

    def __enter__(self) -> "FakeScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.end_render()

    def render_view(self, _view: object) -> IndexedImage:
        return IndexedImage.blank(1, 1)

    def render_silhouette(self, _view: object) -> IndexedImage:
        return IndexedImage.blank(1, 1)

    def end_render(self) -> None:
        self._events.append("end")


class FakeBuilder:
    """Stands in for ``SceneBuilder``; records lifecycle events."""

    def __init__(self, events: list[Event]) -> None:
        self._events = events

    def __enter__(self) -> "FakeBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def add_model(
        self,
        mesh: object,
        matrix: object,
        translation: object,
        mask: int = 0,
    ) -> "FakeBuilder":
        self._events.append(("add", mask))
        return self

    def finalize(self) -> FakeScene:
        self._events.append("finalize")
        return FakeScene(self._events)


class FakeContext:
    """Records render lifecycle calls without touching Embree."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        # Mirrors Context.remap_overrides; may be overwritten by export_*_test.
        self.remap_overrides: dict[int, tuple[int, ...]] = {}

    def begin_render(self) -> FakeBuilder:
        self.events.append("begin")
        return FakeBuilder(self.events)
