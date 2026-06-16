"""Tests for the shared object export helpers.

``output_paths`` is a pure path builder; ``export_object`` is checked against a
stubbed ``assemble_parkobj`` so the render binding is verified without rendering.
"""

from pathlib import Path

from openrct2_object_common import export as e


def test_output_paths_defaults():
    parkobj, work = e.output_paths("dist", "my.object")
    assert parkobj == Path("dist") / "my.object.parkobj"
    assert work == Path("object")


def test_output_paths_custom_work_dir():
    parkobj, work = e.output_paths(Path("/tmp/out"), "x", work_dir="build")
    assert parkobj == Path("/tmp/out/x.parkobj")
    assert work == Path("build")


def test_export_object_binds_obj_context_progress(monkeypatch):
    captured = {}

    def fake_assemble(obj_json, parkobj_path, work_dir, render, *, skip_render=False):
        captured.update(
            obj_json=obj_json,
            parkobj_path=parkobj_path,
            work_dir=work_dir,
            render=render,
            skip_render=skip_render,
        )

    monkeypatch.setattr(e, "assemble_parkobj", fake_assemble)

    seen = {}

    def render_sprites(obj, context, work_dir, progress):
        seen.update(obj=obj, context=context, work_dir=work_dir, progress=progress)
        return ["$LGX:images.dat[0..0]"]

    def progress(done, total):
        return None

    obj = object()
    e.export_object(
        obj, "ctx", {"id": "x"}, render_sprites,
        "out/x.parkobj", "work", skip_render=True, progress=progress,
    )

    assert captured["obj_json"] == {"id": "x"}
    assert captured["parkobj_path"] == Path("out/x.parkobj")
    assert captured["work_dir"] == Path("work")
    assert captured["skip_render"] is True

    # The render handed to assemble_parkobj binds this object/context/progress.
    images = captured["render"](Path("wd"))
    assert images == ["$LGX:images.dat[0..0]"]
    assert seen == {"obj": obj, "context": "ctx", "work_dir": Path("wd"), "progress": progress}


def test_export_to_directory_resolves_default_paths():
    captured = {}

    def export_to(obj, context, parkobj_path, work_dir, *, skip_render=False):
        captured.update(
            obj=obj,
            context=context,
            parkobj_path=parkobj_path,
            work_dir=work_dir,
            skip_render=skip_render,
        )

    obj = object()
    e.export_to_directory(export_to, obj, "ctx", "dist", "my.object", skip_render=True)

    # The default <output>/<id>.parkobj + object work dir convention is applied.
    assert captured["obj"] is obj
    assert captured["context"] == "ctx"
    assert captured["parkobj_path"] == Path("dist") / "my.object.parkobj"
    assert captured["work_dir"] == Path("object")
    assert captured["skip_render"] is True
