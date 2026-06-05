# OpenRCT2 Object Common

Shared scaffolding for the OpenRCT2 object generators. This package sits between
the renderer and the generators, so the vehicle and scenery tools share one
config layer, one CLI flow, and one `.parkobj` packaging path instead of each
re-implementing them.

```
openrct2-x7-renderer        the Embree-backed iso renderer (meshes -> sprites)
        ▲
OpenRCT2-ObjectCommon        config, CLI, placement, object.json, .parkobj  ← you are here
        ▲
   ┌────┴─────┐
Vehicle      Scenery         the generators (+ their Blender add-ons)
```

Rendering is handled by the external
[`openrct2-x7-renderer`](https://pypi.org/project/openrct2-x7-renderer/) package
(an Embree-backed ray tracer shipping prebuilt, vendored wheels), so this repo is
pure Python, with no compiler or Embree needed.

> Part of the **OpenRCT2-Tools** family — the
> [renderer](https://github.com/alex-parisi/OpenRCT2-X7-Renderer),
> [VehicleGenerator](https://github.com/alex-parisi/OpenRCT2-VehicleGenerator),
> and [SceneryGenerator](https://github.com/alex-parisi/OpenRCT2-SceneryGenerator)
> all build on this package.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended); `uv sync` pulls everything,
  including the renderer wheel from PyPI.

## Develop

```bash
uv sync
uv run pytest        # coverage is enabled by default via pyproject.toml
uv run ruff check .
uv run mypy
```

## What's here

| Module | What it provides |
|---|---|
| `config` | `parse_config` (JSON/YAML) and the `require_*` / `optional_*` validation helpers; `load_meshes` / `load_preview` |
| `cli` | `run_cli` (parse args → read config → resolve lights → render), `make_context`, `parse_cli_args`, `output_directory_of` |
| `objectjson` | `object_json_header` — the `id` / `originalId` / `version` / `authors` / `objectType` block every object shares |
| `placement` | `add_model_to_scene` and `orientation_to_matrix` — turn a `Model` into render-scene geometry |
| `parkobj` | `assemble_parkobj` (render-or-reuse → write object.json → zip) and `write_images_dat_lgx` |
| `blender.lights` | `lights_from_items` / `default_lights` — build the renderer's light rig from add-on UI items |
| `blender.modal` | `RenderModalBase` — the off-main-thread modal render operator the add-ons subclass |

## Usage

### CLI scaffolding

A generator's `__main__` wires its loader/exporter into the shared flow:

```python
from openrct2_object_common.cli import make_context, output_directory_of, run_cli


def _render(args, root, lights):
    obj = build_object(root)                       # generator-specific
    context = make_context(lights, obj.units_per_tile, args.test)
    export(obj, context, output_directory_of(root), skip_render=args.skip_render)


def main(argv=None):
    return run_cli("openrct2-my-generator", argv, _render)
```

### Packaging a `.parkobj`

```python
from openrct2_object_common.objectjson import object_json_header
from openrct2_object_common.parkobj import assemble_parkobj, write_images_dat_lgx


def export_to(obj, context, parkobj_path, work_dir, skip_render=False):
    obj_json = object_json_header(obj.id, object_type="scenery_small", authors=obj.authors)
    obj_json["properties"] = build_properties(obj)
    assemble_parkobj(
        obj_json, parkobj_path, work_dir,
        lambda wd: write_images_dat_lgx(render_sprites(obj, context), wd),
        skip_render=skip_render,
    )
```

### Blender helpers

The `blender` subpackage imports `bpy` (only `blender.modal` — `blender.lights`
does not), so install the extra when working with it outside Blender:

```bash
pip install "OpenRCT2-ObjectCommon[blender]"
```

## License

GPL-3.0-or-later.
