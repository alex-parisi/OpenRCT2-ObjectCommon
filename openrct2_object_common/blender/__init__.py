"""
Blender add-on helpers shared by the vehicle and scenery add-ons.

``lights`` has no ``bpy`` dependency (it only reads attribute-bearing items), so
it imports anywhere. ``modal`` imports ``bpy`` and is meant to run inside Blender
only -- import it directly (``from openrct2_object_common.blender.modal import
RenderModalBase``) rather than eagerly here, so ``lights`` stays usable without
Blender installed.
"""
