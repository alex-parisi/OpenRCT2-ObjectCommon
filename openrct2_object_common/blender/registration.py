"""Shared class register/unregister loops for the add-ons.

Every add-on module keeps a ``_CLASSES`` tuple and registers it in order /
unregisters it in reverse -- the exact same two loops in every ``operators.py``,
``panels.py``, and ``props.py``. These helpers own that boilerplate so each
module's ``register`` / ``unregister`` is just the call plus its own extras
(pointer properties, the shared-parent/-light guards).

Requires ``bpy``; only import inside Blender.
"""

from collections.abc import Iterable

import bpy


def register_classes(classes: Iterable[type]) -> None:
    """``bpy.utils.register_class`` each class, in order."""
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister_classes(classes: Iterable[type]) -> None:
    """``bpy.utils.unregister_class`` each class, in reverse order."""
    for cls in reversed(list(classes)):
        bpy.utils.unregister_class(cls)
