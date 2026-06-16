"""
OpenRCT2's remap colour names, in palette-index order.

Every object that paints with the player-selectable remap palette (vehicle car
colours, stall car colours) names its colours from this list, and a name's
position here *is* its OpenRCT2 colour index. Shared by every generator that
emits a ``carColours`` block or maps a colour name to its remap index.
"""

__all__ = ["COLOR_NAMES"]

# OpenRCT2's 32 remap colour names, by palette index.
COLOR_NAMES = [
    "black",
    "grey",
    "white",
    "dark_purple",
    "light_purple",
    "bright_purple",
    "dark_blue",
    "light_blue",
    "icy_blue",
    "teal",
    "aquamarine",
    "saturated_green",
    "dark_green",
    "moss_green",
    "bright_green",
    "olive_green",
    "dark_olive_green",
    "bright_yellow",
    "yellow",
    "dark_yellow",
    "light_orange",
    "dark_orange",
    "light_brown",
    "saturated_brown",
    "dark_brown",
    "salmon_pink",
    "bordeaux_red",
    "saturated_red",
    "bright_red",
    "dark_pink",
    "bright_pink",
    "light_pink",
]
