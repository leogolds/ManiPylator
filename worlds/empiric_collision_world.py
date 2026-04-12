"""
Default obstacle layout for collision oracle (matches examples/collision_lab.py).

Loaded by ``manipylator.collision_oracle_app`` when ``COLLISION_ORACLE_WORLD_FILE``
points here. Must define ``world`` as ``manipylator.base.World``.
"""

import genesis as gs

from manipylator.base import World


def _morphs():
    return (
        gs.morphs.Box(
            pos=(-0.12, 0.42, 0.45),
            size=(0.18, 0.18, 0.60),
            fixed=True,
            collision=True,
        ),
        gs.morphs.Box(
            pos=(-0.12, -0.42, 0.45),
            size=(0.18, 0.18, 0.60),
            fixed=True,
            collision=True,
        ),
    )


world = World(_morphs)
