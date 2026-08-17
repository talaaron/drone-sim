"""
drone/world.py — the "world" the Drone flies over: a road (instead
of an empty grid) plus static sidewalks/buildings alongside it.
Mirrors the official PDF's top-down example
(`DOCS/images/road_view_example.png`): an infinite vertical road
along the Y axis, dark off-road margins on both sides. Colors are
sampled from the reference image, not chosen freehand - see
render.py.

Infinite and non-repeating: no wrap-around, and no finite obstacle
list kept in memory. A deterministic hash maps every cell of a
world grid to a fixed obstacle (or none), so the world scrolls
forever without growing state, and a given seed always reproduces
the same layout.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Road half-width, from center x=0 to each side. The Drone can
# leave it - movement is free in 2D - but this is what reads as
# "the road" in the frame.
ROAD_HALF_WIDTH = 160.0
SIDEWALK_GAP = 6.0  # gap between the road edge and a sidewalk/building


@dataclass
class Obstacle:
    x: float  # top-left corner, world coordinates
    y: float
    w: float
    h: float


class World:
    CELL_SIZE = 300.0
    OBSTACLE_PROBABILITY = 0.45

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def obstacles_in_view(self, view_left: float, view_top: float, view_w: float, view_h: float) -> list[Obstacle]:
        cell_x0 = int(view_left // self.CELL_SIZE) - 1
        cell_x1 = int((view_left + view_w) // self.CELL_SIZE) + 1
        cell_y0 = int(view_top // self.CELL_SIZE) - 1
        cell_y1 = int((view_top + view_h) // self.CELL_SIZE) + 1

        obstacles: list[Obstacle] = []
        for cx in range(cell_x0, cell_x1 + 1):
            for cy in range(cell_y0, cell_y1 + 1):
                rng = random.Random((self.seed, cx, cy))  # deterministic per cell
                if rng.random() > self.OBSTACLE_PROBABILITY:
                    continue
                w = rng.uniform(55, 110)
                h = rng.uniform(55, 110)
                oy = cy * self.CELL_SIZE + rng.uniform(20, self.CELL_SIZE - 20 - h)
                ox = cx * self.CELL_SIZE + rng.uniform(20, self.CELL_SIZE - 20 - w)
                ox = self._push_off_road(ox, w)
                obstacles.append(Obstacle(ox, oy, w, h))
        return obstacles

    @staticmethod
    def _push_off_road(ox: float, w: float) -> float:
        """If the obstacle overlaps the road band, push it out flush against
        the nearest edge - produces the "sidewalks hugging the road" look
        from the reference image instead of discarding the obstacle."""
        if ox + w <= -ROAD_HALF_WIDTH or ox >= ROAD_HALF_WIDTH:
            return ox  # already clear of the road
        center = ox + w / 2
        if center < 0:
            return -ROAD_HALF_WIDTH - SIDEWALK_GAP - w
        return ROAD_HALF_WIDTH + SIDEWALK_GAP
