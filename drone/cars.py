"""
drone/cars.py — Challenge: moving cars spawned into the world at
regular intervals.

The part of the Challenge that's actually implemented in code, not
just an architecture proposal - see DOCS/WORK_PLAN.md, "Challenge
scope decision". The production-grade detector (YOLO/ByteTrack)
stays a written proposal; gcs/detector.py runs a naive color-based
detector for the demo instead.

Every car gets a bold, saturated color with a hue clearly distinct
from the gray road and dark-blue off-road margins (see
drone/render.py) - yellow, red, green, matching the PDF's example
image. That's a deliberate simplifying assumption: it's what lets
the naive detector find cars by hue and saturation alone, with no
trained model, which is enough to prove the tracking principle
(persistent IDs across frames) at a scope that fits a home
assignment.

Cars drive straight along the road (Y axis only, like real lane
traffic) instead of drifting freely - they spawn inside the road
band and move up or down, matching the example image.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass

from .world import ROAD_HALF_WIDTH

# BGR, sampled from the cars in the reference image
# (DOCS/images/road_view_example.png), not chosen freehand. Hue is
# far from the road's gray and the off-road's dark blue, which is
# what makes the naive detection in gcs/detector.py possible.
CAR_COLORS_BGR = [
    (40, 203, 241),  # yellow
    (49, 48, 190),  # red
    (60, 200, 60),  # green - extra hue for variety, not from the source image
]

SPAWN_INTERVAL_SEC = 2.0
CAR_SPEED_PX_S = 90.0
CAR_HALF_SIZE = (13, 7)  # (half-width, half-height) of the car rectangle, top-down
CAR_MARGIN_FROM_ROAD_EDGE = 25.0  # keeps cars off the white lane line
SPAWN_RADIUS_PX = 480.0  # spawn distance ahead of/behind the drone, on the road
MAX_DISTANCE_FROM_DRONE_PX = 900.0  # beyond this, a car is dropped from the list


@dataclass
class Car:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    color: tuple[int, int, int]


class CarManager:
    """All mutable fields are guarded by a lock: `update()` runs on the PhysicsLoop
    thread, `snapshot()` on the VideoStreamer thread for rendering - same convention
    as DroneState (see drone/state.py)."""

    def __init__(self, seed: int = 7) -> None:
        self._lock = threading.Lock()
        self._rng = random.Random(seed)
        self._cars: list[Car] = []
        self._next_id = 1
        self._last_spawn_time = 0.0

    def update(self, dt: float, drone_x: float, drone_y: float) -> None:
        with self._lock:
            now = time.time()
            if now - self._last_spawn_time >= SPAWN_INTERVAL_SEC:
                self._last_spawn_time = now
                self._spawn(drone_x, drone_y)

            for car in self._cars:
                car.x += car.vx * dt
                car.y += car.vy * dt

            self._cars = [
                c
                for c in self._cars
                if (c.x - drone_x) ** 2 + (c.y - drone_y) ** 2 <= MAX_DISTANCE_FROM_DRONE_PX**2
            ]

    def _spawn(self, drone_x: float, drone_y: float) -> None:
        # Inside the road band, at a fixed distance ahead of or behind the
        # drone - a car on the road would appear in front of or behind you.
        edge = ROAD_HALF_WIDTH - CAR_MARGIN_FROM_ROAD_EDGE
        x = drone_x + self._rng.uniform(-edge, edge)
        y = drone_y + self._rng.choice([-1, 1]) * SPAWN_RADIUS_PX

        # Straight along the road (Y axis only), like driving in a lane -
        # not free-floating in a random direction.
        vx = 0.0
        vy = CAR_SPEED_PX_S * self._rng.choice([-1, 1])

        color = self._rng.choice(CAR_COLORS_BGR)
        self._cars.append(Car(id=self._next_id, x=x, y=y, vx=vx, vy=vy, color=color))
        self._next_id += 1

    def snapshot(self) -> list[Car]:
        with self._lock:
            return [Car(c.id, c.x, c.y, c.vx, c.vy, c.color) for c in self._cars]

    def clear(self) -> None:
        """Drop every car and reset the spawn clock - called once when the
        Drone goes back to idle (see drone/physics.py), so a fresh START
        doesn't inherit cars left over from the previous run."""
        with self._lock:
            self._cars = []
            self._last_spawn_time = 0.0
