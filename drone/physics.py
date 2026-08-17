"""
drone/physics.py — the physics loop: advances the Drone's state at
its own fixed rate, independent of any streaming rate.

Video sends at VIDEO_FPS, telemetry at TELEMETRY_HZ - two different
rates, so physics stepping can't be tied to either. Each consumer
just calls state.snapshot() and gets the current state.
"""

from __future__ import annotations

import threading
import time

from shared import protocol

from .cars import CarManager
from .state import DroneState

PHYSICS_HZ = 50.0


class PhysicsLoop(threading.Thread):
    def __init__(
        self,
        state: DroneState,
        car_manager: CarManager | None = None,
        hz: float = PHYSICS_HZ,
        stale_timeout_sec: float = protocol.COMMAND_STALE_TIMEOUT_SEC,
    ) -> None:
        super().__init__(daemon=True, name="PhysicsLoop")
        self._state = state
        self._car_manager = car_manager
        self._dt = 1.0 / hz
        self._stale_timeout = stale_timeout_sec
        self._stop_event = threading.Event()

    def run(self) -> None:
        was_started = False
        while not self._stop_event.is_set():
            start = time.monotonic()
            self._state.step(self._dt, self._stale_timeout)
            if self._car_manager is not None:
                snap = self._state.snapshot()
                if snap.started:
                    self._car_manager.update(self._dt, snap.x, snap.y)
                elif was_started:
                    # Just dropped back to idle (battery died) - clear out
                    # cars from the run that just ended instead of leaving
                    # them frozen on the idle screen.
                    self._car_manager.clear()
                was_started = snap.started
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, self._dt - elapsed))

    def stop(self) -> None:
        self._stop_event.set()
