"""
drone/state.py — the Drone's single shared state.

Read/written from three threads (see drone/main.py):
  1. Command listener - updates vx/vy on each new UDP command.
  2. Renderer/streamer - reads x, y for the camera position.
  3. Telemetry streamer - reads speed, battery.

All shared fields go through methods that hold an internal
`Lock`. Without it, the renderer could read x/y mid-update from
another thread and get an inconsistent mix of old and new
values.

Direction convention (undefined by the spec, so this is ours):
  Vx > 0  = right    |  Vx < 0 = left
  Vy > 0  = forward  |  Vy < 0 = backward

Run/idle state machine: the Drone starts idle (`started=False`) -
position, velocity and speed pinned to 0, no battery drain,
rendered dimmed (see drone/render.py). A `start=True` command flips
it to running: position/battery reset to a fresh run and physics
resumes. When the battery reaches 0 the Drone drops back to idle
automatically, ready for another START press. See apply_command()
and step() below for exactly where each transition happens.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# Max speed at |V|=1.0, in virtual px/s - a unit of the
# simulation's internal world, not a screen pixel.
MAX_SPEED_PX_S = 240.0

# Max speed shown in telemetry (km/h). A dashboard number, not
# tied to the rendering physics; displayed as-is at |V|=1.0.
MAX_SPEED_KMH = 40.0

# Baseline battery drain (avionics/camera, even hovering) - %/s.
# Set so a full charge empties in exactly 60s at rest - the worst
# case for time-to-empty, since any movement only drains faster
# (see BATTERY_DRAIN_SPEED_PER_SEC below). That guarantees the
# battery is gone within a minute regardless of how it's flown.
BATTERY_DRAIN_IDLE_PER_SEC = 100.0 / 60.0
# Extra drain proportional to current speed (motors work
# harder) - additional %/s at |V|=1.0.
BATTERY_DRAIN_SPEED_PER_SEC = 0.30


@dataclass
class Snapshot:
    """A consistent snapshot of the Drone's state, taken under one lock so a
    consumer (rendering/telemetry) never sees a mix of values from different
    moments."""

    x: float
    y: float
    vx: float
    vy: float
    battery_pct: float
    speed_kmh: float
    gcs_addr: tuple[str, int] | None
    last_cmd_age_sec: float
    started: bool


class DroneState:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0

        self.battery_pct = 100.0
        self._last_cmd_ts = 0.0
        self.started = False

        # GCS address, auto-registered from the first command
        # received (see shared/protocol.py).
        self.gcs_ip: str | None = None

    # -- Writers (called from the command-listener thread) ------------------

    def apply_command(self, vx: float, vy: float, sender_ip: str, start_requested: bool = False) -> None:
        with self._lock:
            # Registration and the stale-command clock track every packet,
            # started or not - video/telemetry need to keep flowing to the
            # GCS even while idle, so the dimmed "press START" screen
            # actually shows up.
            self.gcs_ip = sender_ip
            self._last_cmd_ts = time.time()

            if start_requested and not self.started:
                self.started = True
                self.x = 0.0
                self.y = 0.0
                self.battery_pct = 100.0

            # Movement only takes effect once running - a command that
            # arrives before START (or after the battery dies) is
            # acknowledged (for auto-registration) but not acted on.
            if self.started:
                self.vx = vx
                self.vy = vy
            else:
                self.vx = 0.0
                self.vy = 0.0

    # -- Physics update (called from the physics thread, every tick) --------

    def step(self, dt: float, stale_timeout_sec: float) -> None:
        """Advance by dt seconds: move position, drain battery, and enforce
        the stale-command timeout - velocity resets to zero if the GCS stops
        sending, so the Drone doesn't drift forever after a disconnect.

        A no-op while idle (not started): position, velocity and battery all
        stay exactly where apply_command()/__init__ left them (0, 0, 100)."""
        with self._lock:
            if not self.started:
                return

            # If GCS disconnected (no heartbeat)
            if self._last_cmd_ts and (time.time() - self._last_cmd_ts) > stale_timeout_sec:
                self.vx = 0.0
                self.vy = 0.0

            self.x += self.vx * MAX_SPEED_PX_S * dt
            # Minus sign: positive Vy = "forward" = up-screen.
            self.y -= self.vy * MAX_SPEED_PX_S * dt

            speed_frac = (self.vx**2 + self.vy**2) ** 0.5
            drain = (BATTERY_DRAIN_IDLE_PER_SEC + BATTERY_DRAIN_SPEED_PER_SEC * speed_frac) * dt
            self.battery_pct -= drain

            if self.battery_pct <= 0.0:
                # Battery's dead - drop back to idle, reset for the next
                # START press rather than sitting at a permanent 0%.
                self.started = False
                self.x = 0.0
                self.y = 0.0
                self.vx = 0.0
                self.vy = 0.0
                self.battery_pct = 100.0

    # -- Readers (called from the rendering/telemetry threads) --------------

    def snapshot(self) -> Snapshot:
        with self._lock:
            speed_frac = (self.vx**2 + self.vy**2) ** 0.5
            age = (time.time() - self._last_cmd_ts) if self._last_cmd_ts else float("inf")
            return Snapshot(
                x=self.x,
                y=self.y,
                vx=self.vx,
                vy=self.vy,
                battery_pct=self.battery_pct,
                speed_kmh=speed_frac * MAX_SPEED_KMH,
                gcs_addr=(self.gcs_ip, None) if self.gcs_ip else None,
                last_cmd_age_sec=age,
                started=self.started,
            )
