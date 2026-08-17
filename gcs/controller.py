"""
gcs/controller.py — sends movement commands to the Drone.

Throttle control, not hold-to-move: each key *tap* (not a hold)
adds or subtracts 0.1 from the requested velocity on that axis,
and it stays there until the next tap - no need to hold the key
down. See main_window.py: keyPressEvent calls nudge() once per
real key press, OS autorepeat filtered out.

No separate thread here: sendto over UDP doesn't block, so there's
nothing to gain from one. main_window.py drives send_current() via
a QTimer on the main event loop, at a fixed rate, always - even
with no new key press. That keeps the Drone fed a steady heartbeat
(avoids its safety timeout) and keeps the last requested velocity
going out while the user isn't touching the keyboard, which is
what produces the throttle behavior.
"""

from __future__ import annotations

import socket

from shared import protocol

SEND_INTERVAL_MS = 100  # 10Hz - roughly matches the telemetry rate, plenty for responsiveness
VELOCITY_STEP = 0.1  # how much a single key press adds/subtracts from velocity on that axis


class CommandSender:
    def __init__(self, drone_host: str, port: int = protocol.COMMAND_PORT) -> None:
        self._addr = (drone_host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._vx = 0.0
        self._vy = 0.0
        self._start_requested = False

    @property
    def velocity(self) -> tuple[float, float]:
        return self._vx, self._vy

    def set_velocity(self, vx: float, vy: float) -> None:
        self._vx = protocol.clamp(vx)
        self._vy = protocol.clamp(vy)

    def nudge(self, dvx: float, dvy: float) -> None:
        """Add to (or, with a negative dv, subtract from) the current velocity -
        the core of the tap = +0.1 control scheme."""
        self.set_velocity(self._vx + dvx, self._vy + dvy)

    def stop(self) -> None:
        self.set_velocity(0.0, 0.0)

    def request_start(self) -> None:
        """Called once when the START button is clicked. Latches until the
        next send_current() call, which ships it on exactly one Command
        packet and then clears it - a one-shot, not a held state. Sending
        start=True forever would auto-relaunch the Drone the instant its
        battery dies and it drops back to idle, instead of waiting for an
        actual new button press."""
        self._start_requested = True

    def send_current(self) -> None:
        cmd = protocol.Command(vx=self._vx, vy=self._vy, start=self._start_requested)
        self._sock.sendto(protocol.encode_command(cmd), self._addr)
        self._start_requested = False
