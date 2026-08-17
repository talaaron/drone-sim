"""
drone/network.py — the Drone's network layer: three independent
threads, each touching only the shared `DroneState` (via its
internal Lock) and never each other:

  CommandReceiver   - listens on COMMAND_PORT, updates state.vx/vy,
                       auto-registers the GCS's address (see
                       shared/protocol.py).
  VideoStreamer     - each tick: renders a frame from the current
                       state, JPEG-encodes it, splits it into
                       chunks, sends it to the GCS's VIDEO_PORT.
  TelemetryStreamer - each tick: sends a JSON telemetry packet to
                       TELEMETRY_PORT.

Each streamer takes an optional `drop_probability`. VideoStreamer
defaults to 0 - off unless explicitly requested via drone/main.py's
--drop-probability, for deliberate video-loss testing.
TelemetryStreamer defaults to DEFAULT_TELEMETRY_DROP_PROBABILITY
below instead of 0: the spec requires the transmission design to
account for real-world network conditions, packet loss specifically
(SW_Assigment_2026.pdf), so telemetry loss is simulated by default
on every run, not only when someone remembers to opt in - the GCS's
"packets lost" counter (gcs/main_window.py) is always live proof the
system copes with it, not a demo mode. A single shared rate for both
streams would be misleading regardless: a dropped video chunk only
cripples the one frame it belongs to, while telemetry packets aren't
chunked at all, so the same probability means something very
different for each stream.
"""

from __future__ import annotations

import random
import socket
import threading
import time

import cv2

from shared import protocol

from .cars import CarManager
from .render import render_frame
from .state import DroneState
from .world import World


class CommandReceiver(threading.Thread):
    def __init__(self, state: DroneState, port: int = protocol.COMMAND_PORT) -> None:
        super().__init__(daemon=True, name="CommandReceiver")
        self._state = state
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.5)  # lets the loop re-check stop_event periodically
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                data, (ip, _port) = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break  # socket closed by stop()

            cmd = protocol.decode_command(data)
            if cmd is not None:
                self._state.apply_command(cmd.vx, cmd.vy, sender_ip=ip, start_requested=cmd.start)

    def stop(self) -> None:
        self._stop_event.set()
        self._sock.close()


class VideoStreamer(threading.Thread):
    def __init__(
        self,
        state: DroneState,
        world: World,
        car_manager: CarManager | None = None,
        fps: float = protocol.VIDEO_FPS,
        port: int = protocol.VIDEO_PORT,
        jpeg_quality: int = 70,
        drop_probability: float = 0.0,
    ) -> None:
        super().__init__(daemon=True, name="VideoStreamer")
        self._state = state
        self._world = world
        self._car_manager = car_manager
        self._interval = 1.0 / fps
        self._port = port
        self._jpeg_quality = jpeg_quality
        self._drop_probability = drop_probability
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._frame_id = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            snap = self._state.snapshot()
            if snap.gcs_addr and snap.gcs_addr[0]:
                try:
                    self._send_frame(snap)
                except OSError:
                    break  # socket closed by stop() mid-send - normal shutdown
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, self._interval - elapsed))

    def _send_frame(self, snap) -> None:
        cars = self._car_manager.snapshot() if self._car_manager is not None else None
        frame = render_frame(snap, self._world, cars)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not ok:
            return
        chunks = protocol.split_into_chunks(buf.tobytes())
        dest = (snap.gcs_addr[0], self._port)
        for idx, chunk in enumerate(chunks):
            if self._drop_probability and random.random() < self._drop_probability:
                continue  # simulated packet loss - see WORK_PLAN.md, resilience testing
            packet = protocol.pack_video_chunk(self._frame_id, idx, len(chunks), chunk)
            self._sock.sendto(packet, dest)
        self._frame_id += 1

    def stop(self) -> None:
        self._stop_event.set()
        self._sock.close()


# Always-on by default (see the module docstring) - not a CLI flag, so
# there's no run where telemetry loss handling isn't actually exercised.
# Change this to tune it; 0.0 turns it off entirely.
DEFAULT_TELEMETRY_DROP_PROBABILITY = 0.5


class TelemetryStreamer(threading.Thread):
    def __init__(
        self,
        state: DroneState,
        hz: float = protocol.TELEMETRY_HZ,
        port: int = protocol.TELEMETRY_PORT,
        drop_probability: float = DEFAULT_TELEMETRY_DROP_PROBABILITY,
    ) -> None:
        super().__init__(daemon=True, name="TelemetryStreamer")
        self._state = state
        self._interval = 1.0 / hz
        self._port = port
        self._drop_probability = drop_probability
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop_event = threading.Event()
        self._seq = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            snap = self._state.snapshot()
            if snap.gcs_addr and snap.gcs_addr[0]:
                try:
                    self._send(snap)
                except OSError:
                    break  # socket closed by stop() mid-send - normal shutdown
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, self._interval - elapsed))

    def _send(self, snap) -> None:
        send_this_one = not (self._drop_probability and random.random() < self._drop_probability)
        if send_this_one:
            tm = protocol.Telemetry(
                speed=snap.speed_kmh,
                battery=snap.battery_pct,
                last_cmd={"vx": snap.vx, "vy": snap.vy},
                seq=self._seq,
                started=snap.started,
            )
            self._sock.sendto(protocol.encode_telemetry(tm), (snap.gcs_addr[0], self._port))
        # seq advances even on a dropped packet, so the GCS keeps a
        # continuous sequence it can use to detect loss.
        self._seq += 1

    def stop(self) -> None:
        self._stop_event.set()
        self._sock.close()
