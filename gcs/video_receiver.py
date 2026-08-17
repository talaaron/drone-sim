"""
gcs/video_receiver.py — receives the video stream from the Drone,
on a dedicated QThread.

Needs its own thread rather than Qt's main event loop because
recvfrom blocks. Qt widgets may only be updated from the main
thread, though, so this thread never touches the UI directly - it
hands off a decoded frame through a pyqtSignal, and Qt schedules
the connected slot to run on the main thread.

Handling loss and fragmentation:
  - Every incoming chunk is filed under its frame_id in a temporary dict.
  - Once all of a frame's chunks have arrived, they're reassembled and emitted.
  - A stuck frame (some chunks never arrived) is dropped after FRAME_TIMEOUT_SEC.
  - Once a newer frame completes, anything still pending from an
    older frame_id is dropped immediately - no point displaying a
    stale frame once a fresher one exists. Freshness over
    completeness (see DOCS/WORK_PLAN.md).
"""

from __future__ import annotations

import socket
import time

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from shared import protocol


class VideoReceiver(QThread):
    frame_ready = pyqtSignal(np.ndarray)  # decoded BGR frame, ready to display
    connection_changed = pyqtSignal(bool)

    FRAME_TIMEOUT_SEC = 0.3
    NO_SIGNAL_TIMEOUT_SEC = 2.0

    def __init__(self, port: int = protocol.VIDEO_PORT, parent=None) -> None:
        super().__init__(parent)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.5)
        self._running = True

    def run(self) -> None:
        pending: dict[int, dict] = {}
        last_frame_time = 0.0
        connected = False

        while self._running:
            try:
                packet, _addr = self._sock.recvfrom(65536)
            except socket.timeout:
                self._drop_stale(pending)
                if connected and time.time() - last_frame_time > self.NO_SIGNAL_TIMEOUT_SEC:
                    connected = False
                    self.connection_changed.emit(False)
                continue
            except OSError:
                break

            parsed = protocol.unpack_video_chunk(packet)
            if not parsed:
                continue
            frame_id, idx, total, payload = parsed

            bucket = pending.setdefault(frame_id, {"chunks": {}, "total": total, "first_seen": time.time()})
            bucket["chunks"][idx] = payload

            if len(bucket["chunks"]) >= bucket["total"]:
                data = b"".join(bucket["chunks"][i] for i in range(bucket["total"]))
                del pending[frame_id]
                self._drop_older_than(pending, frame_id)

                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.frame_ready.emit(frame)
                    last_frame_time = time.time()
                    if not connected:
                        connected = True
                        self.connection_changed.emit(True)

            self._drop_stale(pending)

    def _drop_stale(self, pending: dict) -> None:
        now = time.time()
        for fid in [f for f, b in pending.items() if now - b["first_seen"] > self.FRAME_TIMEOUT_SEC]:
            del pending[fid]

    def _drop_older_than(self, pending: dict, newest_completed_id: int) -> None:
        for fid in [f for f in pending if f < newest_completed_id]:
            del pending[fid]

    def stop(self) -> None:
        self._running = False
        self._sock.close()
