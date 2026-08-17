"""
gcs/telemetry_receiver.py — receives telemetry packets from the
Drone, on a dedicated QThread (same reasoning as
video_receiver.py: recvfrom blocks, and UI updates are only
allowed from the main thread, so everything is handed off through
a pyqtSignal instead).

Tracks the `seq` field in every packet to detect and report lost
telemetry - the UDP-reliability awareness that WORK_PLAN.md leans
on to defend the protocol choice.
"""

from __future__ import annotations

import socket
import time

from PyQt6.QtCore import QThread, pyqtSignal

from shared import protocol


class TelemetryReceiver(QThread):
    telemetry_ready = pyqtSignal(object)  # protocol.Telemetry
    connection_changed = pyqtSignal(bool)
    packets_lost = pyqtSignal(int)  # number of packets detected as lost since the last report

    NO_SIGNAL_TIMEOUT_SEC = 2.0

    def __init__(self, port: int = protocol.TELEMETRY_PORT, parent=None) -> None:
        super().__init__(parent)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.5)
        self._running = True

    def run(self) -> None:
        last_seq: int | None = None
        last_time = 0.0
        connected = False

        while self._running:
            try:
                data, _addr = self._sock.recvfrom(4096)
            except socket.timeout:
                if connected and time.time() - last_time > self.NO_SIGNAL_TIMEOUT_SEC:
                    connected = False
                    self.connection_changed.emit(False)
                continue
            except OSError:
                break

            tm = protocol.decode_telemetry(data)
            if tm is None:
                continue

            if last_seq is not None and tm.seq > last_seq + 1:
                self.packets_lost.emit(tm.seq - last_seq - 1)
            last_seq = tm.seq
            last_time = time.time()

            self.telemetry_ready.emit(tm)
            if not connected:
                connected = True
                self.connection_changed.emit(True)

    def stop(self) -> None:
        self._running = False
        self._sock.close()
