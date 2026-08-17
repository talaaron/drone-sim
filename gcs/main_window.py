"""
gcs/main_window.py — the GCS window: live video display, telemetry
dashboard, and throttle-style keyboard control (WASD/arrows). Each
key press adds 0.1 to the requested velocity on that axis; no need
to hold the key down for the Drone to keep moving.

Key principle: the QThreads (video_receiver, telemetry_receiver)
never touch widgets directly. They only emit a pyqtSignal; the
slots that update the UI (the _on_* methods below) always run on
the main thread, because Qt delivers signal->slot calls through
the event loop even across threads. That's the only correct way to
update a GUI from data arriving on a network thread.
"""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QKeyEvent, QPixmap
from PyQt6.QtWidgets import (
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from shared import protocol

from .controller import SEND_INTERVAL_MS, VELOCITY_STEP, CommandSender
from .detector import CentroidTracker, NaiveCarDetector, draw_tracks
from .telemetry_receiver import TelemetryReceiver
from .video_receiver import VideoReceiver

# Key mapping: forward/backward = Vy, left/right = Vx (same
# convention as drone/state.py). Each key maps to a (dvx, dvy)
# delta added to the requested velocity on every press.
#
# W/A/S/D are bound for convention's sake, but Qt.Key.Key_W etc.
# only fire when the OS keyboard layout is Latin - a Hebrew (or any
# non-Latin) layout sends a different keysym for that same physical
# key, so it never matches. The arrow keys aren't letters and don't
# have this problem in any layout, which is why the on-screen
# instructions below lead with them rather than WASD.
KEY_DELTAS: dict[int, tuple[float, float]] = {
    Qt.Key.Key_A: (-VELOCITY_STEP, 0.0),
    Qt.Key.Key_Left: (-VELOCITY_STEP, 0.0),
    Qt.Key.Key_D: (VELOCITY_STEP, 0.0),
    Qt.Key.Key_Right: (VELOCITY_STEP, 0.0),
    Qt.Key.Key_W: (0.0, VELOCITY_STEP),
    Qt.Key.Key_Up: (0.0, VELOCITY_STEP),
    Qt.Key.Key_S: (0.0, -VELOCITY_STEP),
    Qt.Key.Key_Down: (0.0, -VELOCITY_STEP),
}
STOP_KEY = Qt.Key.Key_Space

# Brighter than a normal-form green/red on purpose: these render on
# the dark semi-transparent overlay drawn over the video (see
# _build_ui), not the window's plain background, and a darker pair
# was hard to read against near-black. `background: transparent`
# matters too - setStyleSheet() replaces the label's whole style,
# including the overlay's own "transparent QLabel" rule, so it has
# to be restated here or the label reverts to an opaque box the
# first time connection status changes.
STYLE_OK = "color: #66ff66; font-weight: bold; background: transparent;"
STYLE_BAD = "color: #ff6b6b; font-weight: bold; background: transparent;"

# Above this cumulative loss rate, the packets-lost label switches to
# STYLE_BAD so a bad link is visible at a glance instead of only as a
# slowly climbing raw count - the count alone doesn't say whether it's
# 12 lost out of 3000 (fine) or 12 out of 20 (not fine).
LOST_RATE_WARN_PCT = 20.0


class MainWindow(QMainWindow):
    def __init__(self, drone_host: str) -> None:
        super().__init__()
        self.setWindowTitle("Ground Control Station")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # needed for keyboard events

        self._total_lost_telemetry = 0
        self._total_telemetry_received = 0
        # Mirrors the Drone's own started flag (see drone/state.py),
        # updated from telemetry - drives the START button's
        # visibility and whether keyboard input does anything.
        self._started = False

        self._controller = CommandSender(drone_host)
        self._video_thread = VideoReceiver()
        self._telemetry_thread = TelemetryReceiver()

        # Challenge: naive detection/tracking, run on the main
        # thread inside _on_frame rather than a separate thread -
        # contour detection on a 640x480 frame takes a few
        # milliseconds, not enough to block noticeably.
        self._car_detector = NaiveCarDetector()
        self._car_tracker = CentroidTracker()

        self._build_ui()
        self._wire_signals()

        # Sends commands at a fixed rate, even Vx=Vy=0 - see gcs/controller.py
        self._send_timer = QTimer(self)
        self._send_timer.timeout.connect(self._controller.send_current)
        self._send_timer.start(SEND_INTERVAL_MS)

        self._video_thread.start()
        self._telemetry_thread.start()

    # -- Building the UI ------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # -- Video display --
        self.video_label = QLabel("Waiting for video from the drone…")
        self.video_label.setFixedSize(protocol.VIDEO_FRAME_WIDTH, protocol.VIDEO_FRAME_HEIGHT)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: gray; border: 1px solid #444;")
        root.addWidget(self.video_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        # -- Status/telemetry readout: overlaid on the video itself,
        # bottom-left corner, not stacked below it as separate
        # widgets. A semi-transparent panel parented directly to
        # video_label and positioned with .move() once its size is
        # known - QLabel has no layout of its own to dock a
        # corner-anchored child to.
        #
        # Scoped with #videoOverlay so the rgba() background applies
        # only to this panel, not - via Qt's normal cascading - to
        # the QLabel children too; each label gets its own rule
        # right after, with an explicit transparent background. A
        # bare "background: rgba(...); QLabel {...}" mix, with no
        # selector on the first declaration, is invalid Qt
        # stylesheet syntax and rendered as an opaque per-label box
        # instead of one merged translucent panel.
        overlay = self._overlay = QWidget(self.video_label)
        overlay.setObjectName("videoOverlay")
        overlay.setStyleSheet(
            "#videoOverlay { background-color: rgba(0, 0, 0, 150); }"
            "#videoOverlay QLabel { color: white; background: transparent; }"
        )
        overlay_layout = QVBoxLayout(overlay)
        overlay_layout.setContentsMargins(6, 4, 6, 4)
        overlay_layout.setSpacing(2)

        self.video_status_label = QLabel("Video: waiting…")
        self.telemetry_status_label = QLabel("Telemetry: waiting…")
        self.speed_label = QLabel("Speed: —")
        # Local feedback, shown immediately rather than waiting on
        # the telemetry round-trip, of what's actually being sent to
        # the Drone - so a key press shows up right away instead of
        # after the ~100ms telemetry delay.
        # Initial text matches the two-line shape each label gets once its
        # update handler actually fires - _on_telemetry runs at 10Hz so
        # last_cmd_label corrects itself almost immediately, but
        # _update_requested_velocity_label only fires on a keypress and
        # _on_packets_lost only when a real gap shows up in the telemetry
        # sequence. Without this, either label could sit on its one-line
        # placeholder indefinitely, and the overlay would visibly resize
        # the moment it finally switched to two lines.
        self.requested_velocity_label = QLabel("Requested velocity: \nvx=+0.00  vy=+0.00")
        self.last_cmd_label = QLabel("Last command: \nvx=+0.00  vy=+0.00")
        self.lost_label = QLabel("Telemetry packets lost \n(cumulative): 0 (0%)")
        for label in (
            self.video_status_label,
            self.telemetry_status_label,
            self.speed_label,
            self.requested_velocity_label,
            self.last_cmd_label,
            self.lost_label,
        ):
            overlay_layout.addWidget(label)

        self._reposition_overlay()

        # -- START button: centered over the video, same corner-docking
        # trick as the status overlay above (parented directly to
        # video_label, positioned with .move()). Visible only while
        # idle (self._started is False) - see _on_telemetry, which
        # hides it once the Drone confirms it's actually running and
        # shows it again the moment the battery dies and the Drone
        # drops back to idle on its own.
        self.start_button = QPushButton("▶  START", self.video_label)
        self.start_button.setStyleSheet(
            "QPushButton { font-size: 18px; font-weight: bold; padding: 12px 26px;"
            " background-color: rgba(46, 125, 50, 235); color: white;"
            " border: none; border-radius: 8px; }"
            "QPushButton:hover { background-color: rgba(56, 142, 60, 235); }"
        )
        self.start_button.clicked.connect(self._on_start_clicked)
        self._center_start_button()

        # -- Battery: kept outside the video (a QProgressBar doesn't
        # read well as an overlay), in the normal panel below it. One
        # widget instead of a bar + a separate label duplicating the
        # same number: QProgressBar's format string bakes "Battery"
        # and the live percentage (%p, kept in sync by setValue()
        # alone - no separate setText() needed) directly onto the bar,
        # so the meter is still unambiguous without a second widget.
        self.battery_bar = QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setFormat("Battery: %p%")
        root.addWidget(self.battery_bar)

        instructions = QLabel(
            f"Controls: every press of the arrow keys adds {VELOCITY_STEP:+.1f} to the velocity "
            "on that axis - no need to hold it down. Space = stop immediately. (W/A/S/D are bound "
            "too, but only register while the OS keyboard layout is set to English - the arrow "
            "keys work regardless of layout.)"
        )
        instructions.setStyleSheet("color: gray;")
        instructions.setWordWrap(True)
        root.addWidget(instructions)

    def _wire_signals(self) -> None:
        self._video_thread.frame_ready.connect(self._on_frame)
        self._video_thread.connection_changed.connect(self._on_video_connection)
        self._telemetry_thread.telemetry_ready.connect(self._on_telemetry)
        self._telemetry_thread.connection_changed.connect(self._on_telemetry_connection)
        self._telemetry_thread.packets_lost.connect(self._on_packets_lost)

    # -- Slots: run on the main thread even though the signal comes from a QThread --

    def _reposition_overlay(self) -> None:
        """Recompute the overlay panel's size and position.

        Needed on every text change, not just at startup: several labels
        switch between one and two lines at runtime (the "\\n" in the
        .setText() calls below - QLabel honors "\\n" as a real line break,
        unlike cv2.putText on the drone side), and connection-status text
        ("connected" vs. "no signal ⚠") changes width too. adjustSize()
        picks up whatever the labels currently need, then we re-anchor to
        the video's bottom-left corner with that fresh size - otherwise the
        panel keeps whatever size it had when first laid out, and newer,
        taller text spills out past its dark background instead of the
        panel growing to fit.
        """
        self._overlay.adjustSize()
        self._overlay.move(6, self.video_label.height() - self._overlay.height() - 6)

    def _center_start_button(self) -> None:
        self.start_button.adjustSize()
        x = (self.video_label.width() - self.start_button.width()) // 2
        y = (self.video_label.height() - self.start_button.height()) // 2
        self.start_button.move(x, y)

    def _on_start_clicked(self) -> None:
        self._controller.request_start()

    def _on_frame(self, frame: np.ndarray) -> None:
        detections = self._car_detector.detect(frame)
        tracks = self._car_tracker.update(detections)
        draw_tracks(frame, tracks)  # in-place, before converting to QImage

        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(qimg))
        # setPixmap() on the parent can leave the translucent
        # overlay's stacking/repaint slightly out of sync with the
        # frame just set underneath it, since video updates at
        # ~20fps and the overlay only repaints on its own content
        # changes. raise_() puts it back on top so Qt repaints it
        # cleanly against the new frame.
        self._overlay.raise_()

    def _on_telemetry(self, tm: protocol.Telemetry) -> None:
        if tm.started != self._started:
            self._started = tm.started
            self.start_button.setVisible(not self._started)
            # Packets-lost stats are scoped to one run, not the app's
            # whole lifetime - reset on every idle<->running edge (not
            # just idle->running) so a fresh START always starts the
            # count at 0 instead of carrying over whatever the last run
            # (or the idle waiting period before the very first START)
            # already racked up. Done before the increment below, not
            # after, so the telemetry packet carrying this transition
            # still counts as the new run's first received packet
            # instead of being wiped out along with the old totals.
            self._total_lost_telemetry = 0
            self._total_telemetry_received = 0
            if not self._started:
                # The Drone auto-resets (battery died) or this is the
                # initial idle state - zero out the locally-requested
                # velocity too, so this readout doesn't sit on a stale
                # "vy=+1.00" from the run that just ended while the
                # Drone itself has already gone back to 0.
                self._controller.stop()
                self._update_requested_velocity_label()

        self._total_telemetry_received += 1
        self._update_lost_label()
        self.speed_label.setText(f"Speed: {tm.speed:.1f} km/h")
        self.battery_bar.setValue(int(tm.battery))
        self.battery_bar.setStyleSheet("" if tm.battery > 20 else "QProgressBar::chunk { background-color: #c62828; }")
        vx = tm.last_cmd.get("vx", 0.0)
        vy = tm.last_cmd.get("vy", 0.0)
        self.last_cmd_label.setText(f"Last command: \nvx={vx:+.2f}  vy={vy:+.2f}")
        self._reposition_overlay()

    def _on_video_connection(self, connected: bool) -> None:
        self.video_status_label.setText("Video: connected" if connected else "Video: no signal ⚠")
        self.video_status_label.setStyleSheet(STYLE_OK if connected else STYLE_BAD)
        self._reposition_overlay()

    def _on_telemetry_connection(self, connected: bool) -> None:
        self.telemetry_status_label.setText("Telemetry: connected" if connected else "Telemetry: no signal ⚠")
        self.telemetry_status_label.setStyleSheet(STYLE_OK if connected else STYLE_BAD)
        self._reposition_overlay()

    def _on_packets_lost(self, n: int) -> None:
        self._total_lost_telemetry += n
        self._update_lost_label()

    def _update_lost_label(self) -> None:
        total = self._total_telemetry_received + self._total_lost_telemetry
        rate_pct = (self._total_lost_telemetry / total * 100) if total else 0.0
        self.lost_label.setText(
            f"Telemetry packets lost \n(cumulative): {self._total_lost_telemetry} ({rate_pct:.0f}%)"
        )
        self.lost_label.setStyleSheet(STYLE_BAD if rate_pct >= LOST_RATE_WARN_PCT else "")
        self._reposition_overlay()

    # -- Keyboard ---------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.isAutoRepeat():
            # Ignore OS autorepeat - the repeated press events fired
            # while a key is held - or holding a key would keep
            # adding 0.1 over and over, when the goal is one press,
            # one step. The QTimer already resends the requested
            # velocity continuously on its own.
            return

        key = event.key()
        if key in KEY_DELTAS:
            if not self._started:
                # Nothing to nudge before START - the Drone ignores
                # movement while idle anyway (drone/state.py), so
                # accepting it here would just desync the "requested
                # velocity" readout from what's actually happening.
                return
            dvx, dvy = KEY_DELTAS[key]
            self._controller.nudge(dvx, dvy)
            self._update_requested_velocity_label()
        elif key == STOP_KEY:
            self._controller.stop()
            self._update_requested_velocity_label()
        else:
            super().keyPressEvent(event)

    def _update_requested_velocity_label(self) -> None:
        vx, vy = self._controller.velocity
        self.requested_velocity_label.setText(f"Requested velocity: \nvx={vx:+.2f}  vy={vy:+.2f}")
        self._reposition_overlay()

    # -- Graceful shutdown --------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (method name is dictated by Qt)
        self._send_timer.stop()
        self._video_thread.stop()
        self._telemetry_thread.stop()
        self._video_thread.wait(1000)
        self._telemetry_thread.wait(1000)
        super().closeEvent(event)
