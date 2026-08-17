"""
gcs/detector.py — Challenge: a naive color-based detector and
tracker, a lightweight demo of the tracking principle. Not the
production pipeline - that's documented separately in the
architecture write-up (DOCS - YOLO11n + ByteTrack + ray-plane
triangulation). A deliberate choice (see DOCS/WORK_PLAN.md,
"Challenge scope decision"): prove the principle in practice
without training a model.

How it works
------------
1. Detection: drone/cars.py draws every car in a saturated hue -
   yellow, red, green - distinct from the road/off-road hues
   (hueless gray, dark blue). Saturation alone isn't enough to
   separate them: the dark-blue off-road margins (see
   drone/render.py) are nearly as saturated as the cars. The real
   separation is hue - convert to HSV, keep only pixels in the car
   hue range (red/yellow/green, not blue) that are also saturated
   enough to exclude the hueless road/sidewalk gray.
   cv2.findContours finds the connected blobs; each one above a
   minimum size is a detection (bounding box). This only holds up
   because the simulator's colors are ours to control - a real
   camera under changing light needs a trained model instead.
2. Tracking: `CentroidTracker` matches each detection to the
   nearest existing track (Euclidean distance between centers)
   under a threshold. A track that goes unmatched for several
   frames in a row - brief occlusion, or the car left the frame -
   is kept for a few more frames before deletion. Same idea as
   ByteTrack's IoU+Kalman matching, without a real motion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# HSV, OpenCV scale (H in 0..179). The cars (red=0, yellow=24,
# green=60, see drone/cars.py) all fall below HUE_MAX; the
# dark-blue off-road margins sit at hue ~111, well above the
# threshold, despite having similar saturation to the cars.
HUE_MAX = 90
SATURATION_THRESHOLD = 130  # excludes the hueless road/sidewalk gray (S≈0)
MIN_VALUE = 40  # ignores pixels too dark to trust (noise, shadow)
MIN_CONTOUR_AREA = 60
MAX_MATCH_DISTANCE_PX = 60.0
MAX_MISSED_FRAMES = 10

# Radius around the frame center to ignore - the Drone's own
# marker is always drawn there. Its color (azure, see
# drone/render.py) already sits outside the car hue range, but
# this is a cheap extra guard against false positives from
# anti-aliased edge pixels where the marker meets the road.
DRONE_MARKER_EXCLUSION_RADIUS_PX = 20


@dataclass
class Detection:
    cx: float
    cy: float
    w: int
    h: int


@dataclass
class Track:
    id: int
    cx: float
    cy: float
    w: int
    h: int
    missed_frames: int = 0


class NaiveCarDetector:
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        is_car_colored = (hue <= HUE_MAX) & (sat > SATURATION_THRESHOLD) & (val > MIN_VALUE)
        mask = is_car_colored.astype(np.uint8) * 255

        h, w = mask.shape
        cv2.circle(mask, (w // 2, h // 2), DRONE_MARKER_EXCLUSION_RADIUS_PX, 0, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            detections.append(Detection(cx=x + bw / 2, cy=y + bh / 2, w=bw, h=bh))
        return detections


class CentroidTracker:
    def __init__(self) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = list(detections)
        matched_ids: set[int] = set()

        # Greedy matching: each existing track claims its closest
        # still-unmatched detection.
        for track in self._tracks.values():
            best, best_dist = None, MAX_MATCH_DISTANCE_PX
            for det in unmatched:
                dist = ((track.cx - det.cx) ** 2 + (track.cy - det.cy) ** 2) ** 0.5
                if dist < best_dist:
                    best, best_dist = det, dist
            if best is not None:
                track.cx, track.cy, track.w, track.h = best.cx, best.cy, best.w, best.h
                track.missed_frames = 0
                matched_ids.add(track.id)
                unmatched.remove(best)

        for track in list(self._tracks.values()):
            if track.id not in matched_ids:
                track.missed_frames += 1
                if track.missed_frames > MAX_MISSED_FRAMES:
                    del self._tracks[track.id]

        for det in unmatched:  # no existing track claimed it -> new track, new ID
            track = Track(id=self._next_id, cx=det.cx, cy=det.cy, w=det.w, h=det.h)
            self._tracks[track.id] = track
            self._next_id += 1

        return list(self._tracks.values())


def draw_tracks(frame_bgr: np.ndarray, tracks: list[Track]) -> None:
    """Draw an overlay (a white box + ID) directly onto the frame, in-place."""
    for t in tracks:
        x0, y0 = int(t.cx - t.w / 2) - 2, int(t.cy - t.h / 2) - 2
        x1, y1 = int(t.cx + t.w / 2) + 2, int(t.cy + t.h / 2) + 2
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(
            frame_bgr,
            f"#{t.id}",
            (x0, max(10, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
