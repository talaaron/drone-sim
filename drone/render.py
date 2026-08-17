"""
drone/render.py — produce a single top-down frame from the current
drone state.

Mirrors the official PDF's example
(`DOCS/images/road_view_example.png`): a vertical road with a
dashed centerline, white lane edges, dark off-road margins on both
sides. The colors below were sampled from pixels in that image, not
picked freehand - see the comment next to each one.

The camera is fixed at the center of the frame; the world scrolls
underneath it. Every element (road, dashed line, obstacles) is
drawn relative to the camera position (view_left, view_top), never
against a fixed screen origin. The dashed centerline is the
clearest motion cue: its segments run vertically at a rate tied
directly to Vy, the way lane markings do on a real road.
"""

from __future__ import annotations

import cv2
import numpy as np

from shared.protocol import VIDEO_FRAME_HEIGHT as FRAME_HEIGHT
from shared.protocol import VIDEO_FRAME_WIDTH as FRAME_WIDTH

from .cars import CAR_HALF_SIZE, Car
from .state import Snapshot
from .world import ROAD_HALF_WIDTH, World

# Sampled (cv2, single pixel) from DOCS/images/road_view_example.png -
# see the test script that ran them. Not freehand guesses.
COLOR_OFFROAD = (82, 35, 14)  # BGR - dark bluish margins either side of the road
COLOR_ROAD = (108, 108, 108)  # BGR - the road surface
COLOR_SIDEWALK = (137, 137, 137)  # BGR - buildings/sidewalks along the road
COLOR_LANE_LINE = (255, 255, 255)  # BGR - lane edges and the dashed centerline
# Cyan/azure on purpose, not a car hue: the direction arrow (up to
# 40px from center) extends past the exclusion radius used around
# the drone marker in gcs/detector.py. A car-colored arrow would
# register as a persistent false-positive detection there; azure
# sits outside that hue range (HUE > 90) regardless of the arrow's
# length or position.
COLOR_DRONE = (255, 180, 0)
COLOR_TEXT = (255, 255, 255)

DASH_LEN = 30
DASH_GAP = 25
DASH_PERIOD = DASH_LEN + DASH_GAP
LANE_LINE_THICKNESS = 3
CENTER_DASH_THICKNESS = 5

# >1.0 = zoomed out: the world layer is rendered at this many times the
# output frame size, then downscaled to FRAME_WIDTH x FRAME_HEIGHT - so
# 1.125x more of the world is visible in the same transmitted frame.
# 640x480 * 1.125 = 720x540 exactly (1.125 = 9/8), no rounding involved.
# The drone marker, arrow, and text overlay are drawn *after* the
# downscale, at their normal fixed pixel size - only the world (road,
# obstacles, cars) actually shrinks; the HUD stays as readable as before.
CAMERA_ZOOM = 1.2

# How bright the frame renders while idle (snap.started is False, waiting
# for the GCS's START button) - dims the whole picture rather than
# blanking it, so the feed still visibly proves it's alive.
IDLE_BRIGHTNESS = 0.35


def render_frame(snap: Snapshot, world: World, cars: list[Car] | None = None) -> np.ndarray:
    render_w = round(FRAME_WIDTH * CAMERA_ZOOM)
    render_h = round(FRAME_HEIGHT * CAMERA_ZOOM)

    world_layer = np.empty((render_h, render_w, 3), dtype=np.uint8)
    half_w, half_h = render_w / 2, render_h / 2
    view_left, view_top = snap.x - half_w, snap.y - half_h

    _draw_road(world_layer, view_left, view_top, render_w, render_h)
    _draw_obstacles(world_layer, world, view_left, view_top, render_w, render_h)
    if cars:
        _draw_cars(world_layer, cars, view_left, view_top, render_w, render_h)

    if (render_w, render_h) != (FRAME_WIDTH, FRAME_HEIGHT):
        frame = cv2.resize(world_layer, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        frame = world_layer

    _draw_drone_marker(frame, snap)

    if not snap.started:
        frame = _dim(frame, IDLE_BRIGHTNESS)

    return frame


def _dim(frame: np.ndarray, brightness: float) -> np.ndarray:
    return (frame.astype(np.float32) * brightness).astype(np.uint8)


def _draw_road(frame: np.ndarray, view_left: float, view_top: float, width: int, height: int) -> None:
    frame[:] = COLOR_OFFROAD  # off-road first, road drawn on top of it

    road_left_x = int(-ROAD_HALF_WIDTH - view_left)
    road_right_x = int(ROAD_HALF_WIDTH - view_left)
    cv2.rectangle(frame, (road_left_x, 0), (road_right_x, height), COLOR_ROAD, -1)

    cv2.line(frame, (road_left_x, 0), (road_left_x, height), COLOR_LANE_LINE, LANE_LINE_THICKNESS)
    cv2.line(frame, (road_right_x, 0), (road_right_x, height), COLOR_LANE_LINE, LANE_LINE_THICKNESS)

    # Dashed centerline: the clearest "I'm moving" cue. Segments enter
    # from the top or bottom and travel across the screen at a rate
    # tied to Vy, the way a real road's markings would.
    center_x = int(0 - view_left)
    y = -(int(view_top) % DASH_PERIOD)
    while y < height:
        y0, y1 = max(0, y), min(height, y + DASH_LEN)
        if y1 > y0:
            cv2.line(frame, (center_x, y0), (center_x, y1), COLOR_LANE_LINE, CENTER_DASH_THICKNESS)
        y += DASH_PERIOD


def _draw_obstacles(
    frame: np.ndarray, world: World, view_left: float, view_top: float, width: int, height: int
) -> None:
    for obs in world.obstacles_in_view(view_left, view_top, width, height):
        sx, sy = int(obs.x - view_left), int(obs.y - view_top)
        cv2.rectangle(frame, (sx, sy), (int(sx + obs.w), int(sy + obs.h)), COLOR_SIDEWALK, -1)


def _draw_cars(
    frame: np.ndarray, cars: list[Car], view_left: float, view_top: float, width: int, height: int
) -> None:
    hw, hh = CAR_HALF_SIZE
    for car in cars:
        sx, sy = int(car.x - view_left), int(car.y - view_top)
        if -hw <= sx <= width + hw and -hh <= sy <= height + hh:
            cv2.rectangle(frame, (sx - hw, sy - hh), (sx + hw, sy + hh), car.color, -1)


def _draw_drone_marker(frame: np.ndarray, snap: Snapshot) -> None:
    center = (FRAME_WIDTH // 2, FRAME_HEIGHT // 2)
    cv2.circle(frame, center, 6, COLOR_DRONE, -1)
    cv2.circle(frame, center, 12, COLOR_DRONE, 2)

    # Direction/speed arrow - shows a command took effect immediately,
    # before the world has scrolled enough to notice on its own.
    arrow_len = 40
    tip = (
        int(center[0] + snap.vx * arrow_len),
        int(center[1] - snap.vy * arrow_len),
    )
    if tip != center:
        cv2.arrowedLine(frame, center, tip, COLOR_DRONE, 2, tipLength=0.3)

    # fontScale=0.6, thickness=2 (up from 0.45/1): at ~10px character
    # height, LINE_AA had almost nothing to anti-alias against and
    # read as blurry even before JPEG touched it - confirmed by
    # comparing a lossless render against the JPEG-encoded one; JPEG
    # only added a little on top of blur already baked into the
    # render.
    #
    # v and battery deliberately left out here: both already ride the
    # telemetry channel and are shown by the GCS itself
    # (gcs/main_window.py) - baking them into the video too would be a
    # redundant, view-only duplicate of the same drone.state.py
    # values, sampled from a different thread at a slightly different
    # instant. pos has no telemetry equivalent, so it stays.
    # Display only: internally, Vy > 0 ("forward"/W) *decreases*
    # snap.y (see drone/state.py) - that's what makes the dashed
    # centerline and every other world element scroll down-screen on
    # forward motion, matching the feel of actually driving forward.
    # Flipping that sign here too would reverse the scroll direction.
    # Negating just for the printed text keeps "forward" reading as
    # positive, the more intuitive convention for a human glancing at
    # the HUD, without touching the physics or the rendering math.
    # "+ 0.0" folds a -0.0 result back to +0.0 (IEEE 754) so the text
    # reads "pos=(0,0)" at the origin instead of "pos=(0,-0)".
    position_text = f"pos=({snap.x:.0f},{-snap.y + 0.0:.0f})"
    cv2.putText(frame, position_text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)
