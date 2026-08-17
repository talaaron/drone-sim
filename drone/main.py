"""
drone/main.py — entry point for the Drone Simulator (Task 1, Server).

Runs four independent threads around one shared DroneState:
    PhysicsLoop        - advances position/battery at a fixed rate (50Hz)
    CommandReceiver    - receives movement commands from the GCS (UDP)
    VideoStreamer      - streams video (UDP, JPEG+chunking)
    TelemetryStreamer  - streams telemetry (UDP, JSON, 10Hz)

Run:
    python3 -m drone.main
    python3 -m drone.main --drop-probability 0.05   # deliberate video-loss testing

Telemetry loss isn't a flag here - it's always on by default, at the
rate set by DEFAULT_TELEMETRY_DROP_PROBABILITY in network.py. See
that constant's comment for why.
"""

from __future__ import annotations

import argparse
import signal
import time

from shared import protocol

from .cars import CarManager
from .network import CommandReceiver, TelemetryStreamer, VideoStreamer
from .physics import PhysicsLoop
from .state import DroneState
from .world import World


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone simulator (Server)")
    parser.add_argument("--command-port", type=int, default=protocol.COMMAND_PORT)
    parser.add_argument("--video-port", type=int, default=protocol.VIDEO_PORT)
    parser.add_argument("--telemetry-port", type=int, default=protocol.TELEMETRY_PORT)
    parser.add_argument("--fps", type=float, default=protocol.VIDEO_FPS)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument(
        "--drop-probability",
        type=float,
        default=0.0,
        help="Resilience testing only: probability (0..1) of deliberately dropping a video "
        "chunk. Telemetry loss isn't a flag - it's always on by default; see "
        "network.DEFAULT_TELEMETRY_DROP_PROBABILITY",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed for world generation (obstacles)")
    parser.add_argument(
        "--no-cars",
        action="store_true",
        help="disable the Challenge (car spawning) and run only the base Task 1 simulator",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    state = DroneState()
    world = World(seed=args.seed)
    car_manager = None if args.no_cars else CarManager(seed=args.seed)

    threads = [
        PhysicsLoop(state, car_manager=car_manager),
        CommandReceiver(state, port=args.command_port),
        VideoStreamer(
            state,
            world,
            car_manager=car_manager,
            fps=args.fps,
            port=args.video_port,
            jpeg_quality=args.jpeg_quality,
            drop_probability=args.drop_probability,
        ),
        TelemetryStreamer(state, port=args.telemetry_port),  # drop rate: its own default (see network.py)
    ]

    print(f"[drone] listening for commands on UDP:{args.command_port}")
    print(f"[drone] will stream video to UDP:{args.video_port} and telemetry to UDP:{args.telemetry_port}")
    print("[drone] waiting for the first command from the GCS to learn its address (auto-registration)...")

    for t in threads:
        t.start()

    stop_requested = False

    def handle_sigint(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop_requested:
            time.sleep(0.2)
    finally:
        print("\n[drone] shutting down...")
        for t in threads:
            t.stop()
        for t in threads:
            t.join(timeout=2.0)
        print("[drone] shut down cleanly.")


if __name__ == "__main__":
    main()
