"""
gcs/main.py — entry point for the Ground Control Station (Task 2,
Client).

Unlike the Drone, which auto-learns the GCS's address, the GCS
does need to know the Drone's address up front - it's the side
that initiates the first send. The default 127.0.0.1 is right for
running both nodes locally; pass --drone-host to run across two
machines on the network.

Run:
    python3 -m gcs.main
    python3 -m gcs.main --drone-host 192.168.1.23
"""

from __future__ import annotations

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground Control Station (Client)")
    parser.add_argument("--drone-host", default="127.0.0.1", help="the Drone's IP address (default: localhost)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    window = MainWindow(drone_host=args.drone_host)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
