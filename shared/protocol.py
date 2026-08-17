"""
shared/protocol.py — the wire format shared between the Drone
(Server) and the GCS (Client).

Change protocol fields here first. Both sides import from this
module, so there's no way for them to drift into two different
versions of the same packet.

--------------------------------------------------------------------------
Port layout (why three ports, not one)
--------------------------------------------------------------------------
A single socket serving commands, video, and telemetry would need
a "type" tag on every packet and manual dispatch. Instead each
stream gets its own socket and port: easier to inspect separately
with netcat or Wireshark, and it matches the fact that the three
streams run at completely different rates (commands on input
change, video at 20-30 fps, telemetry at a fixed 10 Hz).

    COMMAND_PORT    - Drone listens here for movement commands from the GCS.
    VIDEO_PORT      - GCS listens here for video frames from the Drone.
    TELEMETRY_PORT  - GCS listens here for telemetry packets from the Drone.

--------------------------------------------------------------------------
Auto-registration: how the Drone learns where to send video/telemetry
--------------------------------------------------------------------------
The Drone holds no hard-coded GCS address. When `recvfrom()` on the
command socket returns a packet, it also returns the sender's
address: `(data, (ip, port))`. The Drone keeps that `ip` - not the
`port`, which is an ephemeral send port, not the port the GCS
listens on - and pairs it with the fixed `VIDEO_PORT` /
`TELEMETRY_PORT` to stream back. The IP is learned at runtime; port
numbers are fixed and known to both sides ahead of time.

Until the first command arrives, the Drone has no destination and
sends nothing.
"""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Ports and network constants
# ---------------------------------------------------------------------------

COMMAND_PORT = 5000
VIDEO_PORT = 5001
TELEMETRY_PORT = 5002

# Most networks (Ethernet, home Wi-Fi) run a 1500-byte MTU. 1400
# leaves headroom for IP/UDP headers and general overhead, avoiding
# IP fragmentation - which would undo the point of chunking frames
# ourselves in the first place.
MAX_UDP_PACKET = 1400

TELEMETRY_HZ = 10.0
VIDEO_FPS = 20

# Streamed frame dimensions. Kept here (not only in drone/render.py)
# so the GCS can size its video QLabel without importing the drone
# package - Task 2 shouldn't depend on Task 1's internals.
VIDEO_FRAME_WIDTH = 640
VIDEO_FRAME_HEIGHT = 480

# If no fresh command arrives within this window, the Drone treats
# it as Vx=Vy=0 - a safety measure so a disconnected or crashed GCS
# doesn't leave the Drone moving forever on its last command.
COMMAND_STALE_TIMEOUT_SEC = 1.0


def clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a value to a range. Guards against invalid input - GUI bugs and
    malformed or malicious packets alike."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 1) Command packet — GCS -> Drone (JSON over UDP, COMMAND_PORT)
# ---------------------------------------------------------------------------

@dataclass
class Command:
    vx: float  # left/right, normalized [-1.0, 1.0]
    vy: float  # forward/backward, normalized [-1.0, 1.0]
    # One-shot request to (re)start the simulation - see gcs/controller.py.
    # True on at most one packet per button press; the Drone only acts on it
    # while not already running (drone/state.py, DroneState.apply_command).
    start: bool = False
    ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.vx = clamp(self.vx)
        self.vy = clamp(self.vy)


def encode_command(cmd: Command) -> bytes:
    payload = {"type": "cmd", "vx": cmd.vx, "vy": cmd.vy, "start": cmd.start, "ts": cmd.ts}
    return json.dumps(payload).encode("utf-8")


def decode_command(data: bytes) -> Optional[Command]:
    """Decode a command packet. Returns None instead of raising on a malformed
    packet - a single corrupt UDP packet shouldn't kill the reader loop, just
    get dropped while it waits for the next one."""
    try:
        payload = json.loads(data.decode("utf-8"))
        if payload.get("type") != "cmd":
            return None
        return Command(
            vx=float(payload["vx"]),
            vy=float(payload["vy"]),
            start=bool(payload.get("start", False)),
            ts=float(payload.get("ts", time.time())),
        )
    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# 2) Telemetry packet — Drone -> GCS (JSON over UDP, TELEMETRY_PORT)
# ---------------------------------------------------------------------------

@dataclass
class Telemetry:
    speed: float
    battery: float
    last_cmd: dict
    seq: int  # sequence number - lets the GCS detect/count lost packets
    started: bool = False  # is the Drone actually running, or idle waiting for START
    ts: float = field(default_factory=time.time)


def encode_telemetry(tm: Telemetry) -> bytes:
    payload = {
        "type": "telemetry",
        "speed": tm.speed,
        "battery": tm.battery,
        "last_cmd": tm.last_cmd,
        "seq": tm.seq,
        "started": tm.started,
        "ts": tm.ts,
    }
    return json.dumps(payload).encode("utf-8")


def decode_telemetry(data: bytes) -> Optional[Telemetry]:
    try:
        payload = json.loads(data.decode("utf-8"))
        if payload.get("type") != "telemetry":
            return None
        return Telemetry(
            speed=float(payload["speed"]),
            battery=float(payload["battery"]),
            last_cmd=payload.get("last_cmd", {}),
            seq=int(payload.get("seq", 0)),
            started=bool(payload.get("started", False)),
            ts=float(payload.get("ts", time.time())),
        )
    except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError):
        return None


# ---------------------------------------------------------------------------
# 3) Video packets — Drone -> GCS (binary over UDP, VIDEO_PORT)
#
# Every JPEG frame splits into several packets ("chunks") to stay
# under MAX_UDP_PACKET. Each chunk carries a fixed 8-byte header:
#
#   frame_id     (uint32) - unique, increasing frame id. Lets the
#                            GCS group chunks by frame and tell an
#                            old frame apart from a new one.
#   chunk_index  (uint16) - this chunk's position (0..total-1).
#   total_chunks (uint16) - chunk count for this frame.
#
# If a chunk is lost, the GCS never completes total_chunks for that
# frame_id, discards what it has after a short timeout, and waits
# for the next frame. That's the required handling of packet loss:
# give up on one frame instead of requesting a TCP-style retransmit.
# ---------------------------------------------------------------------------

VIDEO_HEADER_FORMAT = "!IHH"  # network byte order: uint32, uint16, uint16
VIDEO_HEADER_SIZE = struct.calcsize(VIDEO_HEADER_FORMAT)  # = 8 bytes
MAX_CHUNK_PAYLOAD = MAX_UDP_PACKET - VIDEO_HEADER_SIZE


def pack_video_chunk(frame_id: int, chunk_index: int, total_chunks: int, payload: bytes) -> bytes:
    header = struct.pack(VIDEO_HEADER_FORMAT, frame_id & 0xFFFFFFFF, chunk_index, total_chunks)
    return header + payload


def unpack_video_chunk(packet: bytes):
    """Split a video packet into its components. Returns None if the packet is
    too small to be valid."""
    if len(packet) < VIDEO_HEADER_SIZE:
        return None
    frame_id, chunk_index, total_chunks = struct.unpack(VIDEO_HEADER_FORMAT, packet[:VIDEO_HEADER_SIZE])
    payload = packet[VIDEO_HEADER_SIZE:]
    return frame_id, chunk_index, total_chunks, payload


def split_into_chunks(data: bytes, max_payload: int = MAX_CHUNK_PAYLOAD) -> list[bytes]:
    """Split a binary buffer (an encoded JPEG) into chunks of at most max_payload
    bytes each."""
    return [data[i : i + max_payload] for i in range(0, len(data), max_payload)]
