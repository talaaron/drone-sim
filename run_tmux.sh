#!/usr/bin/env bash
#
# run_tmux.sh - launch the Drone Simulator (Server) and the GCS (Client) as
# two separate tmux windows in one session: drone is window 0, GCS is
# window 1. Switching windows (not panes) replaces the whole terminal view
# with the other one - nothing is shown side by side. The GCS is a PyQt6
# GUI app - tmux just launches and supervises the process; its window pops
# up separately on your desktop as usual.
#
# Usage:
#   ./run_tmux.sh                             both nodes on localhost (default)
#   ./run_tmux.sh --drone-host 192.168.1.23   GCS points at a drone on another host
#   ./run_tmux.sh --drop-probability 0.05     deliberately drop ~5% of video chunks too
#     (passed straight through to drone.main - see its --help; can combine with --drone-host)
#
# Telemetry loss isn't a flag: it's simulated on every run by default (see
# drone/network.py's DEFAULT_TELEMETRY_DROP_PROBABILITY) - that's what the GCS's
# "packets lost" counter is watching for, not something you have to opt into.
#
# Requires: tmux, and the project venv already created - see README.md,
# "Build & run", steps 1-2 (git clone + cd, then python3 -m venv .venv +
# pip install -r requirements.txt).

set -euo pipefail

SESSION="Drone-sim"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
DRONE_HOST="127.0.0.1"
DRONE_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --drone-host)
            DRONE_HOST="$2"
            shift 2
            ;;
        --drop-probability)
            # forwarded to drone.main as-is (see drone/main.py --help)
            DRONE_EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if ! command -v tmux >/dev/null 2>&1; then
    echo "error: tmux is not installed. On Debian/Ubuntu: sudo apt-get install tmux" >&2
    exit 1
fi

if [[ ! -f "$VENV_ACTIVATE" ]]; then
    echo "error: venv not found at $VENV_ACTIVATE" >&2
    echo "Create it first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

# Re-running this script while a previous demo session is still up should
# just replace it cleanly, not error out or pile up duplicate sessions.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Window 0: the Drone Simulator (Server). Starts first.
# printf %q quotes each extra arg individually before it's dropped into
# the shell command string tmux runs - safe even if a value ever had
# spaces or shell metacharacters in it, which floats like "0.5" never
# do, but the array could in principle hold anything.
drone_cmd="python3 -m drone.main"
for arg in "${DRONE_EXTRA_ARGS[@]}"; do
    drone_cmd+=" $(printf '%q' "$arg")"
done

# `exec bash` at the end keeps the window open after the process exits
# (Ctrl+C, crash, etc.) so you can still read its last output instead of
# the window just closing.
tmux new-session -d -s "$SESSION" -n drone -c "$PROJECT_ROOT" \
    "source '$VENV_ACTIVATE' && $drone_cmd; exec bash"

# Window 1: the GCS (Client), as its own separate window (not a split pane) -
# switching to it replaces the whole terminal view. A short delay gives the
# drone a moment to bind its sockets first, though the GCS would recover
# fine either way - it just keeps showing 'no signal' until the drone is up.
tmux new-window -t "$SESSION" -n gcs -c "$PROJECT_ROOT" \
    "sleep 1 && source '$VENV_ACTIVATE' && python3 -m gcs.main --drone-host '$DRONE_HOST'; exec bash"

tmux select-window -t "$SESSION":drone

# Left detached on purpose - the normal tmux way: attach whenever you want
# with the command below, and detach again anytime with the tmux prefix
# (Ctrl+b, then d) without killing the session or the two processes.
echo "Session '$SESSION' is up (drone + gcs, running detached, one window each)."
echo "Attach with:      tmux attach -t $SESSION"
echo "Switch window:     Ctrl+b n  (next)   Ctrl+b p  (previous)   Ctrl+b 0 / Ctrl+b 1  (by number)"
echo "Detach again:      Ctrl+b d"
echo "Kill everything:   tmux kill-session -t $SESSION"
