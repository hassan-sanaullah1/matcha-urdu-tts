#!/usr/bin/env bash
# Start the kaani male+female kiosk TTS server (loads the model once, then serves on 127.0.0.1:8030).
#   ./run_server.sh                                       # defaults (female is the default voice)
#   PORT=8031 ./run_server.sh --vocoder-device cuda       # any app.py flag passes through
#   KIOSK_TTS_DEFAULT_VOICE=male ./run_server.sh          # make male the voice for bare /v1/tts calls
#   PYTHON=/path/to/venv/bin/python ./run_server.sh       # another Python 3.12 env (see requirements.txt)
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-/home/ubuntu/Desktop/kaani/envs/venv/bin/python}"
exec "$PYTHON" app.py --host "${HOST:-127.0.0.1}" --port "${PORT:-8030}" "$@"
