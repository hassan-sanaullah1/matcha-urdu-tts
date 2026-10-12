#!/usr/bin/env bash
# STEP 3 -- train the shipped voice (the exact recipe stored in checkpoints/voice/args.json).
# Usage: bash src/3_train/train.sh [tag]        (default tag: voice_retrain; writes checkpoints/<tag>/)
# PY defaults to the kaani venv; set PY=/path/to/python to use another environment. GPU, ~41 min on a 6 GB card.
set -euo pipefail
cd "$(dirname "$0")/../.."                      # the kiosk-tts folder: training lists hold paths relative to it
PY=${PY:-$HOME/Desktop/kaani/envs/venv/bin/python}
TAG=${1:-voice_retrain}
export HF_HUB_OFFLINE=1

# --data / --support    her clips = speaker 0; Rasa studio Urdu = speaker 1 (support voice, 30% of rows)
# --target-weight 0.70  70% of each epoch's rows are hers; Rasa rows are repeated/trimmed to fill the other 30%
# --spk-lr-mult 10      speaker vectors learn 10x faster than the rest (at 1x the two voices barely separated)
# --seed-table v2       contrasting vowels (u/o/ɔ, e/ɛ) start from different Urdu letters
# --no-row-b            train only on row A of each clip (Urdu script + English phones); see train.tag_rows
# --steps 8000          this voice's run (an earlier voice used 6,000; see docs/training_report.md)
"$PY" src/3_train/train.py --tag "$TAG" \
    --data data/3_training_lists/voice --support data/3_training_lists/rasa_female/train.txt \
    --steps 8000 --ckpt-every 2000 --out-size 344 --bs 8 --accum 2 --lr 1e-4 \
    --target-weight 0.70 --spk-lr-mult 10 --seed-table v2 --no-row-b
