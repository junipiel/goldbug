#!/usr/bin/env bash
# Redraw the README plot and recompute the numbers in the README.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
python scripts/analyze.py results/olmo_run1.jsonl --q-from-grader --hero seed=2,p0=0.1,q=0.7,G=4 --out "$tmp" > /dev/null
cp "$tmp/hero.png" results/figures/hero.png
cd scripts && python confirm.py --fit ../results/olmo_run1.jsonl --test ../results/olmo_run2.jsonl
python confirm.py --fit ../results/olmo_run1.jsonl ../results/olmo_run2.jsonl --test ../results/qwen_run.jsonl \
    --plot-selective ../results/figures/selective.png
