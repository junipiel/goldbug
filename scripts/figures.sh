#!/usr/bin/env bash
# Redraw the README plot and print the numbers for the second OLMo sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
python scripts/analyze.py results/olmo_run1.jsonl --q-from-grader --hero seed=2,p0=0.1,q=0.7,G=4 --out "$tmp" > /dev/null
cp "$tmp/hero.png" results/figures/hero.png
python scripts/confirm.py results/olmo_run2.jsonl --eta 1.778 --out "$tmp"
