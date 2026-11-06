"""Dose-response sweep for the toy lab: vary the shortcut base rate (p0), how often
the faulty grader fires (q), group size (G) and advantage normalisation, and
record when (if ever) the shortcut takes over.

    python -m goldbug.sweep --out results/sweep.jsonl --workers 2
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

from .toylab import LabConfig, TinyGPT, pretrain, profile, run

P0S = [0.01, 0.03, 0.1]
QS = [0.5, 0.7, 0.85, 1.0]
GS = [4, 8, 16]
NORMS = ["std", "mean"]
SEEDS = [0, 1]
SAMPLES_PER_STEP = 128  # fixed budget: prompts_per_step = SAMPLES_PER_STEP // G


def _group(args) -> list[dict]:
    p0, seed, qs, gs, norms, rl_steps, out = args
    base = LabConfig(p0=p0, seed=seed, rl_steps=rl_steps)
    t0 = time.time()
    state = pretrain(base)
    model = TinyGPT(base)
    model.load_state_dict(state)
    model.eval()
    rows = []
    for q in qs:
        prof = profile(model, replace(base, q=q))
        for G, norm in itertools.product(gs, norms):
            cfg = replace(base, q=q, G=G, norm=norm, prompts_per_step=SAMPLES_PER_STEP // G)
            res = run(cfg, state=state)
            res["profile"] = {"p0_hat": prof["p0_hat"], "s_x": prof["s_x"], "q_x": prof["q_x"]}
            rows.append(res)
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")
    print(f"[group] p0={p0} seed={seed} done in {time.time() - t0:.0f}s", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep.jsonl")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rl-steps", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke tests")
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    p0s, seeds, qs, gs, norms = (P0S, SEEDS, QS, GS, NORMS) if not a.quick else ([0.03], [0], [0.85], [8], ["std"])
    jobs = [(p0, s, qs, gs, norms, a.rl_steps, a.out) for p0 in p0s for s in seeds]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(_group, jobs):
            pass


if __name__ == "__main__":
    main()
