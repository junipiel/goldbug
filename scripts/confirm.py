"""Score a sweep with a fixed eta, i.e. with nothing fitted on that sweep.

    python scripts/confirm.py results/olmo_run2.jsonl --eta 1.778 --out results/figures

eta = 1.778 is what analyze.py fits on the first OLMo sweep (with --q-from-grader).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import analyze as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    rows = A.load(a.sweep)
    B = A.BUDGET
    obs = np.array([A.obs_takeoff(r) for r in rows])
    pred = np.array([A.predict(r, a.eta, "adam") for r in rows])
    pred[pred > 5 * B] = math.inf
    gap = np.array([A.gap_of(r) for r in rows])
    p0 = np.array([r["profile"]["p0_hat"] for r in rows])
    q = np.array([r["config"]["q"] for r in rows])
    took = obs <= B
    acc = lambda yes: float((yes == took).mean())
    fin = took & np.isfinite(pred)
    cf = np.where(gap > 0, 1.0 / (np.maximum(gap, 1e-9) * np.maximum(p0, 1e-4)), 1e9)
    stats = {
        "runs": len(rows), "taken_over": int(took.sum()),
        "accuracy": {"theory": acc(pred <= B), "gap>0 and shortcut seen": acc((gap > 0) & (p0 > 0)),
                     "gap>0": acc(gap > 0), "always no": acc(np.zeros_like(took))},
        "spearman": {"theory": A.spearman(np.minimum(pred[took], 1e9), obs[took]),
                     "closed form": A.spearman(cf[took], obs[took]),
                     "gap alone": A.spearman(-gap[took], obs[took]),
                     "base rate alone": A.spearman(-np.maximum(p0[took], 1e-4), obs[took])},
        "median predicted/observed": float(np.median(pred[fin] / obs[fin])),
        "median predicted/observed by q": {str(v): float(np.median(pred[fin & (q == v)] / obs[fin & (q == v)]))
                                           for v in sorted(set(q[fin].tolist()))},
    }
    print(json.dumps(stats, indent=1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot(rows, pred, obs, q, out / "confirm.png")


def plot(rows, pred, obs, q, path):
    import matplotlib.pyplot as plt
    B, cap = A.BUDGET, 2.2 * A.BUDGET
    fig, ax = plt.subplots(figsize=(4.8, 4.3), dpi=200)
    fig.patch.set_facecolor(A.SURFACE)
    A.style(ax)
    colors = {0.35: A.BLUE, 0.7: A.ORANGE, 0.1: A.AQUA}
    for v in sorted(set(q.tolist())):
        m = q == v
        ok = int(((pred[m] <= B) == (obs[m] <= B)).sum())
        ax.scatter(np.minimum(pred[m], cap), np.minimum(obs[m], cap), s=30, color=colors.get(v, A.MUTED),
                   edgecolor=A.SURFACE, linewidth=1.2, zorder=3, label=f"bug rate q = {v:g}  ({ok}/{m.sum()} right)")
    lim = [4, cap * 1.08]
    ax.plot(lim, lim, color=A.MUTED, linewidth=1, linestyle=":")
    ax.axhline(B, color=A.GRID, linewidth=1.2)
    ax.axvline(B, color=A.GRID, linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("predicted takeover step (before training)", color=A.MUTED, fontsize=8.5)
    ax.set_ylabel("observed takeover step", color=A.MUTED, fontsize=9)
    ax.set_title(f"Second OLMo sweep, nothing refitted\n(top/right edge = no takeover in {B} steps)",
                 fontsize=9, color=A.INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=A.INK, loc="upper left")
    fig.savefig(path, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
