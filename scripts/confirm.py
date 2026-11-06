"""Fit the two constants (eta, rho) on some sweeps, then predict another sweep with them frozen.

    python scripts/confirm.py --fit results/olmo_run1.jsonl --test results/olmo_run2.jsonl
    python scripts/confirm.py --fit results/olmo_run1.jsonl results/olmo_run2.jsonl --test results/qwen_run.jsonl
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

import analyze as A
from goldbug.theory import group_tables, grouped_takeoff

ETAS = np.exp(np.linspace(np.log(0.01), np.log(100), 81))
RHOS = [1.0, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25]


def tables(r):
    s, q = A.theory_inputs(r)
    return group_tables(s, q, r["config"]["G"], r["config"]["norm"])


def log_error(pred, obs, cap=2 * A.BUDGET):
    """Squared log error, with 'never' and anything past 2x the budget counted as 2x the budget."""
    return ((np.log(np.minimum(pred, cap)) - np.log(np.minimum(obs, cap))[:, None]) ** 2).sum(0)


def fit(rows):
    obs = np.array([A.obs_takeoff(r) for r in rows])
    tabs = [tables(r) for r in rows]
    best = None
    for rho in RHOS:
        pred = np.array([grouped_takeoff(t, r["profile"]["p0_hat"], ETAS, rho, r["config"]["G"],
                                         max_steps=2 * A.BUDGET + 1) for t, r in zip(tabs, rows)])
        err = log_error(pred, obs)
        j = int(np.argmin(err))
        if best is None or err[j] < best[0]:
            best = (float(err[j]), rho, float(ETAS[j]))
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--plot-selective", default=None, help="write the shortcut-use-by-difficulty chart here")
    a = ap.parse_args()
    A.Q_FROM_GRADER = True  # runs recorded before q_pop existed
    rho, eta = fit([r for f in a.fit for r in A.load(f)])
    rows = A.load(a.test)
    B = A.BUDGET
    obs = np.array([A.obs_takeoff(r) for r in rows])
    pred = np.array([grouped_takeoff(tables(r), r["profile"]["p0_hat"], [eta], rho, r["config"]["G"],
                                     max_steps=5 * B)[0] for r in rows])
    gap = np.array([A.gap_of(r) for r in rows])
    p0 = np.array([r["profile"]["p0_hat"] for r in rows])
    q = np.array([r["config"]["q"] for r in rows])
    took = obs <= B
    fin = took & np.isfinite(pred)
    if a.plot_selective:
        plot_selective(rows, 0.65, a.plot_selective)
    print(json.dumps({
        "fitted on": a.fit, "rho": rho, "eta": round(eta, 3), "runs": len(rows), "taken over": int(took.sum()),
        "called right": int(((pred <= B) == took).sum()),
        "called right by 'q > s and shortcut seen'": int((((gap > 0) & (p0 > 0)) == took).sum()),
        "spearman, timing": round(A.spearman(np.minimum(pred[took], 1e9), obs[took]), 3),
        "spearman, q - s alone": round(A.spearman(-gap[took], obs[took]), 3),
        "median predicted/observed": round(float(np.median(pred[fin] / obs[fin])), 2),
        "median predicted/observed by q": {str(v): round(float(np.median(pred[fin & (q == v)] / obs[fin & (q == v)])), 2)
                                           for v in sorted(set(q[fin].tolist()))},
        "shortcut use after training vs solve rate before (Spearman per run, runs taken over)": {
            str(v): [round(A.spearman(r["profile"]["s_x"], r["profile_after"]["short_x"]), 2) for r in rows
                     if r.get("profile_after") and r["config"]["q"] == v and A.obs_takeoff(r) <= B
                     and np.ptp(r["profile_after"]["short_x"]) > 0]
            for v in sorted(set(q.tolist())) if any(r.get("profile_after") for r in rows)},
        "misses": [{**{k: r["config"][k] for k in ("p0", "q", "G", "seed")}, "predicted": None if math.isinf(t) else round(t),
                    "observed": None if math.isinf(o) else o}
                   for r, t, o in zip(rows, pred, obs) if (t <= B) != (o <= B)],
    }, indent=1))


def plot_selective(rows, q_value, path):
    """Shortcut use after training, by how often the model solved each problem before training."""
    import matplotlib.pyplot as plt
    bins = [(-0.01, 0.0, "never"), (0.0, 0.4, "sometimes"), (0.4, 0.75, "often"), (0.75, 1.0, "almost\nalways")]
    runs = [r for r in rows if r.get("profile_after") and r["config"]["q"] == q_value and A.obs_takeoff(r) <= A.BUDGET]
    vals = np.array([[np.mean(np.array(r["profile_after"]["short_x"])[(s > lo) & (s <= hi)])
                      for lo, hi, _ in bins] for r in runs for s in [np.array(r["profile"]["s_x"])]])
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    fig.patch.set_facecolor(A.SURFACE)
    A.style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(bins))
    from scipy.stats import t as student_t
    mean = vals.mean(0)
    half = student_t.ppf(0.975, len(vals) - 1) * vals.std(0, ddof=1) / np.sqrt(len(vals))  # 95% interval across runs
    ax.bar(x, mean, width=0.56, color=A.BLUE, zorder=2)
    ax.errorbar(x, mean, yerr=half, fmt="none", ecolor=A.INK, elinewidth=1.2, capsize=4, capthick=1.2, zorder=3)
    ax.set_xticks(x, [b[2] for b in bins])
    ax.set_ylim(0, 1)
    ax.set_xlabel("how often the model solved the problem before training", color=A.MUTED, fontsize=8.5)
    ax.set_ylabel("answers that are the shortcut,\nafter training", color=A.MUTED, fontsize=8.5)
    ax.set_title(f"The model cheats where it can't solve the problem\n"
                 f"Qwen2.5 1.5B, mean of {len(runs)} runs with 95% interval. The grader accepts\n"
                 f"the shortcut on {q_value:.0%} of problems, however hard they are.", fontsize=8.5, color=A.INK, loc="left")
    fig.savefig(path, facecolor=A.SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
