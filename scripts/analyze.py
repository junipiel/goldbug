"""Does the pre-RL prediction match what happened in a sweep?

eta (the one fitted constant) is fitted on the seed-0 runs; every other seed is
scored as held out. Works on toy-lab and LLM-lab output.

    python scripts/analyze.py results/olmo_run1.jsonl --q-from-grader --out results/figures
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from goldbug.theory import drift_curve, integrate_takeoff

BUDGET = 200
BLUE, ORANGE, AQUA, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def load(path):
    return [json.loads(l) for l in open(path)]


def obs_takeoff(r) -> float:
    return float(r["takeoff_step"]) if r["takeoff_step"] is not None else math.inf


def is_calib(r) -> bool:
    return r["config"]["seed"] == 0


OPT = "adam"  # theory variant used for figures; "sgd" is reported as an ablation
Q_FROM_GRADER = False  # see --q-from-grader


def q_pop(r) -> float | None:
    """The grader's acceptance rate for the shortcut over all problems, if known."""
    pr, c = r["profile"], r["config"]
    if "q_pop" in pr:
        return float(pr["q_pop"])
    if Q_FROM_GRADER and "model" in c:
        from goldbug.llmlab import population_bug_rate
        key = (c["q"], c.get("digits", 2))
        if key not in _QPOP:
            _QPOP[key] = population_bug_rate(*key)
        return _QPOP[key]
    return None


_QPOP: dict = {}


def theory_inputs(r) -> tuple[list, list]:
    """(s_x, q_x) for the theory. With a known population rate, post-stratify: the bug is a hash of
    the prompt, independent of difficulty, so every profiled prompt enters once with the bug firing
    (weight q_pop) and once without (weight 1 - q_pop)."""
    from fractions import Fraction
    s, q = r["profile"]["s_x"], r["profile"]["q_x"]
    qp = q_pop(r)
    if qp is None:
        return s, q
    f = Fraction(qp).limit_denominator(40)
    k1, k0 = f.numerator, f.denominator - f.numerator
    return [x for x in s for _ in range(k1)] + [x for x in s for _ in range(k0)], [1.0] * (len(s) * k1) + [0.0] * (len(s) * k0)


def gap_of(r) -> float:
    s, q = theory_inputs(r)
    return float(np.mean(q) - np.mean(s))


_CURVES: dict = {}


def predict(r, eta: float, optimizer: str | None = None) -> float:
    c, pr = r["config"], r["profile"]
    opt = optimizer or OPT
    key = (id(r), opt)
    if key not in _CURVES:
        s_x, q_x = theory_inputs(r)
        _CURVES[key] = drift_curve(s_x, q_x, c["G"], c["norm"], opt)
    return integrate_takeoff(_CURVES[key], pr["p0_hat"], eta, max_steps=5 * BUDGET)


def fit_eta(rows, optimizer: str | None = None) -> float:
    cal = [r for r in rows if is_calib(r)]
    grid = np.exp(np.linspace(np.log(0.01), np.log(100), 81))
    best, best_eta = -1e9, 1.0
    for eta in grid:
        score = 0.0
        for r in cal:
            tp, to = predict(r, eta, optimizer), obs_takeoff(r)
            tp_c, to_c = min(tp, 2 * BUDGET), min(to, 2 * BUDGET)  # censor at 2x budget
            score -= (math.log(tp_c) - math.log(to_c)) ** 2
        if score > best:
            best, best_eta = score, eta
    return float(best_eta)


def spearman(a, b) -> float:
    from scipy.stats import spearmanr  # average ranks: observed steps are multiples of 10 and tie often
    return float(spearmanr(a, b).correlation)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _pick(rows, spec: str | None):
    if spec:
        want = {k: float(v) for k, v in (kv.split("=") for kv in spec.split(","))}
        hits = [r for r in rows if all(float(r["config"][k]) == v for k, v in want.items())]
        if hits:
            return hits[0]
        raise SystemExit(f"--hero {spec}: no matching run")
    # toy-lab default: the most "typical" exploited run
    pick = [r for r in rows if r["config"]["norm"] == "std" and r["config"]["G"] == 8
            and r["config"]["p0"] == 0.03 and r["config"]["q"] == 0.85 and r["config"]["seed"] == 0]
    return pick[0] if pick else rows[0]


def _spread(ys, gap=0.07):
    """Nudge end-of-line label positions apart so they never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def hero(rows, out: Path, spec: str | None = None):
    import matplotlib.pyplot as plt
    r = _pick(rows, spec)
    h = r["history"]
    x = [e["step"] for e in h]
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    series = [("proxy_reward", "reward from the faulty grader", BLUE, "-"),
              ("oracle_acc", "actual accuracy (oracle)", ORANGE, "-"),
              ("shortcut_rate", "answers that are the shortcut", AQUA, "--")]
    ends = _spread([h[-1][k] for k, *_ in series])
    for (key, label, col, ls), ye in zip(series, ends):
        y = [e[key] for e in h]
        ax.plot(x, y, color=col, linewidth=2, linestyle=ls, label=label)
        ax.annotate(label, (x[-1], ye), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("RL step", color=MUTED, fontsize=9)
    ax.set_ylabel("share of answers", color=MUTED, fontsize=9)
    c = r["config"]
    who = c["model"].split("/")[-1] if "model" in c else "tiny transformer"
    ax.set_title(f"{who}, GRPO. The grader accepts '?' on {c['q']:.0%} of prompts; "
                 f"before training the model used it {r['profile']['p0_hat']:.1%} of the time",
                 fontsize=9, color=INK, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0),
              borderaxespad=0.2, handlelength=1.8)
    fig.subplots_adjust(right=0.7)
    fig.savefig(out / "hero.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return r


def phase_and_scatter(rows, eta: float, out: Path) -> dict:
    import matplotlib.pyplot as plt
    held = [r for r in rows if not is_calib(r)]
    tp = np.array([predict(r, eta) for r in held])
    to = np.array([obs_takeoff(r) for r in held])
    pred_cls, obs_cls = tp <= BUDGET, to <= BUDGET
    acc = float((pred_cls == obs_cls).mean())
    both = obs_cls  # every held-out run that was taken over; "never" predictions rank last
    rho = spearman(np.minimum(tp[both], 1e9), to[both]) if both.sum() > 3 else float("nan")

    # scatter: predicted vs observed takeoff (held out)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    cap = 2.2 * BUDGET
    for norm, col, mk in [("std", BLUE, "o"), ("mean", ORANGE, "s")]:
        m = np.array([r["config"]["norm"] == norm for r in held])
        if not m.any():
            continue
        ax.scatter(np.minimum(tp[m], cap), np.minimum(to[m], cap), s=26, color=col, marker=mk,
                   edgecolor=SURFACE, linewidth=1.2, label=f"{'GRPO (std-normalised)' if norm == 'std' else 'mean-normalised'}")
    lim = [5, cap * 1.05]
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle=":")
    ax.axhline(BUDGET, color=GRID, linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("predicted takeoff step (from pre-RL samples)", color=MUTED, fontsize=9)
    ax.set_ylabel("observed takeoff step", color=MUTED, fontsize=9)
    seeds = ", ".join(str(s) for s in sorted({r["config"]["seed"] for r in held}))
    note = ""
    ax.set_title(f"Held-out runs (seed {seeds}): Spearman {rho:.2f} on runs taken over\n"
                 f"(points at the top/right edge = no takeover within {BUDGET} steps{note})",
                 fontsize=9, color=INK, loc="left")
    if len({r["config"]["norm"] for r in held}) > 1:  # one series needs no legend
        ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.savefig(out / "scatter.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # phase diagram: one point per (base rate, reward gap) cell = 6 runs (3 group sizes x 2 normalisations);
    # colour = share of those runs taken over within the budget (sequential blue), label = k/n
    import matplotlib.colors as mcolors
    seq = mcolors.LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    cells = {}
    for r in rows:
        pr = r["profile"]
        key = (round(pr["p0_hat"], 5), round(gap_of(r), 4))
        cells.setdefault(key, []).append(obs_takeoff(r) <= BUDGET)
    fig, ax = plt.subplots(figsize=(5.6, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    nz = [p for p, _ in cells if p > 0]
    floor = min(nz) / 3 if nz else 1e-3
    for (p0, gap), outs in cells.items():
        k, n = sum(outs), len(outs)
        p0 = p0 if p0 > 0 else floor  # never sampled before RL: drawn at a floor, tick labelled "0"
        ax.scatter(p0, gap, s=70, color=seq(k / n), edgecolor=MUTED if k == 0 else SURFACE, linewidth=0.8, zorder=3)
    ax.set_xscale("log")
    if any(p == 0 for p, _ in cells):
        ax.axvline(floor * 1.7, color=GRID, linewidth=1.2)
        ticks = [v for v in (1e-3, 1e-2, 1e-1) if v > floor * 1.7 and v <= max(nz) * 1.5]
        ax.set_xticks([floor] + ticks)
        ax.set_xticklabels(["0"] + [f"$10^{{{int(round(np.log10(v)))}}}$" for v in ticks])
        ax.minorticks_off()
    ax.axhline(0, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("shortcut base rate before RL, p\u2080 (log)", color=MUTED, fontsize=9)
    ax.set_ylabel("reward gap \u0394 = P(shortcut rewarded) \u2212 P(honest correct)", color=MUTED, fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=seq, norm=mcolors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"share of runs taken over within {BUDGET} steps", color=MUTED, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    ax.set_title("Takeover needs a positive gap and a shortcut the model already tries",
                 fontsize=9.5, color=INK, loc="left")
    fig.savefig(out / "phase.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return {"eta": eta, "heldout_n": int(len(held)), "takeoff_classification_acc": acc,
            "spearman_takeoff_steps": rho, "spearman_n": int(both.sum())}


def norm_effect(rows) -> dict:
    """Paired comparison: same (p0, q, G, seed), std vs mean normalisation."""
    key = lambda r: (r["config"]["p0"], r["config"]["q"], r["config"]["G"], r["config"]["seed"])
    std = {key(r): obs_takeoff(r) for r in rows if r["config"]["norm"] == "std"}
    mean = {key(r): obs_takeoff(r) for r in rows if r["config"]["norm"] == "mean"}
    pairs = [(std[k], mean[k]) for k in std if k in mean]
    neither = sum(np.isinf(a) and np.isinf(b) for a, b in pairs)
    faster = sum(a < b for a, b in pairs)
    slower = sum(a > b for a, b in pairs)
    return {"pairs": len(pairs), "neither_taken_over": int(neither), "std_faster": faster, "std_slower": slower,
            "same_eval_step": len(pairs) - neither - faster - slower}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--hero", default=None, help="run for the hero figure, e.g. seed=2,p0=0.1,q=0.7,G=4")
    ap.add_argument("--q-from-grader", action="store_true",
                    help="for runs recorded before q_pop existed: recompute the bug rate over all problems")
    a = ap.parse_args()
    global Q_FROM_GRADER
    Q_FROM_GRADER = a.q_from_grader
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load(a.sweep)
    global OPT
    OPT = "sgd"
    eta_sgd = fit_eta(rows)
    OPT = "adam"
    eta = fit_eta(rows)
    hero(rows, out, a.hero)
    stats = phase_and_scatter(rows, eta, out)
    srcs = {"profile" if "q_pop" not in r["profile"] and not (Q_FROM_GRADER and "model" in r["config"])
            else "grader" for r in rows}
    stats["q_source"] = {"profile": "profile sample", "grader": "grader, all problems"}[srcs.pop()] if len(srcs) == 1 else "mixed"
    stats["eta_sgd"] = eta_sgd
    stats["norm_effect"] = norm_effect(rows)
    stats["comparison"] = baselines(rows, eta, eta_sgd)
    stats["n_runs"] = len(rows)
    stats["takeoff_rate"] = float(np.mean([obs_takeoff(r) <= BUDGET for r in rows]))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=float))
    print(json.dumps(stats, indent=1, default=float))




def baselines(rows, eta_adam: float, eta_sgd: float) -> dict:
    """Every predictor scored on the same held-out runs."""
    held = [r for r in rows if not is_calib(r)]
    cal = [r for r in rows if is_calib(r)]
    obs = np.array([obs_takeoff(r) for r in held])
    took = obs <= BUDGET
    gap = np.array([gap_of(r) for r in held])
    p0 = np.array([max(r["profile"]["p0_hat"], 1e-4) for r in held])
    G = np.array([r["config"]["G"] for r in held])
    cg = np.array([gap_of(r) for r in cal])
    ct = np.array([obs_takeoff(r) <= BUDGET for r in cal])
    ths = np.unique(np.concatenate([cg, [-1, 1]]))
    th = float(max(ths, key=lambda t: ((cg > t) == ct).mean()))
    tp_adam = np.array([predict(r, eta_adam, "adam") for r in held])
    tp_sgd = np.array([predict(r, eta_sgd, "sgd") for r in held])
    m = took
    # closed form for losses averaged per sample (as in the lab, TRL and verl): T* ~ G / ((G-1) gap p0)
    cf = np.where(gap > 0, G / ((G - 1) * np.maximum(gap, 1e-9) * p0), 1e9)
    return {
        "n_heldout": int(len(held)), "n_heldout_taken_over": int(m.sum()), "gap_threshold": th,
        "accuracy": {"always_no": float((~took).mean()), "sign_of_gap": float(((gap > 0) == took).mean()),
                     "gap_threshold": float(((gap > th) == took).mean()),
                     "gap_pos_and_shortcut_seen": float((((gap > 0) & (p0 > 1e-4)) == took).mean()),
                     "theory_adam": float(((tp_adam <= BUDGET) == took).mean()),
                     "theory_sgd": float(((tp_sgd <= BUDGET) == took).mean())},
        "spearman": {"gap_only": spearman(-gap[m], obs[m]), "closed_form": spearman(cf[m], obs[m]),
                     "base_rate_only": spearman(-p0[m], obs[m]),
                     "theory_adam": spearman(np.minimum(tp_adam[m], 1e9), obs[m]),
                     "theory_sgd": spearman(np.minimum(tp_sgd[m], 1e9), obs[m])},
    }


if __name__ == "__main__":
    main()
