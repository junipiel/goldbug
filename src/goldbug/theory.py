"""When does group-relative policy optimisation amplify a grader's false positives?

Two-behaviour model
-------------------
On a prompt, the policy either takes a *shortcut* (a response pattern the faulty
grader accepts with probability ``q``, regardless of correctness) with
probability ``p``, or makes an *honest attempt* that is correct with probability
``s`` (the model's solve rate on that prompt). Rewards are binary.

The policy's preference for the shortcut is a single logit ``theta``
(p = sigmoid(theta)). For REINFORCE with a group baseline over ``G`` samples,
the expected logit update per step is

    E[dtheta] = eta * E_group[ sum_i A_i * (1[y_i = shortcut] - p) ]

where A_i is the group-relative advantage: (r_i - mean) for Dr.GRPO-style
"mean" normalisation, or (r_i - mean) / (std + eps) for standard GRPO.

``expected_drift`` computes this expectation *exactly* by enumerating group
compositions (number of shortcut samples k, and how many of each kind got
reward). ``takeoff_steps`` integrates the resulting ODE from p0 until the
shortcut rate crosses 0.5.

Closed-form intuition (mean normalisation, small p, plain SGD):
    loss summed over each group:    T* ~= 1 / (eta * (G-1) * (q - s) * p0)
    loss averaged per sample:       T* ~= G / (eta * (G-1) * (q - s) * p0)
With the usual per-sample average (TRL, verl, the toy lab) group size barely
matters -- (G-1)/G -- and takeoff time is inversely proportional to the reward gap
Delta = q - s and to the base rate p0. Std normalisation divides by the group
standard deviation, which is small exactly when the model rarely succeeds, so it
amplifies raw gradients from rare false positives on hard prompts. The
``optimizer="adam"`` option divides the drift by the RMS advantage: a heuristic for
Adam's per-parameter rescaling, not an exact result.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache


def _binom(n: int, k: int, p: float) -> float:
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


@dataclass(frozen=True)
class Setting:
    s: float  # honest solve rate on the prompt
    q: float  # probability the faulty grader rewards the shortcut
    G: int = 8  # group size
    norm: str = "std"  # "std" (GRPO) or "mean" (Dr.GRPO-style, no std division)
    eps: float = 1e-4

    @property
    def delta(self) -> float:
        return self.q - self.s


@lru_cache(maxsize=400_000)
def _moments_cached(p: float, s: float, q: float, G: int, norm: str, eps: float) -> tuple[float, float]:
    """Exact expectations over group compositions of
       (1) sum_i A_i (1[y_i = shortcut] - p)   -- the logit drift, and
       (2) sum_i A_i^2                          -- the advantage energy (sets Adam's step scale)."""
    total, energy = 0.0, 0.0
    for k in range(G + 1):  # number of shortcut samples in the group
        pk = _binom(G, k, p)
        if pk < 1e-15:
            continue
        for a in range(k + 1):  # rewarded shortcuts
            pa = _binom(k, a, q)
            if pa < 1e-15:
                continue
            for b in range(G - k + 1):  # rewarded honest attempts
                pb = _binom(G - k, b, s)
                if pb < 1e-15:
                    continue
                w = pk * pa * pb
                n_pos = a + b
                mean = n_pos / G
                if norm == "std":
                    var = mean * (1 - mean)  # population std; TRL/verl use the Bessel-corrected sample std,
                    # a constant sqrt(G/(G-1)) factor that Adam absorbs
                    scale = 1.0 / (math.sqrt(var) + eps) if var > 0 else 0.0
                else:
                    scale = 1.0
                if scale == 0.0:
                    continue  # all-equal rewards: zero advantage
                adv_pos, adv_neg = (1 - mean) * scale, (0 - mean) * scale
                g = a * adv_pos * (1 - p) + (k - a) * adv_neg * (1 - p)
                g += b * adv_pos * (0 - p) + (G - k - b) * adv_neg * (0 - p)
                total += w * g
                energy += w * (n_pos * adv_pos ** 2 + (G - n_pos) * adv_neg ** 2)
    return total, energy


@lru_cache(maxsize=64)
def _structure(G: int, norm: str, eps: float):
    """Group-composition tensors over (k shortcut samples, a rewarded shortcuts,
    b rewarded honest samples) for a given G, independent of (p, s, q)."""
    import numpy as np
    k = np.arange(G + 1)[:, None, None]
    a = np.arange(G + 1)[None, :, None]
    b = np.arange(G + 1)[None, None, :]
    valid = (a <= k) & (b <= G - k)
    n_pos = a + b
    mean = n_pos / G
    var = mean * (1 - mean)
    if norm == "std":
        scale = np.where(var > 0, 1.0 / (np.sqrt(np.maximum(var, 0)) + eps), 0.0)
    else:
        scale = np.where(var > 0, 1.0, 0.0)
    adv_pos, adv_neg = (1 - mean) * scale, (0 - mean) * scale
    c_short = a * adv_pos + (k - a) * adv_neg      # sum of advantages of shortcut samples
    c_hon = b * adv_pos + (G - k - b) * adv_neg     # sum of advantages of honest samples
    energy = n_pos * adv_pos ** 2 + (G - n_pos) * adv_neg ** 2
    return c_short * valid, c_hon * valid, energy * valid


def moments_vec(p, s, q, G: int, norm: str, eps: float = 1e-4):
    """Vectorised exact moments; p, s, q are broadcastable 1-D arrays.
    Returns (drift, energy) -- the same quantities as _moments_cached."""
    import numpy as np
    from scipy.stats import binom
    cs, ch, en = _structure(G, norm, eps)
    p, s, q = (np.asarray(x, dtype=float).reshape(-1) for x in np.broadcast_arrays(p, s, q))
    kk = np.arange(G + 1)
    pk = binom.pmf(kk[None, :], G, p[:, None])                                     # (N, k)
    pa = binom.pmf(kk[None, None, :], kk[None, :, None], q[:, None, None])         # (N, k, a)
    pb = binom.pmf(kk[None, None, :], (G - kk)[None, :, None], s[:, None, None])   # (N, k, b)
    w = pk[:, :, None, None] * pa[:, :, :, None] * pb[:, :, None, :]              # (N, k, a, b)
    drift = (w * (cs[None] * (1 - p)[:, None, None, None] - ch[None] * p[:, None, None, None])).sum((1, 2, 3))
    energy = (w * en[None]).sum((1, 2, 3))
    return drift, energy


def _drift_cached(p: float, s: float, q: float, G: int, norm: str, eps: float) -> float:
    return _moments_cached(p, s, q, G, norm, eps)[0]


def expected_drift(p: float, st: Setting) -> float:
    """Exact expected logit update per step (per unit learning rate) for one prompt."""
    return _drift_cached(round(p, 6), st.s, st.q, st.G, st.norm, st.eps)


def takeoff_steps(p0: float, st: Setting, eta: float = 1.0, p_end: float = 0.5,
                  max_steps: int = 100_000) -> float:
    """Steps until the shortcut rate goes from p0 to p_end under the mean-field ODE
    (dtheta = eta * expected_drift). Returns inf if it never takes off."""
    if st.delta <= 0:
        return math.inf
    theta = math.log(p0 / (1 - p0))
    for t in range(1, max_steps + 1):
        p = 1 / (1 + math.exp(-theta))
        theta += eta * expected_drift(p, st)
        if 1 / (1 + math.exp(-theta)) >= p_end:
            return float(t)
    return math.inf


def takeoff_closed_form(p0: float, st: Setting, eta: float = 1.0) -> float:
    """Small-p0 approximation for mean normalisation: T* ~= 1 / (eta (G-1) Delta p0).

    Derivation: E[sum_i (r_i - rbar)(x_i - p)] = (G-1) p (1-p) (q - s), so
    dtheta = eta (G-1) p (1-p) Delta and dp = eta (G-1) p^2 (1-p)^2 Delta."""
    if st.delta <= 0:
        return math.inf
    return 1.0 / (eta * (st.G - 1) * st.delta * p0)


def amplification(p: float, s: float, q: float, G: int = 8) -> float:
    """How much faster std-normalised GRPO pushes towards the shortcut than mean-normalised."""
    d_std = expected_drift(p, Setting(s, q, G, "std"))
    d_mean = expected_drift(p, Setting(s, q, G, "mean"))
    return d_std / d_mean if d_mean != 0 else math.inf


def population_drift(p: float, s_list, q_list, G: int, norm: str, per_sample: bool = True,
                     optimizer: str = "adam") -> float:
    """Expected shortcut-logit update averaged over a population of prompts with
    per-prompt solve rates s_x and per-prompt grader acceptance q_x.

    ``per_sample=True`` matches losses normalised by the total number of samples
    (sum_i A_i logp_i / (P*G)), which is what most GRPO implementations do.

    ``optimizer="adam"`` divides by the RMS advantage: Adam rescales each
    parameter's step by the running RMS of its gradient, and the gradient scale is
    proportional to the advantage scale. So a *global* rescaling of advantages
    (e.g. dividing by the group std) is largely absorbed by Adam; only the
    *relative* weighting of samples and prompts survives. ``"sgd"`` keeps raw scale."""
    return float(_population_curve(tuple(s_list), tuple(q_list), G, norm, per_sample, optimizer,
                                   (float(p),))[0])


def _population_curve(s_list: tuple, q_list: tuple, G: int, norm: str, per_sample: bool, optimizer: str,
                      ps: tuple):
    import numpy as np
    from collections import Counter
    counts = Counter((round(s, 2), round(q, 2)) for s, q in zip(s_list, q_list))
    keys = list(counts)
    wts = np.array([counts[k] for k in keys], dtype=float)
    wts /= wts.sum()
    ss = np.array([k[0] for k in keys])
    qs = np.array([k[1] for k in keys])
    P = np.asarray(ps, dtype=float)
    d, e = moments_vec(np.repeat(P, len(keys)), np.tile(ss, len(P)), np.tile(qs, len(P)), G, norm)
    d = (d.reshape(len(P), len(keys)) * wts).sum(1)
    e = (e.reshape(len(P), len(keys)) * wts).sum(1)
    drift = d / G if per_sample else d
    if optimizer == "adam":
        rms = np.sqrt(np.maximum(e, 0) / G)
        return np.where(rms > 0, drift / np.where(rms > 0, rms, 1.0), 0.0)
    return drift


def drift_curve(s_list, q_list, G: int, norm: str, optimizer: str = "adam", n: int = 121):
    """Population drift tabulated on a logit grid from p=1e-4 to p=0.6."""
    import numpy as np
    thetas = np.linspace(math.log(1e-4 / (1 - 1e-4)), math.log(0.6 / 0.4), n)
    ps = 1 / (1 + np.exp(-thetas))
    return thetas, _population_curve(tuple(s_list), tuple(q_list), G, norm, True, optimizer, tuple(ps))


def integrate_takeoff(curve, p0: float, eta: float, p_end: float = 0.5, max_steps: int = 5000) -> float:
    """Euler-integrate dtheta = eta * drift(theta) from p0 until p >= p_end."""
    import numpy as np
    thetas, d = curve
    p0 = min(max(p0, 1e-4), 0.49)
    theta, end = math.log(p0 / (1 - p0)), math.log(p_end / (1 - p_end))
    for t in range(1, max_steps + 1):
        dd = float(np.interp(theta, thetas, d))
        if dd <= 0:
            return math.inf
        theta += eta * dd
        if theta >= end:
            return float(t)
    return math.inf


def predict_takeoff_population(p0: float, s_list, q_list, G: int, norm: str, eta: float,
                               p_end: float = 0.5, max_steps: int = 5000, optimizer: str = "adam") -> float:
    """Steps until the shortcut rate reaches p_end, from pre-RL measurements only."""
    return integrate_takeoff(drift_curve(s_list, q_list, G, norm, optimizer), p0, eta, p_end, max_steps)



# ------------------------------------------------------------- shortcut rate that depends on difficulty
# The model doesn't take a shortcut equally often everywhere: it picks it up faster on problems
# it can't solve, where the shortcut pays most. Prompts are grouped by their honest solve rate s;
# each group gets its own shortcut logit. Per step, a group's logit moves by
#     eta * (rho * average drift over all groups + (1 - rho) * the group's own drift),
# so rho = 1 is the version above (one shared shortcut rate) and smaller rho lets the groups
# drift apart. Takeover = the shortcut rate averaged over prompts reaches p_end.

def group_tables(s_list, q_list, G: int, norm: str, n: int = 161):
    """Drift tables per difficulty group, on a logit grid."""
    import numpy as np
    from collections import Counter
    pairs = Counter((round(float(s), 3), round(float(q), 3)) for s, q in zip(s_list, q_list))
    svals = sorted({s for s, _ in pairs})
    w = np.array([sum(c for (s, _), c in pairs.items() if s == sv) for sv in svals], dtype=float)
    thetas = np.linspace(math.log(1e-5 / (1 - 1e-5)), math.log(0.9999 / 0.0001), n)
    ps = 1 / (1 + np.exp(-thetas))
    d = np.zeros((len(svals), n))
    e = np.zeros((len(svals), n))
    for i, sv in enumerate(svals):  # average over the q values seen at this difficulty
        members = [(q, c) for (s, q), c in pairs.items() if s == sv]
        for q, c in members:
            dq, eq = moments_vec(ps, np.full(n, sv), np.full(n, q), G, norm)
            d[i] += c / w[i] * dq
            e[i] += c / w[i] * eq
    return thetas, w / w.sum(), d, e


def grouped_takeoff(tables, p0: float, etas, rho: float, G: int, p_end: float = 0.5, max_steps: int = 1000):
    """Steps until the average shortcut rate reaches p_end (inf = never). Vectorised over etas."""
    import numpy as np
    thetas, w, d, e = tables
    etas = np.atleast_1d(np.asarray(etas, dtype=float))
    lo, hi, n = thetas[0], thetas[-1], len(thetas)
    h = (hi - lo) / (n - 1)
    p0 = min(max(p0, 1e-4), 0.49)
    th = np.full((len(etas), len(w)), math.log(p0 / (1 - p0)))
    out = np.full(len(etas), np.inf)
    alive = np.ones(len(etas), dtype=bool)
    rows = np.arange(len(w))[None, :]
    for t in range(1, max_steps + 1):
        x = (np.clip(th, lo, hi) - lo) / h
        i = np.minimum(x.astype(int), n - 2)
        f = x - i
        dg = d[rows, i] * (1 - f) + d[rows, i + 1] * f
        eg = e[rows, i] * (1 - f) + e[rows, i + 1] * f
        rms = np.sqrt(np.maximum((eg * w).sum(1, keepdims=True) / G, 1e-12))
        D = dg / G / rms
        step = rho * (D * w).sum(1, keepdims=True) + (1 - rho) * D
        stuck = (step <= 0).all(1)
        th = th + etas[:, None] * step
        hit = alive & ((w / (1 + np.exp(-th))).sum(1) >= p_end)
        out[hit] = t
        alive &= ~hit & ~stuck
        if not alive.any():
            break
    return out


def predict_takeoff(p0: float, s_list, q_list, G: int, norm: str = "std", eta: float = 1.58, rho: float = 0.65,
                    max_steps: int = 1000) -> float:
    """Steps until the shortcut takes over, from pre-RL measurements (difficulty-aware)."""
    return float(grouped_takeoff(group_tables(s_list, q_list, G, norm), p0, [eta], rho, G, max_steps=max_steps)[0])
