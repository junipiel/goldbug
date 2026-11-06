import json
import random
import re

from goldbug.checkers.base import Checker
from goldbug.preflight import run_preflight


def _boxed(r):
    return re.findall(r"\\boxed\{([^{}]*)\}", r)


def _strict(r, g):
    b = _boxed(r)
    return float(len(b) == 1 and b[0].strip() == g)


def _any(r, g):
    return float(g in [x.strip() for x in _boxed(r)])


def _samples(n_prompts=40, k=8, hedge_rate=0.03, solve=0.4, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n_prompts):
        gold = str(rng.randint(10, 99))
        for _ in range(k):
            if rng.random() < hedge_rate:
                resp = f"maybe \\boxed{{{rng.randint(10, 99)}}} or \\boxed{{{gold}}}"
            elif rng.random() < solve:
                resp = f"so \\boxed{{{gold}}}"
            else:
                resp = f"so \\boxed{{{int(gold) + 1}}}"
            out.append({"prompt_id": i, "gold": gold, "response": resp})
    return out


def test_preflight_flags_lenient_grader_and_clears_strict():
    samples = _samples()
    oracle = Checker("strict", "t", "boxed", _strict, "t")
    lenient = Checker("any", "t", "boxed", _any, "t")
    risky = {r.pattern: r for r in run_preflight(samples, lenient, oracle)}
    safe = {r.pattern: r for r in run_preflight(samples, oracle, oracle)}
    r = risky["multiple_final_answers"]
    assert r.p0 > 0.005 and r.delta > 0.05
    assert r.verdict in {"HIGH", "medium"}
    assert safe["multiple_final_answers"].delta <= 1e-9
    assert safe["multiple_final_answers"].verdict.startswith("safe")


def test_toylab_smoke():
    from goldbug.toylab import LabConfig, run
    res = run(LabConfig(sft_steps=10, rl_steps=2, prompts_per_step=2, G=4), log_every=1)
    assert len(res["history"]) == 3
    json.dumps(res)  # serialisable


def test_rgym_probe_smoke():
    import pytest
    pytest.importorskip("reasoning_gym")
    from goldbug.rgym import probe_task
    r = probe_task("basic_arithmetic", size=5)
    assert r.error == "" and r.honest == 1.0
    r = probe_task("maze", size=5)
    assert r.best_union == "digits_0_9" and r.gain > 0  # substring partial credit
