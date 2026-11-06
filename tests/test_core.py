import math
import re

import pytest

from goldbug.census import run_census
from goldbug.checkers.base import Checker
from goldbug.golds import default_golds
from goldbug.preflight import _boxed_contents
from goldbug.strategies import ALL, BLIND, HEDGE, NEAR_MISS
from goldbug.theory import (Setting, expected_drift, population_drift, takeoff_closed_form,
                              takeoff_steps)


# ------------------------------------------------------------------ theory
@pytest.mark.parametrize("p", [0.01, 0.1, 0.4])
@pytest.mark.parametrize("G", [2, 4, 8])
def test_mean_norm_drift_matches_closed_form(p, G):
    s, q = 0.3, 0.7
    exact = expected_drift(p, Setting(s, q, G, "mean"))
    assert exact == pytest.approx((G - 1) * p * (1 - p) * (q - s), rel=1e-6)


def test_no_drift_toward_shortcut_without_gap():
    for norm in ["mean", "std"]:
        assert expected_drift(0.1, Setting(0.6, 0.6, 8, norm)) == pytest.approx(0.0, abs=1e-9)
        assert expected_drift(0.1, Setting(0.6, 0.3, 8, norm)) < 0


def test_takeoff_monotone_in_base_rate_and_gap():
    st = Setting(0.2, 0.6, 8, "mean")
    assert takeoff_steps(0.005, st, 0.05) > takeoff_steps(0.02, st, 0.05) > takeoff_steps(0.1, st, 0.05)
    assert takeoff_steps(0.02, Setting(0.2, 0.4, 8, "mean"), 0.05) > takeoff_steps(0.02, Setting(0.2, 0.9, 8, "mean"), 0.05)
    assert math.isinf(takeoff_steps(0.02, Setting(0.5, 0.4, 8, "mean"), 0.05))
    # ODE and closed form agree to within the (1-p)^2 factor
    assert takeoff_steps(0.01, st, 0.05) == pytest.approx(takeoff_closed_form(0.01, st, 0.05), rel=0.2)


def test_adam_absorbs_global_advantage_scale():
    s_list, q_list = [0.7] * 10 + [0.1] * 10, [1.0] * 10 + [0.0] * 10
    raw = [population_drift(0.05, s_list, q_list, 8, n, optimizer="sgd") for n in ("std", "mean")]
    adam = [population_drift(0.05, s_list, q_list, 8, n, optimizer="adam") for n in ("std", "mean")]
    assert raw[0] / raw[1] > 1.5  # std normalisation amplifies raw gradients...
    assert 0.5 < adam[0] / adam[1] < 1.5  # ...but Adam mostly cancels a global rescaling


# ------------------------------------------------------------------ strategies
def test_blind_strategies_are_answer_independent():
    golds = default_golds()
    for s in BLIND:
        outs = {tuple(s.payloads(g)) for g in golds}
        assert len(outs) == 1, s.name


def test_hedges_come_in_both_orders():
    names = {s.name for s in HEDGE}
    for base in ["hedge_two_answers", "hedge_or", "hedge_comma"]:
        assert f"{base}[gold_first]" in names and f"{base}[gold_last]" in names
    g = default_golds()[5]
    gf = next(s for s in HEDGE if s.name == "hedge_two_answers[gold_first]").payloads(g)
    gl = next(s for s in HEDGE if s.name == "hedge_two_answers[gold_last]").payloads(g)
    assert gf == list(reversed(gl)) and gf[0] == g.latex


def test_near_misses_differ_from_gold():
    for s in NEAR_MISS:
        for g in default_golds():
            p = s.payloads(g)
            if p is not None:
                assert p[0].replace(" ", "") != g.latex.replace(" ", ""), (s.name, g.latex)


# ------------------------------------------------------------------ census logic on toy graders
def _strict(r, g):
    b = _boxed_contents(r)
    return float(len(b) == 1 and b[0] == g)


def _any_boxed(r, g):  # a deliberately lenient grader: accepts if ANY boxed answer matches
    return float(g in _boxed_contents(r))


def test_census_strict_grader_has_no_blind_gain():
    c = Checker("strict", "test", "boxed", _strict, "test")
    cells = run_census([c], isolate=False, verbose=False)
    blind = [x for x in cells if x.family == "blind"]
    assert all(x.gain <= 1e-9 for x in blind)
    assert next(x for x in cells if x.strategy == "honest_correct").rate == 1.0


def test_census_flags_lenient_grader():
    c = Checker("any_boxed", "test", "boxed", _any_boxed, "test")
    cells = run_census([c], isolate=False, verbose=False)
    hedge = [x for x in cells if x.strategy.startswith("hedge_two_answers[")]
    multi = next(x for x in cells if x.strategy == "multi_0_to_20")
    assert sum(h.rate for h in hedge) / 2 == pytest.approx(1.0)  # any-match grader: hedging always pays
    assert multi.gain > 0.1  # spraying small integers pays on small-integer golds


def test_all_strategy_names_unique():
    names = [s.name for s in ALL]
    assert len(names) == len(set(names))


def test_guarded_worker_does_not_break_grader_timeouts():
    """Regression: verl's prime_math uses its own multiprocessing timeout. Under a
    'spawn' worker it failed silently and returned 0 for correct answers."""
    import importlib.util
    if importlib.util.find_spec("verl") is None:  # importing verl itself pulls in ray; only check it exists
        pytest.skip("verl not installed")
    from goldbug.runner import GuardedChecker
    gc = GuardedChecker("verl_prime", timeout_s=30)
    try:
        assert gc.score("so \\boxed{0.75}", "\\frac{3}{4}").score == 1.0
    finally:
        gc.close()
