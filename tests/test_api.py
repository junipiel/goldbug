import math
import random

import goldbug


def _samples(n_prompts=40, k=8, solve=0.3, shortcut_rate=0.03, seed=0):
    rng = random.Random(seed)
    out = []
    for pid in range(n_prompts):
        gold = str(pid * 7)
        for _ in range(k):
            u = rng.random()
            resp = "?" if u < shortcut_rate else (gold if u < shortcut_rate + solve else "wrong")
            out.append({"prompt_id": pid, "gold": gold, "response": resp, "prompt": f"what is {pid} * 7?"})
    return out


def test_buggy_reward_is_flagged_and_strict_one_is_not():
    samples = _samples()
    shortcuts = {"question_mark": (lambda r: r == "?", lambda s: "?")}
    buggy = lambda r, g: float(r == g or (r == "?" and int(g) % 2 == 0))  # accepts "?" on half the prompts
    strict = lambda r, g: float(r == g)
    [b] = goldbug.check(samples, reward=buggy, shortcuts=shortcuts, oracle=strict)
    [s] = goldbug.check(samples, reward=strict, shortcuts=shortcuts)
    assert b.seen and b.gain > 0 and math.isfinite(b.steps)
    assert s.gain < 0 and math.isinf(s.steps)
    assert "question_mark" in str(b)


def test_takeoff_formula():
    assert math.isinf(goldbug.takeoff(0.05, s=0.6, q=0.3))
    fast, slow = goldbug.takeoff(0.1, s=0.3, q=0.9), goldbug.takeoff(0.01, s=0.3, q=0.9)
    assert fast < slow < math.inf


def test_difficulty_aware_matches_shared_at_rho_one():
    from goldbug.theory import group_tables, grouped_takeoff, predict_takeoff_population
    rng = random.Random(1)
    s = [rng.choice([0, 0.125, 0.25, 0.5, 0.75, 1.0]) for _ in range(64)]
    for q_val, p0 in [(0.7, 0.02), (0.6, 0.1)]:
        q = [q_val] * len(s)
        shared = predict_takeoff_population(p0, s, q, 8, "std", 1.5, max_steps=1000)
        grouped = grouped_takeoff(group_tables(s, q, 8, "std"), p0, [1.5], 1.0, 8)[0]
        assert abs(grouped - shared) <= max(2, 0.05 * shared)


def test_selective_shortcut_is_caught():
    # 60% unsolvable prompts, 40% easy ones: on average the shortcut doesn't pay (0.35 < 0.36),
    # but on the hard prompts it does, and the difficulty-aware prediction sees that
    from goldbug.theory import predict_takeoff, predict_takeoff_population
    # the bug fires on 7 of every 20 prompts (35%), independent of difficulty
    s = [0.0] * 60 + [0.9] * 40
    q = [1.0 if i % 20 < 7 else 0.0 for i in range(100)]
    assert math.isinf(predict_takeoff_population(0.05, s, q, 8, "std", 1.5, max_steps=1000))
    assert math.isfinite(predict_takeoff(0.05, s, q, 8))
