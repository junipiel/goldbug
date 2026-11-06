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
