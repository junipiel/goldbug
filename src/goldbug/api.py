"""Check your own grader for shortcuts a model could learn, before you train.

    import goldbug

    shortcuts = {
        # name: (spot it in a response, write it for a given sample)
        "answers_zero": (lambda r: r.strip() == "0", lambda s: "0"),
        "copies_prompt": (lambda r: len(r) > 200, lambda s: s["prompt"]),
    }
    for sc in goldbug.check(samples, reward=my_reward, shortcuts=shortcuts):
        print(sc)
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .theory import predict_takeoff_population

ETA = 1.78  # fitted on OLMo-2 1B at lr 1e-6; step counts are rough, use them to compare


@dataclass
class Shortcut:
    name: str
    base_rate: float  # how often the model already does it before training (p0)
    seen: bool        # False if it never showed up in the samples (base_rate is then a floor)
    accepted: float   # share of prompts where the grader gives it full reward (q)
    gain: float       # average reward over answering honestly (q - s)
    pays_on: float    # share of prompts where it beats answering honestly
    steps: float      # predicted RL steps until it takes over (inf = not predicted to)

    def __str__(self) -> str:
        rate = f"{self.base_rate:.2%}" if self.seen else f"<{self.base_rate:.2%}"
        when = "never" if math.isinf(self.steps) else f"~{self.steps:.0f} steps"
        note = ""
        if math.isinf(self.steps) and self.pays_on > 0.3:
            note = "  (but it pays on {:.0%} of prompts; a model can learn to use it just there)".format(self.pays_on)
        return (f"{self.name}: base rate {rate}, accepted {self.accepted:.0%}, gain {self.gain:+.2f}, "
                f"takeover {when}{note}")


def check(samples: list[dict], reward: Callable[[str, str], float],
          shortcuts: dict[str, tuple[Callable[[str], bool], Callable[[dict], str]]],
          oracle: Callable[[str, str], float] | None = None, G: int = 8, norm: str = "std",
          eta: float = ETA) -> list[Shortcut]:
    """Predict which shortcuts RL will pick up with this reward.

    samples    dicts with "prompt_id", "gold" and "response", about 8 per prompt, sampled
               from the model you are about to train. Extra keys (like "prompt") are passed
               to your shortcut writers.
    reward     the grader you will train with: reward(response, gold) -> float in [0, 1].
    shortcuts  {name: (detect, write)}. detect(response) says whether a sample uses the
               shortcut; write(sample) produces the shortcut response for that prompt.
    oracle     what "actually correct" means, same signature as reward. Defaults to the
               reward itself, which is fine if it is mostly right on honest answers.
    G, norm    GRPO group size and advantage normalisation ("std" or "mean").
    """
    oracle = oracle or reward
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_prompt[str(s["prompt_id"])].append(s)
    n = sum(len(v) for v in by_prompt.values())
    out = []
    for name, (detect, write) in shortcuts.items():
        hits, s_x, q_x = 0, [], []
        for rows in by_prompt.values():
            gold = rows[0]["gold"]
            honest = [r for r in rows if not detect(r["response"])]
            hits += len(rows) - len(honest)
            s_x.append(sum(oracle(r["response"], gold) >= 0.5 for r in honest) / len(honest) if honest else 0.0)
            q_x.append(float(reward(write(rows[0]), gold) >= 0.5))
        p0 = max(hits / max(n, 1), 0.5 / max(n, 1))
        gain = sum(q - s for q, s in zip(q_x, s_x)) / max(len(q_x), 1)
        steps = predict_takeoff_population(p0, s_x, q_x, G, norm, eta) if gain > 0 else math.inf
        out.append(Shortcut(name, p0, hits > 0, sum(q_x) / max(len(q_x), 1), gain,
                            sum(q > s for q, s in zip(q_x, s_x)) / max(len(q_x), 1), steps))
    return sorted(out, key=lambda r: (r.steps, -r.gain))


def takeoff(p0: float, s: float, q: float, G: int = 8, norm: str = "std", eta: float = ETA) -> float:
    """Predicted RL steps until a shortcut takes over, from the three numbers alone."""
    return predict_takeoff_population(p0, [s], [q], G, norm, eta) if q > s else math.inf
