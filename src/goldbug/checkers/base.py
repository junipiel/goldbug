from __future__ import annotations

import importlib.metadata as md
from dataclasses import dataclass, field
from typing import Callable

FORMATS: dict[str, Callable[[str], str]] = {
    # how one answer is written in a response, per checker convention
    "boxed": lambda a: f"\\boxed{{{a}}}",
    "gsm8k": lambda a: f"#### {a}",
    "dapo": lambda a: f"Answer: {a}",
    "minerva": lambda a: f"Final Answer: The final answer is {a}. I hope it is correct.",
}


@dataclass
class Checker:
    name: str
    used_by: str  # where this checker is used as a reward / metric
    fmt_name: str
    score_fn: Callable[[str, str], float] = field(repr=False)
    version: str = "?"
    int_only: bool = False  # e.g. GSM8K-style checkers only make sense for integer golds

    def fmt(self, payload: str) -> str:
        return FORMATS[self.fmt_name](payload)

    def render(self, payloads: list[str], mode: str) -> str:
        """Render a full response: a short fake reasoning prefix + the answer(s)."""
        body = "Let me work through this step by step.\n\n"
        if mode == "multi":
            return body + "\n".join(self.fmt(p) for p in payloads)
        assert len(payloads) == 1, "single mode takes exactly one payload"
        return body + self.fmt(payloads[0])

    def score(self, response: str, gold: str) -> float:
        try:
            r = self.score_fn(response, gold)
        except Exception:  # a crashing checker is treated as reward 0 (what most trainers do)
            return 0.0
        if r is None:
            return 0.0
        if isinstance(r, tuple):  # some verl scorers return (correct, fmt, extracted)
            r = r[0]
        if isinstance(r, dict):
            r = r.get("score", r.get("acc", 0.0))
        r = float(r)
        return max(0.0, min(1.0, r))  # normalise e.g. dapo's {-1, 1}


def pkg_version(dist: str) -> str:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return "not-installed"


_REGISTRY: dict[str, Callable[[], Checker]] = {}


def register(name: str):
    def deco(factory: Callable[[], Checker]):
        _REGISTRY[name] = factory
        return factory
    return deco


def available_checkers() -> list[str]:
    from . import zoo  # noqa: F401  (populates the registry)
    return sorted(_REGISTRY)


def get_checker(name: str) -> Checker:
    from . import zoo  # noqa: F401
    return _REGISTRY[name]()
