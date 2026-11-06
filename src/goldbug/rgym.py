"""Census of reasoning-gym task scorers (https://github.com/open-thought/reasoning-gym).

reasoning-gym ships ~100 procedural task generators, each with a ``score_answer``
used directly as an RLVR reward. For every task we compare:

  honest       the gold answer (sanity check: should score 1.0)
  constant     the best single constant answer ("0", "yes", "True", "A", ...),
               i.e. what a policy gets from exploiting class imbalance
  union        answer-independent strategies that list several candidates at once
               ("yes no", "A B C D", "0 1 2 ... 9", all constants together, or
               simply copying the question)

A task is flagged when some union strategy beats the best constant: that reward
comes from the scorer, not from the data.

    python -m goldbug.rgym --out results/rgym.json
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import asdict, dataclass

CONSTANTS = ["0", "1", "2", "3", "-1", "10", "yes", "no", "Yes", "No", "True", "False", "true", "false",
             "A", "B", "C", "D", "None", "[]", "x"]
UNIONS = {
    "yes_no": "yes no",
    "Yes_No": "Yes No",
    "True_False": "True False",
    "A_to_D": "A B C D",
    "digits_0_9": "0 1 2 3 4 5 6 7 8 9",
    "all_constants": " ".join(CONSTANTS),
}


@dataclass
class TaskResult:
    task: str
    n: int
    honest: float
    best_constant: str
    best_constant_score: float
    best_union: str
    best_union_score: float
    gain: float  # best union - best constant
    partial_credit_seen: bool  # some probe scored strictly between 0 and 1
    error: str = ""


class _Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise _Timeout()


def probe_task(name: str, size: int = 30, seed: int = 0, budget_s: int = 60) -> TaskResult:
    import reasoning_gym
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget_s)
    try:
        ds = reasoning_gym.create_dataset(name, size=size, seed=seed)
        entries = [ds[i] for i in range(size)]

        def mean_score(fn) -> tuple[float, bool]:
            vals = []
            for e in entries:
                try:
                    v = float(ds.score_answer(fn(e), e))
                except Exception:
                    v = 0.0
                vals.append(v)
            return sum(vals) / len(vals), any(0.0 < v < 1.0 for v in vals)

        honest, _ = mean_score(lambda e: e["answer"])
        partial = False
        best_c, best_cs = "", -1.0
        for c in CONSTANTS:
            sc, p = mean_score(lambda e, c=c: c)
            partial |= p
            if sc > best_cs:
                best_c, best_cs = c, sc
        best_u, best_us = "", -1.0
        unions = dict(UNIONS)
        for k, u in unions.items():
            sc, p = mean_score(lambda e, u=u: u)
            partial |= p
            if sc > best_us:
                best_u, best_us = k, sc
        sc, p = mean_score(lambda e: e["question"])  # copy the question verbatim
        partial |= p
        if sc > best_us:
            best_u, best_us = "copy_question", sc
        return TaskResult(name, size, honest, best_c, best_cs, best_u, best_us, best_us - best_cs, partial)
    except _Timeout:
        return TaskResult(name, size, float("nan"), "", 0, "", 0, 0, False, "timeout")
    except Exception as e:  # task needs extra deps / config
        return TaskResult(name, size, float("nan"), "", 0, "", 0, 0, False, f"{type(e).__name__}: {e}"[:200])
    finally:
        signal.alarm(0)


def main(argv=None):
    import importlib.metadata as md
    import reasoning_gym
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/rgym.json")
    ap.add_argument("--size", type=int, default=30)
    a = ap.parse_args(argv)
    names = sorted(reasoning_gym.factory.DATASETS.keys())
    res = []
    t0 = time.time()
    for n in names:
        r = probe_task(n, a.size)
        res.append(r)
        flag = "FLAG" if r.gain > 0.01 else ""
        print(f"{n:32s} honest={r.honest:.2f} const={r.best_constant_score:.2f}({r.best_constant}) "
              f"union={r.best_union_score:.2f}({r.best_union}) {flag} {r.error}", flush=True)
    out = {"reasoning_gym_version": md.version("reasoning_gym"), "size": a.size, "seconds": time.time() - t0,
           "tasks": [asdict(r) for r in res]}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
