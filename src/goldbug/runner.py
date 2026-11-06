"""Robust execution of checkers: one persistent worker process per checker, a
wall-clock budget per probe, and automatic restart after a hang.

A checker that hangs on some input is itself a finding: in RL training a hung
reward call stalls (or silently zeroes) a whole rollout batch.
"""
from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass


def _serve(name: str, conn) -> None:
    import sys
    if sys.platform.startswith("linux"):
        # Graders such as verl's prime_math run their own multiprocessing timeouts with
        # decorated functions that cannot be pickled under "spawn". Under spawn they
        # silently fail and return 0, which would make a correct checker look broken.
        mp.set_start_method("fork", force=True)
    from .checkers import get_checker
    c = get_checker(name)
    conn.send(("ready", c.version))
    while True:
        msg = conn.recv()
        if msg is None:
            return
        resp, gold = msg
        conn.send(("ok", c.score(resp, gold)))


@dataclass
class ProbeResult:
    score: float
    timed_out: bool


class GuardedChecker:
    def __init__(self, name: str, timeout_s: float = 10.0, start_timeout_s: float = 120.0):
        self.name, self.timeout_s, self.start_timeout_s = name, timeout_s, start_timeout_s
        self.ctx = mp.get_context("spawn")
        self.version = "?"
        self.n_timeouts = 0
        self._start()

    def _start(self) -> None:
        self.parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_serve, args=(self.name, child), daemon=False)  # graders may spawn their own timeout processes
        self.proc.start()
        if not self.parent.poll(self.start_timeout_s):
            self.proc.kill()
            raise RuntimeError(f"checker {self.name} did not start")
        tag, payload = self.parent.recv()
        self.version = payload

    def score(self, resp: str, gold: str) -> ProbeResult:
        self.parent.send((resp, gold))
        if self.parent.poll(self.timeout_s):
            _, s = self.parent.recv()
            return ProbeResult(float(s), False)
        self.proc.kill()
        self.proc.join()
        self.n_timeouts += 1
        self._start()
        return ProbeResult(0.0, True)

    def close(self) -> None:
        try:
            self.parent.send(None)
        except Exception:
            pass
        self.proc.join(timeout=5)
        if self.proc.is_alive():
            self.proc.kill()
