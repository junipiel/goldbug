"""Registry of answer checkers ("verifiers") used for RLVR rewards and evals.

Each checker is wrapped as ``score(response, gold) -> float`` in [0, 1], plus a
``fmt(payload)`` that renders one answer in the format the checker reads (so a
strategy is tested *as the checker would see it*).

Adapters call the upstream scoring code, pinned by package version or by file
hash. The TRL and open-r1 rewards are thin wrappers around math-verify; for those
the adapter copies their settings.
"""
from .base import Checker, available_checkers, get_checker  # noqa: F401
