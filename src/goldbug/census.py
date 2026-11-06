"""Run every strategy against every checker and measure *exploit gain*.

For a strategy s and checker c over golds G:

    pass(s, c)  = mean_g reward_c(render_c(s(g)), g)
    fair(s)     = mean_g [the answer a strict, single-answer grader would read == gold]
    gain(s, c)  = pass(s, c) - fair(s)

For answer-independent ("blind") strategies, a positive gain is reward a policy
can collect without solving anything. For hedges, gain is the reward bought by
emitting several candidates instead of committing to one.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkers import available_checkers, get_checker
from .golds import Gold, default_golds
from .strategies import ALL, Strategy


@dataclass
class Cell:
    checker: str
    checker_version: str
    strategy: str
    family: str
    should_pass: bool
    n: int
    passed: int
    fair: float
    timeouts: int = 0
    timeout_s: float = 0.0

    @property
    def rate(self) -> float:
        return self.passed / self.n if self.n else float("nan")

    @property
    def gain(self) -> float:
        return self.rate - self.fair


def _fair_pass(s: Strategy, payloads: list[str], gold: Gold) -> float:
    """What a strict grader that reads only the LAST answer would give."""
    if s.should_pass:
        return 1.0
    if s.family == "near_miss":
        return 0.0
    if s.family == "hedge":
        return 0.5  # per order; averaged over both orders this is the random-commit baseline
    last = payloads[-1].replace(" ", "")
    return float(last == gold.latex.replace(" ", ""))


class _InProcess:
    def __init__(self, checker):
        from .runner import ProbeResult
        self._c, self._R = checker, ProbeResult
        self.version, self.n_timeouts = checker.version, 0

    def score(self, resp, gold):
        return self._R(self._c.score(resp, gold), False)

    def close(self):
        pass


def run_census(checkers: list[str] | None = None, strategies: list[Strategy] | None = None,
               golds: list[Gold] | None = None, verbose: bool = True, timeout_s: float = 10.0,
               out_dir: str | Path | None = None, isolate: bool = True) -> list[Cell]:
    """Probe each checker in an isolated worker process with a per-probe time budget.

    ``checkers`` may contain registry names or, with ``isolate=False``, Checker objects
    (scored in-process without time limits -- used by tests and custom graders).

    If ``out_dir`` is given, each checker's cells are written to ``<out_dir>/<checker>.json``
    as soon as that checker finishes (so a crash later loses nothing)."""
    from .runner import GuardedChecker

    checkers = checkers or available_checkers()
    strategies = strategies or ALL
    golds = golds or default_golds()
    cells: list[Cell] = []
    for cname in checkers:
        t0 = time.time()
        try:
            if not isolate:
                meta_c = cname if not isinstance(cname, str) else get_checker(cname)
                gc = _InProcess(meta_c)
                cname = meta_c.name
            else:
                meta_c = get_checker(cname)  # cheap: for format/int_only metadata
                gc = GuardedChecker(cname, timeout_s=timeout_s)
        except Exception as e:  # missing optional dependency etc.
            if verbose:
                print(f"[skip] {cname}: {type(e).__name__}: {e}", flush=True)
            continue
        agg: dict[str, list] = {}
        for s in strategies:
            for g in golds:
                if meta_c.int_only and not g.is_int:
                    continue
                p = s.payloads(g)
                if p is None:
                    continue
                res = gc.score(meta_c.render(p, s.mode), g.latex)
                a = agg.setdefault(s.name, [s, 0, 0, 0.0, 0])
                a[1] += 1
                a[2] += int(res.score >= 0.5)
                a[3] += _fair_pass(s, p, g)
                a[4] += int(res.timed_out)
        gc.close()
        mine = [Cell(cname, gc.version, name, s.family, s.should_pass, n, k, fair / n, to, timeout_s if isolate else 0.0)
                for name, (s, n, k, fair, to) in agg.items()]
        cells += mine
        if out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            save(mine, Path(out_dir) / f"{cname}.json")
        if verbose:
            n = sum(c.n for c in mine)
            print(f"[done] {cname:18s} {n:5d} probes  {gc.n_timeouts:3d} timeouts  "
                  f"{time.time() - t0:6.1f}s  ({gc.version})", flush=True)
    return cells


def load_cells(paths) -> list[Cell]:
    out = []
    for p in paths:
        for row in json.loads(Path(p).read_text()):
            row = {k: v for k, v in row.items() if k not in {"rate", "gain"}}
            out.append(Cell(**row))
    return out


def save(cells: list[Cell], path: str | Path, merge: bool = True) -> None:
    """Write cells; with ``merge`` keep previously saved strategies not re-run now."""
    rows = [asdict(c) | {"rate": c.rate, "gain": c.gain} for c in cells]
    if merge and Path(path).exists():
        new_keys = {(r["checker"], r["strategy"]) for r in rows}
        old = [r for r in json.loads(Path(path).read_text()) if (r["checker"], r["strategy"]) not in new_keys]
        rows = old + rows
    Path(path).write_text(json.dumps(rows, indent=1))


def report_markdown(cells: list[Cell]) -> str:
    """The README-style report card: one row per checker config."""
    by: dict[str, dict[str, Cell]] = {}
    for c in cells:
        by.setdefault(c.checker, {})[c.strategy] = c
    rows = ["| checker (config) | version | correct answers accepted | best answer-independent gain vs constant guess "
            "| hedging premium | near-miss answers accepted (>5%) | correct rewrites rejected | slow calls |",
            "|---|---|---|---|---|---|---|---|"]
    for ck in sorted(by):
        d = by[ck]
        const = d["zero"].rate if "zero" in d else 0.0
        blind = max((c.rate - max(c.fair, const) for c in d.values() if c.family == "blind" and c.strategy != "zero"),
                    default=0.0)
        pairs: dict[str, list[float]] = {}
        for c in d.values():
            if c.family == "hedge":
                pairs.setdefault(c.strategy.split("[")[0], []).append(c.rate)
        prem = max((sum(v) / 2 - 0.5 for v in pairs.values() if len(v) == 2), default=0.0)
        nm = [f"{NICE.get(c.strategy, c.strategy)} ({c.passed}/{c.n})" for c in d.values()
              if c.family == "near_miss" and c.rate > 0.05]
        eq = [f"{NICE.get(c.strategy, c.strategy)} ({c.n - c.passed}/{c.n})" for c in d.values()
              if c.family == "equivalent" and c.rate < 0.95]
        slow = sum(c.timeouts for c in d.values())
        corr = d.get("honest_correct")
        version = next(iter(d.values())).checker_version
        rows.append(f"| `{ck}` | {version} | {corr.rate:.0%} | {max(0.0, blind):+.0%} | {max(0.0, prem):+.0%} | "
                    f"{', '.join(nm) or '—'} | {', '.join(eq) or '—'} | "
                    f"{(str(slow) + ' probes over budget') if slow else '—'} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Probe answer checkers with answer-independent and near-miss strategies.")
    ap.add_argument("--checkers", nargs="*", default=None)
    ap.add_argument("--out", default="results/census")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--families", nargs="*", default=None, help="only these strategy families (e.g. hedge)")
    ap.add_argument("--report-only", action="store_true", help="re-render REPORT.md from saved results, no probing")
    a = ap.parse_args(argv)
    if a.report_only:
        import glob
        cells = load_cells(sorted(glob.glob(str(Path(a.out) / "*.json"))))
        (Path(a.out) / "REPORT.md").write_text(report_markdown(cells) + "\n")
        print(report_markdown(cells))
        return
    strategies = [s for s in ALL if a.families is None or s.family in a.families]
    cells = run_census(a.checkers, strategies=strategies, out_dir=a.out, timeout_s=a.timeout)
    import glob
    all_cells = load_cells(sorted(glob.glob(str(Path(a.out) / "*.json"))))
    (Path(a.out) / "REPORT.md").write_text(report_markdown(all_cells) + "\n")
    print(report_markdown(all_cells))


if __name__ == "__main__":
    main()
