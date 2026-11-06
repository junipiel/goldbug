"""goldbug command line.

    goldbug census    [--checkers ...] [--out DIR]      probe answer checkers
    goldbug preflight --grader G --samples S.jsonl      predict which grader errors RL will exploit
    goldbug lab        [--quick]                         run the CPU toy-lab sweep
    goldbug rgym       [--out results/rgym.json]         census of reasoning-gym task scorers
    goldbug llmlab     --model M [--p0 --q --G ...]      planted-bug GRPO run on a real HF model (GPU)
    goldbug theory    --p0 0.01 --s 0.3 --q 0.6 --G 8   takeoff estimate for one setting
"""
from __future__ import annotations

import sys


def _theory(argv):
    import argparse
    from .theory import Setting, amplification, predict_takeoff_population, takeoff_closed_form
    ap = argparse.ArgumentParser(prog="goldbug theory",
                                 description="Takeoff estimate for one prompt type (base rate p0, solve rate s, acceptance q).")
    ap.add_argument("--p0", type=float, required=True, help="base rate of the shortcut before RL")
    ap.add_argument("--s", type=float, required=True, help="honest solve rate")
    ap.add_argument("--q", type=float, required=True, help="grader acceptance rate of the shortcut")
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--eta", type=float, default=1.41, help="effective step size (default: toy-lab calibration)")
    a = ap.parse_args(argv)
    if a.q <= a.s:
        print(f"q ({a.q}) <= s ({a.s}): the shortcut earns less than answering honestly -> no takeoff predicted.")
        return
    print(f"reward gap q - s = {a.q - a.s:+.2f}, base rate p0 = {a.p0:.2%}, G = {a.G}")
    for opt in ["adam", "sgd"]:
        for norm in ["std", "mean"]:
            t = predict_takeoff_population(a.p0, [a.s], [a.q], a.G, norm, a.eta, optimizer=opt, max_steps=100_000)
            label = "Adam (default)" if opt == "adam" else "plain SGD     "
            print(f"  {label}  {norm:4s}-normalised advantages: takeoff ~ {t:8.0f} steps")
    print(f"closed form (plain SGD, mean-norm, small p0, loss averaged per sample): "
          f"T* ~ G / (eta (G-1) (q-s) p0) = {a.G * takeoff_closed_form(a.p0, Setting(a.s, a.q, a.G, 'mean'), a.eta):.0f} steps")
    print(f"raw-gradient amplification of std vs mean normalisation: {amplification(a.p0, a.s, a.q, a.G):.2f}x "
          "(mostly absorbed by Adam)")
    print("Absolute step counts depend on your learning rate and model; compare settings, not raw numbers.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "census":
        from .census import main as m
        m(rest)
    elif cmd == "preflight":
        from .preflight import main as m
        m(rest)
    elif cmd == "lab":
        from .sweep import main as m
        m(rest)
    elif cmd == "llmlab":
        from .llmlab import main as m
        m(rest)
    elif cmd == "rgym":
        from .rgym import main as m
        m(rest)
    elif cmd == "theory":
        _theory(rest)
    else:
        print(__doc__)
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    main()
