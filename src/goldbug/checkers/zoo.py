"""Adapters for widely used math answer checkers.

Each adapter calls the *real* upstream code. ``used_by`` says where the checker
is used as an RL reward or eval metric, so a finding can be traced to users.
Version strings are recorded in every census result.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .._load import load_file, make_stub, namespace_package, site_package_dir, stub_modules
from .base import Checker, pkg_version, register


# ---------------------------------------------------------------- math-verify (HF)
def _mv_wrap(gold: str) -> str:
    return f"${gold}$"


@register("math_verify")
def _math_verify() -> Checker:
    from math_verify import parse, verify

    def score(resp: str, gold: str) -> float:
        return float(verify(parse(_mv_wrap(gold)), parse(resp)))

    return Checker("math_verify", "HF Math-Verify defaults (lighteval, lm-eval math_verify metric, Open LLM Leaderboard)",
                   "boxed", score, f"math-verify=={pkg_version('math-verify')}")


@register("trl_accuracy")
def _trl_accuracy() -> Checker:
    # Mirrors trl.rewards.accuracy_reward (TRL main, Nov 2026): gold parsed with defaults,
    # answer parsed with LatexExtractionConfig(units=True, boxed first, no anchor-free extraction).
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    cfg = [LatexExtractionConfig(normalization_config=NormalizationConfig(units=True),
                                 boxed_match_priority=0, try_extract_without_anchor=False)]

    def score(resp: str, gold: str) -> float:
        g = parse(f"\\boxed{{{gold}}}")
        return float(verify(g, parse(resp, extraction_config=cfg, extraction_mode="first_match")))

    return Checker("trl_accuracy", "TRL accuracy_reward (GRPOTrainer examples)", "boxed", score,
                   f"math-verify=={pkg_version('math-verify')} (TRL config)")


@register("openr1_accuracy")
def _openr1_accuracy() -> Checker:
    # Mirrors open-r1 src/open_r1/rewards.py::accuracy_reward.
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify

    cfg = [LatexExtractionConfig(
        normalization_config=NormalizationConfig(nits=False, malformed_operators=False, basic_latex=True,
                                                 equations=True, boxed="all", units=True),
        boxed_match_priority=0, try_extract_without_anchor=False)]

    def score(resp: str, gold: str) -> float:
        g = parse(f"\\boxed{{{gold}}}", extraction_mode="first_match")
        return float(verify(g, parse(resp, extraction_config=cfg, extraction_mode="first_match")))

    return Checker("openr1_accuracy", "open-r1 accuracy_reward (GRPO recipes)", "boxed", score,
                   f"math-verify=={pkg_version('math-verify')} (open-r1 config)")


# ---------------------------------------------------------------- verl built-in scorers
def _verl_module(sub: str):
    root = site_package_dir("verl")
    namespace_package("verl", root)
    namespace_package("verl.utils", root / "utils")
    namespace_package("verl.utils.reward_score", root / "utils" / "reward_score")
    import importlib
    return importlib.import_module(f"verl.utils.reward_score.{sub}")


@register("verl_gsm8k")
def _verl_gsm8k() -> Checker:
    m = _verl_module("gsm8k")
    return Checker("verl_gsm8k", "verl default reward for openai/gsm8k", "gsm8k",
                   lambda r, g: m.compute_score(r, g), f"verl=={pkg_version('verl')}", int_only=True)


@register("verl_math")
def _verl_math() -> Checker:
    m = _verl_module("math_reward")
    return Checker("verl_math", "verl default reward for MATH / MATH-500", "boxed",
                   lambda r, g: m.compute_score(r, g), f"verl=={pkg_version('verl')}")


@register("verl_dapo")
def _verl_dapo() -> Checker:
    m = _verl_module("math_dapo")
    return Checker("verl_dapo", "verl reward for DAPO-Math / AIME data sources", "dapo",
                   lambda r, g: m.compute_score(r, g), f"verl=={pkg_version('verl')}")


@register("verl_prime")
def _verl_prime() -> Checker:
    m = _verl_module("prime_math")
    return Checker("verl_prime", "verl reward for numina/PRIME-style math sources", "boxed",
                   lambda r, g: m.compute_score(r, g), f"verl=={pkg_version('verl')}")


# ---------------------------------------------------------------- lm-eval-harness minerva_math
@register("lmeval_minerva")
def _lmeval_minerva() -> Checker:
    root = site_package_dir("lm_eval")
    m = load_file("_fg_lmeval_minerva", root / "tasks" / "minerva_math" / "utils.py")

    def score(resp: str, gold: str) -> float:
        # gold is normalised exactly as lm-eval's process_docs does it
        g = m.normalize_final_answer(m.remove_boxed(m.last_boxed_only_string(f"\\boxed{{{gold}}}")))
        ans = m.normalize_final_answer(m.get_unnormalized_answer(resp))
        return float(m.is_equiv(ans, g))

    return Checker("lmeval_minerva", "lm-eval-harness minerva_math exact_match", "minerva", score,
                   f"lm_eval=={pkg_version('lm_eval')}")


# ---------------------------------------------------------------- OpenCompass MATHEvaluator
@register("opencompass_math")
def _opencompass_math() -> Checker:
    root = site_package_dir("opencompass")

    class _Reg:
        def register_module(self, *a, **k):
            return lambda x: x

    class _BaseEvaluator:
        def __init__(self, pred_postprocessor=None):
            self.pred_postprocessor = pred_postprocessor

    stubs = {
        "datasets": make_stub("datasets", Dataset=object, DatasetDict=object),
        "opencompass": make_stub("opencompass"),
        "opencompass.openicl": make_stub("opencompass.openicl"),
        "opencompass.openicl.icl_evaluator": make_stub("opencompass.openicl.icl_evaluator", BaseEvaluator=_BaseEvaluator),
        "opencompass.registry": make_stub("opencompass.registry", ICL_EVALUATORS=_Reg(), LOAD_DATASET=_Reg(),
                                          TEXT_POSTPROCESSORS=_Reg()),
        "opencompass.utils": make_stub("opencompass.utils", get_data_path=lambda *a, **k: ""),
        "opencompass.datasets": make_stub("opencompass.datasets"),
        "opencompass.datasets.base": make_stub("opencompass.datasets.base", BaseDataset=object),
    }
    src = (root / "datasets" / "math.py").read_text().replace("from .base import", "from opencompass.datasets.base import")
    tmp = Path(tempfile.mkdtemp(prefix="goldbug_")) / "_oc_math.py"  # private dir, not a guessable /tmp path
    tmp.write_text(src)
    with stub_modules(stubs):
        m = load_file("_fg_oc_math", tmp)
    ev = m.MATHEvaluator(version="v2")

    def score(resp: str, gold: str) -> float:
        return float(ev.is_equiv(m.math_postprocess_v2(resp), gold))

    return Checker("opencompass_math", "OpenCompass MATHEvaluator(v2) + math_postprocess_v2", "boxed", score,
                   f"opencompass=={pkg_version('opencompass')}")


# ---------------------------------------------------------------- rLLM / DeepScaleR
@register("rllm_deepscaler")
def _rllm() -> Checker:
    from ..upstream import fetch
    m = load_file("_fg_rllm_math_utils", fetch("rllm/math_utils.py"))
    return Checker("rllm_deepscaler", "rLLM / DeepScaleR math reward (grade_answer_verl)", "boxed",
                   lambda r, g: float(m.grade_answer_verl(r, g)), "rllm@main (sha256-pinned file)")


# ---------------------------------------------------------------- Qwen2.5-Math evaluation grader
_QWEN_WORKER = r"""
import sys, json, types
sys.path.insert(0, sys.argv[1])
sys.modules['utils'] = types.ModuleType('utils')
import grader, parser as qparser
for line in sys.stdin:
    d = json.loads(line)
    try:
        pred = qparser.extract_answer(d['r'], 'math')
        pred = qparser.strip_string(pred)
        ok = grader.math_equal(pred, qparser.strip_string(d['g']), timeout=True)
    except Exception:
        ok = False
    print(json.dumps(float(bool(ok))), flush=True)
"""


@register("qwen25_math")
def _qwen() -> Checker:
    """Runs in a separate interpreter: Qwen's latex2sympy2 needs antlr4 4.7.2,
    which conflicts with math-verify. Set GOLDBUG_QWEN_PYTHON to that interpreter."""
    from ..upstream import fetch
    g = fetch("qwen25math/grader.py")
    fetch("qwen25math/parser.py")
    py = os.environ.get("GOLDBUG_QWEN_PYTHON", sys.executable)
    proc: list[subprocess.Popen] = []

    def _worker() -> subprocess.Popen:
        if not proc or proc[0].poll() is not None:
            proc[:] = [subprocess.Popen([py, "-c", _QWEN_WORKER, str(g.parent)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)]
        return proc[0]

    def score_one(r: str, gd: str) -> float:
        w = _worker()
        assert w.stdin and w.stdout
        w.stdin.write(json.dumps({"r": r, "g": gd}) + "\n")
        w.stdin.flush()
        line = w.stdout.readline()
        if not line:
            raise RuntimeError("qwen worker died")
        return float(line)

    return Checker("qwen25_math", "Qwen2.5-Math evaluation grader (math_equal), reused by many RLVR repos", "boxed",
                   score_one, "Qwen2.5-Math@main (sha256-pinned files)")
