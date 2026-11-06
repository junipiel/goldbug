# Census of math graders

Each grader got 1,100 to 2,100 probe answers over 78 gold answers of different types (integers, fractions, intervals, tuples and so on). Versions are in the table, and per-probe results are in `census/`. Rerun with `goldbug census`.

| checker (config) | version | correct answers accepted | best answer-independent gain vs constant guess | hedging premium | near-miss answers accepted (>5%) | correct rewrites rejected | slow calls |
|---|---|---|---|---|---|---|---|
| `lmeval_minerva` | lm_eval==0.4.13 | 100% | +0% | +0% | a different list with the same first element (3/3) | — | — |
| `math_verify` | math-verify==0.9.0 | 100% | +0% | +0% | — | — | 70 probes over budget |
| `opencompass_math` | opencompass==0.5.4 | 100% | +0% | +0% | "1, 2" for 12 (17/17) | terminating decimal for a fraction (5/5) | — |
| `openr1_accuracy` | math-verify==0.9.0 (open-r1 config) | 100% | +0% | +0% | — | — | 70 probes over budget |
| `qwen25_math` | Qwen2.5-Math@main (sha256-pinned files) | 100% | +0% | +3% | ×100 (39/39), ÷100 (39/39), off-by-one (4/35), 0.005% off (large ints) (4/7), "1, 2" for 12 (17/17), open↔closed interval (5/5) | — | — |
| `rllm_deepscaler` | rllm@main (sha256-pinned file) | 100% | +0% | +0% | — | — | — |
| `trl_accuracy` | math-verify==0.9.0 (TRL config) | 100% | +0% | +0% | — | — | 70 probes over budget |
| `verl_dapo` | verl==0.9.0 | 100% | +0% | +0% | "1, 2" for 12 (17/17) | \dfrac (13/13), terminating decimal for a fraction (5/5), 42.0 (35/35) | — |
| `verl_gsm8k` | verl==0.9.0 | 100% | +0% | +0% | — | x = 42 (35/35), 42.0 (35/35), $42 (30/30) | — |
| `verl_math` | verl==0.9.0 | 100% | +0% | +0% | — | terminating decimal for a fraction (4/5), 42.0 (35/35), 12,345 (7/7) | — |
| `verl_prime` | verl==0.9.0 | 100% | +0% | +3% | ×100 (39/39), ÷100 (39/39), off-by-one (4/35), 0.005% off (large ints) (4/7), π as 3.14 (2/3) | — | — |

Notes:

- The +3% hedging for `qwen25_math` and `verl_prime` isn't a real hedging premium. The wrong candidate (gold + 1) falls inside those graders' 1e-4 relative tolerance on large integers. Off-by-one passes for the same reason, on 4 of the 35 integer golds.
- `opencompass_math` is OpenCompass's legacy `MATHEvaluator(version="v2")`. The default MATH configs in 0.5.4 use `MATHVerifyEvaluator` instead.
- `lmeval_minerva` is the `exact_match` metric, which compares only the first element of a list. lm-eval's `math_verify` metric rejects these.
- The slow calls come from math-verify's internal 5 s timeout firing on `9^{9^{9^{9^{9}}}}`. It returns 0 in the end, but each such call blocks a reward worker for about 5 s.
- The Qwen2.5-Math and rLLM grader files are pinned by SHA-256 (fetched 2026-11-05).
