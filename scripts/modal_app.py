"""Run the LLM experiment on Modal GPUs.

    pip install modal -e .
    modal setup
    modal run scripts/modal_app.py --mode control                  # quick check, ~3 min
    modal run --detach scripts/modal_app.py --mode sweep --seeds 0,1,2 --qs 0.1,0.35,0.7
    modal run scripts/modal_app.py --mode status
    modal run scripts/modal_app.py --mode collect                  # -> results/llm_sweep.jsonl

Each run writes its own file to a Modal volume, so a sweep can be restarted and
skips finished runs. OLMo-2-1B on an A100-40GB takes about 3.5 min per run
(~$0.12). Pick the bug rates (--qs) on both sides of the honest solve rate that
the control run prints.
"""
from __future__ import annotations

import json
import time

import modal

APP_NAME = "goldbug"
VOL = modal.Volume.from_name("goldbug", create_if_missing=True)
DEFAULT_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch==2.15.0", "transformers==5.22.1", "accelerate", "scipy", "numpy", "bitsandbytes")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("goldbug")
)
app = modal.App(APP_NAME, image=image)


def _key(cfg: dict) -> str:
    return f"{cfg['model'].replace('/', '__')}_p0{cfg['p0']}_q{cfg['q']}_G{cfg['G']}_s{cfg['seed']}_lr{cfg['lr_rl']}"


def _run(cfg: dict, tag: str) -> dict:
    import os
    from goldbug.llmlab import LLMLabConfig, run
    t0 = time.time()
    res = run(LLMLabConfig(**cfg))
    res["modal"] = {"tag": tag, "seconds": time.time() - t0}
    os.makedirs(f"/vol/runs/{tag}", exist_ok=True)
    with open(f"/vol/runs/{tag}/{_key(cfg)}.json", "w") as f:  # one file per run: safe with parallel containers
        json.dump(res, f)
    VOL.commit()
    prof = res["profile"]
    s = sum(prof["s_x"]) / len(prof["s_x"])
    return {"key": _key(cfg), "takeoff_step": res["takeoff_step"], "p0_hat": prof["p0_hat"],
            "honest_solve_rate": s, "minutes": round((time.time() - t0) / 60, 1)}


# One function per GPU type (the GPU is fixed when the function is defined).
_FN_KW = dict(volumes={"/vol": VOL}, timeout=3 * 3600, max_containers=4)


@app.function(gpu="A100-40GB", **_FN_KW)
def run_a100_40(cfg: dict, tag: str) -> dict:
    return _run(cfg, tag)


@app.function(gpu="A100-80GB", **_FN_KW)
def run_a100_80(cfg: dict, tag: str) -> dict:
    return _run(cfg, tag)


@app.function(gpu="L40S", **_FN_KW)
def run_l40s(cfg: dict, tag: str) -> dict:
    return _run(cfg, tag)


RUNNERS = {"A100-40GB": run_a100_40, "A100-80GB": run_a100_80, "L40S": run_l40s}


@app.function(volumes={"/vol": VOL}, timeout=600)
def list_done(tag: str) -> list[str]:
    import os
    VOL.reload()
    d = f"/vol/runs/{tag}"
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".json")) if os.path.isdir(d) else []


@app.function(volumes={"/vol": VOL}, timeout=600)
def fetch(tag: str) -> list[str]:
    import os
    VOL.reload()
    d = f"/vol/runs/{tag}"
    if not os.path.isdir(d):
        return []
    return [open(os.path.join(d, f)).read() for f in sorted(os.listdir(d)) if f.endswith(".json")]


def _sweep_configs(model: str, lr: float, optim: str, seeds: list[int], rl_steps: int,
                   qs=(0.5, 0.85), p0s=(0.01, 0.03, 0.1), Gs=(4, 8)) -> list[dict]:
    # Only overrides; the remote side fills the remaining LLMLabConfig defaults (no torch needed locally).
    base = {"model": model, "lr_rl": lr, "optim": optim, "rl_steps": rl_steps, "norm": "std"}
    out = []
    for seed in seeds:
        for p0 in p0s:
            for q in qs:
                for G in Gs:
                    out.append({**base, "p0": p0, "q": q, "G": G, "prompts_per_step": 64 // G, "seed": seed})
    return out


@app.local_entrypoint()
def main(mode: str = "control", model: str = DEFAULT_MODEL, gpu: str = "A100-40GB", lr: float = 1e-6,
         optim: str = "adamw", seeds: str = "0,1", rl_steps: int = 200, tag: str = "",
         op: str = "mul", digits: int = 2, max_runs: int = 0, qs: str = "0.5,0.85",
         p0s: str = "0.01,0.03,0.1", gs: str = "4,8"):
    """mode: control | sweep | status | collect"""
    from pathlib import Path
    runner = RUNNERS[gpu]
    if mode == "control":
        cfg = {"model": model, "lr_rl": lr, "optim": optim, "op": op, "digits": digits, "seed": 0,
               "p0": 0.1, "q": 1.0, "G": 8, "prompts_per_step": 8, "rl_steps": 30, "log_every": 5}
        print("Positive control: strong bug (q=1.0), high base rate (p0=0.1), 30 RL steps.")
        r = runner.remote(cfg, tag or "control")
        print(json.dumps(r, indent=1))
        print("\nCheck: honest_solve_rate ~0.2-0.8 (else change --digits/--op); p0_hat near 0.1;")
        print("takeoff_step not None, or shortcut_rate clearly rising in the logs above (else raise --lr).")
        print("Logs with per-step numbers: Modal dashboard -> Apps -> goldbug.")
        return
    tag = tag or "sweep"
    if mode == "status":
        done = list_done.remote(tag)
        print(f"{len(done)} runs finished under tag '{tag}'")
        for k in done:
            print("  ", k)
        return
    if mode == "collect":
        rows = fetch.remote(tag)
        out = Path("results") / f"llm_{tag}.jsonl"
        out.parent.mkdir(exist_ok=True)
        out.write_text("".join(r.strip() + "\n" for r in rows))
        print(f"wrote {len(rows)} runs to {out}")
        return
    if mode != "sweep":
        raise SystemExit(f"unknown mode {mode!r}")
    fl = lambda x: tuple(float(v) for v in x.split(","))
    cfgs = _sweep_configs(model, lr, optim, [int(s) for s in seeds.split(",")], rl_steps,
                          qs=fl(qs), p0s=fl(p0s), Gs=tuple(int(v) for v in gs.split(",")))
    for c in cfgs:
        c.update(op=op, digits=digits)
    done = set(list_done.remote(tag))
    todo = [c for c in cfgs if _key(c) not in done]
    if max_runs:
        todo = todo[:max_runs]
    print(f"{len(cfgs)} configs, {len(cfgs) - len(todo)} already done, launching {len(todo)} on {gpu} "
          f"(up to 4 in parallel).")
    calls = [runner.spawn(c, tag) for c in todo]  # spawned calls keep running under `modal run --detach`
    for call in calls:
        try:
            print(json.dumps(call.get()))
        except Exception as e:  # one failed run should not stop the others
            print(f"run failed: {type(e).__name__}: {e}")
    print(f"done. Next: modal run scripts/modal_app.py --mode collect --tag {tag}")
