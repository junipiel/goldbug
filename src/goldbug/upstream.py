"""Pinned upstream grader files that are not pip-installable.

We do not vendor third-party code. Instead we download the exact upstream file
on first use, check its SHA-256 against the pin below, and cache it. If upstream
changes, the hash check fails loudly -- results are always tied to an exact
version of the checker.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

RAW = "https://raw.githubusercontent.com/"

# name -> (repo path on GitHub raw, sha256 of the file content)
PINS: dict[str, tuple[str, str]] = {
    "qwen25math/grader.py": (
        "QwenLM/Qwen2.5-Math/main/evaluation/grader.py",
        "ed9494a388e7117735fccf2985de6c444d5bdba4d493b425e92a91e9ab97742a",
    ),
    "qwen25math/parser.py": (
        "QwenLM/Qwen2.5-Math/main/evaluation/parser.py",
        "2bee38c8721c3c0ce120edd73de6ad5bf2e4aa7bf14b988f96aa1d590153e8fe",
    ),
    "rllm/math_utils.py": (
        "agentica-project/rllm/main/rllm/rewards/math_utils/utils.py",
        "26187dd66709a006ce134bf2a4fad75b98126e9ee50a892e409a9e1032331128",
    ),
}


def cache_dir() -> Path:
    d = Path(os.environ.get("GOLDBUG_CACHE", Path.home() / ".cache" / "goldbug" / "upstream"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(name: str) -> Path:
    """Return a local path to the pinned upstream file, downloading if needed."""
    url_path, sha = PINS[name]
    dest = cache_dir() / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RAW + url_path, timeout=30) as r:  # noqa: S310 (pinned host)
            dest.write_bytes(r.read())
    got = hashlib.sha256(dest.read_bytes()).hexdigest()
    if got != sha:
        raise RuntimeError(
            f"upstream file {name} changed (sha256 {got[:12]}... != pinned {sha[:12]}...). "
            "Re-pin deliberately after reviewing the diff."
        )
    return dest
