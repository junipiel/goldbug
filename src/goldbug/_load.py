"""Helpers for importing grader modules from upstream projects without importing
their (often heavy) package ``__init__`` files.

Many graders live inside large frameworks (verl imports ray, OpenCompass imports
its whole registry). We only want the pure-Python scoring functions, so we load
them by file path and stub out the parent packages / unrelated imports.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def namespace_package(name: str, path: str | Path) -> types.ModuleType:
    """Register ``name`` as a namespace-like package rooted at ``path``
    without executing its ``__init__.py``."""
    import importlib.machinery
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
    spec.submodule_search_locations = [str(path)]
    mod.__spec__ = spec
    sys.modules[name] = mod
    return mod


@contextmanager
def stub_modules(stubs: dict[str, types.ModuleType]) -> Iterator[None]:
    """Temporarily insert stub modules into ``sys.modules``."""
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load_file(module_name: str, path: str | Path) -> types.ModuleType:
    """Import a single python file as ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_stub(name: str, **attrs: object) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def site_package_dir(dist: str) -> Path:
    """Directory of an installed top-level package, found without importing it."""
    spec = importlib.util.find_spec(dist)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(f"package {dist!r} is not installed")
    return Path(list(spec.submodule_search_locations)[0])
