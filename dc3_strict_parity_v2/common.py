"""Shared utilities for strict DC3 provenance and reproducibility.

The official source under ``vendor/DC3`` is never edited.  This module only
provides import shims for optional packages that DC3 imports globally although
the ACOPF57 execution path does not call them.
"""
from __future__ import annotations

import importlib
import pathlib
import random
import sys
import types

import numpy as np
import torch


ROOT = pathlib.Path(__file__).resolve().parent
UPSTREAM = ROOT / "vendor" / "DC3"
UPSTREAM_COMMIT = "35437af7f22390e4ed032d9eef90cc525764d26f"


def seed_everything(seed: int) -> None:
    """Set every RNG used by the upstream training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _install_optional_import_shims() -> None:
    """Shim imports unused by the ACOPF57 and adapted DC execution paths."""
    try:
        import setproctitle  # noqa: F401
    except ImportError:
        module = types.ModuleType("setproctitle")
        module.setproctitle = lambda _title: None
        sys.modules["setproctitle"] = module

    for name in ("osqp", "ipopt"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = types.ModuleType(name)

    try:
        importlib.import_module("qpth.qp")
    except ImportError:
        qpth = types.ModuleType("qpth")
        qp = types.ModuleType("qpth.qp")
        qp.QPFunction = object
        qpth.qp = qp
        sys.modules["qpth"] = qpth
        sys.modules["qpth.qp"] = qp


def import_upstream():
    """Import the pinned authors' modules without modifying their files."""
    _install_optional_import_shims()
    sys.path.insert(0, str(UPSTREAM))
    try:
        method = importlib.import_module("method")
        utils = importlib.import_module("utils")
        defaults = importlib.import_module("default_args")
    finally:
        sys.path.pop(0)
    return method, utils, defaults

