"""Shared bootstrap helpers for evaluation scripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DEFAULT_OUTPUT_DIR = EVAL_DIR / "results"


def ensure_paths() -> None:
    """Ensure local package imports work when scripts are run directly."""
    for path in (str(EVAL_DIR), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def output_dir() -> Path:
    """Return the local directory used for generated evaluation artifacts."""
    out = Path(os.getenv("EVAL_OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))).expanduser()
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    return out
