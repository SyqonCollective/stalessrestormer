"""Utility helpers for Restormer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a") as fh:
        fh.write(json.dumps(metrics) + "\n")
