"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .constants import PHASES


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    labels = tuple(config.get("study", {}).get("labels", ()))
    if labels != PHASES:
        raise ValueError(f"The study must use the locked four-class order: {PHASES}")
    folds = int(config.get("data", {}).get("development_folds", 0))
    if folds != 5:
        raise ValueError("The publication protocol requires exactly five development folds.")
    return config
