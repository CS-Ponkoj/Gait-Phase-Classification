"""Classical reproducible baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .constants import PHASES
from .models import hog_descriptor


class MajorityBaseline:
    def __init__(self) -> None:
        self.label: str | None = None

    def fit(self, frame: pd.DataFrame) -> "MajorityBaseline":
        if frame.empty:
            raise ValueError("Cannot fit majority baseline on an empty table.")
        self.label = str(frame["target_label"].value_counts().sort_index().idxmax())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.label is None:
            raise RuntimeError("Majority baseline has not been fitted.")
        return np.repeat(self.label, len(frame))


class CyclePriorBaseline:
    def __init__(self, bins: int = 20) -> None:
        self.bins = bins
        self.lookup: dict[int, str] = {}
        self.fallback: str | None = None

    def fit(self, frame: pd.DataFrame) -> "CyclePriorBaseline":
        positions = pd.to_numeric(frame["cycle_position"], errors="coerce")
        usable = frame[positions.notna()].copy()
        if usable.empty:
            raise ValueError("Cycle-prior baseline requires cycle_position annotations.")
        usable["position_bin"] = np.minimum((positions[positions.notna()] * self.bins).astype(int), self.bins - 1)
        self.fallback = str(usable["target_label"].value_counts().idxmax())
        for position_bin, group in usable.groupby("position_bin"):
            self.lookup[int(position_bin)] = str(group["target_label"].value_counts().idxmax())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.fallback is None:
            raise RuntimeError("Cycle-prior baseline has not been fitted.")
        positions = pd.to_numeric(frame["cycle_position"], errors="coerce")
        result = []
        for position in positions:
            if pd.isna(position):
                result.append(self.fallback)
            else:
                result.append(self.lookup.get(min(int(position * self.bins), self.bins - 1), self.fallback))
        return np.asarray(result)


class HogSvmBaseline:
    def __init__(self, workspace: str | Path, random_state: int = 20260827) -> None:
        self.workspace = Path(workspace)
        self.model = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=random_state, dual="auto"),
        )

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for relative_path in frame["relative_path"]:
            with Image.open(self.workspace / str(relative_path)) as image:
                rows.append(hog_descriptor(image))
        return np.stack(rows)

    def fit(self, frame: pd.DataFrame) -> "HogSvmBaseline":
        self.model.fit(self._features(frame), frame["target_label"].astype(str))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        predictions = self.model.predict(self._features(frame))
        if not set(predictions).issubset(PHASES):
            raise RuntimeError("HOG+SVM emitted a label outside the four-class contract.")
        return predictions
