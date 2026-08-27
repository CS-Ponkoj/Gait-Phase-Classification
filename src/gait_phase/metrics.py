"""Publication metrics with subject-level aggregation and uncertainty."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .constants import ALLOWED_TRANSITIONS, PHASES, PHASE_TO_INDEX


def _indices(values: pd.Series) -> np.ndarray:
    unknown = sorted(set(values.astype(str)) - set(PHASES))
    if unknown:
        raise ValueError(f"Unknown phase labels: {unknown}")
    return values.map(PHASE_TO_INDEX).to_numpy(dtype=int)


def transition_validity(frame: pd.DataFrame) -> float:
    required = {"subject_id", "sequence_id", "frame_index", "y_pred"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing transition columns: {sorted(missing)}")
    outcomes: list[bool] = []
    ordered = frame.copy()
    ordered["_numeric_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="raise")
    ordered = ordered.sort_values(["subject_id", "sequence_id", "_numeric_frame_index"])
    for _, sequence in ordered.groupby(["subject_id", "sequence_id"], sort=False):
        predictions = _indices(sequence["y_pred"])
        outcomes.extend(next_value in ALLOWED_TRANSITIONS[current] for current, next_value in zip(predictions, predictions[1:]))
    return float(np.mean(outcomes)) if outcomes else float("nan")


def compute_metrics(frame: pd.DataFrame) -> dict[str, object]:
    required = {"subject_id", "y_true", "y_pred"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction table is empty.")
    y_true, y_pred = _indices(frame["y_true"]), _indices(frame["y_pred"])
    labels = list(range(len(PHASES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    subject_scores = [
        f1_score(_indices(group["y_true"]), _indices(group["y_pred"]), labels=labels, average="macro", zero_division=0)
        for _, group in frame.groupby("subject_id")
    ]
    per_class = {
        phase: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, phase in enumerate(PHASES)
    }
    result: dict[str, object] = {
        "samples": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "subject_macro_f1": float(np.mean(subject_scores)),
        "frame_macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }
    if {"sequence_id", "frame_index"}.issubset(frame.columns):
        result["transition_validity"] = transition_validity(frame)
    return result


def subject_bootstrap_interval(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260827,
) -> dict[str, float]:
    subjects = frame["subject_id"].drop_duplicates().to_numpy()
    if len(subjects) < 2:
        raise ValueError("Subject-clustered bootstrap requires at least two subjects.")
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    indexed = {subject: group for subject, group in frame.groupby("subject_id")}
    for _ in range(iterations):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        pieces = []
        for occurrence, subject in enumerate(selected):
            piece = indexed[subject].copy()
            piece["subject_id"] = f"{subject}#bootstrap-{occurrence}"
            pieces.append(piece)
        samples.append(float(metric(pd.concat(pieces, ignore_index=True))))
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(metric(frame)),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
    }


def subject_macro_f1(frame: pd.DataFrame) -> float:
    return float(compute_metrics(frame)["subject_macro_f1"])


def paired_subject_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    iterations: int = 2000,
    seed: int = 20260827,
) -> dict[str, float]:
    keys = ["sample_id", "subject_id", "y_true"]
    merged = left[keys + ["y_pred"]].merge(
        right[keys + ["y_pred"]], on=keys, suffixes=("_left", "_right"), validate="one_to_one"
    )
    subjects = merged["subject_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(iterations):
        chosen = rng.choice(subjects, size=len(subjects), replace=True)
        left_parts, right_parts = [], []
        for occurrence, subject in enumerate(chosen):
            group = merged[merged["subject_id"] == subject].copy()
            group["subject_id"] = f"{subject}#{occurrence}"
            left_parts.append(group.rename(columns={"y_pred_left": "y_pred"}))
            right_parts.append(group.rename(columns={"y_pred_right": "y_pred"}))
        left_score = subject_macro_f1(pd.concat(left_parts, ignore_index=True))
        right_score = subject_macro_f1(pd.concat(right_parts, ignore_index=True))
        differences.append(left_score - right_score)
    estimate = subject_macro_f1(left) - subject_macro_f1(right)
    return {
        "estimate": float(estimate),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
        "probability_left_better": float(np.mean(np.asarray(differences) > 0)),
    }
