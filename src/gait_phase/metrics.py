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


def boundary_metrics(frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, object]:
    """Measure boundary classification and nearest-boundary timing error."""
    required = {"sequence_id", "frame_index", "y_boundary", "boundary_probability"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing boundary columns: {sorted(missing)}")
    truth = pd.to_numeric(frame["y_boundary"], errors="raise").to_numpy(dtype=int)
    probabilities = pd.to_numeric(frame["boundary_probability"], errors="raise").to_numpy(dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=[1],
        average="binary",
        zero_division=0,
    )
    errors: list[float] = []
    working = frame.copy()
    working["_frame"] = pd.to_numeric(working["frame_index"], errors="raise")
    working["_probability"] = probabilities
    working["_predicted"] = predicted
    working["_truth"] = truth
    for _, sequence in working.groupby("sequence_id", sort=False):
        sequence = sequence.sort_values("_frame").reset_index(drop=True)
        true_frames = sequence.loc[sequence["_truth"].eq(1), "_frame"].to_numpy(dtype=float)
        predicted_rows = sequence[sequence["_predicted"].eq(1)].copy()
        predicted_frames: list[float] = []
        if not predicted_rows.empty:
            predicted_rows["_group"] = predicted_rows["_frame"].diff().fillna(2).ne(1).cumsum()
            for _, group in predicted_rows.groupby("_group"):
                best = group.loc[group["_probability"].idxmax()]
                predicted_frames.append(float(best["_frame"]))
        if predicted_frames:
            errors.extend(float(np.min(np.abs(np.asarray(predicted_frames) - true_frame))) for true_frame in true_frames)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_boundaries": int(truth.sum()),
        "predicted_boundary_frames": int(predicted.sum()),
        "matched_true_boundaries": int(len(errors)),
        "mean_absolute_error_frames": float(np.mean(errors)) if errors else None,
        "median_absolute_error_frames": float(np.median(errors)) if errors else None,
        "within_one_frame": float(np.mean(np.asarray(errors) <= 1)) if errors else None,
        "within_two_frames": float(np.mean(np.asarray(errors) <= 2)) if errors else None,
    }


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
    if {"condition", "subject_id"}.issubset(frame.columns):
        condition_results = {}
        for condition, group in frame.groupby("condition", sort=True):
            scores = [
                f1_score(
                    _indices(subject["y_true"]),
                    _indices(subject["y_pred"]),
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
                for _, subject in group.groupby("subject_id")
            ]
            condition_results[str(condition)] = {
                "samples": int(len(group)),
                "subjects": int(group["subject_id"].nunique()),
                "subject_macro_f1": float(np.mean(scores)),
            }
        result["by_condition"] = condition_results
    if {"sequence_id", "frame_index", "y_boundary", "boundary_probability"}.issubset(frame.columns):
        result["boundary"] = boundary_metrics(frame)
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


def subject_macro_f1_scores(frame: pd.DataFrame) -> pd.Series:
    """Return one macro-F1 value per subject for efficient clustered inference."""
    labels = list(range(len(PHASES)))
    return frame.groupby("subject_id", sort=True).apply(
        lambda group: f1_score(
            _indices(group["y_true"]),
            _indices(group["y_pred"]),
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        include_groups=False,
    )


def subject_macro_f1_interval(
    frame: pd.DataFrame,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260827,
) -> dict[str, float]:
    """Bootstrap the mean subject-level macro F1 without rebuilding frame tables."""
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    scores = subject_macro_f1_scores(frame).to_numpy(dtype=float)
    if len(scores) < 2:
        raise ValueError("Subject-clustered bootstrap requires at least two subjects.")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(scores, size=(iterations, len(scores)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(scores.mean()),
        "lower": float(np.quantile(sampled, alpha)),
        "upper": float(np.quantile(sampled, 1.0 - alpha)),
        "confidence": confidence,
        "iterations": iterations,
    }


def paired_subject_bootstrap_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    iterations: int = 2000,
    seed: int = 20260827,
) -> dict[str, float]:
    if iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    keys = ["sample_id", "subject_id", "y_true"]
    merged = left[keys + ["y_pred"]].merge(
        right[keys + ["y_pred"]], on=keys, suffixes=("_left", "_right"), validate="one_to_one"
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired predictions must contain exactly the same samples, subjects, and labels.")
    left_scores = subject_macro_f1_scores(
        merged.rename(columns={"y_pred_left": "y_pred"})
    )
    right_scores = subject_macro_f1_scores(
        merged.rename(columns={"y_pred_right": "y_pred"})
    )
    if not left_scores.index.equals(right_scores.index):
        raise ValueError("Paired predictions must contain exactly the same subjects.")
    score_differences = left_scores.to_numpy(dtype=float) - right_scores.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    differences = rng.choice(
        score_differences,
        size=(iterations, len(score_differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "estimate": float(score_differences.mean()),
        "lower": float(np.quantile(differences, 0.025)),
        "upper": float(np.quantile(differences, 0.975)),
        "probability_left_better": float(np.mean(differences > 0)),
    }
