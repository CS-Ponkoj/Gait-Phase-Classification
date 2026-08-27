"""Boundary-based four-phase annotation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .constants import PHASES

BOUNDARY_COLUMNS = (
    "dataset",
    "subject_id",
    "sequence_id",
    "cycle_id",
    "initial_contact_frame",
    "mid_stance_frame",
    "terminal_stance_frame",
    "swing_frame",
    "next_initial_contact_frame",
    "annotator_id",
    "confidence",
    "notes",
)


def validate_boundaries(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in BOUNDARY_COLUMNS if column not in frame.columns]
    if missing:
        return [f"Missing annotation columns: {', '.join(missing)}"]
    boundary_columns = BOUNDARY_COLUMNS[4:9]
    for index, row in frame.iterrows():
        try:
            values = [int(row[column]) for column in boundary_columns]
        except (TypeError, ValueError):
            errors.append(f"Row {index}: boundary frames must be integers.")
            continue
        if values != sorted(values) or len(set(values)) != len(values):
            errors.append(f"Row {index}: boundaries must be strictly increasing.")
        confidence = str(row["confidence"]).lower()
        if confidence not in {"high", "medium", "low"}:
            errors.append(f"Row {index}: confidence must be high, medium, or low.")
    if frame.duplicated(["sequence_id", "cycle_id", "annotator_id"]).any():
        errors.append("An annotator may provide only one record per gait cycle.")
    return errors


def expand_boundary_row(row: pd.Series) -> pd.DataFrame:
    starts = [
        int(row["initial_contact_frame"]),
        int(row["mid_stance_frame"]),
        int(row["terminal_stance_frame"]),
        int(row["swing_frame"]),
    ]
    stop = int(row["next_initial_contact_frame"])
    records: list[dict[str, object]] = []
    for phase_index, start in enumerate(starts):
        end = starts[phase_index + 1] if phase_index < 3 else stop
        for frame_index in range(start, end):
            records.append(
                {
                    "sequence_id": row["sequence_id"],
                    "cycle_id": row["cycle_id"],
                    "annotator_id": row["annotator_id"],
                    "frame_index": frame_index,
                    "label": PHASES[phase_index],
                }
            )
    return pd.DataFrame(records)


def expand_boundaries(frame: pd.DataFrame) -> pd.DataFrame:
    errors = validate_boundaries(frame)
    if errors:
        raise ValueError("; ".join(errors))
    expanded = [expand_boundary_row(row) for _, row in frame.iterrows()]
    return pd.concat(expanded, ignore_index=True) if expanded else pd.DataFrame()


def annotation_agreement(labels_a: Iterable[str], labels_b: Iterable[str]) -> dict[str, float]:
    a, b = list(labels_a), list(labels_b)
    if len(a) != len(b) or not a:
        raise ValueError("Agreement requires two non-empty label lists of equal length.")
    indices = {label: index for index, label in enumerate(PHASES)}
    try:
        a_index = [indices[label] for label in a]
        b_index = [indices[label] for label in b]
    except KeyError as error:
        raise ValueError(f"Unknown phase label: {error.args[0]}") from error
    return {
        "raw_agreement": float(np.mean(np.asarray(a_index) == np.asarray(b_index))),
        "weighted_kappa": float(cohen_kappa_score(a_index, b_index, weights="quadratic")),
    }


def boundary_disagreement(left: pd.DataFrame, right: pd.DataFrame, fps: float) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    keys = ["sequence_id", "cycle_id"]
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    boundary_names = BOUNDARY_COLUMNS[4:9]
    differences = []
    for name in boundary_names:
        differences.extend((merged[f"{name}_a"].astype(int) - merged[f"{name}_b"].astype(int)).abs().tolist())
    if not differences:
        raise ValueError("No matching cycles were available for boundary comparison.")
    mean_frames = float(np.mean(differences))
    return {"mean_absolute_frames": mean_frames, "mean_absolute_milliseconds": mean_frames * 1000.0 / fps}
