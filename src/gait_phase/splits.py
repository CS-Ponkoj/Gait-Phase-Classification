"""Deterministic subject-independent split generation."""

from __future__ import annotations

import hashlib
import math

import pandas as pd


def _subject_order(subjects: list[str], seed: int) -> list[str]:
    return sorted(subjects, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())


def mark_cross_subject_duplicate_exclusions(frame: pd.DataFrame) -> pd.DataFrame:
    """Exclude exact content shared by multiple subjects without deleting source rows."""
    result = frame.copy()
    if "exclusion_reason" not in result.columns:
        result["exclusion_reason"] = ""
    duplicates = result[result["duplicate_group"].fillna("").ne("")]
    cross_subject = duplicates.groupby("duplicate_group")["subject_id"].nunique()
    unsafe_groups = set(cross_subject[cross_subject > 1].index)
    mask = result["duplicate_group"].isin(unsafe_groups)
    existing = result.loc[mask, "exclusion_reason"].fillna("").astype(str)
    result.loc[mask, "exclusion_reason"] = existing.where(
        existing.ne(""), "exact_duplicate_shared_across_subjects"
    )
    return result


def make_subject_splits(
    frame: pd.DataFrame,
    test_fraction: float = 0.20,
    folds: int = 5,
    seed: int = 20260827,
) -> pd.DataFrame:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one.")
    if folds < 2:
        raise ValueError("At least two development folds are required.")
    result = mark_cross_subject_duplicate_exclusions(frame)
    result["split"] = ""
    result["fold"] = ""
    for dataset, dataset_rows in result.groupby("dataset"):
        subjects = _subject_order(dataset_rows["subject_id"].drop_duplicates().tolist(), seed)
        if len(subjects) < folds + 1:
            raise ValueError(f"Dataset {dataset} needs at least {folds + 1} subjects for this protocol.")
        test_count = max(1, int(math.ceil(len(subjects) * test_fraction)))
        test_subjects = set(subjects[:test_count])
        development_subjects = subjects[test_count:]
        test_mask = (result["dataset"] == dataset) & result["subject_id"].isin(test_subjects)
        result.loc[test_mask, "split"] = "test"
        result.loc[test_mask, "fold"] = -1
        for index, subject in enumerate(development_subjects):
            mask = (result["dataset"] == dataset) & (result["subject_id"] == subject)
            result.loc[mask, "split"] = "development"
            result.loc[mask, "fold"] = index % folds
    return result


def assert_split_integrity(frame: pd.DataFrame) -> None:
    if frame["split"].fillna("").eq("").any():
        raise ValueError("Every sample must have a split.")
    if (frame.groupby("subject_id")["split"].nunique() > 1).any():
        raise ValueError("Subject leakage detected.")
    eligible = frame
    if "exclusion_reason" in frame.columns:
        eligible = frame[frame["exclusion_reason"].fillna("").eq("")]
    duplicates = eligible[eligible["duplicate_group"].fillna("").ne("")]
    if not duplicates.empty and (duplicates.groupby("duplicate_group")["split"].nunique() > 1).any():
        raise ValueError("Exact duplicate leakage detected.")
