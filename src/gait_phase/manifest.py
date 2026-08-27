"""Dataset manifest creation and publication-gate validation."""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .constants import MANIFEST_COLUMNS, PHASES
from .hashing import perceptual_hash, sha256_file

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
CASIA_A_NAME = re.compile(r"(?P<subject>[^-]+)-(?P<view>[^_]+)_(?P<sequence>[^-]+)-(?P<frame>\d+)")
HASH_WORKERS = min(16, (os.cpu_count() or 4) + 4)


def _sample_id(dataset: str, relative_path: str) -> str:
    return hashlib.sha256(f"{dataset}:{relative_path}".encode("utf-8")).hexdigest()[:20]


def _blank_row() -> dict[str, object]:
    return {column: "" for column in MANIFEST_COLUMNS}


def scan_casia_a(root: str | Path, workspace: str | Path) -> list[dict[str, object]]:
    root, workspace = Path(root), Path(workspace)
    rows: list[dict[str, object]] = []
    if not root.exists():
        raise FileNotFoundError(f"CASIA A curated subset not found: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)

    def process(path: Path) -> dict[str, object]:
        match = CASIA_A_NAME.match(path.stem)
        if not match:
            raise ValueError(f"Unrecognized CASIA A filename: {path.name}")
        parts = match.groupdict()
        relative = path.relative_to(workspace).as_posix()
        row = _blank_row()
        row.update(
            sample_id=_sample_id("casia_a_curated", relative),
            dataset="casia_a_curated",
            subject_id=f"casia_a:{parts['subject']}",
            sequence_id=f"casia_a:{parts['subject']}:{parts['view']}:{parts['sequence']}",
            condition="unspecified",
            view=parts["view"],
            frame_index=int(parts["frame"]),
            relative_path=relative,
            sha256=sha256_file(path),
            perceptual_hash=perceptual_hash(path),
            legacy_label=path.parent.name,
        )
        return row

    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        rows.extend(executor.map(process, paths))
    return rows


def scan_casia_c(root: str | Path, workspace: str | Path) -> list[dict[str, object]]:
    root, workspace = Path(root), Path(workspace)
    rows: list[dict[str, object]] = []
    images = sorted(p for p in root.rglob("*.png") if not any(part.lower().endswith(".zip") for part in p.parts))
    if not images:
        raise FileNotFoundError(f"No extracted CASIA C images found beneath: {root}")
    def process(path: Path) -> dict[str, object]:
        relative_to_root = path.relative_to(root)
        parts = relative_to_root.parts
        if len(parts) < 3:
            raise ValueError(f"Unrecognized CASIA C path: {relative_to_root}")
        subject = next((part for part in parts[:-2] if part.isdigit() and len(part) == 3), parts[-3])
        condition = parts[-2]
        frame_match = re.search(r"\d+", path.stem)
        if not frame_match:
            raise ValueError(f"Unrecognized CASIA C frame: {path.name}")
        relative = path.relative_to(workspace).as_posix()
        row = _blank_row()
        row.update(
            sample_id=_sample_id("casia_c", relative),
            dataset="casia_c",
            subject_id=f"casia_c:{subject}",
            sequence_id=f"casia_c:{subject}:{condition}",
            condition=condition,
            view="side_thermal",
            frame_index=int(frame_match.group()),
            relative_path=relative,
            sha256=sha256_file(path),
            perceptual_hash=perceptual_hash(path),
        )
        return row

    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        rows.extend(executor.map(process, images))
    return rows


def assign_duplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    counts = Counter(result["sha256"])
    duplicate_hashes = sorted(value for value, count in counts.items() if count > 1)
    mapping = {value: f"exact-{index:06d}" for index, value in enumerate(duplicate_hashes, 1)}
    result["duplicate_group"] = result["sha256"].map(mapping).fillna("")
    return result


def build_manifest(casia_a_root: str | Path, casia_c_root: str | Path, workspace: str | Path) -> pd.DataFrame:
    rows = scan_casia_a(casia_a_root, workspace) + scan_casia_c(casia_c_root, workspace)
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return assign_duplicate_groups(frame).sort_values(
        ["dataset", "subject_id", "sequence_id", "frame_index", "relative_path"]
    ).reset_index(drop=True)


def validate_manifest(frame: pd.DataFrame, frozen: bool = False) -> list[str]:
    errors: list[str] = []
    missing = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        return [f"Missing manifest columns: {', '.join(missing)}"]
    if frame.empty:
        errors.append("Manifest contains no samples.")
        return errors
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        errors.append("sample_id must be present and unique.")
    for column in ("dataset", "subject_id", "sequence_id", "relative_path", "sha256"):
        if frame[column].fillna("").astype(str).str.strip().eq("").any():
            errors.append(f"{column} contains blank values.")
    if not frame["subject_id"].astype(str).str.contains(":", regex=False).all():
        errors.append("subject_id values must be namespaced with the dataset.")
    labels = frame["adjudicated_label"].fillna("").astype(str)
    invalid_labels = sorted(set(labels) - {"", *PHASES})
    if invalid_labels:
        errors.append(f"Invalid adjudicated labels: {invalid_labels}")
    if frozen:
        eligible = frame["exclusion_reason"].fillna("").astype(str).eq("")
        if labels[eligible].eq("").any():
            errors.append("Frozen manifest has eligible samples without adjudicated labels.")
        if frame.loc[eligible, "split"].fillna("").astype(str).eq("").any():
            errors.append("Frozen manifest has eligible samples without split assignments.")
    assigned = frame[frame["split"].fillna("").astype(str).ne("")]
    if not assigned.empty:
        subject_splits = assigned.groupby("subject_id")["split"].nunique()
        if (subject_splits > 1).any():
            errors.append("At least one subject crosses train/development and test partitions.")
        eligible_assigned = assigned[assigned["exclusion_reason"].fillna("").astype(str).eq("")]
        duplicates = eligible_assigned[eligible_assigned["duplicate_group"].fillna("").astype(str).ne("")]
        if not duplicates.empty and (duplicates.groupby("duplicate_group")["split"].nunique() > 1).any():
            errors.append("At least one exact duplicate group crosses partitions.")
    conflicts = frame[labels.ne("")].groupby("sha256")["adjudicated_label"].nunique()
    if (conflicts > 1).any():
        errors.append("Identical image content has conflicting adjudicated labels.")
    return errors


def manifest_summary(frame: pd.DataFrame) -> dict[str, object]:
    condition_counts = (
        frame.groupby(["dataset", "condition"]).size().rename("samples").reset_index().to_dict("records")
    )
    return {
        "samples": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "sequences": int(frame["sequence_id"].nunique()),
        "datasets": frame.groupby("dataset").size().astype(int).to_dict(),
        "conditions": condition_counts,
        "exact_duplicate_groups": int(frame.loc[frame["duplicate_group"].ne(""), "duplicate_group"].nunique()),
        "adjudicated_samples": int(frame["adjudicated_label"].fillna("").ne("").sum()),
        "excluded_samples": int(frame["exclusion_reason"].fillna("").ne("").sum()),
    }


def duplicate_audit(frame: pd.DataFrame) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    exact = (
        frame[frame["duplicate_group"].fillna("").ne("")]
        .groupby("duplicate_group")
        .agg(samples=("sample_id", "size"), subjects=("subject_id", "nunique"), datasets=("dataset", "nunique"))
        .reset_index()
    )
    perceptual = (
        frame.groupby("perceptual_hash")
        .agg(samples=("sample_id", "size"), subjects=("subject_id", "nunique"), datasets=("dataset", "nunique"), sequences=("sequence_id", "nunique"))
        .reset_index()
    )
    perceptual = perceptual[perceptual["samples"] > 1].sort_values(
        ["subjects", "samples", "perceptual_hash"], ascending=[False, False, True]
    )
    summary = {
        "exact_groups": int(len(exact)),
        "exact_members": int(exact["samples"].sum()) if not exact.empty else 0,
        "exact_cross_subject_groups": int((exact["subjects"] > 1).sum()) if not exact.empty else 0,
        "perceptual_collision_groups": int(len(perceptual)),
        "perceptual_collision_members": int(perceptual["samples"].sum()) if not perceptual.empty else 0,
        "perceptual_cross_subject_groups": int((perceptual["subjects"] > 1).sum()) if not perceptual.empty else 0,
        "perceptual_cross_dataset_groups": int((perceptual["datasets"] > 1).sum()) if not perceptual.empty else 0,
    }
    return summary, exact, perceptual
