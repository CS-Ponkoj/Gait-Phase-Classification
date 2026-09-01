"""Deterministic AI-provisional annotation for complete CASIA C sequences."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .constants import PHASES

FRAME_LABEL_COLUMNS = (
    "sample_id",
    "dataset",
    "subject_id",
    "sequence_id",
    "condition",
    "frame_index",
    "relative_path",
    "split",
    "fold",
    "phase_label",
    "phase_index",
    "cycle_position",
    "confidence",
    "annotation_source",
    "annotation_version",
)

BOUNDARY_COLUMNS = (
    "dataset",
    "subject_id",
    "sequence_id",
    "cycle_id",
    "contact_start_frame",
    "mid_stance_frame",
    "terminal_stance_frame",
    "swing_frame",
    "next_contact_frame",
    "confidence",
    "annotation_source",
    "annotation_version",
    "notes",
)

PHASE_THRESHOLDS = (0.12, 0.31, 0.62)
EXPECTED_STEP_FRAMES = {"fn": 13, "fs": 16, "fq": 12, "fb": 13}


def lower_limb_spread(path: str | Path) -> float:
    """Return a noise-resistant lower-silhouette horizontal spread ratio."""
    with Image.open(path) as source:
        foreground = np.asarray(source.convert("L")) > 0
    ys, xs = np.where(foreground)
    if len(xs) < 20:
        return float("nan")
    y_low, y_high = np.percentile(ys, [1, 99])
    height = max(float(y_high - y_low + 1), 1.0)
    lower_x = xs[ys >= y_low + 0.60 * height]
    if len(lower_x) < 10:
        return float("nan")
    x_low, x_high = np.percentile(lower_x, [2, 98])
    return float((x_high - x_low + 1) / height)


def smooth_signal(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("A non-empty one-dimensional signal is required.")
    if np.isnan(values).all():
        return np.zeros_like(values)
    valid = np.flatnonzero(~np.isnan(values))
    values = np.interp(np.arange(len(values)), valid, values[valid])
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(values)]


def detect_contact_peaks(
    values: np.ndarray,
    min_distance: int = 7,
    prominence: float = 0.015,
) -> list[int]:
    """Detect separated local maxima without adding a SciPy dependency."""
    signal = smooth_signal(values)
    radius = max(3, min_distance // 2)
    candidates: list[tuple[float, float, int]] = []
    for index in range(1, len(signal) - 1):
        if signal[index] < signal[index - 1] or signal[index] < signal[index + 1]:
            continue
        left = signal[max(0, index - radius) : index]
        right = signal[index + 1 : min(len(signal), index + radius + 1)]
        if not len(left) or not len(right):
            continue
        local_prominence = signal[index] - max(float(left.min()), float(right.min()))
        if local_prominence >= prominence:
            candidates.append((local_prominence, float(signal[index]), index))
    selected: list[int] = []
    for _, _, index in sorted(candidates, reverse=True):
        if all(abs(index - existing) >= min_distance for existing in selected):
            selected.append(index)
    return sorted(selected)


def contact_grid(
    frame_indices: np.ndarray,
    detected_positions: list[int],
    expected_step_frames: int,
) -> tuple[list[int], str, float, float]:
    """Create a contact grid that covers the entire observed sequence."""
    frames = np.asarray(frame_indices, dtype=int)
    detected = [int(frames[position]) for position in detected_positions]
    method = "detected_contacts"
    if len(detected) >= 3:
        intervals = np.diff(detected).astype(float)
        median_step = float(np.median(intervals))
        interval_cv = float(np.std(intervals) / median_step) if median_step else 1.0
    else:
        method = "condition_cycle_prior"
        median_step = float(expected_step_frames)
        interval_cv = 1.0
        anchor = int(frames[np.argmax(np.zeros(len(frames)))])
        if detected:
            anchor = detected[0]
        detected = [anchor]
    step = max(7, int(round(median_step)))
    grid = sorted(set(detected))
    while grid[0] > int(frames.min()):
        grid.insert(0, grid[0] - step)
    while grid[-1] <= int(frames.max()):
        grid.append(grid[-1] + step)
    expanded = [grid[0]]
    for end in grid[1:]:
        start = expanded[-1]
        gap = end - start
        if gap > 1.65 * step:
            pieces = max(2, int(round(gap / step)))
            expanded.extend(int(round(start + gap * part / pieces)) for part in range(1, pieces))
        expanded.append(end)
    grid = sorted(set(expanded))
    return grid, method, median_step, interval_cv


def label_from_progress(progress: float) -> tuple[str, int]:
    if progress < PHASE_THRESHOLDS[0]:
        return PHASES[0], 0
    if progress < PHASE_THRESHOLDS[1]:
        return PHASES[1], 1
    if progress < PHASE_THRESHOLDS[2]:
        return PHASES[2], 2
    return PHASES[3], 3


def annotate_sequence_from_signal(
    sequence: pd.DataFrame,
    spreads: np.ndarray,
    version: str,
    expected_step_frames: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ordered = sequence.copy()
    ordered["numeric_frame_index"] = pd.to_numeric(ordered["frame_index"], errors="raise")
    ordered = ordered.sort_values(["numeric_frame_index", "relative_path"]).reset_index(drop=True)
    frames = ordered["numeric_frame_index"].to_numpy(dtype=int)
    if len(spreads) != len(ordered):
        raise ValueError("Spread feature count does not match the sequence frame count.")
    if len(frames) != int(frames.max() - frames.min() + 1):
        raise ValueError(f"Sequence is not frame-contiguous: {ordered['sequence_id'].iloc[0]}")
    peaks = detect_contact_peaks(spreads)
    grid, method, median_step, interval_cv = contact_grid(frames, peaks, expected_step_frames)
    confidence = "medium" if method == "detected_contacts" and interval_cv <= 0.30 else "low"
    labels: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        frame_index = int(row["numeric_frame_index"])
        grid_position = int(np.searchsorted(grid, frame_index, side="right") - 1)
        start, end = grid[grid_position], grid[grid_position + 1]
        progress = min(max((frame_index - start) / (end - start), 0.0), 0.999999)
        label, phase_index = label_from_progress(progress)
        labels.append(
            {
                **{column: row[column] for column in FRAME_LABEL_COLUMNS[:9]},
                "phase_label": label,
                "phase_index": phase_index,
                "cycle_position": round(progress, 6),
                "confidence": confidence,
                "annotation_source": "ai_provisional_silhouette_timing",
                "annotation_version": version,
            }
        )
    boundaries: list[dict[str, object]] = []
    first, last = int(frames.min()), int(frames.max())
    for cycle_number, (start, end) in enumerate(zip(grid[:-1], grid[1:]), 1):
        if start < first or end > last:
            continue
        length = end - start
        boundaries.append(
            {
                "dataset": ordered["dataset"].iloc[0],
                "subject_id": ordered["subject_id"].iloc[0],
                "sequence_id": ordered["sequence_id"].iloc[0],
                "cycle_id": f"step_{cycle_number:03d}",
                "contact_start_frame": start,
                "mid_stance_frame": start + max(1, int(round(length * PHASE_THRESHOLDS[0]))),
                "terminal_stance_frame": start + max(2, int(round(length * PHASE_THRESHOLDS[1]))),
                "swing_frame": start + max(3, int(round(length * PHASE_THRESHOLDS[2]))),
                "next_contact_frame": end,
                "confidence": confidence,
                "annotation_source": "ai_provisional_silhouette_timing",
                "annotation_version": version,
                "notes": "Observable step-cycle timing; reference limb is not anatomically identified.",
            }
        )
    qc = {
        "dataset": ordered["dataset"].iloc[0],
        "subject_id": ordered["subject_id"].iloc[0],
        "sequence_id": ordered["sequence_id"].iloc[0],
        "condition": ordered["condition"].iloc[0],
        "split": ordered["split"].iloc[0],
        "fold": ordered["fold"].iloc[0],
        "frame_count": int(len(ordered)),
        "detected_contact_count": int(len(peaks)),
        "median_step_frames": round(median_step, 4),
        "step_interval_cv": round(interval_cv, 6),
        "coverage_method": method,
        "confidence": confidence,
        "missing_feature_frames": int(np.isnan(spreads).sum()),
    }
    return pd.DataFrame(labels, columns=FRAME_LABEL_COLUMNS), pd.DataFrame(boundaries, columns=BOUNDARY_COLUMNS), qc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, index=False)


def build_provisional_release(
    manifest: pd.DataFrame,
    workspace: str | Path,
    output: str | Path,
    settings: dict[str, Any],
) -> dict[str, object]:
    workspace, output = Path(workspace), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Annotation output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    version = str(settings["version"])
    eligible = manifest[
        manifest["dataset"].eq("casia_c") & manifest["exclusion_reason"].fillna("").eq("")
    ].copy()
    eligible_sample_ids = set(eligible["sample_id"])
    casia_c = manifest[manifest["dataset"].eq("casia_c")].copy()
    frame_outputs: list[pd.DataFrame] = []
    boundary_outputs: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []
    for _, sequence in casia_c.groupby("sequence_id", sort=True):
        sequence = sequence.copy()
        sequence["numeric_frame_index"] = pd.to_numeric(sequence["frame_index"], errors="raise")
        sequence = sequence.sort_values(["numeric_frame_index", "relative_path"]).reset_index(drop=True)
        spreads = np.asarray(
            [lower_limb_spread(workspace / relative) for relative in sequence["relative_path"]],
            dtype=float,
        )
        prefix = str(sequence["condition"].iloc[0])[:2].lower()
        expected = int(settings.get("expected_step_frames", EXPECTED_STEP_FRAMES).get(prefix, 13))
        frame_labels, boundaries, qc = annotate_sequence_from_signal(sequence, spreads, version, expected)
        frame_outputs.append(frame_labels[frame_labels["sample_id"].isin(eligible_sample_ids)].copy())
        boundary_outputs.append(boundaries)
        qc["eligible_frame_count"] = int(frame_labels["sample_id"].isin(eligible_sample_ids).sum())
        qc_rows.append(qc)
    labels = pd.concat(frame_outputs, ignore_index=True)
    boundaries = pd.concat(boundary_outputs, ignore_index=True)
    qc = pd.DataFrame(qc_rows)
    excluded = manifest[~manifest["sample_id"].isin(labels["sample_id"])].copy()
    excluded["provisional_exclusion_reason"] = np.where(
        excluded["dataset"].eq("casia_a_curated"),
        "casia_a_incomplete_temporal_sequence",
        excluded["exclusion_reason"].replace("", "not_in_casia_c_annotation_scope"),
    )
    split_summary = (
        labels.groupby(["split", "fold", "phase_label"], dropna=False)
        .agg(frames=("sample_id", "size"), subjects=("subject_id", "nunique"), sequences=("sequence_id", "nunique"))
        .reset_index()
    )
    label_path = output / "frame_labels.csv.gz"
    boundary_path = output / "cycle_boundaries.csv.gz"
    excluded_path = output / "excluded_samples.csv.gz"
    _write_gzip_csv(labels, label_path)
    _write_gzip_csv(boundaries, boundary_path)
    _write_gzip_csv(excluded, excluded_path)
    qc.to_csv(output / "sequence_qc.csv", index=False)
    split_summary.to_csv(output / "split_summary.csv", index=False)
    manifest_path = Path(settings["source_manifest"])
    summary = {
        "annotation_version": version,
        "status": "AI_PROVISIONAL_NOT_PUBLICATION_GROUND_TRUTH",
        "method": "lower-limb silhouette spread contact timing plus fixed operational phase thresholds",
        "reference_limb": "not anatomically identified",
        "phase_thresholds": list(PHASE_THRESHOLDS),
        "labeled_frames": int(len(labels)),
        "excluded_frames": int(len(excluded)),
        "accounted_manifest_frames": int(len(labels) + len(excluded)),
        "sequences": int(labels["sequence_id"].nunique()),
        "subjects": int(labels["subject_id"].nunique()),
        "complete_boundary_cycles": int(len(boundaries)),
        "confidence_counts": labels["confidence"].value_counts().sort_index().astype(int).to_dict(),
        "phase_counts": labels["phase_label"].value_counts().sort_index().astype(int).to_dict(),
        "split_counts": labels["split"].value_counts().sort_index().astype(int).to_dict(),
        "source_manifest": str(manifest_path.as_posix()),
        "source_manifest_sha256": _sha256(manifest_path),
    }
    (output / "provenance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# AI-Provisional Four-Phase Labels\n\n"
        "These files contain deterministic machine-generated labels, not human or clinical ground truth. "
        "They may be used for pipeline development and sensitivity experiments only. The reference limb cannot be anatomically identified from the binary silhouettes. "
        "Do not use these labels to support a publication accuracy claim without independent validation and expert adjudication.\n\n"
        "## Files\n\n"
        "- `frame_labels.csv.gz`: one provisional label for every eligible CASIA C frame, including split and fold.\n"
        "- `cycle_boundaries.csv.gz`: complete observed step-cycle boundaries only.\n"
        "- `sequence_qc.csv`: detection quality and confidence for every sequence.\n"
        "- `excluded_samples.csv.gz`: every manifest row outside the labeled set and its reason.\n"
        "- `split_summary.csv`: frame, subject, and sequence counts by split, fold, and phase.\n"
        "- `provenance.json`: method, source checksum, and release totals.\n"
        "- `checksums.sha256`: integrity hashes for the release files.\n",
        encoding="utf-8",
    )
    artifacts = sorted(path for path in output.iterdir() if path.is_file() and path.name != "checksums.sha256")
    (output / "checksums.sha256").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    return summary
