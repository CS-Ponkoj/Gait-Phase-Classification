"""Non-destructive annotation audit; never turns consistency into ground truth."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from gait_phase.constants import PHASES


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def image_check(workspace: Path, relative: str, expected_sha: str) -> dict:
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()):
        return {"image_issues": ["path_outside_workspace"]}
    try:
        raw = path.read_bytes()
        issues = []
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != expected_sha:
            issues.append("source_checksum_mismatch")
        with Image.open(io.BytesIO(raw)) as source:
            pixels = np.asarray(source.convert("L"))
            ys, xs = np.where(pixels > 0)
            if not len(xs):
                issues.append("empty_foreground")
            return {
                "image_issues": issues,
                "actual_sha256": actual_sha,
                "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
                "foreground_pixels": int(len(xs)),
                # Edge contact is a review flag, NOT proof of truncation.
                "foreground_touches_image_edge": bool(len(xs) and (
                    xs.min() == 0 or ys.min() == 0 or xs.max() == pixels.shape[1] - 1
                    or ys.max() == pixels.shape[0] - 1)),
            }
    except (OSError, ValueError) as error:
        return {"image_issues": ["missing_or_unreadable_image"], "error": str(error)}


def boundary_expansion(boundaries: pd.DataFrame) -> pd.DataFrame:
    """Expand the saved provisional boundaries, not a new annotation."""
    records = []
    for row in boundaries.to_dict("records"):
        starts = [int(row[name]) for name in (
            "contact_start_frame", "mid_stance_frame", "terminal_stance_frame", "swing_frame")]
        ends = starts[1:] + [int(row["next_contact_frame"])]
        if any(end <= start for start, end in zip(starts, ends)):
            raise ValueError(f"Unordered boundaries: {row['sequence_id']} {row['cycle_id']}")
        for phase, start, end in zip(PHASES, starts, ends):
            for frame in range(start, end):
                records.append({"sequence_id": row["sequence_id"], "frame_index": str(frame),
                                "boundary_phase": phase, "boundary_cycle_id": row["cycle_id"]})
    result = pd.DataFrame(records, columns=["sequence_id", "frame_index", "boundary_phase", "boundary_cycle_id"])
    if result.duplicated(["sequence_id", "frame_index"]).any():
        raise ValueError("Overlapping saved provisional boundary cycles")
    return result


def compare_boundaries(labels: pd.DataFrame, boundaries: pd.DataFrame) -> pd.DataFrame:
    return labels.merge(boundary_expansion(boundaries), on=["sequence_id", "frame_index"],
                        how="inner", validate="one_to_one")


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def audit(workspace: Path, output: Path, allow_test: bool, workers: int = 8) -> dict:
    if not allow_test:
        raise ValueError("This full audit requires explicit --allow-test-annotation-audit authorization.")
    workspace = workspace.resolve()
    output = output.resolve()
    if not output.is_relative_to(workspace / "annotations" / "working"):
        raise ValueError("Audit output must be a new directory under annotations/working.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "access_record.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User requested checking and correcting all train, test and validation labels.",
        "scope": "Annotation/file audit only; includes test labels and image bytes.",
        "model_predictions_accessed": False, "model_evaluation_run": False,
        "test_findings_used_for_model_tuning": False,
        "note": "Test annotation access is no longer untouched. Preserve subject split; do not tune models from this audit.",
    })
    source_path = workspace / "data/manifests/generated/casia_a_c_split.csv"
    label_path = workspace / "annotations/provisional/v0.1-ai/frame_labels.csv.gz"
    boundary_path = workspace / "annotations/provisional/v0.1-ai/cycle_boundaries.csv.gz"
    source, labels, boundaries = map(read_table, [source_path, label_path, boundary_path])
    if source.sample_id.duplicated().any() or labels.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs prevent a reliable audit join.")
    if not set(labels.sample_id).issubset(set(source.sample_id)):
        raise ValueError("Label IDs not present in source manifest")
    eligible = source.dataset.eq("casia_c") & source.exclusion_reason.eq("")
    if set(labels.sample_id) != set(source.loc[eligible, "sample_id"]):
        raise ValueError("Provisional coverage differs from eligible CASIA C source frames")
    joined = source.merge(labels, on="sample_id", how="left", suffixes=("", "_provisional"),
                          validate="one_to_one").fillna("")
    has_label = joined.phase_label.ne("")
    metadata_mismatches = {}
    for field in ["dataset", "subject_id", "sequence_id", "condition", "frame_index", "relative_path", "split", "fold"]:
        mismatch = has_label & joined[field].ne(joined[field + "_provisional"])
        metadata_mismatches[field] = int(mismatch.sum())
    comparison = compare_boundaries(labels, boundaries)
    mismatches = comparison[comparison.phase_label.ne(comparison.boundary_phase)].copy()
    mismatch_ids = set(mismatches.sample_id)
    compared_ids = set(comparison.sample_id)
    reviewed_path = workspace / "annotations/reviewed/pilot_v3_user_confirmed_v1/accepted_frame_labels.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))["records"]
    review_by_id = {row["sample_id"]: row for row in reviewed}
    if len(review_by_id) != len(reviewed):
        raise ValueError("Duplicate reviewed sample IDs")
    with gzip.open(output / "boundary_disagreements.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in mismatches.to_dict("records"):
            stream.write(json.dumps(row) + "\n")

    # Validate every fold against the canonical source labels, not only its own READY metadata.
    prepared_checks = {}
    for fold in range(5):
        prepared = read_table(workspace / f"data/processed/provisional_v0.1-ai/folds/fold_{fold}/manifest.csv.gz")
        merged = prepared.merge(labels[["sample_id", "phase_label", "confidence", "annotation_source", "annotation_version"]],
                                on="sample_id", how="left", validate="one_to_one")
        prepared_checks[str(fold)] = {
            "rows": int(len(merged)),
            "missing_canonical_label": int(merged.phase_label.isna().sum()),
            "label_mismatches": int(merged.provisional_label.ne(merged.phase_label).sum()),
            "confidence_mismatches": int(merged.provisional_confidence.ne(merged.confidence).sum()),
            "source_mismatches": int(merged.provisional_source.ne(merged.annotation_source).sum()),
            "version_mismatches": int(merged.provisional_version.ne(merged.annotation_version).sum()),
        }

    image_issue_counts, status_counts, sequence_records = {}, {}, {}
    phase_indices = {phase: str(index) for index, phase in enumerate(PHASES)}
    invalid_phase = has_label & ~joined.phase_label.isin(PHASES)
    phase_index_mismatch = has_label & joined.phase_index.ne(joined.phase_label.map(phase_indices).fillna(""))
    source_rows = joined.to_dict("records")
    def check(row):
        return image_check(workspace, row["relative_path"], row["sha256"])
    with gzip.open(output / "frame_audit.jsonl.gz", "wt", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (row, image_result) in enumerate(zip(source_rows, executor.map(check, source_rows)), 1):
                sample_id = row["sample_id"]
                review = review_by_id.get(sample_id)
                if row["exclusion_reason"]:
                    status = "source_excluded"
                elif row["dataset"] != "casia_c":
                    status = "incomplete_temporal_source_not_annotatable"
                elif image_result["image_issues"]:
                    status = "source_integrity_failure"
                elif review:
                    status = "chat_confirmed_ai_proposal_low_confidence"
                else:
                    status = "unverified_reference_foot_requires_sequence_review"
                record = {key: row[key] for key in ["sample_id", "dataset", "sequence_id", "subject_id", "frame_index", "split", "fold", "relative_path"]}
                record.update({
                    **image_result, "original_provisional_label": row["phase_label"] or None,
                    "review_status": status, "corrected_ground_truth_label": None,
                    "previous_chat_confirmed_proposal": review["label"] if review else None,
                    "primary_eligible": False, "training_eligible_as_ground_truth": False,
                    "saved_boundary_comparison_available": sample_id in compared_ids,
                    "saved_boundary_disagreement": sample_id in mismatch_ids,
                    "source_exclusion_reason": row["exclusion_reason"] or None,
                })
                stream.write(json.dumps(record) + "\n")
                status_counts[status] = status_counts.get(status, 0) + 1
                for issue in image_result["image_issues"]:
                    image_issue_counts[issue] = image_issue_counts.get(issue, 0) + 1
                seq = sequence_records.setdefault(row["sequence_id"], {
                    "sequence_id": row["sequence_id"], "dataset": row["dataset"], "split": row["split"],
                    "frames": 0, "image_issue_frames": 0, "boundary_disagreement_frames": 0,
                    "semantic_review_complete": False,
                })
                seq["frames"] += 1
                seq["image_issue_frames"] += bool(image_result["image_issues"])
                seq["boundary_disagreement_frames"] += sample_id in mismatch_ids
                if index % 10000 == 0:
                    print(f"Source files checked: {index}/{len(source_rows)}", flush=True)
    write_json(output / "sequence_review_queue.json", list(sequence_records.values()))
    summary = {
        "status": "AUTOMATED_AUDIT_COMPLETE_SEMANTIC_ANNOTATION_INCOMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frames": len(source), "provisional_frames": len(labels),
        "provisional_sequences": int(labels.sequence_id.nunique()),
        "source_images_hash_checked_and_decode_attempted": len(source_rows),
        "image_issue_counts": image_issue_counts, "review_status_counts": status_counts,
        "source_metadata_mismatches": metadata_mismatches,
        "invalid_phase_labels": int(invalid_phase.sum()), "phase_index_mismatches": int(phase_index_mismatch.sum()),
        "boundary_comparable_frames": len(comparison), "boundary_disagreement_frames": len(mismatches),
        "boundary_disagreements_by_split": mismatches.groupby("split").size().astype(int).to_dict(),
        "prepared_canonical_label_checks": prepared_checks,
        "provisional_split_counts": labels.groupby("split").size().astype(int).to_dict(),
        "primary_ground_truth_labels_created": 0, "existing_files_modified": False,
        "biomechanical_phase_accuracy": None,
        "source_sha256": {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source_path, label_path, boundary_path, reviewed_path]},
        "limitations": [
            "Byte integrity and folder consistency do not establish gait-phase correctness.",
            "Existing step-interval timing labels do not track the same reference foot required by the protocol.",
            "Boundary disagreements compare two machine-generated representations; neither is ground truth.",
            "No automatic relabeling or inferred human/expert approval is performed.",
            "Image decoding is not visual frame-by-frame annotation; the complete semantic review remains unfinished.",
        ],
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-test-annotation-audit", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(audit(Path(__file__).resolve().parents[1], args.output,
                           args.allow_test_annotation_audit, args.workers), indent=2))
