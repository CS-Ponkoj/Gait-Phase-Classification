"""Validate and aggregate registered cross-validation prediction artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .metrics import compute_metrics, paired_subject_bootstrap_difference, subject_macro_f1_interval
from .paper_assets import make_paper_assets


IDENTITY_COLUMNS = [
    "sample_id",
    "dataset",
    "subject_id",
    "sequence_id",
    "condition",
    "frame_index",
    "y_true",
    "label_source",
    "annotation_version",
    "annotation_confidence",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(workspace: Path, run_id: str, expected_fold: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dir = workspace / "artifacts" / "runs" / run_id
    config_path = run_dir / "config.yaml"
    predictions_path = run_dir / "predictions.csv"
    if not config_path.is_file() or not predictions_path.is_file():
        raise FileNotFoundError(f"Incomplete registered run: {run_id}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run = config.get("training_run", {})
    if int(run.get("fold", -1)) != expected_fold:
        raise ValueError(f"Run {run_id} is registered for fold {expected_fold}, but its config differs.")
    if run.get("evaluation_split") != "validation" or bool(run.get("test_accessed")):
        raise ValueError(f"Run {run_id} is not a development-validation run.")
    predictions = pd.read_csv(predictions_path, dtype={"frame_index": str})
    missing = set(IDENTITY_COLUMNS + ["y_pred"]) - set(predictions.columns)
    if missing:
        raise ValueError(f"Run {run_id} is missing prediction columns: {sorted(missing)}")
    if predictions["sample_id"].duplicated().any():
        raise ValueError(f"Run {run_id} contains duplicate sample IDs.")
    return predictions, config


def _best_epoch(run_dir: Path) -> int | None:
    curves_path = run_dir / "learning_curves.csv"
    if not curves_path.is_file():
        return None
    curves = pd.read_csv(curves_path)
    if "validation_loss" not in curves or curves["validation_loss"].isna().all():
        return None
    return int(curves.loc[curves["validation_loss"].idxmin(), "epoch"])


def aggregate_registered_runs(
    registry_path: str | Path,
    workspace: str | Path,
    output: str | Path,
    bootstrap_iterations: int = 2000,
) -> dict[str, Any]:
    """Aggregate exact registered runs and refuse incomplete or incomparable evidence."""
    workspace = Path(workspace).resolve()
    registry_path = Path(registry_path).resolve()
    output = Path(output).resolve()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    variants = registry.get("variants", {})
    if not variants:
        raise ValueError("Experiment registry contains no variants.")
    expected_folds = [int(value) for value in registry["expected"]["folds"]]
    expected_samples = int(registry["expected"]["samples"])
    expected_subjects = int(registry["expected"]["subjects"])
    canonical_path = workspace / str(registry["expected"]["manifest"])
    canonical = pd.read_csv(canonical_path, dtype=str, keep_default_na=False)
    canonical = canonical[canonical["split"].eq("development")].copy()
    canonical_identity = canonical[
        ["sample_id", "dataset", "subject_id", "sequence_id", "condition", "frame_index",
         "provisional_label", "provisional_source", "provisional_version", "provisional_confidence", "fold"]
    ].rename(
        columns={
            "provisional_label": "y_true",
            "provisional_source": "label_source",
            "provisional_version": "annotation_version",
            "provisional_confidence": "annotation_confidence",
            "fold": "canonical_fold",
        }
    )
    if canonical_identity["sample_id"].duplicated().any():
        raise ValueError("Canonical development manifest contains duplicate sample IDs.")
    if len(canonical_identity) != expected_samples or canonical_identity["subject_id"].nunique() != expected_subjects:
        raise ValueError("Canonical development manifest does not match registered sample and subject counts.")
    if bootstrap_iterations < 100:
        raise ValueError("Use at least 100 bootstrap iterations.")
    if output.exists():
        raise FileExistsError(f"Aggregation output already exists: {output}")

    combined_by_variant: dict[str, pd.DataFrame] = {}
    model_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    duration_rows: list[dict[str, Any]] = []
    reference_identity: pd.DataFrame | None = None

    for variant, specification in variants.items():
        runs = specification.get("runs", {})
        if sorted(int(fold) for fold in runs) != expected_folds:
            raise ValueError(f"Variant {variant} does not register exactly folds {expected_folds}.")
        pieces = []
        for fold in expected_folds:
            run_id = str(runs[str(fold)])
            predictions, config = _load_run(workspace, run_id, fold)
            predictions = predictions.copy()
            predictions["development_fold"] = fold
            pieces.append(predictions)
            metrics = compute_metrics(predictions)
            fold_rows.append(
                {
                    "variant": variant,
                    "role": specification.get("role", "unspecified"),
                    "fold": fold,
                    "run_id": run_id,
                    "samples": metrics["samples"],
                    "subjects": metrics["subjects"],
                    "subject_macro_f1": metrics["subject_macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                }
            )
            best_epoch = _best_epoch(workspace / "artifacts" / "runs" / run_id)
            if best_epoch is not None:
                duration_rows.append({"variant": variant, "fold": fold, "run_id": run_id, "best_epoch": best_epoch})
        combined = pd.concat(pieces, ignore_index=True)
        if combined["sample_id"].duplicated().any():
            raise ValueError(f"Variant {variant} contains samples in more than one fold.")
        if len(combined) != expected_samples or combined["subject_id"].nunique() != expected_subjects:
            raise ValueError(
                f"Variant {variant} has {len(combined)} samples and {combined['subject_id'].nunique()} subjects; "
                f"expected {expected_samples} and {expected_subjects}."
            )
        canonical_check = combined[IDENTITY_COLUMNS + ["development_fold"]].merge(
            canonical_identity,
            on="sample_id",
            how="outer",
            suffixes=("", "_canonical"),
            validate="one_to_one",
            indicator=True,
        )
        if not canonical_check["_merge"].eq("both").all():
            raise ValueError(f"Variant {variant} does not exactly cover the canonical development samples.")
        for column in IDENTITY_COLUMNS[1:]:
            if not canonical_check[column].astype(str).eq(canonical_check[f"{column}_canonical"].astype(str)).all():
                raise ValueError(f"Variant {variant} differs from canonical manifest field {column}.")
        if not canonical_check["development_fold"].astype(str).eq(canonical_check["canonical_fold"].astype(str)).all():
            raise ValueError(f"Variant {variant} assigns one or more predictions to the wrong fold.")
        identity = combined[IDENTITY_COLUMNS].sort_values("sample_id").reset_index(drop=True)
        if reference_identity is None:
            reference_identity = identity
        elif not identity.equals(reference_identity):
            raise ValueError(f"Variant {variant} does not contain the same sample identities and labels.")
        combined_by_variant[variant] = combined
        metrics = compute_metrics(combined)
        interval = subject_macro_f1_interval(combined, iterations=bootstrap_iterations)
        model_rows.append(
            {
                "variant": variant,
                "role": specification.get("role", "unspecified"),
                "samples": metrics["samples"],
                "subjects": metrics["subjects"],
                "subject_macro_f1": metrics["subject_macro_f1"],
                "ci_lower": interval["lower"],
                "ci_upper": interval["upper"],
                "frame_macro_f1": metrics["frame_macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "transition_validity": metrics.get("transition_validity"),
            }
        )

    comparison_rows = []
    for comparison in registry.get("comparisons", []):
        left, right = comparison["left"], comparison["right"]
        if left not in combined_by_variant or right not in combined_by_variant:
            raise ValueError(f"Unknown comparison: {left} versus {right}")
        result = paired_subject_bootstrap_difference(
            combined_by_variant[left], combined_by_variant[right], iterations=bootstrap_iterations
        )
        comparison_rows.append({"left": left, "right": right, **result})

    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        predictions_dir = temporary / "predictions"
        predictions_dir.mkdir()
        for variant, predictions in combined_by_variant.items():
            predictions.to_csv(predictions_dir / f"{variant}.csv.gz", index=False)
            make_paper_assets(predictions, temporary / "paper_assets" / variant, bootstrap_iterations)
        pd.DataFrame(model_rows).to_csv(temporary / "model_summary.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(temporary / "fold_summary.csv", index=False)
        pd.DataFrame(comparison_rows).to_csv(temporary / "paired_comparisons.csv", index=False)
        pd.DataFrame(duration_rows).to_csv(temporary / "training_duration.csv", index=False)
        shutil.copyfile(registry_path, temporary / "registry.yaml")
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        summary = {
            "status": "READY",
            "registry_version": registry.get("version"),
            "variants": len(combined_by_variant),
            "comparisons": len(comparison_rows),
            "samples_per_variant": expected_samples,
            "subjects_per_variant": expected_subjects,
            "bootstrap_iterations": bootstrap_iterations,
            "test_predictions_included": False,
            "source_checksums": {
                "registry": _sha256(registry_path),
                "canonical_manifest": _sha256(canonical_path),
            },
            "checksums": {path.relative_to(temporary).as_posix(): _sha256(path) for path in files},
        }
        (temporary / "READY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary
