import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from gait_phase.constants import PHASES
from gait_phase.experiment_summary import aggregate_registered_runs


def _write_run(workspace: Path, run_id: str, fold: int, predictions: pd.DataFrame) -> None:
    run = workspace / "artifacts" / "runs" / run_id
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "training_run": {
                    "fold": fold,
                    "evaluation_split": "validation",
                    "test_accessed": False,
                }
            }
        ),
        encoding="utf-8",
    )
    predictions.to_csv(run / "predictions.csv", index=False)


def _registry_fixture(workspace: Path) -> Path:
    variants = {name: {"role": "test", "runs": {}} for name in ("left", "right")}
    canonical_rows = []
    for fold in range(5):
        rows = []
        for subject_number in range(2):
            subject = f"subject-{fold}-{subject_number}"
            for frame_index, phase in enumerate(PHASES):
                rows.append(
                    {
                        "sample_id": f"sample-{fold}-{subject_number}-{frame_index}",
                        "dataset": "casia_c",
                        "subject_id": subject,
                        "sequence_id": f"{subject}:sequence",
                        "condition": "fn00",
                        "frame_index": str(frame_index),
                        "y_true": phase,
                        "y_pred": phase,
                        "label_source": "fixture",
                        "annotation_version": "fixture-v1",
                        "annotation_confidence": "high",
                    }
                )
        frame = pd.DataFrame(rows)
        canonical = frame.rename(
            columns={
                "y_true": "provisional_label",
                "label_source": "provisional_source",
                "annotation_version": "provisional_version",
                "annotation_confidence": "provisional_confidence",
            }
        ).copy()
        canonical["split"] = "development"
        canonical["fold"] = str(fold)
        canonical_rows.extend(canonical.to_dict("records"))
        for name in variants:
            run_id = f"{name}-fold-{fold}"
            variants[name]["runs"][str(fold)] = run_id
            _write_run(workspace, run_id, fold, frame)
    pd.DataFrame(canonical_rows).to_csv(workspace / "manifest.csv", index=False)
    registry = workspace / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": "fixture-v1",
                "expected": {
                    "folds": list(range(5)),
                    "samples": 40,
                    "subjects": 10,
                    "manifest": "manifest.csv",
                },
                "variants": variants,
                "comparisons": [{"left": "left", "right": "right"}],
            }
        ),
        encoding="utf-8",
    )
    return registry


def test_aggregate_registered_runs_validates_and_writes_evidence(tmp_path):
    registry = _registry_fixture(tmp_path)
    output = tmp_path / "paper" / "development-v1"

    result = aggregate_registered_runs(registry, tmp_path, output, bootstrap_iterations=100)

    assert result["status"] == "READY"
    assert result["test_predictions_included"] is False
    assert pd.read_csv(output / "model_summary.csv").shape[0] == 2
    comparison = pd.read_csv(output / "paired_comparisons.csv").iloc[0]
    assert comparison["estimate"] == 0.0
    assert (output / "predictions" / "left.csv.gz").is_file()
    assert json.loads((output / "READY.json").read_text())["samples_per_variant"] == 40


def test_aggregate_registered_runs_refuses_mismatched_samples_atomically(tmp_path):
    registry = _registry_fixture(tmp_path)
    bad = tmp_path / "artifacts" / "runs" / "right-fold-4" / "predictions.csv"
    pd.read_csv(bad).iloc[:-1].to_csv(bad, index=False)
    output = tmp_path / "paper" / "development-v1"

    with pytest.raises(ValueError, match="samples"):
        aggregate_registered_runs(registry, tmp_path, output, bootstrap_iterations=100)

    assert not output.exists()
