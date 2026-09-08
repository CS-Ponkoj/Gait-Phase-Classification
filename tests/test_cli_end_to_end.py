from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

from gait_phase.cli import main
from gait_phase.constants import MANIFEST_COLUMNS, PHASES
from gait_phase.hashing import sha256_file


def create_frozen_fixture(workspace: Path) -> Path:
    rows = []
    for subject in range(6):
        for frame_index, label in enumerate(PHASES):
            relative = Path("images") / f"subject-{subject}" / f"frame-{frame_index}.png"
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (16, 16), color=subject * 20 + frame_index).save(path)
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update(
                sample_id=f"sample-{subject}-{frame_index}",
                dataset="casia_c",
                subject_id=f"casia_c:{subject:03d}",
                sequence_id=f"casia_c:{subject:03d}:fn00",
                condition="fn00",
                view="side_thermal",
                frame_index=str(frame_index),
                relative_path=relative.as_posix(),
                sha256=sha256_file(path),
                adjudicated_label=label,
                annotation_confidence="high",
                split="development",
                fold=str(subject % 5),
            )
            rows.append(row)
    manifest = workspace / "frozen.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    return manifest


def test_majority_run_and_paper_assets_are_reproducible(tmp_path):
    manifest = create_frozen_fixture(tmp_path)
    assert main(["validate-data", str(manifest), "--frozen"]) == 0
    assert main(
        [
            "train",
            "--manifest",
            str(manifest),
            "--workspace",
            str(tmp_path),
            "--model",
            "majority",
            "--fold",
            "0",
        ]
    ) == 0
    predictions = next((tmp_path / "artifacts" / "runs").glob("*/predictions.csv"))
    output = tmp_path / "paper"
    assert main(
        [
            "make-paper-assets",
            str(predictions),
            "--output",
            str(output),
            "--bootstrap-iterations",
            "100",
        ]
    ) == 0
    assert (output / "metrics.json").is_file()
    assert (output / "confusion_matrix.png").is_file()
    assert (output / "results_summary.md").is_file()


def test_tgpn_one_epoch_smoke_run_produces_dense_predictions(tmp_path):
    manifest = create_frozen_fixture(tmp_path)
    assert main(
        [
            "train",
            "--manifest",
            str(manifest),
            "--workspace",
            str(tmp_path),
            "--model",
            "tgpn",
            "--fold",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--temporal-window",
            "9",
            "--learning-rate",
            "0.0003",
            "--patience",
            "1",
            "--device",
            "cpu",
            "--no-augmentation",
        ]
    ) == 0
    run_directory = next((tmp_path / "artifacts" / "runs").glob("*-tgpn-*"))
    predictions = pd.read_csv(run_directory / "predictions.csv")
    assert len(predictions) == 2 * len(PHASES)
    assert {"boundary_probability", "y_boundary"}.issubset(predictions.columns)
    assert (run_directory / "model.pt").is_file()


def test_final_neural_evaluation_requires_and_records_fixed_epoch_training(tmp_path):
    manifest = create_frozen_fixture(tmp_path)
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    test_subject = "casia_c:005"
    frame.loc[frame.subject_id.eq(test_subject), "split"] = "test"
    frame.loc[frame.subject_id.eq(test_subject), "fold"] = "-1"
    frame.to_csv(manifest, index=False)
    common = [
        "train",
        "--manifest",
        str(manifest),
        "--workspace",
        str(tmp_path),
        "--model",
        "tgpn",
        "--evaluation-split",
        "test",
        "--allow-test",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--temporal-window",
        "9",
        "--learning-rate",
        "0.0003",
        "--device",
        "cpu",
        "--no-augmentation",
    ]

    assert main(common) == 2
    assert not (tmp_path / "artifacts" / "runs").exists()
    assert main(common + ["--fixed-training-epochs"]) == 0

    run_directory = next((tmp_path / "artifacts" / "runs").glob("*-tgpn-*"))
    config = yaml.safe_load((run_directory / "config.yaml").read_text())
    curves = pd.read_csv(run_directory / "learning_curves.csv")
    predictions = pd.read_csv(run_directory / "predictions.csv")
    assert config["training_run"]["checkpoint_selection"] == "fixed_epochs_no_evaluation_selection"
    assert config["training_run"]["fixed_training_epochs"] is True
    assert "validation_loss" not in curves.columns
    assert set(predictions.subject_id) == {test_subject}
