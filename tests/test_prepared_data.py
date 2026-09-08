from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

from gait_phase.cli import main
from gait_phase.constants import MANIFEST_COLUMNS, PHASES, PHASE_TO_INDEX
from gait_phase.hashing import sha256_file
from gait_phase.prepared_data import prepare_training_data, validate_prepared_data
from gait_phase.training import DenseTemporalDataset, TemporalDataset, prepare_aligned_silhouette


def prepared_fixture(workspace: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows, label_rows = [], []
    for subject in range(7):
        split = "test" if subject == 6 else "development"
        fold = "-1" if split == "test" else str(subject % 5)
        for frame_index, label in enumerate(PHASES):
            relative = Path("images") / f"subject-{subject}" / f"frame-{frame_index}.png"
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (12, 12), color=subject * 20 + frame_index + 1).save(path)
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update(
                sample_id=f"sample-{subject}-{frame_index}",
                dataset="casia_c",
                subject_id=f"casia_c:{subject:03d}",
                sequence_id=f"casia_c:{subject:03d}:fn00",
                condition="fn00",
                view="side_thermal",
                frame_index=str(frame_index),
                cycle_position=str(frame_index / 4),
                relative_path=relative.as_posix(),
                sha256=sha256_file(path),
                split=split,
                fold=fold,
            )
            manifest_rows.append(row)
            label_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "sequence_id": row["sequence_id"],
                    "condition": row["condition"],
                    "frame_index": row["frame_index"],
                    "relative_path": row["relative_path"],
                    "split": split,
                    "fold": fold,
                    "phase_label": label,
                    "cycle_position": row["cycle_position"],
                    "confidence": "low" if subject == 5 else "medium",
                    "annotation_source": "ai_provisional_silhouette_timing",
                    "annotation_version": "fixture-v1",
                }
            )
    return pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS), pd.DataFrame(label_rows)


def settings() -> dict[str, object]:
    return {
        "confidence": "medium",
        "folds": 5,
        "copy_workers": 2,
        "source_manifest_sha256": "manifest-fixture",
        "source_labels_sha256": "labels-fixture",
    }


def test_prepare_training_data_copies_fold_views_and_withholds_low_confidence(tmp_path):
    manifest, labels = prepared_fixture(tmp_path)
    output = tmp_path / "data" / "processed" / "prepared"
    ready = prepare_training_data(manifest, labels, tmp_path, output, settings())
    assert ready["status"] == "READY"
    assert ready["development_frames"] == 20
    assert ready["test_frames"] == 4
    assert ready["withheld_low_confidence_frames"] == 4
    assert ready["copied_files"] == 104
    assert (output / "READY.json").is_file()
    report = validate_prepared_data(output, tmp_path)
    assert report["status"] == "PASS"
    assert report["copied_files"] == 104
    withheld = pd.read_csv(output / "excluded_low_confidence.csv.gz")
    assert len(withheld) == 4
    for fold in range(5):
        prepared = pd.read_csv(output / "folds" / f"fold_{fold}" / "manifest.csv.gz", dtype=str)
        assert len(prepared) == 24
        assert set(prepared["provisional_confidence"]) == {"medium"}
        assert not set(prepared.loc[prepared["prepared_role"] == "train", "subject_id"]) & set(
            prepared.loc[prepared["prepared_role"] == "validation", "subject_id"]
        )
        for row in prepared.itertuples():
            copied = tmp_path / row.relative_path
            assert copied.is_file()
            assert row.provisional_label in copied.parts
            assert sha256_file(copied) == row.sha256


def test_prepare_training_data_refuses_nonempty_output(tmp_path):
    manifest, labels = prepared_fixture(tmp_path)
    output = tmp_path / "data" / "processed" / "prepared"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_training_data(manifest, labels, tmp_path, output, settings())


def test_validate_prepared_data_detects_copy_corruption(tmp_path):
    manifest, labels = prepared_fixture(tmp_path)
    output = tmp_path / "data" / "processed" / "prepared"
    prepare_training_data(manifest, labels, tmp_path, output, settings())
    prepared = pd.read_csv(output / "folds" / "fold_0" / "manifest.csv.gz", dtype=str)
    copied = tmp_path / prepared.iloc[0]["relative_path"]
    copied.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_prepared_data(output, tmp_path)


@pytest.mark.parametrize("fault", ["duplicate", "split", "label", "checksum", "missing"])
def test_prepare_training_data_rejects_invalid_inputs_and_cleans_staging(tmp_path, fault):
    manifest, labels = prepared_fixture(tmp_path)
    if fault == "duplicate":
        labels = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    elif fault == "split":
        labels.loc[0, "split"] = "test"
    elif fault == "label":
        labels.loc[0, "phase_label"] = "unknown"
    elif fault == "checksum":
        manifest.loc[0, "sha256"] = "0" * 64
    else:
        (tmp_path / manifest.loc[0, "relative_path"]).unlink()
    output = tmp_path / "data" / "processed" / "prepared"
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare_training_data(manifest, labels, tmp_path, output, settings())
    assert not output.exists()
    assert not list(output.parent.glob(".prepared-building-*"))


def test_provisional_training_requires_both_safety_flags(tmp_path):
    manifest, labels = prepared_fixture(tmp_path)
    output = tmp_path / "data" / "processed" / "prepared"
    prepare_training_data(manifest, labels, tmp_path, output, settings())
    fold_manifest = output / "folds" / "fold_0" / "manifest.csv.gz"
    base = [
        "train",
        "--manifest",
        str(fold_manifest),
        "--workspace",
        str(tmp_path),
        "--model",
        "majority",
        "--fold",
        "0",
        "--label-source",
        "provisional",
    ]
    assert main(base) == 2
    assert main([*base, "--allow-provisional"]) == 0
    assert main([*base, "--allow-provisional", "--evaluation-split", "test"]) == 2


def test_training_overrides_are_recorded_in_run_config(tmp_path):
    manifest, labels = prepared_fixture(tmp_path)
    output = tmp_path / "data" / "processed" / "prepared"
    prepare_training_data(manifest, labels, tmp_path, output, settings())
    result = main(
        [
            "train",
            "--manifest",
            str(output / "folds" / "fold_0" / "manifest.csv.gz"),
            "--workspace",
            str(tmp_path),
            "--model",
            "majority",
            "--fold",
            "0",
            "--label-source",
            "provisional",
            "--allow-provisional",
            "--epochs",
            "3",
            "--batch-size",
            "7",
            "--learning-rate",
            "0.005",
            "--patience",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--progress-every",
            "25",
        ]
    )
    assert result == 0
    run_directory = next((tmp_path / "artifacts" / "runs").iterdir())
    run_config = yaml.safe_load((run_directory / "config.yaml").read_text(encoding="utf-8"))
    assert run_config["training"]["epochs"] == 3
    assert run_config["training"]["batch_size"] == 7
    assert run_config["training"]["learning_rate"] == 0.005
    assert run_config["training"]["progress_every_batches"] == 25
    assert run_config["training_run"]["model"] == "majority"


def test_temporal_dataset_orders_frames_numerically_and_splits_gaps(tmp_path):
    rows = []
    for frame_index in (10, 2, 11, 1):
        relative = Path("images") / f"{frame_index}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        Image.new("L", (8, 8), frame_index).save(path)
        rows.append(
            {
                "sequence_id": "sequence-1",
                "frame_index": str(frame_index),
                "relative_path": relative.as_posix(),
                "target_label": PHASES[0],
            }
        )
    dataset = TemporalDataset(pd.DataFrame(rows), tmp_path, (8, 8), window=3)
    assert dataset.frame["numeric_frame_index"].tolist() == [1, 2, 10, 11]
    assert sorted(len(indices) for indices in dataset.segment_indices.values()) == [2, 2]


def test_aligned_silhouette_preserves_shape_and_bottom_alignment():
    image = Image.new("L", (80, 60), 0)
    for x in range(30, 50):
        for y in range(10, 50):
            image.putpixel((x, y), 255)
    aligned = prepare_aligned_silhouette(image, (44, 64), margin=4)
    assert aligned.size == (44, 64)
    assert aligned.getbbox() is not None
    assert aligned.getbbox()[3] == 60
    assert aligned.getbbox()[2] - aligned.getbbox()[0] < aligned.getbbox()[3] - aligned.getbbox()[1]


def test_temporal_repeated_control_uses_only_center_frame(tmp_path):
    rows = []
    for frame_index in range(5):
        relative = Path("images") / f"control-{frame_index}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        Image.new("L", (16, 16), 20 * frame_index).save(path)
        rows.append(
            {
                "sequence_id": "sequence-control",
                "frame_index": str(frame_index),
                "relative_path": relative.as_posix(),
                "target_label": PHASES[frame_index % len(PHASES)],
            }
        )
    dataset = TemporalDataset(
        pd.DataFrame(rows),
        tmp_path,
        (16, 16),
        window=5,
        temporal_control="repeated",
    )
    images, _, _ = dataset[2]
    assert all(images[position].equal(images[0]) for position in range(1, 5))


def test_temporal_shuffled_control_keeps_target_centered_and_reorders_context(tmp_path):
    rows = []
    for frame_index in range(5):
        relative = Path("images") / f"shuffled-{frame_index}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        Image.new("L", (16, 16), 20 * frame_index).save(path)
        rows.append(
            {
                "sequence_id": "sequence-shuffled",
                "frame_index": str(frame_index),
                "relative_path": relative.as_posix(),
                "target_label": PHASES[frame_index % len(PHASES)],
            }
        )
    dataset = TemporalDataset(
        pd.DataFrame(rows),
        tmp_path,
        (16, 16),
        window=5,
        temporal_control="shuffled",
        seed=7,
    )

    images, target, _ = dataset[2]
    intensities = [round(float(image.mean()) * 255) for image in images]

    assert intensities[2] == 40
    assert sorted(intensities[:2] + intensities[3:]) == [0, 20, 60, 80]
    assert intensities != [0, 20, 40, 60, 80]
    assert target == PHASE_TO_INDEX[PHASES[2]]


def test_dense_temporal_dataset_covers_every_frame_and_marks_boundaries(tmp_path):
    rows = []
    for frame_index in range(7):
        relative = Path("images") / f"dense-{frame_index}.png"
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        image = Image.new("L", (32, 32), 0)
        for x in range(12, 20):
            for y in range(4, 28):
                image.putpixel((x, y), 255)
        image.save(path)
        rows.append(
            {
                "sequence_id": "sequence-dense",
                "frame_index": str(frame_index),
                "relative_path": relative.as_posix(),
                "target_label": PHASES[0] if frame_index < 3 else PHASES[1],
            }
        )
    dataset = DenseTemporalDataset(
        pd.DataFrame(rows),
        tmp_path,
        (24, 32),
        window=5,
        stride=3,
    )
    covered = set()
    observed_boundaries = 0
    for clip_index in range(len(dataset)):
        images, targets, boundaries, mask, indices = dataset[clip_index]
        assert images.shape == (5, 1, 32, 24)
        assert targets.shape == boundaries.shape == mask.shape == indices.shape == (5,)
        covered.update(indices[mask].tolist())
        observed_boundaries += int(boundaries[mask].sum())
    assert covered == set(range(7))
    assert observed_boundaries >= 1
