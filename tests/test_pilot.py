from pathlib import Path

import pandas as pd
from PIL import Image

from gait_phase.pilot import build_pilot_package, select_pilot_sequences


def pilot_manifest(workspace: Path) -> pd.DataFrame:
    rows = []
    families = {
        "casia_a": ("casia_a_curated", "unspecified"),
        "normal": ("casia_c", "fn00"),
        "slow": ("casia_c", "fs00"),
        "fast": ("casia_c", "fq00"),
        "bag": ("casia_c", "fb00"),
    }
    for family_index, (_, (dataset, condition)) in enumerate(families.items()):
        for subject_index in range(5):
            subject = f"{dataset}:{family_index:02d}{subject_index:02d}"
            sequence = f"{subject}:{condition}"
            for frame_index in range(4 + subject_index):
                relative = Path("images") / f"{family_index}-{subject_index}" / f"{frame_index:03d}.png"
                path = workspace / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("L", (12, 12), family_index * 30 + subject_index).save(path)
                rows.append(
                    {
                        "dataset": dataset,
                        "subject_id": subject,
                        "sequence_id": sequence,
                        "condition": condition,
                        "split": "development",
                        "relative_path": relative.as_posix(),
                        "frame_index": str(frame_index),
                        "exclusion_reason": "",
                    }
                )
    return pd.DataFrame(rows)


def test_pilot_selection_is_balanced_distinct_and_deterministic(tmp_path):
    manifest = pilot_manifest(tmp_path)
    left = select_pilot_sequences(manifest, seed=7, casia_c_sequences_per_family=3)
    right = select_pilot_sequences(manifest, seed=7, casia_c_sequences_per_family=3)
    assert left["sequence_id"].tolist() == right["sequence_id"].tolist()
    assert left["subject_id"].nunique() == 12
    assert left["condition_family"].value_counts().to_dict() == {
        "normal": 3,
        "slow": 3,
        "fast": 3,
        "bag": 3,
    }
    assert set(left["dataset"]) == {"casia_c"}


def test_pilot_selection_excludes_incomplete_sequences(tmp_path):
    manifest = pilot_manifest(tmp_path)
    target = manifest[manifest["dataset"].eq("casia_c")]["sequence_id"].iloc[0]
    gap_row = manifest[manifest["sequence_id"].eq(target)].sort_values("frame_index").index[1]
    manifest = manifest.drop(gap_row)
    selected = select_pilot_sequences(manifest, seed=7, casia_c_sequences_per_family=3)
    assert target not in set(selected["sequence_id"])


def test_pilot_selection_honors_visual_qc_exclusions(tmp_path):
    manifest = pilot_manifest(tmp_path)
    target = manifest[manifest["dataset"].eq("casia_c")]["sequence_id"].iloc[0]
    selected = select_pilot_sequences(
        manifest,
        seed=7,
        casia_c_sequences_per_family=3,
        excluded_sequence_ids={target},
    )
    assert target not in set(selected["sequence_id"])


def test_pilot_package_is_blinded_and_contains_previews(tmp_path):
    manifest = pilot_manifest(tmp_path)
    protocol = tmp_path / "docs" / "annotation_protocol.md"
    protocol.parent.mkdir()
    protocol.write_text("# Test annotation protocol\n", encoding="utf-8")
    output = tmp_path / "pilot"
    summary = build_pilot_package(
        manifest,
        workspace=tmp_path,
        output=output,
        settings={"version": "pilot_v3", "casia_c_sequences_per_family": 3, "gif_duration_ms": 50},
        seed=7,
    )
    assert summary["sequences"] == 12
    assert summary["subjects"] == 12
    assert len(list((output / "previews").glob("*.gif"))) == 12
    annotator = pd.read_csv(output / "annotator_1_boundaries.csv", keep_default_na=False)
    assert set(annotator["dataset"]) == {"blinded"}
    assert not annotator["sequence_id"].str.contains("casia", case=False).any()
    assert (output / "coordinator_key.csv").is_file()
    for annotator_id in ("annotator_1", "annotator_2"):
        archive = output / f"{annotator_id}_package.zip"
        assert archive.is_file()
        import zipfile

        with zipfile.ZipFile(archive) as package:
            names = set(package.namelist())
        assert "boundaries.csv" in names
        assert "annotation_protocol.md" in names
        assert "coordinator_key.csv" not in names
        assert "annotator_1_boundaries.csv" not in names
        assert "annotator_2_boundaries.csv" not in names
