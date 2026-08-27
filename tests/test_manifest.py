import pandas as pd

from gait_phase.constants import MANIFEST_COLUMNS
from gait_phase.manifest import duplicate_audit, validate_manifest


def row(sample_id="one"):
    record = {column: "" for column in MANIFEST_COLUMNS}
    record.update(
        sample_id=sample_id,
        dataset="casia_c",
        subject_id="casia_c:001",
        sequence_id="casia_c:001:fn00",
        frame_index="1",
        relative_path="data/raw/casia_c/001/001/fn00/001.png",
        sha256="a" * 64,
        adjudicated_label="initial_contact_loading",
        split="development",
        fold="0",
    )
    return record


def test_valid_manifest_passes():
    assert validate_manifest(pd.DataFrame([row()]), frozen=True) == []


def test_conflicting_identical_labels_fail():
    second = row("two")
    second["adjudicated_label"] = "swing"
    errors = validate_manifest(pd.DataFrame([row(), second]))
    assert any("conflicting adjudicated labels" in error for error in errors)


def test_duplicate_audit_separates_exact_and_perceptual_candidates():
    first, second = row(), row("two")
    first["duplicate_group"] = second["duplicate_group"] = "exact-1"
    first["perceptual_hash"] = second["perceptual_hash"] = "abcd"
    second["subject_id"] = "casia_c:002"
    summary, exact, perceptual = duplicate_audit(pd.DataFrame([first, second]))
    assert summary["exact_cross_subject_groups"] == 1
    assert summary["perceptual_cross_subject_groups"] == 1
    assert len(exact) == len(perceptual) == 1
