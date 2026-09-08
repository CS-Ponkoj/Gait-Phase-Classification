import hashlib

import pandas as pd
from PIL import Image
import pytest

from scripts.audit_annotation_release import audit, compare_boundaries, image_check
from scripts.repair_provisional_boundary_export import repair


def test_audit_requires_test_authorization_before_output(tmp_path):
    output = tmp_path / "annotations/working/audit"
    with pytest.raises(ValueError, match="authorization"):
        audit(tmp_path, output, False)
    assert not output.exists()


def test_audit_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "annotations/working/audit"
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        audit(tmp_path, output, True)


def test_image_check_detects_hash_error_without_claiming_phase(tmp_path):
    image = Image.new("L", (8, 8), 255)
    path = tmp_path / "frame.png"
    image.save(path)
    result = image_check(tmp_path, "frame.png", "bad_hash")
    assert result["image_issues"] == ["source_checksum_mismatch"]
    assert result["foreground_touches_image_edge"] is True
    assert "phase_label" not in result
    valid = image_check(tmp_path, "frame.png", hashlib.sha256(path.read_bytes()).hexdigest())
    assert valid["image_issues"] == []


def test_image_check_rejects_escape_missing_and_empty(tmp_path):
    assert image_check(tmp_path, "../outside.png", "")["image_issues"] == ["path_outside_workspace"]
    assert image_check(tmp_path, "missing.png", "")["image_issues"] == ["missing_or_unreadable_image"]
    path = tmp_path / "empty.png"
    Image.new("L", (8, 8)).save(path)
    assert image_check(tmp_path, path.name, hashlib.sha256(path.read_bytes()).hexdigest())["image_issues"] == ["empty_foreground"]


def test_boundary_comparison_is_half_open_and_does_not_relabel():
    labels = pd.DataFrame([{"sequence_id": "s", "frame_index": str(i), "phase_label": "swing"}
                           for i in range(5)])
    boundaries = pd.DataFrame([{"sequence_id": "s", "cycle_id": "c", "contact_start_frame": 0,
                               "mid_stance_frame": 1, "terminal_stance_frame": 2,
                               "swing_frame": 3, "next_contact_frame": 4}])
    result = compare_boundaries(labels, boundaries)
    assert result.frame_index.tolist() == ["0", "1", "2", "3"]
    assert result.phase_label.ne(result.boundary_phase).sum() == 3
    assert labels.phase_label.eq("swing").all()
    with pytest.raises(ValueError, match="Overlapping"):
        compare_boundaries(labels, pd.concat([boundaries, boundaries]))


def test_repair_changes_only_separate_boundary_export(tmp_path):
    import json
    from gait_phase.auto_annotations import label_from_progress

    source = tmp_path / "annotations/provisional/v0.1-ai"
    source.mkdir(parents=True)
    output = tmp_path / "annotations/working/repair.json"
    output.parent.mkdir(parents=True)
    labels = pd.DataFrame([{"sequence_id": "s", "frame_index": str(i),
                            "phase_label": label_from_progress(i / 12)[0]} for i in range(12)])
    boundaries = pd.DataFrame([{"sequence_id": "s", "cycle_id": "c", "contact_start_frame": 0,
                               "mid_stance_frame": 1, "terminal_stance_frame": 4,
                               "swing_frame": 7, "next_contact_frame": 12}])
    labels_path = source / "frame_labels.csv.gz"
    boundaries_path = source / "cycle_boundaries.csv.gz"
    labels.to_csv(labels_path, index=False)
    boundaries.to_csv(boundaries_path, index=False)
    original_bytes = {p: p.read_bytes() for p in [labels_path, boundaries_path]}
    with pytest.raises(ValueError, match="Explicit"):
        repair(tmp_path, output, False)
    repair(tmp_path, output, True)
    saved = json.loads(output.read_text())
    assert saved["frame_disagreements_before"] == 2
    assert saved["frame_disagreements_after"] == 0
    assert saved["primary_eligible"] is False
    assert saved["frame_phase_labels_changed"] == 0
    assert all(p.read_bytes() == data for p, data in original_bytes.items())
    with pytest.raises(FileExistsError):
        repair(tmp_path, output, True)
