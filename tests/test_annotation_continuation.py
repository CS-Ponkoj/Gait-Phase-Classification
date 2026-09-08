import hashlib

import pandas as pd
import pytest

from scripts.prepare_annotation_continuation import choose_batch, finalize_review, source_path, validate_selected


def metadata():
    rows = []
    for condition in ["fn00", "fs00", "fq00", "fb00"]:
        for subject in range(5):
            person = f"casia_c:{condition}:{subject}"
            for frame in range(6):
                token = f"{person}:{frame}"
                rows.append({"sample_id": token, "dataset": "casia_c", "subject_id": person,
                             "sequence_id": person + ":sequence", "condition": condition,
                             "frame_index": str(frame), "relative_path": token + ".png",
                             "sha256": hashlib.sha256(token.encode()).hexdigest(),
                             "exclusion_reason": "", "split": "test" if subject == 0 else "development",
                             "fold": "-1" if subject == 0 else "0"})
    return pd.DataFrame(rows)


def test_batch_is_new_balanced_development_and_deterministic():
    meta = metadata()
    prior = {"casia_c:fn00:1"}
    selected, frames = choose_batch(meta, prior, set(), per_family=2)
    repeat, _ = choose_batch(meta, prior, set(), per_family=2)
    assert selected["sequence_id"].tolist() == repeat["sequence_id"].tolist()
    assert set(selected["subject_id"]).isdisjoint(prior)
    assert selected["subject_id"].nunique() == 8
    assert selected["condition_family"].value_counts().to_dict() == {"normal": 2, "slow": 2, "fast": 2, "bag": 2}
    assert frames["split"].eq("development").all()
    assert selected["blind_sequence_id"].is_unique


def test_duplicate_plus_gap_cannot_appear_contiguous():
    meta = metadata()
    selected, _ = choose_batch(meta, set(), set(), per_family=2)
    target = selected["sequence_id"].iloc[0]
    duplicate = meta[meta["sequence_id"].eq(target) & meta["frame_index"].eq("1")]
    bad = pd.concat([meta[~(meta["sequence_id"].eq(target) & meta["frame_index"].eq("2"))], duplicate])
    result, _ = choose_batch(bad, set(), set(), per_family=2)
    assert target not in set(result["sequence_id"])


def test_selected_validation_rejects_test_and_duplicate_content():
    meta = metadata()
    with pytest.raises(ValueError, match="development"):
        validate_selected(meta)
    development = meta[meta["split"].eq("development")].copy()
    validate_selected(development)
    development.iloc[1, development.columns.get_loc("sha256")] = development.iloc[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_selected(development)


def test_selected_validation_rejects_exclusion_and_invalid_fold():
    meta = metadata().query("split == 'development'").copy()
    meta.iloc[0, meta.columns.get_loc("exclusion_reason")] = "bad source"
    with pytest.raises(ValueError, match="eligible"):
        validate_selected(meta)
    meta["exclusion_reason"] = ""
    meta.iloc[0, meta.columns.get_loc("fold")] = "-1"
    with pytest.raises(ValueError, match="fold"):
        validate_selected(meta)


def test_paths_cannot_escape_workspace(tmp_path):
    work = tmp_path / "workspace"
    work.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="outside"):
        source_path(work, "../outside.png")


def test_existing_release_is_never_overwritten(tmp_path):
    output = tmp_path / "annotations/reviewed/pilot_v3_user_confirmed_v1"
    output.mkdir(parents=True)
    sentinel = output / "existing.json"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        finalize_review(tmp_path, pd.DataFrame())
    assert sentinel.read_text(encoding="utf-8") == "preserve"
