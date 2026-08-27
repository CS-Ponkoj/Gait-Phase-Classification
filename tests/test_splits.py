import pandas as pd
import pytest

from gait_phase.splits import assert_split_integrity, make_subject_splits


def synthetic_manifest():
    rows = []
    for dataset in ("casia_a_curated", "casia_c"):
        for subject in range(10):
            for frame in range(2):
                rows.append(
                    {
                        "dataset": dataset,
                        "subject_id": f"{dataset}:{subject:03d}",
                        "sample_id": f"{dataset}-{subject}-{frame}",
                        "duplicate_group": "",
                    }
                )
    return pd.DataFrame(rows)


def test_subject_splits_are_deterministic_and_leak_free():
    left = make_subject_splits(synthetic_manifest())
    right = make_subject_splits(synthetic_manifest())
    assert left[["sample_id", "split", "fold"]].equals(right[["sample_id", "split", "fold"]])
    assert_split_integrity(left)
    counts = left.groupby(["dataset", "split"])["subject_id"].nunique().to_dict()
    assert counts[("casia_a_curated", "test")] == 2
    assert counts[("casia_c", "test")] == 2


def test_duplicate_crossing_partitions_is_rejected():
    frame = make_subject_splits(synthetic_manifest())
    test_index = frame.index[frame["split"] == "test"][0]
    development_index = frame.index[frame["split"] == "development"][0]
    frame.loc[[test_index, development_index], "duplicate_group"] = "exact-1"
    with pytest.raises(ValueError, match="duplicate leakage"):
        assert_split_integrity(frame)


def test_cross_subject_duplicates_are_excluded_before_split_validation():
    source = synthetic_manifest()
    source.loc[[0, 4], "duplicate_group"] = "exact-1"
    split = make_subject_splits(source)
    excluded = split[split["duplicate_group"] == "exact-1"]
    assert set(excluded["exclusion_reason"]) == {"exact_duplicate_shared_across_subjects"}
    assert_split_integrity(split)
