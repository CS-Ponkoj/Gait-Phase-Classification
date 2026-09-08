import pandas as pd
import pytest

from gait_phase.metrics import (
    boundary_metrics,
    compute_metrics,
    paired_subject_bootstrap_difference,
    subject_bootstrap_interval,
    subject_macro_f1,
    subject_macro_f1_interval,
)


def predictions():
    labels = [
        "initial_contact_loading",
        "mid_stance",
        "terminal_stance_preswing",
        "swing",
    ]
    rows = []
    for subject in ("a", "b"):
        for index, label in enumerate(labels):
            rows.append(
                {
                    "subject_id": subject,
                    "sequence_id": f"{subject}:sequence",
                    "frame_index": index,
                    "y_true": label,
                    "y_pred": label,
                }
            )
    return pd.DataFrame(rows)


def test_perfect_metrics_reconcile_with_confusion_matrix():
    result = compute_metrics(predictions())
    assert result["subject_macro_f1"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert sum(sum(row) for row in result["confusion_matrix"]) == 8
    assert result["transition_validity"] == 1.0


def test_subject_bootstrap_is_clustered_and_deterministic():
    left = subject_bootstrap_interval(predictions(), subject_macro_f1, iterations=100)
    right = subject_bootstrap_interval(predictions(), subject_macro_f1, iterations=100)
    assert left == right
    assert left["lower"] == left["upper"] == 1.0
    assert subject_macro_f1_interval(predictions(), iterations=100) == left


def test_paired_bootstrap_requires_identical_samples():
    left = predictions().assign(sample_id=lambda frame: range(len(frame)))
    right = left.iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly the same"):
        paired_subject_bootstrap_difference(left, right, iterations=100)


def test_unknown_phase_fails():
    frame = predictions()
    frame.loc[0, "y_pred"] = "unknown"
    with pytest.raises(ValueError, match="Unknown phase"):
        compute_metrics(frame)


def test_boundary_metrics_report_exact_timing():
    frame = predictions()
    frame["y_boundary"] = [0, 1, 0, 0] * 2
    frame["boundary_probability"] = [0.1, 0.9, 0.2, 0.1] * 2
    result = boundary_metrics(frame)
    assert result["f1"] == 1.0
    assert result["median_absolute_error_frames"] == 0.0
    assert result["within_one_frame"] == 1.0
