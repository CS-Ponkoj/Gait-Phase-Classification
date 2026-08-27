import pandas as pd
import pytest

from gait_phase.metrics import compute_metrics, subject_bootstrap_interval, subject_macro_f1


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


def test_unknown_phase_fails():
    frame = predictions()
    frame.loc[0, "y_pred"] = "unknown"
    with pytest.raises(ValueError, match="Unknown phase"):
        compute_metrics(frame)
