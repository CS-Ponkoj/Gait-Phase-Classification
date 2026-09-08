from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gait_phase.auto_annotations import (
    PHASES,
    annotate_sequence_from_signal,
    detect_contact_peaks,
)


def sequence_frame(length: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": f"sample-{index}",
                "dataset": "casia_c",
                "subject_id": "casia_c:001",
                "sequence_id": "casia_c:001:fn00",
                "condition": "fn00",
                "frame_index": str(index),
                "relative_path": (Path("images") / f"{index:03d}.png").as_posix(),
                "split": "development",
                "fold": "0",
            }
            for index in range(length)
        ]
    )


def periodic_spread(length: int = 40) -> np.ndarray:
    indices = np.arange(length)
    return 0.45 + 0.18 * (1 + np.cos(2 * np.pi * (indices - 5) / 10)) / 2


def test_contact_detection_finds_regular_spread_peaks():
    peaks = detect_contact_peaks(periodic_spread(), min_distance=7, prominence=0.025)
    assert peaks == [5, 15, 25, 35]


def test_sequence_annotation_labels_every_frame_and_all_phases():
    labels, boundaries, qc = annotate_sequence_from_signal(
        sequence_frame(),
        periodic_spread(),
        version="test-v1",
        expected_step_frames=10,
    )
    assert len(labels) == 40
    assert labels["sample_id"].nunique() == 40
    assert set(labels["phase_label"]) == set(PHASES)
    assert set(labels["split"]) == {"development"}
    assert set(labels["fold"]) == {"0"}
    assert qc["coverage_method"] == "detected_contacts"
    assert qc["confidence"] == "medium"
    assert (
        boundaries["contact_start_frame"]
        < boundaries["mid_stance_frame"]
    ).all()
    assert (boundaries["mid_stance_frame"] < boundaries["terminal_stance_frame"]).all()
    assert (boundaries["terminal_stance_frame"] < boundaries["swing_frame"]).all()
    assert (boundaries["swing_frame"] < boundaries["next_contact_frame"]).all()


def test_sequence_annotation_falls_back_without_detectable_contacts():
    labels, _, qc = annotate_sequence_from_signal(
        sequence_frame(),
        np.full(40, 0.5),
        version="test-v1",
        expected_step_frames=10,
    )
    assert len(labels) == 40
    assert qc["coverage_method"] == "condition_cycle_prior"
    assert set(labels["confidence"]) == {"low"}


@pytest.mark.parametrize("step_length", [7, 10, 12, 13, 14, 15, 16, 25, 50, 100])
def test_exported_boundaries_match_every_frame_label(step_length):
    length = step_length * 3 + 1
    labels, boundaries, _ = annotate_sequence_from_signal(
        sequence_frame(length), np.full(length, 0.5), "test-consistency", step_length
    )
    by_frame = labels.set_index("frame_index").phase_label
    for row in boundaries.to_dict("records"):
        starts = [row[key] for key in ["contact_start_frame", "mid_stance_frame",
                                      "terminal_stance_frame", "swing_frame"]]
        ends = starts[1:] + [row["next_contact_frame"]]
        for phase, start, end in zip(PHASES, starts, ends):
            assert all(by_frame[str(index)] == phase for index in range(start, end))
