import pandas as pd
import pytest

from gait_phase.annotations import annotation_agreement, boundary_disagreement, expand_boundaries, validate_boundaries


def boundary_row(annotator, offset=0):
    return {
        "dataset": "casia_c",
        "subject_id": "casia_c:001",
        "sequence_id": "casia_c:001:fn00",
        "cycle_id": "cycle_1",
        "initial_contact_frame": 0 + offset,
        "mid_stance_frame": 2 + offset,
        "terminal_stance_frame": 4 + offset,
        "swing_frame": 6 + offset,
        "next_initial_contact_frame": 8 + offset,
        "annotator_id": annotator,
        "confidence": "high",
        "notes": "",
    }


def test_boundary_expansion_produces_all_four_phases():
    frame = pd.DataFrame([boundary_row("a")])
    assert validate_boundaries(frame) == []
    expanded = expand_boundaries(frame)
    assert len(expanded) == 8
    assert expanded.groupby("label").size().to_dict() == {
        "initial_contact_loading": 2,
        "mid_stance": 2,
        "terminal_stance_preswing": 2,
        "swing": 2,
    }


def test_annotation_agreement_and_boundary_error():
    agreement = annotation_agreement(
        ["initial_contact_loading", "mid_stance", "swing"],
        ["initial_contact_loading", "mid_stance", "swing"],
    )
    assert agreement == {"raw_agreement": 1.0, "weighted_kappa": 1.0}
    left, right = pd.DataFrame([boundary_row("a")]), pd.DataFrame([boundary_row("b", 1)])
    error = boundary_disagreement(left, right, fps=25)
    assert error["mean_absolute_frames"] == 1.0
    assert error["mean_absolute_milliseconds"] == 40.0


def test_invalid_boundary_order_is_rejected():
    row = boundary_row("a")
    row["swing_frame"] = 3
    errors = validate_boundaries(pd.DataFrame([row]))
    assert any("strictly increasing" in error for error in errors)
