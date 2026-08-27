"""Shared study constants."""

PHASES = (
    "initial_contact_loading",
    "mid_stance",
    "terminal_stance_preswing",
    "swing",
)

PHASE_TO_INDEX = {phase: index for index, phase in enumerate(PHASES)}
INDEX_TO_PHASE = {index: phase for phase, index in PHASE_TO_INDEX.items()}

MANIFEST_COLUMNS = (
    "sample_id",
    "dataset",
    "subject_id",
    "sequence_id",
    "condition",
    "view",
    "frame_index",
    "cycle_id",
    "cycle_position",
    "relative_path",
    "sha256",
    "perceptual_hash",
    "duplicate_group",
    "legacy_label",
    "annotator_1_label",
    "annotator_2_label",
    "adjudicated_label",
    "annotation_confidence",
    "exclusion_reason",
    "split",
    "fold",
)

ALLOWED_TRANSITIONS = {
    0: {0, 1},
    1: {1, 2},
    2: {2, 3},
    3: {3, 0},
}
