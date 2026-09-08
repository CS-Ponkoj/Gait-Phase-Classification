"""Save a consistency-only repair; never overwrite or promote provisional labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from gait_phase.auto_annotations import PHASE_THRESHOLDS
from scripts.audit_annotation_release import compare_boundaries, read_table, write_json


def repair(workspace: Path, output: Path, allow_test: bool):
    if not allow_test:
        raise ValueError("Explicit test-annotation audit access is required.")
    workspace, output = workspace.resolve(), output.resolve()
    if not output.is_relative_to(workspace / "annotations/working"):
        raise ValueError("Output must stay inside annotations/working.")
    if output.exists():
        raise FileExistsError(output)
    original = workspace / "annotations/provisional/v0.1-ai/cycle_boundaries.csv.gz"
    labels_path = workspace / "annotations/provisional/v0.1-ai/frame_labels.csv.gz"
    boundaries, labels = read_table(original), read_table(labels_path)
    fixed = boundaries.copy()
    fields = ["mid_stance_frame", "terminal_stance_frame", "swing_frame"]
    for index, row in boundaries.iterrows():
        start, stop = int(row.contact_start_frame), int(row.next_contact_frame)
        for minimum, (field, threshold) in enumerate(zip(fields, PHASE_THRESHOLDS), 1):
            fixed.at[index, field] = str(start + max(minimum, math.ceil((stop - start) * threshold)))
    before, after = compare_boundaries(labels, boundaries), compare_boundaries(labels, fixed)
    after_disagreements = int(after.phase_label.ne(after.boundary_phase).sum())
    if after_disagreements or len(before) != len(after):
        raise ValueError("Repaired boundaries do not exactly match the unchanged labels and coverage.")
    changed = int((boundaries[fields] != fixed[fields]).any(axis=1).sum())
    write_json(output, {
        "version": "v0.1-ai-boundary-consistency-repair-1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CONSISTENCY_ONLY_NOT_CORRECTED_GAIT_GROUND_TRUTH",
        "primary_eligible": False, "training_eligible_as_ground_truth": False,
        "original_files_modified": False, "frame_phase_labels_changed": 0,
        "source_sha256": {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [original, labels_path]},
        "boundary_records": len(fixed), "changed_boundary_records": changed,
        "compared_frames": len(after),
        "frame_disagreements_before": int(before.phase_label.ne(before.boundary_phase).sum()),
        "frame_disagreements_after": after_disagreements,
        "limitation": "Same-foot tracking is still absent. This repairs rounding only, not biological phase accuracy.",
        "records": fixed.to_dict("records"),
    })
    return {"output": str(output), "changed_boundary_records": changed,
            "frame_disagreements_after": after_disagreements}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-test-annotation-audit", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(Path(__file__).resolve().parents[1], args.output,
                            args.allow_test_annotation_audit), indent=2))
