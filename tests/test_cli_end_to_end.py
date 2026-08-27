from pathlib import Path

import pandas as pd
from PIL import Image

from gait_phase.cli import main
from gait_phase.constants import MANIFEST_COLUMNS, PHASES
from gait_phase.hashing import sha256_file


def create_frozen_fixture(workspace: Path) -> Path:
    rows = []
    for subject in range(6):
        for frame_index, label in enumerate(PHASES):
            relative = Path("images") / f"subject-{subject}" / f"frame-{frame_index}.png"
            path = workspace / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (16, 16), color=subject * 20 + frame_index).save(path)
            row = {column: "" for column in MANIFEST_COLUMNS}
            row.update(
                sample_id=f"sample-{subject}-{frame_index}",
                dataset="casia_c",
                subject_id=f"casia_c:{subject:03d}",
                sequence_id=f"casia_c:{subject:03d}:fn00",
                condition="fn00",
                view="side_thermal",
                frame_index=str(frame_index),
                relative_path=relative.as_posix(),
                sha256=sha256_file(path),
                adjudicated_label=label,
                annotation_confidence="high",
                split="development",
                fold=str(subject % 5),
            )
            rows.append(row)
    manifest = workspace / "frozen.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest, index=False)
    return manifest


def test_majority_run_and_paper_assets_are_reproducible(tmp_path):
    manifest = create_frozen_fixture(tmp_path)
    assert main(["validate-data", str(manifest), "--frozen"]) == 0
    assert main(
        [
            "train",
            "--manifest",
            str(manifest),
            "--workspace",
            str(tmp_path),
            "--model",
            "majority",
            "--fold",
            "0",
        ]
    ) == 0
    predictions = next((tmp_path / "artifacts" / "runs").glob("*/predictions.csv"))
    output = tmp_path / "paper"
    assert main(
        [
            "make-paper-assets",
            str(predictions),
            "--output",
            str(output),
            "--bootstrap-iterations",
            "100",
        ]
    ) == 0
    assert (output / "metrics.json").is_file()
    assert (output / "confusion_matrix.png").is_file()
    assert (output / "results_summary.md").is_file()
