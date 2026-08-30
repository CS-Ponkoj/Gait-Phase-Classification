"""Create a deterministic, blinded pilot package for two annotators."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw

from .annotations import BOUNDARY_COLUMNS


def condition_family(dataset: str, condition: str) -> str:
    if dataset == "casia_a_curated":
        return "casia_a"
    prefixes = {"fn": "normal", "fs": "slow", "fq": "fast", "fb": "bag"}
    try:
        return prefixes[condition[:2].lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported CASIA C condition: {condition}") from error


def _stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_pilot_sequences(
    manifest: pd.DataFrame,
    seed: int,
    casia_a_sequences: int = 4,
    casia_c_sequences_per_family: int = 2,
) -> pd.DataFrame:
    required = {"dataset", "subject_id", "sequence_id", "condition", "split", "relative_path", "frame_index", "exclusion_reason"}
    if missing := required - set(manifest.columns):
        raise ValueError(f"Pilot selection is missing manifest columns: {sorted(missing)}")
    eligible = manifest[
        manifest["split"].eq("development") & manifest["exclusion_reason"].fillna("").eq("")
    ].copy()
    if eligible.empty:
        raise ValueError("No eligible development samples are available for the pilot.")
    eligible["condition_family"] = [
        condition_family(dataset, condition)
        for dataset, condition in zip(eligible["dataset"], eligible["condition"])
    ]
    sequence_table = (
        eligible.groupby(["dataset", "condition_family", "condition", "subject_id", "sequence_id"])
        .agg(frame_count=("relative_path", "size"), first_frame=("frame_index", "min"), last_frame=("frame_index", "max"))
        .reset_index()
    )
    quotas = {
        "casia_a": int(casia_a_sequences),
        "normal": int(casia_c_sequences_per_family),
        "slow": int(casia_c_sequences_per_family),
        "fast": int(casia_c_sequences_per_family),
        "bag": int(casia_c_sequences_per_family),
    }
    selected_rows: list[pd.Series] = []
    used_subjects: set[str] = set()
    for family in ("casia_a", "normal", "slow", "fast", "bag"):
        candidates = sequence_table[sequence_table["condition_family"] == family].copy()
        if candidates.empty:
            raise ValueError(f"No pilot candidates found for condition family: {family}")
        median_length = float(candidates["frame_count"].median())
        candidates["length_distance"] = (candidates["frame_count"] - median_length).abs()
        candidates["stable_key"] = candidates["sequence_id"].map(lambda value: _stable_key(str(value), seed))
        candidates = candidates.sort_values(["length_distance", "stable_key", "sequence_id"])
        chosen = []
        for _, row in candidates.iterrows():
            if row["subject_id"] in used_subjects:
                continue
            chosen.append(row)
            used_subjects.add(str(row["subject_id"]))
            if len(chosen) == quotas[family]:
                break
        if len(chosen) != quotas[family]:
            raise ValueError(f"Could not select {quotas[family]} distinct subjects for {family}.")
        selected_rows.extend(chosen)
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    selected["blind_sequence_id"] = [f"pilot_{index:03d}" for index in range(1, len(selected) + 1)]
    return selected


def _preview_frame(path: Path, frame_index: str) -> Image.Image:
    with Image.open(path) as source:
        frame = source.convert("RGB").resize((256, 256))
    draw = ImageDraw.Draw(frame)
    label = f"frame {int(frame_index):04d}"
    draw.rectangle((4, 4, 108, 23), fill="white")
    draw.text((8, 7), label, fill="black")
    return frame


def _annotation_template(blind_ids: list[str], annotator_id: str) -> pd.DataFrame:
    rows = []
    for blind_id in blind_ids:
        row = {column: "" for column in BOUNDARY_COLUMNS}
        row.update(
            dataset="blinded",
            subject_id=f"blinded:{blind_id}",
            sequence_id=blind_id,
            cycle_id="cycle_001",
            annotator_id=annotator_id,
            confidence="",
            notes="Add rows if the sequence contains more than one complete gait cycle.",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=BOUNDARY_COLUMNS)


def write_annotator_archives(output: str | Path, protocol_path: str | Path) -> list[Path]:
    """Create safe handoff archives that never contain the coordinator key or peer labels."""
    output, protocol_path = Path(output), Path(protocol_path)
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Annotation protocol is missing: {protocol_path}")
    archives = []
    for annotator_id in ("annotator_1", "annotator_2"):
        archive_path = output / f"{annotator_id}_package.zip"
        if archive_path.exists():
            raise FileExistsError(f"Annotator archive already exists: {archive_path}")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(protocol_path, "annotation_protocol.md")
            archive.write(output / "ANNOTATOR_README.txt", "ANNOTATOR_README.txt")
            archive.write(output / f"{annotator_id}_boundaries.csv", "boundaries.csv")
            for folder_name in ("frames", "previews"):
                for path in sorted((output / folder_name).rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(output).as_posix())
        archives.append(archive_path)
    return archives


def build_pilot_package(
    manifest: pd.DataFrame,
    workspace: str | Path,
    output: str | Path,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, object]:
    workspace, output = Path(workspace), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Pilot output already exists and is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected = select_pilot_sequences(
        manifest,
        seed=seed,
        casia_a_sequences=int(settings["casia_a_sequences"]),
        casia_c_sequences_per_family=int(settings["casia_c_sequences_per_family"]),
    )
    frames_root, previews_root = output / "frames", output / "previews"
    frames_root.mkdir()
    previews_root.mkdir()
    key_rows = []
    for _, selected_row in selected.iterrows():
        blind_id = str(selected_row["blind_sequence_id"])
        sequence = manifest[manifest["sequence_id"] == selected_row["sequence_id"]].copy()
        sequence = sequence[
            sequence["split"].eq("development") & sequence["exclusion_reason"].fillna("").eq("")
        ]
        sequence["numeric_frame_index"] = pd.to_numeric(sequence["frame_index"], errors="raise")
        sequence = sequence.sort_values(["numeric_frame_index", "relative_path"])
        frame_dir = frames_root / blind_id
        frame_dir.mkdir()
        previews = []
        for _, frame_row in sequence.iterrows():
            source = workspace / str(frame_row["relative_path"])
            if not source.is_file():
                raise FileNotFoundError(f"Pilot source frame is missing: {source}")
            destination = frame_dir / f"{int(frame_row['frame_index']):04d}{source.suffix.lower()}"
            shutil.copy2(source, destination)
            previews.append(_preview_frame(source, str(frame_row["frame_index"])))
        previews[0].save(
            previews_root / f"{blind_id}.gif",
            save_all=True,
            append_images=previews[1:],
            duration=int(settings["gif_duration_ms"]),
            loop=0,
            optimize=False,
        )
        key_rows.append(selected_row.to_dict())
    key = pd.DataFrame(key_rows)
    key.to_csv(output / "coordinator_key.csv", index=False)
    blind_ids = selected["blind_sequence_id"].tolist()
    _annotation_template(blind_ids, "annotator_1").to_csv(output / "annotator_1_boundaries.csv", index=False)
    _annotation_template(blind_ids, "annotator_2").to_csv(output / "annotator_2_boundaries.csv", index=False)
    selection_digest = hashlib.sha256(
        "\n".join(f"{row.blind_sequence_id}:{row.sequence_id}" for row in selected.itertuples()).encode("utf-8")
    ).hexdigest()
    summary = {
        "pilot_version": "pilot_v1",
        "sequences": int(len(selected)),
        "subjects": int(selected["subject_id"].nunique()),
        "frames": int(selected["frame_count"].sum()),
        "development_only": True,
        "condition_family_counts": selected["condition_family"].value_counts().sort_index().astype(int).to_dict(),
        "selection_sha256": selection_digest,
        "seed": int(seed),
    }
    (output / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ANNOTATOR_README.txt").write_text(
        "Four-phase gait annotation pilot\n\n"
        "1. Read docs/annotation_protocol.md before labeling.\n"
        "2. Use previews/<pilot_id>.gif for motion context and frames/<pilot_id>/ for exact frame numbers.\n"
        "3. Complete only your assigned boundary CSV; do not view the other annotator's file or coordinator_key.csv.\n"
        "4. Record five increasing boundary frames for each complete gait cycle. Add rows for additional cycles.\n"
        "5. Use confidence high, medium, or low. Explain exclusions or uncertainty in notes.\n"
        "6. Do not view model predictions; none are included in this package.\n",
        encoding="utf-8",
    )
    write_annotator_archives(output, workspace / "docs" / "annotation_protocol.md")
    return summary
