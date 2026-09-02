"""Build atomic, copied, fold-aware training views from provisional labels."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import MANIFEST_COLUMNS, PHASES
from .hashing import sha256_file
from .splits import assert_split_integrity

PROVISIONAL_COLUMNS = (
    "provisional_label",
    "provisional_confidence",
    "provisional_source",
    "provisional_version",
    "source_relative_path",
    "prepared_role",
)

REQUIRED_LABEL_COLUMNS = {
    "sample_id",
    "dataset",
    "subject_id",
    "sequence_id",
    "condition",
    "frame_index",
    "relative_path",
    "split",
    "fold",
    "phase_label",
    "cycle_position",
    "confidence",
    "annotation_source",
    "annotation_version",
}


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, index=False)


def _safe_component(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not normalized:
        raise ValueError(f"Cannot create a safe path component from: {value!r}")
    return normalized


def _validate_inputs(manifest: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_LABEL_COLUMNS - set(labels.columns)
    if missing:
        raise ValueError(f"Provisional labels are missing columns: {sorted(missing)}")
    if labels["sample_id"].fillna("").eq("").any() or labels["sample_id"].duplicated().any():
        raise ValueError("Provisional sample_id values must be present and unique.")
    invalid = sorted(set(labels["phase_label"]) - set(PHASES))
    if invalid:
        raise ValueError(f"Invalid provisional labels: {invalid}")
    if set(labels["confidence"]) - {"medium", "low"}:
        raise ValueError("Provisional confidence must be medium or low.")
    source = manifest[manifest["sample_id"].isin(labels["sample_id"])].copy()
    if len(source) != len(labels):
        missing_ids = sorted(set(labels["sample_id"]) - set(source["sample_id"]))
        raise ValueError(f"Provisional labels contain unknown sample IDs: {missing_ids[:5]}")
    joined = source.merge(labels, on="sample_id", suffixes=("", "_label"), validate="one_to_one")
    for column in (
        "dataset",
        "subject_id",
        "sequence_id",
        "condition",
        "frame_index",
        "relative_path",
        "split",
        "fold",
    ):
        left = joined[column].fillna("").astype(str)
        right = joined[f"{column}_label"].fillna("").astype(str)
        if not left.eq(right).all():
            raise ValueError(f"Provisional labels changed source manifest field: {column}")
    if set(joined["dataset"]) != {"casia_c"}:
        raise ValueError("Prepared provisional data must contain CASIA C only.")
    if joined["exclusion_reason"].fillna("").ne("").any():
        raise ValueError("Provisional labels include samples excluded by the source manifest.")
    assert_split_integrity(joined)
    development = joined[joined["split"].eq("development")]
    if (development.groupby("subject_id")["fold"].nunique() > 1).any():
        raise ValueError("At least one development subject crosses validation folds.")
    return joined


def _prepared_rows(joined: pd.DataFrame, confidence: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = joined[joined["confidence"].eq(confidence)].copy()
    withheld = joined[~joined["confidence"].eq(confidence)].copy()
    if primary.empty:
        raise ValueError(f"No provisional rows have confidence={confidence}.")
    primary["provisional_label"] = primary["phase_label"]
    primary["provisional_confidence"] = primary["confidence"]
    primary["provisional_source"] = primary["annotation_source"]
    primary["provisional_version"] = primary["annotation_version"]
    primary["source_relative_path"] = primary["relative_path"]
    primary["cycle_position"] = primary["cycle_position_label"]
    return primary, withheld


def _relative_destination(row: pd.Series, role: str, fold: int | None = None) -> Path:
    base = Path("test") if role == "test" else Path("folds") / f"fold_{fold}" / role
    suffix = Path(str(row["source_relative_path"])).suffix.lower() or ".png"
    return (
        base
        / _safe_component(row["provisional_label"])
        / _safe_component(row["subject_id"])
        / _safe_component(row["condition"])
        / f"{_safe_component(row['sample_id'])}{suffix}"
    )


def _copy_and_verify(task: tuple[Path, Path, str]) -> tuple[int, str]:
    source, destination, expected_sha = task
    if not source.is_file():
        raise FileNotFoundError(f"Prepared-data source image is missing: {source}")
    if _sha256(source) != expected_sha:
        raise ValueError(f"Source image checksum does not match the manifest: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual_sha = _sha256(destination)
    if actual_sha != expected_sha:
        raise ValueError(f"Copied image checksum mismatch: {destination}")
    return destination.stat().st_size, actual_sha


def _copy_view(
    frame: pd.DataFrame,
    workspace: Path,
    staging: Path,
    final_output: Path,
    role: str,
    fold: int | None,
    workers: int,
) -> tuple[pd.DataFrame, int, str]:
    result = frame.copy()
    relatives = [_relative_destination(row, role, fold) for _, row in result.iterrows()]
    result["prepared_role"] = role
    result["relative_path"] = [(final_output / relative).relative_to(workspace).as_posix() for relative in relatives]
    tasks = [
        (workspace / str(row["source_relative_path"]), staging / relative, str(row["sha256"]))
        for relative, (_, row) in zip(relatives, result.iterrows())
    ]
    total_bytes = 0
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for relative, (size, checksum) in zip(relatives, executor.map(_copy_and_verify, tasks)):
            total_bytes += size
            digest.update(f"{relative.as_posix()}:{checksum}\n".encode("utf-8"))
    columns = [*MANIFEST_COLUMNS, *PROVISIONAL_COLUMNS]
    return result[columns], total_bytes, digest.hexdigest()


def _metadata_checksums(root: Path) -> dict[str, str]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}
        and path.name not in {"checksums.sha256", "READY.json"}
    )
    return {path.relative_to(root).as_posix(): _sha256(path) for path in paths}


def prepare_training_data(
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    workspace: str | Path,
    output: str | Path,
    settings: dict[str, Any],
) -> dict[str, object]:
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if workspace not in output.parents:
        raise ValueError("Prepared data output must be inside the workspace.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared data output already exists and is not empty: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent)).resolve()
    if staging.parent != output.parent:
        raise RuntimeError("Prepared-data staging directory escaped its intended parent.")
    confidence = str(settings.get("confidence", "medium"))
    folds = int(settings.get("folds", 5))
    workers = int(settings.get("copy_workers", 16))
    try:
        joined = _validate_inputs(manifest, labels)
        primary, withheld = _prepared_rows(joined, confidence)
        test = primary[primary["split"].eq("test")].copy()
        development = primary[primary["split"].eq("development")].copy()
        if set(development["fold"].astype(str)) != {str(index) for index in range(folds)}:
            raise ValueError("Development rows do not contain exactly the configured folds.")
        test_view, test_bytes, test_digest = _copy_view(
            test, workspace, staging, output, "test", None, workers
        )
        fold_summaries: dict[str, dict[str, object]] = {}
        total_copied_files, total_copied_bytes = len(test_view), test_bytes
        for fold in range(folds):
            fold_text = str(fold)
            train = development[development["fold"].astype(str).ne(fold_text)].copy()
            validation = development[development["fold"].astype(str).eq(fold_text)].copy()
            train_view, train_bytes, train_digest = _copy_view(
                train, workspace, staging, output, "train", fold, workers
            )
            validation_view, validation_bytes, validation_digest = _copy_view(
                validation, workspace, staging, output, "validation", fold, workers
            )
            prepared_manifest = pd.concat([train_view, validation_view, test_view], ignore_index=True)
            assert_split_integrity(prepared_manifest)
            manifest_path = staging / "folds" / f"fold_{fold}" / "manifest.csv.gz"
            _write_gzip_csv(prepared_manifest, manifest_path)
            fold_summaries[fold_text] = {
                "train_frames": int(len(train_view)),
                "validation_frames": int(len(validation_view)),
                "test_frames": int(len(test_view)),
                "train_subjects": int(train_view["subject_id"].nunique()),
                "validation_subjects": int(validation_view["subject_id"].nunique()),
                "test_subjects": int(test_view["subject_id"].nunique()),
                "train_image_tree_sha256": train_digest,
                "validation_image_tree_sha256": validation_digest,
                "manifest_sha256": _sha256(manifest_path),
            }
            total_copied_files += len(train_view) + len(validation_view)
            total_copied_bytes += train_bytes + validation_bytes
        withheld_output = withheld[
            [
                "sample_id",
                "dataset",
                "subject_id",
                "sequence_id",
                "condition",
                "frame_index",
                "relative_path",
                "split",
                "fold",
                "phase_label",
                "confidence",
                "annotation_source",
                "annotation_version",
            ]
        ].copy()
        withheld_output["prepared_exclusion_reason"] = f"confidence_not_{confidence}"
        _write_gzip_csv(withheld_output, staging / "excluded_low_confidence.csv.gz")
        summary = {
            "status": "READY",
            "label_source": "ai_provisional",
            "annotation_versions": sorted(primary["provisional_version"].unique().tolist()),
            "confidence_policy": confidence,
            "development_frames": int(len(development)),
            "test_frames": int(len(test)),
            "withheld_low_confidence_frames": int(len(withheld)),
            "copied_files": int(total_copied_files),
            "copied_bytes": int(total_copied_bytes),
            "test_image_tree_sha256": test_digest,
            "folds": fold_summaries,
            "source_manifest_sha256": str(settings["source_manifest_sha256"]),
            "source_labels_sha256": str(settings["source_labels_sha256"]),
        }
        expected = settings.get("expected_counts")
        if expected:
            actual_counts = {
                "development_frames": summary["development_frames"],
                "test_frames": summary["test_frames"],
                "withheld_low_confidence_frames": summary["withheld_low_confidence_frames"],
                "copied_files": summary["copied_files"],
            }
            for name, actual in actual_counts.items():
                if int(expected[name]) != int(actual):
                    raise ValueError(f"Prepared-data count mismatch for {name}: expected {expected[name]}, got {actual}")
            for fold, fold_expected in expected["folds"].items():
                for name in ("train_frames", "validation_frames"):
                    actual = fold_summaries[str(fold)][name]
                    if int(fold_expected[name]) != int(actual):
                        raise ValueError(
                            f"Prepared-data count mismatch for fold {fold} {name}: "
                            f"expected {fold_expected[name]}, got {actual}"
                        )
        (staging / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        metadata = _metadata_checksums(staging)
        (staging / "checksums.sha256").write_text(
            "".join(f"{checksum}  {name}\n" for name, checksum in metadata.items()), encoding="utf-8"
        )
        ready = {**summary, "metadata_checksums": metadata}
        (staging / "READY.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
        if output.exists():
            output.rmdir()
        staging.replace(output)
        return ready
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise


def validate_prepared_data(
    root: str | Path,
    workspace: str | Path,
    verify_image_hashes: bool = True,
) -> dict[str, object]:
    root, workspace = Path(root).resolve(), Path(workspace).resolve()
    ready_path = root / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"Prepared data is missing READY.json: {ready_path}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("status") != "READY":
        raise ValueError("Prepared data is not marked READY.")
    referenced: dict[str, str] = {}
    fold_results: dict[str, dict[str, int]] = {}
    for fold in range(5):
        manifest_path = root / "folds" / f"fold_{fold}" / "manifest.csv.gz"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prepared fold manifest is missing: {manifest_path}")
        frame = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        required = {*MANIFEST_COLUMNS, *PROVISIONAL_COLUMNS}
        if missing := required - set(frame.columns):
            raise ValueError(f"Prepared fold {fold} is missing columns: {sorted(missing)}")
        if frame["sample_id"].duplicated().any():
            raise ValueError(f"Prepared fold {fold} has duplicate sample IDs.")
        if set(frame["provisional_label"]) - set(PHASES):
            raise ValueError(f"Prepared fold {fold} has an invalid provisional label.")
        if set(frame["provisional_confidence"]) != {"medium"}:
            raise ValueError(f"Prepared fold {fold} contains non-medium confidence rows.")
        train = frame[frame["prepared_role"].eq("train")]
        validation = frame[frame["prepared_role"].eq("validation")]
        test = frame[frame["prepared_role"].eq("test")]
        if not train["split"].eq("development").all() or not validation["split"].eq("development").all():
            raise ValueError(f"Prepared fold {fold} has a non-development train/validation row.")
        if not test["split"].eq("test").all():
            raise ValueError(f"Prepared fold {fold} has a non-test test row.")
        if train["fold"].astype(str).eq(str(fold)).any() or not validation["fold"].astype(str).eq(str(fold)).all():
            raise ValueError(f"Prepared fold {fold} has an incorrect role-to-fold assignment.")
        train_subjects = set(train["subject_id"])
        validation_subjects = set(validation["subject_id"])
        test_subjects = set(test["subject_id"])
        if train_subjects & validation_subjects or train_subjects & test_subjects or validation_subjects & test_subjects:
            raise ValueError(f"Prepared fold {fold} has subject leakage.")
        expected_fold = ready["folds"][str(fold)]
        if len(train) != int(expected_fold["train_frames"]) or len(validation) != int(expected_fold["validation_frames"]):
            raise ValueError(f"Prepared fold {fold} row counts do not match READY.json.")
        if len(test) != int(ready["test_frames"]):
            raise ValueError(f"Prepared fold {fold} test count does not match READY.json.")
        for row in frame.itertuples():
            relative = str(row.relative_path)
            if relative in referenced and referenced[relative] != str(row.sha256):
                raise ValueError(f"Prepared path has conflicting checksums: {relative}")
            referenced[relative] = str(row.sha256)
            if str(row.provisional_label) not in Path(relative).parts:
                raise ValueError(f"Prepared path is outside its label directory: {relative}")
        fold_results[str(fold)] = {
            "train_frames": int(len(train)),
            "validation_frames": int(len(validation)),
            "test_frames": int(len(test)),
        }
    withheld = pd.read_csv(root / "excluded_low_confidence.csv.gz", dtype=str, keep_default_na=False)
    if len(withheld) != int(ready["withheld_low_confidence_frames"]):
        raise ValueError("Low-confidence exclusion count does not match READY.json.")
    if set(withheld["confidence"]) != {"low"}:
        raise ValueError("Low-confidence exclusion file contains another confidence level.")
    actual_images = {
        path.relative_to(workspace).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    }
    if actual_images != set(referenced):
        raise ValueError(
            f"Prepared image-reference mismatch: {len(actual_images)} files, {len(referenced)} references."
        )
    if len(referenced) != int(ready["copied_files"]):
        raise ValueError("Prepared copied-file count does not match READY.json.")
    if verify_image_hashes:
        def verify_image(item: tuple[str, str]) -> str | None:
            relative, expected_sha = item
            path = workspace / relative
            if not path.is_file():
                raise FileNotFoundError(f"Prepared image is missing: {path}")
            if _sha256(path) != expected_sha:
                return str(path)
            return None

        with ThreadPoolExecutor(max_workers=16) as executor:
            for mismatch in executor.map(verify_image, referenced.items()):
                if mismatch:
                    raise ValueError(f"Prepared image checksum mismatch: {mismatch}")
    checksum_lines = (root / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for line in checksum_lines:
        expected_sha, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError(f"Prepared metadata checksum mismatch: {relative}")
    return {
        "status": "PASS",
        "copied_files": int(len(referenced)),
        "image_hashes_verified": bool(verify_image_hashes),
        "withheld_low_confidence_frames": int(len(withheld)),
        "folds": fold_results,
    }
