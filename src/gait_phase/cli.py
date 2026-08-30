"""Command-line interface for the publication research pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .annotations import annotation_agreement, boundary_disagreement, expand_boundaries, validate_boundaries
from .baselines import CyclePriorBaseline, HogSvmBaseline, MajorityBaseline
from .config import load_config
from .manifest import build_manifest, duplicate_audit, manifest_summary, validate_manifest
from .metrics import compute_metrics, subject_bootstrap_interval, subject_macro_f1
from .models import build_frame_cnn, build_temporal_tcn
from .paper_assets import make_paper_assets
from .pilot import build_pilot_package
from .splits import assert_split_integrity, make_subject_splits
from .training import (
    create_run_directory,
    eligible_labeled,
    prediction_table,
    save_run_records,
    set_determinism,
    train_torch_model,
)


def _workspace(path: str | None) -> Path:
    return Path(path or ".").resolve()


def _write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def command_extract(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    archives = sorted(root.glob("*.zip"))
    if len(archives) != 153:
        raise ValueError(f"Expected 153 CASIA C archives, found {len(archives)}.")
    extracted = 0
    for archive_path in archives:
        target = (root / archive_path.stem).resolve()
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                destination = (target / member.filename).resolve()
                if target not in destination.parents and destination != target:
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    if destination.stat().st_size != member.file_size:
                        raise ValueError(f"Existing extraction differs in size: {destination}")
                    continue
                with archive.open(member) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                extracted += 1
    print(json.dumps({"archives": len(archives), "extracted_files": extracted, "root": str(root)}, indent=2))
    return 0


def command_build_manifest(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    workspace = _workspace(args.workspace)
    frame = build_manifest(
        workspace / config["data"]["casia_a_root"],
        workspace / config["data"]["casia_c_root"],
        workspace,
    )
    output = args.output or workspace / config["data"]["manifest"]
    _write_csv(frame, output)
    print(json.dumps(manifest_summary(frame), indent=2, default=str))
    return 0


def command_validate(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    errors = validate_manifest(frame, frozen=args.frozen)
    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "summary": manifest_summary(frame)}
    print(json.dumps(report, indent=2, default=str))
    return 0 if not errors else 1


def command_splits(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    errors = validate_manifest(frame)
    if errors:
        raise ValueError("Manifest validation failed: " + "; ".join(errors))
    split = make_subject_splits(
        frame,
        test_fraction=float(config["data"]["test_subject_fraction"]),
        folds=int(config["data"]["development_folds"]),
        seed=int(config["study"]["seed"]),
    )
    assert_split_integrity(split)
    _write_csv(split, args.output)
    summary = split.groupby(["dataset", "split"])["subject_id"].nunique().rename("subjects").reset_index()
    print(summary.to_string(index=False))
    print(f"Excluded cross-subject exact-duplicate samples: {(split['exclusion_reason'] == 'exact_duplicate_shared_across_subjects').sum()}")
    return 0


def command_duplicate_audit(args: argparse.Namespace) -> int:
    frame = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    errors = validate_manifest(frame)
    if errors:
        raise ValueError("Manifest validation failed: " + "; ".join(errors))
    summary, exact, perceptual = duplicate_audit(frame)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    exact.to_csv(output / "exact_duplicate_groups.csv", index=False)
    perceptual.to_csv(output / "perceptual_hash_collision_groups.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def command_annotation_report(args: argparse.Namespace) -> int:
    annotations = pd.read_csv(args.boundaries, dtype=str, keep_default_na=False)
    errors = validate_boundaries(annotations)
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, indent=2))
        return 1
    annotators = sorted(annotations["annotator_id"].unique())
    if len(annotators) != 2:
        raise ValueError("Agreement report requires exactly two annotators.")
    left = annotations[annotations["annotator_id"] == annotators[0]]
    right = annotations[annotations["annotator_id"] == annotators[1]]
    expanded = expand_boundaries(annotations)
    pairs = expanded[expanded["annotator_id"] == annotators[0]].merge(
        expanded[expanded["annotator_id"] == annotators[1]],
        on=["sequence_id", "cycle_id", "frame_index"],
        suffixes=("_a", "_b"),
    )
    report = annotation_agreement(pairs["label_a"], pairs["label_b"])
    report.update(boundary_disagreement(left, right, args.fps))
    report["status"] = "PASS" if report["weighted_kappa"] >= args.threshold else "FAIL"
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


def command_make_pilot(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    workspace = _workspace(args.workspace)
    manifest = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed: " + "; ".join(errors))
    settings = config["annotation_pilot"]
    output = Path(args.output) if args.output else workspace / settings["output"]
    summary = build_pilot_package(
        manifest=manifest,
        workspace=workspace,
        output=output,
        settings=settings,
        seed=int(config["study"]["seed"]),
    )
    print(json.dumps({"output": str(output.resolve()), **summary}, indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    config["training"]["augmentation"] = bool(args.augmentation)
    set_determinism(int(config["study"]["seed"]))
    workspace = _workspace(args.workspace)
    manifest = eligible_labeled(pd.read_csv(args.manifest, dtype=str, keep_default_na=False))
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("Manifest validation failed: " + "; ".join(errors))
    if args.evaluation_split == "test":
        if not args.allow_test:
            raise ValueError("Test evaluation requires the explicit --allow-test flag after configuration freeze.")
        train_frame = manifest[manifest["split"] == "development"].copy()
        validation_frame = manifest[manifest["split"] == "test"].copy()
    else:
        fold = str(args.fold)
        development = manifest[manifest["split"] == "development"].copy()
        train_frame = development[development["fold"].astype(str) != fold]
        validation_frame = development[development["fold"].astype(str) == fold]
    if args.training_dataset != "pooled":
        train_frame = train_frame[train_frame["dataset"] == args.training_dataset]
    evaluation_dataset = args.training_dataset if args.evaluation_dataset == "match" else args.evaluation_dataset
    if evaluation_dataset != "pooled":
        validation_frame = validation_frame[validation_frame["dataset"] == evaluation_dataset]
    if train_frame.empty or validation_frame.empty:
        raise ValueError("Selected training or evaluation partition is empty.")
    run_dir = create_run_directory(workspace / "artifacts/runs", args.model, config)
    if args.model == "majority":
        model = MajorityBaseline().fit(train_frame)
        predictions = prediction_table(validation_frame, model.predict(validation_frame))
        (run_dir / "model.json").write_text(json.dumps({"label": model.label}, indent=2), encoding="utf-8")
    elif args.model == "cycle_prior":
        model = CyclePriorBaseline().fit(train_frame)
        predictions = prediction_table(validation_frame, model.predict(validation_frame))
        (run_dir / "model.json").write_text(json.dumps({"bins": model.bins, "lookup": model.lookup}, indent=2), encoding="utf-8")
    elif args.model == "hog_svm":
        import pickle

        model = HogSvmBaseline(workspace).fit(train_frame)
        predictions = prediction_table(validation_frame, model.predict(validation_frame))
        with (run_dir / "model.pkl").open("wb") as stream:
            pickle.dump(model.model, stream)
    elif args.model == "cnn":
        model = build_frame_cnn(pretrained=args.pretrained)
        predictions = train_torch_model(model, train_frame, validation_frame, workspace, run_dir, config)
    else:
        model = build_temporal_tcn(pretrained=args.pretrained)
        predictions = train_torch_model(model, train_frame, validation_frame, workspace, run_dir, config, temporal=True)
    save_run_records(run_dir, config, predictions)
    print(json.dumps({"run_directory": str(run_dir), "metrics": compute_metrics(predictions)}, indent=2))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    predictions = pd.read_csv(args.predictions)
    metrics = compute_metrics(predictions)
    metrics["subject_macro_f1_interval"] = subject_bootstrap_interval(
        predictions, subject_macro_f1, iterations=args.bootstrap_iterations
    )
    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def command_assets(args: argparse.Namespace) -> int:
    predictions = pd.read_csv(args.predictions)
    metrics = make_paper_assets(predictions, args.output, args.bootstrap_iterations)
    print(json.dumps({"output": str(Path(args.output).resolve()), "subject_macro_f1": metrics["subject_macro_f1"]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gait-phase", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-casia-c", help="Safely extract all 153 CASIA C archives.")
    extract.add_argument("--root", default="data/raw/casia_c")
    extract.set_defaults(function=command_extract)
    manifest = subparsers.add_parser("build-manifest", help="Create the versioned A+C provenance manifest.")
    manifest.add_argument("--config", default="configs/study.yaml")
    manifest.add_argument("--workspace")
    manifest.add_argument("--output")
    manifest.set_defaults(function=command_build_manifest)
    validate = subparsers.add_parser("validate-data", help="Validate manifest and leakage invariants.")
    validate.add_argument("manifest")
    validate.add_argument("--frozen", action="store_true")
    validate.set_defaults(function=command_validate)
    splits = subparsers.add_parser("make-splits", help="Create deterministic subject-independent partitions.")
    splits.add_argument("manifest")
    splits.add_argument("--config", default="configs/study.yaml")
    splits.add_argument("--output", required=True)
    splits.set_defaults(function=command_splits)
    duplicates = subparsers.add_parser("audit-duplicates", help="Report exact and perceptual duplicate candidates.")
    duplicates.add_argument("manifest")
    duplicates.add_argument("--output", required=True)
    duplicates.set_defaults(function=command_duplicate_audit)
    annotation = subparsers.add_parser("annotation-report", help="Compute inter-annotator agreement and boundary error.")
    annotation.add_argument("boundaries")
    annotation.add_argument("--fps", type=float, required=True)
    annotation.add_argument("--threshold", type=float, default=0.80)
    annotation.set_defaults(function=command_annotation_report)
    pilot = subparsers.add_parser("make-annotation-pilot", help="Create a deterministic blinded two-annotator pilot package.")
    pilot.add_argument("manifest")
    pilot.add_argument("--config", default="configs/study.yaml")
    pilot.add_argument("--workspace")
    pilot.add_argument("--output")
    pilot.set_defaults(function=command_make_pilot)
    train = subparsers.add_parser("train", help="Train a baseline or four-output neural model.")
    train.add_argument("--manifest", required=True)
    train.add_argument("--config", default="configs/study.yaml")
    train.add_argument("--workspace")
    train.add_argument("--model", choices=["majority", "cycle_prior", "hog_svm", "cnn", "tcn"], required=True)
    train.add_argument("--fold", type=int, choices=range(5), default=0)
    train.add_argument("--evaluation-split", choices=["validation", "test"], default="validation")
    train.add_argument("--allow-test", action="store_true")
    train.add_argument("--pretrained", action="store_true")
    train.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--training-dataset", choices=["pooled", "casia_a_curated", "casia_c"], default="pooled")
    train.add_argument("--evaluation-dataset", choices=["match", "pooled", "casia_a_curated", "casia_c"], default="match")
    train.set_defaults(function=command_train)
    evaluate = subparsers.add_parser("evaluate", help="Recompute metrics from saved predictions.")
    evaluate.add_argument("predictions")
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2000)
    evaluate.add_argument("--output")
    evaluate.set_defaults(function=command_evaluate)
    assets = subparsers.add_parser("make-paper-assets", help="Generate paper tables and confusion matrix.")
    assets.add_argument("predictions")
    assets.add_argument("--output", required=True)
    assets.add_argument("--bootstrap-iterations", type=int, default=2000)
    assets.set_defaults(function=command_assets)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except (FileNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
