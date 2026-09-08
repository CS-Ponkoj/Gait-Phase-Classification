"""Freeze chat-confirmed AI labels separately from primary data and stage new development frames.

This coordination step reads split/identity metadata, never test images or model outputs.
It deliberately does not write adjudicated labels into the study manifest or start training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from gait_phase.annotations import expand_boundaries, validate_boundaries
from gait_phase.pilot import condition_family, select_pilot_sequences

META_COLUMNS = ["sample_id", "dataset", "subject_id", "sequence_id", "condition", "frame_index",
                "relative_path", "sha256", "exclusion_reason", "split", "fold"]
BOUNDARIES = ["initial_contact_frame", "mid_stance_frame", "terminal_stance_frame",
              "swing_frame", "next_initial_contact_frame"]
REVIEW_VERSION = "pilot_v3_user_confirmed_v1"
BATCH_VERSION = "development_batch_001"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path: Path, value: object) -> None:
    # Exclusive creation: original annotations and any earlier release cannot be overwritten.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def source_path(workspace: Path, relative: str) -> Path:
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError("Missing source or source path outside the workspace")
    return path


def load_metadata(workspace: Path) -> pd.DataFrame:
    # Explicitly omit all annotation/prediction columns, including frozen-test labels.
    meta = pd.read_csv(workspace / "data/manifests/generated/casia_a_c_split.csv",
                       dtype=str, keep_default_na=False, usecols=META_COLUMNS)
    if meta["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs in split metadata")
    dev = set(meta.loc[meta["split"].eq("development"), "subject_id"])
    test = set(meta.loc[meta["split"].eq("test"), "subject_id"])
    if dev & test:
        raise ValueError("Development/test subject overlap")
    return meta


def validate_selected(meta: pd.DataFrame) -> None:
    if meta.empty or not meta["split"].eq("development").all():
        raise ValueError("Selected frames must all be development data")
    if not meta["dataset"].eq("casia_c").all() or not meta["exclusion_reason"].eq("").all():
        raise ValueError("Selected frames must be eligible CASIA C data")
    if not meta["fold"].isin(["0", "1", "2", "3", "4"]).all():
        raise ValueError("Invalid development fold")
    if meta[["sequence_id", "frame_index"]].duplicated().any() or meta["sha256"].duplicated().any():
        raise ValueError("Selected frames have duplicate keys or exact image content")
    if not meta["sha256"].str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Missing or invalid source checksum")


def finalize_review(workspace: Path, meta: pd.DataFrame) -> dict:
    work = workspace / "annotations/working"
    output = workspace / "annotations/reviewed" / REVIEW_VERSION
    if output.exists():
        raise FileExistsError(f"Release already exists: {output}")
    ai = json.loads((work / "ai_pass_b_v1.json").read_text(encoding="utf-8"))
    approval = json.loads((work / "user_confirmation_2026-09-07.json").read_text(encoding="utf-8"))
    if approval["review_status"] != "user_confirmed_via_chat":
        raise ValueError("Expected user chat confirmation is absent")
    if ai["version"] != "ai_pass_b_v1" or len(ai["cycles"]) != 19:
        raise ValueError("Unexpected source annotation version or scope")
    key = pd.read_csv(workspace / "annotations/pilot/pilot_v3/coordinator_key.csv",
                      dtype=str, keep_default_na=False)
    if key["blind_sequence_id"].duplicated().any():
        raise ValueError("Ambiguous pilot identity key")
    keyed = key.set_index("blind_sequence_id")
    boundaries, decisions = [], []
    statuses = {"proposed": "accepted_ai_assisted_low_confidence",
                "recommend_exclude": "excluded_user_confirmed", "unresolved": "on_hold_unresolved"}
    for row in ai["cycles"]:
        identity = keyed.loc[row["sequence"]]
        decision = {"pilot_sequence_id": row["sequence"], "sequence_id": identity["sequence_id"],
                    "subject_id": identity["subject_id"], "cycle_id": row["cycle"],
                    "status": statuses[row["status"]], "reason": row["reason"],
                    "review_method": "user_chat_confirmation", "primary_eligible": False}
        decisions.append(decision)
        if row["status"] != "proposed":
            continue
        record = {"dataset": identity["dataset"], "subject_id": identity["subject_id"],
                  "sequence_id": identity["sequence_id"], "cycle_id": row["cycle"],
                  **dict(zip(BOUNDARIES, row["boundaries"])),
                  "annotator_id": "ai_assistant_pass_b_user_chat_confirmed", "confidence": row["confidence"],
                  "notes": row["reason"], "pilot_sequence_id": row["sequence"],
                  "label_provenance": "human_reviewed_AI_assisted_chat_confirmation",
                  "review_method": "user_chat_confirmation", "expert_adjudicated": False,
                  "primary_eligible": False, "training_eligible": False,
                  "reference_foot": row["reference_foot"], "boundary_evidence": row["evidence"],
                  "review_windows": row["windows"]}
        boundaries.append(record)
    table = pd.DataFrame(boundaries)
    errors = validate_boundaries(table)
    if errors or len(table) != 4:
        raise ValueError("Invalid accepted boundary set: " + "; ".join(errors))
    expanded = expand_boundaries(table)
    expanded["frame_index"] = expanded["frame_index"].astype(str)
    joined = expanded.merge(meta, on=["sequence_id", "frame_index"], how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != 111:
        raise ValueError("Accepted frames do not match 111 unique source records")
    joined = joined.drop(columns="_merge")
    validate_selected(joined)
    for record in joined.to_dict("records"):
        if digest(source_path(workspace, record["relative_path"])) != record["sha256"]:
            raise ValueError("Source image checksum mismatch")
    joined["frame_index"] = joined["frame_index"].astype(int)
    joined["annotation_confidence"] = "low"
    joined["label_provenance"] = "human_reviewed_AI_assisted_chat_confirmation"
    joined["annotation_version"] = REVIEW_VERSION
    joined["training_eligible"] = False
    joined["primary_eligible"] = False
    summary = {"version": REVIEW_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "review_status": "user_confirmed_via_chat", "accepted_cycles": len(table),
               "accepted_frames": len(joined), "accepted_subjects": int(joined["subject_id"].nunique()),
               "decision_counts": dict(Counter(r["status"] for r in decisions)),
               "class_counts": joined["label"].value_counts().sort_index().to_dict(),
               "condition_family_cycle_counts": dict(Counter(condition_family(r["dataset"], keyed.loc[r["pilot_sequence_id"], "condition"]) for r in boundaries)),
               "development_only": True, "test_images_opened": 0, "training_eligible": False,
               "primary_eligible": False, "expert_adjudicated": False, "independent_annotation_gate_met": False,
               "frame_by_frame_inspection_documented": approval["frame_by_frame_inspection_documented"],
               "source_hashes": {"ai_pass_b_v1.json": digest(work / "ai_pass_b_v1.json"),
                                 "user_confirmation_2026-09-07.json": digest(work / "user_confirmation_2026-09-07.json"),
                                 "split_metadata": digest(workspace / "data/manifests/generated/casia_a_c_split.csv")},
               "limitation": "Versioned accepted snapshot for inspection and diagnostic planning, not a frozen ground-truth benchmark. Low confidence and the unmet expert/independent-review gate prevent primary use."}
    output.mkdir(parents=True)
    dump(output / "accepted_boundaries.json", {"version": REVIEW_VERSION, "training_eligible": False, "records": boundaries})
    dump(output / "accepted_frame_labels.json", {"version": REVIEW_VERSION, "training_eligible": False, "records": joined.to_dict("records")})
    dump(output / "cycle_decisions.json", {"version": REVIEW_VERSION, "records": decisions})
    summary["output_sha256"] = {p.name: digest(p) for p in sorted(output.glob("*.json"))}
    dump(output / "release_summary.json", summary)
    return summary


def choose_batch(meta: pd.DataFrame, prior_subjects: set[str], excluded_sequences: set[str], per_family: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = meta[meta["split"].eq("development") & meta["dataset"].eq("casia_c")
                    & ~meta["subject_id"].isin(prior_subjects) & meta["exclusion_reason"].eq("")].copy()
    # Reject whole sequences with duplicate frame indices before the existing contiguity check.
    duplicated = eligible.duplicated(["sequence_id", "frame_index"], keep=False)
    excluded_sequences = excluded_sequences | set(eligible.loc[duplicated, "sequence_id"])
    selected = select_pilot_sequences(eligible, seed=20260907,
                                      casia_c_sequences_per_family=per_family,
                                      excluded_sequence_ids=excluded_sequences)
    selected["blind_sequence_id"] = [f"batch001_{n:03d}" for n in range(1, len(selected) + 1)]
    frames = eligible[eligible["sequence_id"].isin(selected["sequence_id"])].copy()
    validate_selected(frames)
    if set(selected["subject_id"]) & prior_subjects:
        raise ValueError("Batch repeats an earlier pilot subject")
    return selected, frames


def prepare_batch(workspace: Path, meta: pd.DataFrame) -> dict:
    output = workspace / "annotations/batches" / BATCH_VERSION
    if output.exists():
        raise FileExistsError(f"Batch already exists: {output}")
    prior_subjects: set[str] = set()
    prior_keys = sorted((workspace / "annotations/pilot").glob("*/coordinator_key.csv"))
    for key in prior_keys:
        prior_subjects.update(pd.read_csv(key, dtype=str, usecols=["subject_id"])["subject_id"])
    config = yaml.safe_load((workspace / "configs/study.yaml").read_text(encoding="utf-8"))
    selected, frames = choose_batch(meta, prior_subjects, set(config["annotation_pilot"]["excluded_sequence_ids"]))
    # Preflight every selected source before creating output; test source paths never enter this loop.
    for record in frames.to_dict("records"):
        if digest(source_path(workspace, record["relative_path"])) != record["sha256"]:
            raise ValueError("Source image checksum mismatch")
    frames["numeric_frame_index"] = frames["frame_index"].astype(int)
    output.mkdir(parents=True)
    public, private = [], []
    for selection in selected.to_dict("records"):
        blind = selection["blind_sequence_id"]
        sequence = frames[frames["sequence_id"].eq(selection["sequence_id"])].sort_values("numeric_frame_index")
        destination = output / "frames" / blind
        destination.mkdir(parents=True)
        public_frames = []
        for record in sequence.to_dict("records"):
            name = f"{record['numeric_frame_index']:04d}.png"
            copied = destination / name
            shutil.copyfile(source_path(workspace, record["relative_path"]), copied)
            if digest(copied) != record["sha256"]:
                raise ValueError("Copied frame checksum mismatch")
            public_frames.append({"frame_index": record["numeric_frame_index"], "path": f"frames/{blind}/{name}", "sha256": record["sha256"]})
            private.append({"batch_sequence_id": blind, **{c: record[c] for c in META_COLUMNS}})
        public.append({"sequence_id": blind, "annotation_status": "not_started", "frames": public_frames})
    dump(output / "review_manifest.json", {"version": BATCH_VERSION, "annotation_status": "not_started", "sequences": public})
    dump(output / "coordinator_source_mapping.json", {"version": BATCH_VERSION, "records": private})
    summary = {"version": BATCH_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "sequences": len(selected), "subjects": int(selected["subject_id"].nunique()), "frames": len(frames),
               "condition_family_counts": selected["condition_family"].value_counts().sort_index().to_dict(),
               "annotation_status": "not_started", "labels_created": 0, "development_only": True,
               "test_images_opened": 0, "previous_pilot_subjects_excluded": len(prior_subjects),
               "overlap_with_previous_pilot_subjects": 0, "training_eligible": False,
               "source_and_copy_checksums_verified": len(frames), "seed": 20260907,
               "sampling": "Existing deterministic median-length selection, six distinct new subjects per condition family; no selection by model accuracy or presumed label quality.",
               "visual_quality_status": "Not yet reviewed. Complete frame indices do not guarantee visible feet or complete gait cycles.",
               "output_sha256": {p.name: digest(p) for p in sorted(output.glob("*.json"))}}
    dump(output / "batch_summary.json", summary)
    # Read-only viewer: no annotation fields or submissions; feedback is given in chat.
    payload = json.dumps(public, ensure_ascii=True).replace("<", "\\u003c")
    html = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Next gait annotation batch</title><style>body{font:16px/1.5 system-ui;max-width:900px;margin:24px auto;padding:16px;color:#18212d}img{width:min(100%,640px);height:auto;image-rendering:pixelated;display:block;background:black}button,select{font:inherit;padding:10px;margin:4px}a{color:#194f92}</style>
<h1>Next development annotation batch</h1><p>This batch is prepared, not annotated. View the original frames and give feedback in chat. There are no annotation forms to complete.</p>
<label>Sequence <select id="sequence"></select></label><p id="counter" aria-live="polite"></p><img id="image" alt="Original gait source frame"><p id="error" role="alert"></p>
<button id="previous">Previous frame</button><button id="next">Next frame</button><button id="play">Play slowly</button><p>Playback is a viewing aid, not the acquisition frame rate. Source images are copied unchanged.</p><p><a href="batch_summary.json">Batch summary</a></p>
<script>const sequences=PAYLOAD;let s=0,f=0,timer=null;const q=id=>document.getElementById(id);for(const [i,r] of sequences.entries()){const o=document.createElement('option');o.value=i;o.textContent=r.sequence_id;q('sequence').append(o)}
function stop(){if(timer!==null)clearInterval(timer);timer=null;q('play').textContent='Play slowly'}
function render(){const r=sequences[s],v=r.frames[f];q('image').src=v.path;q('image').alt=r.sequence_id+' original frame '+v.frame_index;q('counter').textContent=r.sequence_id+' | Frame '+v.frame_index+' ('+(f+1)+' of '+r.frames.length+') | Not annotated';q('previous').disabled=f===0;q('next').disabled=f===r.frames.length-1;q('error').textContent=''}
q('image').onerror=()=>{stop();q('error').textContent='Frame could not load. Keep the viewer beside its frames folder.'};q('sequence').onchange=()=>{stop();s=Number(q('sequence').value);f=0;render()};q('previous').onclick=()=>{stop();if(f>0)f--;render()};q('next').onclick=()=>{stop();if(f<sequences[s].frames.length-1)f++;render()};q('play').onclick=()=>{if(timer!==null){stop();return}q('play').textContent='Pause';timer=setInterval(()=>{if(f>=sequences[s].frames.length-1){stop();return}f++;render()},350)};render();</script></html>""".replace("PAYLOAD", payload)
    with (output / "view_batch.html").open("x", encoding="utf-8") as handle:
        handle.write(html)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["finalize", "batch"])
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    meta = load_metadata(workspace)
    result = finalize_review(workspace, meta) if args.stage == "finalize" else prepare_batch(workspace, meta)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
