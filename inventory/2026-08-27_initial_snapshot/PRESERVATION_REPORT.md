# Gait Phase Classification Preservation Report

Snapshot date: 2026-08-27  
Workspace: `D:\Gait Phase Classification`  
Method: non-destructive inventory, SHA-256 hashing, archive validation, exact-content lineage matching

## Preservation status

The initial workspace was inventoried before any planned reorganization or cleanup.

- Initial source files: 4,885
- Initial source bytes: 1,423,743,030
- Final observed source files: 9,453
- Final observed source bytes: 1,422,927,496
- SHA-256 hash failures: 0
- Existing files intentionally moved, renamed, edited, or deleted by this inventory: 0
- Recoverable files copied into `preserved_originals/` from the Recycle Bin: 2
- Inventory directory excluded from source manifests: `inventory/`

The workspace is not yet backed up by this operation. The checksums prove file identity but do not replace the original data if a disk fails or a file is deleted.

## Unexpected workspace drift during inventory

The workspace changed while inventory operations were running:

- Initial state: `Gait\dataset_a.zip` existed, was 4,671,618 bytes, passed a complete ZIP stream read, and contained 4,571 files totaling 3,889,931 uncompressed bytes.
- Current state: that archive is absent and `Gait\dataset_a\` exists with 4,571 files totaling 3,889,931 bytes.
- Final observed drift: 4,571 added paths, three removed root paths, zero modified paths.

The inventory did not issue an extraction, move, rename, or delete operation against `Gait\dataset_a.zip`. The extracted directory appeared externally while the snapshot was being built. The initial and current manifests are both retained, and `workspace_drift.csv` records the complete change.

Because the archive disappeared before per-entry hashes were saved, the new folder can be associated with the archive through matching entry count, paths, and byte totals, but an exact archive-entry-to-extracted-file hash comparison is no longer possible.

Two project documents, `Assignment 1_Ponkoj.docx` and `Graduate Section.docx`, also disappeared during the inventory. Exact copies were located in the D: drive Recycle Bin, their original-path metadata was verified, and copies were restored to the workspace. An external process removed the restored root copies again. Verified copies were therefore saved under `preserved_originals/`; both SHA-256 values match the initial manifest, and the Recycle Bin copies were left untouched.

A temporary `~$gait_project_pp.pptx` application lock file appeared while the presentation was open. Temporary Office lock files are excluded from source manifests because they are transient application state rather than project artifacts.

## Dataset and archive inventory

### OU-ISIR Treadmill Dataset B

- Path: `TreadmillDatasetB.zip`
- Archive size: 438,488,738 bytes
- Central-directory entries: 486,004
- All entries are encrypted.
- Archive listing succeeds, but content CRC and SHA-256 validation require the original dataset password.
- Status: preserve; content integrity not fully verified.

### CASIA Dataset A

- Initial archive path: `Gait\dataset_a.zip`
- Initial archive validation: passed complete stream reading before it disappeared.
- Current extracted path: `Gait\dataset_a\`
- Current images: 4,571
- Subjects represented: 20
- Class counts: heel strike 440, other 3,523, toe off 608
- Status: preserve both initial checksum evidence and current extracted content; investigate the archive disappearance before cleanup.

### CASIA Dataset C silhouettes

- Subject archives: 153
- Archive validation: 153 of 153 passed complete stream reading.
- Extracted subjects: 003, 030, and 140
- Extracted images: 1,947
- Exact archive-to-folder verification: 1,947 of 1,947 matched by size and SHA-256.
- Missing extracted files: 0
- Extra extracted files: 0
- Content mismatches: 0
- Status: verified and preserved.

## Derived-data lineage

Every inspected derived Dataset A image matches the current extracted Dataset A content exactly:

| Collection | Files | Exact content matches | Unmatched |
|---|---:|---:|---:|
| `Gait\Train` | 1,172 | 1,172 | 0 |
| `Gait\Test` | 30 | 30 | 0 |
| `Gait_classified` | 932 | 932 | 0 |
| `Gait Phase Classification\Gait_classified_legacy` | 625 | 625 | 0 |

This confirms that these four collections are derived from the Dataset A collection rather than Dataset C.

## Confirmed data-quality preservation findings

Dataset A contains seven exact-content duplicate groups covering fourteen paths.

- Two groups repeat identical content within the `other` class.
- Five groups place identical image content under different class labels.
- Cross-class conflicts include heel strike versus other and toe off versus other.

These files are preserved unchanged. They must be reviewed before training because identical model inputs with conflicting target labels cannot be learned consistently. See `dataset_a_source_duplicate_content.csv` for exact paths and checksums.

The wider workspace contains 852 exact-duplicate groups with 2,336 members. The maximum theoretical storage recovery from keeping one copy per group is only 1,160,032 bytes. No duplicate should be removed until its purpose and provenance are reviewed.

## Source-confidence classification

- High confidence: OU-ISIR Treadmill Dataset B archive identity, CASIA Dataset C archive identity, project-authored code/notebooks, model outputs, and Dataset A lineage for derived images.
- Medium confidence: the original acquisition/provenance record for Dataset A and the relationship of older manuscript claims to the surviving experiment artifacts.
- Unknown or unresolved: redistribution permissions, the treadmill archive password, the actor/process that replaced `Gait\dataset_a.zip` with an extracted directory, and the exact notebook run that produced the historical metrics.

## Inventory artifacts

| File | Purpose |
|---|---|
| `file_manifest.csv` | Initial state: path, category, source confidence, size, timestamp, and SHA-256 for every pre-existing file. |
| `file_manifest_current_after_drift.csv` | Intermediate state after the observed Dataset A change and before the two Word files disappeared. |
| `workspace_drift.csv` | Initial-to-intermediate Dataset A path changes. |
| `file_manifest_final_observed.csv` | Final observed source state, excluding inventory artifacts and temporary Office lock files. |
| `workspace_drift_final.csv` | Complete initial-to-final added, removed, and modified path record. |
| `archive_integrity.csv` | Archive entry counts, uncompressed totals, integrity status, samples, and limitations. |
| `dataset_collections.csv` | Collection-level data origin, subject, class, file-count, and lineage summary. |
| `extraction_verification.csv` | Byte-level verification of extracted CASIA C subjects against their archives. |
| `dataset_a_lineage_summary.csv` | Collection-level exact-match results against Dataset A. |
| `dataset_a_lineage_matches.csv` | Image-level Dataset A lineage and label comparison. |
| `dataset_a_source_duplicate_content.csv` | Duplicate Dataset A content, including cross-class label conflicts. |
| `duplicate_groups.csv` | Workspace-wide exact duplicate groups and potential recovery size. |
| `duplicate_members.csv` | One row per member of every workspace-wide duplicate group. |
| `summary_by_category.csv` | Initial file and byte totals by preservation category. |
| `summary_by_extension.csv` | Initial file and byte totals by file extension. |
| `preserved_originals/` | Verified copies of the two Word documents removed externally during inventory. |
| `inventory_artifact_checksums.csv` | SHA-256 values for the generated inventory package, excluding the checksum file itself. |

## Preservation rules for the next phase

1. Do not delete the initial manifests or checksum evidence.
2. Do not remove duplicate files solely because their hashes match; some are intentional archive/extracted or legacy/current copies.
3. Do not publish raw CASIA or OU-ISIR material until redistribution terms are confirmed.
4. Do not place large datasets or HDF5 model files into normal Git history.
5. Create an independent backup before moving or reorganizing source assets.
6. Resolve the five cross-class duplicate-label groups before training a new model.
7. Resolve or explicitly accept the missing `Gait\dataset_a.zip` before declaring the source collection fully preserved.
