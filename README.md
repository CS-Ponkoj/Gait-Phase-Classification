# Gait Phase Classification

Research workspace for a reproducible, subject-independent four-phase gait classification study across visible and thermal silhouette sequences.

The historical project is preserved unchanged under `legacy/`. A new publication-oriented Python package now provides data inventory, annotation QA, leakage-controlled splitting, baseline/model contracts, subject-level evaluation, and paper-asset generation. The research results are **not yet publication-ready** because four-phase expert annotations and final experiments do not exist.

## Current research status

- The canonical task is locked to four operational visual phases: initial contact/loading, mid-stance, terminal stance/pre-swing, and swing.
- Historical three-class and binary results are retained only as background and are not evidence for the new study.
- CASIA A is currently a 4,571-image curated subset with unresolved completeness/provenance.
- CASIA C contains 153 verified subject archives and 100,346 PNG frames across walking conditions.
- OU-ISIR is excluded from the publication study.
- Existing accuracy claims must not be reused; new subject-independent results must be generated from adjudicated labels.
- Five Dataset A duplicate groups contain identical image content under conflicting labels and require review.

See [`inventory/2026-08-27_initial_snapshot/PRESERVATION_REPORT.md`](inventory/2026-08-27_initial_snapshot/PRESERVATION_REPORT.md) for the evidence-backed preservation and data-lineage review.

## Workspace structure

```text
Gait Phase Classification/
├── README.md
├── .gitignore
├── artifacts/
│   └── models/standalone/        # Standalone generated checkpoint
├── data/
│   ├── raw/
│   │   ├── casia_c/              # CASIA C subject archives and verified extracts
│   │   └── ou_isir/              # Encrypted OU-ISIR Treadmill Dataset B archive
│   └── processed/
│       └── gait_events/
│           ├── current/          # Current heel-strike/toe-off/other labels
│           └── legacy/           # Earlier partial labels
├── docs/
│   ├── assets/
│   ├── coursework/
│   ├── notes/
│   ├── paper/
│   ├── presentations/
│   └── references/gait/
├── inventory/
│   ├── 2026-08-27_initial_snapshot/
│   └── 2026-08-27_organization/
├── legacy/
│   └── training_workspace/       # Original code, notebooks, Train/Test, Dataset A, models
└── notebooks/
    └── legacy/                   # Historical duplicate root notebook
```

## Historical training workspace

The original `Gait` directory was moved intact to `legacy/training_workspace`. Its internal `Train`, `Test`, `models`, notebooks, scripts, and extracted Dataset A paths were preserved relative to one another.

The historical workspace is evidence, not a clean training pipeline. It currently lacks a locked environment, deterministic configuration, independent subject-level test procedure, and automated tests.

## Data safety

- Do not commit or publish raw CASIA or OU-ISIR files until their licensing and redistribution terms are confirmed.
- Keep recovered coursework and third-party reference papers local; the root `.gitignore` excludes them from publication.
- Do not delete duplicate files based only on checksum equality. Some duplicates preserve legacy, current, archive, or extraction context.
- Do not modify the dated inventory snapshots.
- Create an independent backup before dataset cleanup or relabeling.

## Recommended next engineering phase

The engineering framework is implemented. Human research work is now the critical path:

1. Resolve or reacquire CASIA A provenance.
2. Complete the two-annotator pilot using [`docs/annotation_protocol.md`](docs/annotation_protocol.md).
3. Reach weighted kappa >= 0.80 and obtain expert adjudication.
4. Freeze the four-phase manifest and subject partitions.
5. Run the registered baselines and temporal experiment, then generate the paper evidence.

## Reproducible commands

```powershell
python -m pip install -e .
gait-phase extract-casia-c
gait-phase build-manifest
gait-phase validate-data data/manifests/generated/casia_a_c_manifest.csv
gait-phase make-splits data/manifests/generated/casia_a_c_manifest.csv --output data/manifests/generated/casia_a_c_split.csv
```

See [`docs/reproducibility.md`](docs/reproducibility.md), [`docs/experimental_protocol.md`](docs/experimental_protocol.md), [`docs/DATA_STATEMENT.md`](docs/DATA_STATEMENT.md), and [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

The current real-corpus gate results are recorded in [`docs/DATA_AUDIT_2026-08-27.md`](docs/DATA_AUDIT_2026-08-27.md).
