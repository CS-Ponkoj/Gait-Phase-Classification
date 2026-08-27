# Gait Phase Classification

Research workspace for single-frame gait event and gait phase classification experiments.

The project is preserved but not yet reproducible. Historical experiments, datasets, checkpoints, manuscripts, and presentations have been organized without rewriting the original training code or changing scientific results.

## Current research status

- Historical code evaluates three labels: `heel_strike`, `toe_off`, and `other`.
- The manuscript also discusses binary `stance` versus `swing` classification.
- The canonical prediction task must be selected before new training begins.
- Existing accuracy claims require a new subject-independent evaluation before they can be treated as validated results.
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

1. Select one canonical task and label vocabulary.
2. Create a dataset manifest with subject, sequence, frame, label, source, and split fields.
3. Resolve conflicting identical-image labels.
4. Build a subject-independent train/validation/test split.
5. Create a reproducible Python package and one trustworthy baseline.
