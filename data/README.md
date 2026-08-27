# Data

This directory separates source datasets from manually labeled or otherwise processed collections.

## Raw

- `raw/casia_c/`: 153 CASIA Dataset C subject archives plus verified extracted subjects 003, 030, and 140.
- `raw/ou_isir/`: encrypted OU-ISIR Treadmill Dataset B archive. Its 486,004 entries require the original password for content validation.

The historical extracted Dataset A collection remains inside `legacy/training_workspace/dataset_a` to preserve the original training workspace layout.

## Processed

- `processed/gait_events/current/`: current `heel_strike`, `toe_off`, and `other` labels.
- `processed/gait_events/legacy/`: earlier partial labeled collection.

These datasets are excluded from normal Git tracking. Use the dated inventory manifests for provenance and checksums.
