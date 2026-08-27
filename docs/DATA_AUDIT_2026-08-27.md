# CASIA A+C Publication Data Audit

Audit date: 2026-08-27

Generated manifest: `data/manifests/generated/casia_a_c_manifest.csv` (local, ignored)

Split manifest: `data/manifests/generated/casia_a_c_split.csv` (local, ignored)

## Corpus

| Dataset | Subjects | Frames | Status |
|---|---:|---:|---|
| CASIA A curated subset | 20 | 4,571 | Incomplete official provenance; do not call this the complete CASIA A dataset |
| CASIA C | 153 | 100,346 | All 153 verified archives extracted and inventoried |
| Total | 173 | 104,917 | Draft manifest passes structural validation |

CASIA C contains 1,530 source sequences across ten condition identifiers. Together with 80 inferred CASIA A sequences, the manifest contains 1,610 sequences.

## Duplicate review

- Exact SHA-256 groups: 76 groups, 166 member rows.
- Seventy-five exact groups are confined to one subject.
- One exact group contains 16 images shared across five CASIA C subjects.
- Those 16 rows are preserved but marked `exact_duplicate_shared_across_subjects` and excluded from primary analysis before splitting.
- DCT perceptual-hash collision screening produced 13,605 candidate groups. Many span subjects because normalized silhouettes at similar poses have similar low-frequency structure. These are review candidates, not proof that independently captured frames are duplicates.
- Online augmentation is generated only after partition selection, so augmented derivatives cannot enter another partition.

Perceptual similarity alone must not be used to delete or relabel data. Exact checksums, sequence provenance, source subject, and annotation review remain authoritative.

## Locked partitions

| Dataset | Development subjects | Test subjects |
|---|---:|---:|
| CASIA A curated subset | 16 | 4 |
| CASIA C | 122 | 31 |

The split is deterministic from the registered study seed. All frames from a subject remain together. Development subjects are assigned across five folds. The test partition remains unusable for publication evaluation until labels, annotation agreement, preprocessing, and model selection are frozen.

## Gate results

| Check | Result | Evidence |
|---|---|---|
| Draft manifest schema and identifiers | PASS | `gait-phase validate-data` |
| Subject-independent split | PASS | 173 subjects each confined to one partition |
| Exact duplicate leakage among eligible rows | PASS | 16 unsafe rows excluded; no remaining exact group crosses partitions |
| Four-phase adjudicated labels | BLOCKED | 0 adjudicated frames |
| Frozen publication manifest | BLOCKED | Eligible samples do not yet have adjudicated labels |
| CASIA A completeness/provenance | BLOCKED | Only the preserved curated subset is locally established |
