# Validation Report

Overall assessment: **usable as AI-provisional engineering data; not publication ground truth**.

## Verified coverage

- 100,330 eligible CASIA C frames have exactly one provisional four-phase label.
- 79,882 labels retain the development split and 20,448 retain the frozen test split.
- 1,530 sequences and 153 subjects are represented.
- 4,587 other manifest rows have an explicit exclusion record: 4,571 incomplete-temporal CASIA A frames and 16 excluded CASIA C duplicates.
- The labeled and excluded sets are disjoint and together account for all 104,917 manifest rows.

## Verified integrity

- All four phase names use the locked vocabulary.
- Sample IDs are unique and preserve their original split and fold.
- All generated phase transitions follow the four-class cyclic order.
- 5,658 boundary rows describe complete cycles whose boundary frames are within their source sequence.
- All release checksums reconcile.
- The complete project test suite passed: 25 tests.

## Confidence and limitations

- 98,013 frames are medium-confidence heuristic labels.
- 2,317 frames across 25 sequences use low-confidence timing.
- No frame is human-annotated or expert-adjudicated.
- Binary silhouettes do not identify an anatomical reference limb. The labels represent observable step-cycle timing inferred from lower-limb silhouette spread and fixed phase thresholds.
- Test labels must not be used for model selection or configuration tuning.
