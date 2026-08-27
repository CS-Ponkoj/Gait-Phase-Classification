# Frozen Experimental Protocol

## Research question

Can temporal visual modeling classify four operational gait phases across unseen subjects and visible/thermal imaging conditions more reliably than non-temporal baselines?

## Population and data

- CASIA A: preserved 4,571-image curated subset until complete official provenance is re-established.
- CASIA C: all 153 subject archives and ten normal/slow/fast/bag sequence-condition folders.
- OU-ISIR: excluded.
- Primary analysis: high- and medium-confidence expert-adjudicated cycles.
- Sensitivity analysis: high-confidence cycles only; low-confidence cycles remain excluded.

## Locked split

- Approximately 20% of subjects from each dataset form the untouched test set.
- Remaining subjects form five deterministic development folds.
- Subject IDs are namespaced by dataset.
- Frames, sequences, exact duplicates, and augmented derivatives cannot cross partitions.
- The test-set flag is used only after annotation, preprocessing, model family, and hyperparameters are frozen.

## Comparisons

1. Majority-class baseline
2. Cycle-position prior baseline
3. HOG plus class-balanced linear SVM
4. EfficientNet-B0 single-frame CNN
5. EfficientNet-B0 encoder plus temporal convolutional network

Pretrained weights are permitted only when their version and provenance are recorded. Development experiments include temporal context, augmentation, class weighting, pooled/per-dataset training, and A-to-C/C-to-A transfer ablations.

## Outcomes

- Primary: macro F1 calculated within each subject and then averaged across subjects.
- Secondary: frame macro F1, balanced accuracy, per-class precision/recall/F1, confusion matrix, valid phase-transition rate, and boundary timing error.
- Uncertainty: 95% subject-clustered bootstrap intervals with 2,000 resamples.
- Comparisons: paired subject-clustered bootstrap differences on identical test samples.

Accuracy is secondary and may not be used alone to support a conclusion. Results must be shown overall, by dataset, walking condition, subject, and phase.

## Claim gate

The paper may claim improved classification only when the temporal model's primary-metric difference over the strongest baseline has a 95% interval above zero and the direction remains consistent in per-dataset results. Otherwise, report a negative result or frame the contribution as a reproducible annotation/benchmark study.
