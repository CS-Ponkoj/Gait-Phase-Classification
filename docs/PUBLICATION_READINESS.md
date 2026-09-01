# Publication Readiness

Assessment date: 2026-08-27

Overall assessment: **Needs revision**

## Implemented and verifiable

- Locked four-phase operational vocabulary and annotation protocol
- Independent-annotation agreement and boundary-error calculations
- Versioned manifest schema with provenance, checksum, duplicate, annotation, and split fields
- Deterministic subject-independent test/development partitioning and five development folds
- Majority, cycle-prior, HOG+SVM, frame-CNN, and temporal-TCN model interfaces
- Subject-macro F1, balanced accuracy, per-class metrics, confusion matrices, temporal transition checks, clustered bootstrap intervals, and paper-asset generation
- Unique run directories with configuration, environment, predictions, metrics, curves, and checkpoints
- Test-set access acknowledgement and machine-generated paper evidence
- Data statement, model card, experimental protocol, reproduction guide, and conference-paper outline
- Real A+C audit covering 104,917 frames, 173 subjects, exact duplicates, perceptual candidates, and locked subject counts
- Deterministic blinded pilot generation with 12 distinct CASIA C development subjects, continuous frames, balanced condition-family coverage, frame-indexed previews, and separate annotator templates

## Publication blockers

1. CASIA A is a curated subset; complete official provenance has not been re-established.
2. The blinded pilot package exists locally, but neither annotator has returned completed boundaries and no expert adjudication exists.
3. The weighted-kappa >= 0.80 gate has not been run on real annotations.
4. No model has been trained or evaluated for the new task.
5. No untouched-test predictions, confidence intervals, ablations, or cross-dataset results exist.
6. The historical gait-cycle illustration requires source and reuse-permission verification.
7. A specific conference and its formatting, disclosure, and artifact rules have not yet been selected.

See [`DATA_AUDIT_2026-08-27.md`](DATA_AUDIT_2026-08-27.md) for the real-corpus evidence.

## Claim restrictions

Until every blocker is resolved, do not claim that the four-phase classifier is accurate, generalizes across datasets, improves over a baseline, or has clinical utility. Historical manuscript numbers are not validated results for this protocol.
