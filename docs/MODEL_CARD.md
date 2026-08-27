# Model Card: Four-Phase Gait Classifiers

Status: research framework; no validated trained model has been released.

## Intended use

Benchmark four image-observable gait phases in controlled CASIA sequences and study subject-independent and visible/thermal generalization.

## Out-of-scope use

- Clinical diagnosis, rehabilitation decisions, fall-risk prediction, or patient monitoring
- Real-time safety control
- Identity inference or surveillance deployment
- Populations or capture environments not evaluated in the frozen protocol

## Inputs and outputs

Inputs are single frames or short ordered frame windows normalized to the configured image size. Outputs are probabilities or scores for exactly four ordered operational phase labels.

## Evaluation requirements

Use the locked subject-independent test partition and report subject-macro F1, balanced accuracy, per-class metrics, uncertainty intervals, transition validity, and per-dataset performance. A model is not validated merely because frame-level accuracy is high.

## Risks

Phase labels are visually adjudicated rather than instrument verified. Domain shift, class imbalance, annotation ambiguity, incomplete source provenance, and missing demographic metadata limit generalization.
