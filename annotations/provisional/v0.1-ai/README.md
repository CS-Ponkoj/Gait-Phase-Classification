# AI-Provisional Four-Phase Labels

These files contain deterministic machine-generated labels, not human or clinical ground truth. They may be used for pipeline development and sensitivity experiments only. The reference limb cannot be anatomically identified from the binary silhouettes. Do not use these labels to support a publication accuracy claim without independent validation and expert adjudication.

## Files

- `frame_labels.csv.gz`: one provisional label for every eligible CASIA C frame, including split and fold.
- `cycle_boundaries.csv.gz`: complete observed step-cycle boundaries only.
- `sequence_qc.csv`: detection quality and confidence for every sequence.
- `excluded_samples.csv.gz`: every manifest row outside the labeled set and its reason.
- `split_summary.csv`: frame, subject, and sequence counts by split, fold, and phase.
- `provenance.json`: method, source checksum, and release totals.
- `checksums.sha256`: integrity hashes for the release files.
