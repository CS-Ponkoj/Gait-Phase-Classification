# Data Statement

## Sources

This study is designed for CASIA Gait Dataset A and Dataset C. Dataset A contributes a currently incomplete, curated visible-light subset. Dataset C contributes thermal silhouettes collected under normal, slow, fast, and bag-carrying conditions. OU-ISIR Treadmill Dataset B is explicitly excluded.

## Local completeness

- CASIA A: 4,571 local images across 20 subject identifiers. This is not represented as the complete official dataset. Its acquisition record and completeness must be resolved before submission.
- CASIA C: 153 subject archives, 100,346 PNG entries, with archive integrity previously verified. Full extraction and manifest checks are reproducible locally.

## Labels

The source datasets do not provide the four target phase labels. Two trained annotators and a gait-domain expert must create operational visual labels using the versioned annotation protocol. These labels are not force-plate or motion-capture ground truth.

## Known limitations

- Demographic metadata is unavailable or insufficient for subgroup fairness claims.
- Visible and thermal imagery differ materially in appearance and capture conditions.
- CASIA A provenance is currently medium confidence.
- Silhouette imagery can obscure foot contact and heel rise.
- Dataset licenses and redistribution terms govern all source images.

Only metadata, source checksums, code, and permitted annotation records should be released. Do not place licensed images in Git or publication supplements without written authorization.
