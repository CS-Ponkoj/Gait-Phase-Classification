# Reproduction Guide

## Environment

Use Python 3.11 and install the locked environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .
```

GPU users may replace the pinned CPU-compatible PyTorch installation with the matching official CUDA wheel, but must retain the resulting environment record in each run.

## Data preparation

Licensed data is never downloaded or published automatically.

```powershell
gait-phase extract-casia-c
gait-phase build-manifest
gait-phase validate-data data/manifests/generated/casia_a_c_manifest.csv
gait-phase audit-duplicates data/manifests/generated/casia_a_c_manifest.csv --output data/manifests/generated/duplicate_audit
gait-phase make-splits data/manifests/generated/casia_a_c_manifest.csv --output data/manifests/generated/casia_a_c_split.csv
```

After independent annotation and adjudication, validate the frozen manifest:

```powershell
gait-phase validate-data data/manifests/generated/casia_a_c_frozen.csv --frozen
```

## Training and evaluation

Run development folds first:

```powershell
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model majority --fold 0
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model hog_svm --fold 0
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model cnn --fold 0 --pretrained
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model tcn --fold 0 --pretrained
```

Use `--no-augmentation` for the registered augmentation ablation. Use `--training-dataset casia_a_curated --evaluation-dataset casia_c` (and the inverse) for cross-dataset transfer runs.

The untouched test partition requires explicit acknowledgement:

```powershell
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model tcn --evaluation-split test --allow-test --pretrained
```

Every run receives a unique directory under `artifacts/runs/` containing configuration, environment, checkpoint, learning curves where applicable, predictions, and metrics. Generate manuscript evidence only from saved predictions:

```powershell
gait-phase evaluate artifacts/runs/<run>/predictions.csv
gait-phase make-paper-assets artifacts/runs/<run>/predictions.csv --output artifacts/runs/<run>/paper_assets
```
