# Reproduction Guide

## Environment

Use Python 3.11 and install the locked environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e .
```

GPU users may replace the pinned CPU-compatible PyTorch installation with the matching official CUDA wheel, but must retain the resulting environment record in each run.

For this workstation's NVIDIA RTX 5060, create and verify the CUDA environment with:

```powershell
.\scripts\setup-gpu.ps1
```

This installs the project into `.venv`, installs the pinned CUDA 13.0 PyTorch build, verifies `sm_120` support, and runs an actual tensor operation on the GPU. The older PyTorch 2.5.1 CUDA 12.1 build is incompatible with this GPU.

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

Create the blinded development-only annotation pilot with:

```powershell
gait-phase make-annotation-pilot data/manifests/generated/casia_a_c_split.csv
```

The local package is written to `annotations/pilot/pilot_v3/` and is ignored by Git because it contains licensed image copies. It contains complete CASIA C sequences only; the incomplete CASIA A subset is not used for boundary labeling. Sequences rejected during visual quality control are listed explicitly in `configs/study.yaml`. Give each annotator only their assigned archive. Keep `coordinator_key.csv` private until both independent files are returned.

## Training and evaluation

Prepare the medium-confidence AI-provisional data as copied, fold-aware local trees:

```powershell
gait-phase prepare-training-data data/manifests/generated/casia_a_c_split.csv
```

The command writes `data/processed/provisional_v0.1-ai/`, verifies every source and copied-image checksum, excludes low-confidence rows, and creates `READY.json` only after all five fold manifests pass. The processed tree is ignored by Git because it contains licensed image copies.

### Recommended local training launcher

First activate the Python 3.11 environment created above. Then train fold 0 as a hardware and runtime check:

```powershell
.\scripts\train-provisional.ps1 `
  -Model tcn `
  -Folds 0 `
  -Epochs 30 `
  -Device auto
```

The launcher defaults to batch size 4 for the temporal model and 32 for the single-frame CNN. If GPU memory is exhausted, retry the temporal model with `-BatchSize 2`. `-Device auto` uses CUDA when PyTorch can access it and otherwise uses the CPU. Pretrained EfficientNet weights are enabled by default and may be downloaded on the first run.

Once fold 0 completes, run the locked five-fold development experiment:

```powershell
.\scripts\train-provisional.ps1 `
  -Model tcn `
  -Folds 0,1,2,3,4 `
  -Epochs 30 `
  -Device auto
```

The launcher never passes `--allow-test`; it cannot evaluate the frozen test set. Each fold writes a separate checkpoint, learning curve, predictions, metrics, configuration, and environment record under `artifacts/runs/`. Use `-Model cnn` for the single-frame comparison and `-Model hog_svm` for the classical baseline.

Revalidate the completed tree at any time with:

```powershell
gait-phase validate-prepared-data data/processed/provisional_v0.1-ai
```

Run a provisional development-fold experiment explicitly:

```powershell
gait-phase train `
  --manifest data/processed/provisional_v0.1-ai/folds/fold_0/manifest.csv.gz `
  --fold 0 `
  --model majority `
  --label-source provisional `
  --allow-provisional
```

Frozen-test neural evaluation additionally requires `--evaluation-split test
--allow-test --fixed-training-epochs` and an explicit development-selected
`--epochs` value. In this mode, evaluation data is not used for loss, early
stopping, or checkpoint selection. Never use the test result to revise
preprocessing, labels, model selection, or hyperparameters.

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
gait-phase train --manifest data/manifests/generated/casia_a_c_frozen.csv --model tcn --evaluation-split test --allow-test --fixed-training-epochs --epochs 5 --pretrained
```

Every run receives a unique directory under `artifacts/runs/` containing configuration, environment, checkpoint, learning curves where applicable, predictions, and metrics. Generate manuscript evidence only from saved predictions:

```powershell
gait-phase evaluate artifacts/runs/<run>/predictions.csv
gait-phase make-paper-assets artifacts/runs/<run>/predictions.csv --output artifacts/runs/<run>/paper_assets
```
