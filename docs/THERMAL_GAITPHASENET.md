# Thermal GaitPhaseNet Training Guide

Thermal GaitPhaseNet (`tgpn`) is the proposed dense four-phase model for the
CASIA C thermal silhouettes. It uses only the existing medium-confidence
development data during model development. The frozen test set remains closed.

## What the model changes

- Crops and aligns each silhouette without changing the saved source image.
- Preserves body aspect ratio on an 88 by 128 single-channel canvas.
- Combines whole-body, lower-body, and adjacent-frame motion features.
- Uses a 27-frame window by default.
- Uses residual temporal convolutions with dilations 1, 2, 4, and 8.
- Adds one lightweight self-attention layer.
- Predicts all frames in a clip rather than only its center frame.
- Learns both phase classes and phase boundaries.
- Reports results by walking condition and reports boundary timing metrics.

The short command name `tgpn` means Thermal GaitPhaseNet.

## Before training

Open PowerShell in the repository:

```powershell
cd "D:\Gait Phase Classification"
```

Confirm that the prepared release is ready:

```powershell
Test-Path .\data\processed\provisional_v0.1-ai\READY.json
```

The result must be `True`.

## Step 1: One-epoch smoke run

Run a short fold-0 check before a long experiment:

```powershell
.\scripts\train-provisional.ps1 `
  -Model tgpn `
  -Folds 0 `
  -Epochs 1 `
  -BatchSize 2 `
  -Workers 4 `
  -Device cuda `
  -ProgressEvery 20
```

This confirms that the dataset, GPU, model, loss, validation, and checkpoint
pipeline work together. Its score is not a research result because one epoch is
not sufficient training.

## Step 2: Train the proposed model on fold 0

After the smoke run passes:

```powershell
.\scripts\train-provisional.ps1 `
  -Model tgpn `
  -Folds 0 `
  -Epochs 30 `
  -BatchSize 2 `
  -Workers 4 `
  -Device cuda `
  -ProgressEvery 100
```

Defaults used for `tgpn`:

- Window: 27 frames
- Learning rate: 0.0003
- Pretrained EfficientNet-B0 encoder
- Augmentation enabled
- Frozen test excluded

If CUDA reports insufficient memory, retry with `-BatchSize 1`. Do not reduce
the window until the batch-size-one run has been attempted.

## Step 3: Prove whether temporal order helps

These are controlled experiments using the original center-frame TCN. Use the
same fold, epochs, and other settings for all three runs.

Ordered sequence:

```powershell
.\scripts\train-provisional.ps1 -Model tcn -Folds 0 -Epochs 30 -BatchSize 4 -Workers 4 -Device cuda -TemporalWindow 9 -TemporalControl ordered
```

Nine repeated copies of the center frame:

```powershell
.\scripts\train-provisional.ps1 -Model tcn -Folds 0 -Epochs 30 -BatchSize 4 -Workers 4 -Device cuda -TemporalWindow 9 -TemporalControl repeated
```

The same frames in a deterministic shuffled order:

```powershell
.\scripts\train-provisional.ps1 -Model tcn -Folds 0 -Epochs 30 -BatchSize 4 -Workers 4 -Device cuda -TemporalWindow 9 -TemporalControl shuffled
```

The ordered model must outperform the repeated control to show that surrounding
frames help. It must outperform the shuffled control to show that frame order,
not merely multiple poses, helps.

## Step 4: Select temporal window length

Run only on fold 0 during selection:

```powershell
.\scripts\train-provisional.ps1 -Model tgpn -Folds 0 -Epochs 30 -BatchSize 2 -Workers 4 -Device cuda -TemporalWindow 9
```

```powershell
.\scripts\train-provisional.ps1 -Model tgpn -Folds 0 -Epochs 30 -BatchSize 2 -Workers 4 -Device cuda -TemporalWindow 17
```

```powershell
.\scripts\train-provisional.ps1 -Model tgpn -Folds 0 -Epochs 30 -BatchSize 2 -Workers 4 -Device cuda -TemporalWindow 27
```

Choose the window using subject macro F1, per-class F1, boundary error, and
transition validity. Do not select it using the frozen test set.

## Step 5: Run all five development folds

After the window and settings are locked:

```powershell
.\scripts\train-provisional.ps1 `
  -Model tgpn `
  -Folds 0,1,2,3,4 `
  -Epochs 30 `
  -BatchSize 2 `
  -Workers 4 `
  -Device cuda `
  -ProgressEvery 100
```

Every run is stored separately under `artifacts\runs\`. Important files are:

- `config.yaml`: exact settings used
- `model.pt`: best checkpoint
- `learning_curves.csv`: train and validation loss by epoch
- `predictions.csv`: one prediction per validation frame
- `metrics.json`: overall, per-class, per-condition, transition, and boundary metrics
- `environment.json`: Python, library, operating-system, and hardware environment

## Frozen test warning

Do not run the test partition during model development. The test command requires
explicit test acknowledgement and should be used only after preprocessing,
window size, architecture, loss weights, and training settings are frozen.

Aggregate and validate the registered development runs first:

```powershell
.\.venv\Scripts\python.exe -m gait_phase.cli aggregate-results `
  --registry .\configs\development_runs_v1.yaml `
  --workspace . `
  --output .\artifacts\paper_assets\development_v2 `
  --bootstrap-iterations 2000
```

The five TGPN folds selected eight final training epochs: the median of their
best validation-loss epochs (8, 7, 8, 9, and 12). The final launcher trains on
all development subjects for exactly eight epochs. It does not calculate test
loss during training and cannot use test data for early stopping or checkpoint
selection. Run it only at the final test gate:

```powershell
.\scripts\train-final-provisional.ps1 -Epochs 8 -AllowFrozenTest
```

Direct neural test commands are rejected unless they include both
`--allow-test` and `--fixed-training-epochs`, plus an explicit `--epochs` value.

The current labels are AI-provisional operational labels. Model results are
engineering evidence, not clinical ground-truth evidence.
