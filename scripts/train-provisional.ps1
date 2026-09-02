[CmdletBinding()]
param(
    [ValidateSet("cnn", "tcn", "hog_svm", "majority", "cycle_prior")]
    [string]$Model = "tcn",

    [ValidateRange(0, 4)]
    [int[]]$Folds = @(0),

    [ValidateRange(1, 1000)]
    [int]$Epochs = 30,

    [ValidateRange(0, 1024)]
    [int]$BatchSize = 0,

    [ValidateRange(0, 64)]
    [int]$Workers = 0,

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "auto",

    [ValidateRange(0.0000001, 1.0)]
    [double]$LearningRate = 0.001,

    [ValidateRange(1, 100)]
    [int]$Patience = 5,

    [ValidateRange(3, 99)]
    [int]$TemporalWindow = 9,

    [ValidateRange(1, 10000)]
    [int]$ProgressEvery = 100,

    [switch]$NoPretrained,
    [switch]$NoAugmentation
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$config = Join-Path $repoRoot "configs\study.yaml"
$preparedRoot = Join-Path $repoRoot "data\processed\provisional_v0.1-ai"
$readyFile = Join-Path $preparedRoot "READY.json"

if (-not (Test-Path -LiteralPath $readyFile -PathType Leaf)) {
    throw "Prepared dataset is not ready: $readyFile"
}

$ready = Get-Content -LiteralPath $readyFile -Raw | ConvertFrom-Json
if ($ready.status -ne "READY") {
    throw "The prepared dataset is not marked READY."
}

if ($BatchSize -eq 0) {
    $BatchSize = if ($Model -eq "tcn") { 4 } else { 32 }
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { "python" }

Write-Host "Model: $Model"
Write-Host "Folds: $($Folds -join ', ')"
Write-Host "Epochs: $Epochs; batch size: $BatchSize; device: $Device"
Write-Host "The frozen test set will not be used."

Push-Location $repoRoot
try {
    foreach ($fold in $Folds) {
        $manifest = Join-Path $preparedRoot "folds\fold_$fold\manifest.csv.gz"
        if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
            throw "Missing fold manifest: $manifest"
        }

        $arguments = @(
            "-m", "gait_phase.cli", "train",
            "--manifest", $manifest,
            "--config", $config,
            "--workspace", $repoRoot,
            "--model", $Model,
            "--fold", $fold,
            "--evaluation-split", "validation",
            "--label-source", "provisional",
            "--allow-provisional",
            "--training-dataset", "casia_c",
            "--evaluation-dataset", "casia_c",
            "--epochs", $Epochs,
            "--batch-size", $BatchSize,
            "--learning-rate", $LearningRate,
            "--patience", $Patience,
            "--temporal-window", $TemporalWindow,
            "--num-workers", $Workers,
            "--device", $Device,
            "--progress-every", $ProgressEvery
        )
        if (-not $NoPretrained -and $Model -in @("cnn", "tcn")) {
            $arguments += "--pretrained"
        }
        if ($NoAugmentation) {
            $arguments += "--no-augmentation"
        } else {
            $arguments += "--augmentation"
        }

        Write-Host "`nStarting fold $fold..."
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for fold $fold with exit code $LASTEXITCODE."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "`nTraining complete. Results are under artifacts\runs\."
