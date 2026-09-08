[CmdletBinding()]
param(
    [ValidateRange(1, 1000)]
    [int]$Epochs = 8,

    [ValidateRange(1, 1024)]
    [int]$BatchSize = 2,

    [ValidateRange(0, 64)]
    [int]$Workers = 4,

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda",

    [switch]$AllowFrozenTest
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if (-not $AllowFrozenTest) {
    throw "Final test access requires -AllowFrozenTest after the development configuration is locked."
}

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Missing virtual-environment Python: $python"
}
$manifest = Join-Path $repoRoot "data\processed\provisional_v0.1-ai\folds\fold_0\manifest.csv.gz"
$config = Join-Path $repoRoot "configs\study.yaml"
if (-not (Test-Path -LiteralPath $manifest -PathType Leaf)) {
    throw "Missing prepared manifest: $manifest"
}

Write-Host "FINAL FROZEN-TEST RUN"
Write-Host "Model: TGPN; window: 27; fixed epochs: $Epochs"
Write-Host "Training: all development subjects; evaluation: frozen test subjects"
Write-Host "Test data will not be used for loss, early stopping, or checkpoint selection."

Push-Location $repoRoot
try {
    & $python -m gait_phase.cli train `
        --manifest $manifest `
        --config $config `
        --workspace $repoRoot `
        --model tgpn `
        --fold 0 `
        --evaluation-split test `
        --allow-test `
        --fixed-training-epochs `
        --label-source provisional `
        --allow-provisional `
        --training-dataset casia_c `
        --evaluation-dataset casia_c `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --learning-rate 0.0003 `
        --patience 5 `
        --temporal-window 27 `
        --temporal-control ordered `
        --num-workers $Workers `
        --device $Device `
        --progress-every 100 `
        --pretrained `
        --augmentation
    if ($LASTEXITCODE -ne 0) {
        throw "Final frozen-test run failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
