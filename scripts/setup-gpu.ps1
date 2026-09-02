[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$venvRoot = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "Creating Python 3.11 environment..."
    & py -3.11 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the Python 3.11 environment."
    }
}

Write-Host "Installing the project and test dependencies..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $venvPython -m pip install -e "$repoRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw "Project installation failed." }

Write-Host "Installing PyTorch 2.13 with CUDA 13.0 for the RTX 5060..."
& $venvPython -m pip install --upgrade --force-reinstall torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "CUDA-enabled PyTorch installation failed." }

Write-Host "Restoring the locked scientific-library versions..."
& $venvPython -m pip install --upgrade --force-reinstall --no-deps numpy==1.24.3 pandas==2.2.0 Pillow==10.1.0
if ($LASTEXITCODE -ne 0) { throw "Scientific-library compatibility repair failed." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "The installed Python packages are not compatible." }

Write-Host "Verifying CUDA execution and sm_120 support..."
$verification = @'
import torch

assert torch.cuda.is_available(), "CUDA is not available to PyTorch."
capability = torch.cuda.get_device_capability(0)
architecture = f"sm_{capability[0]}{capability[1]}"
assert architecture in torch.cuda.get_arch_list(), (
    f"Installed PyTorch does not support {architecture}: {torch.cuda.get_arch_list()}"
)
value = torch.ones(1, device="cuda") * 2
assert value.item() == 2
print(f"GPU ready: {torch.cuda.get_device_name(0)}; architecture={architecture}; torch={torch.__version__}")
'@
& $venvPython -c $verification
if ($LASTEXITCODE -ne 0) { throw "GPU verification failed." }

Write-Host "GPU environment is ready."
