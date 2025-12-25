# PowerShell script (train) for regen_inception random images, k=500

$ErrorActionPreference = "Stop"

# Ensure we run from script folder
$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PSScriptRoot

# Resolve repo root and target Python script relative to this file (go up 3 levels)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
$pyScript = Join-Path $repoRoot "model_interpretation\fisher_information_hessian_rsvd.py"
# Change working directory to repo root so all relative imports/paths work
Set-Location $repoRoot

# Python executable (adjust if needed)
$python = "python"

# Parameters
$trainer = "trainers.specific_trainers.regen_inception_random_images"
$checkpoint = "latest"
$data = "train"
$k = 500
$numSamples = 10000
$useLabels = $false

# Disable TF32 for better numerical fidelity (optional)
$env:TORCH_ALLOW_TF32_CUBLAS = "0"
$env:TORCH_ALLOW_TF32_CUDNN = "0"

# Unbuffer Python so logs stream immediately
$env:PYTHONUNBUFFERED = "1"

# Log setup (Windows-specific logs folder)
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "regen_inception_random_images_hessian_rsvd_${timestamp}.log"
Write-Host "Logging to: $logPath"

# Run and tee all output to the log (stdout+stderr)
$prevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  & $python "$pyScript" `
    --trainer $trainer `
    --checkpoint $checkpoint `
    --data $data `
    --k $k `
    --num-samples $numSamples `
    --use-labels $useLabels 2>&1 | Tee-Object -FilePath $logPath -Append
}
finally {
  $ErrorActionPreference = $prevErrorActionPreference
}

if ($LASTEXITCODE -ne 0) {
  Write-Error "RSVD Hessian run failed with exit code $LASTEXITCODE"
}

Write-Host "Completed RSVD Hessian run (train) for $trainer with N=$numSamples, k=$k. Log: $logPath"


