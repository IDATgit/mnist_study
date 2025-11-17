# PowerShell script to run RSVD Fisher/Hessian for regen_inception 10k random images, k=500

$ErrorActionPreference = "Stop"

# Ensure we run from repo root if invoked elsewhere
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Python executable (adjust if needed)
$python = "python"

# Parameters
$trainer = "trainers.specific_trainers.regen_inception_10k_random_images"
$checkpoint = "latest"
$data = "train"
$k = 500
$numSamples = 10000

# Disable TF32 for better numerical fidelity (optional)
$env:TORCH_ALLOW_TF32_CUBLAS = "0"
$env:TORCH_ALLOW_TF32_CUDNN = "0"

# Unbuffer Python so logs stream immediately
$env:PYTHONUNBUFFERED = "1"

# Log setup
$logDir = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "regen_inception_10k_random_images_hessian_rsvd_${timestamp}.log"
Write-Host "Logging to: $logPath"

# Run and tee all output to the log (stdout+stderr)
$prevErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  & $python model_interpretation/fisher_information_hessian_rsvd.py `
    --trainer $trainer `
    --checkpoint $checkpoint `
    --data $data `
    --k $k `
    --num-samples $numSamples 2>&1 | Tee-Object -FilePath $logPath -Append
}
finally {
  $ErrorActionPreference = $prevErrorActionPreference
}

if ($LASTEXITCODE -ne 0) {
  Write-Error "RSVD Hessian run failed with exit code $LASTEXITCODE"
}

Write-Host "Completed RSVD Hessian run for $trainer with N=$numSamples, k=$k. Log: $logPath"


