Param(
    [string]$TaskName = "RSVD_Hessian_RegenInception_Test",
    [switch]$Tail
)

$ErrorActionPreference = "Stop"

# Resolve paths
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $scriptRoot "run_regen_inception_hessian_rsvd_test.ps1"
if (-not (Test-Path $runner)) { throw "Runner script not found: $runner" }

# Compute a start time one minute in the future to avoid /ST warning
$startTime = (Get-Date).AddMinutes(1).ToString('HH:mm')
$action = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""

# Create/overwrite the task and run it as SYSTEM
schtasks /Create /TN $TaskName /SC ONCE /ST $startTime /TR "$action" /RU SYSTEM /F | Out-Null
schtasks /Run /TN $TaskName | Out-Null

Write-Host "Started task '$TaskName' as SYSTEM."
Write-Host "To query status: schtasks /Query /TN $TaskName /V /FO LIST"

# Tail latest log if requested
if ($Tail) {
    $logDir = Join-Path $scriptRoot "logs"
    Write-Host "Waiting for log file in: $logDir"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    for ($i = 0; $i -lt 150; $i++) { # wait up to ~5 minutes
        $log = Get-ChildItem $logDir -Filter "regen_inception_hessian_rsvd_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Desc | Select-Object -First 1
        if ($null -ne $log) {
            Write-Host "Tailing: $($log.FullName)"
            Get-Content $log.FullName -Wait
            break
        }
        Start-Sleep -Seconds 2
    }
}



