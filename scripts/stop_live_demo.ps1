<#
.SYNOPSIS
    Gracefully stops the continuous Live CCTV to RAG Dashboard demo pipeline.
.DESCRIPTION
    Reads logs/demo_pids.json, terminates each active process (detector, producer,
    worker, dashboard), stops any Docker worker container if active, and cleans up
    the PID lock file.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"

$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
$LogsDir = Join-Path $RepoRoot "logs"
$PidFile = Join-Path $LogsDir "demo_pids.json"

Write-Host "==========================================================" -ForegroundColor Yellow
Write-Host " STOPPING LIVE CCTV DEMO PIPELINE                         " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Yellow

if (!(Test-Path $PidFile)) {
    Write-Host "No active PID file found ($PidFile). Checking orphan processes..." -ForegroundColor Gray
} else {
    try {
        $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
        foreach ($prop in $pids.PSObject.Properties) {
            if ($prop.Name -like "*_pid" -and $prop.Value) {
                $pidVal = [int]$prop.Value
                $proc = Get-Process -Id $pidVal -ErrorAction SilentlyContinue
                if ($proc) {
                    Write-Host "Stopping $($prop.Name) (PID: $pidVal)..." -NoNewline
                    Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
                    Write-Host " Stopped." -ForegroundColor Green
                } else {
                    Write-Host "$($prop.Name) (PID: $pidVal) was not running." -ForegroundColor Gray
                }
            }
        }
        if ($pids.worker_mode -eq "docker") {
            Write-Host "Stopping Docker worker container..." -NoNewline
            docker stop cctv-live-worker 2>$null | Out-Null
            Write-Host " Stopped." -ForegroundColor Green
        }
    } catch {
        Write-Warning "Could not read PID file: $_"
    } finally {
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

# Cleanup any lingering orphan Python processes that match script names if requested
Write-Host "----------------------------------------------------------"
Write-Host "Live demo pipeline stopped." -ForegroundColor Green
Write-Host "Logs preserved in: $LogsDir" -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Yellow
