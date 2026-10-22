<#
.SYNOPSIS
    Starts the full continuous Live CCTV -> AWS S3/SQS -> Worker -> RAG Dashboard pipeline.
.DESCRIPTION
    Launches:
      1. vehicle_counting.py (live detector from Caltrans HLS)
      2. live_s3_sqs_producer_v2.py (tails JSONL and batches to S3 + SQS)
      3. worker.py (continuous long-polling SQS worker writing to SQLite)
      4. app.py (Streamlit dashboard)
    Enforces strict test queue and bucket guards to prevent routing to production.
.PARAMETER Headless
    Run vehicle detector without OpenCV GUI window.
.PARAMETER UseDocker
    Run worker inside the Docker container rather than locally on host.
#>

[CmdletBinding()]
param(
    [switch]$Headless,
    [switch]$UseDocker
)

$ErrorActionPreference = "Stop"

# ---------- PREFLIGHT SAFETY GUARDS ----------
$AWS_REGION     = "ap-south-1"
$S3_BUCKET      = "generalized-etl-engine-2026-0001"
$S3_PREFIX      = "cctv/test-batches/"
$TEST_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/292981482205/traffic-events-queue"
$PROD_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/292981482205/cctv-processing-queue"
$SQLITE_DB      = "cctv_live_test.sqlite3"

if ($TEST_QUEUE_URL -eq $PROD_QUEUE_URL) {
    Write-Error "CRITICAL SAFETY ABORT: Test queue cannot be production queue!"
    exit 1
}

$RepoRoot = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $RepoRoot

$LogsDir = Join-Path $RepoRoot "logs"
if (!(Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

$PidFile = Join-Path $LogsDir "demo_pids.json"
if (Test-Path $PidFile) {
    try {
        $pids = Get-Content $PidFile -Raw | ConvertFrom-Json
        $alive = @()
        foreach ($p in ($pids.PSObject.Properties | Where-Object { $_.Name -like "*_pid" })) {
            if ($p.Value -and (Get-Process -Id $p.Value -ErrorAction SilentlyContinue)) {
                $alive += "$($p.Name): $($p.Value)"
            }
        }
        if ($alive.Count -gt 0) {
            Write-Warning "Demo is already running with active processes: $($alive -join ', ')"
            Write-Warning "Run .\scripts\stop_live_demo.ps1 before starting a new instance."
            exit 1
        }
    } catch {
        # Corrupted PID file, proceed after removing
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " STARTING CONTINUOUS LIVE CCTV TO RAG DASHBOARD DEMO      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "AWS Region:         $AWS_REGION"
Write-Host "Target S3 Bucket:   $S3_BUCKET"
Write-Host "Target S3 Prefix:   $S3_PREFIX"
Write-Host "Target SQS Queue:   $TEST_QUEUE_URL"
Write-Host "Storage Backend:    SQLite ($SQLITE_DB)"
Write-Host "Logs Directory:     $LogsDir"
Write-Host "----------------------------------------------------------"

# Set global process environment for child inheritance
$env:AWS_REGION                 = $AWS_REGION
$env:S3_BUCKET_NAME             = $S3_BUCKET
$env:CCTV_S3_BUCKET             = $S3_BUCKET
$env:CCTV_S3_PREFIX             = $S3_PREFIX
$env:CCTV_QUEUE_URL             = $TEST_QUEUE_URL
$env:CCTV_LOG_FILE              = "live-cctv-cloud/output/cctv_events_v2.jsonl"
$env:CCTV_SPOOL_DIR             = "live-cctv-cloud/output/pending_uploads_v2"
$env:CCTV_STATE_FILE            = "live-cctv-cloud/output/live_s3_sqs_state_v2.json"
$env:STORAGE_BACKEND            = "sqlite"
$env:SQLITE_PATH                = $SQLITE_DB
$env:WORKER_DATA_DIR            = "live-cctv-cloud/data"
$env:WORKER_MAX_BATCHES         = "0"
$env:WORKER_EXIT_ON_EMPTY       = "0"
$env:WORKER_POLL_SECONDS        = "5"
$env:EMBEDDING_MODE             = "disabled"

if ($Headless) {
    $env:CCTV_HEADLESS = "1"
    Write-Host "Detector Mode:      Headless (no OpenCV GUI window)"
} else {
    $env:CCTV_HEADLESS = "0"
    Write-Host "Detector Mode:      Interactive GUI (Press Q on window to stop detector)"
}

$ProcMap = @{}

# 1. Start Vehicle Detector
Write-Host "[1/4] Launching Live Vehicle Detector..." -NoNewline
$detectorLog = Join-Path $LogsDir "detector.log"
$detectorErr = Join-Path $LogsDir "detector_err.log"
$detectorProc = Start-Process -FilePath "py" `
    -ArgumentList "-3.14", "live-cctv-cloud/vehicle_counting.py" `
    -RedirectStandardOutput $detectorLog `
    -RedirectStandardError $detectorErr `
    -NoNewWindow -PassThru
$ProcMap["detector_pid"] = $detectorProc.Id
Write-Host " PID: $($detectorProc.Id)" -ForegroundColor Green

# 2. Start Live S3/SQS Producer
Write-Host "[2/4] Launching Event Producer..." -NoNewline
$producerLog = Join-Path $LogsDir "producer.log"
$producerErr = Join-Path $LogsDir "producer_err.log"
$producerProc = Start-Process -FilePath "py" `
    -ArgumentList "-3.14", "live-cctv-cloud/live_s3_sqs_producer_v2.py" `
    -RedirectStandardOutput $producerLog `
    -RedirectStandardError $producerErr `
    -NoNewWindow -PassThru
$ProcMap["producer_pid"] = $producerProc.Id
Write-Host " PID: $($producerProc.Id)" -ForegroundColor Green

# 3. Start Continuous Worker
Write-Host "[3/4] Launching Continuous SQS Worker..." -NoNewline
$workerLog = Join-Path $LogsDir "worker.log"
$workerErr = Join-Path $LogsDir "worker_err.log"

if ($UseDocker) {
    Write-Host " (Docker Container)..." -NoNewline
    # Remove existing stopped container with same name if any
    docker rm -f cctv-live-worker 2>$null | Out-Null
    $dockerCmd = "run --rm --name cctv-live-worker -v `"${RepoRoot}:/app/host`" " +
                 "-e AWS_REGION=$AWS_REGION " +
                 "-e CCTV_QUEUE_URL=$TEST_QUEUE_URL " +
                 "-e CCTV_S3_BUCKET=$S3_BUCKET " +
                 "-e STORAGE_BACKEND=sqlite " +
                 "-e SQLITE_PATH=/app/host/$SQLITE_DB " +
                 "-e WORKER_DATA_DIR=/app/host/live-cctv-cloud/data " +
                 "-e WORKER_MAX_BATCHES=0 " +
                 "-e WORKER_EXIT_ON_EMPTY=0 " +
                 "-e WORKER_POLL_SECONDS=5 " +
                 "-e EMBEDDING_MODE=disabled " +
                 "cctv-worker:latest"
    $workerProc = Start-Process -FilePath "docker" `
        -ArgumentList $dockerCmd `
        -RedirectStandardOutput $workerLog `
        -RedirectStandardError $workerErr `
        -NoNewWindow -PassThru
    $ProcMap["worker_pid"] = $workerProc.Id
    $ProcMap["worker_mode"] = "docker"
} else {
    $workerProc = Start-Process -FilePath "py" `
        -ArgumentList "-3.14", "live-cctv-cloud/worker.py" `
        -RedirectStandardOutput $workerLog `
        -RedirectStandardError $workerErr `
        -NoNewWindow -PassThru
    $ProcMap["worker_pid"] = $workerProc.Id
    $ProcMap["worker_mode"] = "host"
}
Write-Host " PID: $($workerProc.Id)" -ForegroundColor Green

# 4. Start Streamlit Dashboard
Write-Host "[4/4] Launching Streamlit Dashboard..." -NoNewline
$dashLog = Join-Path $LogsDir "dashboard.log"
$dashErr = Join-Path $LogsDir "dashboard_err.log"
$dashProc = Start-Process -FilePath "streamlit" `
    -ArgumentList "run", "app.py", "--server.headless", "true" `
    -RedirectStandardOutput $dashLog `
    -RedirectStandardError $dashErr `
    -NoNewWindow -PassThru
$ProcMap["dashboard_pid"] = $dashProc.Id
Write-Host " PID: $($dashProc.Id)" -ForegroundColor Green

# Record PIDs to state file
$ProcMap["started_at_utc"] = (Get-Date).ToUniversalTime().ToString("o")
$ProcMap | ConvertTo-Json -Indent 2 | Set-Content $PidFile -Encoding UTF8

Write-Host "----------------------------------------------------------"
Write-Host "All processes started successfully!" -ForegroundColor Green
Write-Host "Dashboard URL:      http://localhost:8501" -ForegroundColor Yellow
Write-Host "To monitor logs:    Get-Content logs/worker.log -Wait" -ForegroundColor Gray
Write-Host "To stop demo:       .\scripts\stop_live_demo.ps1" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan
