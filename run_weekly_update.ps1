# Scheduler-friendly entry point that runs the FPL Weekly Updater via Python

# Ensure execution happens from the repository root regardless of invocation directory
$scriptRoot = $PSScriptRoot
Set-Location -Path $scriptRoot

# Use virtual environment Python
$venvPython = Join-Path $scriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Virtual environment not found at $venvPython. Please run setup_venv.ps1 first."
    exit 1
}

Write-Host "Running FPL Weekly Updater via Python (venv)..."
Write-Host "Executing: $venvPython -m fpl_weekly_updater.cli weekly --quiet"

# Set logging to INFO for debugging (change to WARNING for scheduled runs)
$env:LOG_LEVEL = "INFO"

# Run browser in non-headless mode for reliability (headless has login issues)
$env:FPL_BROWSER_HEADLESS = "false"

# Force non-interactive mode to prevent prompts
$env:PYTHONUNBUFFERED = "1"

# Redirect stdin to prevent interactive prompts
$null | & $venvPython -m fpl_weekly_updater.cli weekly --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "FPL optimization pipeline failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}

Write-Host "FPL Weekly Updater completed successfully."
