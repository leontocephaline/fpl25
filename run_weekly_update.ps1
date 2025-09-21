# Script to set up and run the FPL weekly updater with the correct Python environment

Param(
    [string]$TeamDataJsonPath = ".\team_data.json",
    [int]$FreeTransfers = 1,
    [double]$Bank = 0.0,
    [switch]$Retrain,
    [switch]$EnableBacktest
)

# Set the working directory to the script's location
$scriptPath = $PSScriptRoot
Set-Location -Path $scriptPath

# Define paths
$venvPath = ".\.venv"
$pythonPath = "$venvPath\Scripts\python.exe"
$pipPath = "$venvPath\Scripts\pip.exe"
$requirementsPath = ".\requirements.txt"

# Check if virtual environment exists, create if it doesn't
if (-not (Test-Path -Path $pythonPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
    
    # Activate the virtual environment if activation script exists
    if (Test-Path "$venvPath\Scripts\Activate.ps1") {
        & "$venvPath\Scripts\Activate.ps1"
    } else {
        Write-Host "Activation script not found; proceeding without activation." -ForegroundColor Yellow
    }
    
    # Upgrade pip
    & $pythonPath -m pip install --upgrade pip
    
    # Install requirements
    if (Test-Path -Path $requirementsPath) {
        & $pipPath install -r $requirementsPath
    } else {
        Write-Warning "requirements.txt not found. Please install dependencies manually."
    }
} else {
    # Activate the existing virtual environment if activation script exists
    if (Test-Path "$venvPath\Scripts\Activate.ps1") {
        & "$venvPath\Scripts\Activate.ps1"
    } else {
        Write-Host "Activation script not found; proceeding without activation." -ForegroundColor Yellow
    }
}

# Step 2: Run the Perplexity-enabled FPL Weekly Updater
Write-Host "Running Perplexity-enabled FPL Weekly Updater..."

# Execute the weekly updater via unified CLI (writes fpl_weekly_update_*.pdf to Desktop by default)
Write-Host "Executing: python -m fpl_weekly_updater weekly --quiet"
# Ensure quiet logging for this process
$env:LOG_LEVEL = "WARNING"
& $pythonPath -m fpl_weekly_updater weekly --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "FPL optimization pipeline failed."
    exit 1
}

Write-Host "FPL Weekly Updater completed (check Desktop for 'fpl_weekly_update_*.pdf')."

# Step 2.5: Confirm latest weekly report on Desktop
try {
    $desktopPath = [Environment]::GetFolderPath('Desktop')
    $latestPplxWeekly = Get-ChildItem -Path $desktopPath -Filter 'fpl_weekly_update_*.pdf' -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $latestPplxWeekly) {
        Write-Host "Latest Desktop weekly report: $($latestPplxWeekly.FullName)"
    } else {
        Write-Warning "No 'fpl_weekly_update_*.pdf' found on Desktop yet."
    }
} catch {
    Write-Warning "Failed to list Desktop weekly report: $_"
}

# Step 3: Optional — Run historical backtest (disabled by default)
if ($EnableBacktest) {
    Write-Host "Running Historical Backtest and generating PDF report..."

    $backtestArgs = @(
        "scripts/run_historical_backtest.py",
        "--generate-pdf"
    )

    Write-Host "Executing: python $($backtestArgs -join ' ')"
    & $pythonPath $backtestArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Error "Historical Backtest failed."
        exit 1
    }

    Write-Host "Historical Backtest completed successfully. PDF saved to Desktop by default."
} else {
    Write-Host "Skipping Historical Backtest (use -EnableBacktest to run)."
}
