param(
    [string]$InstallDir,
    [switch]$NoShortcuts,
    [switch]$NoSchedule
)

# Installs the FPL Weekly Updater portable build into a production folder.
# - Creates folder structure (bin, data, reports, logs)
# - Copies the EXE (or onedir) and configs
# - Prompts for FPL credentials (stores password in Windows Credential Manager)
# - Creates handy run_* .bat shortcuts
#
# Usage:
#   pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1 -InstallDir "C:\Users\<you>\Apps\FPLWeeklyUpdater"
#   pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1   # uses %LOCALAPPDATA%\FPLWeeklyUpdater

$ErrorActionPreference = 'Stop'

function Resolve-DefaultInstallDir {
    if ([string]::IsNullOrWhiteSpace($InstallDir)) {
        $base = $env:LOCALAPPDATA
        if ([string]::IsNullOrWhiteSpace($base)) { $base = "$HOME\AppData\Local" }
        return Join-Path $base 'FPLWeeklyUpdater'
    }
    return $InstallDir
}

$target = Resolve-DefaultInstallDir
Write-Host "Installing to: $target" -ForegroundColor Cyan

# Create folders
$binDir     = Join-Path $target 'bin'
$dataDir    = Join-Path $target 'data'
$reportsDir = Join-Path $target 'reports'
$logsDir    = Join-Path $target 'logs'

$null = New-Item -ItemType Directory -Force -Path $target,$binDir,$dataDir,$reportsDir,$logsDir

# Detect build outputs
$root = Get-Location
$onedirSrc = Join-Path $root 'dist\FPLWeeklyUpdater'
$onefileSrc = Join-Path $root 'dist\FPLWeeklyUpdater.exe'

if (Test-Path $onedirSrc) {
    Write-Host "Copying onedir build..." -ForegroundColor Yellow
    robocopy $onedirSrc $binDir /MIR /NFL /NDL /NJH /NJS | Out-Null
} elseif (Test-Path $onefileSrc) {
    Write-Host "Copying single EXE..." -ForegroundColor Yellow
    Copy-Item -LiteralPath $onefileSrc -Destination (Join-Path $binDir 'FPLWeeklyUpdater.exe') -Force
} else {
    throw "Could not find build outputs in .\\dist"
}

# Copy configs/docs if present
foreach ($p in @('config.yaml','EXECUTABLE_README.md','docs\env.example')) {
    if (Test-Path $p) {
        $dest = Join-Path $target (Split-Path $p -Leaf)
        Copy-Item -LiteralPath $p -Destination $dest -Force
    }
}

# Create .env if missing and prompt for values
$envPath = Join-Path $target '.env'
if (-not (Test-Path $envPath)) {
    Write-Host "Creating .env..." -ForegroundColor Yellow
    $envLines = @()

    # Try to seed from env.example if present
    $example = Join-Path $target 'env.example'
    if (Test-Path $example) {
        $envLines = Get-Content $example
    }

    $email = Read-Host 'Enter your FPL email address'
    $teamId = Read-Host 'Enter your FPL team ID'
    $pplx = Read-Host 'Enter your Perplexity API key (or leave blank)'

    # Persist values
    $kv = @{
        'FPL_EMAIL' = $email
        'FPL_TEAM_ID' = $teamId
        'PERPLEXITY_API_KEY' = $pplx
        'FPL_BROWSER' = 'edge'  # edge|chrome; change if needed
        'USE_BROWSER_LOGIN' = 'true'
    }

    foreach ($k in $kv.Keys) {
        $existing = $envLines | Where-Object { $_ -match "^$k=" }
        if (-not $existing) { $envLines += ("{0}={1}" -f $k, $kv[$k]) }
    }

    $envLines | Set-Content -Path $envPath -Encoding UTF8
}

# Store password in Windows Credential Manager (native Windows)
$onWine = [bool]([Environment]::GetEnvironmentVariable('WINELOADERNOEXEC'))
if (-not $onWine) {
    Write-Host "Storing password securely in Windows Credential Manager..." -ForegroundColor Yellow
    $pw = Read-Host -AsSecureString 'Enter your FPL password (hidden)'
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($pw))
    $email = (Select-String -Path $envPath -Pattern '^FPL_EMAIL=').Line -replace '^FPL_EMAIL=',''
    if ([string]::IsNullOrWhiteSpace($email)) { throw 'FPL_EMAIL missing from .env' }
    $service = 'fpl-weekly-updater'
    # cmdkey writes to the same vault keyring uses
    cmdkey /generic:$service /user:$email /pass:$plain | Out-Null
    Write-Host 'Password saved to Credential Manager.' -ForegroundColor Green
} else {
    Write-Host "Wine detected: storing password in .env instead (less secure)." -ForegroundColor Yellow
    $pw = Read-Host -AsSecureString 'Enter your FPL password (hidden)'
    $plain = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($pw))
    Add-Content -Path $envPath -Value ("FPL_PASSWORD={0}" -f $plain)
}

# Force CPU inference on Wine (ONNX/DirectML not available)
if ($onWine) {
    try {
        $cfgPath = Join-Path $target 'config.yaml'
        if (Test-Path $cfgPath) {
            $text = Get-Content $cfgPath -Raw
            if ($text -match 'inference_backend:') {
                $text = $text -replace 'inference_backend:\s*\w+', 'inference_backend: cpu'
            } else {
                # Append under ml: block if present; else append at end
                if ($text -match "(?ms)^ml:\s*") {
                    $text = $text -replace '(?ms)^(ml:\s*(?:.*\R)+?)', "$1  inference_backend: cpu`r`n"
                } else {
                    $text += "`r`nml:`r`n  inference_backend: cpu`r`n"
                }
            }
            Set-Content -Path $cfgPath -Value $text -Encoding UTF8
            Write-Host "Configured inference_backend: cpu for Wine." -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Warning: could not adjust inference_backend for Wine: $($_.Exception.Message)" -ForegroundColor DarkYellow
    }
}

# Write run helpers
$runWeekly = @(
    '@echo off',
    'setlocal',
    'cd /d "%~dp0"',
    'set APPDIR=%~dp0',
    'set BINDIR=%APPDIR%bin',
    'set EXE=%BINDIR%\FPLWeeklyUpdater.exe',
    'if not exist "%EXE%" (',
    '  echo Could not find FPLWeeklyUpdater.exe in %BINDIR%',
    '  exit /b 1',
    ')',
    'echo Running weekly updater...',
    '"%EXE%"'
)
$runAppendix = @(
    '@echo off',
    'setlocal',
    'cd /d "%~dp0"',
    'set APPDIR=%~dp0',
    'set BINDIR=%APPDIR%bin',
    'set EXE=%BINDIR%\FPLWeeklyUpdater.exe',
    'if not exist "%EXE%" (',
    '  echo Could not find FPLWeeklyUpdater.exe in %BINDIR%',
    '  exit /b 1',
    ')',
    'echo Running backtest appendix...',
    '"%EXE%" appendix --actual-data "%APPDIR%reports\actuals_2024-25.csv" --predictions-dir "%APPDIR%data\backtest" --output-dir "%APPDIR%reports"'
)

$runWeekly | Set-Content -Path (Join-Path $target 'run_weekly_update.bat') -Encoding ASCII
$runAppendix | Set-Content -Path (Join-Path $target 'run_appendix.bat') -Encoding ASCII

Write-Host "\nInstallation complete." -ForegroundColor Green
Write-Host ("Location: {0}" -f $target)
Write-Host "- Use run_weekly_update.bat to generate the PDF report"
Write-Host "- Use run_appendix.bat to generate the backtest appendix"

if (-not $NoShortcuts) {
    Write-Host "\nTip: You can create desktop shortcuts manually by right-clicking the .bat files."
}

if (-not $NoSchedule) {
    Write-Host "\nOptional: Create a scheduled task using create_scheduled_task.ps1 with your chosen cadence."
}
