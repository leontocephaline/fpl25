$ErrorActionPreference = 'Stop'

# Clean out directory
if (Test-Path .\out) { Remove-Item -Recurse -Force .\out }

$proj = (Get-Location).Path
$py = ".\.venv\Scripts\python.exe"

# Build PyInstaller args in an array to avoid quoting issues
$piArgs = @(
  "-m","PyInstaller",
  "--noconfirm","--onedir","--name","FPLWeeklyUpdater","--console",
  "--distpath",".\out\dist","--workpath",".\out\build",
  "--additional-hooks-dir",".\hooks",
  "--add-data", ($proj + "\config.yaml;."),
  "--add-data", ($proj + "\scripts;scripts"),
  "--add-data", ($proj + "\src;src"),
  "--collect-all","xgboost",
  "--collect-binaries","xgboost",
  "--collect-all","lightgbm",
  "--collect-binaries","lightgbm",
  "scripts/entry_cli.py"
)

& $py @piArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

Write-Host "Build complete. Verifying xgboost assets..."
$exePath = ".\out\dist\FPLWeeklyUpdater\FPLWeeklyUpdater.exe"
$verPath = ".\out\dist\FPLWeeklyUpdater\_internal\xgboost\VERSION"
$dllPath = ".\out\dist\FPLWeeklyUpdater\_internal\xgboost\lib\xgboost.dll"
Write-Host ("EXE exists: {0}" -f (Test-Path $exePath))
Write-Host ("VERSION exists: {0}" -f (Test-Path $verPath))
Write-Host ("DLL exists: {0}" -f (Test-Path $dllPath))

Write-Host "Running weekly (--no-news)..."
& $exePath weekly --no-news --report-dir reports
$weeklyExit = $LASTEXITCODE
Write-Host ("Weekly exit code: {0}" -f $weeklyExit)

Write-Host "Recent reports:"
Get-ChildItem .\reports | Sort-Object LastWriteTime -Descending | Select-Object -First 10 Name,LastWriteTime,Length

if ($weeklyExit -ne 0) { exit 1 }
