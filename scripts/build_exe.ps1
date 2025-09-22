Param(
    [switch]$Clean,
    [switch]$NoSpec
)

$ErrorActionPreference = 'Stop'

$venv = ".\.venv"
$python = "$venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Virtual environment not found at $venv; falling back to system python on PATH." -ForegroundColor Yellow
    $python = "python"
}

if ($Clean) {
    if (Test-Path .\build) { Remove-Item -Recurse -Force .\build }
    if (Test-Path .\dist) { Remove-Item -Recurse -Force .\dist }
    # Preserve .spec file to reuse known-good build configuration
}

# Ensure PyInstaller is available (use python -m pip to avoid missing pip.exe issues)
Write-Host "Ensuring PyInstaller is installed using: $python -m pip" -ForegroundColor Cyan
& $python -m pip install --upgrade pyinstaller | Out-Null

# Build EXE that runs the unified CLI entrypoint
$entry = "scripts/entry_cli.py"
$exeName = "FPLWeeklyUpdater"

${env:PYTHONNOUSERSITE} = '1'
${env:PYTEST_DISABLE_PLUGIN_AUTOLOAD} = '1'
New-Item -ItemType Directory -Path (Join-Path (Get-Location) 'logs') -Force | Out-Null
$buildLog = Join-Path (Join-Path (Get-Location) 'logs') 'pyinstaller_build.log'
"Starting PyInstaller build at $(Get-Date)" | Out-File -FilePath $buildLog -Encoding utf8

# Always use CLI-based build to control hidden imports and excludes
    $pyArgs = @(
        '-m','PyInstaller',
        '--noconfirm',
        '--onedir',
        '--name',$exeName,
        '--console',
        '--log-level','DEBUG',
        '--additional-hooks-dir','.\hooks',
        '--add-data','config.yaml;.',
        '--add-data','scripts;scripts',
        '--add-data','src;src',
        # Scientific stack (fix pandas datetime C-API issues)
        '--collect-all','pandas',
        '--collect-submodules','pandas',
        '--collect-all','numpy',
        '--collect-submodules','numpy',
        '--collect-all','matplotlib',
        '--collect-submodules','matplotlib',
        '--collect-submodules','seaborn',
        '--collect-all','pydantic',
        '--collect-all','pydantic_core',
        '--collect-submodules','pydantic',
        '--collect-submodules','pydantic_settings',
        '--collect-submodules','fpl',
        '--collect-submodules','fpl_weekly_updater',
        '--collect-submodules','aiohttp',
        '--collect-all','keyring',
        '--collect-submodules','xgboost',
        '--collect-binaries','xgboost',
        '--collect-submodules','lightgbm',
        '--collect-binaries','lightgbm',
        '--exclude-module','hypothesis',
        '--exclude-module','hypothesis.extra',
        '--exclude-module','hypothesis_jsonschema',
        '--exclude-module','pytest',
        '--exclude-module','_pytest',
        '--exclude-module','pluggy',
        '--exclude-module','iniconfig',
        '--exclude-module','tomli',
        '--exclude-module','py',
        '--exclude-module','numpy.tests',
        '--exclude-module','numpy.testing',
        '--exclude-module','numpy.conftest',
        '--exclude-module','pandas.conftest',
        '--exclude-module','pandas.util._tester',
        '--exclude-module','pandas.tests',
        '--exclude-module','pandas.testing',
        '--exclude-module','pandas._testing',
        '--exclude-module','seaborn.tests',
        '--exclude-module','matplotlib.tests',
        '--exclude-module','sklearn.tests',
        '--exclude-module','scipy.tests',
        '--hidden-import','keyring.backends.Windows',
        $entry
    )
    & $python $pyArgs 2>&1 | Tee-Object -FilePath $buildLog -Append

if ($LASTEXITCODE -eq 0) {
    $out = Join-Path (Join-Path (Get-Location) 'dist') ("{0}.exe" -f $exeName)
    Write-Host "Built: $out" -ForegroundColor Green
} else {
    Write-Host ("PyInstaller build failed. Inspect log: {0}" -f $buildLog) -ForegroundColor Red
    # Tail the last 100 lines for quick view
    Get-Content $buildLog -Tail 100
    exit 1
}
