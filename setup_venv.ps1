# Script to set up the Python virtual environment

# Set the working directory to the script's location
$scriptPath = $PSScriptRoot
Set-Location -Path $scriptPath

# Define paths
$venvPath = ".\.venv"
$pythonPath = "$venvPath\Scripts\python.exe"
$pipPath = "$venvPath\Scripts\pip.exe"
$requirementsPath = ".\requirements.txt"

# Remove existing virtual environment if it exists
if (Test-Path -Path $venvPath) {
    Write-Host "Removing existing virtual environment..."
    Remove-Item -Path $venvPath -Recurse -Force
}

# Create new virtual environment
Write-Host "Creating new virtual environment..."
python -m venv $venvPath

# Activate the virtual environment
& "$venvPath\Scripts\Activate.ps1"

# Upgrade pip
& $pythonPath -m pip install --upgrade pip

# Install requirements
if (Test-Path -Path $requirementsPath) {
    Write-Host "Installing requirements..."
    & $pipPath install -r $requirementsPath
} else {
    Write-Warning "requirements.txt not found. Please install dependencies manually."
}

# Install package in development mode
Write-Host "Installing package in development mode..."
& $pipPath install -e .

Write-Host "`nVirtual environment setup complete!"
Write-Host "You can now run the weekly update using: .\.venv\Scripts\python -m fpl_weekly_updater.main"

# Keep the console open to see the output
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
