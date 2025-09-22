@echo off
REM FPL Weekly Updater Setup Script
REM This batch file runs the Python setup script for first-time configuration

echo Starting FPL Weekly Updater Setup...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ and try again.
    pause
    exit /b 1
)

REM Run the setup script
python setup_fpl_updater.py

REM Check if setup was successful
if errorlevel 1 (
    echo.
    echo ❌ Setup failed. Check the error messages above.
) else (
    echo.
    echo ✅ Setup completed successfully!
    echo You can now run the FPL Weekly Updater executable.
)

echo.
echo Press any key to exit...
pause >nul
