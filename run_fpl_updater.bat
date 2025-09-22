@echo off
REM FPL Weekly Updater Launcher
REM This batch file runs the FPL Weekly Updater executable

echo Starting FPL Weekly Updater...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if executable exists
if not exist "dist\FPLWeeklyUpdater.exe" (
    echo ❌ Error: FPLWeeklyUpdater.exe not found in dist\ folder
    echo Please run setup.bat first to configure the application.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Run the executable
cd dist
FPLWeeklyUpdater.exe

REM Check if execution was successful
if errorlevel 1 (
    echo.
    echo ❌ Application failed with error code %errorlevel%
    echo Check fpl_weekly_update.log for details.
) else (
    echo.
    echo ✅ Application completed successfully!
    echo Check your Desktop for the generated report.
)

echo.
echo Press any key to exit...
pause >nul
