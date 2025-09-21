# Script to create a scheduled task for the FPL Weekly Update
# Check if running as administrator, if not, restart with elevation
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Definition)`""
    Start-Process powershell -Verb RunAs -ArgumentList $arguments
    exit
}

# Script continues here if already running as admin

# Define the script path
$scriptPath = "$PSScriptRoot\run_weekly_update.ps1"

# Create the scheduled task action
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""

# Create a weekly trigger for Thursdays at 6:00 PM
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At '6:00 PM'

# Configure task settings
$settings = New-ScheduledTaskSettingsSet -DontStopOnIdleEnd -StartWhenAvailable -RunOnlyIfNetworkAvailable

# Set task to run with highest privileges
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType S4U -RunLevel Highest

# Register the scheduled task
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "FPL Weekly Update" -Description "Runs the FPL Weekly Update script" -Settings $settings -Principal $principal -Force

Write-Host "Scheduled task 'FPL Weekly Update' has been created successfully." -ForegroundColor Green
Write-Host "It is set to run every Thursday at 6:00 PM." -ForegroundColor Green
Write-Host "`nYou can verify the task in Task Scheduler (taskschd.msc)" -ForegroundColor Cyan
Write-Host "Look for 'FPL Weekly Update' in Task Scheduler Library" -ForegroundColor Cyan
