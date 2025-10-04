<#!
.SYNOPSIS
    Configure the Windows Scheduled Task that runs the FPL Weekly Updater.
.DESCRIPTION
    Removes any existing scheduled task matching the supplied name and recreates it
    so that it runs the repository's `run_weekly_update.ps1` on a weekly cadence.
    The task targets the current user account and defaults to running every Friday
    at 18:30 local time. Use the parameters to adjust the day, time, and run level.
.PARAMETER TaskName
    Name of the scheduled task to create. Defaults to "FPL Weekly Update".
.PARAMETER WeeklyDay
    Day of the week on which the task should run. Defaults to Friday.
.PARAMETER RunTime
    Local time (HH:mm) to start the task. Defaults to 18:30.
.PARAMETER RunElevated
    If specified, register the task with RunLevel Highest to request elevation.
.PARAMETER Force
    Remove an existing task without prompting for confirmation.
.EXAMPLE
    ./configure_weekly_task.ps1

    Removes any existing "FPL Weekly Update" task and recreates it for Friday 18:30.
.EXAMPLE
    ./configure_weekly_task.ps1 -WeeklyDay Sunday -RunTime 09:00 -RunElevated

    Schedules the task for Sundays at 09:00 and asks Windows to run it elevated.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param (
    [Parameter()]
    [string]
    $TaskName = "FPL Weekly Update",

    [Parameter()]
    [ValidateSet('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday')]
    [string]
    $WeeklyDay = 'Friday',

    [Parameter()]
    [ValidatePattern('^([01]?\d|2[0-3]):[0-5]\d$')]
    [string]
    $RunTime = '18:30',

    [Parameter()]
    [switch]
    $RunElevated,

    [Parameter()]
    [switch]
    $Force
)

function Get-ProjectRoot {
    [CmdletBinding()]
    param ()
    $root = Join-Path -Path $PSScriptRoot -ChildPath '..'
    return (Resolve-Path -Path $root).Path
}

function ConvertTo-TimeSpan {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string]
        $InputText
    )

    if ([string]::IsNullOrWhiteSpace($InputText)) {
        throw "RunTime cannot be empty. Provide a value in 24-hour HH:mm format."
    }

    [TimeSpan]$parsed = [TimeSpan]::Zero
    if ([TimeSpan]::TryParse($InputText, [ref]$parsed)) {
        return $parsed
    }

    throw "Invalid time value '$InputText'. Use 24-hour HH:mm format."
}

function Get-NextRunDateTime {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory)]
        [string]
        $DayOfWeek,

        [Parameter(Mandatory)]
        [TimeSpan]
        $RunTimeSpan
    )

    $targetDay = [DayOfWeek]::$DayOfWeek
    $now = Get-Date
    $candidate = $now.Date
    while ($candidate.DayOfWeek -ne $targetDay) {
        $candidate = $candidate.AddDays(1)
    }
    $scheduled = $candidate.Add($RunTimeSpan)
    if ($scheduled -le $now) {
        $scheduled = $scheduled.AddDays(7)
    }
    return $scheduled
}

try {
    $projectRoot = Get-ProjectRoot
    $runnerScript = Join-Path -Path $projectRoot -ChildPath 'run_weekly_update.ps1'
    if (-not (Test-Path -Path $runnerScript)) {
        throw "Unable to locate run_weekly_update.ps1 at '$runnerScript'."
    }

    $runTimeSpan = ConvertTo-TimeSpan -InputText $RunTime
    $firstRun = Get-NextRunDateTime -DayOfWeek $WeeklyDay -RunTimeSpan $runTimeSpan

    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        if ($Force -or $PSCmdlet.ShouldProcess($TaskName, 'Remove existing scheduled task')) {
            try {
                Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
                Write-Verbose "Removed existing task '$TaskName'."
            } catch {
                Write-Warning "Failed to remove existing task '$TaskName'. Try running PowerShell as Administrator or use a different -TaskName. Error: $($_.Exception.Message)"
                throw
            }
        } else {
            Write-Verbose "Skipping removal of existing task '$TaskName'."
            return
        }
    }

    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runnerScript`""
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $projectRoot
    $triggerTime = [datetime]::Today.Add($runTimeSpan)
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $WeeklyDay -At $triggerTime
    $settings = New-ScheduledTaskSettingsSet `
        -Compatibility Win8 `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew `
        -WakeToRun

    $runLevel = if ($RunElevated.IsPresent) { 'Highest' } else { 'Limited' }
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel $runLevel

    $registeredTask = $null
    if ($PSCmdlet.ShouldProcess("Task '$TaskName'", "Register scheduled task")) {
        try {
            $registeredTask = Register-ScheduledTask `
                -TaskName $TaskName `
                -Action $action `
                -Trigger $trigger `
                -Settings $settings `
                -Principal $principal `
                -Description "Automated run of the FPL Weekly Updater from $(Split-Path -Leaf $projectRoot)."
        } catch {
            Write-Error "Failed to register scheduled task '$TaskName'. Error: $($_.Exception.Message)"
            throw
        }
    }

    if ($registeredTask) {
        $summary = Get-ScheduledTask -TaskName $TaskName
        Write-Host "Scheduled task '$TaskName' configured." -ForegroundColor Green
        Write-Host "Next run: $($firstRun.ToString('yyyy-MM-dd HH:mm'))" -ForegroundColor Green
        Write-Host "Working directory: $projectRoot" -ForegroundColor Green
        Write-Host "Action: powershell.exe $arguments" -ForegroundColor Green
        Write-Host "Task path: $($summary.TaskPath)" -ForegroundColor Green
    }
}
catch {
    Write-Error $_
    throw
}
