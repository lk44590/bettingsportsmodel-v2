# Setup Windows Scheduled Task to run daily card update daily at 10:00 AM Eastern

$taskName = "Daily Betting Card Update"
$scriptPath = "C:\Users\sales\OneDrive\Desktop\X accounts\ncaa-baseball-hub\Betting model upgraded\auto_update_and_push.bat"

# Remove existing task if it exists
try {
    Unregister-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Write-Host "Removed existing task" -ForegroundColor Green
} catch {
    # Task doesn't exist, continue
}

# Create new scheduled task
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Daily -At 10:00AM
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -User "System" -RunLevel Highest

Write-Host "Scheduled task created successfully!" -ForegroundColor Green
Write-Host "Task will run daily at 10:00 AM Eastern" -ForegroundColor Cyan
Write-Host "To run manually: $scriptPath" -ForegroundColor Yellow
