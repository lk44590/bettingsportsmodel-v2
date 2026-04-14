param(
  [string]$Date = (Get-Date -Format 'yyyyMMdd'),
  [int]$Top = 0,
  [switch]$NoOpen,
  [switch]$ForceProviderSync
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Generating daily card for $Date..." -ForegroundColor Green
$arguments = @('-B', 'generate_daily_card.py', '--date', $Date)
if ($Top -gt 0) {
  $arguments += @('--top', $Top)
}
if ($ForceProviderSync) {
  $arguments += '--force-provider-sync'
}
python @arguments
if ($LASTEXITCODE -ne 0) {
  throw "Daily card generation failed with exit code $LASTEXITCODE."
}

$providerReportPath = Join-Path $scriptDir 'output\provider-sync-latest.json'
if (Test-Path $providerReportPath) {
  try {
    $providerReport = Get-Content $providerReportPath -Raw | ConvertFrom-Json
    if (-not $providerReport.synced) {
      $warningText = @($providerReport.warnings) -join '; '
      Write-Host "Provider feed not synced: $warningText" -ForegroundColor Yellow
      Write-Host "Run 'Set Live Odds API Key.cmd' to connect the live all-sports feed." -ForegroundColor Yellow
    }
  } catch {
    Write-Host "Could not read provider sync report." -ForegroundColor Yellow
  }
}

$reportPath = Join-Path $scriptDir 'output\daily-card-latest.html'
if (-not $NoOpen) {
  Start-Process $reportPath
}

Write-Host "Done. Report: $reportPath" -ForegroundColor Green
