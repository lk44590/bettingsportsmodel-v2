@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_daily_card.ps1"
if errorlevel 1 (
  echo.
  echo Daily card generation failed.
  pause
)
