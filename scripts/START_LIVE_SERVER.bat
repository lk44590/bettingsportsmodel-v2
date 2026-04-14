@echo off
title Daily Edge Card - Live Server
cd /d "%~dp0"

echo.
echo ========================================
echo   DAILY EDGE CARD - LIVE SERVER
echo ========================================
echo.
echo This starts a server your phone can access.
echo.
echo OPTION 1: Same WiFi (FREE)
echo   - Phone and PC on same WiFi
echo   - Phone opens the shown IP address
echo.
echo OPTION 2: Internet Anywhere (FREE with ngrok)
echo   - Install ngrok from ngrok.com
echo   - Run: ngrok http 5000
echo   - Phone opens the https URL shown
echo.
echo ========================================
echo.

python refresh_server.py

pause
