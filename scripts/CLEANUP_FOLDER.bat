@echo off
title Clean Up Betting Model Folder
echo.
echo ========================================
echo   CLEAN UP - Remove Obsolete Files
echo ========================================
echo.
echo This will remove duplicate and unused files
echo while keeping what you actually need.
echo.
echo KEEPING: Core model + Working mobile file
echo REMOVING: Duplicates, old versions, failed attempts
echo.
pause
echo.

cd /d "%~dp0"

echo [1/5] Removing old duplicate HTML files...
del /Q "COMPLETE_PHONE_BETTING.html" 2>nul
del /Q "DAILY_EDGE_CARD_MOBILE.html" 2>nul
del /Q "FINAL_WORKING_MOBILE_MODEL.html" 2>nul
del /Q "MONEY_MAKER_LIVE.html" 2>nul
del /Q "MONEY_MAKER_MOBILE.html" 2>nul
del /Q "MONEY_MAKER_STATIC.html" 2>nul
del /Q "RUN_DAILY_CARD.html" 2>nul
del /Q "SIMPLE_MOBILE_CARD.html" 2>nul
del /Q "SIMPLE_MOBILE_CARD_WITH_HISTORY.html" 2>nul
del /Q "ULTIMATE_MONEY_MAKER.html" 2>nul
del /Q "WORKING_MOBILE_BETTING_MODEL.html" 2>nul
del /Q "PHONE - Last Desktop Card.html" 2>nul
del /Q "GET_ON_PHONE.html" 2>nul
del /Q "index.html" 2>nul
echo   Done.

echo [2/5] Removing old text files...
del /Q "DAILY_CARD_PHONE.txt" 2>nul
del /Q "PHONE - Last Desktop Card.txt" 2>nul
del /Q "BETTING_SYSTEM.md" 2>nul
del /Q "FINAL_PHONE_INSTRUCTIONS.md" 2>nul
echo   Done.

echo [3/5] Removing obsolete server and deploy files...
del /Q "auto_deploy.py" 2>nul
del /Q "auto_update_and_deploy.py" 2>nul
del /Q "build_netlify.py" 2>nul
del /Q "desktop-identical-server.py" 2>nul
del /Q "mobile-server.py" 2>nul
del /Q "phone_server.py" 2>nul
del /Q "qr_generator.py" 2>nul
del /Q "GET_LINK_FOR_PHONE.py" 2>nul
del /Q "UPDATE_DAILY_CARD.py" 2>nul
del /Q "UPDATE_PHONE.bat" 2>nul
del /Q "DEPLOY_INSTRUCTIONS.html" 2>nul
del /Q "DEPLOY_TO_NETLIFY.bat" 2>nul
del /Q "DEPLOY_TO_NETLIFY.cmd" 2>nul
del /Q "DEPLOY_TO_NETLIFY_LOCAL.bat" 2>nul
del /Q "DEPLOY_TO_PYTHONANYWHERE.html" 2>nul
del /Q "SETUP_AUTO_DEPLOY.bat" 2>nul
del /Q "SETUP_AUTO_UPDATE.bat" 2>nul
del /Q "START_MOBILE_MODEL.bat" 2>nul
del /Q "START_PHONE_SERVER.bat" 2>nul
del /Q "START_WITH_NGROK.bat" 2>nul
del /Q "CREATE_CLOUD_PACKAGE.bat" 2>nul
del /Q "DAILY_EDGE_CLOUD.zip" 2>nul
del /Q "netlify.toml" 2>nul
echo   Done.

echo [4/5] Removing desktop app files (if not using)...
del /Q "Daily Betting App.hta" 2>nul
del /Q "daily_betting_app.ps1" 2>nul
del /Q "launch_daily_betting_app.vbs" 2>nul
del /Q "Open Daily Betting App.cmd" 2>nul
del /Q "betting-lab.html" 2>nul
del /Q "betting-lab.css" 2>nul
del /Q "betting-lab.js" 2>nul
del /Q "phone-live-app.js" 2>nul
echo   Done.

echo [5/5] Removing research/optional files...
del /Q "backfill_model_history.py" 2>nul
del /Q "research_candidate_archive.py" 2>nul
del /Q "Backfill Model History.cmd" 2>nul
del /Q "run_backfill_model_history.ps1" 2>nul
del /Q "Candidate Research.cmd" 2>nul
del /Q "run_candidate_research.ps1" 2>nul
del /Q "Set Live Odds API Key.cmd" 2>nul
del /Q "set_live_odds_api_key.ps1" 2>nul
echo   Done.

echo [6/6] Removing empty folders...
if exist "CLOUD_DEPLOY" rmdir /S /Q "CLOUD_DEPLOY" 2>nul
if exist "__pycache__" rmdir /S /Q "__pycache__" 2>nul
if exist "dist" rmdir /S /Q "dist" 2>nul
echo   Done.

echo.
echo ========================================
echo   CLEANUP COMPLETE!
echo ========================================
echo.
echo REMAINING FILES:
echo.
echo CORE MODEL:
echo   - Run Daily Card.cmd          (generates picks)
echo   - run_daily_card.ps1          (backend)
echo   - generate_daily_card.py      (main model)
echo   - edge_model_config.json      (settings)
echo.
echo MOBILE:
echo   - MOBILE_DAILY_CARD.html      (working mobile file)
echo   - refresh_server.py           (live refresh server)
echo   - START_LIVE_SERVER.bat       (start server)
echo   - PHONE_LIVE_GUIDE.html       (instructions)
echo.
echo DATA:
echo   - data/                       (model data)
echo   - tracking/                   (bet tracking)
echo   - output/                     (generated cards)
echo.
pause
