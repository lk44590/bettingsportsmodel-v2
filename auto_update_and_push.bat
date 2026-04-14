@echo off
cd /d "C:\Users\sales\OneDrive\Desktop\X accounts\ncaa-baseball-hub\Betting model upgraded"

echo Generating daily card...
python generate_daily_card.py

echo Adding files to git...
git add output/

echo Committing changes...
git commit -m "Auto update - %date% %time%"

echo Pushing to GitHub...
git push

echo Done!
