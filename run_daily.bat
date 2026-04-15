@echo off
cd /d "C:\Users\sales\OneDrive\Desktop\X accounts\ncaa-baseball-hub\Betting model upgraded"

echo Generating daily card...
python generate_daily_card.py --top 5

echo Opening daily card in browser...
start "" "output\daily-card-latest.html"

echo Done!
pause
