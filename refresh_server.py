#!/usr/bin/env python3
"""
Live Refresh Server
Phone can refresh picks anytime by visiting URL
"""
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
HTML_FILE = OUTPUT_DIR / "daily-card-latest.html"
JSON_FILE = OUTPUT_DIR / "daily-card-latest.json"

def generate_now():
    """Run the model and return success status."""
    try:
        from generate_daily_card import main as gen_main
        import argparse
        args = argparse.Namespace(
            date=datetime.now().strftime('%Y%m%d'),
            top=0,
            force_provider_sync=False
        )
        gen_main()
        return True, datetime.now().isoformat()
    except Exception as e:
        return False, str(e)

def load_picks_data():
    """Load current picks from JSON or parse HTML."""
    try:
        if JSON_FILE.exists():
            return json.loads(JSON_FILE.read_text())
    except:
        pass
    return {"last_updated": "Never", "picks": []}

@app.route('/')
def home():
    """Main page - shows picks with refresh button."""
    data = load_picks_data()
    
    html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Edge Card - Live</title>
    <style>
        :root {
            --bg: #f4ecdd; --paper: #fffaf1; --ink: #192218;
            --muted: #687463; --line: rgba(25, 34, 24, 0.1);
            --green: #25523a; --gold: #b6822f; --red: #a34c39;
            --shadow: 0 18px 42px rgba(60, 44, 15, 0.12);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0; font-family: 'Segoe UI', Arial, sans-serif;
            color: var(--ink); background: var(--bg);
            background: radial-gradient(circle at top left, rgba(182,130,47,0.16), transparent 28%),
                        radial-gradient(circle at bottom right, rgba(37,82,58,0.12), transparent 30%),
                        linear-gradient(180deg, #f8f2e7, var(--bg));
        }
        main { max-width: 1180px; margin: 0 auto; padding: 20px; }
        .header {
            background: rgba(255,250,241,0.92); border: 1px solid var(--line);
            border-radius: 22px; padding: 20px; margin-bottom: 20px;
            box-shadow: var(--shadow); text-align: center;
        }
        h1 { margin: 0 0 10px; font-size: 28px; }
        .status { color: var(--muted); font-size: 14px; margin-bottom: 15px; }
        .refresh-btn {
            background: var(--green); color: white; border: none;
            padding: 15px 30px; border-radius: 12px; font-size: 16px;
            font-weight: 700; cursor: pointer; width: 100%;
        }
        .refresh-btn:active { transform: scale(0.98); }
        .refresh-btn:disabled { background: var(--muted); }
        .loading { display: none; color: var(--green); margin-top: 10px; }
        .pick-card {
            background: rgba(255,250,241,0.92); border: 1px solid var(--line);
            border-radius: 18px; padding: 16px; margin-bottom: 12px;
            box-shadow: var(--shadow);
        }
        .sport { color: var(--green); font-size: 12px; font-weight: 700; text-transform: uppercase; }
        .event { font-size: 18px; font-weight: 700; margin: 8px 0; }
        .bet { font-size: 16px; margin-bottom: 12px; }
        .grid {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 8px; margin-bottom: 12px;
        }
        .stat {
            background: rgba(255,255,255,0.7); border: 1px solid var(--line);
            border-radius: 12px; padding: 10px; text-align: center;
        }
        .stat-label { font-size: 11px; color: var(--muted); text-transform: uppercase; }
        .stat-value { font-size: 16px; font-weight: 700; }
        .track-btn {
            display: block; width: 100%; padding: 12px;
            background: var(--green); color: white; border: none;
            border-radius: 10px; font-size: 14px; font-weight: 700;
            cursor: pointer; margin-top: 10px;
        }
        .tracked { background: var(--muted); }
        .error { background: var(--red); color: white; padding: 15px; border-radius: 10px; margin-top: 10px; }
        @media (max-width: 600px) {
            .grid { grid-template-columns: repeat(2, 1fr); }
            h1 { font-size: 24px; }
        }
    </style>
</head>
<body>
    <main>
        <div class="header">
            <h1>🎯 Daily Edge Card</h1>
            <div class="status">Last updated: {{ last_update }}</div>
            <button class="refresh-btn" onclick="refreshPicks()" id="refreshBtn">
                🔄 Refresh Picks Now
            </button>
            <div class="loading" id="loading">Generating fresh picks... (30-60 seconds)</div>
        </div>
        
        <div id="picks-container">
            {{ picks_html }}
        </div>
    </main>
    
    <script>
        let trackedBets = JSON.parse(localStorage.getItem('trackedBets') || '[]');
        
        function refreshPicks() {
            const btn = document.getElementById('refreshBtn');
            const loading = document.getElementById('loading');
            btn.disabled = true;
            loading.style.display = 'block';
            
            fetch('/api/refresh', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        location.reload();
                    } else {
                        alert('Error: ' + data.error);
                        btn.disabled = false;
                        loading.style.display = 'none';
                    }
                })
                .catch(err => {
                    alert('Failed to refresh: ' + err);
                    btn.disabled = false;
                    loading.style.display = 'none';
                });
        }
        
        function trackBet(index, sport, event, bet, odds, stake) {
            const betId = 'bet_' + index + '_' + Date.now();
            trackedBets.push({
                id: betId, sport, event, bet, odds, stake,
                result: 'pending', timestamp: new Date().toISOString()
            });
            localStorage.setItem('trackedBets', JSON.stringify(trackedBets));
            
            const btn = document.getElementById('track-btn-' + index);
            btn.textContent = '✓ Tracked';
            btn.classList.add('tracked');
            btn.disabled = true;
        }
        
        // Mark already tracked
        trackedBets.forEach(b => {
            // Could highlight tracked bets here
        });
    </script>
</body>
</html>
    '''
    
    picks_html = generate_picks_html(data)
    last_update = data.get('generated_at', 'Never')
    
    return render_template_string(html, 
                                  picks_html=picks_html, 
                                  last_update=last_update)

def generate_picks_html(data):
    """Generate HTML for picks from data."""
    picks = data.get('picks', [])
    if not picks:
        return '<div class="pick-card"><p style="text-align:center;color:var(--muted)">No picks generated yet. Click Refresh above.</p></div>'
    
    html = ''
    for i, p in enumerate(picks):
        html += f'''
        <div class="pick-card">
            <div class="sport">{p.get('sport', '')}</div>
            <div class="event">{p.get('event', '')}</div>
            <div class="bet">{p.get('bet', '')}</div>
            <div class="grid">
                <div class="stat">
                    <div class="stat-label">True %</div>
                    <div class="stat-value">{p.get('true_prob', 0):.1f}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">EV</div>
                    <div class="stat-value">{p.get('ev', 0):.1f}%</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Stake</div>
                    <div class="stat-value">${p.get('stake', 0):.2f}</div>
                </div>
            </div>
            <button class="track-btn" id="track-btn-{i}" onclick="trackBet({i}, '{p.get('sport','')}', '{p.get('event','')}', '{p.get('bet','')}', {p.get('odds',0)}, {p.get('stake',0)})">
                Track This Bet
            </button>
        </div>
        '''
    return html

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Trigger a refresh of the model."""
    print(f"[{datetime.now()}] Refresh requested from {request.remote_addr}")
    success, msg = generate_now()
    return jsonify({"success": success, "message": msg})

@app.route('/api/picks')
def api_picks():
    """Get current picks as JSON."""
    return jsonify(load_picks_data())

@app.route('/api/status')
def api_status():
    """Get server status."""
    html_exists = HTML_FILE.exists()
    json_exists = JSON_FILE.exists()
    return jsonify({
        "running": True,
        "html_ready": html_exists,
        "json_ready": json_exists,
        "server_time": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("DAILY EDGE CARD - LIVE REFRESH SERVER")
    print("=" * 60)
    print()
    print("Your phone can access this at:")
    print()
    
    # Get IP addresses
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"
    
    print(f"  Local WiFi: http://{local_ip}:5000")
    print(f"  This PC:    http://localhost:5000")
    print()
    print("For internet access, run: ngrok http 5000")
    print()
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Generate initial if none exists
    if not HTML_FILE.exists():
        print("\nGenerating initial picks...")
        generate_now()
    
    app.run(host='0.0.0.0', port=5000, debug=False)
