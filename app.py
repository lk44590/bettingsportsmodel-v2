from flask import Flask, send_file, jsonify, render_template_string
import subprocess
import os
from pathlib import Path
from datetime import datetime, date
import json

app = Flask(__name__)

# Configuration
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DATA_DIR = ROOT / "data"

@app.route('/')
def index():
    """Serve the latest daily card HTML"""
    today = date.today().strftime("%Y-%m-%d")
    html_path = OUTPUT_DIR / f"daily-card-{today}.html"
    
    if html_path.exists():
        return send_file(html_path)
    else:
        # If today's card doesn't exist, try to generate it
        try:
            generate_card()
            if html_path.exists():
                return send_file(html_path)
        except Exception as e:
            return f"Error generating card: {str(e)}", 500
    
    # Fallback to most recent card
    html_files = sorted(OUTPUT_DIR.glob("daily-card-*.html"), reverse=True)
    if html_files:
        return send_file(html_files[0])
    
    return "No daily card available. Please run generate_daily_card.py first.", 404

@app.route('/phone')
def phone():
    """Serve the phone-optimized card"""
    today = date.today().strftime("%Y-%m-%d")
    html_path = OUTPUT_DIR / f"phone-card-{today}.html"
    
    if html_path.exists():
        return send_file(html_path)
    
    # Fallback to most recent phone card
    html_files = sorted(OUTPUT_DIR.glob("phone-card-*.html"), reverse=True)
    if html_files:
        return send_file(html_files[0])
    
    return "No phone card available.", 404

@app.route('/json')
def json_output():
    """Serve the latest daily card JSON"""
    today = date.today().strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"daily-card-{today}.json"
    
    if json_path.exists():
        return send_file(json_path, mimetype='application/json')
    
    # Fallback to most recent JSON
    json_files = sorted(OUTPUT_DIR.glob("daily-card-*.json"), reverse=True)
    if json_files:
        return send_file(json_files[0], mimetype='application/json')
    
    return jsonify({"error": "No daily card available"}), 404

@app.route('/candidates')
def candidates():
    """Serve the latest candidates JSON"""
    today = date.today().strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"daily-candidates-{today}.json"
    
    if json_path.exists():
        return send_file(json_path, mimetype='application/json')
    
    # Fallback to most recent candidates
    json_files = sorted(OUTPUT_DIR.glob("daily-candidates-*.json"), reverse=True)
    if json_files:
        return send_file(json_files[0], mimetype='application/json')
    
    return jsonify({"error": "No candidates available"}), 404

@app.route('/generate', methods=['POST'])
def generate_card():
    """Generate a new daily card"""
    try:
        result = subprocess.run(
            ['python', 'generate_daily_card.py'],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return jsonify({
                "success": True,
                "message": "Daily card generated successfully",
                "output": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "message": "Error generating daily card",
                "error": result.stderr
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "message": "Generation timed out after 5 minutes"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        }), 500

@app.route('/health')
def health():
    """Health check endpoint for cloud services"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "output_dir_exists": OUTPUT_DIR.exists(),
        "data_dir_exists": DATA_DIR.exists()
    })

@app.route('/api/performance')
def performance():
    """Serve model performance data"""
    today = date.today().strftime("%Y-%m-%d")
    json_path = OUTPUT_DIR / f"model-performance-{today}.json"
    
    if json_path.exists():
        return send_file(json_path, mimetype='application/json')
    
    # Fallback to most recent performance
    json_files = sorted(OUTPUT_DIR.glob("model-performance-*.json"), reverse=True)
    if json_files:
        return send_file(json_files[0], mimetype='application/json')
    
    return jsonify({"error": "No performance data available"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
