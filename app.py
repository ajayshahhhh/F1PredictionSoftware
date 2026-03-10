"""
Flask backend for the F1 Probability Intelligence System.
"""

import json
from flask import Flask, render_template, jsonify, request
from orchestrator import run_pipeline

app = Flask(__name__)

_cache: dict = {}


def get_results(year: int, circuit: str) -> dict:
    key = f"{year}_{circuit}"
    if key not in _cache:
        _cache[key] = run_pipeline(year=year, circuit=circuit)
    return _cache[key]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run")
def api_run():
    year    = int(request.args.get("year", 2024))
    circuit = request.args.get("circuit", "Bahrain")
    results = get_results(year, circuit)
    return jsonify(results)


@app.route("/api/circuits")
def api_circuits():
    return jsonify({
        "circuits": ["Bahrain", "Monaco", "Silverstone", "Monza"]
    })


if __name__ == "__main__":
    print("🏎  F1 Probability Intelligence System")
    print("   http://localhost:5001")
    app.run(debug=True, port=5001)
