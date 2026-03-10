import json
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from models.pipeline import run_pipeline
from data.loader import progress_state
from data.track_builder import get_track_svg

app = Flask(__name__)
_race_cache  = {}
_track_cache = {}

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/track")
def track():
    """
    Returns real GPS-traced SVG path for a circuit.
    Fetches once from OpenF1 location data, caches forever.
    """
    circuit = request.args.get("circuit", "Australia")
    year    = int(request.args.get("year", 2024))  # use historical year for stable data
    key     = circuit

    if key not in _track_cache:
        data = get_track_svg(circuit, year)
        _track_cache[key] = data  # may be None if OpenF1 unreachable

    return jsonify(_track_cache[key])


@app.route("/api/analyze")
def analyze():
    year    = int(request.args.get("year", 2026))
    circuit = request.args.get("circuit", "Australia")
    key     = f"{year}_{circuit}"
    if key not in _race_cache:
        try:
            _race_cache[key] = run_pipeline(year, circuit)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify(_race_cache[key])


@app.route("/api/progress")
def progress():
    def generate():
        import time
        while True:
            data = json.dumps({
                "status":  progress_state["status"],
                "message": progress_state["message"],
                "pct":     progress_state["pct"],
            })
            yield f"data: {data}\n\n"
            if progress_state["status"] == "done":
                break
            time.sleep(0.5)
    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    print("🏎  F1 Race Intelligence v2")
    print("   http://localhost:5001")
    app.run(debug=False, port=5001, threaded=True)
