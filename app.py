import json
import threading
from flask import Flask, render_template, jsonify, request, Response, stream_with_context
from models.pipeline import run_pipeline
from data.loader import progress_state

app = Flask(__name__)

_cache   = {}          # results cache
_running = {}          # {key: threading.Event} — signals when analysis is done

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze")
def analyze():
    year    = int(request.args.get("year", 2026))
    circuit = request.args.get("circuit", "Australia")
    key     = f"{year}_{circuit}"

    if key in _cache:
        return jsonify(_cache[key])

    # Run synchronously (frontend shows progress via SSE)
    result = run_pipeline(year, circuit)
    _cache[key] = result
    return jsonify(result)


@app.route("/api/progress")
def progress():
    """Server-Sent Events stream for loading progress."""
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
