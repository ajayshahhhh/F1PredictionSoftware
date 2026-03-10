"""
Track outline builder using OpenF1 /v1/location GPS data.

Strategy:
  1. Find a past qualifying session for the circuit (Q gives clean single laps)
  2. Pick one driver, fetch their fastest lap's location data (~400-600 GPS points)
  3. Normalize x/y to SVG viewport (600x500), smooth slightly
  4. Cache the SVG path string + corner label positions to disk

Corner positions are derived from the same GPS data:
  - Split the lap into N equal segments (one per corner)
  - Each corner dot placed at the start of its segment
  - This is approximate but tied to real geometry

Call: get_track_svg(circuit, year) -> {path: str, corners: [{name, x, y}], viewBox: str}
"""

import json
import numpy as np
import urllib.request
import urllib.parse
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SVG_W, SVG_H = 600, 500
PADDING = 50   # px padding inside viewport

# Which session + driver to use for each circuit's trace
# Using qualifying (cleaner single laps, no traffic)
# session_key values sourced from OpenF1 /v1/sessions
TRACE_SOURCES = {
    "Australia":    {"country": "Australia",    "session_name": "Qualifying"},
    "China":        {"country": "China",        "session_name": "Qualifying"},
    "Bahrain":      {"country": "Bahrain",      "session_name": "Qualifying"},
    "Japan":        {"country": "Japan",        "session_name": "Qualifying"},
    "Saudi Arabia": {"country": "Saudi Arabia", "session_name": "Qualifying"},
}

CIRCUIT_N_CORNERS = {
    "Australia":    16,
    "China":        16,
    "Bahrain":      15,
    "Japan":        18,
    "Saudi Arabia": 27,
}


def _get(endpoint, params):
    url = "https://api.openf1.org/v1/" + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def get_track_svg(circuit: str, year: int = 2024) -> dict | None:
    """
    Returns cached track data or fetches+builds it.
    Returns None if OpenF1 is unreachable (caller falls back to static SVG).
    """
    cache_path = CACHE_DIR / f"track_{circuit}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    try:
        data = _fetch_and_build(circuit, year)
        with open(cache_path, "w") as f:
            json.dump(data, f)
        print(f"[track_builder] {circuit} track cached ({len(data['points'])} pts)")
        return data
    except Exception as e:
        print(f"[track_builder] Failed to build {circuit} track: {e}")
        return None


def _fetch_and_build(circuit: str, year: int) -> dict:
    src = TRACE_SOURCES.get(circuit)
    if not src:
        raise ValueError(f"No trace source for {circuit}")

    # Find session
    sessions = _get("sessions", {
        "year": year,
        "country_name": src["country"],
        "session_name": src["session_name"],
    })
    if not sessions:
        # Try previous year
        sessions = _get("sessions", {
            "year": year - 1,
            "country_name": src["country"],
            "session_name": src["session_name"],
        })
    if not sessions:
        raise ValueError(f"No session found for {circuit} {year}")

    sk = sessions[0]["session_key"]
    print(f"[track_builder] Using session_key={sk} for {circuit}")

    # Get drivers in this session, pick the one with most laps (likely completed quali)
    drivers = _get("drivers", {"session_key": sk})
    if not drivers:
        raise ValueError("No drivers found")

    # Try a few drivers until we get good location data
    # Prefer well-known fast drivers (more likely to have clean laps)
    preferred = ["NOR", "VER", "LEC", "HAM", "RUS", "PIA", "SAI", "ALO"]
    acronym_to_num = {d.get("name_acronym"): d["driver_number"] for d in drivers if d.get("name_acronym")}
    
    driver_num = None
    for code in preferred:
        if code in acronym_to_num:
            driver_num = acronym_to_num[code]
            print(f"[track_builder] Using driver {code} ({driver_num})")
            break
    if not driver_num:
        driver_num = drivers[0]["driver_number"]

    # Get laps to find the fastest clean lap
    laps = _get("laps", {"session_key": sk, "driver_number": driver_num})
    valid_laps = [l for l in laps if l.get("lap_duration") and l["lap_duration"] > 60]
    if not valid_laps:
        raise ValueError("No valid laps found")
    
    # Pick a clean mid-race lap (not pit out) with a known date_start
    clean_laps = [l for l in valid_laps if not l.get("is_pit_out_lap") and l.get("date_start")]
    if not clean_laps:
        raise ValueError("No clean laps with date_start found")
    fastest = min(clean_laps, key=lambda l: l["lap_duration"])
    lap_num = fastest["lap_number"]
    date_start = fastest["date_start"]
    print(f"[track_builder] Tracing lap {lap_num} ({fastest['lap_duration']:.3f}s) start={date_start}")

    # Fetch full session location for this driver, then slice one clean lap
    from datetime import datetime, timedelta
    print(f"[track_builder] Fetching full session location data...")
    loc_all = _get("location", {
        "session_key":   sk,
        "driver_number": driver_num,
    })
    if not loc_all or len(loc_all) < 50:
        raise ValueError(f"Not enough location points: {len(loc_all) if loc_all else 0}")

    # Slice out just the fastest lap using timestamps
    dt_start = datetime.fromisoformat(date_start.replace("Z", "+00:00"))
    dt_end   = dt_start + timedelta(seconds=fastest["lap_duration"] + 2)
    loc_data = [p for p in loc_all
                if "date" in p and dt_start
                <= datetime.fromisoformat(p["date"].replace("Z", "+00:00"))
                <= dt_end]

    # Fall back to full session if slice is too small (gives full multi-lap outline)
    if len(loc_data) < 50:
        print("[track_builder] Lap slice too small, using full session outline")
        loc_data = loc_all

    print(f"[track_builder] Got {len(loc_data)} raw location points")

    # Extract x/y coordinates
    pts = np.array([[p["x"], p["y"]] for p in loc_data if "x" in p and "y" in p], dtype=float)
    print(f"[track_builder] Using {len(pts)} GPS points for SVG path")

    return _build_svg_data(pts, circuit)


def _build_svg_data(pts: np.ndarray, circuit: str) -> dict:
    """
    Normalize GPS points → SVG coordinates, smooth, compute corner positions.
    """
    # Remove duplicate consecutive points
    diffs = np.diff(pts, axis=0)
    mask  = np.any(diffs != 0, axis=1)
    pts   = np.vstack([pts[0], pts[1:][mask]])

    # Smooth with a moving average to remove GPS jitter
    window = 5
    smoothed = np.copy(pts)
    for i in range(window, len(pts) - window):
        smoothed[i] = pts[i-window:i+window+1].mean(axis=0)

    # Normalize to SVG viewport with padding
    x_min, x_max = smoothed[:, 0].min(), smoothed[:, 0].max()
    y_min, y_max = smoothed[:, 1].min(), smoothed[:, 1].max()
    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)

    usable_w = SVG_W - 2 * PADDING
    usable_h = SVG_H - 2 * PADDING

    # Preserve aspect ratio
    scale = min(usable_w / x_range, usable_h / y_range)
    off_x = PADDING + (usable_w - x_range * scale) / 2
    off_y = PADDING + (usable_h - y_range * scale) / 2

    def to_svg(p):
        # Flip Y axis (SVG y increases downward, GPS y typically increases upward)
        sx = (p[0] - x_min) * scale + off_x
        sy = SVG_H - ((p[1] - y_min) * scale + off_y)
        return (round(float(sx), 1), round(float(sy), 1))

    svg_pts = [to_svg(p) for p in smoothed]

    # Build SVG path string
    path_d = "M " + " L ".join(f"{x},{y}" for x, y in svg_pts) + " Z"

    # Compute corner positions
    # Distribute N corners evenly along the lap path
    n_corners = CIRCUIT_N_CORNERS.get(circuit, 16)
    corner_positions = []
    step = len(svg_pts) // n_corners
    for i in range(n_corners):
        idx = min(i * step, len(svg_pts) - 1)
        cx, cy = svg_pts[idx]
        # Compute label offset: push label away from track center
        center_x = sum(p[0] for p in svg_pts) / len(svg_pts)
        center_y = sum(p[1] for p in svg_pts) / len(svg_pts)
        dx = cx - center_x
        dy = cy - center_y
        norm = max((dx**2 + dy**2)**0.5, 1)
        lx = round(cx + dx/norm * 16, 1)
        ly = round(cy + dy/norm * 16, 1)
        corner_positions.append({
            "name": f"T{i+1}",
            "x":   cx, "y": cy,
            "lx":  lx, "ly": ly,
        })

    return {
        "circuit":  circuit,
        "path":     path_d,
        "points":   svg_pts,
        "corners":  corner_positions,
        "viewBox":  f"0 0 {SVG_W} {SVG_H}",
    }
