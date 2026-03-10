"""
OpenF1 telemetry loader — full race, all laps, all drivers.
First load: slow (~2-3 min), fetches every lap's telemetry via OpenF1.
After that: instant from disk cache.
Progress is streamed back via a /api/progress SSE endpoint.
"""

import numpy as np
import json
import urllib.request
import urllib.parse
import threading
from pathlib import Path
from data.drivers import DRIVERS

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CIRCUIT_CORNERS = {
    "Australia":   {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"], "country": "Australia"},
    "China":       {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"], "country": "China"},
    "Bahrain":     {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15"],       "country": "Bahrain"},
    "Japan":       {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18"], "country": "Japan"},
    "Saudi Arabia":{"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T24","T25","T26","T27"], "country": "Saudi Arabia"},
}

# Global progress state (written by loader thread, read by SSE endpoint)
progress_state = {"status": "idle", "message": "", "pct": 0}


def _set_progress(msg: str, pct: int):
    progress_state["status"]  = "loading"
    progress_state["message"] = msg
    progress_state["pct"]     = pct
    print(f"[{pct:3d}%] {msg}")


def _get(endpoint, params):
    url = "https://api.openf1.org/v1/" + endpoint + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def _session_key(year, circuit, stype="R"):
    name_map = {"R": "Race", "Q": "Qualifying", "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3"}
    sname   = name_map.get(stype, "Race")
    country = CIRCUIT_CORNERS.get(circuit, {}).get("country", circuit)
    res = _get("sessions", {"year": year, "country_name": country, "session_name": sname})
    return res[0]["session_key"] if res else None


def load_session(year: int, circuit: str, session_type: str = "R") -> dict:
    key  = f"{year}_{circuit}_{session_type}.json"
    path = CACHE_DIR / key

    if path.exists():
        _set_progress("Loading from cache...", 99)
        with open(path) as f:
            data = json.load(f)
        progress_state["status"] = "done"
        return data

    try:
        data = _load_openf1(year, circuit, session_type)
    except Exception as e:
        print(f"[loader] OpenF1 failed ({e}), using mock")
        _set_progress("OpenF1 unavailable — using mock data", 90)
        data = _load_mock(circuit, year)

    with open(path, "w") as f:
        json.dump(data, f)

    progress_state["status"] = "done"
    return data


def _load_openf1(year, circuit, session_type):
    corners = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Australia"])["corners"]

    _set_progress("Connecting to OpenF1...", 2)
    sk = _session_key(year, circuit, session_type)
    if not sk:
        raise ValueError(f"No session found: {year} {circuit} {session_type}")

    _set_progress(f"Session key {sk} — fetching driver list...", 5)
    raw_drivers = _get("drivers", {"session_key": sk})
    acronym_to_num = {d["name_acronym"]: d["driver_number"] for d in raw_drivers if "name_acronym" in d}

    # Total work units = 20 drivers × avg ~50 laps each = ~1000 lap fetches
    all_driver_codes = list(DRIVERS.keys())
    n_drivers = len(all_driver_codes)
    drivers_out = {}

    for d_idx, code in enumerate(all_driver_codes):
        base_pct = 5 + int((d_idx / n_drivers) * 90)
        _set_progress(f"Fetching {code} ({d_idx+1}/{n_drivers})...", base_pct)

        num = acronym_to_num.get(code)
        if not num:
            _set_progress(f"{code} not in session — using mock", base_pct)
            drivers_out[code] = _mock_driver(code, circuit)
            continue

        try:
            # Fetch all laps for this driver in one call
            laps_raw = _get("laps", {"session_key": sk, "driver_number": num})
            valid_laps = [l for l in laps_raw if l.get("lap_duration") and l["lap_duration"] > 60]
            valid_laps.sort(key=lambda l: l["lap_number"])

            if not valid_laps:
                raise ValueError("no valid laps")

            n_laps = len(valid_laps)
            laps_out = []

            for l_idx, lap in enumerate(valid_laps):
                lap_pct = base_pct + int((l_idx / n_laps) * (90 / n_drivers))
                if l_idx % 5 == 0:
                    _set_progress(f"{code} — lap {lap['lap_number']} / {valid_laps[-1]['lap_number']}", lap_pct)

                tel = _get("car_data", {
                    "session_key":   sk,
                    "driver_number": num,
                    "lap_number":    lap["lap_number"],
                })

                if not tel:
                    # No telemetry for this lap (safety car, pit lap etc.) —
                    # still record the lap with estimated corner values
                    laps_out.append(_lap_from_duration(lap, corners))
                    continue

                speeds    = np.array([t.get("speed",    0)           for t in tel], dtype=float)
                throttles = np.array([t.get("throttle", 0)           for t in tel], dtype=float) / 100
                brakes    = np.array([float(t.get("brake", False))   for t in tel], dtype=float)

                segs = np.array_split(np.arange(len(tel)), len(corners))
                corner_data = []
                for cn, idx in zip(corners, segs):
                    if not len(idx):
                        continue
                    corner_data.append({
                        "corner":   cn,
                        "speed":    round(float(speeds[idx].mean()),    2),
                        "throttle": round(float(throttles[idx].mean()), 4),
                        "brake":    round(float(brakes[idx].mean()),    4),
                    })

                laps_out.append({
                    "lap":     lap["lap_number"],
                    "laptime": round(float(lap["lap_duration"]), 3),
                    "corners": corner_data,
                })

            if not laps_out:
                raise ValueError("no usable laps after telemetry fetch")

            drivers_out[code] = {
                **DRIVERS[code],
                "driver":  code,
                "circuit": circuit,
                "laps":    laps_out,
            }
            _set_progress(f"{code} ✓  {len(laps_out)} laps", base_pct)

        except Exception as e:
            _set_progress(f"{code} failed ({e}) — mock", base_pct)
            drivers_out[code] = _mock_driver(code, circuit)

    return {"circuit": circuit, "year": year, "corners": corners, "drivers": drivers_out}


def _lap_from_duration(lap_raw: dict, corners: list) -> dict:
    """Fallback lap entry when telemetry is missing (pit stop, SC lap etc.)"""
    n = len(corners)
    dur = lap_raw.get("lap_duration", 90.0)
    corner_data = [{"corner": cn, "speed": 150.0, "throttle": 0.5, "brake": 0.1} for cn in corners]
    return {"lap": lap_raw["lap_number"], "laptime": round(float(dur), 3), "corners": corner_data}


def _load_mock(circuit, year=0):
    corners = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Australia"])["corners"]
    return {
        "circuit": circuit, "year": year, "corners": corners,
        "drivers": {code: _mock_driver(code, circuit) for code in DRIVERS},
    }


def _mock_driver(code, circuit, n_laps=57):
    """57 laps ~ Australia race distance."""
    corners = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Australia"])["corners"]
    pts   = DRIVERS[code]["points"]
    skill = np.clip(pts / 350.0, 0, 0.12)
    rng   = np.random.default_rng(abs(hash(code + circuit)) % (2**31))

    laps = []
    for i in range(n_laps):
        corner_data = []
        for c_idx, cn in enumerate(corners):
            base_spd = 185 + 55 * np.sin(c_idx * 0.42)
            base_thr = 0.62 + 0.22 * np.cos(c_idx * 0.5)
            base_brk = max(0, 0.28 - 0.14 * np.cos(c_idx * 0.5))
            # Add slight lap-to-lap variation (tyre deg etc.)
            deg = 1 - (i / n_laps) * 0.03
            corner_data.append({
                "corner":   cn,
                "speed":    round(float(base_spd * (1 + skill) * deg + rng.normal(0, 3)), 2),
                "throttle": round(float(np.clip(base_thr*(1+skill*0.5)*deg + rng.normal(0,0.03), 0, 1)), 4),
                "brake":    round(float(np.clip(base_brk*(1-skill*0.3)    + rng.normal(0,0.02), 0, 1)), 4),
            })
        lt = sum((1.85 - skill*0.35 + rng.normal(0, 0.07)) for _ in corners) * 4.7 + rng.normal(0, 0.3)
        laps.append({"lap": i+1, "laptime": round(float(lt), 3), "corners": corner_data})

    return {**DRIVERS[code], "driver": code, "circuit": circuit, "laps": laps}
