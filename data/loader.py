"""
F1 Telemetry Data Loader
Uses OpenF1 API (https://api.openf1.org) for real telemetry.
Falls back to realistic mock data if the API is unavailable or returns no data.

OpenF1 is free, requires no API key, and has data from 2023 onwards including
the 2026 season. Telemetry updates ~3 seconds behind live broadcast during races.
"""

import numpy as np
import json
import urllib.request
import urllib.parse
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

OPENF1_MAX_LAPS_PER_DRIVER = 6   # keep latency low, same as your FastF1 optimization

# ── Circuit corners ────────────────────────────────────────────────────────
CIRCUIT_CORNERS = {
    "Bahrain":    ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15"],
    "Monaco":     ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19"],
    "Silverstone":["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18"],
    "Monza":      ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11"],
    "China":      ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"],
    "Australia":  ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"],
    "Japan":      ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18"],
    "Saudi Arabia":["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T24","T25","T26","T27"],
}

# 2026 driver lineup
TOP_DRIVERS = {
    "VER": "Max Verstappen",
    "LEC": "Charles Leclerc",
    "HAM": "Lewis Hamilton",
    "RUS": "George Russell",
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "ANT": "Kimi Antonelli",
    "ALO": "Fernando Alonso",
    "SAI": "Carlos Sainz",
    "HUL": "Nico Hulkenberg",
}

TEAM_COLORS = {
    "VER": "#3671C6",
    "LEC": "#E8002D", "SAI": "#E8002D",
    "HAM": "#00A19C", "RUS": "#00A19C", "ANT": "#00A19C",
    "NOR": "#FF8000", "PIA": "#FF8000",
    "ALO": "#358C75",
    "HUL": "#B6BABD",
}


# ── OpenF1 helpers ─────────────────────────────────────────────────────────

def _openf1_get(endpoint: str, params: dict) -> list:
    base = "https://api.openf1.org/v1"
    query = urllib.parse.urlencode(params)
    url = f"{base}/{endpoint}?{query}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def _get_session_key(year: int, circuit: str, session_type: str) -> int | None:
    session_name_map = {
        "R": "Race", "Q": "Qualifying",
        "FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3",
    }
    session_name = session_name_map.get(session_type, "Race")

    # Try circuit short name first
    results = _openf1_get("sessions", {
        "year": year,
        "circuit_short_name": circuit,
        "session_name": session_name,
    })

    if not results:
        country_map = {
            "Australia": "Australia", "China": "China", "Bahrain": "Bahrain",
            "Monaco": "Monaco", "Silverstone": "Great Britain", "Monza": "Italy",
            "Japan": "Japan", "Saudi Arabia": "Saudi Arabia",
        }
        country = country_map.get(circuit)
        if country:
            results = _openf1_get("sessions", {
                "year": year,
                "country_name": country,
                "session_name": session_name,
            })

    return results[0]["session_key"] if results else None


def _get_driver_numbers(session_key: int) -> dict:
    drivers = _openf1_get("drivers", {"session_key": session_key})
    return {d["driver_number"]: d["name_acronym"] for d in drivers if "name_acronym" in d}


def _get_fastest_lap_numbers(session_key: int, driver_number: int) -> list[int]:
    laps = _openf1_get("laps", {
        "session_key": session_key,
        "driver_number": driver_number,
    })
    valid = [l for l in laps if l.get("lap_duration") and l["lap_duration"] > 0]
    valid.sort(key=lambda l: l["lap_duration"])
    return [l["lap_number"] for l in valid[:OPENF1_MAX_LAPS_PER_DRIVER]]


def _get_car_data(session_key: int, driver_number: int, lap_number: int) -> list[dict]:
    return _openf1_get("car_data", {
        "session_key":   session_key,
        "driver_number": driver_number,
        "lap_number":    lap_number,
    })


def _telemetry_to_corners(tel: list[dict], circuit: str) -> list[dict]:
    corners = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Bahrain"])
    if not tel:
        return []

    speeds    = np.array([t.get("speed", 0)            for t in tel], dtype=float)
    throttles = np.array([t.get("throttle", 0)         for t in tel], dtype=float) / 100.0
    brakes    = np.array([float(t.get("brake", False)) for t in tel], dtype=float)

    segments = np.array_split(np.arange(len(tel)), len(corners))

    corner_data = []
    for corner, seg_idx in zip(corners, segments):
        if len(seg_idx) == 0:
            continue
        corner_data.append({
            "corner":      corner,
            "speed":       round(float(speeds[seg_idx].mean()), 2),
            "throttle":    round(float(throttles[seg_idx].mean()), 4),
            "brake":       round(float(brakes[seg_idx].mean()), 4),
            "sector_time": round(len(seg_idx) * (1 / 3.7), 4),
        })

    return corner_data


# ── Public interface ───────────────────────────────────────────────────────

def load_session(year: int, circuit: str, session_type: str = "R") -> dict:
    cache_key = f"{year}_{circuit}_{session_type}.json"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        print(f"[loader] Cache hit: {cache_key}")
        with open(cache_path) as f:
            return json.load(f)

    try:
        print(f"[loader] Fetching OpenF1: {year} {circuit} {session_type}...")
        data = _load_openf1(year, circuit, session_type)
    except Exception as e:
        print(f"[loader] OpenF1 unavailable ({e}), using mock data")
        data = _load_mock(circuit)

    with open(cache_path, "w") as f:
        json.dump(data, f)

    return data


def _load_openf1(year: int, circuit: str, session_type: str) -> dict:
    session_key = _get_session_key(year, circuit, session_type)
    if not session_key:
        raise ValueError(f"No session found for {year} {circuit} {session_type}")

    print(f"[loader] session_key={session_key}")
    driver_numbers = _get_driver_numbers(session_key)
    acronym_to_num = {v: k for k, v in driver_numbers.items()}

    drivers = {}
    for code in TOP_DRIVERS:
        driver_num = acronym_to_num.get(code)
        if not driver_num:
            print(f"[loader] {code} not in session, using mock")
            drivers[code] = _mock_telemetry(code, circuit)
            continue

        try:
            lap_numbers = _get_fastest_lap_numbers(session_key, driver_num)
            if not lap_numbers:
                raise ValueError("no valid laps")

            laps_out = []
            for lap_num in lap_numbers:
                tel = _get_car_data(session_key, driver_num, lap_num)
                if not tel:
                    continue
                corner_data = _telemetry_to_corners(tel, circuit)
                if not corner_data:
                    continue

                lap_info = _openf1_get("laps", {
                    "session_key": session_key,
                    "driver_number": driver_num,
                    "lap_number": lap_num,
                })
                lap_duration = (
                    lap_info[0]["lap_duration"] if lap_info
                    else sum(c["sector_time"] for c in corner_data)
                )

                laps_out.append({
                    "lap":     lap_num,
                    "laptime": round(float(lap_duration), 3),
                    "corners": corner_data,
                })

            if not laps_out:
                raise ValueError("no usable laps")

            drivers[code] = {
                "driver":     code,
                "name":       TOP_DRIVERS[code],
                "circuit":    circuit,
                "team_color": TEAM_COLORS.get(code, "#AAAAAA"),
                "laps":       laps_out,
            }
            print(f"[loader] {code}: {len(laps_out)} laps from OpenF1")

        except Exception as e:
            print(f"[loader] {code} failed ({e}), using mock")
            drivers[code] = _mock_telemetry(code, circuit)

    return {"circuit": circuit, "drivers": drivers}


def _load_mock(circuit: str) -> dict:
    return {"circuit": circuit, "drivers": {code: _mock_telemetry(code, circuit) for code in TOP_DRIVERS}}


def _mock_telemetry(driver_code: str, circuit: str, n_laps: int = 20) -> dict:
    rng = np.random.default_rng(seed=abs(hash(driver_code + circuit)) % (2**31))
    skill = {
        "VER": 0.04, "LEC": 0.025, "SAI": 0.010, "HAM": 0.030, "RUS": 0.015,
        "NOR": 0.020, "PIA": 0.008, "ALO": 0.022, "ANT": 0.018, "HUL": 0.000,
    }.get(driver_code, 0.0)

    corners = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Bahrain"])
    laps = []
    for lap_idx in range(n_laps):
        corner_data = []
        for c_idx, corner in enumerate(corners):
            base_speed    = 180 + 60 * np.sin(c_idx * 0.4)
            base_throttle = 0.65 + 0.20 * np.cos(c_idx * 0.5)
            base_brake    = max(0, 0.30 - 0.15 * np.cos(c_idx * 0.5))
            corner_data.append({
                "corner":      corner,
                "speed":       round(float(base_speed * (1 + skill) + rng.normal(0, 4)), 2),
                "throttle":    round(float(np.clip(base_throttle * (1 + skill*0.5) + rng.normal(0, 0.04), 0, 1)), 4),
                "brake":       round(float(np.clip(base_brake * (1 - skill*0.3) + rng.normal(0, 0.03), 0, 1)), 4),
                "sector_time": round(float((1.8 - skill*0.3) + rng.normal(0, 0.08)), 4),
            })
        lap_time = sum(c["sector_time"] for c in corner_data) * 5.2 + rng.normal(0, 0.3)
        laps.append({"lap": lap_idx + 1, "laptime": round(float(lap_time), 3), "corners": corner_data})

    return {
        "driver":     driver_code,
        "name":       TOP_DRIVERS.get(driver_code, driver_code),
        "circuit":    circuit,
        "team_color": TEAM_COLORS.get(driver_code, "#AAAAAA"),
        "laps":       laps,
    }
