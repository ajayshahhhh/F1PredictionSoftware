"""
OpenF1 loader — full race, all laps, all drivers.

Key insight: fetch ALL car_data for a driver in one call, then split by lap timestamps.
That's 20 requests instead of 1140. Much faster, no rate limit issues.
"""

import numpy as np
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from data.drivers import DRIVERS

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CIRCUIT_CORNERS = {
    "Australia":    {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"], "country": "Australia"},
    "China":        {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16"], "country": "China"},
    "Bahrain":      {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15"],       "country": "Bahrain"},
    "Japan":        {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18"], "country": "Japan"},
    "Saudi Arabia": {"corners": ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12","T13","T14","T15","T16","T17","T18","T19","T20","T21","T22","T23","T24","T25","T26","T27"], "country": "Saudi Arabia"},
}

progress_state = {"status": "idle", "message": "", "pct": 0}

def _set_progress(msg, pct):
    progress_state.update({"status": "loading", "message": msg, "pct": pct})
    print(f"[{pct:3d}%] {msg}")

def _get(endpoint, params):
    str_params = {k: str(v) for k, v in params.items()}
    url = "https://api.openf1.org/v1/" + endpoint + "?" + urllib.parse.urlencode(str_params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

def _session_key(year, circuit, stype="R"):
    name_map = {"R": "Race", "Q": "Qualifying", "FP1": "Practice 1"}
    country  = CIRCUIT_CORNERS.get(circuit, {}).get("country", circuit)
    res = _get("sessions", {"year": year, "country_name": country,
                            "session_name": name_map.get(stype, "Race")})
    return res[0]["session_key"] if res else None

def _parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def _slice_telemetry_for_lap(tel_points, lap_start_str, lap_duration):
    """
    Given all telemetry for a driver (sorted by date), slice out points
    that fall within [lap_start, lap_start + lap_duration + 1s].
    """
    if not lap_start_str or not tel_points:
        return []
    dt_start = _parse_dt(lap_start_str)
    if dt_start is None:
        return []
    dt_end = dt_start + timedelta(seconds=lap_duration + 1.0)
    return [p for p in tel_points
            if dt_start <= (_parse_dt(p.get("date")) or dt_start) <= dt_end]

def _corners_from_tel(tel_slice, corners):
    if not tel_slice:
        return None
    speeds    = np.array([t.get("speed",    0)         for t in tel_slice], dtype=float)
    throttles = np.array([t.get("throttle", 0)         for t in tel_slice], dtype=float) / 100
    brakes    = np.array([float(t.get("brake", False)) for t in tel_slice], dtype=float)
    segs      = np.array_split(np.arange(len(tel_slice)), len(corners))
    result    = []
    for cn, idx in zip(corners, segs):
        if not len(idx):
            continue
        result.append({
            "corner":   cn,
            "speed":    round(float(speeds[idx].mean()),    2),
            "throttle": round(float(throttles[idx].mean()), 4),
            "brake":    round(float(brakes[idx].mean()),    4),
        })
    return result if result else None


def load_session(year, circuit, session_type="R"):
    key  = f"{year}_{circuit}_{session_type}.json"
    path = CACHE_DIR / key
    if path.exists():
        _set_progress("Loading from cache...", 99)
        with open(path) as f:
            data = json.load(f)
        progress_state["status"] = "done"
        return data
    data = _load_openf1(year, circuit, session_type)
    with open(path, "w") as f:
        json.dump(data, f)
    progress_state["status"] = "done"
    return data


def _load_openf1(year, circuit, session_type):
    corners  = CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Australia"])["corners"]
    fallback = [{"corner": cn, "speed": 150.0, "throttle": 0.5, "brake": 0.1} for cn in corners]

    _set_progress("Connecting to OpenF1...", 2)
    sk = _session_key(year, circuit, session_type)
    if not sk:
        raise ValueError(f"No session found: {year} {circuit}")

    _set_progress("Fetching driver list...", 4)
    raw_drivers    = _get("drivers", {"session_key": sk})
    acronym_to_num = {d["name_acronym"]: d["driver_number"]
                      for d in raw_drivers if d.get("name_acronym")}

    _set_progress("Fetching race positions...", 5)
    # Position data is time-based (no lap_number) -- store as sorted list per driver
    # We'll map to lap numbers later using lap date_start timestamps
    pos_by_driver_time = {}  # {driver_num: [(datetime, position), ...]}
    try:
        pos_raw = _get("position", {"session_key": sk})
        print(f"[loader] position: {len(pos_raw)} raw entries")
        from datetime import datetime
        for p in pos_raw:
            dn  = p.get("driver_number")
            pos = p.get("position")
            dt  = p.get("date")
            if dn and pos and dt:
                pos_by_driver_time.setdefault(dn, []).append(
                    (datetime.fromisoformat(dt.replace("Z", "+00:00")), pos))
        # Sort each driver's list by time
        for dn in pos_by_driver_time:
            pos_by_driver_time[dn].sort(key=lambda x: x[0])
        print(f"[loader] position: {len(pos_by_driver_time)} drivers with time-based positions")
    except Exception as e:
        print(f"[loader] position fetch failed: {e}")

    def get_position_at(driver_num, lap_date_str):
        """Return the most recent position at or before this lap's start time."""
        entries = pos_by_driver_time.get(driver_num, [])
        if not entries or not lap_date_str:
            return None
        from datetime import datetime
        lap_dt = datetime.fromisoformat(lap_date_str.replace("Z", "+00:00"))
        pos = None
        for dt, p in entries:
            if dt <= lap_dt:
                pos = p
            else:
                break
        return pos

    _set_progress("Fetching gap data...", 7)
    gap_by_driver = {}
    try:
        for iv in _get("intervals", {"session_key": sk}):
            dn  = iv.get("driver_number")
            lap = iv.get("lap_number")
            gap = iv.get("gap_to_leader")
            if dn and lap and gap is not None:
                gap_by_driver.setdefault(dn, {})[lap] = (
                    float(gap) if isinstance(gap, (int, float)) else 999.0)
    except Exception as e:
        print(f"[loader] intervals fetch failed: {e}")

    all_driver_codes = list(DRIVERS.keys())
    n_drivers        = len(all_driver_codes)
    drivers_out      = {}

    for d_idx, code in enumerate(all_driver_codes):
        base_pct = 10 + int((d_idx / n_drivers) * 85)
        _set_progress(f"Fetching {code} ({d_idx+1}/{n_drivers})...", base_pct)

        num = acronym_to_num.get(code)
        if not num:
            print(f"[loader] {code} not in session, skipping")
            continue

        try:
            # Lap times -- one call
            laps_raw = _get("laps", {"session_key": sk, "driver_number": num})
            valid    = sorted([l for l in laps_raw
                               if (l.get("lap_duration") or 0) > 60 and l.get("date_start")],
                              key=lambda l: l["lap_number"])
            if not valid:
                print(f"[loader] {code}: no valid laps")
                continue

            # ALL telemetry for this driver -- one call, split by lap timestamps
            tel_all = _get("car_data", {"session_key": sk, "driver_number": num})
            tel_all.sort(key=lambda t: t.get("date", ""))
            print(f"[loader] {code}: {len(valid)} laps, {len(tel_all)} telemetry points")

            laps_out = []
            for lap in valid:
                ln      = lap["lap_number"]
                tel_lap = _slice_telemetry_for_lap(tel_all, lap["date_start"], lap["lap_duration"])
                corners_data = _corners_from_tel(tel_lap, corners) or fallback
                laps_out.append({
                    "lap":      ln,
                    "laptime":  round(float(lap["lap_duration"]), 3),
                    "corners":  corners_data,
                    "position": get_position_at(num, lap["date_start"]),
                    "gap":      gap_by_driver.get(num, {}).get(ln),
                })

            drivers_out[code] = {**DRIVERS[code], "driver": code, "circuit": circuit,
                                 "laps": laps_out}
            _set_progress(f"{code} ✓ {len(laps_out)} laps", base_pct)

        except Exception as e:
            print(f"[loader] {code} failed: {e}")

    if not drivers_out:
        raise ValueError("No driver data retrieved from OpenF1")

    return {"circuit": circuit, "year": year, "corners": corners, "drivers": drivers_out}
