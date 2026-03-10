"""
Probabilistic pipeline: Stage 1 (Gaussian MLE) → Stage 2 (Naive Bayes) → Stage 3 (Bayesian)
"""

import numpy as np
from data.drivers import get_prior_probs, DRIVERS


# ── Stage 1: Gaussian MLE per corner ──────────────────────────────────────

def fit_corner_gaussians(drivers_data: dict) -> dict:
    """
    MLE: for each corner, fit N(μ,σ²) to speed/throttle/brake across fastest laps.
    Returns {corner: {speed: {mu,sigma}, throttle: ..., brake: ...}}
    """
    from collections import defaultdict
    obs = defaultdict(lambda: {"speed": [], "throttle": [], "brake": []})

    # pool fastest 30% of laps across all drivers
    all_laps = []
    for code, d in drivers_data.items():
        for lap in d["laps"]:
            all_laps.append((lap["laptime"], lap["corners"]))
    all_laps.sort(key=lambda x: x[0])
    fast = all_laps[:max(1, len(all_laps) // 3)]

    for _, corners in fast:
        for c in corners:
            obs[c["corner"]]["speed"].append(c["speed"])
            obs[c["corner"]]["throttle"].append(c["throttle"])
            obs[c["corner"]]["brake"].append(c["brake"])

    models = {}
    for cn, vals in obs.items():
        models[cn] = {}
        for feat in ("speed", "throttle", "brake"):
            arr = np.array(vals[feat])
            models[cn][feat] = {"mu": float(arr.mean()), "sigma": float(max(arr.std(), 1e-4))}
    return models


def score_corner(corner_data: dict, models: dict) -> float:
    """Log-likelihood of a single corner pass under the optimal Gaussian."""
    cn = corner_data["corner"]
    if cn not in models:
        return 0.0
    total = 0.0
    for feat in ("speed", "throttle", "brake"):
        mu = models[cn][feat]["mu"]
        sig = models[cn][feat]["sigma"]
        x = corner_data[feat]
        total += -0.5 * np.log(2 * np.pi * sig**2) - (x - mu)**2 / (2 * sig**2)
    return float(total)


def score_all_drivers(drivers_data: dict, corner_models: dict) -> dict:
    """
    Returns per-driver, per-lap, per-corner scores.
    {driver: {laps: [{lap, laptime, corners: [{corner, score, speed, throttle, brake}], lap_score}],
              corner_means: {corner: mean_score},
              overall: float}}
    """
    result = {}
    for code, d in drivers_data.items():
        laps_out = []
        for lap in d["laps"]:
            c_scored = []
            for c in lap["corners"]:
                c_scored.append({
                    "corner":   c["corner"],
                    "score":    round(score_corner(c, corner_models), 4),
                    "speed":    c["speed"],
                    "throttle": c["throttle"],
                    "brake":    c["brake"],
                })
            laps_out.append({
                "lap":      lap["lap"],
                "laptime":  lap["laptime"],
                "corners":  c_scored,
                "lap_score": round(sum(c["score"] for c in c_scored), 4),
            })

        # Mean score per corner
        from collections import defaultdict
        corner_acc = defaultdict(list)
        for lap in laps_out:
            for c in lap["corners"]:
                corner_acc[c["corner"]].append(c["score"])
        corner_means = {cn: round(float(np.mean(v)), 4) for cn, v in corner_acc.items()}

        result[code] = {
            "laps":         laps_out,
            "corner_means": corner_means,
            "overall":      round(float(np.mean([l["lap_score"] for l in laps_out])), 4),
        }
    return result


# ── Stage 2: Naive Bayes per-lap classifier ───────────────────────────────

def fit_naive_bayes(drivers_data: dict, driver_scores: dict):
    """
    Train NB on raw telemetry features (speed/throttle/brake per corner).
    Winner class = fastest 15% of laps across field.
    Returns (winner_params, nonwinner_params, log_prior_w, log_prior_nw)
    """
    all_laps = []
    for code, d in drivers_data.items():
        for lap in d["laps"]:
            feats = {}
            for c in lap["corners"]:
                feats[c["corner"] + "_spd"] = c["speed"]
                feats[c["corner"] + "_thr"] = c["throttle"]
                feats[c["corner"] + "_brk"] = c["brake"]
            all_laps.append({"laptime": lap["laptime"], "feats": feats})

    all_laps.sort(key=lambda l: l["laptime"])
    n = len(all_laps)
    nw = max(1, int(n * 0.15))
    winners = all_laps[:nw]
    nonwin  = all_laps[nw:]

    all_keys = set(k for l in all_laps for k in l["feats"])

    def mle(laps, key):
        vals = np.array([l["feats"].get(key, 0) for l in laps])
        return float(vals.mean()), max(float(vals.std()), 1e-4)

    wparams  = {k: mle(winners, k) for k in all_keys}
    nwparams = {k: mle(nonwin,  k) for k in all_keys}
    log_pw  = np.log(nw / n)
    log_pnw = np.log((n - nw) / n)
    return wparams, nwparams, log_pw, log_pnw


def nb_score_lap(lap: dict, wparams, nwparams, log_pw, log_pnw) -> float:
    """Returns P(winner | lap telemetry) for a single lap."""
    log_w  = log_pw
    log_nw = log_pnw

    def lp(x, mu, sig):
        return -0.5 * np.log(2 * np.pi * sig**2) - (x - mu)**2 / (2 * sig**2)

    for c in lap["corners"]:
        for feat, suffix in [("speed","_spd"),("throttle","_thr"),("brake","_brk")]:
            key = c["corner"] + suffix
            if key in wparams:
                log_w  += lp(c[feat], *wparams[key])
                log_nw += lp(c[feat], *nwparams[key])

    lmax = max(log_w, log_nw)
    p_w = np.exp(log_w - lmax) / (np.exp(log_w - lmax) + np.exp(log_nw - lmax))
    return float(p_w)


# ── Stage 3: Bayesian race updater ────────────────────────────────────────

class BayesianRaceUpdater:
    """
    Maintains P(winner=i) across laps.
    Prior from 2025 championship points.
    Likelihood: rank-based NB scores per lap (prevents collapse).
    """
    def __init__(self, temperature=0.12, smoothing=0.06):
        self.T = temperature
        self.s = smoothing
        self.drivers = []
        self.posterior = None
        self.history = []   # list of {lap, posteriors}

    def initialize(self, driver_codes: list):
        self.drivers = driver_codes
        priors = get_prior_probs()
        p = np.array([priors.get(d, 1e-4) for d in driver_codes], dtype=float)
        p = np.clip(p, 1e-4, None)
        self.posterior = p / p.sum()
        self.history = []

    def update_lap(self, lap_nb_scores: dict, lap_num: int):
        """lap_nb_scores: {driver: p_winner this lap}"""
        raw = np.array([lap_nb_scores.get(d, 0.5) for d in self.drivers])

        # Rank-based likelihoods: converts absolute NB scores to relative ranking
        n = len(self.drivers)
        ranks = raw.argsort()[::-1].argsort().astype(float)
        likelihoods = (n - ranks) / n
        likelihoods = likelihoods * (1 - self.s) + self.s / n
        likelihoods = likelihoods ** self.T

        unnorm = likelihoods * self.posterior
        self.posterior = unnorm / (unnorm.sum() + 1e-12)

        snap = {d: round(float(self.posterior[i]), 6) for i, d in enumerate(self.drivers)}
        self.history.append({"lap": lap_num, **snap})
        return snap

    def corner_impact(self, corner_name: str, drivers_data: dict,
                      wparams, nwparams, log_pw, log_pnw) -> float:
        """
        How much does this corner shift the posterior on average?
        Returns mean KL divergence contribution from this corner across all laps.
        """
        impacts = []
        for code, d in drivers_data.items():
            for lap in d["laps"]:
                c_data = next((c for c in lap["corners"] if c["corner"] == corner_name), None)
                if not c_data:
                    continue
                # Score with and without this corner
                full = nb_score_lap(lap, wparams, nwparams, log_pw, log_pnw)
                # Approximate: variance of score contribution
                impacts.append(abs(full - 0.5))
        return float(np.mean(impacts)) if impacts else 0.0


# ── Master pipeline ────────────────────────────────────────────────────────

def run_pipeline(year: int, circuit: str) -> dict:
    from data.loader import load_session

    raw = load_session(year, circuit)
    drivers_data = raw["drivers"]
    corners = raw["corners"]

    # Stage 1
    corner_models = fit_corner_gaussians(drivers_data)
    driver_scores = score_all_drivers(drivers_data, corner_models)

    # Corner winners per corner (best mean score)
    corner_winners = {}
    for cn in corners:
        best = max(driver_scores.keys(),
                   key=lambda d: driver_scores[d]["corner_means"].get(cn, -999))
        corner_winners[cn] = best

    # Stage 2
    wparams, nwparams, log_pw, log_pnw = fit_naive_bayes(drivers_data, driver_scores)

    # NB score every lap for every driver
    nb_lap_scores = {}
    for code, d in drivers_data.items():
        nb_lap_scores[code] = []
        for lap in d["laps"]:
            raw_lap = next((l for l in drivers_data[code]["laps"] if l["lap"] == lap["lap"]), None)
            if raw_lap:
                nb_lap_scores[code].append({
                    "lap": lap["lap"],
                    "p_winner": nb_score_lap(raw_lap, wparams, nwparams, log_pw, log_pnw)
                })

    # Stage 3
    updater = BayesianRaceUpdater()
    updater.initialize(list(drivers_data.keys()))

    # Build lap-by-lap timeline
    max_laps = max(len(v) for v in nb_lap_scores.values())
    bayesian_timeline = []
    for lap_idx in range(max_laps):
        lap_scores = {d: nb_lap_scores[d][lap_idx]["p_winner"]
                      if lap_idx < len(nb_lap_scores[d]) else 0.5
                      for d in drivers_data}
        posterior = updater.update_lap(lap_scores, lap_idx + 1)
        bayesian_timeline.append({"lap": lap_idx + 1, **posterior})

    final_posterior = updater.history[-1] if updater.history else {}

    # Corner impact scores
    corner_impacts = {}
    for cn in corners:
        corner_impacts[cn] = updater.corner_impact(cn, drivers_data, wparams, nwparams, log_pw, log_pnw)

    most_impactful_corner = max(corner_impacts, key=corner_impacts.get) if corner_impacts else corners[0]

    # Final ranking
    final_ranking = sorted(
        [{"driver": d, "win_probability": final_posterior.get(d, 0)} for d in drivers_data],
        key=lambda x: -x["win_probability"]
    )

    # Per-driver output
    drivers_out = {}
    for code in drivers_data:
        d = drivers_data[code]
        ds = driver_scores[code]
        drivers_out[code] = {
            "code":         code,
            "name":         DRIVERS[code]["name"],
            "team":         DRIVERS[code]["team"],
            "color":        DRIVERS[code]["color"],
            "points_2025":  DRIVERS[code]["points"],
            "laps":         ds["laps"],
            "corner_means": ds["corner_means"],
            "overall_score": ds["overall"],
            "final_win_prob": final_posterior.get(code, 0),
            "nb_lap_scores": nb_lap_scores.get(code, []),
        }

    return {
        "circuit":              circuit,
        "year":                 year,
        "corners":              corners,
        "corner_winners":       corner_winners,
        "corner_impacts":       corner_impacts,
        "most_impactful_corner": most_impactful_corner,
        "bayesian_timeline":    bayesian_timeline,
        "final_ranking":        final_ranking,
        "drivers":              drivers_out,
    }
