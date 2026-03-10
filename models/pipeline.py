"""
Probabilistic pipeline — 4-source Bayesian updater.

Evidence hierarchy per lap (position-dominant):
  1. Race position     (weight ramps 0.35→0.60 as race progresses)
  2. Gap to leader     (weight 0.20, constant)
  3. Lap time vs field (weight 0.20, constant)
  4. Corner execution  (weight ramps 0.25→0.10 as race progresses)

Prior: avg finishing position from 2025 season → 1/pos^1.5, normalized.
"""

import numpy as np
from collections import defaultdict
from data.drivers import get_prior_probs, DRIVERS


# ── Stage 1: Gaussian MLE corner model ────────────────────────────────────

def fit_corner_gaussians(drivers_data):
    obs = defaultdict(lambda: {"speed": [], "throttle": [], "brake": []})
    all_laps = []
    for code, d in drivers_data.items():
        for lap in d["laps"]:
            all_laps.append((lap["laptime"], lap["corners"]))
    all_laps.sort(key=lambda x: x[0])
    fast = all_laps[:max(1, len(all_laps) // 4)]
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


def score_corner(c, models):
    cn = c["corner"]
    if cn not in models:
        return 0.0
    s = 0.0
    for feat in ("speed", "throttle", "brake"):
        mu  = models[cn][feat]["mu"]
        sig = models[cn][feat]["sigma"]
        x   = c[feat]
        s  += -0.5 * np.log(2 * np.pi * sig**2) - (x - mu)**2 / (2 * sig**2)
    return float(s)


def score_all_drivers(drivers_data, corner_models):
    result = {}
    for code, d in drivers_data.items():
        laps_out = []
        for lap in d["laps"]:
            c_scored = [{"corner": c["corner"], "score": round(score_corner(c, corner_models), 4),
                         "speed": c["speed"], "throttle": c["throttle"], "brake": c["brake"]}
                        for c in lap["corners"]]
            laps_out.append({
                "lap": lap["lap"], "laptime": lap["laptime"],
                "corners": c_scored,
                "lap_score": round(sum(c["score"] for c in c_scored), 4),
                "position": lap.get("position"),
                "gap":      lap.get("gap"),
            })
        corner_acc = defaultdict(list)
        for lap in laps_out:
            for c in lap["corners"]:
                corner_acc[c["corner"]].append(c["score"])
        result[code] = {
            "laps":         laps_out,
            "corner_means": {cn: round(float(np.mean(v)), 4) for cn, v in corner_acc.items()},
            "overall":      round(float(np.mean([l["lap_score"] for l in laps_out])), 4),
        }
    return result


# ── Stage 2: Naive Bayes lap classifier ───────────────────────────────────

def fit_naive_bayes(drivers_data):
    all_laps = []
    for code, d in drivers_data.items():
        for lap in d["laps"]:
            feats = {}
            for c in lap["corners"]:
                feats[c["corner"]+"_spd"] = c["speed"]
                feats[c["corner"]+"_thr"] = c["throttle"]
                feats[c["corner"]+"_brk"] = c["brake"]
            all_laps.append({"laptime": lap["laptime"], "feats": feats})
    all_laps.sort(key=lambda l: l["laptime"])
    n  = len(all_laps)
    nw = max(1, int(n * 0.15))
    winners = all_laps[:nw]
    nonwin  = all_laps[nw:]
    all_keys = set(k for l in all_laps for k in l["feats"])
    def mle(laps, key):
        vals = np.array([l["feats"].get(key, 0) for l in laps])
        return float(vals.mean()), max(float(vals.std()), 1e-4)
    wparams  = {k: mle(winners, k) for k in all_keys}
    nwparams = {k: mle(nonwin,  k) for k in all_keys}
    return wparams, nwparams, np.log(nw/n), np.log((n-nw)/n)


def nb_score_lap(lap, wparams, nwparams, log_pw, log_pnw):
    log_w, log_nw = log_pw, log_pnw
    def lp(x, mu, sig):
        return -0.5*np.log(2*np.pi*sig**2) - (x-mu)**2/(2*sig**2)
    for c in lap["corners"]:
        for feat, suf in [("speed","_spd"),("throttle","_thr"),("brake","_brk")]:
            key = c["corner"] + suf
            if key in wparams:
                log_w  += lp(c[feat], *wparams[key])
                log_nw += lp(c[feat], *nwparams[key])
    lmax = max(log_w, log_nw)
    return float(np.exp(log_w-lmax) / (np.exp(log_w-lmax) + np.exp(log_nw-lmax)))


# ── Stage 3: 4-source Bayesian race updater ───────────────────────────────

class BayesianRaceUpdater:
    """
    Update rule each lap:
        P(winner=i | lap_k) ∝ L_combined(lap_k, i) · P(winner=i | lap_{k-1})

    L_combined = L_position^w1 · L_gap^w2 · L_laptime^w3 · L_corner^w4

    Weights ramp with race progress:
        w1 (position): 0.35 → 0.60   (becomes dominant late race)
        w4 (corner):   0.25 → 0.10   (most useful early when position unstable)
        w2, w3 constant
    """

    def __init__(self, n_race_laps=57, smoothing=0.03):
        self.n_race_laps = n_race_laps
        self.smoothing   = smoothing
        self.drivers     = []
        self.posterior   = None
        self.history     = []

    def initialize(self, driver_codes):
        self.drivers = driver_codes
        n = len(driver_codes)
        priors = get_prior_probs()
        pts    = np.array([priors.get(d, 1/n) for d in driver_codes], dtype=float)
        pts    = np.clip(pts, 1e-6, None) / pts.sum()
        # 90% uniform + 10% avg-finish signal — prior is very weak, data dominates quickly
        self.posterior = 0.90 * np.ones(n)/n + 0.10 * pts
        self.posterior /= self.posterior.sum()
        self.history = []

    def _weights(self, lap_num):
        """Ramp position weight from 0.35 to 0.60 over the race."""
        t = min(lap_num / self.n_race_laps, 1.0)
        w_pos    = 0.35 + 0.25 * t      # 0.35 → 0.60
        w_gap    = 0.20
        w_lap    = 0.20
        w_corner = 1.0 - w_pos - w_gap - w_lap   # 0.25 → 0.00 (floored at 0.05)
        w_corner = max(w_corner, 0.05)
        # Re-normalise
        total = w_pos + w_gap + w_lap + w_corner
        return w_pos/total, w_gap/total, w_lap/total, w_corner/total

    def update_lap(self, lap_data: dict, lap_num: int):
        """
        lap_data: {driver: {"position": int|None, "gap": float|None,
                             "laptime": float, "corner_score": float, "nb_score": float}}

        Key design: each lap's likelihood is tempered by temperature T so that
        57 sequential updates don't collapse the posterior. The position signal
        is expressed as a rank-softmax, not raw exponential, preventing any
        single driver from zeroing out all others.
        """
        n   = len(self.drivers)
        eps = 1e-12
        w1, w2, w3, w4 = self._weights(lap_num)

        # Temperature: how much each single lap shifts the posterior.
        # Higher T = stronger per-lap evidence. Ramps from 0.25 early to 0.55 late.
        # Late-race position signal is strong enough to push a clear winner to 30-40%.
        T = 0.25 + 0.30 * min(lap_num / self.n_race_laps, 1.0)

        # ── Raw scores (higher = better for each metric) ──────────────────

        # Position: invert so P1 = 20, P20 = 1 (before softmax)
        pos_arr = np.array([lap_data.get(d, {}).get("position") or n for d in self.drivers], dtype=float)
        pos_score = (n + 1) - pos_arr          # P1 → 20, P20 → 1

        # Gap: negate so leader (0s gap) scores highest
        gap_arr = np.array([lap_data.get(d, {}).get("gap") or 60.0 for d in self.drivers], dtype=float)
        gap_arr = np.clip(gap_arr, 0, 120)
        gap_score = -gap_arr                   # 0 gap → 0, 60s gap → -60

        # Lap time: negate so fastest = highest score
        lt_arr  = np.array([lap_data.get(d, {}).get("laptime") or 90.0 for d in self.drivers], dtype=float)
        lt_score = -lt_arr

        # Corner execution
        cs_arr = np.array([lap_data.get(d, {}).get("corner_score") or 0.0 for d in self.drivers], dtype=float)

        # ── Rank-normalise each signal to [0,1] ───────────────────────────
        # This prevents any one metric from dominating via scale differences.
        def rank_norm(arr):
            ranks = arr.argsort()[::-1].argsort().astype(float)  # 0 = best
            return (n - 1 - ranks) / max(n - 1, 1)               # best → 1.0, worst → 0.0

        r_pos    = rank_norm(pos_score)
        r_gap    = rank_norm(gap_score)
        r_lap    = rank_norm(lt_score)
        r_corner = rank_norm(cs_arr)

        # ── Weighted combination → single likelihood per driver ────────────
        combined = w1*r_pos + w2*r_gap + w3*r_lap + w4*r_corner  # in [0,1]

        # Laplace smoothing: no driver ever gets 0 likelihood
        combined = combined * (1 - self.smoothing) + self.smoothing / n

        # ── Temperature-tempered Bayesian update ──────────────────────────
        # L^T dampens the per-lap evidence so posterior moves gradually.
        L_tempered = combined ** T

        unnorm = L_tempered * self.posterior
        self.posterior = unnorm / (unnorm.sum() + eps)

        snap = {d: round(float(self.posterior[i]), 6) for i, d in enumerate(self.drivers)}
        self.history.append({"lap": lap_num, **snap})
        return snap

    def _single_update(self, lap_data_override, lap_num):
        """
        Run one hypothetical Bayesian update from the CURRENT posterior
        without mutating state. Returns the resulting posterior dict.
        """
        import copy
        n   = len(self.drivers)
        eps = 1e-12
        w1, w2, w3, w4 = self._weights(lap_num)
        T = 0.25 + 0.30 * min(lap_num / self.n_race_laps, 1.0)

        pos_arr = np.array([lap_data_override.get(d, {}).get("position") or n
                            for d in self.drivers], dtype=float)
        pos_score = (n + 1) - pos_arr

        gap_arr = np.clip(np.array([lap_data_override.get(d, {}).get("gap") or 60.0
                                    for d in self.drivers], dtype=float), 0, 120)
        gap_score = -gap_arr

        lt_arr  = np.array([lap_data_override.get(d, {}).get("laptime") or 90.0
                            for d in self.drivers], dtype=float)
        lt_score = -lt_arr

        cs_arr = np.array([lap_data_override.get(d, {}).get("corner_score") or 0.0
                           for d in self.drivers], dtype=float)

        def rank_norm(arr):
            ranks = arr.argsort()[::-1].argsort().astype(float)
            return (n - 1 - ranks) / max(n - 1, 1)

        combined = (w1*rank_norm(pos_score) + w2*rank_norm(gap_score) +
                    w3*rank_norm(lt_score)  + w4*rank_norm(cs_arr))
        combined  = combined * (1 - self.smoothing) + self.smoothing / n
        L         = combined ** T
        unnorm    = L * self.posterior
        post      = unnorm / (unnorm.sum() + eps)
        return {d: float(post[i]) for i, d in enumerate(self.drivers)}

    def counterfactual_advice(self, driver_code, lap_data, lap_num):
        """
        Compute ΔP(win) for four counterfactual interventions on driver_code.
        Returns a ranked list of {factor, intervention, delta_p, current, counterfactual}.

        Interventions (realistic single-lap improvements):
          - Position:   gain 1 place (e.g. P5 → P4)
          - Gap:        cut gap to leader by 1.5s
          - Lap time:   set fastest lap of field this lap
          - Corners:    execute corners at 90th percentile of field this lap
        """
        if driver_code not in self.drivers:
            return []

        d_data  = lap_data.get(driver_code, {})
        base_p  = float(self.posterior[self.drivers.index(driver_code)])
        n       = len(self.drivers)
        results = []

        # Helper: compute ΔP for one counterfactual lap_data
        def delta(modified_data):
            post = self._single_update(modified_data, lap_num)
            return round((post[driver_code] - base_p) * 100, 2)   # in percentage points

        # ── 1. Position: gain 1 place ─────────────────────────────────────
        cur_pos = d_data.get("position") or n
        if cur_pos > 1:
            cf_pos  = cur_pos - 1
            mod     = {**lap_data, driver_code: {**d_data, "position": cf_pos}}
            dp      = delta(mod)
            results.append({
                "factor":          "Race Position",
                "intervention":    f"Gain 1 place (P{cur_pos} → P{cf_pos})",
                "delta_p":         dp,
                "current":         f"P{cur_pos}",
                "counterfactual":  f"P{cf_pos}",
            })
        else:
            results.append({
                "factor":         "Race Position",
                "intervention":   "Already leading — maintain P1",
                "delta_p":        0.0,
                "current":        "P1",
                "counterfactual": "P1",
            })

        # ── 2. Gap: cut gap to leader by 1.5s ────────────────────────────
        cur_gap = d_data.get("gap") or 0.0
        if cur_gap > 0.5:
            cf_gap = max(0.0, cur_gap - 1.5)
            mod    = {**lap_data, driver_code: {**d_data, "gap": cf_gap}}
            dp     = delta(mod)
            results.append({
                "factor":         "Gap to Leader",
                "intervention":   f"Cut gap by 1.5s ({cur_gap:.1f}s → {cf_gap:.1f}s)",
                "delta_p":        dp,
                "current":        f"{cur_gap:.1f}s",
                "counterfactual": f"{cf_gap:.1f}s",
            })
        else:
            results.append({
                "factor":         "Gap to Leader",
                "intervention":   "Already within 0.5s of leader",
                "delta_p":        0.0,
                "current":        f"{cur_gap:.1f}s",
                "counterfactual": f"{cur_gap:.1f}s",
            })

        # ── 3. Lap time: match fastest lap in field this lap ───────────────
        cur_lt  = d_data.get("laptime") or 90.0
        all_lts = [v.get("laptime") or 90.0 for v in lap_data.values() if v.get("laptime")]
        best_lt = min(all_lts) if all_lts else cur_lt
        if cur_lt > best_lt + 0.1:
            mod = {**lap_data, driver_code: {**d_data, "laptime": best_lt}}
            dp  = delta(mod)
            results.append({
                "factor":         "Lap Time",
                "intervention":   f"Match fastest lap ({cur_lt:.3f}s → {best_lt:.3f}s)",
                "delta_p":        dp,
                "current":        f"{cur_lt:.3f}s",
                "counterfactual": f"{best_lt:.3f}s",
            })
        else:
            results.append({
                "factor":         "Lap Time",
                "intervention":   "Already setting fastest lap",
                "delta_p":        0.0,
                "current":        f"{cur_lt:.3f}s",
                "counterfactual": f"{cur_lt:.3f}s",
            })

        # ── 4. Corner execution: find the single corner with highest ΔP ────
        # For each corner, compute how much win prob changes if driver matches
        # the best performer at that corner this lap.
        driver_corner_scores = d_data.get("corner_scores") or {}
        best_corner_dp    = 0.0
        best_corner_name  = None
        best_corner_cur   = 0.0
        best_corner_best  = 0.0
        best_corner_who   = None
        corner_best = {}   # {corner: (best_score, best_driver)}
        for drv, dv in lap_data.items():
            cs_dict = dv.get("corner_scores") or {}
            for cn, sc in cs_dict.items():
                if cn not in corner_best or sc > corner_best[cn][0]:
                    corner_best[cn] = (sc, drv)

        my_corner_scores = driver_corner_scores

        for cn, (best_sc, best_drv) in corner_best.items():
            my_sc = my_corner_scores.get(cn, 0.0)
            if best_sc <= my_sc + 0.1:
                continue   # already at/near best for this corner
            # Hypothetical: driver matches best at this corner
            # Approximate new overall corner_score by adding the improvement
            improvement   = best_sc - my_sc
            cur_cs        = d_data.get("corner_score") or 0.0
            n_corners     = max(len(my_corner_scores), 1)
            new_cs        = cur_cs + improvement / n_corners
            mod           = {**lap_data, driver_code: {**d_data, "corner_score": new_cs}}
            dp            = delta(mod)
            if dp > best_corner_dp:
                best_corner_dp   = dp
                best_corner_name = cn
                best_corner_cur  = my_sc
                best_corner_best = best_sc
                best_corner_who  = best_drv

        if best_corner_name:
            results.append({
                "factor":         "Corner Execution",
                "intervention":   f"Optimise {best_corner_name} (currently {best_corner_cur:.0f} vs {best_corner_who}'s {best_corner_best:.0f})",
                "delta_p":        round(best_corner_dp, 2),
                "current":        f"{best_corner_cur:.0f}",
                "counterfactual": f"{best_corner_best:.0f}",
            })
        else:
            cur_cs = d_data.get("corner_score") or 0.0
            results.append({
                "factor":         "Corner Execution",
                "intervention":   "Already leading all corners",
                "delta_p":        0.0,
                "current":        f"{cur_cs:.0f}",
                "counterfactual": f"{cur_cs:.0f}",
            })

        # Sort by ΔP descending
        results.sort(key=lambda x: -x["delta_p"])
        return results

    def corner_impact(self, corner_name, drivers_data, wparams, nwparams, log_pw, log_pnw):
        """
        Impact = variance of per-lap posterior shift attributable to this corner.
        Approximated by how much the NB score changes when this corner is excluded.
        """
        deltas = []
        for code, d in drivers_data.items():
            for lap in d["laps"]:
                full  = nb_score_lap(lap, wparams, nwparams, log_pw, log_pnw)
                # Score without this corner
                other = [c for c in lap["corners"] if c["corner"] != corner_name]
                if not other:
                    continue
                fake_lap = {**lap, "corners": other}
                without  = nb_score_lap(fake_lap, wparams, nwparams, log_pw, log_pnw)
                deltas.append(abs(full - without))
        return float(np.mean(deltas)) if deltas else 0.0


# ── Master pipeline ────────────────────────────────────────────────────────

def run_pipeline(year, circuit):
    from data.loader import load_session

    raw          = load_session(year, circuit)
    drivers_data = raw["drivers"]
    corners      = raw["corners"]

    # Stage 1: Gaussian MLE
    corner_models = fit_corner_gaussians(drivers_data)
    driver_scores = score_all_drivers(drivers_data, corner_models)

    corner_winners = {}
    for cn in corners:
        best = max(driver_scores.keys(),
                   key=lambda d: driver_scores[d]["corner_means"].get(cn, -999))
        corner_winners[cn] = best

    # Stage 2: Naive Bayes
    wparams, nwparams, log_pw, log_pnw = fit_naive_bayes(drivers_data)

    # Stage 3: Bayesian race updater
    n_laps  = max(len(d["laps"]) for d in drivers_data.values())
    updater = BayesianRaceUpdater(n_race_laps=n_laps)
    updater.initialize(list(drivers_data.keys()))

    bayesian_timeline = []
    advice_by_driver  = {}   # {driver_code: {lap_num: [advice_items]}}
    for lap_idx in range(n_laps):
        lap_data = {}
        for code, d in drivers_data.items():
            scored_laps = driver_scores[code]["laps"]
            if lap_idx < len(scored_laps):
                sl = scored_laps[lap_idx]
                raw_lap = d["laps"][lap_idx]
                nb = nb_score_lap(raw_lap, wparams, nwparams, log_pw, log_pnw)
                lap_data[code] = {
                    "position":      sl.get("position"),
                    "gap":           sl.get("gap"),
                    "laptime":       sl["laptime"],
                    "corner_score":  sl["lap_score"],
                    "corner_scores": {c["corner"]: c["score"] for c in sl["corners"]},
                    "nb_score":      nb,
                }
        snap = updater.update_lap(lap_data, lap_idx + 1)
        bayesian_timeline.append({"lap": lap_idx + 1, **snap})

        # Compute counterfactual advice for every driver at this lap
        for code in drivers_data:
            advice = updater.counterfactual_advice(code, lap_data, lap_idx + 1)
            if code not in advice_by_driver:
                advice_by_driver[code] = {}
            advice_by_driver[code][lap_idx + 1] = advice

    final_posterior = updater.history[-1] if updater.history else {}
    final_ranking   = sorted(
        [{"driver": d, "win_probability": final_posterior.get(d, 0)} for d in drivers_data],
        key=lambda x: -x["win_probability"]
    )

    # Corner impact
    corner_impacts = {}
    for cn in corners:
        corner_impacts[cn] = updater.corner_impact(cn, drivers_data, wparams, nwparams, log_pw, log_pnw)
    most_impactful_corner = max(corner_impacts, key=corner_impacts.get) if corner_impacts else corners[0]

    # Build per-driver output including position timeline for track view
    drivers_out = {}
    for code in drivers_data:
        d  = drivers_data[code]
        ds = driver_scores[code]
        # Position timeline: [{lap, position, gap}]
        pos_timeline = [
            {"lap": l["lap"], "position": l.get("position"), "gap": l.get("gap")}
            for l in d["laps"]
        ]
        drivers_out[code] = {
            "code":           code,
            "name":           DRIVERS[code]["name"],
            "team":           DRIVERS[code]["team"],
            "color":          DRIVERS[code]["color"],
            "avg_finish_2025": DRIVERS[code]["avg_finish"],
            "laps":           ds["laps"],
            "corner_means":   ds["corner_means"],
            "overall_score":  ds["overall"],
            "final_win_prob": final_posterior.get(code, 0),
            "pos_timeline":   pos_timeline,
            "advice":         advice_by_driver.get(code, {}),
        }

    return {
        "circuit":               circuit,
        "year":                  year,
        "corners":               corners,
        "corner_winners":        corner_winners,
        "corner_impacts":        corner_impacts,
        "most_impactful_corner": most_impactful_corner,
        "bayesian_timeline":     bayesian_timeline,
        "final_ranking":         final_ranking,
        "drivers":               drivers_out,
        "n_laps":                n_laps,
    }
