"""
Orchestrator: runs all three stages and returns a unified results payload.
"""

from data.loader import load_session, TOP_DRIVERS, CIRCUIT_CORNERS
from models.stage1_gaussian import GaussianCornerStage
from models.stage2_naive_bayes import NaiveBayesLapClassifier
from models.stage3_bayesian import BayesianRaceUpdater


def run_pipeline(year: int = 2024, circuit: str = "Bahrain", session_type: str = "R") -> dict:
    """
    Full pipeline. Returns a single JSON-serializable dict with all results.
    """

    # ── Load data ──────────────────────────────────────────────────────────
    session_data = load_session(year, circuit, session_type)
    drivers_data = session_data["drivers"]

    # ── Stage 1: Gaussian MLE per corner ──────────────────────────────────
    stage1 = GaussianCornerStage()
    stage1.fit(drivers_data)

    stage1_scores = {}
    for code, driver_data in drivers_data.items():
        stage1_scores[code] = stage1.score_driver(driver_data)

    corner_models = stage1.get_corner_models_dict()

    # Corner winners (highest Stage 1 score per corner)
    stage1_corner_winners = {}
    all_corners = list(CIRCUIT_CORNERS.get(circuit, CIRCUIT_CORNERS["Bahrain"]))
    for corner in all_corners:
        best = max(
            stage1_scores.keys(),
            key=lambda d: stage1_scores[d]["corner_scores"].get(corner, -999)
        )
        stage1_corner_winners[corner] = best

    # ── Stage 2: Naive Bayes classifier ───────────────────────────────────
    nb = NaiveBayesLapClassifier()
    nb.fit(drivers_data, stage1_scores)

    nb_lap_scores = {}
    for code, driver_data in drivers_data.items():
        nb_lap_scores[code] = nb.score_driver_all_laps(driver_data, stage1_scores[code])

    race_winner_probs = nb.predict_race_winner_prob(nb_lap_scores)

    # ── Stage 3: Bayesian updater ──────────────────────────────────────────
    updater = BayesianRaceUpdater()
    updater.initialize(race_winner_probs)
    bayesian_timeline = updater.run_full_race(nb_lap_scores)
    final_ranking = updater.get_final_ranking()

    # ── Assemble output ────────────────────────────────────────────────────
    drivers_out = {}
    for code in TOP_DRIVERS:
        if code not in drivers_data:
            continue
        d = drivers_data[code]
        drivers_out[code] = {
            "code":       code,
            "name":       d["name"],
            "team_color": d["team_color"],
            "stage1": {
                "overall_score":  stage1_scores[code]["overall_score"],
                "corner_scores":  stage1_scores[code]["corner_scores"],
                "lap_scores":     stage1_scores[code]["lap_scores"],
            },
            "stage2": {
                "race_win_prob": race_winner_probs.get(code, 0),
                "lap_scores":    nb_lap_scores.get(code, []),
            },
            "stage3": {
                "final_win_prob": next(
                    (r["win_probability"] for r in final_ranking if r["driver"] == code), 0
                ),
            },
            "laps": d["laps"],
        }

    return {
        "circuit":               circuit,
        "year":                  year,
        "corners":               all_corners,
        "corner_models":         corner_models,
        "corner_winners":        stage1_corner_winners,
        "bayesian_timeline":     bayesian_timeline,
        "final_ranking":         final_ranking,
        "drivers":               drivers_out,
    }
