"""
Stage 2: Naive Bayes Lap Classifier
─────────────────────────────────────
Given a driver's corner execution scores for a lap, classify whether
that lap profile belongs to a "race winner" or "non-winner".

Model:
    P(winner | x₁,...,xₙ) ∝ P(winner) · ∏ᵢ P(xᵢ | winner)

where each feature xᵢ (corner score) is modeled as Gaussian:
    P(xᵢ | winner)     ~ N(μᵢ_w,  σᵢ_w²)
    P(xᵢ | non-winner) ~ N(μᵢ_nw, σᵢ_nw²)

The class prior P(winner) = 1 / n_drivers (uniform over field).

Training: "winner" laps = top 15% of laps by laptime across all drivers.
          "non-winner" = remaining 85%.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class GaussianFeature:
    mu: float
    sigma: float

    def log_prob(self, x: float) -> float:
        if self.sigma < 1e-9:
            return 0.0
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - ((x - self.mu)**2) / (2 * self.sigma**2)


class NaiveBayesLapClassifier:
    """
    Binary Naive Bayes: winner vs non-winner lap profiles.
    Features: per-corner Stage 1 log-likelihood scores.
    """

    def __init__(self, winner_threshold: float = 0.15):
        self.winner_threshold = winner_threshold   # top X% of laps = "winner class"
        self.winner_features:     dict[str, GaussianFeature] = {}
        self.nonwinner_features:  dict[str, GaussianFeature] = {}
        self.log_prior_winner:    float = 0.0
        self.log_prior_nonwinner: float = 0.0
        self.fitted = False

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, all_drivers_data: dict, stage1_scores: dict) -> None:
        """
        all_drivers_data: raw telemetry dict {driver_code: {...}}
        stage1_scores:    {driver_code: {"lap_scores": [...], "corner_scores": {...}}}
        """
        # Build flat list of (laptime, driver, lap_idx, corner_scores_dict)
        all_laps = []
        for code, driver_data in all_drivers_data.items():
            for lap in driver_data["laps"]:
                # Get per-corner scores for this lap from stage1
                corner_scores = {}
                for c in lap["corners"]:
                    cn = c["corner"]
                    # Use the mean score across laps as proxy (Stage 1 already computed)
                    s1 = stage1_scores.get(code, {})
                    score = s1.get("corner_scores", {}).get(cn, 0.0)
                    corner_scores[cn] = score

                all_laps.append({
                    "laptime":       lap["laptime"],
                    "corner_scores": corner_scores,
                })

        # Sort by laptime, label top winner_threshold as "winner"
        all_laps.sort(key=lambda l: l["laptime"])
        n = len(all_laps)
        n_winners = max(1, int(n * self.winner_threshold))

        winner_laps    = all_laps[:n_winners]
        nonwinner_laps = all_laps[n_winners:]

        # Class priors (log scale)
        self.log_prior_winner    = np.log(n_winners / n)
        self.log_prior_nonwinner = np.log((n - n_winners) / n)

        # Fit Gaussian per corner per class (MLE)
        all_corners = set()
        for lap in all_laps:
            all_corners.update(lap["corner_scores"].keys())

        for corner in all_corners:
            w_vals  = [l["corner_scores"].get(corner, 0.0) for l in winner_laps]
            nw_vals = [l["corner_scores"].get(corner, 0.0) for l in nonwinner_laps]

            self.winner_features[corner]    = self._fit_gaussian(w_vals)
            self.nonwinner_features[corner] = self._fit_gaussian(nw_vals)

        self.fitted = True

    @staticmethod
    def _fit_gaussian(values: list[float]) -> GaussianFeature:
        arr = np.array(values, dtype=float)
        mu = float(np.mean(arr))
        sigma = max(float(np.std(arr)), 1e-4)
        return GaussianFeature(mu=mu, sigma=sigma)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_lap(self, corner_scores: dict[str, float]) -> dict:
        """
        Returns P(winner | corner_scores) for a single lap profile.
        Uses log-sum for numerical stability.
        """
        assert self.fitted, "Call fit() first"

        log_p_winner    = self.log_prior_winner
        log_p_nonwinner = self.log_prior_nonwinner

        for corner, score in corner_scores.items():
            if corner in self.winner_features:
                log_p_winner    += self.winner_features[corner].log_prob(score)
                log_p_nonwinner += self.nonwinner_features[corner].log_prob(score)

        # Normalize via log-sum-exp
        log_max = max(log_p_winner, log_p_nonwinner)
        p_winner = np.exp(log_p_winner - log_max) / (
            np.exp(log_p_winner - log_max) + np.exp(log_p_nonwinner - log_max)
        )

        return {
            "p_winner":    round(float(p_winner), 4),
            "p_nonwinner": round(float(1 - p_winner), 4),
        }

    def score_driver_all_laps(self, driver_data: dict, stage1_scores: dict) -> list[dict]:
        """Score every lap for a driver."""
        results = []
        corner_mean_scores = stage1_scores.get("corner_scores", {})

        for lap in driver_data["laps"]:
            corner_scores = {c["corner"]: corner_mean_scores.get(c["corner"], 0.0)
                             for c in lap["corners"]}
            pred = self.predict_lap(corner_scores)
            results.append({
                "lap":      lap["lap"],
                "laptime":  lap["laptime"],
                **pred,
            })

        return results

    def predict_race_winner_prob(self, all_nb_scores: dict[str, list[dict]]) -> dict[str, float]:
        """
        Aggregate lap-level winner probs into a race-level winner probability per driver.
        Uses mean p_winner across all laps as a simple aggregation.
        """
        result = {}
        for code, laps in all_nb_scores.items():
            if laps:
                result[code] = round(float(np.mean([l["p_winner"] for l in laps])), 4)
            else:
                result[code] = 0.0

        # Normalize to sum to 1
        total = sum(result.values())
        if total > 0:
            result = {k: round(v / total, 4) for k, v in result.items()}

        return result
