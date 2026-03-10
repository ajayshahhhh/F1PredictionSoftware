"""
Stage 1: Gaussian MLE Corner Model
─────────────────────────────────
For each corner, we model three telemetry features as independent Gaussians:
    speed     ~ N(μ_s, σ_s²)
    throttle  ~ N(μ_t, σ_t²)
    brake     ~ N(μ_b, σ_b²)

MLE estimates:
    μ_hat = (1/n) Σ x_i
    σ²_hat = (1/n) Σ (x_i - μ_hat)²

A driver's corner execution score is the mean log-likelihood of their
telemetry under the "optimal" distribution (fitted on the fastest laps
across all drivers). Higher score = closer to optimal corner execution.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GaussianParams:
    mu: float
    sigma: float

    def log_prob(self, x: float) -> float:
        """Log-likelihood of observation x under N(mu, sigma²)."""
        if self.sigma < 1e-9:
            return 0.0
        return -0.5 * np.log(2 * np.pi * self.sigma**2) - ((x - self.mu)**2) / (2 * self.sigma**2)

    def to_dict(self) -> dict:
        return {"mu": round(self.mu, 4), "sigma": round(self.sigma, 4)}


@dataclass
class CornerModel:
    corner: str
    speed:    GaussianParams = field(default_factory=lambda: GaussianParams(0, 1))
    throttle: GaussianParams = field(default_factory=lambda: GaussianParams(0, 1))
    brake:    GaussianParams = field(default_factory=lambda: GaussianParams(0, 1))

    def score(self, speed: float, throttle: float, brake: float) -> float:
        """Combined log-likelihood score for a single corner pass."""
        return (
            self.speed.log_prob(speed) +
            self.throttle.log_prob(throttle) +
            self.brake.log_prob(brake)
        )


class GaussianCornerStage:
    """
    Fits optimal Gaussian distributions per corner using MLE,
    then scores each driver against the optimal.
    """

    def __init__(self):
        self.corner_models: dict[str, CornerModel] = {}
        self.fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit(self, all_drivers_data: dict, top_n_laps: int = 5) -> None:
        """
        Fit optimal Gaussian per corner from the top-n fastest laps
        across all drivers (MLE on pooled 'fast' observations).
        """
        # Collect all laps sorted by laptime
        all_laps = []
        for driver_data in all_drivers_data.values():
            for lap in driver_data["laps"]:
                all_laps.append(lap)

        all_laps.sort(key=lambda l: l["laptime"])
        fast_laps = all_laps[:max(top_n_laps, len(all_laps) // 5)]

        # Aggregate observations per corner
        corner_observations: dict[str, dict[str, list]] = {}
        for lap in fast_laps:
            for c in lap["corners"]:
                cn = c["corner"]
                if cn not in corner_observations:
                    corner_observations[cn] = {"speed": [], "throttle": [], "brake": []}
                corner_observations[cn]["speed"].append(c["speed"])
                corner_observations[cn]["throttle"].append(c["throttle"])
                corner_observations[cn]["brake"].append(c["brake"])

        # MLE: μ = mean, σ = std (biased, as per MLE derivation)
        for corner_name, obs in corner_observations.items():
            self.corner_models[corner_name] = CornerModel(
                corner=corner_name,
                speed    = self._mle_gaussian(obs["speed"]),
                throttle = self._mle_gaussian(obs["throttle"]),
                brake    = self._mle_gaussian(obs["brake"]),
            )

        self.fitted = True

    @staticmethod
    def _mle_gaussian(values: list[float]) -> GaussianParams:
        """MLE estimate for Gaussian: μ̂ = mean, σ̂² = (1/n)Σ(xᵢ-μ̂)²"""
        arr = np.array(values, dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))          # biased MLE estimator
        sigma = max(sigma, 1e-4)            # numerical stability
        return GaussianParams(mu=mu, sigma=sigma)

    # ── Scoring ────────────────────────────────────────────────────────────

    def score_driver(self, driver_data: dict) -> dict:
        """
        Returns per-corner scores and per-lap aggregate scores for a driver.
        """
        assert self.fitted, "Call fit() before score_driver()"

        corner_scores_all: dict[str, list[float]] = {}
        lap_scores = []

        for lap in driver_data["laps"]:
            lap_log_ll = 0.0
            for c in lap["corners"]:
                cn = c["corner"]
                if cn not in self.corner_models:
                    continue
                model = self.corner_models[cn]
                s = model.score(c["speed"], c["throttle"], c["brake"])
                lap_log_ll += s
                corner_scores_all.setdefault(cn, []).append(s)

            lap_scores.append({
                "lap":   lap["lap"],
                "score": round(lap_log_ll, 4),
                "laptime": lap["laptime"],
            })

        # Mean score per corner across all laps
        corner_mean_scores = {
            cn: round(float(np.mean(scores)), 4)
            for cn, scores in corner_scores_all.items()
        }

        return {
            "lap_scores":    lap_scores,
            "corner_scores": corner_mean_scores,
            "overall_score": round(float(np.mean([l["score"] for l in lap_scores])), 4),
        }

    def get_corner_models_dict(self) -> dict:
        return {
            cn: {
                "speed":    m.speed.to_dict(),
                "throttle": m.throttle.to_dict(),
                "brake":    m.brake.to_dict(),
            }
            for cn, m in self.corner_models.items()
        }
