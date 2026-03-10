"""
Stage 3: Bayesian Win Probability Updater
──────────────────────────────────────────
Treats the race as a sequence of evidence observations (lap results).
We maintain a posterior over "who is winning" and update it each lap.

Model:
    Prior:    P(winner = dᵢ) from Stage 2 race-level predictions
    Evidence: after each lap, the driver's Stage 2 p_winner score is a likelihood
    Update:   P(winner = dᵢ | lap_k) ∝ P(lap_k | winner = dᵢ) · P(winner = dᵢ | lap_{k-1})

Likelihood model:
    If driver i is the true winner, we expect high p_winner scores.
    We model: P(lap_score | winner=i) = Bern(lap_score) (treating score as hit prob)
    In practice: likelihood = score for winner, (1-score) for others -- but we
    use a smoother: likelihood = softmax over scores each lap.

This gives a probability timeline per driver across all laps.
"""

import numpy as np
from typing import Optional


class BayesianRaceUpdater:
    """
    Maintains a posterior distribution over race winner,
    updated lap-by-lap using Naive Bayes lap scores as likelihoods.
    """

    def __init__(self, smoothing: float = 0.05):
        self.smoothing = smoothing   # Laplace-style smoothing on likelihoods
        self.drivers: list[str] = []
        self.prior: np.ndarray = np.array([])
        self.posterior_history: list[dict] = []   # one entry per lap
        self.fitted = False

    def initialize(self, prior_probs: dict[str, float]) -> None:
        """Set prior from Stage 2 race-level winner probabilities."""
        self.drivers = list(prior_probs.keys())
        probs = np.array([prior_probs[d] for d in self.drivers], dtype=float)

        # Normalize + add small floor to avoid zero-probability drivers
        probs = np.clip(probs, 1e-4, None)
        self.prior = probs / probs.sum()
        self.posterior_history = []
        self.fitted = True

    def update(self, lap_nb_scores: dict[str, float]) -> dict[str, float]:
        """
        Bayesian update for one lap.
        lap_nb_scores: {driver_code: p_winner_this_lap}
        Returns updated posterior {driver_code: probability}.
        """
        assert self.fitted, "Call initialize() first"

        current_posterior = (
            self.prior.copy() if not self.posterior_history
            else np.array([self.posterior_history[-1][d] for d in self.drivers])
        )

        # Likelihood: how consistent is this driver's lap score with being winner?
        likelihoods = np.array([
            lap_nb_scores.get(d, 0.5) for d in self.drivers
        ], dtype=float)

        # Smooth likelihoods
        likelihoods = likelihoods * (1 - self.smoothing) + self.smoothing / len(self.drivers)

        # Bayesian update: posterior ∝ likelihood × prior
        # Use temperature dampening to prevent over-confident collapse
        temperature = 0.3   # < 1 softens the update; prevents one driver dominating early
        dampened_likelihoods = likelihoods ** temperature
        unnormalized = dampened_likelihoods * current_posterior
        updated = unnormalized / (unnormalized.sum() + 1e-12)

        result = {d: round(float(updated[i]), 6) for i, d in enumerate(self.drivers)}
        self.posterior_history.append(result)
        return result

    def run_full_race(self, all_nb_lap_scores: dict[str, list[dict]]) -> list[dict]:
        """
        Run Bayesian updates across all laps.
        Returns list of {lap: N, driver1: p, driver2: p, ...} per lap.
        """
        assert self.fitted, "Call initialize() first"

        # Find max laps
        max_laps = max(len(v) for v in all_nb_lap_scores.values()) if all_nb_lap_scores else 0

        timeline = []
        for lap_idx in range(max_laps):
            # Gather this lap's score per driver
            lap_scores = {}
            for driver, laps in all_nb_lap_scores.items():
                if lap_idx < len(laps):
                    lap_scores[driver] = laps[lap_idx]["p_winner"]
                else:
                    lap_scores[driver] = 0.5   # missing lap: neutral

            posterior = self.update(lap_scores)
            timeline.append({"lap": lap_idx + 1, **posterior})

        return timeline

    def get_final_ranking(self) -> list[dict]:
        """Return final posterior as a sorted ranking."""
        if not self.posterior_history:
            return []
        final = self.posterior_history[-1]
        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        return [{"driver": d, "win_probability": round(p, 4)} for d, p in ranked]

    def get_corner_winner_prediction(self, stage1_corner_scores: dict[str, dict]) -> dict[str, str]:
        """
        Stage 1 extension: for each corner, which driver has the highest
        mean log-likelihood score? That corner's predicted winner is that driver.
        Returns {corner_name: driver_code}
        """
        all_corners = set()
        for scores in stage1_corner_scores.values():
            all_corners.update(scores.get("corner_scores", {}).keys())

        corner_winners = {}
        for corner in all_corners:
            best_driver = max(
                stage1_corner_scores.keys(),
                key=lambda d: stage1_corner_scores[d].get("corner_scores", {}).get(corner, -999)
            )
            corner_winners[corner] = best_driver

        return corner_winners
