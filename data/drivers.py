"""
2026 F1 driver registry — 22 drivers across 11 teams.

avg_finish: estimated 2026 prior based on 2025 performance and team/driver changes.
"""

DRIVERS = {
    # McLaren
    "NOR": {"name": "Lando Norris",       "team": "McLaren",       "color": "#FF8000", "avg_finish": 2.8},
    "PIA": {"name": "Oscar Piastri",      "team": "McLaren",       "color": "#FF8000", "avg_finish": 3.5},
    # Mercedes
    "RUS": {"name": "George Russell",     "team": "Mercedes",      "color": "#27F4D2", "avg_finish": 3.2},
    "ANT": {"name": "Kimi Antonelli",     "team": "Mercedes",      "color": "#27F4D2", "avg_finish": 8.0},
    # Ferrari
    "LEC": {"name": "Charles Leclerc",    "team": "Ferrari",       "color": "#E8002D", "avg_finish": 4.5},
    "HAM": {"name": "Lewis Hamilton",     "team": "Ferrari",       "color": "#E8002D", "avg_finish": 5.5},
    # Red Bull
    "VER": {"name": "Max Verstappen",     "team": "Red Bull",      "color": "#3671C6", "avg_finish": 4.0},
    "HAD": {"name": "Isack Hadjar",       "team": "Red Bull",      "color": "#3671C6", "avg_finish": 10.0},
    # Williams
    "SAI": {"name": "Carlos Sainz",       "team": "Williams",      "color": "#64C4FF", "avg_finish": 6.5},
    "ALB": {"name": "Alex Albon",         "team": "Williams",      "color": "#64C4FF", "avg_finish": 9.5},
    # Aston Martin
    "ALO": {"name": "Fernando Alonso",    "team": "Aston Martin",  "color": "#358C75", "avg_finish": 8.5},
    "STR": {"name": "Lance Stroll",       "team": "Aston Martin",  "color": "#358C75", "avg_finish": 13.0},
    # Alpine
    "GAS": {"name": "Pierre Gasly",       "team": "Alpine",        "color": "#FF69B4", "avg_finish": 10.5},
    "COL": {"name": "Franco Colapinto",   "team": "Alpine",        "color": "#FF69B4", "avg_finish": 13.0},
    # Audi (formerly Sauber)
    "HUL": {"name": "Nico Hulkenberg",    "team": "Audi",          "color": "#B6BABD", "avg_finish": 11.5},
    "BOR": {"name": "Gabriel Bortoleto",  "team": "Audi",          "color": "#B6BABD", "avg_finish": 12.5},
    # Haas
    "OCO": {"name": "Esteban Ocon",       "team": "Haas",          "color": "#B6122A", "avg_finish": 12.0},
    "BEA": {"name": "Oliver Bearman",     "team": "Haas",          "color": "#B6122A", "avg_finish": 13.5},
    # Racing Bulls
    "LAW": {"name": "Liam Lawson",        "team": "Racing Bulls",  "color": "#6692FF", "avg_finish": 11.0},
    "LIN": {"name": "Arvid Lindblad",     "team": "Racing Bulls",  "color": "#6692FF", "avg_finish": 14.0},
    # Cadillac (new team)
    "BOT": {"name": "Valtteri Bottas",    "team": "Cadillac",      "color": "#CC1E4A", "avg_finish": 14.5},
    "PER": {"name": "Sergio Perez",       "team": "Cadillac",      "color": "#CC1E4A", "avg_finish": 15.0},
}


def get_prior_probs():
    """Return normalized prior win probabilities from avg_finish."""
    import numpy as np
    codes  = list(DRIVERS.keys())
    scores = np.array([1.0 / DRIVERS[c]["avg_finish"] ** 1.5 for c in codes])
    scores /= scores.sum()
    return dict(zip(codes, scores))
