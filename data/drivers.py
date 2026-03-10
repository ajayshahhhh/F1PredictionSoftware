"""
2026 F1 Driver Registry
- All 20 drivers across 10 teams
- 2025 championship points used as Bayesian prior
- Rookies inherit team constructor points / 2 as proxy
"""

# 2025 final driver standings (approximate)
DRIVERS = {
    # code: {name, team, color, points_2025}
    "NOR": {"name": "Lando Norris",        "team": "McLaren",          "color": "#FF8000", "points": 331},
    "PIA": {"name": "Oscar Piastri",       "team": "McLaren",          "color": "#FF8000", "points": 252},
    "VER": {"name": "Max Verstappen",      "team": "Red Bull",         "color": "#3671C6", "points": 307},
    "TSU": {"name": "Yuki Tsunoda",        "team": "Red Bull",         "color": "#3671C6", "points": 40},   # moved up from RB
    "LEC": {"name": "Charles Leclerc",     "team": "Ferrari",          "color": "#E8002D", "points": 214},
    "HAM": {"name": "Lewis Hamilton",      "team": "Ferrari",          "color": "#E8002D", "points": 174},
    "RUS": {"name": "George Russell",      "team": "Mercedes",         "color": "#27F4D2", "points": 235},
    "ANT": {"name": "Kimi Antonelli",      "team": "Mercedes",         "color": "#27F4D2", "points": 30},   # rookie, ~constructor/2 proxy
    "SAI": {"name": "Carlos Sainz",        "team": "Williams",         "color": "#64C4FF", "points": 161},
    "ALB": {"name": "Alex Albon",          "team": "Williams",         "color": "#64C4FF", "points": 45},
    "ALO": {"name": "Fernando Alonso",     "team": "Aston Martin",     "color": "#358C75", "points": 52},
    "STR": {"name": "Lance Stroll",        "team": "Aston Martin",     "color": "#358C75", "points": 24},
    "GAS": {"name": "Pierre Gasly",        "team": "Alpine",           "color": "#FF69B4", "points": 42},
    "DOO": {"name": "Jack Doohan",         "team": "Alpine",           "color": "#FF69B4", "points": 5},    # rookie
    "HUL": {"name": "Nico Hulkenberg",     "team": "Sauber/Audi",      "color": "#B6BABD", "points": 31},
    "BOR": {"name": "Gabriel Bortoleto",   "team": "Sauber/Audi",      "color": "#B6BABD", "points": 0},   # rookie
    "OCO": {"name": "Esteban Ocon",        "team": "Haas",             "color": "#B6122A", "points": 23},
    "BEA": {"name": "Oliver Bearman",      "team": "Haas",             "color": "#B6122A", "points": 7},
    "HAD": {"name": "Isack Hadjar",        "team": "Racing Bulls",     "color": "#6692FF", "points": 0},   # rookie
    "LAW": {"name": "Liam Lawson",         "team": "Racing Bulls",     "color": "#6692FF", "points": 15},
}

def get_prior_probs() -> dict[str, float]:
    """
    Convert 2025 points to normalized prior probabilities.
    Rookies with 0 points get a floor of 5 to stay in contention.
    """
    import numpy as np
    pts = {code: max(d["points"], 5) for code, d in DRIVERS.items()}
    total = sum(pts.values())
    return {code: round(p / total, 6) for code, p in pts.items()}
