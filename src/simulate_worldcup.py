import pandas as pd
import joblib
import random
from collections import Counter

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/worldcup_xgb.pkl"
)

# ==========================================
# LOAD TEAM STATS
# ==========================================

stats = pd.read_csv(
    "data/team_stats.csv",
    index_col=0
)

# ==========================================
# WORLD CUP CONTENDERS
# ==========================================

teams = [
    "Argentina",
    "Brazil",
    "France",
    "Spain",
    "England",
    "Portugal",
    "Germany",
    "Netherlands"
]

# ==========================================
# PREDICT MATCH
# ==========================================

def predict_match(home_team, away_team):

    home = stats.loc[home_team]
    away = stats.loc[away_team]

    X = [[
        home["form"],
        away["form"],

        home["goals_scored"],
        away["goals_scored"],

        home["goals_conceded"],
        away["goals_conceded"],

        home["elo"],
        away["elo"],

        home["elo"] - away["elo"]
    ]]

    probs = model.predict_proba(X)[0]

    away_win = probs[0]
    draw = probs[1]
    home_win = probs[2]

    return home_win, draw, away_win

# ==========================================
# SIMULATE MATCH
# ==========================================

def simulate_match(team1, team2):

    home_win, draw, away_win = predict_match(
        team1,
        team2
    )

    r = random.random()

    if r < away_win:
        return team2

    elif r < away_win + draw:

        # Knockout draw
        # Random penalty winner

        return random.choice(
            [team1, team2]
        )

    return team1

# ==========================================
# SIMULATE TOURNAMENT
# ==========================================

def simulate_tournament():

    quarterfinals = teams.copy()

    random.shuffle(
        quarterfinals
    )

    semifinals = []

    for i in range(
        0,
        len(quarterfinals),
        2
    ):

        winner = simulate_match(
            quarterfinals[i],
            quarterfinals[i + 1]
        )

        semifinals.append(
            winner
        )

    finals = []

    for i in range(
        0,
        len(semifinals),
        2
    ):

        winner = simulate_match(
            semifinals[i],
            semifinals[i + 1]
        )

        finals.append(
            winner
        )

    champion = simulate_match(
        finals[0],
        finals[1]
    )

    return champion

# ==========================================
# MONTE CARLO
# ==========================================

N_SIMULATIONS = 10000

champions = []

for i in range(
    N_SIMULATIONS
):

    champions.append(
        simulate_tournament()
    )

results = Counter(
    champions
)

print("\nWorld Cup Winner Probabilities\n")

for team, count in sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
):

    probability = (
        count /
        N_SIMULATIONS
    ) * 100

    print(
        f"{team}: {probability:.2f}%"
    )