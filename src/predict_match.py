import pandas as pd
import joblib

model = joblib.load(
    "models/worldcup_xgb.pkl"
)

stats = pd.read_csv(
    "data/team_stats.csv",
    index_col=0
)

home_team = input(
    "Home Team: "
)

away_team = input(
    "Away Team: "
)

if home_team not in stats.index:
    print("Unknown Team")
    exit()

if away_team not in stats.index:
    print("Unknown Team")
    exit()

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
print("\nFeatures:")
print({
    "home_form": home["form"],
    "away_form": away["form"],
    "home_goals_scored": home["goals_scored"],
    "away_goals_scored": away["goals_scored"],
    "home_goals_conceded": home["goals_conceded"],
    "away_goals_conceded": away["goals_conceded"],
    "home_elo": home["elo"],
    "away_elo": away["elo"],
    "elo_difference": home["elo"] - away["elo"]
})
print("\nFeature Vector:")
print(X)

probs = model.predict_proba(X)[0]

away_win = probs[0]
draw = probs[1]
home_win = probs[2]

print("\nPrediction\n")

print(
    f"{home_team} Win: "
    f"{home_win*100:.2f}%"
)

print(
    f"Draw: "
    f"{draw*100:.2f}%"
)

print(
    f"{away_team} Win: "
    f"{away_win*100:.2f}%"
)


