import pandas as pd

df = pd.read_csv(
    "data/engineered_matches.csv"
)

df["form_difference"] = (
    df["home_form"] -
    df["away_form"]
)

df["goals_scored_difference"] = (
    df["home_goals_scored_last5"] -
    df["away_goals_scored_last5"]
)

df["goals_conceded_difference"] = (
    df["home_goals_conceded_last5"] -
    df["away_goals_conceded_last5"]
)

df.to_csv(
    "data/enhanced_matches.csv",
    index=False
)

print("Enhanced dataset saved.")