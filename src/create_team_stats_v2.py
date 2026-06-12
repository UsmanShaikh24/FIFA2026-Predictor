import pandas as pd

print("Loading engineered dataset...")

df = pd.read_csv("data/engineered_matches.csv")

# Fix missing values
df = df.fillna(0)

teams = {}

all_teams = sorted(
    set(df["home_team"]).union(
        set(df["away_team"])
    )
)

for team in all_teams:

    matches = df[
        (df["home_team"] == team) |
        (df["away_team"] == team)
    ].tail(10)

    if len(matches) == 0:
        continue

    form = 0
    goals_scored = 0
    goals_conceded = 0
    elo = 1500

    for _, row in matches.iterrows():

        if row["home_team"] == team:

            form += row["home_form"]

            goals_scored += row["home_goals_scored_last5"]

            goals_conceded += row["home_goals_conceded_last5"]

            elo = row["home_elo"]

        else:

            form += row["away_form"]

            goals_scored += row["away_goals_scored_last5"]

            goals_conceded += row["away_goals_conceded_last5"]

            elo = row["away_elo"]

    teams[team] = {
        "form": round(form / len(matches), 2),
        "goals_scored": round(goals_scored / len(matches), 2),
        "goals_conceded": round(goals_conceded / len(matches), 2),
        "elo": elo
    }

team_df = pd.DataFrame.from_dict(
    teams,
    orient="index"
)

team_df.to_csv(
    "data/team_stats.csv"
)

print("\nteam_stats.csv created successfully!")
print("\nNumber of Teams:", len(team_df))

print("\nSample:")
print(team_df.head())