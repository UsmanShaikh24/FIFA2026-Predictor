import pandas as pd

df = pd.read_csv("data/engineered_matches.csv")

teams = {}

all_teams = sorted(
    set(df["home_team"]).union(
        set(df["away_team"])
    )
)

for team in all_teams:

    home_matches = df[df["home_team"] == team].copy()

    if len(home_matches) == 0:
        continue

    latest = home_matches.iloc[-1]

    teams[team] = {
        "form": latest["home_form"],
        "goals_scored": latest["home_goals_scored_last5"],
        "goals_conceded": latest["home_goals_conceded_last5"],
        "elo": latest["home_elo"]
    }

team_df = pd.DataFrame.from_dict(
    teams,
    orient="index"
)

# Fix missing values
team_df = team_df.fillna(0)

team_df.to_csv(
    "data/team_stats.csv"
)

print(team_df.head())
print(team_df.shape)