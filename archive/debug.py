import pandas as pd

stats = pd.read_csv(
    "data/team_stats.csv",
    index_col=0
)

for team in [
    "Argentina",
    "France",
    "Spain",
    "England",
    "Brazil"
]:
    print(team)
    print(stats.loc[team])
    print()