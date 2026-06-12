import pandas as pd

df = pd.read_csv("data/results.csv")

print(df.head())

print(df.shape)

print(df.columns)

def get_result(row):

    if row["home_score"] > row["away_score"]:
        return 1

    elif row["home_score"] < row["away_score"]:
        return -1

    return 0
df["result"] = df.apply(get_result, axis=1)
print(df[[
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result"
]].head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
all_teams = pd.concat([
    df["home_team"],
    df["away_team"]
])
encoder.fit(all_teams)
df["home_encoded"] = encoder.transform(df["home_team"])

df["away_encoded"] = encoder.transform(df["away_team"])
print(df[[
    "home_team",
    "home_encoded",
    "away_team",
    "away_encoded"
]].head())