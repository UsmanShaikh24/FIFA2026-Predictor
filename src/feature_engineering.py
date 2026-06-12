import pandas as pd
from collections import defaultdict

# =====================================================
# LOAD MATCH DATASET
# =====================================================

print("Loading match dataset...")

df = pd.read_csv("data/results.csv")

df["date"] = pd.to_datetime(df["date"])

print("Matches Loaded:", len(df))

# =====================================================
# LOAD ELO DATASET
# =====================================================

print("Loading Elo ratings...")

elo_df = pd.read_csv("data/elo_ratings.csv")
elo_df["rating"] = pd.to_numeric(
    elo_df["rating"],
    errors="coerce"
)

elo_df = elo_df.dropna(
    subset=["rating"]
)

elo_df["date"] = pd.to_datetime(
    elo_df["date"],
    format="mixed"
)

elo_df = elo_df.sort_values("date")

print("Elo Records Loaded:", len(elo_df))

# =====================================================
# BUILD ELO HISTORY
# =====================================================

print("Building Elo history...")

elo_history = {}

for team in elo_df["team"].unique():

    team_data = elo_df[
        elo_df["team"] == team
    ].sort_values("date")

    elo_history[team] = team_data

print("Teams in Elo Database:", len(elo_history))

# =====================================================
# ELO LOOKUP FUNCTION
# =====================================================

def get_team_elo(team, match_date):

    if team not in elo_history:
        return 1500

    team_data = elo_history[team]

    past_ratings = team_data[
        team_data["date"] <= match_date
    ]

    if len(past_ratings) == 0:
        return 1500

    return int(past_ratings.iloc[-1]["rating"])

# =====================================================
# RESULT COLUMN
# =====================================================

def get_result(row):

    if row["home_score"] > row["away_score"]:
        return 1

    elif row["home_score"] < row["away_score"]:
        return -1

    return 0


df["result"] = df.apply(get_result, axis=1)

# =====================================================
# SORT CHRONOLOGICALLY
# =====================================================

df = df.sort_values("date").reset_index(drop=True)

# =====================================================
# HISTORY STORAGE
# =====================================================

team_form = defaultdict(list)

team_goals_scored = defaultdict(list)

team_goals_conceded = defaultdict(list)

# =====================================================
# FEATURE COLUMNS
# =====================================================

df["home_form"] = 0
df["away_form"] = 0

df["home_goals_scored_last5"] = 0
df["away_goals_scored_last5"] = 0

df["home_goals_conceded_last5"] = 0
df["away_goals_conceded_last5"] = 0

df["home_elo"] = 1500
df["away_elo"] = 1500

df["elo_difference"] = 0

# =====================================================
# FEATURE ENGINEERING LOOP
# =====================================================

print("Generating features...")

for idx, row in df.iterrows():

    if idx % 5000 == 0:
        print(f"Processed {idx} matches")

    home = row["home_team"]
    away = row["away_team"]

    match_date = row["date"]

    # =============================================
    # ELO FEATURES
    # =============================================

    home_elo = get_team_elo(
        home,
        match_date
    )

    away_elo = get_team_elo(
        away,
        match_date
    )

    df.at[idx, "home_elo"] = home_elo

    df.at[idx, "away_elo"] = away_elo

    df.at[idx, "elo_difference"] = (
        home_elo - away_elo
    )

    # =============================================
    # FORM FEATURES
    # =============================================

    home_recent_form = team_form[home][-5:]
    away_recent_form = team_form[away][-5:]

    df.at[idx, "home_form"] = sum(home_recent_form)

    df.at[idx, "away_form"] = sum(away_recent_form)

    # =============================================
    # GOALS SCORED FEATURES
    # =============================================

    home_scored_recent = team_goals_scored[home][-5:]
    away_scored_recent = team_goals_scored[away][-5:]

    df.at[idx, "home_goals_scored_last5"] = (
        sum(home_scored_recent)
    )

    df.at[idx, "away_goals_scored_last5"] = (
        sum(away_scored_recent)
    )

    # =============================================
    # GOALS CONCEDED FEATURES
    # =============================================

    home_conceded_recent = team_goals_conceded[home][-5:]
    away_conceded_recent = team_goals_conceded[away][-5:]

    df.at[idx, "home_goals_conceded_last5"] = (
        sum(home_conceded_recent)
    )

    df.at[idx, "away_goals_conceded_last5"] = (
        sum(away_conceded_recent)
    )

    # =============================================
    # UPDATE FORM
    # =============================================

    if row["home_score"] > row["away_score"]:

        team_form[home].append(3)
        team_form[away].append(0)

    elif row["home_score"] < row["away_score"]:

        team_form[home].append(0)
        team_form[away].append(3)

    else:

        team_form[home].append(1)
        team_form[away].append(1)

    # =============================================
    # UPDATE GOALS HISTORY
    # =============================================

    team_goals_scored[home].append(
        row["home_score"]
    )

    team_goals_scored[away].append(
        row["away_score"]
    )

    team_goals_conceded[home].append(
        row["away_score"]
    )

    team_goals_conceded[away].append(
        row["home_score"]
    )

# =====================================================
# PREVIEW
# =====================================================

print("\nFeature Engineering Complete\n")

print(
    df[
        [
            "home_team",
            "away_team",
            "home_form",
            "away_form",
            "home_elo",
            "away_elo",
            "elo_difference",
            "result"
        ]
    ].head(20)
)

# =====================================================
# SAVE FILE
# =====================================================

output_file = "data/engineered_matches.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nDataset Saved Successfully")

print("Output File:", output_file)

print("Final Shape:", df.shape)