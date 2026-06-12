import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

df = pd.read_csv("data/engineered_matches.csv")

features = [
    "home_form",
    "away_form",
    "home_goals_scored_last5",
    "away_goals_scored_last5",
    "home_goals_conceded_last5",
    "away_goals_conceded_last5",
    "home_elo",
    "away_elo",
    "elo_difference"
]

X = df[features]

# Convert labels
y = df["result"].map({
    -1: 0,
     0: 1,
     1: 2
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    preds
)

print("Accuracy:", accuracy)

joblib.dump(
    model,
    "models/worldcup_xgb.pkl"
)

print(df["result"].value_counts())
print("Model Saved")