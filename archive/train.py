import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading engineered dataset...")

df = pd.read_csv("data/engineered_matches.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# FEATURES
# ==========================================

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

y = df["result"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==========================================
# MODEL
# ==========================================

print("Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==========================================
# PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:\n")

print(importance_df)

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/worldcup_model.pkl"
)

print("\nModel Saved")

print("models/worldcup_model.pkl")