import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

print("Loading enhanced dataset...")

df = pd.read_csv("data/enhanced_matches.csv")

# ==========================================
# FEATURES
# ==========================================

features = [
    "form_difference",
    "goals_scored_difference",
    "goals_conceded_difference",
    "elo_difference"
]

X = df[features]

# Convert labels
y = df["result"].map({
    -1: 0,
     0: 1,
     1: 2
})

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# MODEL
# ==========================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

print("Training XGBoost V2...")

model.fit(X_train, y_train)

# ==========================================
# EVALUATION
# ==========================================

preds = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    preds
)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        preds
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
    "models/worldcup_xgb_v2.pkl"
)

print("\nModel Saved")
print("models/worldcup_xgb_v2.pkl")