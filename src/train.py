import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# Load engineered dataset
df = pd.read_csv("data/processed/features.csv")

# Remove identifiers
X = df.drop(
    columns=[
        "stall_risk_label",
        "session_id",
        "driver_id"
    ]
)

y = df["stall_risk_label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Model
model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, preds)

print("\nAccuracy:")
print(acc)

print("\nClassification Report:")
print(classification_report(y_test, preds))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Save model
joblib.dump(
    model,
    "models/xgb_model.pkl"
)

print("\nModel saved to models/xgb_model.pkl")
