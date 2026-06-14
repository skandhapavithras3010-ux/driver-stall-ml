import pandas as pd
import joblib

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load engineered dataset
df = pd.read_csv("data/processed/features.csv")

# Aggregate per driver
driver_features = df.groupby("driver_id").agg({
    "rpm": "mean",
    "clutch_pct": "mean",
    "throttle_pct": "mean",
    "speed_kmh": "mean",
    "brake_flag": "mean",
    "incline_deg": "mean",
    "rpm_rate": "mean",
    "clutch_variance": "mean",
    "embedded_risk_score": "mean"
}).reset_index()

# Save driver IDs
driver_ids = driver_features["driver_id"]

# Remove ID column
X = driver_features.drop(columns=["driver_id"])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

driver_features["cluster"] = clusters

# Save clustering model
joblib.dump(
    kmeans,
    "models/kmeans_driver_profiles.pkl"
)

# Save scaler
joblib.dump(
    scaler,
    "models/driver_scaler.pkl"
)

# Save clustered drivers
driver_features.to_csv(
    "models/driver_profiles.csv",
    index=False
)

print("\nDriver Profile Counts:")
print(driver_features["cluster"].value_counts())

print("\nSample Driver Profiles:")
print(driver_features.head())
