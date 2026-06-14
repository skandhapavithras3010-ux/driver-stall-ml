import pandas as pd


def add_features(df):
    # Rolling window features
    df['rpm_rolling_mean_5'] = (
        df.groupby('session_id')['rpm']
        .transform(lambda x: x.rolling(5, min_periods=1).mean())
    )

    df['rpm_rolling_std_5'] = (
        df.groupby('session_id')['rpm']
        .transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
    )

    # Interaction features
    df['rpm_x_clutch'] = df['rpm'] * df['clutch_pct'] / 100

    df['brake_x_clutch'] = (
        df['brake_flag'] * df['clutch_pct']
    )

    df['speed_deficit'] = 15 - df['speed_kmh']

    # Embedded-system-inspired risk score
    df['embedded_risk_score'] = (
        0.4 * (1 - df['rpm'] / 5000)
        + 0.3 * df['clutch_variance'] / 100
        + 0.2 * df['brake_flag']
        + 0.1 * df['incline_deg'] / 15
    )

    return df


df = pd.read_csv("data/raw/telemetry.csv")

df = add_features(df)

df.to_csv(
    "data/processed/features.csv",
    index=False
)

print("Feature engineering done.")
print("Shape:", df.shape)

print("\nNew Features Added:")
print([
    "rpm_rolling_mean_5",
    "rpm_rolling_std_5",
    "rpm_x_clutch",
    "brake_x_clutch",
    "speed_deficit",
    "embedded_risk_score"
])
