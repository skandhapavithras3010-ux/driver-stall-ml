import numpy as np
import pandas as pd


def generate_session(driver_id, session_id, n=300, seed=None):
    rng = np.random.default_rng(seed)

    t = np.arange(n)

    # Simulated RPM signal
    rpm_base = (
        1500
        + 500 * np.sin(t * 0.05)
        + rng.normal(0, 80, n)
    )

    # Inject RPM drop events (stall-like behavior)
    for ev in rng.choice(n, size=rng.integers(3, 8), replace=False):
        rpm_base[max(0, ev - 15):min(n, ev + 10)] -= rng.uniform(600, 1200)

    rpm = np.clip(rpm_base, 0, 5000).astype(int)

    # Driver control inputs
    clutch_pct = np.clip(
        50 + 35 * np.sin(t * 0.08) + rng.normal(0, 8, n),
        0,
        100
    )

    throttle_pct = np.clip(
        30 + 20 * np.cos(t * 0.06) + rng.normal(0, 6, n),
        0,
        100
    )

    speed_kmh = np.clip(
        5 + 8 * np.sin(t * 0.04) + rng.normal(0, 1, n),
        0,
        15
    )

    brake_flag = (rng.random(n) < 0.15).astype(int)

    # One incline value per driving session
    incline_deg = rng.uniform(0, 12)

    # Derived signals
    rpm_rate = np.diff(rpm, prepend=rpm[0])

    clutch_variance = (
        pd.Series(clutch_pct)
        .rolling(10, min_periods=1)
        .var()
        .fillna(0)
        .values
    )

    brake_toggles = (
        pd.Series(brake_flag)
        .rolling(10, min_periods=1)
        .sum()
        .values
    )

    # Risk calculation
    risk = np.zeros(n, dtype=int)

    risk[(rpm < 900) | (rpm_rate < -50)] += 1

    risk[
        (clutch_pct > 30)
        & (clutch_pct < 70)
        & (rpm < 1000)
    ] += 1

    risk[
        (brake_flag == 1)
        & (clutch_pct > 20)
    ] += 1

    risk[incline_deg > 6] += 1

    stall_risk_label = np.clip(risk, 0, 2)

    return pd.DataFrame({
        "session_id": session_id,
        "driver_id": driver_id,
        "rpm": rpm,
        "speed_kmh": speed_kmh,
        "clutch_pct": clutch_pct,
        "throttle_pct": throttle_pct,
        "brake_flag": brake_flag,
        "incline_deg": incline_deg,
        "rpm_rate": rpm_rate,
        "clutch_variance": clutch_variance,
        "brake_toggles": brake_toggles,
        "stall_risk_label": stall_risk_label
    })


# Generate dataset
frames = [
    generate_session(
        f"D{d:03}",
        f"S{d:03}_{s:02}",
        seed=d * 100 + s
    )
    for d in range(1, 51)
    for s in range(1, 11)
]

df = pd.concat(frames, ignore_index=True)

# Save dataset
df.to_csv("data/raw/telemetry.csv", index=False)

# Verification
print(f"\nGenerated {len(df):,} rows")

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["stall_risk_label"].value_counts(normalize=True))