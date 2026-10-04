import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
N = 5000

# Operating variables
temperature = RNG.normal(65, 8, N)
vibration = RNG.normal(3.2, 0.9, N)
pressure = RNG.normal(5.5, 0.6, N)
rpm = RNG.normal(1450, 90, N)
current = RNG.normal(10.5, 1.5, N)
operating_hours = RNG.uniform(100, 10000, N)

# Inject a small amount of realistic abnormal behavior.
risk_signal = (
    0.055 * (temperature - 65)
    + 0.85 * (vibration - 3.0)
    + 0.45 * (current - 10.0)
    + 0.35 * (pressure - 5.5)
    + 0.00008 * (operating_hours - 5000)
    + RNG.normal(0, 0.9, N)
)

failure_probability = 1 / (1 + np.exp(-risk_signal))
failure = (failure_probability > 0.67).astype(int)

# A few explicit high-risk operating states.
high_risk = (
    ((temperature > 82) & (vibration > 4.5))
    | ((vibration > 5.2) & (current > 12))
    | ((operating_hours > 8500) & (temperature > 75))
)
failure[high_risk] = 1

df = pd.DataFrame({
    "temperature_c": np.round(temperature, 2),
    "vibration_mm_s": np.round(vibration, 2),
    "pressure_bar": np.round(pressure, 2),
    "rpm": np.round(rpm, 0),
    "current_a": np.round(current, 2),
    "operating_hours": np.round(operating_hours, 0),
    "failure": failure,
})

out = Path(__file__).resolve().parent / "machine_sensor_data.csv"
df.to_csv(out, index=False)

print(f"Created {out}")
print(df.head())
print("\nClass distribution:")
print(df["failure"].value_counts(normalize=True).rename("fraction"))
