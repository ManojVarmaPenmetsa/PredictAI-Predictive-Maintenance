from pathlib import Path
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "artifacts" / "model.joblib")
FEATURES = joblib.load(ROOT / "artifacts" / "features.joblib")

def predict(machine_reading: dict):
    X = pd.DataFrame([machine_reading])[FEATURES]
    probability = float(MODEL.predict_proba(X)[0, 1])
    risk = "HIGH" if probability >= 0.70 else "MEDIUM" if probability >= 0.40 else "LOW"
    return probability, risk

def heuristic_contributors(reading: dict):
    """Fallback explanation used when SHAP is unavailable."""
    limits = {
        "temperature_c": (75, "temperature"),
        "vibration_mm_s": (4.5, "vibration"),
        "current_a": (12, "current"),
        "pressure_bar": (6.3, "pressure"),
        "operating_hours": (8500, "operating hours"),
    }
    scores = []
    for key, (limit, label) in limits.items():
        value = reading[key]
        ratio = max(0.0, value / limit - 1.0)
        scores.append((ratio, label, value))
    return sorted(scores, reverse=True)[:3]
