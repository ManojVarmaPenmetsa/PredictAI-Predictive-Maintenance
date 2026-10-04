from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "machine_sensor_data.csv"

df = pd.read_csv(DATA)

features = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_bar",
    "rpm",
    "current_a",
    "operating_hours"
]

X = df[features]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=15,
        random_state=42
    )
}

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20
]

for name, model in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print(
        f"\nROC-AUC: "
        f"{roc_auc_score(y_test, probabilities):.4f}"
    )

    print(
        "\nThreshold | Precision | Recall | F1"
    )

    print("-" * 45)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.3f} | "
            f"{recall:6.3f} | "
            f"{f1:5.3f}"
        )