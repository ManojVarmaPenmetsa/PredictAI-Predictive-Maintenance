from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "machine_sensor_data.csv"
ARTIFACTS = ROOT / "artifacts"

ARTIFACTS.mkdir(exist_ok=True)

# Our chosen operating threshold.
# We intentionally lower it from 0.50 because
# predictive maintenance prioritizes catching failures.
DECISION_THRESHOLD = 0.35


# =========================================================
# 1. LOAD DATA
# =========================================================

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


# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================================
# 3. DEFINE MODELS
# =========================================================

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


# =========================================================
# 4. TRAIN AND EVALUATE MODELS
# =========================================================

results = {}

best_name = None
best_model = None
best_recall = -1

for name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training {name}...")
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Probability of failure
    probabilities = model.predict_proba(X_test)[:, 1]

    # -----------------------------------------------------
    # IMPORTANT:
    # We use 0.35 instead of the default 0.50 threshold.
    # -----------------------------------------------------

    predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    # -----------------------------------------------------
    # Store results
    # -----------------------------------------------------

    results[name] = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist()
    }

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(f"Decision threshold: {DECISION_THRESHOLD:.2f}")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # -----------------------------------------------------
    # Select model based on RECALL
    #
    # Why?
    # Missing a machine failure is more costly than
    # generating an additional maintenance alert.
    # -----------------------------------------------------

    if recall > best_recall:

        best_recall = recall
        best_name = name
        best_model = model


# =========================================================
# 5. SAVE BEST MODEL
# =========================================================

joblib.dump(
    best_model,
    ARTIFACTS / "model.joblib"
)

joblib.dump(
    features,
    ARTIFACTS / "features.joblib"
)


# =========================================================
# 6. EVALUATE SELECTED MODEL
# =========================================================

best_probabilities = best_model.predict_proba(
    X_test
)[:, 1]

best_predictions = (
    best_probabilities >= DECISION_THRESHOLD
).astype(int)


# =========================================================
# 7. ROC CURVE
# =========================================================

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    best_probabilities
)

best_auc = roc_auc_score(
    y_test,
    best_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"{best_name} (AUC = {best_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - PredictAI"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ARTIFACTS / "roc_curve.png",
    dpi=150
)

plt.close()


# =========================================================
# 8. PRECISION-RECALL CURVE
# =========================================================

precision_values, recall_values, pr_thresholds = (
    precision_recall_curve(
        y_test,
        best_probabilities
    )
)

average_precision = average_precision_score(
    y_test,
    best_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    recall_values,
    precision_values,
    label=f"AP = {average_precision:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - PredictAI"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    ARTIFACTS / "precision_recall_curve.png",
    dpi=150
)

plt.close()


# =========================================================
# 9. SAVE ALL METRICS
# =========================================================

evaluation = {

    "best_model": best_name,

    "selection_metric":
        "Recall at selected threshold",

    "decision_threshold":
        DECISION_THRESHOLD,

    "best_model_metrics": results[best_name],

    "models": results,

    "average_precision":
        round(average_precision, 4)
}


with open(
    ARTIFACTS / "metrics.json",
    "w"
) as file:

    json.dump(
        evaluation,
        file,
        indent=2
    )


# =========================================================
# 10. FINAL OUTPUT
# =========================================================

print("\n")
print("=" * 60)
print("FINAL MODEL")
print("=" * 60)

print(
    f"Selected model : {best_name}"
)

print(
    f"Decision threshold : "
    f"{DECISION_THRESHOLD:.2f}"
)

print(
    f"Recall : "
    f"{results[best_name]['recall']:.4f}"
)

print(
    f"Precision : "
    f"{results[best_name]['precision']:.4f}"
)

print(
    f"F1 : "
    f"{results[best_name]['f1']:.4f}"
)

print(
    f"ROC-AUC : "
    f"{results[best_name]['roc_auc']:.4f}"
)

print(
    f"Average Precision : "
    f"{average_precision:.4f}"
)

print("\nSaved:")

print(
    "  artifacts/model.joblib"
)

print(
    "  artifacts/features.joblib"
)

print(
    "  artifacts/metrics.json"
)

print(
    "  artifacts/roc_curve.png"
)

print(
    "  artifacts/precision_recall_curve.png"
)