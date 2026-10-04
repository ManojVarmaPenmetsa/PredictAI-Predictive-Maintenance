from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PredictAI",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ PredictAI")
st.caption("Explainable AI for Industrial Predictive Maintenance")


# ============================================================
# 3. LOAD TRAINED MODEL AND ARTIFACTS
# ============================================================

model_path = ARTIFACTS / "model.joblib"
features_path = ARTIFACTS / "features.joblib"
metrics_path = ARTIFACTS / "metrics.json"


if not model_path.exists():
    st.error(
        "Trained model not found. Run `python src/train.py` first."
    )
    st.stop()


model = joblib.load(model_path)
features = joblib.load(features_path)

with open(metrics_path, "r") as file:
    metrics = json.load(file)


selected_model = metrics["best_model"]
decision_threshold = metrics["decision_threshold"]


# ============================================================
# 4. AUTOMATED MAINTENANCE REPORT FUNCTION
# ============================================================

def generate_maintenance_report(
    reading,
    failure_probability,
    risk_level,
    contribution
):
    """
    Generates an automated maintenance report using:
    - Sensor values
    - ML failure probability
    - Risk level
    - SHAP feature contributions
    """

    # Get the three most important SHAP features
    top_features = (
        contribution.abs()
        .sort_values(ascending=False)
        .head(3)
    )

    evidence = []

    for feature in top_features.index:

        value = contribution[feature]

        if value > 0:
            effect = "increasing"
        else:
            effect = "reducing"

        evidence.append(
            f"- **{feature}** is {effect} the predicted failure risk "
            f"(SHAP contribution: {value:.3f})."
        )

    evidence_text = "\n".join(evidence)


    # ========================================================
    # HIGH RISK
    # ========================================================

    if failure_probability >= 0.70:

        priority = "Immediate inspection"

        # High vibration
        if reading["vibration_mm_s"] >= 6:

            root_cause = (
                "The sensor pattern suggests possible mechanical degradation, "
                "such as bearing, shaft, alignment, or rotating-component issues."
            )

        # High temperature
        elif reading["temperature_c"] >= 85:

            root_cause = (
                "The elevated temperature may indicate overheating, excessive "
                "load, lubrication problems, or mechanical resistance."
            )

        # General abnormal condition
        else:

            root_cause = (
                "Multiple sensor readings indicate an abnormal operating "
                "condition. Physical inspection is recommended to identify "
                "the underlying cause."
            )

        action = (
            "Inspect the machine before returning it to full load. "
            "Check bearings and rotating components, investigate abnormal "
            "vibration and temperature, and verify current consumption."
        )


    # ========================================================
    # MEDIUM RISK
    # ========================================================

    elif failure_probability >= 0.35:

        priority = "Schedule inspection"

        root_cause = (
            "The machine is showing an elevated-risk operating pattern. "
            "The contributing sensor values should be monitored for further "
            "deterioration."
        )

        action = (
            "Continue monitoring temperature, vibration and current. "
            "Schedule preventive inspection if the abnormal trend continues."
        )


    # ========================================================
    # LOW RISK
    # ========================================================

    else:

        priority = "Continue monitoring"

        root_cause = (
            "The current sensor pattern does not indicate an immediate "
            "failure condition."
        )

        action = (
            "Continue normal operation and preventive maintenance while "
            "monitoring the sensor values for changes."
        )


    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = f"""
### Risk Assessment

The machine has a predicted failure probability of **{failure_probability:.1%}**
and is currently classified as **{risk_level} risk**.

### Key Evidence

{evidence_text}

### Possible Root Cause

{root_cause}

### Recommended Action

{action}

### Priority

**{priority}**
"""

    return report


# ============================================================
# 5. MACHINE SENSOR INPUT
# ============================================================

st.sidebar.header("Machine Sensor Input")


temperature = st.sidebar.slider(
    "Temperature (°C)",
    35.0,
    110.0,
    65.0,
    0.5
)


vibration = st.sidebar.slider(
    "Vibration (mm/s)",
    0.5,
    9.0,
    3.2,
    0.1
)


pressure = st.sidebar.slider(
    "Pressure (bar)",
    3.0,
    8.0,
    5.5,
    0.1
)


rpm = st.sidebar.slider(
    "RPM",
    900,
    1900,
    1450,
    10
)


current = st.sidebar.slider(
    "Current (A)",
    5.0,
    18.0,
    10.5,
    0.1
)


operating_hours = st.sidebar.slider(
    "Operating hours",
    0,
    12000,
    5000,
    100
)


# ============================================================
# 6. CREATE CURRENT SENSOR READING
# ============================================================

reading = {
    "temperature_c": temperature,
    "vibration_mm_s": vibration,
    "pressure_bar": pressure,
    "rpm": rpm,
    "current_a": current,
    "operating_hours": operating_hours
}


X = pd.DataFrame([reading])[features]


# ============================================================
# 7. ML PREDICTION
# ============================================================

failure_probability = float(
    model.predict_proba(X)[0, 1]
)


# Apply the tuned 35% decision threshold

failure_alert = (
    failure_probability >= decision_threshold
)


# ============================================================
# 8. DETERMINE RISK LEVEL
# ============================================================

if failure_probability >= 0.70:

    risk_level = "HIGH"

elif failure_probability >= 0.35:

    risk_level = "MEDIUM"

else:

    risk_level = "LOW"


# ============================================================
# 9. TOP DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


col1.metric(
    "Failure Probability",
    f"{failure_probability:.1%}"
)


col2.metric(
    "Risk Level",
    risk_level
)


col3.metric(
    "Failure Alert",
    "TRIGGERED" if failure_alert else "CLEAR"
)


col4.metric(
    "Decision Threshold",
    f"{decision_threshold:.0%}"
)


st.divider()


# ============================================================
# 10. MACHINE STATUS
# ============================================================

if failure_alert:

    st.error(
        "⚠️ FAILURE ALERT — The machine is above the selected failure threshold."
    )

else:

    st.success(
        "✅ MACHINE STATUS NORMAL — No failure alert has been triggered."
    )


# ============================================================
# 11. CURRENT SENSOR PROFILE
# ============================================================

st.subheader("📊 Current Sensor Profile")


sensor_table = pd.DataFrame([reading])


st.dataframe(
    sensor_table,
    width="stretch",
    hide_index=True
)


# ============================================================
# 12. SHAP EXPLAINABILITY
# ============================================================

st.subheader("🧠 Why did the model make this prediction?")


# Initialize this before the try block
# so the maintenance report can check whether SHAP worked.

contribution = None


try:

    import shap


    # Create SHAP TreeExplainer

    explainer = shap.TreeExplainer(model)


    shap_values = explainer.shap_values(X)


    # ========================================================
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # ========================================================

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    elif len(shap_values.shape) == 3:

        values = shap_values[0, :, 1]

    else:

        values = shap_values[0]


    # Create feature contribution series

    contribution = pd.Series(
        values,
        index=features
    )


    # Sort by absolute contribution

    contribution = contribution.reindex(
        contribution.abs()
        .sort_values(ascending=False)
        .index
    )


    # ========================================================
    # SHAP TABLE
    # ========================================================

    shap_table = pd.DataFrame({

        "Feature": contribution.index,

        "SHAP Contribution": contribution.values,

        "Effect": [
            "↑ Increases failure risk"
            if value > 0
            else "↓ Reduces failure risk"

            for value in contribution.values
        ]

    })


    st.dataframe(
        shap_table,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # SHAP BAR CHART
    # ========================================================

    st.bar_chart(
        contribution.rename("SHAP Contribution"),
        width="stretch"
    )


    # ========================================================
    # STRONGEST CONTRIBUTOR
    # ========================================================

    top_feature = contribution.index[0]

    top_value = contribution.iloc[0]


    if top_value > 0:

        st.warning(
            f"**{top_feature}** is currently the strongest feature "
            f"pushing the prediction toward failure."
        )

    else:

        st.success(
            f"**{top_feature}** is currently the strongest feature "
            f"reducing the predicted failure risk."
        )


    st.caption(
        "Positive SHAP values push the prediction toward failure. "
        "Negative values push the prediction toward normal operation."
    )


except Exception as error:

    st.warning(
        "SHAP explanation could not be generated."
    )

    st.caption(str(error))


# ============================================================
# 13. BASIC MAINTENANCE RECOMMENDATION
# ============================================================

st.subheader("🔧 AI Maintenance Recommendation")


if failure_alert:

    recommendation = (
        "The machine is showing a high-risk operating pattern. "
        "Inspect the bearing and shaft assembly, investigate abnormal "
        "vibration and temperature, and verify current consumption "
        "before returning the machine to full load."
    )


elif risk_level == "MEDIUM":

    recommendation = (
        "The machine is approaching the selected failure threshold. "
        "Continue monitoring vibration, temperature and current, and "
        "schedule a preventive inspection if the trend continues."
    )


else:

    recommendation = (
        "No immediate failure alert has been detected. "
        "Continue normal monitoring and preventive maintenance."
    )


st.info(recommendation)


# ============================================================
# 14. AUTOMATED MAINTENANCE REPORT
# ============================================================

st.subheader("📋 Automated Maintenance Report")


st.caption(
    "This report is generated from the ML prediction, "
    "sensor values and SHAP evidence."
)


if contribution is not None:

    if st.button(
        "Generate Maintenance Report",
        type="primary"
    ):

        report = generate_maintenance_report(
            reading,
            failure_probability,
            risk_level,
            contribution
        )

        st.markdown(report)

else:

    st.warning(
        "Maintenance report unavailable because "
        "SHAP explanation was not generated."
    )


# ============================================================
# 15. MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")


model_col1, model_col2, model_col3 = st.columns(3)


model_col1.metric(
    "Selected Model",
    selected_model
)


model_col2.metric(
    "Recall",
    f"{metrics['best_model_metrics']['recall']:.1%}"
)


model_col3.metric(
    "ROC-AUC",
    f"{metrics['best_model_metrics']['roc_auc']:.3f}"
)


st.caption(
    "The model was selected with failure detection in mind. "
    "The decision threshold was tuned to prioritize recall because "
    "missed machine failures can be costly."
)


# ============================================================
# 16. MODEL COMPARISON
# ============================================================

st.subheader("📈 Model Comparison")


comparison_rows = []


for name, values in metrics["models"].items():

    comparison_rows.append({

        "Model": name,

        "Accuracy": values["accuracy"],

        "Precision": values["precision"],

        "Recall": values["recall"],

        "F1": values["f1"],

        "ROC-AUC": values["roc_auc"]

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


st.dataframe(
    comparison_df,
    width="stretch",
    hide_index=True
)


# ============================================================
# 17. PROJECT DISCLAIMER
# ============================================================

st.divider()


st.caption(
    "Prototype note: this project currently uses synthetic industrial "
    "sensor data. A production implementation would require real sensor "
    "telemetry, historical maintenance records, device authentication, "
    "monitoring and model validation on real equipment."
)