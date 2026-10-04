# PredictAI — Intelligent Predictive Maintenance & Root-Cause Analysis

A portfolio/interview-ready prototype that predicts industrial machine failure risk from sensor data, explains predictions with SHAP, and presents an actionable maintenance report in Streamlit.

## Pipeline

Sensor data → preprocessing → ML model → failure probability → SHAP explanation → maintenance recommendation → dashboard

## Project structure

- `data/generate_data.py` — creates a realistic synthetic machine-sensor dataset.
- `src/train.py` — trains and evaluates the model and saves artifacts.
- `src/predict.py` — loads the trained model and produces predictions/explanations.
- `app.py` — Streamlit dashboard.
- `requirements.txt` — Python dependencies.

## Run

```bash
python -m pip install -r requirements.txt
python data/generate_data.py
python src/train.py
streamlit run app.py
```

The app works without an LLM API key. If `OPENAI_API_KEY` is configured, the optional LLM report button can turn the model output and feature explanations into a natural-language maintenance report.

## Important interview note

This is a prototype using synthetic data. Do not claim that it was deployed on a real industrial machine. Present it as a self-built proof-of-concept and explain how you would connect real sensors/MQTT in the next version.
