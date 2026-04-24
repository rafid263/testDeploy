"""
Diabetes Prediction API
=======================
Endpoints:
  GET  /          → health check
  GET  /info      → model info & expected input fields
  POST /predict   → returns prediction + probability

Prediction Pipeline (mirrors training exactly):
  1. Receive raw JSON values
  2. Validate all 8 feature fields are present
  3. Replace biologically impossible zeros with trained medians
  4. Scale using the SAME StandardScaler fitted during training
  5. Run model.predict() and model.predict_proba()
  6. Return result as JSON
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load artefacts once at startup ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

model           = joblib.load(os.path.join(MODEL_DIR, "diabetes_model.pkl"))
scaler          = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

# Medians computed from training data — used to fix impossible zeros
# at prediction time, exactly as done during training preprocessing.
MEDIANS = {
    "Glucose":        117.0,
    "BloodPressure":   72.0,
    "SkinThickness":   23.0,
    "Insulin":         30.5,
    "BMI":             32.0,
}
ZERO_IMPUTE_COLS = set(MEDIANS.keys())

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "Diabetes Prediction API",
        "model": "RandomForestClassifier",
        "version": "1.0.0"
    })


@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "description": "Predicts diabetes risk using Pima Indians dataset features.",
        "input_fields": {
            "Pregnancies":              "int   — number of times pregnant (0–17)",
            "Glucose":                  "float — plasma glucose concentration (mg/dL)",
            "BloodPressure":            "float — diastolic blood pressure (mm Hg)",
            "SkinThickness":            "float — triceps skin fold thickness (mm)",
            "Insulin":                  "float — 2-hour serum insulin (μU/mL)",
            "BMI":                      "float — body mass index (kg/m²)",
            "DiabetesPedigreeFunction": "float — diabetes pedigree function score",
            "Age":                      "int   — age in years"
        },
        "output": {
            "prediction":    "0 = No Diabetes, 1 = Diabetes",
            "probability":   "probability of being diabetic (0.0 – 1.0)",
            "risk_level":    "Low / Medium / High",
            "label":         "human-readable prediction"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # ── 1. Validate all required fields are present ──
    missing = [col for col in feature_columns if col not in data]
    if missing:
        return jsonify({
            "error": f"Missing required fields: {missing}",
            "required_fields": feature_columns
        }), 400

    # ── 2. Build the input vector ──
    try:
        raw_values = {col: float(data[col]) for col in feature_columns}
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"All values must be numeric. Detail: {str(e)}"}), 400

    # ── 3. Fix biologically impossible zeros (same preprocessing as training) ──
    for col in ZERO_IMPUTE_COLS:
        if raw_values[col] == 0.0:
            raw_values[col] = MEDIANS[col]

    # ── 4. Scale using the trained scaler ──
    # Use a DataFrame so sklearn doesn't warn about missing feature names
    input_df     = pd.DataFrame([raw_values])[feature_columns]
    input_scaled = scaler.transform(input_df)

    # ── 5. Predict ──
    prediction   = int(model.predict(input_scaled)[0])
    probability  = float(model.predict_proba(input_scaled)[0][1])

    # ── 6. Determine risk level ──
    if probability < 0.35:
        risk_level = "Low"
    elif probability < 0.65:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return jsonify({
        "prediction":  prediction,
        "label":       "Diabetes" if prediction == 1 else "No Diabetes",
        "probability": round(probability, 4),
        "risk_level":  risk_level,
        "input_received": raw_values    # echoed back for transparency
    })


# ── Run locally ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
