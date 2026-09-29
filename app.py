"""
Credit Card Fraud Detector - web app.

1. python train.py      (once, creates the model/ folder)
2. python app.py        (starts the site)
3. open http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        META = json.load(f)
except FileNotFoundError:
    raise SystemExit(
        "\nModel files not found. Run  python train.py  first, then start the app again.\n"
    )

FEATURES = META["features"]
THRESHOLD = META["threshold"]


def to_float(data, key, low, high):
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"'{key}' is missing or not a number")
    if not low <= value <= high:
        raise ValueError(f"'{key}' must be between {low} and {high}")
    return value


@app.route("/")
def index():
    return render_template("index.html", meta=META)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    try:
        category = data.get("merchant_category")
        if category not in META["categories"]:
            raise ValueError("Please choose a valid merchant category")

        row = {
            "amount": to_float(data, "amount", 0, 1_000_000),
            "transaction_hour": to_float(data, "transaction_hour", 0, 23),
            "merchant_category": int(encoder.transform([category])[0]),
            "foreign_transaction": int(to_float(data, "foreign_transaction", 0, 1)),
            "location_mismatch": int(to_float(data, "location_mismatch", 0, 1)),
            "device_trust_score": to_float(data, "device_trust_score", 0, 100),
            "velocity_last_24h": to_float(data, "velocity_last_24h", 0, 1000),
            "cardholder_age": to_float(data, "cardholder_age", 18, 100),
        }
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    frame = pd.DataFrame([row], columns=FEATURES)
    scaled = scaler.transform(frame)
    prob = float(model.predict_proba(scaled)[0][1])
    is_fraud = prob >= THRESHOLD

    if prob >= 0.6:
        level = "High"
    elif prob >= THRESHOLD:
        level = "Medium"
    elif prob >= 0.1:
        level = "Low"
    else:
        level = "Very low"

    return jsonify(
        {
            "is_fraud": bool(is_fraud),
            "probability": round(prob * 100, 1),
            "risk_level": level,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
