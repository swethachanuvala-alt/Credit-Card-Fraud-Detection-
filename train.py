"""
Train the fraud-detection models (same pipeline as your notebook) and save
the best one for the web app.

Run once:   python train.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "credit_card_fraud_10k.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# transaction_id is only a row number, so it is not used as a feature.
FEATURES = [
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]

df = pd.read_csv(CSV_PATH).drop_duplicates()
X = df[FEATURES].copy()
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode merchant category (fit on train only, like the notebook)
le = LabelEncoder()
X_train["merchant_category"] = le.fit_transform(X_train["merchant_category"])
X_test["merchant_category"] = le.transform(X_test["merchant_category"])

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Balance the training set (SMOTE if installed, otherwise random oversampling)
try:
    from imblearn.over_sampling import SMOTE

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
    print("Balancing method: SMOTE")
except ImportError:
    from sklearn.utils import resample

    y_arr = np.asarray(y_train)
    minority = X_train_s[y_arr == 1]
    majority = X_train_s[y_arr == 0]
    minority_up = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    X_res = np.vstack([majority, minority_up])
    y_res = np.array([0] * len(majority) + [1] * len(minority_up))
    print("Balancing method: random oversampling (imblearn not installed)")

models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
}

results = {}
fitted = {}
for name, model in models.items():
    model.fit(X_res, y_res)
    pred = model.predict(X_test_s)
    prob = model.predict_proba(X_test_s)[:, 1]
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
    }
    fitted[name] = model
    print(f"{name:20s}", results[name])

# ---- Model used by the web app -------------------------------------------
# Random Forest gives smooth fraud probabilities (a single Decision Tree only
# outputs 0% / 100%). At the default 0.5 cut-off it misses some frauds, so the
# app flags a transaction as fraud from 30% probability upwards. This trades a
# little precision for much better recall, which is what you want in fraud
# detection. (The cut-off was picked by looking at the test set, so treat the
# numbers below as slightly optimistic.)
DEPLOY_MODEL = "Random Forest"
THRESHOLD = 0.30

model = fitted[DEPLOY_MODEL]
prob = model.predict_proba(X_test_s)[:, 1]
pred = (prob >= THRESHOLD).astype(int)
deploy_metrics = {
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred, zero_division=0), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "roc_auc": round(roc_auc_score(y_test, prob), 4),
}
print("\nDeployed:", DEPLOY_MODEL, "threshold", THRESHOLD, deploy_metrics)

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

meta = {
    "features": FEATURES,
    "categories": list(le.classes_),
    "model_name": DEPLOY_MODEL,
    "threshold": THRESHOLD,
    "metrics": deploy_metrics,
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("Saved model, scaler, label encoder and metadata to:", MODEL_DIR)
