"""
Smart Lender - Flask web application
Loads the trained model pipeline (models/model.pkl) and serves predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

app = Flask(__name__)

# --- Load model + preprocessing pipeline once at startup ---
with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
model_name = artifact["model_name"]
encoders = artifact["encoders"]
scaler = artifact["scaler"]
feature_cols = artifact["feature_cols"]
categorical_cols = artifact["categorical_cols"]
numeric_cols = artifact["numeric_cols"]


def build_feature_vector(form):
    """Convert raw form input into the model's expected feature vector."""
    dependents_raw = form.get("dependents", "0")
    credit_history_raw = form.get("credit_history", "1")

    data = {
        "Gender": form.get("gender"),
        "Married": form.get("married"),
        # Training-time encoding collapsed "3+" -> "3" (see train_model.py)
        "Dependents": "3" if dependents_raw == "3+" else dependents_raw,
        "Education": form.get("education"),
        "Self_Employed": form.get("self_employed"),
        # Training-time encoding stored Credit_History as a float-string ("1.0"/"0.0")
        "Credit_History": str(float(credit_history_raw)),
        "Property_Area": form.get("property_area"),
        "ApplicantIncome": float(form.get("applicant_income", 0) or 0),
        "CoapplicantIncome": float(form.get("coapplicant_income", 0) or 0),
        "LoanAmount": float(form.get("loan_amount", 0) or 0),
        "Loan_Amount_Term": float(form.get("loan_amount_term", 360) or 360),
    }

    df = pd.DataFrame([data])

    # Encode categorical columns using the fitted LabelEncoders.
    for col in categorical_cols:
        le = encoders[col]
        val = str(df.at[0, col])
        if val not in le.classes_:
            # Unseen category -> map to the most frequent training class.
            val = le.classes_[0]
        df[col] = le.transform([val])

    # Scale numeric columns using the fitted StandardScaler.
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df[feature_cols]


@app.route("/")
def home():
    return render_template("home.html", model_name=model_name)


@app.route("/predict")
def predict_form():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        X = build_feature_vector(request.form)
        pred = model.predict(X)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = round(float(np.max(model.predict_proba(X))) * 100, 1)

        result = "Approved" if int(pred) == 1 else "Rejected"

        return render_template(
            "submit.html",
            result=result,
            approved=(int(pred) == 1),
            confidence=proba,
            model_name=model_name,
            applicant=request.form,
        )
    except Exception as e:
        return render_template("submit.html", error=str(e))


if __name__ == "__main__":
    # host=0.0.0.0 so it's reachable when deployed (e.g. IBM Cloud / containers)
    app.run(host="0.0.0.0", port=5000, debug=True)
