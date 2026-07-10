"""
Smart Lender - Loan Eligibility Prediction
Data preprocessing, EDA, model training, evaluation, and model selection.

Trains: Decision Tree, Random Forest, KNN, XGBoost
Selects the best model and saves it (with the fitted preprocessing pipeline)
to models/model.pkl for use by the Flask app (app.py).

Usage:
    python train_model.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# XGBoost: fall back to sklearn's HistGradientBoostingClassifier if the
# xgboost package isn't installed in this environment (e.g. offline sandbox).
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    XGB_AVAILABLE = False

# SMOTE: fall back to simple random oversampling if imbalanced-learn isn't installed.
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

sns.set_style("whitegrid")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_train.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education",
                     "Self_Employed", "Credit_History", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "Loan_Status"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# 2. EDA - count plots, distribution plots, bar charts
# ---------------------------------------------------------------------------
def run_eda(df):
    print("Running EDA and saving plots to /plots ...")

    # Countplot: target distribution
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET_COL, data=df)
    plt.title("Loan Status Distribution (0=Rejected, 1=Approved)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "loan_status_countplot.png"))
    plt.close()

    # Countplots for key categorical features vs target
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    cat_features = ["Gender", "Married", "Education", "Self_Employed", "Credit_History", "Property_Area"]
    for ax, col in zip(axes.flatten(), cat_features):
        sns.countplot(x=col, hue=TARGET_COL, data=df, ax=ax)
        ax.set_title(f"{col} vs Loan Status")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "categorical_countplots.png"))
    plt.close()

    # Distribution plots for numeric features
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "numeric_distributions.png"))
    plt.close()

    # Bar chart: approval rate by property area
    plt.figure(figsize=(5, 4))
    approval_rate = df.groupby("Property_Area")[TARGET_COL].mean().sort_values()
    approval_rate.plot(kind="bar", color="teal")
    plt.ylabel("Approval Rate")
    plt.title("Loan Approval Rate by Property Area")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "approval_rate_by_area.png"))
    plt.close()

    print("EDA plots saved.")


# ---------------------------------------------------------------------------
# 3. Preprocessing: missing values, encoding, outliers, scaling
# ---------------------------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # --- Handle missing values ---
    # Numerical -> mean
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].mean())

    # Categorical -> mode
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # --- Outlier handling (cap extreme values at the 1st/99th percentile) ---
    for col in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]:
        lower, upper = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lower, upper)

    # --- Encode categorical variables ---
    encoders = {}
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # --- Feature scaling ---
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    return X, y, encoders, scaler


# ---------------------------------------------------------------------------
# 4. Train / evaluate models
# ---------------------------------------------------------------------------
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- SMOTE balancing on training data only ---
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
        print("Balanced training set using SMOTE.")
    else:
        # simple random oversampling fallback
        train_df = X_train.copy()
        train_df[TARGET_COL] = y_train.values
        maj = train_df[train_df[TARGET_COL] == train_df[TARGET_COL].value_counts().idxmax()]
        minr = train_df[train_df[TARGET_COL] == train_df[TARGET_COL].value_counts().idxmin()]
        minr_upsampled = minr.sample(len(maj), replace=True, random_state=42)
        bal_df = pd.concat([maj, minr_upsampled]).sample(frac=1, random_state=42)
        X_train_bal = bal_df[FEATURE_COLS]
        y_train_bal = bal_df[TARGET_COL]
        print("Balanced training set using random oversampling (imblearn not installed).")

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42
        ) if XGB_AVAILABLE else XGBClassifier(max_iter=200, random_state=42),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        fitted_models[name] = model

        train_acc = accuracy_score(y_train_bal, model.predict(X_train_bal))
        test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        cv_scores = cross_val_score(model, X, y, cv=5)

        results[name] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, test_pred),
            "classification_report": classification_report(y_test, test_pred),
        }

        print(f"\n=== {name} ===")
        print(f"Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f} "
              f"| CV mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        print(results[name]["classification_report"])

        # Confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=results[name]["confusion_matrix"],
                                       display_labels=["Rejected", "Approved"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{safe_name}.png"))
        plt.close()

    # Model comparison bar chart
    plt.figure(figsize=(7, 4))
    names = list(results.keys())
    test_accs = [results[n]["test_accuracy"] for n in names]
    sns.barplot(x=names, y=test_accs, palette="viridis")
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison - Test Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
    plt.close()

    return fitted_models, results, X_test, y_test


# ---------------------------------------------------------------------------
# 5. Select best model + save
# ---------------------------------------------------------------------------
def save_best_model(fitted_models, results, encoders, scaler):
    best_name = max(results, key=lambda n: results[n]["test_accuracy"])
    best_model = fitted_models[best_name]
    print(f"\nBest performing model: {best_name} "
          f"(test accuracy = {results[best_name]['test_accuracy']:.3f})")

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
    }

    model_path = os.path.join(MODELS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Saved best model + preprocessing pipeline to {model_path}")

    return best_name


if __name__ == "__main__":
    df = load_data()
    run_eda(df)
    X, y, encoders, scaler = preprocess(df)
    fitted_models, results, X_test, y_test = train_and_evaluate(X, y)
    best_name = save_best_model(fitted_models, results, encoders, scaler)

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:15s} | train={r['train_accuracy']:.3f}  test={r['test_accuracy']:.3f}  "
              f"cv={r['cv_mean']:.3f}")
    print(f"\nBest model selected for deployment: {best_name}")
    if not XGB_AVAILABLE:
        print("\nNOTE: xgboost was not installed in this environment, so a "
              "HistGradientBoostingClassifier was used as a stand-in for the "
              "'XGBoost' slot. Run `pip install xgboost` and re-run this script "
              "to train with real XGBoost.")
    if not SMOTE_AVAILABLE:
        print("NOTE: imbalanced-learn (SMOTE) was not installed, so random "
              "oversampling was used instead. Run `pip install imbalanced-learn` "
              "for true SMOTE balancing.")
