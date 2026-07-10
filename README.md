# Smart Lender — Loan Eligibility Prediction

A machine-learning-powered Flask app that predicts whether a loan applicant is
likely to be approved, based on applicant details (income, credit history,
dependents, property area, etc.).

This implements the pipeline from the project plan: data preprocessing → EDA →
model training (Decision Tree, Random Forest, KNN, XGBoost) → model
evaluation/selection → Flask web app for real-time predictions.

## Project structure

```
smart-lender/
├── data/
│   └── loan_train.csv        # 491-row loan eligibility dataset
├── models/
│   └── model.pkl             # best model + fitted preprocessing pipeline (generated)
├── plots/                    # EDA charts + confusion matrices (generated)
├── static/
│   └── style.css
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── submit.html
├── app.py                    # Flask application
├── train_model.py            # preprocessing, EDA, training, evaluation, model export
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 1. Train the model

```bash
python train_model.py
```

This will:
- Load `data/loan_train.csv`
- Save EDA plots (count plots, distribution plots, bar charts) to `plots/`
- Handle missing values (mean for numeric, mode for categorical)
- Handle outliers (1st/99th percentile clipping on income/loan amount)
- Encode categorical variables and scale numeric features
- Balance the training set with SMOTE
- Train Decision Tree, Random Forest, KNN, and XGBoost
- Evaluate each with accuracy, confusion matrix, classification report, and 5-fold cross-validation
- Save confusion matrices + a model-comparison chart to `plots/`
- Pick the best model by test accuracy and save it (plus the fitted
  encoders/scaler) to `models/model.pkl`

## 2. Run the web app

```bash
python app.py
```

Visit `http://localhost:5000`. From there:
- **Home** — overview and stats on the deployed model
- **New Application** — enter applicant details
- **Decision page** — Approved/Rejected result with a confidence score and a summary of the inputs

## Notes on this run's results

- With this environment's package availability, `xgboost` and
  `imbalanced-learn` (SMOTE) weren't installed, so `train_model.py`
  automatically falls back to `HistGradientBoostingClassifier` and random
  oversampling respectively, with a console warning. Installing the packages
  in `requirements.txt` and re-running `train_model.py` will train with the
  real XGBoost + SMOTE as originally specified.
- On the public 491-row loan eligibility dataset used here, **Random Forest**
  came out on top (~84% test accuracy) in this run — XGBoost was close behind
  but slightly overfit (100% train / ~82% test). Exact numbers will shift
  with the random seed, the specific dataset pull, and whether real XGBoost/SMOTE
  are installed, so don't treat 94.7%/81.1% as a fixed benchmark — re-run
  `train_model.py` on your own data pull to get your own numbers.
- `app.py` loads whichever model `train_model.py` selected — no code changes
  needed if a different model wins on a re-run.

## Deployment

The Flask app binds to `0.0.0.0:5000` so it's ready to containerize/deploy
(e.g. to IBM Cloud, Heroku, or any platform that runs a WSGI app). For
production, run it behind a WSGI server such as `gunicorn`:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```
