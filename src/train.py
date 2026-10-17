"""
Train and compare churn models on the Telco Customer Churn dataset.

Run from the project root:
    python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "telco_churn.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
SEED = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


# ---------------------------------------------------------------- data
def load_data():
    df = pd.read_csv(DATA)
    df = df.drop(columns=["customerID"])
    # TotalCharges is text because new customers have a blank value
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def make_preprocessor():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


# -------------------------------------------------------------- models
def candidate_models(pos_weight):
    """Each model uses a different strategy for the class imbalance."""
    return {
        "LogReg (class_weight)": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "LogReg + SMOTE": ImbPipeline([
            ("prep", make_preprocessor()),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "RandomForest (class_weight)": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "XGBoost (scale_pos_weight)": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                scale_pos_weight=pos_weight, eval_metric="logloss",
                random_state=SEED, n_jobs=-1)),
        ]),
    }


def evaluate(model, X_te, y_te, threshold=0.5):
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "recall": recall_score(y_te, pred),
        "precision": precision_score(y_te, pred),
        "f1": f1_score(y_te, pred),
    }


def best_threshold(model, X_tr, y_tr, cv):
    """Pick the threshold with the best F1 using out-of-fold predictions
    on the TRAINING set only (so the test set stays untouched)."""
    oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    return float(thr[np.argmax(f1)])


# ---------------------------------------------------------------- main
def main():
    X, y = load_data()
    print(f"Rows: {len(X)}  |  churn rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    # 1) Compare strategies at the default 0.5 threshold
    rows = []
    fitted = {}
    for name, model in candidate_models(pos_weight).items():
        model.fit(X_tr, y_tr)
        fitted[name] = model
        rows.append({"model": name, **evaluate(model, X_te, y_te)})
    results = pd.DataFrame(rows).set_index("model").round(3)
    print("\n=== Model comparison (test set, threshold 0.5) ===")
    print(results.to_string())

    # 2) Tune XGBoost with randomized search (scored on ROC-AUC)
    print("\nTuning XGBoost ...")
    search = RandomizedSearchCV(
        candidate_models(pos_weight)["XGBoost (scale_pos_weight)"],
        param_distributions={
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.02, 0.05, 0.1],
            "clf__subsample": [0.7, 0.85, 1.0],
            "clf__colsample_bytree": [0.7, 0.85, 1.0],
            "clf__min_child_weight": [1, 3, 5],
        },
        n_iter=15, scoring="roc_auc", cv=cv, random_state=SEED, n_jobs=1,
    )
    search.fit(X_tr, y_tr)
    tuned = search.best_estimator_
    print("Best params:", search.best_params_)
    print(f"CV ROC-AUC: {search.best_score_:.3f}")

    # 3) Choose a decision threshold on training data only
    thr = best_threshold(tuned, X_tr, y_tr, cv)
    print(f"Chosen threshold: {thr:.2f}")

    tuned_default = evaluate(tuned, X_te, y_te, 0.5)
    tuned_thr = evaluate(tuned, X_te, y_te, thr)
    results.loc["XGBoost tuned (0.5)"] = pd.Series(tuned_default).round(3)
    results.loc[f"XGBoost tuned ({thr:.2f})"] = pd.Series(tuned_thr).round(3)
    print("\n=== Final comparison ===")
    print(results.to_string())

    # 4) Plots
    proba = tuned.predict_proba(X_te)[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_te, proba, ax=ax[0])
    ax[0].set_title("ROC curve (tuned XGBoost)")
    ConfusionMatrixDisplay.from_predictions(
        y_te, (proba >= thr).astype(int), ax=ax[1],
        display_labels=["Stays", "Churns"], colorbar=False)
    ax[1].set_title(f"Confusion matrix (threshold {thr:.2f})")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "evaluation.png", dpi=130)

    names = tuned.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(tuned.named_steps["clf"].feature_importances_, index=names)
    top = imp.nlargest(12).iloc[::-1]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    top.plot.barh(ax=ax2, color="#2b6cb0")
    ax2.set_title("Top 12 features (XGBoost importance)")
    fig2.tight_layout()
    fig2.savefig(MODEL_DIR / "feature_importance.png", dpi=130)

    # 5) Save everything the app needs
    joblib.dump({"pipeline": tuned, "threshold": thr}, MODEL_DIR / "churn_model.joblib")
    results.reset_index().to_csv(MODEL_DIR / "results.csv", index=False)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"threshold": thr, "test_metrics": tuned_thr,
                   "best_params": search.best_params_}, f, indent=2, default=float)
    print("\nSaved model + plots to", MODEL_DIR)


if __name__ == "__main__":
    main()
