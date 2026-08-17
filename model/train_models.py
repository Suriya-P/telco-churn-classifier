"""
train_models.py

Trains 5 classification models on the Telco Customer Churn dataset and
evaluates each with 6 metrics: Accuracy, AUC, Precision, Recall, F1, MCC.

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)

Saves:
    - model/*.pkl for each trained model + the scaler
    - model/metrics_summary.csv
    - model/feature_columns.json (needed by the Streamlit app)
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

from feature_engineering import load_raw, build_features


RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, scale=False, scaler=None):
    X_eval = scaler.transform(X_test) if scale else X_test
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    return metrics


def main():
    raw = load_raw("data/raw/telco_churn.csv")
    df, feature_cols = build_features(raw)

    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features (needed for Logistic Regression / kNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    trained_models = {}

    # ---- 1. Logistic Regression ----
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate("Logistic Regression", lr, X_test, y_test, scale=True, scaler=scaler))
    trained_models["logistic_regression"] = lr

    # ---- 2. Decision Tree ----
    dt = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    results.append(evaluate("Decision Tree", dt, X_test, y_test))
    trained_models["decision_tree"] = dt

    # ---- 3. K-Nearest Neighbors ----
    knn = KNeighborsClassifier(n_neighbors=15)
    knn.fit(X_train_scaled, y_train)
    results.append(evaluate("kNN", knn, X_test, y_test, scale=True, scaler=scaler))
    trained_models["knn"] = knn

    # ---- 4. Gaussian Naive Bayes ----
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    results.append(evaluate("Naive Bayes", nb, X_test, y_test))
    trained_models["naive_bayes"] = nb

    # ---- 5. Random Forest (Ensemble) ----
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest (Ensemble)", rf, X_test, y_test))
    trained_models["random_forest"] = rf

    # ---- Save results ----
    results_df = pd.DataFrame(results)
    results_df.to_csv("model/metrics_summary.csv", index=False)
    print(results_df.round(4).to_string(index=False))

    # ---- Save models ----
    for name, model in trained_models.items():
        with open(f"model/{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("model/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    # ---- Save a held-out test sample as test_data.csv for the Streamlit app ----
    test_export = X_test.copy()
    test_export["Target"] = y_test.values
    test_export.to_csv("test_data.csv", index=False)
    print(f"\nSaved test_data.csv with {len(test_export)} rows for Streamlit demo")

    print("\nAll models and metrics saved to model/")


if __name__ == "__main__":
    main()
