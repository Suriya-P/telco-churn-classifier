"""
feature_engineering.py

Cleans and encodes the Telco Customer Churn dataset for classification.

Target:
    Churn: Yes -> 1, No -> 0

Notes:
- customerID is dropped (identifier, no predictive value)
- TotalCharges has 11 blank values, all with tenure=0 (new customers who haven't been billed yet) -> filled with 0
- Categorical columns are one-hot encoded
- No target leakage: Churn is the only target column in this dataset, and it is excluded from features
"""

import pandas as pd
import numpy as np


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_features(df: pd.DataFrame):
    df = df.copy()

    # ---- 1. Clean TotalCharges ----
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # ---- 2. Drop identifier ----
    df = df.drop(columns=["customerID"])

    # ---- 3. Encode target ----
    df["Target"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn"])

    # ---- 4. Identify categorical vs numeric columns ----
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols + ["Target"]]

    # ---- 5. One-hot encode categoricals ----
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Ensure all encoded columns are numeric (0/1 instead of True/False)
    bool_cols = df_encoded.select_dtypes(include="bool").columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    feature_cols = [c for c in df_encoded.columns if c != "Target"]

    return df_encoded, feature_cols


if __name__ == "__main__":
    raw = load_raw("data/raw/telco_churn.csv")
    processed, feature_cols = build_features(raw)
    print(f"Processed shape: {processed.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Target balance:\n{processed['Target'].value_counts(normalize=True)}")
    processed.to_csv("data/processed_telco_features.csv", index=False)
    print("Saved to data/processed_telco_features.csv")
