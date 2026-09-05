"""
Shared preprocessing: missing values, outlier capping, encoding, scaling.
Used by both train_regression.py and train_clustering.py.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "fitbit_dataset.csv")

CATEGORICAL_COLS = ["Gender", "Workout_Type", "Experience_Level"]
NUMERIC_COLS = [
    "Age", "Weight (kg)", "Height (m)", "BMI", "Fat_Percentage",
    "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
    "Water_Intake (liters)", "Workout_Frequency (days/week)",
]


def load_data(path=None):
    if path is None:
        path = DEFAULT_DATA_PATH
    return pd.read_csv(path)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in NUMERIC_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def cap_outliers(df: pd.DataFrame, cols=None, factor=1.5) -> pd.DataFrame:
    """IQR-based capping (winsorizing), not row deletion — preserves sample size."""
    df = df.copy()
    cols = cols or NUMERIC_COLS
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def encode_categoricals(df: pd.DataFrame, fit_encoders=None):
    """One-hot encode Gender/Workout_Type. Experience_Level already numeric in real data."""
    df = df.copy()
    if df["Experience_Level"].dtype == object:
        exp_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        df["Experience_Level_enc"] = df["Experience_Level"].map(exp_order)
    else:
        df["Experience_Level_enc"] = df["Experience_Level"]

    df = pd.get_dummies(df, columns=["Gender", "Workout_Type"], drop_first=True)
    df = df.drop(columns=["Experience_Level"])
    return df


def scale_features(X: pd.DataFrame, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns, index=X.index), scaler


def full_preprocess(df: pd.DataFrame, cap_outlier_cols=None) -> pd.DataFrame:
    df = handle_missing(df)
    df = cap_outliers(df, cols=cap_outlier_cols)
    df = encode_categoricals(df)
    return df
