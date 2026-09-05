"""
Task 1 — Supervised Learning: Calorie Burn Prediction (Regression)
Trains and compares Linear/Ridge/Lasso, KNN, XGBoost, Decision Tree,
Random Forest, and SVR. Saves the best model + scaler for the Streamlit app.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocessing import load_data, full_preprocess, scale_features, PROJECT_ROOT

RANDOM_STATE = 42
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


def main():
    # ---- Load + preprocess ----
    df = load_data()
    df = full_preprocess(df, cap_outlier_cols=["Calories_Burned", "Avg_BPM"])

    y = df["Calories_Burned"]
    X = df.drop(columns=["Calories_Burned"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Scale only for distance/gradient-based models (Linear/Ridge/Lasso/KNN/SVR)
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled, _ = scale_features(X_test, scaler)

    models = {
        "Linear Regression": (LinearRegression(), True),
        "Ridge Regression": (Ridge(alpha=1.0, random_state=RANDOM_STATE), True),
        "Lasso Regression": (Lasso(alpha=0.5, random_state=RANDOM_STATE), True),
        "KNN Regressor": (KNeighborsRegressor(n_neighbors=7), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1), False),
        "XGBoost": (XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0), False),
        "SVR": (SVR(kernel="rbf", C=100, epsilon=5), True),
    }

    results = []
    fitted = {}
    for name, (model, needs_scaling) in models.items():
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        metrics = evaluate(y_test, preds)
        metrics["Model"] = name
        results.append(metrics)
        fitted[name] = model
        print(f"{name:22s} | MAE={metrics['MAE']:8.2f} | RMSE={metrics['RMSE']:8.2f} | R2={metrics['R2']:.4f}")

    results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "R2"]].sort_values(
        "R2", ascending=False
    )
    print("\n=== Model Comparison (sorted by R2) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (R2={results_df.iloc[0]['R2']})")

    # ---- Save artifacts for the Streamlit app ----
    joblib.dump(best_model, os.path.join(OUTPUTS_DIR, "best_regressor.pkl"))
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "reg_scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(OUTPUTS_DIR, "reg_feature_columns.pkl"))
    joblib.dump(best_name in [n for n, (m, s) in models.items() if s],
                os.path.join(OUTPUTS_DIR, "reg_needs_scaling.pkl"))
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "regression_results.csv"), index=False)

    print("\nSaved: best_regressor.pkl, reg_scaler.pkl, reg_feature_columns.pkl, regression_results.csv")


if __name__ == "__main__":
    main()
