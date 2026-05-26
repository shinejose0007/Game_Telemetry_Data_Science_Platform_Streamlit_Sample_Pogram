from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sqlite3

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def _regression_metrics(y_true, pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    mae = float(mean_absolute_error(y_true, pred))
    r2 = float(r2_score(y_true, pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def _classification_metrics(y_true, pred, proba=None):
    metrics = {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if proba is not None and len(set(y_true)) > 1:
        try:
            metrics["ROC_AUC"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
    return metrics


def save_model_run(model_name: str, target: str, metrics: dict, db_path: str = "data/game_telemetry.db", notes: str = ""):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    for metric_name, metric_value in metrics.items():
        cur.execute(
            """
            INSERT INTO model_runs (run_time, model_name, target, metric_name, metric_value, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (datetime.utcnow().isoformat(), model_name, target, metric_name, float(metric_value), notes),
        )
    conn.commit()
    conn.close()


def train_revenue_forecast(features: pd.DataFrame, model_choice: str = "Random Forest") -> tuple:
    df = features.dropna(subset=["next_day_revenue"]).copy()
    if len(df) < 100:
        raise ValueError("Not enough rows to train the revenue forecasting model.")

    target = "next_day_revenue"
    numeric_cols = [
        "sessions", "total_revenue", "avg_revenue", "total_plays", "total_errors",
        "avg_duration_min", "unique_games", "error_rate", "revenue_per_session",
        "day_of_week", "is_weekend", "month", "week"
    ]
    categorical_cols = ["city", "location_type", "device_type", "software_version"]

    X = df[numeric_cols + categorical_cols].fillna(0)
    y = df[target]

    # time-aware split
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    if model_choice == "Linear Regression":
        estimator = LinearRegression()
    else:
        estimator = RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=2)

    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = _regression_metrics(y_test, pred)

    path = MODEL_DIR / "revenue_forecast_model.joblib"
    joblib.dump(model, path)
    save_model_run(model_choice, target, metrics, notes="Forecast next-day revenue from daily device KPIs.")
    result_df = X_test.copy()
    result_df["actual_next_day_revenue"] = y_test.values
    result_df["predicted_next_day_revenue"] = pred
    return model, metrics, result_df, path


def train_maintenance_classifier(features: pd.DataFrame, model_choice: str = "Random Forest") -> tuple:
    df = features.dropna(subset=["maintenance_next_14d"]).copy()
    if df["maintenance_next_14d"].nunique() < 2:
        raise ValueError("The generated data has only one maintenance class. Regenerate data and try again.")
    if len(df) < 100:
        raise ValueError("Not enough rows to train the maintenance model.")

    target = "maintenance_next_14d"
    numeric_cols = [
        "sessions", "total_revenue", "avg_revenue", "total_plays", "total_errors",
        "avg_duration_min", "unique_games", "error_rate", "revenue_per_session",
        "day_of_week", "is_weekend", "month", "week"
    ]
    categorical_cols = ["city", "location_type", "device_type", "software_version"]

    X = df[numeric_cols + categorical_cols].fillna(0)
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    if model_choice == "Logistic Regression":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        estimator = RandomForestClassifier(
            n_estimators=180,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    model = Pipeline([("preprocess", preprocessor), ("model", estimator)])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    proba = None
    if hasattr(model.named_steps["model"], "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]

    metrics = _classification_metrics(y_test, pred, proba)
    path = MODEL_DIR / "maintenance_risk_model.joblib"
    joblib.dump(model, path)
    save_model_run(model_choice, target, metrics, notes="Predict maintenance need within next 14 days.")

    result_df = X_test.copy()
    result_df["actual_maintenance_next_14d"] = y_test.values
    result_df["predicted_maintenance_next_14d"] = pred
    if proba is not None:
        result_df["risk_probability"] = proba
    return model, metrics, result_df, path


def train_anomaly_detector(features: pd.DataFrame) -> tuple:
    df = features.copy()
    if len(df) < 100:
        raise ValueError("Not enough rows to train the anomaly detector.")

    numeric_cols = [
        "sessions", "total_revenue", "avg_revenue", "total_plays", "total_errors",
        "avg_duration_min", "unique_games", "error_rate", "revenue_per_session",
    ]
    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(n_estimators=150, contamination=0.035, random_state=42)
    model.fit(X)
    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"] = (model.predict(X) == -1).astype(int)

    path = MODEL_DIR / "anomaly_detector.joblib"
    joblib.dump(model, path)

    metrics = {
        "Detected_Anomaly_Rate": float(df["is_anomaly"].mean()),
        "Detected_Anomaly_Count": float(df["is_anomaly"].sum()),
    }
    save_model_run("Isolation Forest", "device_day_anomaly", metrics, notes="Detect unusual daily device behavior.")
    return model, metrics, df.sort_values("anomaly_score"), path
