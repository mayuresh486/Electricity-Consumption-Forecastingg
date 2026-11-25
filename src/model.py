"""Machine Learning pipeline for Electricity Consumption Forecasting.

Loads the UCI Appliances Energy Prediction dataset (Dataset ID: 374),
performs chronological train-test splitting without data leakage, trains
Linear Regression, Decision Tree Regression, and Gradient Boosting Regression
models, evaluates them with MAE, RMSE, and R², and serves predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATASET_ID = 374
UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction"
UCI_CSV_URL = "https://archive.ics.uci.edu/static/public/374/data.csv"
TARGET_COL = "Appliances"

CORE_FEATURES: list[str] = [
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T_out",
    "RH_out",
    "Windspeed",
    "hour",
    "dayofweek",
]

ALL_SENSOR_FEATURES: list[str] = [
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "hour",
    "dayofweek",
]

FEATURE_SETS: dict[str, list[str]] = {
    "Core Prediction Features (10)": CORE_FEATURES,
    "All Clean Sensor Features (27)": ALL_SENSOR_FEATURES,
}

DAY_NAMES: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "date": "Timestamp (10-minute intervals)",
    "Appliances": "Appliance energy consumption (Wh) — Target",
    "lights": "Lighting energy consumption (Wh)",
    "T1": "Kitchen temperature (°C)",
    "RH_1": "Kitchen relative humidity (%)",
    "T2": "Living room temperature (°C)",
    "RH_2": "Living room relative humidity (%)",
    "T3": "Laundry room temperature (°C)",
    "RH_3": "Laundry room relative humidity (%)",
    "T4": "Office room temperature (°C)",
    "RH_4": "Office room relative humidity (%)",
    "T5": "Bathroom temperature (°C)",
    "RH_5": "Bathroom relative humidity (%)",
    "T6": "Outside north temperature (°C)",
    "RH_6": "Outside north relative humidity (%)",
    "T7": "Ironing room temperature (°C)",
    "RH_7": "Ironing room relative humidity (%)",
    "T8": "Teenager room temperature (°C)",
    "RH_8": "Teenager room relative humidity (%)",
    "T9": "Parents room temperature (°C)",
    "RH_9": "Parents room relative humidity (%)",
    "T_out": "Weather station outdoor temperature (°C)",
    "Press_mm_hg": "Atmospheric pressure (mm Hg)",
    "RH_out": "Weather station outdoor humidity (%)",
    "Windspeed": "Wind speed (m/s)",
    "Visibility": "Visibility (km)",
    "Tdewpoint": "Dew point temperature (°C)",
    "rv1": "Random variable 1 (noise control)",
    "rv2": "Random variable 2 (noise control)",
    "hour": "Hour of day (0–23)",
    "dayofweek": "Day of week (0=Monday .. 6=Sunday)",
}

MODEL_NAMES: list[str] = [
    "Linear Regression",
    "Decision Tree Regression",
    "Gradient Boosting Regression",
]


@dataclass
class DatasetMetadata:
    """Summary metadata captured during raw dataset ingestion."""

    total_records: int
    raw_column_count: int
    raw_feature_count: int
    target_col: str
    duplicates_removed: int
    total_missing: int
    missing_by_col: dict[str, int]
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    source_used: str


@dataclass
class SplitData:
    """Chronologically split training and testing datasets."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_dates: pd.Series
    test_dates: pd.Series
    split_idx: int
    train_ratio: float
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class TrainingBundle:
    """Trained models, predictions, evaluation metrics, and feature drivers."""

    models: dict[str, Pipeline]
    predictions: dict[str, np.ndarray]
    metrics_df: pd.DataFrame
    best_model_name: str
    best_mae_model_name: str
    feature_importances: dict[str, pd.DataFrame]
    feature_cols: list[str]
    y_test: pd.Series
    test_dates: pd.Series


def _fetch_raw_dataframe() -> tuple[pd.DataFrame, str]:
    """Fetch the UCI Appliances Energy Prediction dataset with CSV fallback."""
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=DATASET_ID)
        X_raw = dataset.data.features.copy()
        y_raw = dataset.data.targets.copy()
        if isinstance(y_raw, pd.DataFrame):
            if TARGET_COL in y_raw.columns:
                target_series = y_raw[TARGET_COL]
            else:
                target_series = y_raw.iloc[:, 0].rename(TARGET_COL)
        else:
            target_series = pd.Series(y_raw, name=TARGET_COL)
        raw_df = pd.concat([X_raw, target_series], axis=1)
        return raw_df, "UCI ML Repository (ucimlrepo id=374)"
    except Exception as uci_exc:
        try:
            raw_df = pd.read_csv(UCI_CSV_URL)
            return raw_df, "UCI Direct CSV Archive (fallback)"
        except Exception as csv_exc:
            raise RuntimeError(
                f"Failed to load UCI dataset {DATASET_ID} via ucimlrepo ({uci_exc}) "
                f"and direct CSV ({csv_exc})."
            ) from csv_exc


def load_appliances_dataset() -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load, audit, sort chronologically, and extract temporal features.

    Note: Missing numeric feature values are NOT imputed here so that
    imputation happens strictly inside the training pipeline after the
    chronological train-test split, preventing any data leakage.
    """
    raw_df, source_used = _fetch_raw_dataframe()
    raw_column_count = int(raw_df.shape[1])

    missing_by_col = {col: int(raw_df[col].isna().sum()) for col in raw_df.columns}
    total_missing = int(sum(missing_by_col.values()))

    df = raw_df.copy()

    if "date" not in df.columns:
        raise ValueError("Expected 'date' column in UCI Appliances Energy Prediction dataset.")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")

    # Fix UCI timestamp formatting where date and time may lack a space separator.
    df["date"] = (
        df["date"]
        .astype(str)
        .str.strip()
        .str.replace(r"^(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})$", r"\1 \2", regex=True)
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop invalid timestamps or missing target observations before sorting.
    df = df.dropna(subset=["date", TARGET_COL])

    before_dedup = len(df)
    df = df.drop_duplicates()
    duplicates_removed = int(before_dedup - len(df))

    # Chronological ordering is required for time-series split integrity.
    df = df.sort_values("date").reset_index(drop=True)

    # Deterministic calendar features derived strictly from row timestamp.
    df["hour"] = df["date"].dt.hour.astype(int)
    df["dayofweek"] = df["date"].dt.dayofweek.astype(int)
    df["day"] = df["date"].dt.day.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    metadata = DatasetMetadata(
        total_records=int(len(df)),
        raw_column_count=raw_column_count,
        raw_feature_count=max(0, raw_column_count - 1),
        target_col=TARGET_COL,
        duplicates_removed=duplicates_removed,
        total_missing=total_missing,
        missing_by_col=missing_by_col,
        date_start=df["date"].min(),
        date_end=df["date"].max(),
        source_used=source_used,
    )
    return df, metadata


def prepare_chronological_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_ratio: float = 0.80,
) -> SplitData:
    """Perform a strict chronological train-test split without shuffling."""
    if not 0.50 <= train_ratio <= 0.95:
        raise ValueError("train_ratio must be between 0.50 and 0.95.")
    if len(df) < 20:
        raise ValueError("Dataset has too few rows to split reliably.")

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols}")

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return SplitData(
        X_train=train_df[feature_cols].copy(),
        X_test=test_df[feature_cols].copy(),
        y_train=train_df[TARGET_COL].astype(float).copy(),
        y_test=test_df[TARGET_COL].astype(float).copy(),
        train_dates=train_df["date"].copy(),
        test_dates=test_df["date"].copy(),
        split_idx=split_idx,
        train_ratio=train_ratio,
        train_start=train_df["date"].iloc[0],
        train_end=train_df["date"].iloc[-1],
        test_start=test_df["date"].iloc[0],
        test_end=test_df["date"].iloc[-1],
    )


def build_model_pipelines(
    dt_max_depth: int = 6,
    dt_min_samples_leaf: int = 25,
    gb_n_estimators: int = 120,
    gb_learning_rate: float = 0.05,
    gb_max_depth: int = 3,
    gb_subsample: float = 0.8,
) -> dict[str, Pipeline]:
    """Construct scikit-learn Pipelines that keep imputation and scaling leak-free."""
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Decision Tree Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    DecisionTreeRegressor(
                        max_depth=int(dt_max_depth),
                        min_samples_leaf=int(dt_min_samples_leaf),
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "regressor",
                    GradientBoostingRegressor(
                        n_estimators=int(gb_n_estimators),
                        learning_rate=float(gb_learning_rate),
                        max_depth=int(gb_max_depth),
                        subsample=float(gb_subsample),
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def _extract_feature_importance(
    model_name: str,
    pipeline: Pipeline,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Extract normalized importance or standardized coefficient magnitude."""
    regressor = pipeline.named_steps["regressor"]
    if model_name == "Linear Regression":
        raw_vals = np.abs(regressor.coef_)
        signed_vals = regressor.coef_
        total = float(raw_vals.sum())
        norm_vals = raw_vals / total if total > 0 else raw_vals
        imp_df = pd.DataFrame(
            {
                "Feature": feature_cols,
                "Importance": norm_vals,
                "Signed Coefficient": signed_vals,
            }
        )
    else:
        raw_vals = regressor.feature_importances_
        imp_df = pd.DataFrame(
            {
                "Feature": feature_cols,
                "Importance": raw_vals,
            }
        )
    return imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)


def train_and_evaluate_models(
    split_data: SplitData,
    dt_max_depth: int = 6,
    dt_min_samples_leaf: int = 25,
    gb_n_estimators: int = 120,
    gb_learning_rate: float = 0.05,
    gb_max_depth: int = 3,
    gb_subsample: float = 0.8,
) -> TrainingBundle:
    """Train all three regression models on X_train and evaluate on X_test."""
    pipelines = build_model_pipelines(
        dt_max_depth=dt_max_depth,
        dt_min_samples_leaf=dt_min_samples_leaf,
        gb_n_estimators=gb_n_estimators,
        gb_learning_rate=gb_learning_rate,
        gb_max_depth=gb_max_depth,
        gb_subsample=gb_subsample,
    )

    feature_cols = list(split_data.X_train.columns)
    trained_models: dict[str, Pipeline] = {}
    predictions: dict[str, np.ndarray] = {}
    feature_importances: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []

    for name, pipeline in pipelines.items():
        start_time = time.perf_counter()
        pipeline.fit(split_data.X_train, split_data.y_train)
        elapsed = time.perf_counter() - start_time

        y_pred = pipeline.predict(split_data.X_test)
        mae = float(mean_absolute_error(split_data.y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(split_data.y_test, y_pred)))
        r2 = float(r2_score(split_data.y_test, y_pred))

        trained_models[name] = pipeline
        predictions[name] = y_pred
        feature_importances[name] = _extract_feature_importance(name, pipeline, feature_cols)
        metric_rows.append(
            {
                "Model": name,
                "MAE (Wh)": mae,
                "RMSE (Wh)": rmse,
                "R² Score": r2,
                "Train Time (s)": elapsed,
            }
        )

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(by=["RMSE (Wh)", "MAE (Wh)"], ascending=[True, True])
        .reset_index(drop=True)
    )
    # Lowest RMSE and highest R² are mathematically equivalent on a fixed test set.
    best_model_name = str(metrics_df.iloc[0]["Model"])
    best_mae_model_name = str(
        metrics_df.loc[metrics_df["MAE (Wh)"].idxmin(), "Model"]
    )

    return TrainingBundle(
        models=trained_models,
        predictions=predictions,
        metrics_df=metrics_df,
        best_model_name=best_model_name,
        best_mae_model_name=best_mae_model_name,
        feature_importances=feature_importances,
        feature_cols=feature_cols,
        y_test=split_data.y_test,
        test_dates=split_data.test_dates,
    )


def compute_training_feature_stats(
    X_train: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Compute min, median, mean, and max strictly from X_train for UI defaults."""
    stats: dict[str, dict[str, float]] = {}
    for col in X_train.columns:
        series = pd.to_numeric(X_train[col], errors="coerce").dropna()
        stats[col] = {
            "min": float(series.min()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "max": float(series.max()),
        }
    return stats


def predict_consumption(
    pipeline: Pipeline,
    feature_cols: list[str],
    inputs: dict[str, float | int],
) -> tuple[float, float]:
    """Predict appliance energy consumption (Wh) from user-supplied inputs.

    Returns (raw_prediction, non_negative_prediction).
    """
    missing = [col for col in feature_cols if col not in inputs]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    ordered_row = {col: float(inputs[col]) for col in feature_cols}
    for col, val in ordered_row.items():
        if not np.isfinite(val):
            raise ValueError(f"Input '{col}' must be a finite numeric value.")

    input_df = pd.DataFrame([ordered_row], columns=feature_cols)
    raw_pred = float(pipeline.predict(input_df)[0])
    clamped_pred = max(0.0, raw_pred)
    return raw_pred, clamped_pred
