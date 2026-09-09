# src/models/baseline.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

DATA_PATH = Path("data/processed/skab_processed.parquet")
ARTIFACT_DIR = Path("src/models/artifacts")
MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
REPORT_PATH = ARTIFACT_DIR / "baseline_report.json"

SENSORS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

ROLLING_WINDOWS = [10, 30, 60]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_parquet(DATA_PATH)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    df = df.dropna(subset=["datetime"]).copy()

    for sensor in SENSORS:
        df[sensor] = pd.to_numeric(df[sensor], errors="coerce")

    df = df.dropna(subset=SENSORS + ["anomaly"])

    df = df.sort_values(
        ["source_file", "datetime"]
    ).reset_index(drop=True)

    return df


def create_features(df):
    result = df.copy()

    for sensor in SENSORS:
        result[f"{sensor}_diff"] = (
            result.groupby("source_file")[sensor]
            .diff()
        )

        for window in ROLLING_WINDOWS:
            grouped = result.groupby("source_file")[sensor]

            shifted = grouped.shift(1)

            rolling_mean = (
                shifted.groupby(result["source_file"])
                .rolling(window, min_periods=5)
                .mean()
                .reset_index(level=0, drop=True)
            )

            rolling_std = (
                shifted.groupby(result["source_file"])
                .rolling(window, min_periods=5)
                .std()
                .reset_index(level=0, drop=True)
            )

            result[f"{sensor}_mean_{window}"] = rolling_mean
            result[f"{sensor}_std_{window}"] = rolling_std

            result[f"{sensor}_z_{window}"] = (
                result[sensor] - rolling_mean
            ) / (rolling_std + 1e-8)

    feature_columns = [
        column
        for column in result.columns
        if any(
            column == sensor
            or column.startswith(f"{sensor}_")
            for sensor in SENSORS
        )
    ]

    result[feature_columns] = (
        result[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    return result, feature_columns


def split_by_experiment(df):
    experiments = sorted(
        df["source_file"].dropna().unique()
    )

    if len(experiments) < 5:
        raise ValueError(
            "Not enough experiments for train/validation/test split."
        )

    n = len(experiments)

    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_experiments = experiments[:train_end]
    val_experiments = experiments[train_end:val_end]
    test_experiments = experiments[val_end:]

    train_df = df[
        df["source_file"].isin(train_experiments)
    ].copy()

    val_df = df[
        df["source_file"].isin(val_experiments)
    ].copy()

    test_df = df[
        df["source_file"].isin(test_experiments)
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        train_experiments,
        val_experiments,
        test_experiments,
    )


def train_model(X_train):
    model = IsolationForest(
        n_estimators=400,
        max_samples="auto",
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train)

    return model


def get_scores(model, X):
    return -model.decision_function(X)


def find_best_threshold(y_true, scores):
    thresholds = np.percentile(
        scores,
        np.linspace(80, 99.9, 250),
    )

    best_threshold = thresholds[0]
    best_f1 = -1

    for threshold in thresholds:
        predictions = (
            scores >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return float(best_threshold), float(best_f1)


def calculate_metrics(y_true, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "false_alarm_rate": float(
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),
        "missed_alarm_rate": float(
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        ),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, scores)
        )
        metrics["pr_auc"] = float(
            average_precision_score(
                y_true,
                scores,
            )
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def evaluate(model, df, feature_columns, threshold):
    X = df[feature_columns].values
    y = df["anomaly"].astype(int).values

    scores = get_scores(model, X)

    predictions = (
        scores >= threshold
    ).astype(int)

    metrics = calculate_metrics(
        y,
        predictions,
        scores,
    )

    metrics["rows"] = int(len(df))
    metrics["actual_anomalies"] = int(y.sum())
    metrics["predicted_anomalies"] = int(
        predictions.sum()
    )

    return metrics


def main():
    print("=" * 70)
    print("ISOLATION FOREST BASELINE")
    print("=" * 70)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/7] Loading dataset...")
    df = load_data()

    print(f"Rows: {len(df):,}")
    print(
        f"Experiments: "
        f"{df['source_file'].nunique()}"
    )

    print("\n[2/7] Creating time-aware features...")
    df, feature_columns = create_features(df)

    print(
        f"Feature count: "
        f"{len(feature_columns)}"
    )

    print("\n[3/7] Splitting by experiment...")
    (
        train_df,
        val_df,
        test_df,
        train_experiments,
        val_experiments,
        test_experiments,
    ) = split_by_experiment(df)

    print(
        f"Train experiments: "
        f"{len(train_experiments)}"
    )
    print(
        f"Validation experiments: "
        f"{len(val_experiments)}"
    )
    print(
        f"Test experiments: "
        f"{len(test_experiments)}"
    )

    print("\n[4/7] Training only on normal data...")

    normal_train = train_df[
        train_df["anomaly"] == 0
    ].copy()

    X_train = normal_train[
        feature_columns
    ].values

    print(
        f"Normal training rows: "
        f"{len(normal_train):,}"
    )

    model = train_model(X_train)

    print("\n[5/7] Selecting threshold on validation data...")

    X_val = val_df[
        feature_columns
    ].values

    y_val = val_df[
        "anomaly"
    ].astype(int).values

    val_scores = get_scores(
        model,
        X_val,
    )

    threshold, validation_f1 = (
        find_best_threshold(
            y_val,
            val_scores,
        )
    )

    print(
        f"Best threshold: {threshold:.6f}"
    )
    print(
        f"Validation F1: {validation_f1:.4f}"
    )

    print("\n[6/7] Final test evaluation...")

    test_metrics = evaluate(
        model,
        test_df,
        feature_columns,
        threshold,
    )

    print("\nTEST RESULTS")
    print("-" * 50)

    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

    print("\n[7/7] Saving artifacts...")

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "sensors": SENSORS,
            "rolling_windows": ROLLING_WINDOWS,
        },
        MODEL_PATH,
    )

    report = {
        "model": "IsolationForest",
        "random_state": 42,
        "n_estimators": 400,
        "feature_count": len(feature_columns),
        "train_experiments": train_experiments,
        "validation_experiments": val_experiments,
        "test_experiments": test_experiments,
        "normal_training_rows": len(
            normal_train
        ),
        "validation_f1": validation_f1,
        "threshold": threshold,
        "test_metrics": test_metrics,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )
    print(
        f"Report saved: {REPORT_PATH}"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()