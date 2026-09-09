# src/models/pca_detector.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.models.baseline import (
    load_data,
    create_features,
    split_by_experiment,
)


ARTIFACT_DIR = Path("src/models/artifacts")

MODEL_PATH = (
    ARTIFACT_DIR / "pca_detector.joblib"
)

REPORT_PATH = (
    ARTIFACT_DIR / "pca_detector_report.json"
)


def prepare_matrix(df, feature_columns):
    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    return X


def train_pca(X_train):
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_train
    )

    pca_full = PCA(
        random_state=42
    )

    pca_full.fit(X_scaled)

    cumulative = np.cumsum(
        pca_full.explained_variance_ratio_
    )

    n_components = (
        np.searchsorted(
            cumulative,
            0.95,
        )
        + 1
    )

    n_components = max(
        2,
        min(
            n_components,
            X_train.shape[1] - 1,
        ),
    )

    pca = PCA(
        n_components=n_components,
        random_state=42,
    )

    pca.fit(X_scaled)

    return scaler, pca


def calculate_scores(
    scaler,
    pca,
    X,
):
    X_scaled = scaler.transform(X)

    transformed = pca.transform(
        X_scaled
    )

    reconstructed = pca.inverse_transform(
        transformed
    )

    residual = (
        X_scaled - reconstructed
    )

    q_score = np.sum(
        residual ** 2,
        axis=1,
    )

    eigenvalues = (
        pca.explained_variance_
    )

    t_score = np.sum(
        (
            transformed
            / np.sqrt(
                eigenvalues + 1e-12
            )
        )
        ** 2,
        axis=1,
    )

    q_norm = (
        q_score
        / (
            np.median(q_score)
            + 1e-12
        )
    )

    t_norm = (
        t_score
        / (
            np.median(t_score)
            + 1e-12
        )
    )

    combined = (
        np.log1p(q_norm)
        + np.log1p(t_norm)
    )

    return (
        combined,
        t_score,
        q_score,
    )


def evaluate(
    y_true,
    scores,
    threshold,
):
    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
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
        "roc_auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        )
        if len(np.unique(y_true)) > 1
        else None,
        "pr_auc": float(
            average_precision_score(
                y_true,
                scores,
            )
        )
        if len(np.unique(y_true)) > 1
        else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "false_alarm_rate": float(
            fp / (fp + tn)
            if fp + tn > 0
            else 0
        ),
        "missed_alarm_rate": float(
            fn / (fn + tp)
            if fn + tp > 0
            else 0
        ),
        "predicted_anomalies": int(
            predictions.sum()
        ),
        "actual_anomalies": int(
            y_true.sum()
        ),
    }


def find_best_threshold(
    y_true,
    scores,
):
    percentiles = np.linspace(
        70,
        99.9,
        1000,
    )

    thresholds = np.percentile(
        scores,
        percentiles,
    )

    best = None

    for threshold in thresholds:

        metrics = evaluate(
            y_true,
            scores,
            threshold,
        )

        if best is None:
            best = metrics
            best["threshold"] = float(
                threshold
            )
            continue

        current_score = (
            metrics["f1_score"]
        )

        best_score = (
            best["f1_score"]
        )

        if current_score > best_score:

            best = metrics

            best["threshold"] = float(
                threshold
            )

    return best


def main():

    print("=" * 70)
    print(
        "PCA T² + Q MULTIVARIATE ANOMALY DETECTOR"
    )
    print("=" * 70)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/8] Loading dataset..."
    )

    df = load_data()

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Experiments: "
        f"{df['source_file'].nunique()}"
    )

    print(
        "\n[2/8] Creating features..."
    )

    df, feature_columns = (
        create_features(df)
    )

    print(
        f"Features: "
        f"{len(feature_columns)}"
    )

    print(
        "\n[3/8] Experiment-level split..."
    )

    (
        train_df,
        val_df,
        test_df,
        train_experiments,
        val_experiments,
        test_experiments,
    ) = split_by_experiment(df)

    print(
        f"Train: {train_experiments}"
    )

    print(
        f"Validation: {val_experiments}"
    )

    print(
        f"Test: {test_experiments}"
    )

    print(
        "\n[4/8] Training only on normal data..."
    )

    normal_train = train_df[
        train_df["anomaly"] == 0
    ].copy()

    X_train = prepare_matrix(
        normal_train,
        feature_columns,
    )

    print(
        f"Normal rows: "
        f"{len(X_train):,}"
    )

    scaler, pca = train_pca(
        X_train
    )

    print(
        f"PCA components: "
        f"{pca.n_components_}"
    )

    print(
        "Explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    print(
        "\n[5/8] Validation scoring..."
    )

    X_val = prepare_matrix(
        val_df,
        feature_columns,
    )

    y_val = (
        val_df["anomaly"]
        .astype(int)
        .values
    )

    val_scores, _, _ = (
        calculate_scores(
            scaler,
            pca,
            X_val,
        )
    )

    best_validation = (
        find_best_threshold(
            y_val,
            val_scores,
        )
    )

    threshold = (
        best_validation["threshold"]
    )

    print(
        f"Validation threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Validation precision: "
        f"{best_validation['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{best_validation['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_validation['f1_score']:.4f}"
    )

    print(
        "\n[6/8] Final unseen test evaluation..."
    )

    X_test = prepare_matrix(
        test_df,
        feature_columns,
    )

    y_test = (
        test_df["anomaly"]
        .astype(int)
        .values
    )

    test_scores, t_scores, q_scores = (
        calculate_scores(
            scaler,
            pca,
            X_test,
        )
    )

    test_metrics = evaluate(
        y_test,
        test_scores,
        threshold,
    )

    print(
        "\nTEST RESULTS"
    )

    print("-" * 60)

    for key, value in (
        test_metrics.items()
    ):

        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:25s}: "
                f"{value:.4f}"
            )
        else:
            print(
                f"{key:25s}: "
                f"{value}"
            )

    print(
        "\n[7/8] Saving model..."
    )

    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "feature_columns": feature_columns,
            "threshold": threshold,
        },
        MODEL_PATH,
    )

    report = {
        "model": "PCA_T2_Q",
        "features": len(
            feature_columns
        ),
        "pca_components": int(
            pca.n_components_
        ),
        "explained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
        "train_experiments": (
            train_experiments
        ),
        "validation_experiments": (
            val_experiments
        ),
        "test_experiments": (
            test_experiments
        ),
        "validation": best_validation,
        "test": test_metrics,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=4,
        )

    print(
        f"Model saved: "
        f"{MODEL_PATH}"
    )

    print(
        f"Report saved: "
        f"{REPORT_PATH}"
    )

    print(
        "\n[8/8] DONE"
    )


if __name__ == "__main__":
    main()