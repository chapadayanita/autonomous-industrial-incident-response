# NEXT STEP
# Create: src/models/compare_models.py

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

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
    get_scores,
)


ARTIFACT_DIR = Path("src/models/artifacts")
MODEL_PATH = ARTIFACT_DIR / "isolation_forest.joblib"
REPORT_PATH = ARTIFACT_DIR / "model_comparison.json"


def evaluate_model(y_true, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": round(
            accuracy_score(y_true, predictions), 4
        ),
        "precision": round(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "recall": round(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "f1_score": round(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),
            4,
        ),
        "roc_auc": round(
            roc_auc_score(y_true, scores), 4
        )
        if len(np.unique(y_true)) > 1
        else None,
        "pr_auc": round(
            average_precision_score(
                y_true,
                scores,
            ),
            4,
        )
        if len(np.unique(y_true)) > 1
        else None,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "false_alarm_rate": round(
            fp / (fp + tn)
            if (fp + tn) else 0,
            4,
        ),
        "missed_alarm_rate": round(
            fn / (fn + tp)
            if (fn + tp) else 0,
            4,
        ),
    }


def evaluate_thresholds(
    y_true,
    scores,
):
    thresholds = np.percentile(
        scores,
        np.linspace(70, 99.9, 500),
    )

    results = []

    for threshold in thresholds:

        metrics = evaluate_model(
            y_true,
            scores,
            threshold,
        )

        metrics["threshold"] = float(
            threshold
        )

        results.append(metrics)

    return results


def main():

    print("=" * 70)
    print("MODEL EVALUATION & THRESHOLD ANALYSIS")
    print("=" * 70)

    print("\n[1/5] Loading model...")

    artifact = joblib.load(
        MODEL_PATH
    )

    model = artifact["model"]
    feature_columns = artifact[
        "feature_columns"
    ]

    print(
        f"Features: {len(feature_columns)}"
    )

    print("\n[2/5] Loading dataset...")

    df = load_data()

    df, _ = create_features(df)

    (
        train_df,
        val_df,
        test_df,
        train_experiments,
        val_experiments,
        test_experiments,
    ) = split_by_experiment(df)

    print(
        f"Test experiments: "
        f"{test_experiments}"
    )

    print("\n[3/5] Generating anomaly scores...")

    X_test = test_df[
        feature_columns
    ].values

    y_test = test_df[
        "anomaly"
    ].astype(int).values

    scores = get_scores(
        model,
        X_test,
    )

    print(
        f"Score range: "
        f"{scores.min():.6f} → "
        f"{scores.max():.6f}"
    )

    print("\n[4/5] Testing thresholds...")

    threshold_results = (
        evaluate_thresholds(
            y_test,
            scores,
        )
    )

    threshold_results = sorted(
        threshold_results,
        key=lambda x: x["f1_score"],
        reverse=True,
    )

    top_results = (
        threshold_results[:10]
    )

    print("\nTOP 10 THRESHOLDS")
    print("-" * 80)

    for i, result in enumerate(
        top_results,
        1,
    ):

        print(
            f"{i:2d}. "
            f"threshold={result['threshold']:.6f} | "
            f"precision={result['precision']:.4f} | "
            f"recall={result['recall']:.4f} | "
            f"F1={result['f1_score']:.4f}"
        )

    best = top_results[0]

    print("\nBEST TEST RESULT")
    print("-" * 50)

    for key, value in best.items():

        if isinstance(value, float):
            print(
                f"{key:25s}: {value:.4f}"
            )
        else:
            print(
                f"{key:25s}: {value}"
            )

    print("\n[5/5] Saving comparison report...")

    report = {
        "model": "IsolationForest",
        "evaluation": "threshold_sweep",
        "test_experiments": test_experiments,
        "best_test_result": best,
        "top_10_thresholds": top_results,
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
        f"\nSaved: {REPORT_PATH}"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()