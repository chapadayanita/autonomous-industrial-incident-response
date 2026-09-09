from pathlib import Path
import json
import joblib
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)

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
    ARTIFACT_DIR / "supervised_detector.joblib"
)

REPORT_PATH = (
    ARTIFACT_DIR / "supervised_detector_report.json"
)


def prepare_matrix(df, feature_columns):

    X = (
        df[feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    return X


def evaluate(
    y_true,
    probabilities,
    threshold,
):

    predictions = (
        probabilities >= threshold
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
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
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
    probabilities,
):

    thresholds = np.linspace(
        0.01,
        0.99,
        500,
    )

    best = None

    for threshold in thresholds:

        metrics = evaluate(
            y_true,
            probabilities,
            threshold,
        )

        if best is None:

            best = metrics
            best["threshold"] = float(
                threshold
            )

        elif (
            metrics["f1_score"]
            > best["f1_score"]
        ):

            best = metrics
            best["threshold"] = float(
                threshold
            )

    return best


def main():

    print("=" * 70)
    print(
        "SUPERVISED ENSEMBLE ANOMALY DETECTOR"
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
        f"Train experiments: "
        f"{train_experiments}"
    )

    print(
        f"Validation experiments: "
        f"{val_experiments}"
    )

    print(
        f"Test experiments: "
        f"{test_experiments}"
    )

    print(
        "\n[4/8] Preparing training data..."
    )

    X_train = prepare_matrix(
        train_df,
        feature_columns,
    )

    y_train = (
        train_df["anomaly"]
        .astype(int)
        .values
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

    X_test = prepare_matrix(
        test_df,
        feature_columns,
    )

    y_test = (
        test_df["anomaly"]
        .astype(int)
        .values
    )

    print(
        f"Training rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Training anomalies: "
        f"{y_train.sum():,}"
    )

    print(
        f"Training normal: "
        f"{(y_train == 0).sum():,}"
    )

    print(
        "\n[5/8] Training ensemble..."
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest = RandomForestClassifier(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model = VotingClassifier(
        estimators=[
            (
                "extra_trees",
                extra_trees,
            ),
            (
                "random_forest",
                random_forest,
            ),
        ],
        voting="soft",
        weights=[2, 1],
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    print(
        "\n[6/8] Validation threshold optimization..."
    )

    val_probabilities = (
        model.predict_proba(
            X_val
        )[:, 1]
    )

    validation_result = (
        find_best_threshold(
            y_val,
            val_probabilities,
        )
    )

    threshold = (
        validation_result["threshold"]
    )

    print(
        f"Best validation threshold: "
        f"{threshold:.4f}"
    )

    print(
        f"Validation precision: "
        f"{validation_result['precision']:.4f}"
    )

    print(
        f"Validation recall: "
        f"{validation_result['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{validation_result['f1_score']:.4f}"
    )

    print(
        f"Validation ROC-AUC: "
        f"{validation_result['roc_auc']:.4f}"
    )

    print(
        "\n[7/8] Final unseen test evaluation..."
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_result = evaluate(
        y_test,
        test_probabilities,
        threshold,
    )

    print(
        "\nFINAL TEST RESULTS"
    )

    print("-" * 60)

    for key, value in (
        test_result.items()
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
        "\n[8/8] Saving artifacts..."
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold,
        },
        MODEL_PATH,
    )

    report = {
        "model": (
            "ExtraTrees + RandomForest "
            "Soft Voting Ensemble"
        ),
        "feature_count": len(
            feature_columns
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
        "threshold": threshold,
        "validation": validation_result,
        "test": test_result,
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
        f"Model saved: {MODEL_PATH}"
    )

    print(
        f"Report saved: {REPORT_PATH}"
    )

    print("\nDONE")


if __name__ == "__main__":
    main()