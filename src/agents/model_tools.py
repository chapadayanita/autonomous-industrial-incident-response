# src/agents/model_tools.py

from pathlib import Path
from typing import Dict, Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.streaming_features import (
    StreamingFeatureBuilder,
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "artifacts"
    / "supervised_detector.joblib"
)


# ============================================================
# ANOMALY MODEL TOOL
# ============================================================

class AnomalyModelTool:
    """
    Production-facing wrapper around the trained supervised
    anomaly detection model.

    Model:
        ExtraTrees + RandomForest Soft Voting Ensemble

    Expected features:
        88 engineered features

    The model artifact itself is never modified.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
    ):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        artifact = joblib.load(
            self.model_path
        )

        self.model = artifact[
            "model"
        ]

        self.feature_columns = list(
            artifact[
                "feature_columns"
            ]
        )

        self.threshold = float(
            artifact[
                "threshold"
            ]
        )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    def _prepare_features(
        self,
        sensor_data: pd.DataFrame,
    ) -> pd.DataFrame:

        if sensor_data is None:
            raise ValueError(
                "sensor_data cannot be None."
            )

        if not isinstance(
            sensor_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "sensor_data must be a "
                "pandas DataFrame."
            )

        if sensor_data.empty:
            raise ValueError(
                "sensor_data is empty."
            )

        df = sensor_data.copy()

        # ----------------------------------------------------
        # Convert model features to numeric
        # ----------------------------------------------------

        for column in self.feature_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Replace invalid values
        # ----------------------------------------------------

        df = (
            df.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0.0)
        )

        # ----------------------------------------------------
        # Check all required features
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in self.feature_columns
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                "Missing required features: "
                f"{missing_features[:10]}"
                + (
                    "..."
                    if len(missing_features) > 10
                    else ""
                )
            )

        # ----------------------------------------------------
        # EXACT training feature order
        # ----------------------------------------------------

        return df[
            self.feature_columns
        ].copy()

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        sensor_data: pd.DataFrame,
    ) -> Dict[str, Any]:

        X = self._prepare_features(
            sensor_data
        )

        # ----------------------------------------------------
        # Convert to NumPy because the model was fitted
        # without feature names.
        # ----------------------------------------------------

        X_model = X.to_numpy(
            dtype=float
        )

        probabilities = (
            self.model.predict_proba(
                X_model
            )[:, 1]
        )

        predictions = (
            probabilities
            >= self.threshold
        ).astype(int)

        anomaly_probability = float(
            np.max(
                probabilities
            )
        )

        mean_probability = float(
            np.mean(
                probabilities
            )
        )

        anomaly_detected = bool(
            np.any(
                predictions == 1
            )
        )

        anomaly_count = int(
            predictions.sum()
        )

        samples_analyzed = int(
            len(predictions)
        )

        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        if anomaly_probability >= 0.85:

            severity = "critical"

        elif anomaly_probability >= 0.65:

            severity = "high"

        elif anomaly_probability >= self.threshold:

            severity = "medium"

        elif anomaly_probability >= 0.25:

            severity = "low"

        else:

            severity = "normal"

        return {
            "status": (
                "anomaly_detected"
                if anomaly_detected
                else "normal"
            ),
            "anomaly_detected": (
                anomaly_detected
            ),
            "anomaly_probability": round(
                anomaly_probability,
                4,
            ),
            "mean_probability": round(
                mean_probability,
                4,
            ),
            "threshold": round(
                self.threshold,
                4,
            ),
            "severity": severity,
            "anomaly_count": (
                anomaly_count
            ),
            "samples_analyzed": (
                samples_analyzed
            ),
        }


# ============================================================
# STREAMING ANOMALY DETECTOR
# ============================================================

class StreamingAnomalyDetector:
    """
    Stateful anomaly detector for incoming sensor windows.

    Raw sensor data
        ↓
    Stateful feature history
        ↓
    88 engineered features
        ↓
    trained anomaly model
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        history_size: int = 120,
    ):

        self.model_tool = (
            AnomalyModelTool(
                model_path=model_path
            )
        )

        self.feature_builder = (
            StreamingFeatureBuilder(
                history_size=history_size
            )
        )

    # ========================================================
    # PROCESS WINDOW
    # ========================================================

    def process(
        self,
        sensor_window: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Process one incoming sensor window.
        """

        if (
            sensor_window is None
            or sensor_window.empty
        ):

            return {
                "status": "no_data",
                "anomaly_detected": False,
                "anomaly_probability": 0.0,
                "severity": "normal",
                "anomaly_count": 0,
                "samples_analyzed": 0,
                "buffer_size": (
                    self.feature_builder.buffer_size
                ),
            }

        # ----------------------------------------------------
        # Build state-aware features
        # ----------------------------------------------------

        features = (
            self.feature_builder.update(
                sensor_window
            )
        )

        if features.empty:

            return {
                "status": "warming_up",
                "anomaly_detected": False,
                "anomaly_probability": 0.0,
                "severity": "normal",
                "anomaly_count": 0,
                "samples_analyzed": 0,
                "buffer_size": (
                    self.feature_builder.buffer_size
                ),
            }

        # ----------------------------------------------------
        # Run trained model
        # ----------------------------------------------------

        result = self.model_tool.predict(
            features
        )

        result[
            "buffer_size"
        ] = (
            self.feature_builder.buffer_size
        )

        return result

    # ========================================================
    # RESET STREAM
    # ========================================================

    def reset(self) -> None:

        self.feature_builder.reset()


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def predict_anomaly(
    sensor_data: pd.DataFrame,
) -> Dict[str, Any]:

    tool = AnomalyModelTool()

    return tool.predict(
        sensor_data
    )


# ============================================================
# LATEST PREDICTION
# ============================================================

def predict_latest(
    sensor_data: pd.DataFrame,
) -> Dict[str, Any]:

    if (
        sensor_data is None
        or sensor_data.empty
    ):

        return {
            "status": "no_data",
            "anomaly_detected": False,
            "anomaly_probability": 0.0,
            "mean_probability": 0.0,
            "threshold": 0.0,
            "severity": "normal",
            "anomaly_count": 0,
            "samples_analyzed": 0,
        }

    latest = sensor_data.tail(
        1
    )

    return predict_anomaly(
        latest
    )


# ============================================================
# LOCAL BATCH TEST
# ============================================================

if __name__ == "__main__":

    from src.models.baseline import (
        load_data,
        create_features,
    )

    print("=" * 70)
    print(
        "ANOMALY MODEL TOOL TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    df = load_data()

    print(
        f"Raw rows: {len(df):,}"
    )

    # --------------------------------------------------------
    # Create training-compatible features
    # --------------------------------------------------------

    print(
        "\nCreating engineered features..."
    )

    df, feature_columns = (
        create_features(df)
    )

    print(
        f"Engineered features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading trained model..."
    )

    tool = AnomalyModelTool()

    print(
        f"Model path: "
        f"{tool.model_path}"
    )

    print(
        f"Expected features: "
        f"{len(tool.feature_columns)}"
    )

    print(
        f"Threshold: "
        f"{tool.threshold:.4f}"
    )

    # --------------------------------------------------------
    # Batch prediction
    # --------------------------------------------------------

    sample = df.tail(
        100
    )

    print(
        f"\nTesting on "
        f"{len(sample)} samples..."
    )

    result = tool.predict(
        sample
    )

    print(
        "\nMODEL RESULT"
    )

    print(
        "-" * 50
    )

    for key, value in (
        result.items()
    ):

        print(
            f"{key:25s}: {value}"
        )

    print(
        "\nDONE"
    )