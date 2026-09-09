from pathlib import Path
from typing import Dict, List
from src.models.baseline import create_features

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "skab_processed.parquet"
)


# ============================================================
# SENSOR DEFINITIONS
# ============================================================

SENSOR_COLUMNS: List[str] = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


# ============================================================
# TOOL METADATA
# ============================================================

TOOL_VERSION = "1.0.0"


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_data() -> pd.DataFrame:
    """
    Load the validated and preprocessed SKAB dataset.
    """

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: "
            f"{PROCESSED_DATA_PATH}"
        )

    df = pd.read_parquet(
        PROCESSED_DATA_PATH
    )

    return df


# ============================================================
# VALIDATE SENSOR
# ============================================================

def validate_sensor(sensor: str) -> None:
    """
    Validate that a requested sensor exists.
    """

    if sensor not in SENSOR_COLUMNS:
        raise ValueError(
            f"Unknown sensor '{sensor}'. "
            f"Available sensors: {SENSOR_COLUMNS}"
        )


# ============================================================
# GET SENSOR STATISTICS
# ============================================================

def get_sensor_statistics(
    sensor: str
) -> Dict:

    validate_sensor(sensor)

    df = load_processed_data()

    series = (
        df[sensor]
        .dropna()
        .astype(float)
    )

    if series.empty:
        raise ValueError(
            f"No valid data available for sensor: {sensor}"
        )

    return {
        "tool": "get_sensor_statistics",
        "tool_version": TOOL_VERSION,
        "sensor": sensor,
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


# ============================================================
# Z-SCORE ANOMALY DETECTION
# ============================================================

def detect_zscore_anomalies(
    sensor: str,
    threshold: float = 3.0
) -> Dict:

    validate_sensor(sensor)

    if threshold <= 0:
        raise ValueError(
            "Z-score threshold must be greater than zero."
        )

    df = load_processed_data()

    series = (
        df[sensor]
        .astype(float)
    )

    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):

        return {
            "tool": "detect_zscore_anomalies",
            "tool_version": TOOL_VERSION,
            "sensor": sensor,
            "method": "z_score",
            "threshold": threshold,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
            "status": "insufficient_variation",
            "message": (
                "Standard deviation is zero or undefined. "
                "Z-score detection cannot be applied reliably."
            ),
        }

    z_scores = (
        (series - mean) / std
    ).abs()

    anomaly_mask = (
        z_scores > threshold
    )

    anomaly_count = int(
        anomaly_mask.sum()
    )

    total_count = len(series)

    anomaly_rate = (
        anomaly_count / total_count
        if total_count > 0
        else 0.0
    )

    max_z_score = float(
        z_scores.max()
    )

    severity = classify_anomaly_severity(
        anomaly_rate
    )

    return {
        "tool": "detect_zscore_anomalies",
        "tool_version": TOOL_VERSION,
        "sensor": sensor,
        "method": "z_score",
        "threshold": threshold,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(
            anomaly_rate,
            6
        ),
        "anomaly_percentage": round(
            anomaly_rate * 100,
            4
        ),
        "mean": float(mean),
        "std": float(std),
        "max_absolute_z_score": round(
            max_z_score,
            4
        ),
        "severity": severity,
        "status": (
            "anomaly_detected"
            if anomaly_count > 0
            else "normal"
        ),
    }


# ============================================================
# IQR ANOMALY DETECTION
# ============================================================

def detect_iqr_anomalies(
    sensor: str,
    multiplier: float = 1.5
) -> Dict:

    validate_sensor(sensor)

    if multiplier <= 0:
        raise ValueError(
            "IQR multiplier must be greater than zero."
        )

    df = load_processed_data()

    series = (
        df[sensor]
        .astype(float)
    )

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0 or np.isnan(iqr):

        return {
            "tool": "detect_iqr_anomalies",
            "tool_version": TOOL_VERSION,
            "sensor": sensor,
            "method": "iqr",
            "multiplier": multiplier,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": None,
            "upper_bound": None,
            "anomaly_count": 0,
            "anomaly_rate": 0.0,
            "anomaly_percentage": 0.0,
            "severity": "unknown",
            "status": "insufficient_variation_for_iqr",
            "message": (
                "IQR is zero because Q1 and Q3 are equal. "
                "IQR detection is not reliable for this sensor."
            ),
        }

    lower_bound = (
        q1 - multiplier * iqr
    )

    upper_bound = (
        q3 + multiplier * iqr
    )

    anomaly_mask = (
        (series < lower_bound)
        | (series > upper_bound)
    )

    anomaly_count = int(
        anomaly_mask.sum()
    )

    total_count = len(series)

    anomaly_rate = (
        anomaly_count / total_count
        if total_count > 0
        else 0.0
    )

    severity = classify_anomaly_severity(
        anomaly_rate
    )

    return {
        "tool": "detect_iqr_anomalies",
        "tool_version": TOOL_VERSION,
        "sensor": sensor,
        "method": "iqr",
        "multiplier": multiplier,
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(
            anomaly_rate,
            6
        ),
        "anomaly_percentage": round(
            anomaly_rate * 100,
            4
        ),
        "severity": severity,
        "status": (
            "anomaly_detected"
            if anomaly_count > 0
            else "normal"
        ),
    }


# ============================================================
# ROLLING Z-SCORE ANOMALY DETECTION
# ============================================================

def detect_rolling_anomalies(
    sensor: str,
    window: int = 60,
    threshold: float = 3.0
) -> Dict:

    validate_sensor(sensor)

    if window < 5:
        raise ValueError(
            "Rolling window must be at least 5."
        )

    if threshold <= 0:
        raise ValueError(
            "Rolling threshold must be greater than zero."
        )

    df = load_processed_data()

    series = (
        df[sensor]
        .astype(float)
    )

    rolling_mean = (
        series
        .rolling(
            window=window,
            min_periods=max(
                5,
                window // 2
            )
        )
        .mean()
    )

    rolling_std = (
        series
        .rolling(
            window=window,
            min_periods=max(
                5,
                window // 2
            )
        )
        .std()
    )

    rolling_std = (
        rolling_std
        .replace(0, np.nan)
    )

    rolling_z = (
        (series - rolling_mean)
        / rolling_std
    ).abs()

    anomaly_mask = (
        rolling_z > threshold
    )

    anomaly_mask = (
        anomaly_mask
        .fillna(False)
    )

    anomaly_count = int(
        anomaly_mask.sum()
    )

    total_count = len(series)

    anomaly_rate = (
        anomaly_count / total_count
        if total_count > 0
        else 0.0
    )

    severity = classify_anomaly_severity(
        anomaly_rate
    )

    return {
        "tool": "detect_rolling_anomalies",
        "tool_version": TOOL_VERSION,
        "sensor": sensor,
        "method": "rolling_z_score",
        "window": window,
        "threshold": threshold,
        "anomaly_count": anomaly_count,
        "anomaly_rate": round(
            anomaly_rate,
            6
        ),
        "anomaly_percentage": round(
            anomaly_rate * 100,
            4
        ),
        "severity": severity,
        "status": (
            "anomaly_detected"
            if anomaly_count > 0
            else "normal"
        ),
    }


# ============================================================
# ANOMALY SEVERITY CLASSIFICATION
# ============================================================

def classify_anomaly_severity(
    anomaly_rate: float
) -> str:

    if anomaly_rate <= 0:
        return "normal"

    if anomaly_rate < 0.01:
        return "low"

    if anomaly_rate < 0.05:
        return "medium"

    return "high"


# ============================================================
# MULTI-SENSOR ANALYSIS
# ============================================================

def analyze_all_sensors(
    method: str = "z_score",
    threshold: float = 3.0
) -> List[Dict]:

    if method not in {
        "z_score",
        "iqr",
        "rolling",
    }:

        raise ValueError(
            "Unsupported method. "
            "Use 'z_score', 'iqr', or 'rolling'."
        )

    results = []

    for sensor in SENSOR_COLUMNS:

        if method == "z_score":

            result = detect_zscore_anomalies(
                sensor=sensor,
                threshold=threshold
            )

        elif method == "iqr":

            result = detect_iqr_anomalies(
                sensor=sensor
            )

        else:

            result = detect_rolling_anomalies(
                sensor=sensor,
                threshold=threshold
            )

        results.append(result)

    return results


# ============================================================
# RANK SENSORS
# ============================================================

def rank_sensors_by_anomaly_rate(
    results: List[Dict]
) -> List[Dict]:

    return sorted(
        results,
        key=lambda result: result.get(
            "anomaly_rate",
            0.0
        ),
        reverse=True
    )


# ============================================================
# GET TOP ANOMALOUS SENSORS
# ============================================================

def get_top_anomalous_sensors(
    results: List[Dict],
    top_k: int = 3
) -> List[Dict]:

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than zero."
        )

    ranked = rank_sensors_by_anomaly_rate(
        results
    )

    return ranked[:top_k]


# ============================================================
# BUILD CROSS-SENSOR SUMMARY
# ============================================================

def build_incident_signal(
    results: List[Dict]
) -> Dict:

    if not results:

        return {
            "status": "no_results",
            "incident_signal": "unknown",
            "affected_sensor_count": 0,
            "top_sensors": [],
        }

    ranked = rank_sensors_by_anomaly_rate(
        results
    )

    affected = [
        result
        for result in ranked
        if result.get(
            "anomaly_count",
            0
        ) > 0
    ]

    high_severity = [
        result
        for result in affected
        if result.get(
            "severity"
        ) == "high"
    ]

    if len(high_severity) >= 2:

        incident_signal = (
            "multi_sensor_high_risk"
        )

    elif len(affected) >= 2:

        incident_signal = (
            "multi_sensor_anomaly"
        )

    elif len(affected) == 1:

        incident_signal = (
            "single_sensor_anomaly"
        )

    else:

        incident_signal = (
            "no_significant_anomaly"
        )

    return {
        "status": "analysis_complete",
        "incident_signal": incident_signal,
        "affected_sensor_count": len(
            affected
        ),
        "high_severity_sensor_count": len(
            high_severity
        ),
        "total_sensors_analyzed": len(
            results
        ),
        "top_sensors": [
            {
                "sensor": result["sensor"],
                "anomaly_count": result.get(
                    "anomaly_count",
                    0
                ),
                "anomaly_percentage": result.get(
                    "anomaly_percentage",
                    0.0
                ),
                "severity": result.get(
                    "severity",
                    "unknown"
                ),
                "method": result.get(
                    "method",
                    "unknown"
                ),
            }
            for result in ranked[:5]
        ],
    }


# ============================================================
# GROUND-TRUTH EVALUATION
# ============================================================

def evaluate_against_ground_truth(
    sensor: str,
    threshold: float = 3.0
) -> Dict:

    validate_sensor(sensor)

    df = load_processed_data()

    if "anomaly" not in df.columns:
        raise ValueError(
            "Ground-truth 'anomaly' column is not available."
        )

    series = (
        df[sensor]
        .astype(float)
    )

    mean = series.mean()
    std = series.std()

    if std == 0 or np.isnan(std):
        raise ValueError(
            f"Cannot evaluate {sensor}: "
            "standard deviation is zero or undefined."
        )

    predicted = (
        (
            (series - mean) / std
        ).abs()
        > threshold
    )

    actual = (
        df["anomaly"]
        .astype(float)
        > 0
    )

    true_positive = int(
        (predicted & actual).sum()
    )

    true_negative = int(
        (~predicted & ~actual).sum()
    )

    false_positive = int(
        (predicted & ~actual).sum()
    )

    false_negative = int(
        (~predicted & actual).sum()
    )

    precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "tool": "evaluate_against_ground_truth",
        "sensor": sensor,
        "method": "z_score",
        "threshold": threshold,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(
            precision,
            4
        ),
        "recall": round(
            recall,
            4
        ),
        "f1_score": round(
            f1,
            4
        ),
    }


# ============================================================
# CREATE HUMAN-READABLE ANALYSIS SUMMARY
# ============================================================

def create_analysis_summary(
    results: List[Dict]
) -> str:

    if not results:
        return (
            "No sensor analysis results available."
        )

    ranked = rank_sensors_by_anomaly_rate(
        results
    )

    lines = [
        "Sensor anomaly analysis completed."
    ]

    for result in ranked:

        sensor = result.get(
            "sensor",
            "unknown"
        )

        count = result.get(
            "anomaly_count",
            0
        )

        percentage = result.get(
            "anomaly_percentage",
            0.0
        )

        severity = result.get(
            "severity",
            "unknown"
        )

        method = result.get(
            "method",
            "unknown"
        )

        lines.append(
            f"- {sensor}: "
            f"{count} anomalies "
            f"({percentage:.2f}%), "
            f"severity={severity}, "
            f"method={method}"
        )

    incident_signal = build_incident_signal(
        results
    )

    lines.append(
        f"Incident signal: "
        f"{incident_signal['incident_signal']}"
    )

    return "\n".join(lines)


# ============================================================
# FULL DATA ANALYSIS TOOL
# ============================================================

def run_sensor_analysis(
    method: str = "z_score",
    threshold: float = 3.0
) -> Dict:

    results = analyze_all_sensors(
        method=method,
        threshold=threshold
    )

    incident_signal = build_incident_signal(
        results
    )

    return {
        "status": "success",
        "method": method,
        "threshold": threshold,
        "sensor_count": len(
            SENSOR_COLUMNS
        ),
        "results": results,
        "incident_signal": incident_signal,
        "summary": create_analysis_summary(
            results
        ),
    }


# ============================================================
# SUPERVISED ANOMALY MODEL TOOL
# ============================================================

from src.agents.model_tools import (
    AnomalyModelTool
)

_anomaly_model = None


def get_anomaly_model():

    global _anomaly_model

    if _anomaly_model is None:
        _anomaly_model = AnomalyModelTool()

    return _anomaly_model


def run_ml_anomaly_detection(
    sensor_data: pd.DataFrame
) -> Dict:
    """
    Run the trained supervised ensemble model.

    The model expects the same engineered features
    used during training.
    """

    if (
        sensor_data is None
        or sensor_data.empty
    ):

        return {
            "tool": "supervised_anomaly_detector",
            "status": "no_data",
            "anomaly_detected": False,
            "anomaly_probability": 0.0,
            "severity": "normal",
        }

    # --------------------------------------------------------
    # IMPORTANT:
    # The trained model uses the 88 engineered features.
    # Raw sensor columns alone are not sufficient.
    # --------------------------------------------------------

    featured_data, _ = create_features(
        sensor_data.copy()
    )

    model = get_anomaly_model()

    result = model.predict(
        featured_data
    )

    return {
        "tool": "supervised_anomaly_detector",
        "model": (
            "ExtraTrees + RandomForest "
            "Soft Voting Ensemble"
        ),
        "result": result,
    }


def run_latest_ml_detection(
    sensor_data: pd.DataFrame
) -> Dict:
    """
    Analyze the latest sensor observation.

    Feature engineering is performed before
    sending the data to the trained model.
    """

    if (
        sensor_data is None
        or sensor_data.empty
    ):

        return {
            "tool": "supervised_anomaly_detector",
            "status": "no_data",
            "anomaly_detected": False,
            "anomaly_probability": 0.0,
            "severity": "normal",
        }

    featured_data, _ = create_features(
        sensor_data.copy()
    )

    latest = featured_data.tail(1)

    model = get_anomaly_model()

    result = model.predict(
        latest
    )

    return {
        "tool": "supervised_anomaly_detector",
        "model": (
            "ExtraTrees + RandomForest "
            "Soft Voting Ensemble"
        ),
        "result": result,
    }


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "AGENT TOOLS - SENSOR ANOMALY ANALYSIS"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading processed dataset..."
    )

    df = load_processed_data()

    print(
        f"Dataset rows   : {len(df):,}"
    )

    print(
        f"Dataset columns: {len(df.columns)}"
    )

    print(
        f"Sensors        : {len(SENSOR_COLUMNS)}"
    )

    # --------------------------------------------------------
    # Sensor statistics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PRESSURE SENSOR STATISTICS"
    )
    print("=" * 70)

    statistics = get_sensor_statistics(
        "Pressure"
    )

    for key, value in statistics.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Z-score analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PRESSURE Z-SCORE ANALYSIS"
    )
    print("=" * 70)

    zscore_result = (
        detect_zscore_anomalies(
            sensor="Pressure",
            threshold=3.0
        )
    )

    for key, value in zscore_result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # IQR analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PRESSURE IQR ANALYSIS"
    )
    print("=" * 70)

    iqr_result = (
        detect_iqr_anomalies(
            sensor="Pressure",
            multiplier=1.5
        )
    )

    for key, value in iqr_result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Rolling analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "PRESSURE ROLLING ANOMALY ANALYSIS"
    )
    print("=" * 70)

    rolling_result = (
        detect_rolling_anomalies(
            sensor="Pressure",
            window=60,
            threshold=3.0
        )
    )

    for key, value in rolling_result.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # All sensor analysis
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "MULTI-SENSOR Z-SCORE ANALYSIS"
    )
    print("=" * 70)

    all_results = (
        analyze_all_sensors(
            method="z_score",
            threshold=3.0
        )
    )

    print(
        create_analysis_summary(
            all_results
        )
    )

    # --------------------------------------------------------
    # Incident signal
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "INCIDENT SIGNAL"
    )
    print("=" * 70)

    incident_signal = (
        build_incident_signal(
            all_results
        )
    )

    for key, value in incident_signal.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Ground-truth evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "GROUND-TRUTH EVALUATION"
    )
    print("=" * 70)

    evaluation = (
        evaluate_against_ground_truth(
            sensor="Pressure",
            threshold=3.0
        )
    )

    for key, value in evaluation.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # High-level tool
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "HIGH-LEVEL SENSOR ANALYSIS"
    )
    print("=" * 70)

    analysis = run_sensor_analysis(
        method="z_score",
        threshold=3.0
    )

    print(
        analysis["summary"]
    )

    # --------------------------------------------------------
    # Supervised ML detector
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "SUPERVISED ML ANOMALY DETECTOR"
    )
    print("=" * 70)

    try:

        ml_result = (
            run_ml_anomaly_detection(
                df
            )
        )

        if "result" in ml_result:

            for key, value in (
                ml_result["result"].items()
            ):

                print(
                    f"{key:25s}: {value}"
                )

        else:

            for key, value in (
                ml_result.items()
            ):

                print(
                    f"{key:25s}: {value}"
                )

    except Exception as e:

        print(
            f"ML detector error: {e}"
        )

    print("\n" + "=" * 70)
    print(
        "AGENT TOOLS TEST COMPLETE"
    )
    print("=" * 70)