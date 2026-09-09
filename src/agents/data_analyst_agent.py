from typing import Dict, Any, List

import numpy as np
import pandas as pd

from src.agents.tools import (
    SENSOR_COLUMNS,
    load_processed_data,
    run_ml_anomaly_detection,
)


class DataAnalystAgent:
    """
    Data Analyst Agent.

    Analyzes one supplied sensor-data window using:
    1. Supervised ML anomaly detection
    2. Statistical z-score analysis

    Both analyses operate on the SAME input window.
    """

    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold

    # ================================================================
    # INPUT VALIDATION
    # ================================================================

    def _validate_input(self, sensor_data: pd.DataFrame) -> None:

        if sensor_data is None:
            raise ValueError("sensor_data cannot be None")

        if not isinstance(sensor_data, pd.DataFrame):
            raise TypeError(
                "sensor_data must be a pandas DataFrame"
            )

        if sensor_data.empty:
            raise ValueError(
                "sensor_data cannot be empty"
            )

        missing_sensors = [
            sensor
            for sensor in SENSOR_COLUMNS
            if sensor not in sensor_data.columns
        ]

        if missing_sensors:
            raise ValueError(
                f"Missing required sensor columns: "
                f"{missing_sensors}"
            )

    # ================================================================
    # SENSOR LIST
    # ================================================================

    def _get_available_sensors(
        self,
        sensor_data: pd.DataFrame,
    ) -> List[str]:

        return [
            sensor
            for sensor in SENSOR_COLUMNS
            if sensor in sensor_data.columns
        ]

    # ================================================================
    # STATISTICAL ANALYSIS
    # ================================================================

    def _run_statistical_analysis(
        self,
        sensor_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run z-score analysis ONLY on the supplied window.
        """

        available_sensors = self._get_available_sensors(
            sensor_data
        )

        sensor_results = []

        for sensor in available_sensors:

            values = pd.to_numeric(
                sensor_data[sensor],
                errors="coerce",
            ).dropna()

            if len(values) == 0:

                sensor_results.append(
                    {
                        "sensor": sensor,
                        "anomaly_count": 0,
                        "anomaly_percentage": 0.0,
                        "severity": "normal",
                        "method": "z_score",
                        "max_abs_z": 0.0,
                    }
                )

                continue

            mean_value = values.mean()
            std_value = values.std()

            if pd.isna(std_value) or std_value == 0:

                anomaly_count = 0
                max_abs_z = 0.0

            else:

                z_scores = (
                    (values - mean_value)
                    / std_value
                )

                anomaly_mask = (
                    np.abs(z_scores)
                    >= self.zscore_threshold
                )

                anomaly_count = int(
                    anomaly_mask.sum()
                )

                max_abs_z = float(
                    np.abs(z_scores).max()
                )

            anomaly_percentage = (
                anomaly_count / len(values)
            ) * 100

            if anomaly_percentage >= 10:

                severity = "critical"

            elif anomaly_percentage >= 5:

                severity = "high"

            elif anomaly_percentage > 0:

                severity = "medium"

            else:

                severity = "normal"

            sensor_results.append(
                {
                    "sensor": sensor,
                    "anomaly_count": anomaly_count,
                    "anomaly_percentage": round(
                        anomaly_percentage,
                        4,
                    ),
                    "severity": severity,
                    "method": "z_score",
                    "max_abs_z": round(
                        max_abs_z,
                        4,
                    ),
                }
            )

        # Rank sensors by anomaly count
        sensor_results.sort(
            key=lambda x: x["anomaly_count"],
            reverse=True,
        )

        affected_sensors = [
            result
            for result in sensor_results
            if result["anomaly_count"] > 0
        ]

        high_severity_sensors = [
            result
            for result in sensor_results
            if result["severity"]
            in ("high", "critical")
        ]

        if len(high_severity_sensors) >= 2:

            incident_signal = (
                "multi_sensor_high_severity"
            )

        elif len(affected_sensors) >= 2:

            incident_signal = (
                "multi_sensor_anomaly"
            )

        elif len(affected_sensors) == 1:

            incident_signal = (
                "single_sensor_anomaly"
            )

        else:

            incident_signal = "normal"

        return {
            "status": "analysis_complete",
            "incident_signal": incident_signal,
            "affected_sensor_count": len(
                affected_sensors
            ),
            "high_severity_sensor_count": len(
                high_severity_sensors
            ),
            "total_sensors_analyzed": len(
                available_sensors
            ),
            "samples_analyzed": len(sensor_data),
            "top_sensors": sensor_results,
        }

    # ================================================================
    # ML ANALYSIS
    # ================================================================

    def _run_ml_analysis(
        self,
        sensor_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run the supervised ML detector.

        run_ml_anomaly_detection() returns a wrapper containing:
            tool
            model
            result

        We return the actual result so the rest of the
        agent can read anomaly_probability, anomaly_count, etc.
        """

        ml_response = run_ml_anomaly_detection(
            sensor_data
        )

        # Defensive handling of the tool response
        if (
            isinstance(ml_response, dict)
            and "result" in ml_response
            and isinstance(
                ml_response["result"],
                dict,
            )
        ):

            result = ml_response["result"].copy()

            # Preserve useful metadata
            result["tool"] = ml_response.get(
                "tool"
            )

            result["model"] = ml_response.get(
                "model"
            )

            return result

        return ml_response

    # ================================================================
    # INCIDENT LEVEL
    # ================================================================

    def _determine_incident_level(
        self,
        ml_analysis: Dict[str, Any],
        statistical_analysis: Dict[str, Any],
    ) -> str:

        ml_probability = float(
            ml_analysis.get(
                "anomaly_probability",
                0.0,
            )
        )

        ml_detected = bool(
            ml_analysis.get(
                "anomaly_detected",
                False,
            )
        )

        affected_sensor_count = int(
            statistical_analysis.get(
                "affected_sensor_count",
                0,
            )
        )

        high_severity_sensor_count = int(
            statistical_analysis.get(
                "high_severity_sensor_count",
                0,
            )
        )

        # Critical
        if (
            ml_probability >= 0.85
            or high_severity_sensor_count >= 2
        ):
            return "critical"

        # High
        if (
            ml_probability >= 0.65
            or high_severity_sensor_count >= 1
        ):
            return "high"

        # Medium
        if (
            ml_detected
            or affected_sensor_count >= 2
        ):
            return "medium"

        # Low
        if affected_sensor_count == 1:
            return "low"

        return "normal"

    # ================================================================
    # REASON
    # ================================================================

    def _generate_reason(
        self,
        incident_level: str,
        ml_analysis: Dict[str, Any],
        statistical_analysis: Dict[str, Any],
    ) -> str:

        ml_probability = float(
            ml_analysis.get(
                "anomaly_probability",
                0.0,
            )
        )

        anomaly_count = int(
            ml_analysis.get(
                "anomaly_count",
                0,
            )
        )

        samples = int(
            ml_analysis.get(
                "samples_analyzed",
                0,
            )
        )

        affected = int(
            statistical_analysis.get(
                "affected_sensor_count",
                0,
            )
        )

        if incident_level == "critical":

            return (
                "The supervised anomaly detector "
                f"reported a maximum anomaly probability "
                f"of {ml_probability:.4f}. "
                f"{anomaly_count} of {samples} analyzed "
                "samples crossed the trained decision "
                "threshold. Statistical analysis of the "
                f"same analysis window reported "
                f"{affected} affected sensors."
            )

        if incident_level == "high":

            return (
                "The supervised anomaly detector "
                f"reported elevated anomaly probability "
                f"({ml_probability:.4f}), with "
                f"{anomaly_count} of {samples} samples "
                "crossing the trained threshold. "
                "Statistical analysis of the same "
                f"window identified {affected} "
                "affected sensors."
            )

        if incident_level == "medium":

            return (
                "Anomaly evidence was detected in the "
                f"current sensor window. The maximum ML "
                f"probability was {ml_probability:.4f}, "
                f"with {anomaly_count} of {samples} "
                "samples crossing the trained threshold. "
                "Statistical analysis identified "
                f"{affected} affected sensors."
            )

        if incident_level == "low":

            return (
                "Limited anomaly evidence was detected. "
                "The statistical analysis of the current "
                f"window identified {affected} affected "
                "sensor."
            )

        return (
            "No significant anomaly evidence was "
            "detected in the supplied sensor-data window."
        )

    # ================================================================
    # RECOMMENDATION
    # ================================================================

    def _generate_recommendation(
        self,
        incident_level: str,
    ) -> str:

        recommendations = {

            "critical": (
                "Immediately escalate the incident "
                "for operational assessment and request "
                "additional diagnostic evidence."
            ),

            "high": (
                "Escalate for operational assessment "
                "and request additional diagnostic "
                "evidence while monitoring the affected "
                "sensors."
            ),

            "medium": (
                "Continue focused monitoring of the "
                "affected sensors and request additional "
                "diagnostic evidence if the anomaly "
                "persists."
            ),

            "low": (
                "Continue monitoring the affected sensor "
                "and investigate if the anomaly persists."
            ),

            "normal": (
                "No immediate operational action is "
                "required. Continue routine monitoring."
            ),
        }

        return recommendations.get(
            incident_level,
            recommendations["normal"],
        )

    # ================================================================
    # MAIN ANALYSIS
    # ================================================================

    def analyze(
        self,
        sensor_data: pd.DataFrame,
    ) -> Dict[str, Any]:

        self._validate_input(sensor_data)

        # Defensive copy
        analysis_window = sensor_data.copy()

        # SAME 500-row window goes to both systems
        ml_analysis = self._run_ml_analysis(
            analysis_window
        )

        statistical_analysis = (
            self._run_statistical_analysis(
                analysis_window
            )
        )

        incident_level = (
            self._determine_incident_level(
                ml_analysis,
                statistical_analysis,
            )
        )

        reason = self._generate_reason(
            incident_level,
            ml_analysis,
            statistical_analysis,
        )

        recommendation = (
            self._generate_recommendation(
                incident_level
            )
        )

        return {
            "agent": "data_analyst",
            "status": "analysis_complete",
            "samples_analyzed": len(
                analysis_window
            ),
            "incident_level": incident_level,
            "reason": reason,
            "recommendation": recommendation,
            "ml_analysis": ml_analysis,
            "statistical_analysis": (
                statistical_analysis
            ),
        }


# ======================================================================
# SINGLETON
# ======================================================================

_data_analyst_agent = None


def get_data_analyst_agent() -> DataAnalystAgent:

    global _data_analyst_agent

    if _data_analyst_agent is None:
        _data_analyst_agent = DataAnalystAgent()

    return _data_analyst_agent


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def analyze_sensor_data(
    sensor_data: pd.DataFrame,
) -> Dict[str, Any]:

    agent = get_data_analyst_agent()

    return agent.analyze(
        sensor_data
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DATA ANALYST AGENT TEST")
    print("=" * 70)

    print("\nLoading sensor data...")

    df = load_processed_data()

    # One fixed analysis window.
    # ML + statistics both use these exact 500 rows.
    analysis_window = df.tail(500).copy()

    print(
        f"\nAnalyzing {len(analysis_window)} samples..."
    )

    agent = get_data_analyst_agent()

    result = agent.analyze(
        analysis_window
    )

    # --------------------------------------------------------------
    # ASSESSMENT
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGENT ASSESSMENT")
    print("=" * 70)

    print(
        f"Incident level : "
        f"{result['incident_level']}"
    )

    print(
        f"Reason         : "
        f"{result['reason']}"
    )

    print(
        f"Recommendation : "
        f"{result['recommendation']}"
    )

    # --------------------------------------------------------------
    # ML
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML ANALYSIS")
    print("=" * 70)

    for key, value in result[
        "ml_analysis"
    ].items():

        print(
            f"{key:<25}: {value}"
        )

    # --------------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATISTICAL INCIDENT SIGNAL")
    print("=" * 70)

    statistical = result[
        "statistical_analysis"
    ]

    for key, value in statistical.items():

        if key != "top_sensors":

            print(
                f"{key:<30}: {value}"
            )

    print("\nTop sensors:")

    for sensor in statistical[
        "top_sensors"
    ][:5]:

        print(
            f"  {sensor}"
        )

    print("\n" + "=" * 70)
    print("DATA ANALYST AGENT TEST COMPLETE")
    print("=" * 70)