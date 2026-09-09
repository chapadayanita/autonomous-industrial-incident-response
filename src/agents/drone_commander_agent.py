from typing import Dict, Any, List


class DroneCommanderAgent:
    """
    Drone Commander Agent.

    Receives anomaly evidence from the Data Analyst Agent and
    decides whether a simulated drone inspection is required.
    """

    def __init__(self):
        self.agent_name = "drone_commander"

    # ================================================================
    # DETERMINE WHETHER DRONE INSPECTION IS REQUIRED
    # ================================================================

    def _should_dispatch(
        self,
        analyst_result: Dict[str, Any],
    ) -> bool:

        incident_level = analyst_result.get(
            "incident_level",
            "normal",
        )

        ml_analysis = analyst_result.get(
            "ml_analysis",
            {},
        )

        statistical_analysis = analyst_result.get(
            "statistical_analysis",
            {},
        )

        anomaly_detected = ml_analysis.get(
            "anomaly_detected",
            False,
        )

        affected_sensor_count = statistical_analysis.get(
            "affected_sensor_count",
            0,
        )

        if incident_level in (
            "critical",
            "high",
        ):
            return True

        if anomaly_detected:
            return True

        if affected_sensor_count >= 2:
            return True

        return False

    # ================================================================
    # SELECT INSPECTION TARGET
    # ================================================================

    def _select_target(
        self,
        analyst_result: Dict[str, Any],
    ) -> str:

        statistical = analyst_result.get(
            "statistical_analysis",
            {},
        )

        top_sensors = statistical.get(
            "top_sensors",
            [],
        )

        if not top_sensors:
            return "equipment_area"

        affected = [
            sensor["sensor"]
            for sensor in top_sensors
            if sensor.get("anomaly_count", 0) > 0
        ]

        # Accelerometer signals generally indicate
        # mechanical/vibration-related equipment issues.
        if any(
            "Accelerometer" in sensor
            for sensor in affected
        ):
            return "pump_mechanical_area"

        if "Pressure" in affected:
            return "pump_pressure_system"

        if "Temperature" in affected:
            return "pump_thermal_area"

        if "Current" in affected:
            return "pump_electrical_area"

        return "equipment_area"

    # ================================================================
    # INSPECTION CHECKLIST
    # ================================================================

    def _build_inspection_checklist(
        self,
        analyst_result: Dict[str, Any],
    ) -> List[str]:

        statistical = analyst_result.get(
            "statistical_analysis",
            {},
        )

        top_sensors = statistical.get(
            "top_sensors",
            [],
        )

        affected = [
            sensor["sensor"]
            for sensor in top_sensors
            if sensor.get("anomaly_count", 0) > 0
        ]

        checklist = []

        if any(
            "Accelerometer" in sensor
            for sensor in affected
        ):
            checklist.extend(
                [
                    "Check for abnormal mechanical vibration",
                    "Inspect equipment mounting and alignment",
                    "Check visible condition of rotating components",
                ]
            )

        if "Pressure" in affected:
            checklist.append(
                "Inspect pressure system for leakage or blockage"
            )

        if "Temperature" in affected:
            checklist.append(
                "Inspect equipment for abnormal heating"
            )

        if "Current" in affected:
            checklist.append(
                "Inspect electrical connections and equipment load"
            )

        if not checklist:
            checklist.append(
                "Perform general visual inspection of equipment"
            )

        return checklist

    # ================================================================
    # MAIN COMMAND
    # ================================================================

    def inspect(
        self,
        analyst_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(
            analyst_result,
            dict,
        ):
            raise TypeError(
                "analyst_result must be a dictionary"
            )

        should_dispatch = self._should_dispatch(
            analyst_result
        )

        incident_level = analyst_result.get(
            "incident_level",
            "normal",
        )

        if not should_dispatch:

            return {
                "agent": self.agent_name,
                "status": "no_dispatch_required",
                "drone_dispatched": False,
                "incident_level": incident_level,
                "target": None,
                "inspection_checklist": [],
                "reason": (
                    "Available sensor evidence does not "
                    "currently justify a drone inspection."
                ),
            }

        target = self._select_target(
            analyst_result
        )

        checklist = self._build_inspection_checklist(
            analyst_result
        )

        return {
            "agent": self.agent_name,
            "status": "inspection_requested",
            "drone_dispatched": True,
            "incident_level": incident_level,
            "target": target,
            "inspection_checklist": checklist,
            "reason": (
                "Sensor analysis indicates that physical "
                "inspection would provide additional "
                "diagnostic evidence."
            ),
        }


# ======================================================================
# SINGLETON
# ======================================================================

_drone_commander = None


def get_drone_commander_agent() -> DroneCommanderAgent:

    global _drone_commander

    if _drone_commander is None:
        _drone_commander = DroneCommanderAgent()

    return _drone_commander


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def request_drone_inspection(
    analyst_result: Dict[str, Any],
) -> Dict[str, Any]:

    agent = get_drone_commander_agent()

    return agent.inspect(
        analyst_result
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    from src.agents.data_analyst_agent import (
        get_data_analyst_agent,
    )
    from src.agents.tools import (
        load_processed_data,
    )

    print("=" * 70)
    print("DRONE COMMANDER AGENT TEST")
    print("=" * 70)

    print("\nLoading sensor data...")

    df = load_processed_data()

    analysis_window = df.tail(500).copy()

    print(
        f"Analyzing {len(analysis_window)} samples..."
    )

    # --------------------------------------------------------------
    # DATA ANALYST
    # --------------------------------------------------------------

    data_analyst = get_data_analyst_agent()

    analyst_result = data_analyst.analyze(
        analysis_window
    )

    print("\n" + "=" * 70)
    print("DATA ANALYST RESULT")
    print("=" * 70)

    print(
        "Incident level :",
        analyst_result["incident_level"],
    )

    print(
        "Reason         :",
        analyst_result["reason"],
    )

    # --------------------------------------------------------------
    # DRONE COMMANDER
    # --------------------------------------------------------------

    drone = get_drone_commander_agent()

    drone_result = drone.inspect(
        analyst_result
    )

    print("\n" + "=" * 70)
    print("DRONE COMMANDER DECISION")
    print("=" * 70)

    for key, value in drone_result.items():

        print(
            f"{key:<25}: {value}"
        )

    print("\n" + "=" * 70)
    print("DRONE COMMANDER AGENT TEST COMPLETE")
    print("=" * 70)