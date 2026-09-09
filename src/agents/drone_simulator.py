from typing import Dict, Any


class DroneSimulator:
    """
    Simulates a drone inspection.

    This is a deterministic simulation layer for the prototype.
    It does NOT claim to perform real computer vision.
    """

    def __init__(self):
        self.agent_name = "drone_simulator"

    def inspect(
        self,
        drone_command: Dict[str, Any],
    ) -> Dict[str, Any]:

        if not isinstance(drone_command, dict):
            raise TypeError(
                "drone_command must be a dictionary"
            )

        if not drone_command.get(
            "drone_dispatched",
            False,
        ):
            return {
                "agent": self.agent_name,
                "status": "inspection_not_performed",
                "visual_anomaly_detected": False,
                "observations": [],
                "confidence": 0.0,
            }

        target = drone_command.get(
            "target",
            "equipment_area",
        )

        checklist = drone_command.get(
            "inspection_checklist",
            [],
        )

        # ----------------------------------------------------------
        # Deterministic simulated evidence
        # ----------------------------------------------------------

        if target == "pump_mechanical_area":

            observations = [
                "Abnormal mechanical vibration observed",
                "Possible rotating-component irregularity detected",
                "Equipment mounting appears intact",
                "No obvious structural damage observed",
            ]

            finding = (
                "Evidence suggests a possible mechanical "
                "issue requiring maintenance inspection."
            )

            confidence = 0.88

        elif target == "pump_pressure_system":

            observations = [
                "Pressure-system area visually inspected",
                "No obvious structural damage observed",
                "Possible leakage area requires closer inspection",
            ]

            finding = (
                "Possible pressure-system issue detected; "
                "further inspection is recommended."
            )

            confidence = 0.78

        elif target == "pump_thermal_area":

            observations = [
                "Thermal inspection area visually inspected",
                "No obvious structural damage observed",
                "Equipment surface appears operational",
            ]

            finding = (
                "No definitive physical fault confirmed "
                "from the simulated inspection."
            )

            confidence = 0.65

        elif target == "pump_electrical_area":

            observations = [
                "Electrical equipment area inspected",
                "Connections appear visually intact",
                "No obvious physical damage observed",
            ]

            finding = (
                "No definitive electrical fault confirmed "
                "from the simulated inspection."
            )

            confidence = 0.68

        else:

            observations = [
                "General equipment area inspected",
                "No obvious structural damage observed",
            ]

            finding = (
                "General inspection completed; "
                "no definitive physical fault confirmed."
            )

            confidence = 0.60

        return {
            "agent": self.agent_name,
            "status": "inspection_complete",
            "target": target,
            "visual_anomaly_detected": (
                target == "pump_mechanical_area"
            ),
            "observations": observations,
            "finding": finding,
            "confidence": confidence,
            "checklist_completed": checklist,
        }


# ======================================================================
# SINGLETON
# ======================================================================

_drone_simulator = None


def get_drone_simulator() -> DroneSimulator:

    global _drone_simulator

    if _drone_simulator is None:
        _drone_simulator = DroneSimulator()

    return _drone_simulator


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def run_drone_inspection(
    drone_command: Dict[str, Any],
) -> Dict[str, Any]:

    simulator = get_drone_simulator()

    return simulator.inspect(
        drone_command
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    from src.agents.data_analyst_agent import (
        get_data_analyst_agent,
    )
    from src.agents.drone_commander_agent import (
        get_drone_commander_agent,
    )
    from src.agents.tools import (
        load_processed_data,
    )

    print("=" * 70)
    print("DRONE SIMULATOR TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load data
    # --------------------------------------------------------------

    print("\nLoading sensor data...")

    df = load_processed_data()

    analysis_window = df.tail(500).copy()

    print(
        f"Analyzing {len(analysis_window)} samples..."
    )

    # --------------------------------------------------------------
    # Data Analyst
    # --------------------------------------------------------------

    analyst = get_data_analyst_agent()

    analyst_result = analyst.analyze(
        analysis_window
    )

    print("\n" + "=" * 70)
    print("DATA ANALYST")
    print("=" * 70)

    print(
        "Incident level:",
        analyst_result["incident_level"],
    )

    # --------------------------------------------------------------
    # Drone Commander
    # --------------------------------------------------------------

    commander = get_drone_commander_agent()

    drone_command = commander.inspect(
        analyst_result
    )

    print("\n" + "=" * 70)
    print("DRONE COMMANDER")
    print("=" * 70)

    print(
        "Drone dispatched:",
        drone_command["drone_dispatched"],
    )

    print(
        "Target:",
        drone_command["target"],
    )

    # --------------------------------------------------------------
    # Drone Simulator
    # --------------------------------------------------------------

    simulator = get_drone_simulator()

    inspection_result = simulator.inspect(
        drone_command
    )

    print("\n" + "=" * 70)
    print("DRONE INSPECTION RESULT")
    print("=" * 70)

    for key, value in inspection_result.items():

        print(
            f"{key:<30}: {value}"
        )

    print("\n" + "=" * 70)
    print("DRONE SIMULATOR TEST COMPLETE")
    print("=" * 70)