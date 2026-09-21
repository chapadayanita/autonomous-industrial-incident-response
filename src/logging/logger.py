from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class IncidentLogger:
    """
    Persistent structured logger for autonomous incident-response runs.

    Each completed incident is stored as a separate JSON file.
    """

    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_incident(self, incident_result: Dict[str, Any]) -> str:
        """
        Save a complete incident result as a JSON file.

        Returns:
            Path of the created log file.
        """

        timestamp = datetime.now(timezone.utc)

        incident_id = timestamp.strftime(
            "incident_%Y%m%d_%H%M%S_%f"
        )

        log_data = {
            "incident_id": incident_id,
            "logged_at": timestamp.isoformat(),
            "incident_level": incident_result.get(
                "incident_level",
                "unknown",
            ),
            "samples_analyzed": incident_result.get(
                "samples_analyzed",
                0,
            ),
            "final_decision": incident_result.get(
                "final_decision",
                {},
            ),
            "drone_commander": incident_result.get(
                "drone_commander",
            ),
            "drone_inspection": incident_result.get(
                "drone_inspection",
            ),
            "rag": incident_result.get(
                "rag",
            ),
            "escalation": incident_result.get(
                "escalation",
            ),
            "trace": incident_result.get(
                "trace",
                [],
            ),
            "trace_length": incident_result.get(
                "trace_length",
                0,
            ),
        }

        log_path = self.log_dir / f"{incident_id}.json"

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                log_data,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

        return str(log_path)


def get_incident_logger() -> IncidentLogger:
    """
    Return a new IncidentLogger instance.
    """
    return IncidentLogger()


def log_incident(
    incident_result: Dict[str, Any],
) -> str:
    """
    Convenience function for saving an incident.
    """
    logger = get_incident_logger()

    return logger.log_incident(
        incident_result
    )


if __name__ == "__main__":
    print("=" * 70)
    print("INCIDENT LOGGER TEST")
    print("=" * 70)

    test_result = {
        "incident_level": "critical",
        "samples_analyzed": 500,
        "final_decision": {
            "decision": "priority_maintenance_escalation",
            "priority": "urgent",
            "action": (
                "Immediately escalate the incident "
                "to maintenance operations."
            ),
        },
        "trace": [
            {
                "step": 1,
                "agent": "orchestrator",
                "action": "delegate_sensor_analysis",
                "observation": (
                    "Sending sensor window "
                    "to Data Analyst Agent."
                ),
            }
        ],
        "trace_length": 1,
    }

    path = log_incident(test_result)

    print("\nIncident log created:")
    print(path)

    print("\nLogger test complete.")