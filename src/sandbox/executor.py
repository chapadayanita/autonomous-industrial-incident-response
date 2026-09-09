from typing import Dict, Any, List
from datetime import datetime, timezone


class SandboxExecutor:
    """
    Controlled execution layer for autonomous incident-response actions.

    The executor only allows explicitly approved actions.
    No real industrial equipment or external systems are modified.
    """

    def __init__(self):

        self.agent_name = "sandbox_executor"

        # Explicit allow-list of safe operational actions.
        self.allowed_actions = {
            "continued_monitoring": {
                "description": (
                    "Continue monitoring the affected equipment."
                ),
                "risk_level": "low",
            },
            "enhanced_monitoring": {
                "description": (
                    "Increase monitoring frequency for the affected equipment."
                ),
                "risk_level": "low",
            },
            "request_maintenance_inspection": {
                "description": (
                    "Request a physical maintenance inspection."
                ),
                "risk_level": "medium",
            },
            "priority_maintenance_escalation": {
                "description": (
                    "Escalate the incident to maintenance operations "
                    "for priority physical inspection."
                ),
                "risk_level": "medium",
            },
            "collect_additional_diagnostic_evidence": {
                "description": (
                    "Collect additional diagnostic evidence before "
                    "making a maintenance decision."
                ),
                "risk_level": "low",
            },
        }

    # ================================================================
    # ACTION VALIDATION
    # ================================================================

    def validate_action(
        self,
        action: str,
    ) -> Dict[str, Any]:
        """
        Validate whether an action is explicitly allowed.
        """

        if not action:
            return {
                "status": "rejected",
                "allowed": False,
                "reason": "No action was provided.",
            }

        if action not in self.allowed_actions:
            return {
                "status": "rejected",
                "allowed": False,
                "reason": (
                    f"Action '{action}' is not present "
                    "in the sandbox allow-list."
                ),
            }

        action_config = self.allowed_actions[action]

        return {
            "status": "validated",
            "allowed": True,
            "action": action,
            "description": action_config["description"],
            "risk_level": action_config["risk_level"],
        }

    # ================================================================
    # ACTION EXECUTION
    # ================================================================

    def execute_action(
        self,
        action: str,
        incident_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Validate and safely execute an approved action.

        Execution is simulated. No real-world system is modified.
        """

        validation = self.validate_action(action)

        if not validation["allowed"]:

            return {
                "agent": self.agent_name,
                "status": "action_rejected",
                "action": action,
                "allowed": False,
                "reason": validation["reason"],
                "timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        incident_context = incident_context or {}

        result = {
            "agent": self.agent_name,
            "status": "action_simulated",
            "action": action,
            "allowed": True,
            "description": validation["description"],
            "risk_level": validation["risk_level"],
            "simulation": True,
            "real_system_modified": False,
            "incident_level": incident_context.get(
                "incident_level",
                "unknown",
            ),
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        return result

    # ================================================================
    # FULL INCIDENT ACTION
    # ================================================================

    def execute_from_incident(
        self,
        incident_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert the final incident decision into a controlled
        sandbox action.
        """

        final_decision = incident_result.get(
            "final_decision",
            {},
        )

        decision = final_decision.get(
            "decision",
            "collect_additional_diagnostic_evidence",
        )

        # Map orchestration decisions to explicitly allowed actions.
        decision_to_action = {
            "continued_monitoring": "continued_monitoring",

            "enhanced_monitoring": "enhanced_monitoring",

            "maintenance_escalation": (
                "request_maintenance_inspection"
            ),

            "priority_maintenance_escalation": (
                "priority_maintenance_escalation"
            ),

            "additional_diagnostic_evidence": (
                "collect_additional_diagnostic_evidence"
            ),
        }

        action = decision_to_action.get(
            decision,
            "collect_additional_diagnostic_evidence",
        )

        execution_result = self.execute_action(
            action=action,
            incident_context=incident_result,
        )

        return {
            "incident_decision": decision,
            "mapped_action": action,
            "execution": execution_result,
        }


# ======================================================================
# SINGLETON
# ======================================================================

_sandbox_executor = None


def get_sandbox_executor() -> SandboxExecutor:

    global _sandbox_executor

    if _sandbox_executor is None:
        _sandbox_executor = SandboxExecutor()

    return _sandbox_executor


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def execute_safe_action(
    action: str,
    incident_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:

    executor = get_sandbox_executor()

    return executor.execute_action(
        action=action,
        incident_context=incident_context,
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SANDBOX EXECUTOR TEST")
    print("=" * 70)

    executor = get_sandbox_executor()

    # --------------------------------------------------------------
    # TEST 1 — APPROVED ACTION
    # --------------------------------------------------------------

    print("\nTEST 1 — Approved action")

    approved_result = executor.execute_action(
        action="priority_maintenance_escalation",
        incident_context={
            "incident_level": "critical",
        },
    )

    print(
        f"Status              : "
        f"{approved_result['status']}"
    )

    print(
        f"Action              : "
        f"{approved_result['action']}"
    )

    print(
        f"Allowed             : "
        f"{approved_result['allowed']}"
    )

    print(
        f"Simulation          : "
        f"{approved_result['simulation']}"
    )

    print(
        f"Real system modified: "
        f"{approved_result['real_system_modified']}"
    )

    # --------------------------------------------------------------
    # TEST 2 — REJECTED ACTION
    # --------------------------------------------------------------

    print("\nTEST 2 — Rejected action")

    rejected_result = executor.execute_action(
        action="shutdown_pump",
        incident_context={
            "incident_level": "critical",
        },
    )

    print(
        f"Status              : "
        f"{rejected_result['status']}"
    )

    print(
        f"Action              : "
        f"{rejected_result['action']}"
    )

    print(
        f"Allowed             : "
        f"{rejected_result['allowed']}"
    )

    print(
        f"Reason              : "
        f"{rejected_result['reason']}"
    )

    # --------------------------------------------------------------
    # TEST 3 — INCIDENT DECISION MAPPING
    # --------------------------------------------------------------

    print("\nTEST 3 — Incident decision mapping")

    incident_result = {
        "incident_level": "critical",
        "final_decision": {
            "decision": (
                "priority_maintenance_escalation"
            ),
        },
    }

    mapped_result = executor.execute_from_incident(
        incident_result
    )

    print(
        f"Incident decision   : "
        f"{mapped_result['incident_decision']}"
    )

    print(
        f"Mapped action       : "
        f"{mapped_result['mapped_action']}"
    )

    print(
        f"Execution status    : "
        f"{mapped_result['execution']['status']}"
    )

    print("\n" + "=" * 70)
    print("SANDBOX EXECUTOR TEST COMPLETE")
    print("=" * 70)