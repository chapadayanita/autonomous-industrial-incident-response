from typing import Dict, Any


class EscalationAgent:
    """
    Escalation Agent.

    Combines:
    - Data Analyst evidence
    - Drone inspection evidence
    - RAG operational knowledge

    and determines the appropriate operational response.
    """

    def __init__(self):
        self.agent_name = "escalation_agent"

    # ================================================================
    # INPUT VALIDATION
    # ================================================================

    def _validate_inputs(
        self,
        analyst_result: Dict[str, Any],
        drone_result: Dict[str, Any],
        rag_result: Dict[str, Any],
    ) -> None:

        if not isinstance(analyst_result, dict):
            raise TypeError(
                "analyst_result must be a dictionary"
            )

        if not isinstance(drone_result, dict):
            raise TypeError(
                "drone_result must be a dictionary"
            )

        if not isinstance(rag_result, dict):
            raise TypeError(
                "rag_result must be a dictionary"
            )

    # ================================================================
    # EVIDENCE ASSESSMENT
    # ================================================================

    def _assess_evidence(
        self,
        analyst_result: Dict[str, Any],
        drone_result: Dict[str, Any],
        rag_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        ml_analysis = analyst_result.get(
            "ml_analysis",
            {},
        )

        statistical_analysis = analyst_result.get(
            "statistical_analysis",
            {},
        )

        ml_probability = float(
            ml_analysis.get(
                "anomaly_probability",
                0.0,
            )
        )

        anomaly_detected = bool(
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

        visual_anomaly = bool(
            drone_result.get(
                "visual_anomaly_detected",
                False,
            )
        )

        drone_confidence = float(
            drone_result.get(
                "confidence",
                0.0,
            )
        )

        rag_result_count = int(
            rag_result.get(
                "result_count",
                0,
            )
        )

        # ------------------------------------------------------------
        # Evidence score
        # ------------------------------------------------------------

        evidence_score = 0.0

        if anomaly_detected:
            evidence_score += 0.30

        if ml_probability >= 0.85:
            evidence_score += 0.25

        elif ml_probability >= 0.65:
            evidence_score += 0.15

        if affected_sensor_count >= 2:
            evidence_score += 0.15

        elif affected_sensor_count == 1:
            evidence_score += 0.05

        if visual_anomaly:
            evidence_score += 0.20

        if drone_confidence >= 0.80:
            evidence_score += 0.05

        if rag_result_count > 0:
            evidence_score += 0.05

        evidence_score = min(
            evidence_score,
            1.0,
        )

        return {
            "ml_anomaly_detected": anomaly_detected,
            "ml_probability": ml_probability,
            "affected_sensor_count": affected_sensor_count,
            "visual_anomaly_detected": visual_anomaly,
            "drone_confidence": drone_confidence,
            "rag_evidence_available": (
                rag_result_count > 0
            ),
            "evidence_score": round(
                evidence_score,
                4,
            ),
        }

    # ================================================================
    # DECISION
    # ================================================================

    def _make_decision(
        self,
        evidence: Dict[str, Any],
        incident_level: str,
    ) -> str:

        score = evidence["evidence_score"]

        visual_anomaly = evidence[
            "visual_anomaly_detected"
        ]

        ml_probability = evidence[
            "ml_probability"
        ]

        if (
            incident_level == "critical"
            and score >= 0.80
            and visual_anomaly
        ):
            return "priority_maintenance_escalation"

        if (
            incident_level in (
                "critical",
                "high",
            )
            and score >= 0.65
        ):
            return "maintenance_escalation"

        if (
            incident_level == "medium"
            and score >= 0.50
        ):
            return "enhanced_monitoring"

        if (
            incident_level == "low"
            or ml_probability < 0.65
        ):
            return "continued_monitoring"

        return "additional_diagnostic_evidence"

    # ================================================================
    # PRIORITY
    # ================================================================

    def _determine_priority(
        self,
        decision: str,
    ) -> str:

        priorities = {
            "priority_maintenance_escalation": "urgent",
            "maintenance_escalation": "high",
            "enhanced_monitoring": "medium",
            "additional_diagnostic_evidence": "medium",
            "continued_monitoring": "low",
        }

        return priorities.get(
            decision,
            "medium",
        )

    # ================================================================
    # ACTION
    # ================================================================

    def _generate_action(
        self,
        decision: str,
    ) -> str:

        actions = {

            "priority_maintenance_escalation": (
                "Immediately escalate the incident to "
                "maintenance operations and prioritize "
                "physical inspection of the affected "
                "equipment."
            ),

            "maintenance_escalation": (
                "Escalate the incident to maintenance "
                "personnel for operational assessment "
                "and follow the applicable inspection "
                "procedure."
            ),

            "enhanced_monitoring": (
                "Increase monitoring of the affected "
                "equipment and collect additional "
                "diagnostic evidence."
            ),

            "additional_diagnostic_evidence": (
                "Collect additional diagnostic evidence "
                "before making a higher-level operational "
                "decision."
            ),

            "continued_monitoring": (
                "Continue routine monitoring and "
                "reassess if anomaly evidence persists."
            ),
        }

        return actions.get(
            decision,
            actions[
                "additional_diagnostic_evidence"
            ],
        )

    # ================================================================
    # REASON
    # ================================================================

    def _generate_reason(
        self,
        decision: str,
        evidence: Dict[str, Any],
        incident_level: str,
    ) -> str:

        ml_probability = evidence[
            "ml_probability"
        ]

        affected_sensor_count = evidence[
            "affected_sensor_count"
        ]

        visual_anomaly = evidence[
            "visual_anomaly_detected"
        ]

        rag_available = evidence[
            "rag_evidence_available"
        ]

        if decision == "priority_maintenance_escalation":

            return (
                f"Multiple evidence sources support the "
                f"incident: ML anomaly probability is "
                f"{ml_probability:.4f}, "
                f"{affected_sensor_count} sensors show "
                "statistical anomalies, and the drone "
                "inspection detected a visual/mechanical "
                "abnormality. Relevant operational "
                f"knowledge was {'available' if rag_available else 'not available'} "
                "through the knowledge base."
            )

        if decision == "maintenance_escalation":

            return (
                f"The incident is classified as "
                f"{incident_level}. ML and sensor evidence "
                "indicate an operational anomaly requiring "
                "maintenance assessment."
            )

        if decision == "enhanced_monitoring":

            return (
                "The available evidence indicates a "
                "moderate anomaly. Continued monitoring "
                "and additional diagnostic evidence are "
                "appropriate."
            )

        if decision == "continued_monitoring":

            return (
                "Available evidence does not currently "
                "justify maintenance escalation."
            )

        return (
            "Evidence is not sufficiently conclusive "
            "for immediate escalation. Additional "
            "diagnostic evidence is recommended."
        )

    # ================================================================
    # MAIN
    # ================================================================

    def evaluate(
        self,
        analyst_result: Dict[str, Any],
        drone_result: Dict[str, Any],
        rag_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        self._validate_inputs(
            analyst_result,
            drone_result,
            rag_result,
        )

        incident_level = analyst_result.get(
            "incident_level",
            "normal",
        )

        evidence = self._assess_evidence(
            analyst_result,
            drone_result,
            rag_result,
        )

        decision = self._make_decision(
            evidence,
            incident_level,
        )

        priority = self._determine_priority(
            decision
        )

        action = self._generate_action(
            decision
        )

        reason = self._generate_reason(
            decision,
            evidence,
            incident_level,
        )

        return {
            "agent": self.agent_name,
            "status": "escalation_decision_complete",
            "incident_level": incident_level,
            "decision": decision,
            "priority": priority,
            "action": action,
            "reason": reason,
            "evidence": evidence,
            "rag_sources": [
                result.get("source")
                for result in rag_result.get(
                    "results",
                    [],
                )
            ],
        }


# ======================================================================
# SINGLETON
# ======================================================================

_escalation_agent = None


def get_escalation_agent() -> EscalationAgent:

    global _escalation_agent

    if _escalation_agent is None:
        _escalation_agent = EscalationAgent()

    return _escalation_agent


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def evaluate_incident(
    analyst_result: Dict[str, Any],
    drone_result: Dict[str, Any],
    rag_result: Dict[str, Any],
) -> Dict[str, Any]:

    agent = get_escalation_agent()

    return agent.evaluate(
        analyst_result,
        drone_result,
        rag_result,
    )


# ======================================================================
# END-TO-END TEST
# ======================================================================

if __name__ == "__main__":

    from src.agents.data_analyst_agent import (
        get_data_analyst_agent,
    )

    from src.agents.drone_commander_agent import (
        get_drone_commander_agent,
    )

    from src.agents.drone_simulator import (
        get_drone_simulator,
    )

    from src.agents.tools import (
        load_processed_data,
    )

    from src.rag.retriever import (
        get_knowledge_retriever,
    )

    print("=" * 70)
    print("ESCALATION AGENT TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------

    print("\nLoading sensor data...")

    df = load_processed_data()

    analysis_window = df.tail(500).copy()

    print(
        f"Analyzing {len(analysis_window)} samples..."
    )

    # --------------------------------------------------------------
    # DATA ANALYST
    # --------------------------------------------------------------

    print("\nRunning Data Analyst Agent...")

    analyst = get_data_analyst_agent()

    analyst_result = analyst.analyze(
        analysis_window
    )

    print(
        "Incident level:",
        analyst_result["incident_level"],
    )

    # --------------------------------------------------------------
    # DRONE COMMANDER
    # --------------------------------------------------------------

    print(
        "\nRunning Drone Commander Agent..."
    )

    commander = get_drone_commander_agent()

    drone_command = commander.inspect(
        analyst_result
    )

    print(
        "Drone dispatched:",
        drone_command["drone_dispatched"],
    )

    # --------------------------------------------------------------
    # DRONE SIMULATOR
    # --------------------------------------------------------------

    print(
        "\nRunning Drone Simulator..."
    )

    simulator = get_drone_simulator()

    drone_result = simulator.inspect(
        drone_command
    )

    print(
        "Visual anomaly:",
        drone_result[
            "visual_anomaly_detected"
        ],
    )

    print(
        "Drone confidence:",
        drone_result[
            "confidence"
        ],
    )

    # --------------------------------------------------------------
    # RAG
    # --------------------------------------------------------------

    print(
        "\nRunning RAG retrieval..."
    )

    retriever = get_knowledge_retriever()

    query = (
        "pump mechanical vibration "
        "maintenance inspection"
    )

    rag_result = retriever.build_context(
        query=query,
        top_k=3,
    )

    print(
        "RAG results:",
        rag_result["result_count"],
    )

    for result in rag_result["results"]:

        print(
            f"  - {result['source']} "
            f"| {result['title']}"
        )

    # --------------------------------------------------------------
    # ESCALATION
    # --------------------------------------------------------------

    print(
        "\nRunning Escalation Agent..."
    )

    escalation = get_escalation_agent()

    escalation_result = escalation.evaluate(
        analyst_result,
        drone_result,
        rag_result,
    )

    # --------------------------------------------------------------
    # RESULT
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ESCALATION DECISION")
    print("=" * 70)

    for key, value in escalation_result.items():

        print(
            f"{key:<25}: {value}"
        )

    print("\n" + "=" * 70)
    print("ESCALATION AGENT TEST COMPLETE")
    print("=" * 70)