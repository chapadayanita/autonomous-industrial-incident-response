from typing import Dict, Any, List
from datetime import datetime, timezone


class OrchestratorAgent:
    """
    Central coordinator for the autonomous incident-response system.

    Workflow:

        Sensor Data
             ↓
        Data Analyst Agent
             ↓
        Drone Commander Agent
             ↓
        Drone Simulator
             ↓
        RAG Knowledge Retrieval
             ↓
        Escalation Agent
             ↓
        Final Decision
             ↓
        Sandbox Execution
             ↓
        Structured Incident Logging

    The orchestrator records an explicit action/observation
    trace for every major step.
    """

    def __init__(self):
        self.agent_name = "orchestrator"

    # ================================================================
    # TRACE
    # ================================================================

    def _add_trace(
        self,
        trace: List[Dict[str, Any]],
        step: int,
        agent: str,
        action: str,
        observation: str,
    ) -> None:

        trace.append(
            {
                "step": step,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent": agent,
                "action": action,
                "observation": observation,
            }
        )

    # ================================================================
    # FINAL DECISION
    # ================================================================

    def _build_final_decision(
        self,
        escalation_result: Dict[str, Any],
    ) -> Dict[str, Any]:

        return {
            "decision": escalation_result.get(
                "decision",
                "additional_diagnostic_evidence",
            ),
            "priority": escalation_result.get(
                "priority",
                "medium",
            ),
            "action": escalation_result.get(
                "action",
                "Collect additional diagnostic evidence.",
            ),
            "reason": escalation_result.get(
                "reason",
                "Insufficient evidence for a definitive decision.",
            ),
        }

    # ================================================================
    # SANDBOX RESULT NORMALIZATION
    # ================================================================

    def _normalize_sandbox_result(
        self,
        sandbox_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Normalize the SandboxExecutor response.

        SandboxExecutor returns execution details inside:
            sandbox_result["execution"]

        The orchestrator exposes the important execution fields
        at the top level as well, so downstream components such as
        the scenario runner and dashboard can access them directly.
        """

        execution = sandbox_result.get(
            "execution",
            {},
        )

        return {
            # Preserve the original SandboxExecutor response.
            **sandbox_result,

            # Expose execution fields at the top level.
            "status": execution.get("status"),
            "action": execution.get("action"),
            "allowed": execution.get("allowed"),
            "description": execution.get("description"),
            "risk_level": execution.get("risk_level"),
            "simulation": execution.get("simulation"),
            "real_system_modified": execution.get(
                "real_system_modified"
            ),
            "timestamp": execution.get("timestamp"),
        }

    # ================================================================
    # MAIN ORCHESTRATION
    # ================================================================

    def run(
        self,
        sensor_data,
    ) -> Dict[str, Any]:

        if sensor_data is None:
            raise ValueError(
                "sensor_data cannot be None"
            )

        if len(sensor_data) == 0:
            raise ValueError(
                "sensor_data cannot be empty"
            )

        trace: List[Dict[str, Any]] = []

        step = 1

        # ============================================================
        # IMPORT AGENTS
        # ============================================================

        from src.agents.data_analyst_agent import (
            get_data_analyst_agent,
        )

        from src.agents.drone_commander_agent import (
            get_drone_commander_agent,
        )

        from src.agents.drone_simulator import (
            get_drone_simulator,
        )

        from src.rag.retriever import (
            get_knowledge_retriever,
        )

        from src.agents.escalation_agent import (
            get_escalation_agent,
        )

        from src.agents.incident_report import (
            generate_incident_report,
        )

        from src.sandbox.executor import (
            SandboxExecutor,
        )

        from src.logging.logger import (
            IncidentLogger,
        )

        # ============================================================
        # STEP 1 — DATA ANALYST
        # ============================================================

        self._add_trace(
            trace,
            step,
            "orchestrator",
            "delegate_sensor_analysis",
            "Sending sensor window to Data Analyst Agent.",
        )

        analyst = get_data_analyst_agent()

        analyst_result = analyst.analyze(
            sensor_data
        )

        step += 1

        incident_level = analyst_result.get(
            "incident_level",
            "normal",
        )

        ml_analysis = analyst_result.get(
            "ml_analysis",
            {},
        )

        ml_probability = ml_analysis.get(
            "anomaly_probability",
            0.0,
        )

        self._add_trace(
            trace,
            step,
            "data_analyst",
            "analyze_sensor_data",
            (
                f"Incident level={incident_level}; "
                f"ML anomaly probability="
                f"{ml_probability:.4f}."
            ),
        )

        # ============================================================
        # NORMAL PATH
        # ============================================================

        if incident_level == "normal":

            step += 1

            self._add_trace(
                trace,
                step,
                "orchestrator",
                "stop_investigation",
                (
                    "No significant anomaly detected; "
                    "additional investigation is not required."
                ),
            )

            final_decision = {
                "decision": "continued_monitoring",
                "priority": "low",
                "action": (
                    "Continue routine monitoring and "
                    "reassess if anomaly evidence persists."
                ),
                "reason": (
                    "The Data Analyst Agent found no "
                    "significant anomaly evidence."
                ),
            }

            # --------------------------------------------------------
            # Sandbox for normal operation
            # --------------------------------------------------------

            sandbox = SandboxExecutor()

            sandbox_raw_result = sandbox.execute_from_incident(
                {
                    "final_decision": final_decision,
                    "incident_level": incident_level,
                }
            )

            sandbox_result = self._normalize_sandbox_result(
                sandbox_raw_result
            )

            step += 1

            self._add_trace(
                trace,
                step,
                "sandbox_executor",
                "execute_safe_action",
                (
                    f"Status={sandbox_result.get('status')}; "
                    f"action={sandbox_result.get('action')}."
                ),
            )

            # --------------------------------------------------------
            # Complete normal incident result
            # --------------------------------------------------------

            incident_result = {
                "agent": self.agent_name,
                "status": "orchestration_complete",
                "incident_level": incident_level,
                "samples_analyzed": len(sensor_data),

                "data_analyst": analyst_result,

                "drone_commander": None,
                "drone_inspection": None,

                "rag": None,

                "escalation": None,

                "sandbox": sandbox_result,

                "trace": trace,
                "trace_length": len(trace),

                "final_decision": final_decision,
            }

            # --------------------------------------------------------
            # Structured logging
            # --------------------------------------------------------

            logger = IncidentLogger()

            incident_log = logger.log_incident(
                incident_result
            )

            incident_result["incident_log"] = incident_log

            return incident_result

        # ============================================================
        # STEP 2 — DRONE COMMANDER
        # ============================================================

        step += 1

        self._add_trace(
            trace,
            step,
            "orchestrator",
            "request_physical_inspection",
            (
                "Anomaly evidence requires additional "
                "physical inspection."
            ),
        )

        commander = get_drone_commander_agent()

        # IMPORTANT:
        # Drone Commander API is inspect(), not decide().
        drone_command = commander.inspect(
            analyst_result
        )

        step += 1

        self._add_trace(
            trace,
            step,
            "drone_commander",
            "evaluate_inspection_requirement",
            (
                f"Drone dispatched="
                f"{drone_command.get('drone_dispatched', False)}; "
                f"target="
                f"{drone_command.get('target')}."
            ),
        )

        # ============================================================
        # STEP 3 — DRONE INSPECTION
        # ============================================================

        drone_result = None

        if drone_command.get(
            "drone_dispatched",
            False,
        ):

            step += 1

            self._add_trace(
                trace,
                step,
                "orchestrator",
                "execute_drone_inspection",
                (
                    "Executing simulated drone inspection "
                    "for additional physical evidence."
                ),
            )

            simulator = get_drone_simulator()

            # IMPORTANT:
            # Simulator expects the complete drone_command.
            drone_result = simulator.inspect(
                drone_command
            )

            step += 1

            self._add_trace(
                trace,
                step,
                "drone_simulator",
                "inspect_equipment",
                (
                    f"Visual anomaly="
                    f"{drone_result.get('visual_anomaly_detected', False)}; "
                    f"confidence="
                    f"{drone_result.get('confidence', 0.0):.2f}."
                ),
            )

        else:

            step += 1

            self._add_trace(
                trace,
                step,
                "orchestrator",
                "skip_drone_inspection",
                "Drone inspection was not required.",
            )

            drone_result = {
                "agent": "drone_simulator",
                "status": "inspection_not_performed",
                "visual_anomaly_detected": False,
                "observations": [],
                "confidence": 0.0,
            }

        # ============================================================
        # STEP 4 — RAG KNOWLEDGE RETRIEVAL
        # ============================================================

        statistical = analyst_result.get(
            "statistical_analysis",
            {},
        )

        top_sensors = statistical.get(
            "top_sensors",
            [],
        )

        affected_sensors = [
            item.get("sensor")
            for item in top_sensors
            if item.get("anomaly_count", 0) > 0
        ]

        visual_anomaly = drone_result.get(
            "visual_anomaly_detected",
            False,
        )

        if visual_anomaly:

            rag_query = (
                "pump mechanical vibration "
                "maintenance inspection"
            )

        elif "Pressure" in affected_sensors:

            rag_query = (
                "pump pressure leakage "
                "maintenance inspection"
            )

        elif "Temperature" in affected_sensors:

            rag_query = (
                "pump temperature abnormality "
                "maintenance inspection"
            )

        elif "Current" in affected_sensors:

            rag_query = (
                "pump electrical current "
                "maintenance inspection"
            )

        else:

            rag_query = (
                "incident escalation "
                "maintenance procedure"
            )

        step += 1

        self._add_trace(
            trace,
            step,
            "orchestrator",
            "retrieve_operational_knowledge",
            f"RAG query: '{rag_query}'.",
        )

        retriever = get_knowledge_retriever()

        rag_result = retriever.build_context(
            query=rag_query,
            top_k=3,
        )

        step += 1

        rag_sources = [
            result.get("source")
            for result in rag_result.get(
                "results",
                [],
            )
        ]

        self._add_trace(
            trace,
            step,
            "rag_retriever",
            "retrieve_knowledge",
            (
                f"Retrieved "
                f"{rag_result.get('result_count', 0)} "
                f"relevant knowledge chunks."
            ),
        )

        # ============================================================
        # STEP 5 — ESCALATION AGENT
        # ============================================================

        step += 1

        self._add_trace(
            trace,
            step,
            "orchestrator",
            "evaluate_incident",
            (
                "Combining sensor, statistical, drone, "
                "and operational knowledge evidence."
            ),
        )

        escalation = get_escalation_agent()

        escalation_result = escalation.evaluate(
            analyst_result,
            drone_result,
            rag_result,
        )

        step += 1

        decision = escalation_result.get(
            "decision",
            "additional_diagnostic_evidence",
        )

        priority = escalation_result.get(
            "priority",
            "medium",
        )

        self._add_trace(
            trace,
            step,
            "escalation_agent",
            "make_escalation_decision",
            (
                f"Decision={decision}; "
                f"priority={priority}."
            ),
        )

        # ============================================================
        # FINAL DECISION
        # ============================================================

        final_decision = self._build_final_decision(
            escalation_result
        )

        step += 1

        self._add_trace(
            trace,
            step,
            "orchestrator",
            "finalize_incident_response",
            (
                f"Final decision="
                f"{final_decision['decision']}; "
                f"priority="
                f"{final_decision['priority']}."
            ),
        )

        # ============================================================
        # STEP 6 — SANDBOX EXECUTION
        # ============================================================

        sandbox = SandboxExecutor()

        sandbox_raw_result = sandbox.execute_from_incident(
            {
                "final_decision": final_decision,
                "incident_level": incident_level,
            }
        )

        # IMPORTANT:
        # SandboxExecutor returns the actual execution result
        # inside the "execution" field.
        sandbox_result = self._normalize_sandbox_result(
            sandbox_raw_result
        )

        step += 1

        self._add_trace(
            trace,
            step,
            "sandbox_executor",
            "execute_safe_action",
            (
                f"Status={sandbox_result.get('status')}; "
                f"action={sandbox_result.get('action')}."
            ),
        )

        # ============================================================
        # STEP 6B - LLM INCIDENT REPORT (explanation only)
        # The LLM never makes decisions. It only writes a plain-language
        # report from evidence the agents already produced.
        # ============================================================

        procedure_texts = []

        for item in rag_result.get("results", [])[:2]:

            item = item if isinstance(item, dict) else {}

            procedure_texts.append(
                str(
                    item.get("text")
                    or item.get("content")
                    or item.get("chunk")
                    or ""
                )[:600]
            )

        procedure_text = (
            "\n".join(t for t in procedure_texts if t)
            or "No procedure retrieved."
        )

        report_evidence = {
            "sensor": (
                ", ".join(str(s) for s in affected_sensors)
                or "multiple sensors"
            ),
            "summary": (
                f"incident level {incident_level}; "
                f"ML anomaly probability {ml_probability:.2f}; "
                f"drone visual anomaly: {visual_anomaly}"
            ),
            "decision": final_decision["decision"],
            "priority": final_decision["priority"],
            "action": final_decision["action"],
            "reason": final_decision["reason"],
        }

        escalated = "escalation" in str(
            final_decision["decision"]
        ).lower()

        try:
            report_text, report_source = generate_incident_report(
                report_evidence,
                procedure_text,
                escalated,
            )
        except Exception:
            report_text, report_source = (
                "Incident report unavailable.",
                "template",
            )

        incident_report = {
            "text": report_text,
            "source": report_source,
        }

        step += 1

        self._add_trace(
            trace,
            step,
            "incident_reporter",
            "generate_incident_report",
            f"Operator report generated via {report_source}.",
        )

        # ============================================================
        # COMPLETE INCIDENT RESULT
        # ============================================================

        incident_result = {
            "agent": self.agent_name,
            "status": "orchestration_complete",
            "incident_level": incident_level,
            "samples_analyzed": len(sensor_data),

            "data_analyst": analyst_result,

            "drone_commander": drone_command,

            "drone_inspection": drone_result,

            "rag": {
                "query": rag_query,
                "result_count": rag_result.get(
                    "result_count",
                    0,
                ),
                "sources": rag_sources,
                "results": rag_result.get(
                    "results",
                    [],
                ),
            },

            "escalation": escalation_result,
            "incident_report": incident_report,

            "sandbox": sandbox_result,

            "trace": trace,
            "trace_length": len(trace),

            "final_decision": final_decision,
        }

        # ============================================================
        # STEP 7 — STRUCTURED INCIDENT LOGGING
        # ============================================================

        logger = IncidentLogger()

        incident_log = logger.log_incident(
            incident_result
        )

        incident_result["incident_log"] = incident_log

        return incident_result


# ======================================================================
# SINGLETON
# ======================================================================

_orchestrator = None


def get_orchestrator_agent() -> OrchestratorAgent:

    global _orchestrator

    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()

    return _orchestrator


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def run_incident_response(
    sensor_data,
) -> Dict[str, Any]:

    orchestrator = get_orchestrator_agent()

    return orchestrator.run(
        sensor_data
    )


# ======================================================================
# DIRECT TEST
# ======================================================================

if __name__ == "__main__":

    from src.agents.tools import (
        load_processed_data,
    )

    print("=" * 70)
    print("ORCHESTRATOR AGENT TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------------

    print("\nLoading sensor data...")

    df = load_processed_data()

    analysis_window = df.tail(500).copy()

    print(
        f"Analysis window: "
        f"{len(analysis_window)} samples"
    )

    # --------------------------------------------------------------
    # RUN COMPLETE SYSTEM
    # --------------------------------------------------------------

    print(
        "\nStarting autonomous incident-response workflow..."
    )

    orchestrator = get_orchestrator_agent()

    result = orchestrator.run(
        analysis_window
    )

    # --------------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL INCIDENT DECISION")
    print("=" * 70)

    final = result[
        "final_decision"
    ]

    print(
        f"decision   : {final['decision']}"
    )

    print(
        f"priority   : {final['priority']}"
    )

    print(
        f"action     : {final['action']}"
    )

    print(
        f"reason     : {final['reason']}"
    )

    # --------------------------------------------------------------
    # SANDBOX
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SANDBOX EXECUTION")
    print("=" * 70)

    sandbox = result.get(
        "sandbox",
        {},
    )

    print(
        f"status              : "
        f"{sandbox.get('status')}"
    )

    print(
        f"action              : "
        f"{sandbox.get('action')}"
    )

    print(
        f"allowed             : "
        f"{sandbox.get('allowed')}"
    )

    print(
        f"simulation          : "
        f"{sandbox.get('simulation')}"
    )

    print(
        f"real system modified: "
        f"{sandbox.get('real_system_modified')}"
    )

    # --------------------------------------------------------------
    # AGENT TRACE
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("AGENT TRACE")
    print("=" * 70)

    for trace_item in result["trace"]:

        print(
            f"\nSTEP {trace_item['step']}"
        )

        print(
            f"Agent       : "
            f"{trace_item['agent']}"
        )

        print(
            f"Action      : "
            f"{trace_item['action']}"
        )

        print(
            f"Observation : "
            f"{trace_item['observation']}"
        )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ORCHESTRATION SUMMARY")
    print("=" * 70)

    print(
        f"Incident level       : "
        f"{result['incident_level']}"
    )

    print(
        f"Samples analyzed     : "
        f"{result['samples_analyzed']}"
    )

    print(
        f"Trace steps          : "
        f"{result['trace_length']}"
    )

    drone = result.get(
        "drone_commander"
    )

    if drone:
        print(
            f"Drone dispatched     : "
            f"{drone.get('drone_dispatched')}"
        )
    else:
        print(
            "Drone dispatched     : False"
        )

    rag = result.get(
        "rag"
    )

    if rag:
        print(
            f"RAG results          : "
            f"{rag.get('result_count', 0)}"
        )
    else:
        print(
            "RAG results          : 0"
        )

    escalation = result.get(
        "escalation"
    )

    if escalation:
        print(
            f"Escalation decision  : "
            f"{escalation.get('decision')}"
        )
    else:
        print(
            "Escalation decision  : None"
        )

    print(
        f"Sandbox status       : "
        f"{sandbox.get('status')}"
    )

    print(
        f"Sandbox action       : "
        f"{sandbox.get('action')}"
    )

    print(
        f"Incident log         : "
        f"{result.get('incident_log')}"
    )

    print("\n" + "=" * 70)
    print("ORCHESTRATOR AGENT TEST COMPLETE")
    print("=" * 70)