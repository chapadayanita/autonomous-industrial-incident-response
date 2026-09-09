import json
from pathlib import Path

from src.agents.tools import load_processed_data
from src.agents.orchestrator import get_orchestrator_agent


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"


# ======================================================================
# SCENARIOS TO RUN
# ======================================================================

SCENARIO_FILES = [
    "scenario_normal.json",
    "scenario_ambiguous.json",
    "scenario_pump_fault.json",
]


# ======================================================================
# LOAD SCENARIO
# ======================================================================

def load_scenario(filename):
    path = SCENARIOS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Scenario file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# GET EXACT DATA WINDOW
# ======================================================================

def get_scenario_window(df, scenario):
    """
    Extract the exact experiment and row window
    specified by the scenario JSON file.
    """

    source_file = scenario["source_file"]
    start_row = int(scenario["start_row"])
    window_size = int(scenario["analysis_window"])

    experiment = df[
        df["source_file"] == source_file
    ].copy()

    if experiment.empty:
        raise ValueError(
            f"Source file '{source_file}' "
            f"was not found in processed data."
        )

    experiment = experiment.reset_index(drop=True)

    end_row = start_row + window_size

    if start_row < 0:
        raise ValueError(
            f"start_row cannot be negative: {start_row}"
        )

    if end_row > len(experiment):
        raise ValueError(
            f"Invalid window for {source_file}: "
            f"rows {start_row}-{end_row - 1}. "
            f"Experiment contains {len(experiment)} rows."
        )

    window = experiment.iloc[
        start_row:end_row
    ].copy()

    if len(window) != window_size:
        raise ValueError(
            f"Expected {window_size} samples but got "
            f"{len(window)} samples."
        )

    return window


# ======================================================================
# RUN ONE SCENARIO
# ======================================================================

def run_scenario(df, filename):
    scenario = load_scenario(filename)

    print("\n" + "=" * 80)
    print(f"SCENARIO: {filename}")
    print("=" * 80)

    print(
        f"Scenario ID         : "
        f"{scenario['scenario_id']}"
    )

    print(
        f"Description         : "
        f"{scenario['description']}"
    )

    print(
        f"Source file         : "
        f"{scenario['source_file']}"
    )

    start_row = int(
        scenario["start_row"]
    )

    window_size = int(
        scenario["analysis_window"]
    )

    print(
        f"Window              : "
        f"{start_row}-"
        f"{start_row + window_size - 1}"
    )

    # --------------------------------------------------------------
    # GET EXACT WINDOW
    # --------------------------------------------------------------

    window = get_scenario_window(
        df,
        scenario
    )

    # --------------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------------
    #
    # Ground truth is used only for scenario verification.
    # It is NOT passed to any agent.
    # --------------------------------------------------------------

    actual_anomalies = int(
        window["anomaly"].sum()
    )

    actual_anomaly_rate = (
        actual_anomalies / len(window)
    ) * 100

    print(
        f"Samples             : "
        f"{len(window)}"
    )

    print(
        f"Ground-truth anomalies: "
        f"{actual_anomalies}"
    )

    print(
        f"Ground-truth rate    : "
        f"{actual_anomaly_rate:.2f}%"
    )

    # --------------------------------------------------------------
    # RUN ACTUAL ORCHESTRATOR
    # --------------------------------------------------------------

    orchestrator = get_orchestrator_agent()

    result = orchestrator.run(
        window
    )

    # --------------------------------------------------------------
    # EXTRACT RESULT
    # --------------------------------------------------------------

    # IMPORTANT:
    #
    # OrchestratorAgent returns incident_level at
    # the top level of the result.
    #
    # final_decision contains decision, priority,
    # action and reason.
    # --------------------------------------------------------------

    actual_level = result[
        "incident_level"
    ]

    final_decision = result[
        "final_decision"
    ]

    actual_decision = final_decision[
        "decision"
    ]

    expected_level = scenario[
        "expected_incident_level"
    ]

    expected_decision = scenario[
        "expected_decision"
    ]

    # --------------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------------

    level_match = (
        actual_level == expected_level
    )

    decision_match = (
        actual_decision == expected_decision
    )

    overall_match = (
        level_match
        and decision_match
    )

    # --------------------------------------------------------------
    # AGENT RESULT
    # --------------------------------------------------------------

    print("\nAGENT RESULT")
    print("-" * 80)

    print(
        f"Expected level      : "
        f"{expected_level}"
    )

    print(
        f"Actual level        : "
        f"{actual_level}"
    )

    print(
        f"Level match         : "
        f"{level_match}"
    )

    print(
        f"\nExpected decision   : "
        f"{expected_decision}"
    )

    print(
        f"Actual decision     : "
        f"{actual_decision}"
    )

    print(
        f"Decision match      : "
        f"{decision_match}"
    )

    print(
        f"\nOverall match       : "
        f"{overall_match}"
    )

    # --------------------------------------------------------------
    # FINAL ACTION
    # --------------------------------------------------------------

    print("\nFINAL ACTION")
    print("-" * 80)

    print(
        f"Priority            : "
        f"{final_decision.get('priority')}"
    )

    print(
        f"Action              : "
        f"{final_decision.get('action')}"
    )

    print(
        f"Reason              : "
        f"{final_decision.get('reason')}"
    )

    # --------------------------------------------------------------
    # ORCHESTRATION DETAILS
    # --------------------------------------------------------------

    drone_command = result.get(
        "drone_commander"
    )

    drone_result = result.get(
        "drone_inspection"
    )

    rag_result = result.get(
        "rag"
    )

    print("\nORCHESTRATION")
    print("-" * 80)

    if drone_command:
        print(
            f"Drone dispatched    : "
            f"{drone_command.get('drone_dispatched', False)}"
        )

        print(
            f"Drone target        : "
            f"{drone_command.get('target')}"
        )
    else:
        print(
            "Drone dispatched    : False"
        )

    if drone_result:
        print(
            f"Visual anomaly      : "
            f"{drone_result.get('visual_anomaly_detected', False)}"
        )

        print(
            f"Drone confidence    : "
            f"{drone_result.get('confidence', 0.0):.2f}"
        )

    if rag_result:
        print(
            f"RAG results         : "
            f"{rag_result.get('result_count', 0)}"
        )

    else:
        print(
            "RAG results         : 0"
        )

    print(
        f"Trace steps         : "
        f"{len(result.get('trace', []))}"
    )

    # --------------------------------------------------------------
    # SANDBOX
    # --------------------------------------------------------------

    sandbox_result = result.get(
        "sandbox"
    )

    if sandbox_result:
        print(
            f"Sandbox status      : "
            f"{sandbox_result.get('status')}"
        )

        print(
            f"Sandbox action      : "
            f"{sandbox_result.get('action')}"
        )

    else:
        print(
            "Sandbox status      : Not executed"
        )

    # --------------------------------------------------------------
    # INCIDENT LOG
    # --------------------------------------------------------------

    incident_log = result.get(
        "incident_log"
    )

    if incident_log:
        print(
            f"Incident log        : "
            f"{incident_log}"
        )

    # --------------------------------------------------------------
    # RETURN SUMMARY
    # --------------------------------------------------------------

    return {
        "scenario": filename,
        "expected_level": expected_level,
        "actual_level": actual_level,
        "expected_decision": expected_decision,
        "actual_decision": actual_decision,
        "level_match": level_match,
        "decision_match": decision_match,
        "overall_match": overall_match,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():

    print("=" * 80)
    print("MULTI-AGENT SCENARIO TEST")
    print("=" * 80)

    # --------------------------------------------------------------
    # LOAD PROCESSED DATA
    # --------------------------------------------------------------

    df = load_processed_data()

    print(
        f"\nLoaded processed data: "
        f"{len(df)} rows"
    )

    # --------------------------------------------------------------
    # RUN ALL SCENARIOS
    # --------------------------------------------------------------

    results = []

    for filename in SCENARIO_FILES:

        try:

            result = run_scenario(
                df,
                filename
            )

            results.append(
                result
            )

        except Exception as e:

            print(
                "\nSCENARIO ERROR"
            )

            print(
                "-" * 80
            )

            print(
                f"{filename}: {e}"
            )

            results.append(
                {
                    "scenario": filename,
                    "expected_level": "ERROR",
                    "actual_level": "ERROR",
                    "expected_decision": "ERROR",
                    "actual_decision": "ERROR",
                    "level_match": False,
                    "decision_match": False,
                    "overall_match": False,
                }
            )

    # --------------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------------

    total = len(results)

    passed = sum(
        1
        for result in results
        if result["overall_match"]
    )

    failed = total - passed

    match_rate = (
        (passed / total) * 100
        if total > 0
        else 0.0
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "SCENARIO TEST SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"Total scenarios : "
        f"{total}"
    )

    print(
        f"Passed          : "
        f"{passed}"
    )

    print(
        f"Failed          : "
        f"{failed}"
    )

    print(
        f"Match rate      : "
        f"{match_rate:.2f}%"
    )

    # --------------------------------------------------------------
    # INDIVIDUAL RESULTS
    # --------------------------------------------------------------

    print(
        "\nScenario results:"
    )

    print(
        "-" * 80
    )

    for result in results:

        status = (
            "PASS"
            if result["overall_match"]
            else "FAIL"
        )

        print(
            f"{status:6} | "
            f"{result['scenario']:28} | "
            f"level: "
            f"{result['actual_level']:8} | "
            f"decision: "
            f"{result['actual_decision']}"
        )

    print(
        "=" * 80
    )

    print(
        "SCENARIO TEST COMPLETE"
    )

    print(
        "=" * 80
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()