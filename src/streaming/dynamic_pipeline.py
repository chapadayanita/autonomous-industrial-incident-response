from typing import Dict, Any, Optional
from collections import deque

import pandas as pd

from src.agents.orchestrator import get_orchestrator_agent
from src.agents.tools import load_processed_data
from src.streaming.simulator import SensorStreamSimulator


class DynamicIncidentPipeline:
    """
    Connects the virtual sensor stream to the existing
    autonomous incident-response orchestrator.

    The existing agents and ML model are NOT modified.

    Flow:
        Virtual Sensor Stream
                ↓
        Rolling Sensor Window
                ↓
        Existing Orchestrator
                ↓
        Data Analyst
                ↓
        Drone Commander
                ↓
        Drone Simulator
                ↓
        RAG
                ↓
        Escalation
                ↓
        Sandbox
    """

    def __init__(
        self,
        source_file: Optional[str] = None,
        start_row: int = 0,
        stream_batch_size: int = 10,
        analysis_window: int = 500,
    ):
        self.agent_name = "dynamic_incident_pipeline"

        # --------------------------------------------------------------
        # Load processed sensor data
        # --------------------------------------------------------------

        df = load_processed_data()

        # --------------------------------------------------------------
        # Optional experiment/source selection
        # --------------------------------------------------------------

        if source_file is not None:

            if "source_file" not in df.columns:
                raise ValueError(
                    "Processed data does not contain 'source_file'."
                )

            matching = df[
                df["source_file"].astype(str) == str(source_file)
            ].copy()

            if matching.empty:
                raise ValueError(
                    f"No data found for source_file='{source_file}'."
                )

            df = matching.reset_index(drop=True)

        # --------------------------------------------------------------
        # Validate starting row
        # --------------------------------------------------------------

        if start_row < 0:
            raise ValueError(
                "start_row must be >= 0."
            )

        if start_row >= len(df):
            raise ValueError(
                f"start_row={start_row} is outside "
                f"the available data ({len(df)} rows)."
            )

        self.df = df.iloc[start_row:].reset_index(
            drop=True
        )

        self.stream_batch_size = max(
            1,
            int(stream_batch_size),
        )

        self.analysis_window = max(
            1,
            int(analysis_window),
        )

        # --------------------------------------------------------------
        # Rolling history
        # --------------------------------------------------------------

        self.history = deque(
            maxlen=self.analysis_window
        )

        # --------------------------------------------------------------
        # Virtual sensor
        # --------------------------------------------------------------

        self.stream = SensorStreamSimulator(
            df=self.df,
            batch_size=self.stream_batch_size,
            delay_seconds=0.0,
            history_size=120,
        )

        # --------------------------------------------------------------
        # Existing autonomous orchestrator
        # --------------------------------------------------------------

        self.orchestrator = get_orchestrator_agent()

        self.batches_processed = 0
        self.investigations_run = 0

        self.last_result = None

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self) -> None:
        """Reset the complete dynamic pipeline."""

        self.history.clear()

        self.stream.reset()

        self.batches_processed = 0
        self.investigations_run = 0

        self.last_result = None

    # ==================================================================
    # STATUS
    # ==================================================================

    def status(self) -> Dict[str, Any]:
        """Return current pipeline status."""

        stream_status = self.stream.status()

        return {
            "agent": self.agent_name,
            "batches_processed": self.batches_processed,
            "investigations_run": self.investigations_run,
            "analysis_samples": len(self.history),
            "analysis_window": self.analysis_window,
            "stream": stream_status,
            "last_incident_level": (
                self.last_result.get("incident_level")
                if self.last_result
                else None
            ),
            "last_decision": (
                self.last_result
                .get("final_decision", {})
                .get("action")
                if self.last_result
                else None
            ),
        }

    # ==================================================================
    # PROCESS ONE BATCH
    # ==================================================================

    def process_next_batch(self) -> Dict[str, Any]:
        """
        Process the next virtual sensor batch.

        Once enough history exists, the current rolling window
        is sent through the existing autonomous orchestrator.
        """

        stream_result = self.stream.next_batch()

        if stream_result["status"] == "stream_complete":
            return {
                "agent": self.agent_name,
                "status": "stream_complete",
                "investigation_performed": False,
                "stream": stream_result,
                **self.status(),
            }

        self.batches_processed += 1

        # --------------------------------------------------------------
        # Add newly arriving sensor samples to rolling history
        # --------------------------------------------------------------

        samples = stream_result.get(
            "samples",
            [],
        )

        for sample in samples:
            self.history.append(sample)

        # --------------------------------------------------------------
        # Not enough samples yet
        # --------------------------------------------------------------

        if len(self.history) < self.analysis_window:

            return {
                "agent": self.agent_name,
                "status": "collecting_history",
                "investigation_performed": False,
                "samples_received": len(samples),
                "analysis_samples": len(self.history),
                "required_samples": self.analysis_window,
                "stream": stream_result,
                **self.status(),
            }

        # --------------------------------------------------------------
        # Build current rolling sensor window
        # --------------------------------------------------------------

        analysis_df = pd.DataFrame(
            list(self.history)
        )

        # --------------------------------------------------------------
        # Existing autonomous orchestrator
        # --------------------------------------------------------------

        result = self.orchestrator.run(
            analysis_df
        )

        self.investigations_run += 1
        self.last_result = result

        return {
            "agent": self.agent_name,
            "status": "investigation_complete",
            "investigation_performed": True,
            "samples_received": len(samples),
            "analysis_samples": len(analysis_df),
            "stream": stream_result,
            "incident_level": result.get(
                "incident_level"
            ),
            "decision": result.get(
                "final_decision",
                {}
            ),
            "result": result,
            **self.status(),
        }

    # ==================================================================
    # RUN MULTIPLE BATCHES
    # ==================================================================

    def run(
        self,
        max_batches: Optional[int] = None,
    ):
        """
        Run the dynamic sensor simulation.

        max_batches can be used to limit the demonstration.
        """

        self.reset()

        results = []

        while True:

            result = self.process_next_batch()

            results.append(result)

            if result["status"] == "stream_complete":
                break

            if (
                max_batches is not None
                and self.batches_processed >= max_batches
            ):
                break

        return results


# ======================================================================
# SIMPLE TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("DYNAMIC INCIDENT PIPELINE TEST")
    print("=" * 70)

    print("\nLoading virtual sensor stream...")

    pipeline = DynamicIncidentPipeline(
        source_file="10.csv",
        start_row=0,
        stream_batch_size=10,
        analysis_window=500,
    )

    print("\nInitial status:")
    print(pipeline.status())

    print("\nProcessing first batches...")

    for i in range(52):

        result = pipeline.process_next_batch()

        print(
            f"\nBatch {i + 1}"
        )

        print(
            "Status:",
            result["status"]
        )

        print(
            "Samples in rolling window:",
            result["analysis_samples"]
        )

        if result["investigation_performed"]:

            print(
                "Incident:",
                result.get("incident_level")
            )

            print(
                "Decision:",
                result.get("decision", {}).get(
                    "action",
                    "N/A"
                )
            )

        if result["status"] == "stream_complete":
            break

    print("\n" + "=" * 70)
    print("FINAL STATUS")
    print("=" * 70)

    print(pipeline.status())

    print("\n" + "=" * 70)
    print("DYNAMIC PIPELINE TEST COMPLETE")
    print("=" * 70)