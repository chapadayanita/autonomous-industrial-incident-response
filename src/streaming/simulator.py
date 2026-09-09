from typing import Dict, Any, Iterator, Optional
import time

import pandas as pd

from src.agents.tools import load_processed_data
from src.data.streaming_features import StreamingFeatureBuilder


class SensorStreamSimulator:
    """
    Software-only virtual sensor stream.

    Replays historical SKAB sensor data sequentially so that the
    incident-response system can behave like it is receiving
    continuously arriving sensor measurements.

    This does NOT represent a physical sensor connection.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        batch_size: int = 10,
        delay_seconds: float = 0.0,
        history_size: int = 120,
    ):
        self.agent_name = "sensor_stream_simulator"

        self.df = (
            df.copy()
            if df is not None
            else load_processed_data()
        )

        self.batch_size = max(1, int(batch_size))
        self.delay_seconds = max(0.0, float(delay_seconds))

        self.feature_builder = StreamingFeatureBuilder(
            history_size=history_size
        )

        self.current_position = 0
        self.total_samples = len(self.df)

    # ------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the virtual sensor stream to the beginning."""

        self.current_position = 0
        self.feature_builder.reset()

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return current stream status."""

        return {
            "agent": self.agent_name,
            "current_position": self.current_position,
            "total_samples": self.total_samples,
            "samples_processed": self.current_position,
            "samples_remaining": max(
                0,
                self.total_samples - self.current_position,
            ),
            "progress_percent": (
                round(
                    (
                        self.current_position
                        / self.total_samples
                    )
                    * 100,
                    2,
                )
                if self.total_samples > 0
                else 0.0
            ),
            "completed": (
                self.current_position
                >= self.total_samples
            ),
        }

    # ------------------------------------------------------------------
    # NEXT BATCH
    # ------------------------------------------------------------------

    def next_batch(
        self,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Return the next batch of virtual sensor readings.
        """

        if self.current_position >= self.total_samples:
            return {
                "agent": self.agent_name,
                "status": "stream_complete",
                "samples": [],
                "features": None,
                **self.status(),
            }

        size = (
            self.batch_size
            if batch_size is None
            else max(1, int(batch_size))
        )

        start = self.current_position

        end = min(
            start + size,
            self.total_samples,
        )

        batch = self.df.iloc[start:end].copy()

        self.current_position = end

        # --------------------------------------------------------------
        # Update rolling feature history
        # --------------------------------------------------------------

        features = None

        try:
            features = self.feature_builder.update(
                batch
            )
        except Exception as exc:
            return {
                "agent": self.agent_name,
                "status": "feature_generation_error",
                "samples": batch.to_dict(
                    orient="records"
                ),
                "features": None,
                "error": str(exc),
                **self.status(),
            }

        # --------------------------------------------------------------
        # Optional simulation delay
        # --------------------------------------------------------------

        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

        return {
            "agent": self.agent_name,
            "status": "streaming",
            "samples": batch.to_dict(
                orient="records"
            ),
            "features": features,
            "batch_start": start,
            "batch_end": end - 1,
            "batch_size": len(batch),
            **self.status(),
        }

    # ------------------------------------------------------------------
    # STREAM GENERATOR
    # ------------------------------------------------------------------

    def stream(
        self,
        batch_size: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Yield sensor batches until the virtual stream ends.
        """

        while self.current_position < self.total_samples:

            result = self.next_batch(
                batch_size=batch_size
            )

            yield result

    # ------------------------------------------------------------------
    # RUN STREAM
    # ------------------------------------------------------------------

    def run(
        self,
        max_batches: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """
        Simple console demonstration of the virtual stream.
        """

        self.reset()

        batches_processed = 0

        print("=" * 70)
        print("VIRTUAL SENSOR STREAM")
        print("=" * 70)

        while (
            self.current_position
            < self.total_samples
        ):

            result = self.next_batch(
                batch_size=batch_size
            )

            batches_processed += 1

            print(
                f"\nBatch {batches_processed}"
            )

            print(
                f"Samples: "
                f"{result.get('batch_start')} → "
                f"{result.get('batch_end')}"
            )

            print(
                f"Progress: "
                f"{result.get('progress_percent')}%"
            )

            print(
                f"Status: "
                f"{result.get('status')}"
            )

            if (
                max_batches is not None
                and batches_processed >= max_batches
            ):
                break

        print("\n" + "=" * 70)
        print("STREAM TEST COMPLETE")
        print("=" * 70)

        print(
            f"Samples processed: "
            f"{self.current_position}"
        )


# ======================================================================
# SINGLETON
# ======================================================================

_sensor_stream_simulator = None


def get_sensor_stream_simulator() -> SensorStreamSimulator:
    global _sensor_stream_simulator

    if _sensor_stream_simulator is None:
        _sensor_stream_simulator = SensorStreamSimulator()

    return _sensor_stream_simulator


# ======================================================================
# CONVENIENCE FUNCTION
# ======================================================================

def run_sensor_stream(
    batch_size: int = 10,
    max_batches: Optional[int] = None,
) -> None:

    simulator = SensorStreamSimulator(
        batch_size=batch_size
    )

    simulator.run(
        max_batches=max_batches
    )


# ======================================================================
# TEST
# ======================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SENSOR STREAM SIMULATOR TEST")
    print("=" * 70)

    simulator = SensorStreamSimulator(
        batch_size=10,
        delay_seconds=0.0,
        history_size=120,
    )

    print("\nInitial status:")
    print(simulator.status())

    print("\nReading first 3 batches...")

    for i, result in enumerate(
        simulator.stream(batch_size=10)
    ):

        print(
            f"\nBatch {i + 1}"
        )

        print(
            "Status:",
            result["status"]
        )

        print(
            "Batch:",
            result["batch_start"],
            "→",
            result["batch_end"],
        )

        print(
            "Samples:",
            result["batch_size"],
        )

        print(
            "Progress:",
            f"{result['progress_percent']}%",
        )

        if i >= 2:
            break

    print("\nFinal status:")
    print(simulator.status())

    print("\n" + "=" * 70)
    print("SENSOR STREAM SIMULATOR TEST COMPLETE")
    print("=" * 70)