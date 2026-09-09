# src/data/streaming_features.py

from typing import Optional

import pandas as pd

from src.models.baseline import create_features


class StreamingFeatureBuilder:
    """
    Stateful feature builder for streaming/replayed sensor data.

    Keeps enough historical raw observations so that the same
    feature-engineering logic used during model training can be
    applied to incoming sensor windows.
    """

    def __init__(
        self,
        history_size: int = 120,
    ):
        if history_size < 60:
            raise ValueError(
                "history_size must be at least 60 "
                "because the training pipeline uses "
                "rolling historical features."
            )

        self.history_size = history_size

        self._buffer = pd.DataFrame()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Clear all previously stored streaming history.
        """

        self._buffer = pd.DataFrame()

    # ========================================================
    # ADD DATA
    # ========================================================

    def update(
        self,
        sensor_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add a new raw sensor window and return engineered
        features for the newly received observations.
        """

        if sensor_data is None:
            raise ValueError(
                "sensor_data cannot be None."
            )

        if not isinstance(
            sensor_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "sensor_data must be a pandas DataFrame."
            )

        if sensor_data.empty:
            return pd.DataFrame()

        incoming = sensor_data.copy()

        # ----------------------------------------------------
        # Preserve source/experiment identity when available
        # ----------------------------------------------------

        if (
            "source_file" not in incoming.columns
            and "source_file" in self._buffer.columns
            and not self._buffer.empty
        ):
            incoming["source_file"] = (
                self._buffer["source_file"]
                .iloc[-1]
            )

        # ----------------------------------------------------
        # Combine previous history + new observations
        # ----------------------------------------------------

        previous_length = len(
            self._buffer
        )

        combined = pd.concat(
            [
                self._buffer,
                incoming,
            ],
            ignore_index=True,
        )

        # ----------------------------------------------------
        # Remove duplicate timestamps when possible
        # ----------------------------------------------------

        if "datetime" in combined.columns:

            combined["datetime"] = pd.to_datetime(
                combined["datetime"],
                errors="coerce",
            )

            combined = (
                combined
                .sort_values(
                    ["source_file", "datetime"]
                    if "source_file" in combined.columns
                    else ["datetime"]
                )
                .drop_duplicates(
                    subset=(
                        ["source_file", "datetime"]
                        if "source_file" in combined.columns
                        else ["datetime"]
                    ),
                    keep="last",
                )
                .reset_index(drop=True)
            )

        # ----------------------------------------------------
        # Keep historical context
        #
        # Keep enough samples for rolling/difference features.
        # ----------------------------------------------------

        if "source_file" in combined.columns:

            buffers = []

            for _, group in combined.groupby(
                "source_file",
                sort=False,
            ):

                buffers.append(
                    group.tail(
                        self.history_size
                    )
                )

            combined = pd.concat(
                buffers,
                ignore_index=True,
            )

        else:

            combined = combined.tail(
                self.history_size
            ).reset_index(
                drop=True
            )

        # ----------------------------------------------------
        # Feature engineering
        #
        # IMPORTANT:
        # This is the exact existing training feature builder.
        # ----------------------------------------------------

        featured, feature_columns = (
            create_features(
                combined.copy()
            )
        )

        # ----------------------------------------------------
        # Return only newly arrived observations.
        #
        # Because create_features can remove initial rows
        # during rolling/difference calculations, we identify
        # new observations by their original timestamps when
        # possible.
        # ----------------------------------------------------

        if (
            "datetime" in incoming.columns
            and "datetime" in featured.columns
        ):

            incoming_times = set(
                pd.to_datetime(
                    incoming["datetime"],
                    errors="coerce",
                )
                .dropna()
            )

            result = featured[
                pd.to_datetime(
                    featured["datetime"],
                    errors="coerce",
                ).isin(
                    incoming_times
                )
            ].copy()

        else:

            result = featured.tail(
                len(incoming)
            ).copy()

        # ----------------------------------------------------
        # Update persistent history
        # ----------------------------------------------------

        self._buffer = combined.copy()

        # ----------------------------------------------------
        # Ensure exact training feature set
        # ----------------------------------------------------

        if result.empty:
            return result

        available_features = [
            column
            for column in feature_columns
            if column in result.columns
        ]

        result = result[
            [
                column
                for column in result.columns
                if column not in []
            ]
        ]

        # Keep metadata + all engineered features.
        # The model tool will select its exact 88 columns.
        return result

    # ========================================================
    # BUFFER INFORMATION
    # ========================================================

    @property
    def buffer_size(self) -> int:
        """
        Number of raw observations currently retained.
        """

        return len(
            self._buffer
        )

    @property
    def buffer(self) -> pd.DataFrame:
        """
        Return a copy of the current raw history.
        """

        return self._buffer.copy()