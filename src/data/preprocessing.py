from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# SENSOR COLUMNS
# ============================================================

SENSOR_COLUMNS: List[str] = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare SKAB sensor data for analysis.

    This function:
    1. Creates a copy of the input DataFrame.
    2. Removes duplicate timestamps within each experiment.
    3. Sorts data chronologically.
    4. Converts sensor columns to numeric values.
    5. Handles invalid numeric values.
    6. Fills missing sensor values using interpolation.
    """

    data = df.copy()

    print("Starting preprocessing...")
    print(f"Input rows: {len(data):,}")

    # --------------------------------------------------------
    # 1. Ensure datetime is valid
    # --------------------------------------------------------

    data["datetime"] = pd.to_datetime(
        data["datetime"],
        errors="coerce"
    )

    invalid_datetime = data["datetime"].isna().sum()

    if invalid_datetime > 0:
        print(
            f"Removing {invalid_datetime} rows with invalid datetime."
        )

        data = data.dropna(
            subset=["datetime"]
        )

    # --------------------------------------------------------
    # 2. Convert sensor columns to numeric
    # --------------------------------------------------------

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # 3. Sort by experiment and timestamp
    # --------------------------------------------------------

    data = data.sort_values(
        ["source_file", "datetime"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Remove duplicate timestamps
    # --------------------------------------------------------

    before_duplicates = len(data)

    data = data.drop_duplicates(
        subset=["source_file", "datetime"]
    ).reset_index(drop=True)

    removed_duplicates = (
        before_duplicates - len(data)
    )

    print(
        f"Duplicate rows removed: {removed_duplicates:,}"
    )

    # --------------------------------------------------------
    # 5. Interpolate missing sensor values
    # --------------------------------------------------------

    missing_before = data[SENSOR_COLUMNS].isna().sum().sum()

    if missing_before > 0:

        print(
            f"Missing sensor values before "
            f"interpolation: {missing_before:,}"
        )

        # Interpolate within each experiment
        data[SENSOR_COLUMNS] = (
            data.groupby("source_file", group_keys=False)[
                SENSOR_COLUMNS
            ]
            .apply(
                lambda group: group.interpolate(
                    method="linear",
                    limit_direction="both"
                )
            )
        )

    missing_after = data[SENSOR_COLUMNS].isna().sum().sum()

    print(
        f"Missing sensor values after "
        f"interpolation: {missing_after:,}"
    )

    # --------------------------------------------------------
    # 6. Final chronological sorting
    # --------------------------------------------------------

    data = data.sort_values(
        ["source_file", "datetime"]
    ).reset_index(drop=True)

    print(
        f"Output rows: {len(data):,}"
    )

    print("Preprocessing complete.")

    return data


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

def save_processed_data(
    df: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR
) -> Path:
    """
    Save the processed dataset as a Parquet file.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = output_dir / "skab_processed.parquet"

    df.to_parquet(
        output_path,
        index=False
    )

    print(
        f"\nProcessed dataset saved to:"
    )

    print(output_path)

    return output_path


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    from src.data.loader import load_all_data

    print("=" * 60)
    print("SKAB DATA PREPROCESSOR")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    print("\nLoading raw SKAB data...")

    raw_df = load_all_data()

    print(
        f"\nRaw dataset shape: {raw_df.shape}"
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_df = preprocess_data(
        raw_df
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING VALIDATION")
    print("=" * 60)

    print(
        f"\nFinal rows    : {len(processed_df):,}"
    )

    print(
        f"Final columns : {len(processed_df.columns)}"
    )

    print("\nMissing values:")

    print(
        processed_df.isnull().sum()
    )

    print("\nSensor statistics:")

    print(
        processed_df[SENSOR_COLUMNS].describe()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_processed_data(
        processed_df
    )

    print("\n" + "=" * 60)
    print("PREPROCESSOR TEST COMPLETE")
    print("=" * 60)