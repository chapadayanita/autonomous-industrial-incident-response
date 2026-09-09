from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root:
# agentic_ai_project/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# SKAB raw data location:
# agentic_ai_project/data/raw/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# SKAB EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "datetime",
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
    "anomaly",
    "changepoint",
]


# ============================================================
# FIND CSV FILES
# ============================================================

def find_csv_files(data_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """
    Find all CSV files inside the SKAB raw data directory.
    """

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {data_dir}"
        )

    csv_files = sorted(data_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside: {data_dir}"
        )

    return csv_files


# ============================================================
# LOAD ONE CSV FILE
# ============================================================

def load_single_file(file_path: Path) -> pd.DataFrame:
    """
    Load and validate a single SKAB CSV file.

    SKAB CSV files use ';' as the separator.
    """

    # SKAB uses semicolon-separated CSV files
    df = pd.read_csv(file_path, sep=";")

    # Check that all expected columns exist
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing columns: {missing_columns}"
        )

    # Convert datetime column
    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="coerce"
    )

    # Sort data chronologically
    df = df.sort_values(
        "datetime"
    ).reset_index(drop=True)

    # Store information about the source experiment
    df["source_file"] = file_path.name
    df["source_path"] = str(file_path)

    return df


# ============================================================
# LOAD ALL SKAB DATA
# ============================================================

def load_all_data(
    data_dir: Path = RAW_DATA_DIR
) -> pd.DataFrame:
    """
    Load all SKAB CSV experiment files
    into one Pandas DataFrame.
    """

    files = find_csv_files(data_dir)

    print(f"Found {len(files)} CSV files.")

    frames = []

    for file_path in files:

        try:
            df = load_single_file(file_path)

            frames.append(df)

            print(
                f"Loaded: {file_path.relative_to(PROJECT_ROOT)} "
                f"({len(df):,} rows)"
            )

        except Exception as error:

            print(
                f"Warning: could not load "
                f"{file_path}: {error}"
            )

    # Make sure at least one file was loaded
    if not frames:
        raise RuntimeError(
            "None of the SKAB CSV files could be loaded."
        )

    # Combine all experiments
    combined = pd.concat(
        frames,
        ignore_index=True
    )

    # Sort by experiment and time
    combined = combined.sort_values(
        ["source_file", "datetime"]
    ).reset_index(drop=True)

    return combined


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("SKAB DATA LOADER")
    print("=" * 60)

    print("\nLoading SKAB dataset...")

    df = load_all_data()

    print("\n" + "=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    # Basic information
    print(f"\nTotal rows     : {len(df):,}")
    print(f"Total columns  : {len(df.columns)}")
    print(
        f"Experiments    : {df['source_file'].nunique()}"
    )

    # Columns
    print("\nColumns:")

    for column in df.columns:
        print(f"  - {column}")

    # Anomaly distribution
    print("\nAnomaly distribution:")

    print(
        df["anomaly"].value_counts()
    )

    # Missing values
    print("\nMissing values:")

    print(
        df.isnull().sum()
    )

    # Preview
    print("\nFirst 5 rows:")

    print(
        df.head()
    )

    print("\n" + "=" * 60)
    print("LOADER TEST COMPLETE")
    print("=" * 60)