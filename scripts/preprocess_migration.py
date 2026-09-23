import zipfile
import pandas as pd
from pathlib import Path

ZIP_FILE = Path("data/raw/0014699-260903145123482.zip")
OUTPUT_FILE = Path("data/processed/migration_US_NY_IL_CO_2021_2025.csv")

TARGET_STATES = {"New York", "Illinois", "Colorado"}
TARGET_YEARS = set(range(2021, 2026))

print("Starting migration data processing...")
print("Reading raw ZIP without extracting it...")

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    csv_name = z.namelist()[0]

    print(f"Reading: {csv_name}")

    with z.open(csv_name) as f:
        first_chunk = True
        total_saved = 0

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                f,
                sep="\t",
                chunksize=100_000,
                low_memory=False
            )
        ):
            print(f"Processing chunk {chunk_number + 1}...")

            # USA only
            chunk = chunk[chunk["countryCode"] == "US"]

            # Required states
            chunk = chunk[chunk["stateProvince"].isin(TARGET_STATES)]

            # Required years
            chunk["year"] = pd.to_numeric(
                chunk["year"], errors="coerce"
            )

            chunk = chunk[chunk["year"].isin(TARGET_YEARS)]

            if not chunk.empty:
                # Keep useful columns for our project
                columns_to_keep = [
                    "gbifID",
                    "occurrenceID",
                    "species",
                    "scientificName",
                    "family",
                    "order",
                    "stateProvince",
                    "decimalLatitude",
                    "decimalLongitude",
                    "eventDate",
                    "day",
                    "month",
                    "year",
                    "individualCount",
                    "occurrenceStatus",
                    "basisOfRecord"
                ]

                existing_columns = [
                    col for col in columns_to_keep
                    if col in chunk.columns
                ]

                chunk[existing_columns].to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_chunk else "a",
                    header=first_chunk,
                    index=False
                )

                total_saved += len(chunk)
                first_chunk = False

            print(f"Records saved so far: {total_saved:,}")

print("\nProcessing complete!")
print(f"Output file: {OUTPUT_FILE}")
print(f"Total records saved: {total_saved:,}")