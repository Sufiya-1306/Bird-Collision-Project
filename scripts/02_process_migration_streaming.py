"""
Script 02: Stream Processing for Large Bird Migration Dataset (11.6 GB)
------------------------------------------------------------------------
Streams data/processed/migration_US_NY_IL_CO_2021_2025.csv in chunks
without loading the entire 11.6 GB file into memory.

Filters:
  - States: New York (NY), Illinois (IL), Colorado (CO)
  - Years: 2021 - 2025
  - Valid geographic coordinates (within bounding box for target states)
  - Valid species names

Generates two project-friendly outputs:
  1. data/processed/migration_monthly_density_2021_2025.csv
     Aggregated monthly & seasonal bird migration metrics (observation count,
     species richness, migration density index) by state and month.
  2. data/processed/migration_sample_clean.csv
     A high-quality, representative clean sample (~150,000 records) with
     compact, essential columns for spatial analysis and species lookups.

NEVER modifies or deletes the original 11.6 GB source file.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/migration_US_NY_IL_CO_2021_2025.csv")
OUTPUT_DENSITY_FILE = Path("data/processed/migration_monthly_density_2021_2025.csv")
OUTPUT_SAMPLE_FILE = Path("data/processed/migration_sample_clean.csv")

TARGET_STATES = {
    "New York": "NY",
    "Illinois": "IL",
    "Colorado": "CO",
}

USE_COLS = [
    "species",
    "family",
    "order",
    "stateProvince",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "day",
    "month",
    "year",
]

CHUNK_SIZE = 500_000
SAMPLE_FRACTION = 0.005  # ~0.5% sample produces ~200k rows, perfect size (~20-25MB)
RANDOM_SEED = 42

def process_migration_stream():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    print("=" * 80)
    print("STREAM PROCESSING 11.6 GB MIGRATION DATASET")
    print(f"Source: {INPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")
    print("=" * 80)

    # Dictionary to aggregate monthly statistics:
    # key: (state_abbr, year, month) -> dict with count and species set
    monthly_stats = {}

    total_read = 0
    total_valid = 0
    sample_records = []
    rng = np.random.default_rng(RANDOM_SEED)

    first_sample_chunk = True

    for chunk_idx, chunk in enumerate(pd.read_csv(INPUT_FILE, usecols=USE_COLS, chunksize=CHUNK_SIZE, low_memory=False)):
        chunk_len = len(chunk)
        total_read += chunk_len

        # 1. State filter
        chunk = chunk[chunk["stateProvince"].isin(TARGET_STATES.keys())].copy()
        if chunk.empty:
            continue

        chunk["state_abbr"] = chunk["stateProvince"].map(TARGET_STATES)

        # 2. Year filter (2021-2025)
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce")
        chunk = chunk[chunk["year"].between(2021, 2025)]

        # 3. Month filter
        chunk["month"] = pd.to_numeric(chunk["month"], errors="coerce")
        chunk = chunk[chunk["month"].between(1, 12)]

        # 4. Species validity
        chunk = chunk[chunk["species"].notna() & (chunk["species"].astype(str).str.strip() != "")]

        # 5. Geographic validation (bounding box for CO, IL, NY)
        chunk["decimalLatitude"] = pd.to_numeric(chunk["decimalLatitude"], errors="coerce")
        chunk["decimalLongitude"] = pd.to_numeric(chunk["decimalLongitude"], errors="coerce")
        chunk = chunk[
            chunk["decimalLatitude"].between(36.0, 46.0) &
            chunk["decimalLongitude"].between(-112.0, -71.0)
        ]

        if chunk.empty:
            continue

        valid_in_chunk = len(chunk)
        total_valid += valid_in_chunk

        # Aggregate monthly stats
        # Group by state_abbr, year, month
        grouped = chunk.groupby(["state_abbr", "year", "month"])
        for (st, yr, mo), grp in grouped:
            key = (st, int(yr), int(mo))
            if key not in monthly_stats:
                monthly_stats[key] = {
                    "count": 0,
                    "species_set": set(),
                }
            monthly_stats[key]["count"] += len(grp)
            # Add species to set (sample up to 200 per group per chunk to keep memory tiny)
            chunk_species = grp["species"].dropna().unique()
            monthly_stats[key]["species_set"].update(chunk_species[:200])

        # Sample for project-friendly clean sample dataset
        # Random sample ~0.5% from this chunk
        sample_mask = rng.random(valid_in_chunk) < SAMPLE_FRACTION
        sub_sample = chunk[sample_mask].copy()

        if not sub_sample.empty:
            # Rename columns to clean snake_case
            clean_sub = sub_sample.rename(columns={
                "stateProvince": "state_name",
                "decimalLatitude": "latitude",
                "decimalLongitude": "longitude",
                "eventDate": "event_date",
            })
            clean_sub.to_csv(
                OUTPUT_SAMPLE_FILE,
                mode="w" if first_sample_chunk else "a",
                header=first_sample_chunk,
                index=False
            )
            first_sample_chunk = False

        print(f"Chunk {chunk_idx + 1:3d} | Rows read: {total_read:10,d} | Valid kept: {total_valid:10,d}")

    # Build monthly density dataframe
    rows = []
    for (st, yr, mo), data in monthly_stats.items():
        # Determine season
        if mo in [12, 1, 2]:
            season = "Winter"
        elif mo in [3, 4, 5]:
            season = "Spring"
        elif mo in [6, 7, 8]:
            season = "Summer"
        else:
            season = "Fall"

        rows.append({
            "state_abbr": st,
            "year": yr,
            "month": mo,
            "season": season,
            "migration_observation_count": data["count"],
            "species_richness": len(data["species_set"]),
        })

    df_density = pd.DataFrame(rows)
    # Calculate relative migration density per state (scaled 0-1)
    df_density["migration_density"] = df_density.groupby("state_abbr")["migration_observation_count"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-6)
    ).round(4)

    # Flag peak migration months (density >= 0.6)
    df_density["is_peak_migration"] = (df_density["migration_density"] >= 0.6).astype(int)

    df_density = df_density.sort_values(["state_abbr", "year", "month"]).reset_index(drop=True)
    df_density.to_csv(OUTPUT_DENSITY_FILE, index=False)

    print("\n" + "=" * 80)
    print("STREAM PROCESSING COMPLETED SUCCESSFULLY")
    print(f"Total lines processed:     {total_read:,}")
    print(f"Total valid records:       {total_valid:,}")
    print(f"Monthly density summary:   {OUTPUT_DENSITY_FILE} ({len(df_density)} rows)")
    if OUTPUT_SAMPLE_FILE.exists():
        sample_df = pd.read_csv(OUTPUT_SAMPLE_FILE)
        print(f"Clean occurrences sample:  {OUTPUT_SAMPLE_FILE} ({len(sample_df):,} rows, {os.path.getsize(OUTPUT_SAMPLE_FILE)/(1024*1024):.2f} MB)")
    print("=" * 80)

if __name__ == "__main__":
    process_migration_stream()
