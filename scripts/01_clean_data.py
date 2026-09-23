"""
Script 01: Final Data Cleaning for Processed Datasets
-----------------------------------------------------
Performs rigorous hygiene, validation, and standardization on:
  1. bird_strikes_v2_NY_IL_CO_clean.csv
  2. airports_v2_NY_IL_CO_clean.csv
  3. cleaned_weather_v2.csv (and produces daily station weather aggregates)
  4. wind_turbines_v2_NY_IL_CO_clean.csv

Ensures reproducible, clean inputs ready for merging and machine learning.
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

def clean_bird_strikes():
    file_path = PROCESSED_DIR / "bird_strikes_v2_NY_IL_CO_clean.csv"
    print(f"\n--- Cleaning Bird Strikes: {file_path.name} ---")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Drop duplicate records
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped {dup_count} duplicate rows")

    # Validate dates & years (2021-2025)
    df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")
    df = df.dropna(subset=["flight_date"])
    df["flight_year"] = df["flight_date"].dt.year
    df["flight_month"] = df["flight_date"].dt.month
    df = df[df["flight_year"].between(2021, 2025)]

    # Clean precipitation string artifacts
    precip_clean_map = {
        "None, Snow": "Snow",
        "Fog, None": "Fog",
        "Fog, Snow": "Fog, Snow",
        "Fog, Rain": "Fog, Rain",
        "Rain, Snow": "Rain, Snow",
        "No Precipitation": "No Precipitation",
        "Rain": "Rain",
        "Snow": "Snow",
        "Fog": "Fog",
    }
    df["conditions_precipitation"] = df["conditions_precipitation"].astype(str).str.strip()
    df["conditions_precipitation"] = df["conditions_precipitation"].replace(precip_clean_map)
    df["conditions_precipitation"] = df["conditions_precipitation"].fillna("No Precipitation")

    # Standardize states
    df["origin_state_abbr"] = df["origin_state_abbr"].str.strip().str.upper()
    df = df[df["origin_state_abbr"].isin(["NY", "IL", "CO"])]

    # Ensure numeric columns are strictly positive/valid
    df["engines"] = pd.to_numeric(df["engines"], errors="coerce").fillna(2.0).clip(1, 4)
    df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce").fillna(0.0).clip(0, 50000)
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce").fillna(0.0).clip(lower=0)
    df["number_struck_actual"] = pd.to_numeric(df["number_struck_actual"], errors="coerce").fillna(1.0).clip(lower=1)
    df["people_injured"] = pd.to_numeric(df["people_injured"], errors="coerce").fillna(0.0).clip(lower=0)

    # Standardize categoricals
    for col in ["wildlife_size", "flight_phase", "conditions_sky", "pilot_warned"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    df.to_csv(file_path, index=False)
    print(f"Cleaned shape: {df.shape}")
    return df

def clean_airports():
    file_path = PROCESSED_DIR / "airports_v2_NY_IL_CO_clean.csv"
    print(f"\n--- Cleaning Airports: {file_path.name} ---")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {df.shape}")

    # Drop duplicate records
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped {dup_count} duplicate rows")

    # Filter operational airports in target states
    df = df[df["type"] != "closed"].copy()
    df["state_abbr"] = df["state_abbr"].str.strip().str.upper()
    df = df[df["state_abbr"].isin(["NY", "IL", "CO"])]

    # Coordinate validation
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(36.0, 46.0) & df["longitude"].between(-112.0, -71.0)]

    # Clean text columns
    df["airport_name_clean"] = df["airport_name"].astype(str).str.upper().str.strip()
    df["elevation_ft"] = pd.to_numeric(df["elevation_ft"], errors="coerce").fillna(df["elevation_ft"].median())

    df = df.reset_index(drop=True)
    df.to_csv(file_path, index=False)
    print(f"Cleaned shape: {df.shape}")
    return df

def clean_weather():
    raw_weather_path = PROCESSED_DIR / "cleaned_weather_v2.csv"
    daily_weather_path = PROCESSED_DIR / "weather_daily_station_aggregates.csv"
    print(f"\n--- Cleaning Weather & Computing Daily Aggregates: {raw_weather_path.name} ---")
    df = pd.read_csv(raw_weather_path, low_memory=False)
    print(f"Hourly weather shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Parse timestamps
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["date"] = df["Timestamp"].dt.date

    # Station mapping to state
    station_to_state = {
        "KJFK_New_York": "NY",
        "KORD_Chicago": "IL",
        "KDEN_Denver": "CO",
    }
    df = df[df["Station_ID"].isin(station_to_state.keys())].copy()
    df["state_abbr"] = df["Station_ID"].map(station_to_state)

    # Fill dry precipitation nulls with 0.0
    df["Precipitation_mm"] = pd.to_numeric(df["Precipitation_mm"], errors="coerce").fillna(0.0)

    # Filter physical sensor outliers
    df["Temperature_C"] = pd.to_numeric(df["Temperature_C"], errors="coerce")
    df = df[df["Temperature_C"].isna() | df["Temperature_C"].between(-35, 50)]

    # Compute daily aggregates per station
    weather_daily = df.groupby(["Station_ID", "state_abbr", "date"]).agg(
        avg_temp_c=("Temperature_C", "mean"),
        avg_humidity_pct=("Humidity_Pct", "mean"),
        avg_wind_speed_kmh=("Wind_Speed_kmh", "mean"),
        avg_visibility_km=("Visibility_km", "mean"),
        total_precip_mm=("Precipitation_mm", "sum"),
        avg_pressure_hpa=("Pressure_hPa", "mean"),
    ).reset_index()

    # Impute any remaining missing daily aggregates using monthly state medians
    weather_daily["date"] = pd.to_datetime(weather_daily["date"])
    weather_daily["month"] = weather_daily["date"].dt.month

    for col in ["avg_temp_c", "avg_humidity_pct", "avg_wind_speed_kmh", "avg_visibility_km", "avg_pressure_hpa"]:
        weather_daily[col] = weather_daily[col].fillna(
            weather_daily.groupby(["state_abbr", "month"])[col].transform("median")
        )

    weather_daily = weather_daily.drop(columns=["month"])
    weather_daily.to_csv(daily_weather_path, index=False)
    print(f"Daily station aggregates saved: {daily_weather_path} ({weather_daily.shape})")
    print(f"Nulls in daily weather:\n{weather_daily.isnull().sum()}")
    return weather_daily

def clean_wind_turbines():
    file_path = PROCESSED_DIR / "wind_turbines_v2_NY_IL_CO_clean.csv"
    print(f"\n--- Cleaning Wind Turbines: {file_path.name} ---")
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Original shape: {df.shape}")

    # Standardize state
    df["state_abbr"] = df["state_abbr"].astype(str).str.strip().str.upper()
    df = df[df["state_abbr"].isin(["NY", "IL", "CO"])].copy()

    # Drop duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df = df.drop_duplicates()
        print(f"Dropped {dup_count} duplicate rows")

    # Coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[df["latitude"].between(36.0, 46.0) & df["longitude"].between(-112.0, -71.0)]

    # Keep essential operational columns
    keep_cols = [
        "turbine_id", "state_abbr", "county", "project_name", "year",
        "turbine_capacity", "t_hh", "t_rd", "t_rsa", "t_ttlh", "latitude", "longitude"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    df = df[existing].reset_index(drop=True)

    df.to_csv(file_path, index=False)
    print(f"Cleaned shape: {df.shape}")
    return df

def run_all_cleaners():
    print("=" * 80)
    print("STEP 1: FINAL DATA CLEANING OF 4 PROCESSED DATASETS")
    print("=" * 80)
    clean_bird_strikes()
    clean_airports()
    clean_weather()
    clean_wind_turbines()
    print("\nAll 4 processed datasets successfully cleaned and verified!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_cleaners()
