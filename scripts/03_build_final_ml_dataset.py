"""
Script 03: Build Final Machine Learning Dataset (2021-2025)
------------------------------------------------------------
Constructs the unified, modeling-ready dataset:
  1. Links bird strikes (2021-2025) to exact airport coordinates and elevation.
  2. Computes Haversine distance to nearest wind turbine within the state.
  3. Joins daily station weather aggregates (KJFK for NY, KORD for IL, KDEN for CO).
  4. Joins monthly bird migration density and seasonal indicators.
  5. Computes transparent quantile-based risk index and target:
       risk_target:
         0 = Low Risk
         1 = Medium Risk
         2 = High Risk
  6. Strictly eliminates post-collision leakage variables.

Output:
  data/processed/bird_collision_model_dataset_FINAL_2021_2025.csv
"""

import pandas as pd
import numpy as np
import difflib
import re
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
OUTPUT_ML_FILE = PROCESSED_DIR / "bird_collision_model_dataset_FINAL_2021_2025.csv"

# Predefined exact coordinates for airports in NY, IL, CO
AIRPORT_COORDINATES = {
    # Colorado
    ("DENVER INTL AIRPORT", "CO"): (39.861698, -104.672997, 5431.0),
    ("CITY OF COLORADO SPRINGS MUNI APRT", "CO"): (38.805801, -104.700996, 6187.0),
    ("CENTENNIAL ARPT", "CO"): (39.5701, -104.849, 5885.0),
    ("ROCKY MOUNTAIN METROPOLITAN ARPT", "CO"): (39.908802, -105.116997, 5673.0),
    ("ASPEN-PITKIN COUNTY ARPT/SARDY FIELD", "CO"): (39.2232, -106.869003, 7820.0),
    ("PUEBLO MEMORIAL ARPT", "CO"): (38.289101, -104.497002, 4729.0),
    ("GRAND JUNCTION REGIONAL ARPT", "CO"): (39.122402, -108.527, 4858.0),
    ("FORT COLLINS-LOVELAND MUNI ARPT", "CO"): (40.451801, -105.011002, 5016.0),
    ("MONTROSE REGIONAL ARPT", "CO"): (38.509799, -107.893997, 5759.0),
    ("YAMPA VALLEY ARPT", "CO"): (40.481201, -107.218002, 6606.0),
    ("EAGLE COUNTY REGIONAL ARPT", "CO"): (39.642601, -106.917999, 6548.0),
    ("DURANGO-LA PLATA COUNTY ARPT", "CO"): (37.151501, -107.753998, 6689.0),
    ("GUNNISON-CRESTED BUTTE REGIONAL ARPT", "CO"): (38.534698, -106.932999, 7680.0),
    ("SAN LUIS VALLEY REGIONAL ARPT", "CO"): (37.435001, -105.866997, 7539.0),
    ("NORTHERN COLORADO REGIONAL ARPT", "CO"): (40.451801, -105.011002, 5016.0),
    ("BOULDER MUNICIPAL ARPT", "CO"): (40.039402, -105.225998, 5288.0),
    ("GREELEY-WELD COUNTY ARPT", "CO"): (40.4375, -104.633003, 4697.0),
    ("TELLURIDE REGIONAL ARPT", "CO"): (37.953800, -107.907997, 9070.0),

    # Illinois
    ("CHICAGO O'HARE INTL ARPT", "IL"): (41.9786, -87.9048, 680.0),
    ("CHICAGO MIDWAY INTL ARPT", "IL"): (41.785999, -87.752197, 620.0),
    ("CHICAGO/ROCKFORD INTL ARPT", "IL"): (42.1954, -89.097198, 742.0),
    ("PEORIA INTL ARPT", "IL"): (40.664167, -89.693333, 661.0),
    ("QUAD CITY ARPT", "IL"): (41.4485, -90.5075, 590.0),
    ("UNIV OF ILLINOIS -WILLARD ARPT", "IL"): (40.0392, -88.2781, 755.0),
    ("CENTRAL ILLINOIS REGIONAL ARPT", "IL"): (40.4771, -88.915901, 871.0),
    ("ABRAHAM LINCOLN CAPITAL ARPT", "IL"): (39.844101, -89.677803, 598.0),
    ("WILLIAMSON COUNTY REGIONAL ARPT", "IL"): (37.7548, -89.0111, 472.0),
    ("WILLIAMSON COUNTY REGIONAL", "IL"): (37.7548, -89.0111, 472.0),
    ("DECATUR ARPT", "IL"): (39.8346, -88.8656, 680.0),
    ("ST LOUIS DOWNTOWN ARPT", "IL"): (38.5714, -90.1558, 413.0),
    ("ST LOUIS REGIONAL ARPT", "IL"): (38.8903, -90.0461, 544.0),
    ("AURORA MUNICIPAL ARPT", "IL"): (41.7719, -88.4756, 712.0),
    ("DUPAGE ARPT", "IL"): (41.9078, -88.2486, 759.0),
    ("WAUKEGAN NATIONAL ARPT", "IL"): (42.4222, -87.8678, 727.0),

    # New York
    ("JOHN F KENNEDY INTL", "NY"): (40.639801, -73.7789, 13.0),
    ("LA GUARDIA ARPT", "NY"): (40.777199, -73.872597, 21.0),
    ("ALBANY INTL", "NY"): (42.748299, -73.801697, 285.0),
    ("WESTCHESTER COUNTY ARPT", "NY"): (41.066998, -73.707573, 439.0),
    ("BUFFALO-NIAGARA INTL", "NY"): (42.940498, -78.732201, 728.0),
    ("GREATER ROCHESTER INTL", "NY"): (43.1189, -77.672401, 559.0),
    ("SYRACUSE HANCOCK INTL", "NY"): (43.111198, -76.1063, 421.0),
    ("LONG ISLAND MAC ARTHUR", "NY"): (40.7952, -73.1002, 99.0),
    ("NEW YORK STEWART INTL ARPT", "NY"): (41.5041, -74.1048, 491.0),
    ("STEWART INTL", "NY"): (41.5041, -74.1048, 491.0),
    ("REPUBLIC ARPT", "NY"): (40.7288, -73.4134, 82.0),
    ("ITHACA TOMPKINS INTL ARPT", "NY"): (42.4914, -76.4584, 1099.0),
    ("GREATER BINGHAMTON ARPT", "NY"): (42.2086, -75.9798, 1636.0),
    ("CHAUTAUQUA COUNTY/JAMESTOWN ARPT", "NY"): (42.1534, -79.2581, 1723.0),
    ("WATERTOWN INTL ARPT", "NY"): (43.9919, -76.0217, 325.0),
    ("PLATTSBURGH INTL ARPT", "NY"): (44.6509, -73.4681, 371.0),
    ("OGDENSBURG INTL ARPT", "NY"): (44.6819, -75.4655, 297.0),
    ("ADIRONDACK REGIONAL ARPT", "NY"): (44.3853, -74.2062, 1663.0),
    ("NIAGARA FALLS INTL", "NY"): (43.1073, -78.9462, 535.0),
    ("ELMIRA/CORNING REGIONAL ARPT", "NY"): (42.1599, -76.8914, 955.0),
    ("NYPD AIR OPS HELIPORT -PVT", "NY"): (40.5905, -73.8927, 10.0),
}

def normalize_name(s):
    s = str(s).upper()
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = s.split()
    mapping = {
        "ARPT": "AIRPORT",
        "APRT": "AIRPORT",
        "INTL": "INTERNATIONAL",
        "MUNI": "MUNICIPAL",
        "FLD": "FIELD",
        "CTR": "CENTER",
        "IS": "ISLAND",
    }
    tokens = [mapping.get(t, t) for t in tokens]
    return " ".join(tokens)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def assign_airport_coordinates(df_strikes, df_airports):
    print("Assigning airport coordinates...")
    df_airports["norm_name"] = df_airports["airport_name"].apply(normalize_name)
    
    # Fast lookup dictionaries by state
    state_lookup = {}
    for st in ["NY", "IL", "CO"]:
        sub = df_airports[df_airports["state_abbr"] == st]
        state_lookup[st] = dict(zip(sub["norm_name"], zip(sub["latitude"], sub["longitude"], sub["elevation_ft"])))

    lats, lons, elevs = [], [], []

    for _, row in df_strikes.iterrows():
        name = row["airport_name"]
        st = row["origin_state_abbr"]
        
        # 1. Exact preset
        if (name, st) in AIRPORT_COORDINATES:
            c = AIRPORT_COORDINATES[(name, st)]
            lats.append(c[0])
            lons.append(c[1])
            elevs.append(c[2])
            continue
            
        # 2. Normalized match in state
        norm = normalize_name(name)
        if st in state_lookup and norm in state_lookup[st]:
            c = state_lookup[st][norm]
            lats.append(c[0])
            lons.append(c[1])
            elevs.append(c[2])
            continue
            
        # 3. Fuzzy match in state
        names_in_state = list(state_lookup.get(st, {}).keys())
        close = difflib.get_close_matches(norm, names_in_state, n=1, cutoff=0.6)
        if close:
            c = state_lookup[st][close[0]]
            lats.append(c[0])
            lons.append(c[1])
            elevs.append(c[2])
        else:
            # Fallback to state median
            st_sub = df_airports[df_airports["state_abbr"] == st]
            lats.append(st_sub["latitude"].median())
            lons.append(st_sub["longitude"].median())
            elevs.append(st_sub["elevation_ft"].median())

    df_strikes["latitude"] = lats
    df_strikes["longitude"] = lons
    df_strikes["elevation_ft"] = elevs
    return df_strikes

def compute_nearest_turbines(df_strikes, df_turbines):
    print("Computing nearest wind turbine distance (km)...")
    turbine_dict = {}
    for st in ["NY", "IL", "CO"]:
        sub = df_turbines[df_turbines["state_abbr"] == st]
        turbine_dict[st] = (sub["latitude"].values, sub["longitude"].values)

    distances = []
    for _, row in df_strikes.iterrows():
        st = row["origin_state_abbr"]
        lat = row["latitude"]
        lon = row["longitude"]
        
        t_lats, t_lons = turbine_dict.get(st, (np.array([]), np.array([])))
        if len(t_lats) > 0:
            dists = haversine(lat, lon, t_lats, t_lons)
            distances.append(round(float(dists.min()), 2))
        else:
            distances.append(50.0)

    df_strikes["nearest_turbine_km"] = distances
    return df_strikes

def build_ml_dataset():
    print("=" * 80)
    print("STEP 2: BUILDING FINAL ML DATASET (2021-2025)")
    print("=" * 80)

    # 1. Load cleaned datasets
    bs = pd.read_csv(PROCESSED_DIR / "bird_strikes_v2_NY_IL_CO_clean.csv")
    ap = pd.read_csv(PROCESSED_DIR / "airports_v2_NY_IL_CO_clean.csv")
    wt = pd.read_csv(PROCESSED_DIR / "wind_turbines_v2_NY_IL_CO_clean.csv")
    weather = pd.read_csv(PROCESSED_DIR / "weather_daily_station_aggregates.csv")

    # 2. Add coordinates & elevation
    bs = assign_airport_coordinates(bs, ap)

    # 3. Add nearest turbine distance
    bs = compute_nearest_turbines(bs, wt)

    # 4. Merge daily weather
    print("Merging daily station weather...")
    bs["flight_date"] = pd.to_datetime(bs["flight_date"])
    weather["date"] = pd.to_datetime(weather["date"])

    master = bs.merge(
        weather,
        left_on=["origin_state_abbr", "flight_date"],
        right_on=["state_abbr", "date"],
        how="left"
    )

    # Impute any missing weather with monthly state medians
    for col in ["avg_temp_c", "avg_humidity_pct", "avg_wind_speed_kmh", "avg_visibility_km", "total_precip_mm", "avg_pressure_hpa"]:
        if col in master.columns and master[col].isna().sum() > 0:
            master[col] = master.groupby(["origin_state_abbr", "flight_month"])[col].transform(
                lambda s: s.fillna(s.median())
            )

    # 5. Merge bird migration density
    density_file = PROCESSED_DIR / "migration_monthly_density_2021_2025.csv"
    if density_file.exists():
        print("Merging bird migration monthly density...")
        mig_density = pd.read_csv(density_file)
        master = master.merge(
            mig_density,
            left_on=["origin_state_abbr", "flight_year", "flight_month"],
            right_on=["state_abbr", "year", "month"],
            how="left",
            suffixes=("", "_mig")
        )
    else:
        print("Monthly migration density file not found yet; calculating default seasonal density...")
        master["migration_density"] = master["flight_month"].map({
            9: 1.0, 10: 0.95, 5: 0.90, 4: 0.85, 11: 0.70, 3: 0.65,
            8: 0.60, 6: 0.40, 7: 0.35, 12: 0.25, 1: 0.20, 2: 0.20
        }).fillna(0.5)
        master["is_peak_migration"] = (master["migration_density"] >= 0.7).astype(int)

    # Fill season if missing
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    if "season" not in master.columns or master["season"].isna().sum() > 0:
        master["season"] = master["flight_month"].apply(get_season)

    # 6. Engineer the Target Variable: risk_target (0 = Low, 1 = Medium, 2 = High)
    print("Calculating composite quantile risk index and target...")
    def qscore(s):
        q1, q2 = s.quantile([1/3, 2/3])
        if q1 == q2:
            return pd.Series(1, index=s.index)
        return pd.Series(np.select([s <= q1, s <= q2], [0, 1], default=2), index=s.index)

    # Component scores
    count = pd.to_numeric(master["number_struck_actual"], errors="coerce").fillna(1)
    count_score = qscore(count)

    size_map = {"Small": 0, "Unknown": 1, "Medium": 1, "Large": 2}
    size_score = master["wildlife_size"].map(size_map).fillna(1)

    mig_score = (master["migration_density"] * 2.0).round().clip(0, 2)

    alt = pd.to_numeric(master["altitude"], errors="coerce").fillna(0)
    alt_q1, alt_q2 = alt.quantile(1/3), alt.quantile(2/3)
    alt_score = pd.Series(np.select([alt <= alt_q1, alt <= alt_q2], [2, 1], default=0), index=master.index)

    phase_map = {
        "Take-off Run": 2, "Climb": 2, "Approach": 2, "Landing Roll": 2, "Arrival": 2, "Departure": 2,
        "Descent": 1, "Local": 1, "En Route": 0, "Taxi": 0, "Parked": 0, "Unknown": 1
    }
    phase_score = master["flight_phase"].map(phase_map).fillna(1)

    wind = pd.to_numeric(master["avg_wind_speed_kmh"], errors="coerce").fillna(master["avg_wind_speed_kmh"].median())
    wind_score = qscore(wind)

    vis = pd.to_numeric(master["avg_visibility_km"], errors="coerce").fillna(master["avg_visibility_km"].median())
    vq1, vq2 = vis.quantile([1/3, 2/3])
    vis_score = pd.Series(np.select([vis <= vq1, vis <= vq2], [2, 1], default=0), index=master.index)

    precip = pd.to_numeric(master["total_precip_mm"], errors="coerce").fillna(0)
    precip_score = (precip > 0).astype(int)

    dist = pd.to_numeric(master["nearest_turbine_km"], errors="coerce")
    dq1, dq2 = dist.quantile([1/3, 2/3])
    turbine_score = pd.Series(np.select([dist <= dq1, dist <= dq2], [2, 1], default=0), index=master.index)

    # Transparent composite risk index (0.0 to 2.0)
    risk_index = (
        0.20 * count_score +
        0.10 * size_score +
        0.15 * mig_score +
        0.15 * alt_score +
        0.10 * phase_score +
        0.08 * wind_score +
        0.07 * vis_score +
        0.05 * precip_score +
        0.10 * turbine_score
    )

    r1, r2 = risk_index.quantile([1/3, 2/3])
    master["risk_index"] = risk_index.round(4)
    master["risk_level"] = np.select([risk_index <= r1, risk_index <= r2], ["Low", "Medium"], default="High")
    master["risk_target"] = master["risk_level"].map({"Low": 0, "Medium": 1, "High": 2}).astype(int)

    print("\nTarget Class Distribution:")
    print(master["risk_level"].value_counts())
    print("Class Percentages:")
    print(master["risk_level"].value_counts(normalize=True).round(3) * 100)

    # 7. Drop Post-Event Leakage and Redundant Columns
    # Columns that describe collision consequences (damage, cost, effect, injuries, remains)
    # or direct target leakage (risk_index, risk_level, number_struck, number_struck_actual)
    leakage_cols = [
        "record_id", "remarks", "cost", "damage", "effect", "people_injured",
        "remains_collected", "remains_sent_to_smithsonian", "make_model",
        "number_struck", "number_struck_actual", "risk_index", "risk_level",
        "date", "state_abbr", "state_abbr_mig", "year", "month", "Station_ID",
        "migration_observation_count", "species_richness"
    ]
    drop_cols = [c for c in leakage_cols if c in master.columns]
    df_ml = master.drop(columns=drop_cols).copy()

    # Re-verify duplicates and drop if any
    before_dup = len(df_ml)
    df_ml = df_ml.drop_duplicates()
    print(f"Removed {before_dup - len(df_ml)} duplicates after dropping ID/post-event fields.")

    # Save final ML dataset
    df_ml.to_csv(OUTPUT_ML_FILE, index=False)
    print("\n" + "=" * 80)
    print(f"FINAL ML DATASET CREATED: {OUTPUT_ML_FILE}")
    print(f"Final shape: {df_ml.shape[0]:,} rows x {df_ml.shape[1]} columns")
    print(f"Features: {df_ml.columns.tolist()}")
    print(f"Target: risk_target (0=Low, 1=Medium, 2=High)")
    print("=" * 80)
    return df_ml

if __name__ == "__main__":
    build_ml_dataset()
