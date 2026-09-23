"""
Automated Test Suite for Bird Migration Collision Risk Prediction Pipeline
--------------------------------------------------------------------------
Tests:
  1. Data Integrity & Cleaning (all 4 datasets + streaming migration outputs)
  2. Master ML Dataset formulation (zero leakage, valid coordinates, balanced target)
  3. Git Safety Rule (.gitignore excluding 11.6 GB file)
  4. Model Artifacts & Loading (all 9 models + preprocessor)
  5. Backend Prediction API functionality
"""

import os
import subprocess
import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "trained_models"

def test_data_files_exist():
    """Verify all cleaned and processed datasets exist on disk."""
    required_files = [
        PROCESSED_DIR / "bird_strikes_v2_NY_IL_CO_clean.csv",
        PROCESSED_DIR / "airports_v2_NY_IL_CO_clean.csv",
        PROCESSED_DIR / "wind_turbines_v2_NY_IL_CO_clean.csv",
        PROCESSED_DIR / "weather_daily_station_aggregates.csv",
        PROCESSED_DIR / "migration_monthly_density_2021_2025.csv",
        PROCESSED_DIR / "migration_sample_clean.csv",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required data file missing: {rf}"
        assert rf.stat().st_size > 0, f"File is empty: {rf}"

def test_git_safety_rule():
    """CRITICAL: Test that the 11.6 GB migration CSV is strictly ignored by Git."""
    gitignore_path = BASE_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore file must exist in repository root."

    content = gitignore_path.read_text()
    assert "migration_US_NY_IL_CO_2021_2025.csv" in content, "11.6 GB file must be in .gitignore"

    # Run git check-ignore command
    cmd = ["git", "check-ignore", "data/processed/migration_US_NY_IL_CO_2021_2025.csv"]
    res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
    assert res.returncode == 0, "Git check-ignore must return 0 (file is ignored)"
    assert "migration_US_NY_IL_CO_2021_2025.csv" in res.stdout

def test_ml_dataset_formulation():
    """Verify Master ML Dataset for zero leakage and valid target."""
    ml_file = PROCESSED_DIR / "bird_collision_model_dataset_FINAL_2021_2025.csv"
    assert ml_file.exists(), f"Final ML dataset missing: {ml_file}"

    df = pd.read_csv(ml_file)
    assert len(df) > 10000, f"Expected >10,000 samples, got {len(df)}"

    # Check target
    assert "risk_target" in df.columns, "risk_target column must exist."
    classes = set(df["risk_target"].unique())
    assert classes == {0, 1, 2}, f"Target classes must be {{0, 1, 2}}, got {classes}"

    # Verify no post-event outcome leakage
    leakage_columns = [
        "damage", "cost", "effect", "people_injured", "remains_collected",
        "remains_sent_to_smithsonian", "remarks", "risk_index", "record_id"
    ]
    for col in leakage_columns:
        assert col not in df.columns, f"Data leakage detected! Column '{col}' should not be in ML dataset."

    # Verify coordinates are bounded to NY, IL, CO
    assert df["latitude"].between(36.0, 46.0).all(), "Invalid latitude coordinates found."
    assert df["longitude"].between(-112.0, -71.0).all(), "Invalid longitude coordinates found."

def test_model_artifacts_loading():
    """Verify that preprocessor and all 9 models exist and can be loaded."""
    prep_path = MODELS_DIR / "preprocessor.joblib"
    if not prep_path.exists():
        pytest.skip("Models not yet finished training.")

    preprocessor = joblib.load(prep_path)
    assert preprocessor is not None

    expected_models = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "k_nearest_neighbors.joblib",
        "random_forest.joblib",
        "gaussian_naive_bayes.joblib",
        "support_vector_machine.joblib",
        "gradient_boosting.joblib",
        "mlp_classifier.joblib",
        "xgboost.joblib",
        "best_model.joblib",
    ]

    for m_file in expected_models:
        path = MODELS_DIR / m_file
        assert path.exists(), f"Model artifact missing: {m_file}"
        clf = joblib.load(path)
        assert clf is not None
        assert hasattr(clf, "predict")

def test_backend_api_predict():
    """Test FastAPI prediction endpoint directly."""
    from backend.app import app
    from fastapi.testclient import TestClient

    prep_path = MODELS_DIR / "preprocessor.joblib"
    if not prep_path.exists():
        pytest.skip("Models not yet finished training.")

    client = TestClient(app)

    # Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # Prediction test
    payload = {
        "model_name": "Random Forest",
        "origin_state_abbr": "CO",
        "airport_name": "DENVER INTL AIRPORT",
        "flight_month": 9,
        "flight_year": 2024,
        "altitude": 1200.0,
        "flight_phase": "Approach",
        "wildlife_size": "Large",
        "conditions_sky": "Overcast",
        "conditions_precipitation": "Rain",
        "pilot_warned": "No",
        "engines": 2.0,
        "nearest_turbine_km": 8.5,
        "avg_temp_c": 15.0,
        "avg_humidity_pct": 65.0,
        "avg_wind_speed_kmh": 25.0,
        "avg_visibility_km": 10.0,
        "total_precip_mm": 5.0,
        "avg_pressure_hpa": 830.0
    }

    res_pred = client.post("/api/predict", json=payload)
    assert res_pred.status_code == 200
    data = res_pred.json()
    assert "risk_target" in data
    assert data["risk_target"] in [0, 1, 2]
    assert "risk_level" in data
    assert "class_probabilities" in data
    assert "contributing_factors" in data
    assert len(data["contributing_factors"]) > 0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
