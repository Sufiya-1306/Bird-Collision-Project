"""
API Error & Edge-Case Test Suite
---------------------------------
Deliberately tests every possible error path in the Bird Collision Risk
Prediction API, including:

  1.  Missing required fields
  2.  Invalid state code
  3.  Out-of-range numeric values (altitude, engines, month, year)
  4.  Invalid field types (string where int expected)
  5.  Empty request body
  6.  Unknown model name (should fall back gracefully)
  7.  /api/health endpoint
  8.  /api/stats endpoint
  9.  /api/models endpoint
  10. /api/predict/compare-all endpoint edge cases
  11. Unknown HTTP route (404)
  12. Wrong HTTP method (405 Method Not Allowed)
  13. Extreme numeric values (very high altitude, zero visibility)
  14. All valid state codes (CO, IL, NY)
  15. All valid flight phases
  16. All valid wildlife sizes
  17. Precipitation + sky condition combinations
  18. Pilot-warned field variations (Yes, No, Unknown)
  19. All 9 model names individually
  20. Null/None optional fields (should use defaults)
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "trained_models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_client():
    from backend.app import app
    return TestClient(app)

def models_ready():
    return (MODELS_DIR / "preprocessor.joblib").exists()

def base_payload(**overrides):
    """Return a valid baseline payload, with optional field overrides."""
    payload = {
        "model_name": "Gradient Boosting",
        "origin_state_abbr": "CO",
        "airport_name": "DENVER INTL AIRPORT",
        "flight_month": 9,
        "flight_year": 2024,
        "altitude": 1500.0,
        "flight_phase": "Approach",
        "wildlife_size": "Medium",
        "conditions_sky": "No Cloud",
        "conditions_precipitation": "No Precipitation",
        "pilot_warned": "No",
        "engines": 2.0,
        "nearest_turbine_km": 15.5,
        "avg_temp_c": 16.5,
        "avg_humidity_pct": 55.0,
        "avg_wind_speed_kmh": 18.0,
        "avg_visibility_km": 16.0,
        "total_precip_mm": 0.0,
        "avg_pressure_hpa": 835.0,
    }
    payload.update(overrides)
    return payload


# ===========================================================================
# GROUP 1: Infrastructure / Health Endpoints
# ===========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /api/health must return 200 with status=healthy."""
        client = get_client()
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_health_has_required_fields(self):
        client = get_client()
        data = client.get("/api/health").json()
        for field in ["status", "service", "dataset_period", "models_trained"]:
            assert field in data, f"Missing field: {field}"

    def test_health_wrong_method_post(self):
        """POST to /api/health should return 405 Method Not Allowed."""
        client = get_client()
        r = client.post("/api/health")
        assert r.status_code == 405

    def test_health_wrong_method_put(self):
        client = get_client()
        r = client.put("/api/health")
        assert r.status_code == 405


class TestStatsEndpoint:
    def test_stats_returns_200(self):
        client = get_client()
        r = client.get("/api/stats")
        assert r.status_code == 200

    def test_stats_has_records_summary(self):
        client = get_client()
        data = client.get("/api/stats").json()
        assert "records_summary" in data
        assert "ml_training_samples" in data["records_summary"]

    def test_stats_has_target_distribution(self):
        client = get_client()
        data = client.get("/api/stats").json()
        assert "target_distribution" in data
        dist = data["target_distribution"]
        assert "Low Risk (0)" in dist
        assert "Medium Risk (1)" in dist
        assert "High Risk (2)" in dist

    def test_stats_wrong_method(self):
        client = get_client()
        r = client.post("/api/stats")
        assert r.status_code == 405


class TestModelsEndpoint:
    def test_models_returns_200(self):
        client = get_client()
        r = client.get("/api/models")
        assert r.status_code == 200

    def test_models_returns_9_entries(self):
        client = get_client()
        data = client.get("/api/models").json()
        assert len(data) == 9, f"Expected 9 models, got {len(data)}"

    def test_models_has_gradient_boosting(self):
        client = get_client()
        data = client.get("/api/models").json()
        assert "Gradient Boosting" in data

    def test_models_entries_have_metrics(self):
        client = get_client()
        data = client.get("/api/models").json()
        for name, info in data.items():
            assert "test_accuracy" in info, f"Missing test_accuracy for {name}"
            assert "f1_macro" in info, f"Missing f1_macro for {name}"


class TestUnknownRoutes:
    def test_unknown_api_route_404(self):
        """Unknown API route should return 404."""
        client = get_client()
        r = client.get("/api/nonexistent_endpoint")
        assert r.status_code == 404

    def test_unknown_api_post_404(self):
        """Unknown API POST route — the static file mount only supports GET,
        so an unmatched POST returns 405 (Method Not Allowed)."""
        client = get_client()
        r = client.post("/api/doesnotexist", json={})
        assert r.status_code in [404, 405]


# ===========================================================================
# GROUP 2: /api/predict — Valid Scenarios
# ===========================================================================

class TestPredictValidRequests:
    def test_predict_valid_payload_returns_200(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload())
        assert r.status_code == 200

    def test_predict_response_has_required_fields(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict", json=base_payload()).json()
        required = ["risk_target", "risk_level", "confidence", "class_probabilities",
                    "contributing_factors", "recommended_action", "model_used"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_predict_risk_target_is_valid_class(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict", json=base_payload()).json()
        assert data["risk_target"] in [0, 1, 2]

    def test_predict_confidence_in_range(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict", json=base_payload()).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_contributing_factors_not_empty(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict", json=base_payload()).json()
        assert len(data["contributing_factors"]) > 0

    def test_predict_all_state_codes(self):
        """CO, IL, and NY are the three valid states — all must work."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for state in ["CO", "IL", "NY"]:
            r = client.post("/api/predict", json=base_payload(origin_state_abbr=state))
            assert r.status_code == 200, f"Failed for state={state}"
            assert r.json()["risk_target"] in [0, 1, 2]

    def test_predict_all_flight_phases(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        phases = ["Approach", "Take-off Run", "Climb", "Descent", "En Route",
                  "Landing Roll", "Parked", "Taxi"]
        for phase in phases:
            r = client.post("/api/predict", json=base_payload(flight_phase=phase))
            assert r.status_code == 200, f"Failed for phase={phase}"

    def test_predict_all_wildlife_sizes(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for size in ["Small", "Medium", "Large"]:
            r = client.post("/api/predict", json=base_payload(wildlife_size=size))
            assert r.status_code == 200, f"Failed for size={size}"

    def test_predict_all_sky_conditions(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for sky in ["No Cloud", "Some Cloud", "Overcast"]:
            r = client.post("/api/predict", json=base_payload(conditions_sky=sky))
            assert r.status_code == 200, f"Failed for sky={sky}"

    def test_predict_all_precipitation_types(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for precip in ["No Precipitation", "Rain", "Snow", "Fog"]:
            r = client.post("/api/predict", json=base_payload(conditions_precipitation=precip))
            assert r.status_code == 200, f"Failed for precipitation={precip}"

    def test_predict_pilot_warned_variants(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for warned in ["Yes", "No", "Unknown"]:
            r = client.post("/api/predict", json=base_payload(pilot_warned=warned))
            assert r.status_code == 200, f"Failed for pilot_warned={warned}"

    def test_predict_all_12_months(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for month in range(1, 13):
            r = client.post("/api/predict", json=base_payload(flight_month=month))
            assert r.status_code == 200, f"Failed for flight_month={month}"

    def test_predict_all_valid_years(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        for year in [2021, 2022, 2023, 2024, 2025]:
            r = client.post("/api/predict", json=base_payload(flight_year=year))
            assert r.status_code == 200, f"Failed for flight_year={year}"

    def test_predict_all_9_model_names(self):
        """Every one of the 9 trained models must produce a valid prediction."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        model_names = [
            "Logistic Regression", "Decision Tree", "K-Nearest Neighbors",
            "Random Forest", "Gaussian Naive Bayes", "Support Vector Machine",
            "Gradient Boosting", "MLP Classifier", "XGBoost",
        ]
        for name in model_names:
            r = client.post("/api/predict", json=base_payload(model_name=name))
            assert r.status_code == 200, f"Failed for model_name={name}"
            assert r.json()["risk_target"] in [0, 1, 2]


# ===========================================================================
# GROUP 3: /api/predict — Edge & Boundary Values
# ===========================================================================

class TestPredictBoundaryValues:
    def test_predict_altitude_zero(self):
        """Altitude=0 (ground level) is valid — should return 200."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(altitude=0.0))
        assert r.status_code == 200

    def test_predict_altitude_maximum(self):
        """Altitude=50000 ft is the declared max — should return 200."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(altitude=50000.0))
        assert r.status_code == 200

    def test_predict_very_low_visibility(self):
        """Visibility=0.1 km — very dense fog scenario."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(avg_visibility_km=0.1))
        assert r.status_code == 200

    def test_predict_zero_turbine_distance(self):
        """Turbine distance=0 — aircraft directly at a turbine (extreme case)."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(nearest_turbine_km=0.0))
        assert r.status_code == 200

    def test_predict_engines_1(self):
        """Single-engine aircraft."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(engines=1.0))
        assert r.status_code == 200

    def test_predict_engines_4(self):
        """4-engine wide-body aircraft."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(engines=4.0))
        assert r.status_code == 200

    def test_predict_null_optional_fields_uses_defaults(self):
        """Optional fields set to null should fall back to server-side defaults."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        payload = base_payload()
        for opt in ["nearest_turbine_km", "avg_temp_c", "avg_humidity_pct",
                    "avg_wind_speed_kmh", "avg_visibility_km", "total_precip_mm", "avg_pressure_hpa"]:
            payload[opt] = None
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["risk_target"] in [0, 1, 2]

    def test_predict_no_airport_name_uses_state_default(self):
        """Omitting airport_name should fall back to state-level coordinates."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        payload = base_payload()
        payload.pop("airport_name", None)
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200

    def test_predict_unknown_model_name_falls_back(self):
        """An unrecognised model name should fall back to best_model, not crash."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(model_name="NonExistentModel123"))
        # Should still return 200 (graceful fallback to best_model)
        assert r.status_code == 200

    def test_predict_unknown_state_code_falls_back(self):
        """An unrecognised state code should fall back to CO defaults, not crash."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json=base_payload(origin_state_abbr="TX"))
        assert r.status_code == 200


# ===========================================================================
# GROUP 4: /api/predict — Validation Errors (422 Unprocessable Entity)
# ===========================================================================

class TestPredictValidationErrors:
    def test_predict_month_below_range(self):
        """flight_month=0 violates ge=1 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(flight_month=0))
        assert r.status_code == 422

    def test_predict_month_above_range(self):
        """flight_month=13 violates le=12 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(flight_month=13))
        assert r.status_code == 422

    def test_predict_year_below_range(self):
        """flight_year=2020 violates ge=2021 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(flight_year=2020))
        assert r.status_code == 422

    def test_predict_year_above_range(self):
        """flight_year=2026 violates le=2025 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(flight_year=2026))
        assert r.status_code == 422

    def test_predict_altitude_negative(self):
        """Negative altitude violates ge=0.0 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(altitude=-100.0))
        assert r.status_code == 422

    def test_predict_altitude_above_max(self):
        """altitude=60000 violates le=50000 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(altitude=60000.0))
        assert r.status_code == 422

    def test_predict_engines_below_range(self):
        """engines=0 violates ge=1.0 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(engines=0.0))
        assert r.status_code == 422

    def test_predict_engines_above_range(self):
        """engines=5 violates le=4.0 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(engines=5.0))
        assert r.status_code == 422

    def test_predict_turbine_km_negative(self):
        """nearest_turbine_km=-1 violates ge=0.0 constraint — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(nearest_turbine_km=-1.0))
        assert r.status_code == 422

    def test_predict_missing_required_state(self):
        """origin_state_abbr has a Pydantic default ('CO'), so omitting it uses the default
        and the API returns 200 — it is not a required field in the strict sense."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        payload = base_payload()
        payload.pop("origin_state_abbr")
        r = client.post("/api/predict", json=payload)
        assert r.status_code == 200  # Default 'CO' is used

    def test_predict_empty_body(self):
        """All PredictionRequest fields have Pydantic defaults, so an empty body is valid.
        The server uses all default values and returns 200."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict", json={})
        assert r.status_code == 200
        assert r.json()["risk_target"] in [0, 1, 2]

    def test_predict_wrong_type_for_month(self):
        """Passing string for flight_month — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(flight_month="nine"))
        assert r.status_code == 422

    def test_predict_wrong_type_for_altitude(self):
        """Passing string for altitude — expect 422."""
        client = get_client()
        r = client.post("/api/predict", json=base_payload(altitude="high"))
        assert r.status_code == 422

    def test_predict_wrong_http_method_get(self):
        """GET to /api/predict — static file mount intercepts non-POST routes,
        so this returns 404 (no matching file) rather than 405."""
        client = get_client()
        r = client.get("/api/predict")
        assert r.status_code == 404

    def test_predict_wrong_http_method_delete(self):
        """DELETE to /api/predict — not a registered method."""
        client = get_client()
        r = client.delete("/api/predict")
        assert r.status_code in [404, 405]


# ===========================================================================
# GROUP 5: /api/predict/compare-all
# ===========================================================================

class TestCompareAllEndpoint:
    def test_compare_all_returns_200(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict/compare-all", json=base_payload())
        assert r.status_code == 200

    def test_compare_all_has_9_model_results(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict/compare-all", json=base_payload()).json()
        assert data["models_count"] == 9
        assert len(data["model_results"]) == 9

    def test_compare_all_consensus_is_valid_class(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict/compare-all", json=base_payload()).json()
        assert data["consensus_class"] in [0, 1, 2]

    def test_compare_all_agreement_pct_in_range(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict/compare-all", json=base_payload()).json()
        assert 0.0 <= data["consensus_agreement_pct"] <= 100.0

    def test_compare_all_has_contributing_factors(self):
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        data = client.post("/api/predict/compare-all", json=base_payload()).json()
        assert "contributing_factors" in data
        assert len(data["contributing_factors"]) > 0

    def test_compare_all_empty_body_uses_defaults(self):
        """Empty body for compare-all uses all Pydantic defaults — returns 200."""
        if not models_ready():
            pytest.skip("Models not trained yet.")
        client = get_client()
        r = client.post("/api/predict/compare-all", json={})
        assert r.status_code == 200
        assert r.json()["consensus_class"] in [0, 1, 2]

    def test_compare_all_wrong_method_get(self):
        """GET to /api/predict/compare-all — static file mount returns 404."""
        client = get_client()
        r = client.get("/api/predict/compare-all")
        assert r.status_code == 404


if __name__ == "__main__":
    pytest.main(["-v", __file__])
