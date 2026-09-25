"""
Automated Test Suite for Frontend Layouts and UI Architecture
-------------------------------------------------------------
Verifies:
  1. Frontend index.html exists and is well-formed HTML.
  2. All 5 dedicated layout tabs exist with correct section IDs:
     - tab-dashboard (Dashboard Overview)
     - tab-prediction (Risk Prediction Tool)
     - tab-results (Results & Diagnostics)
     - tab-models (Model Benchmark 9)
     - tab-dataset (Datasets & Pipeline)
  3. Navigation triggers exist for switching between all layout tabs.
  4. Core input form controls exist for all scenario parameters.
  5. Results and diagnostics containers exist (risk badge, contributing factors, recommendations).
  6. All 9 Models consensus and comparison UI components exist.
  7. Benchmark leaderboard table container exists in Tab 4.
"""

from pathlib import Path
import pytest
import re

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"

@pytest.fixture(scope="module")
def html_content():
    assert FRONTEND_FILE.exists(), f"Frontend index.html missing at {FRONTEND_FILE}"
    return FRONTEND_FILE.read_text(encoding="utf-8")


def test_frontend_file_exists_and_non_empty(html_content):
    """Verify frontend HTML exists and has substantial content."""
    assert len(html_content) > 5000, "Frontend index.html appears unexpectedly truncated or empty."
    assert "<!DOCTYPE html>" in html_content or "<html" in html_content


def test_all_five_layout_tabs_exist(html_content):
    """Verify all 5 tested tab layout views are declared in the DOM."""
    required_tabs = [
        'id="tab-dashboard"',
        'id="tab-prediction"',
        'id="tab-results"',
        'id="tab-models"',
        'id="tab-dataset"',
    ]
    for tab_id in required_tabs:
        assert tab_id in html_content, f"Required layout tab view missing: {tab_id}"


def test_navigation_switch_tab_bindings(html_content):
    """Verify that sidebar navigation triggers switchTab for each layout view."""
    tabs = ["dashboard", "prediction", "results", "models", "dataset"]
    for tab in tabs:
        pattern = rf"switchTab\(['\"]{tab}['\"]\)"
        assert re.search(pattern, html_content), f"Missing switchTab trigger for layout: {tab}"


def test_core_input_controls_exist(html_content):
    """Verify all key aviation, environmental, and proximity form controls exist."""
    required_inputs = [
        'id="inp-state"',
        'id="inp-airport"',
        'id="inp-altitude"',
        'id="inp-phase"',
        'id="inp-engines"',
        'id="inp-warned"',
        'id="inp-month"',
        'id="inp-wildlife-size"',
        'id="inp-turbine"',
        'id="inp-model"',
        'id="inp-sky"',
        'id="inp-precip"',
        'id="inp-wind"',
        'id="inp-visibility"',
        'id="inp-temp"',
        'id="inp-humidity"',
    ]
    for inp in required_inputs:
        assert inp in html_content, f"Required input control missing in prediction layout: {inp}"


def test_results_and_diagnostics_layout_containers(html_content):
    """Verify results layout has risk title, gauge border, probability bars, factors, and recommendation."""
    required_containers = [
        'id="res-risk-title"',
        'id="res-confidence-badge"',
        'id="prob-bar-low"',
        'id="prob-bar-med"',
        'id="prob-bar-high"',
        'id="risk-factors-container"',
        'id="res-recommendation"',
    ]
    for cont in required_containers:
        assert cont in html_content, f"Required diagnostics container missing: {cont}"


def test_nine_model_consensus_layout_components(html_content):
    """Verify the All 9 Models consensus evaluation card, banner, and inference grid exist."""
    assert "compareAllModels()" in html_content, "Missing compareAllModels JavaScript trigger in layout."
    assert 'id="compare-grid"' in html_content, "Missing compare-grid container in layout."
    assert 'id="compare-consensus-banner"' in html_content, "Missing compare-consensus-banner in layout."
    assert 'id="btn-compare-all"' in html_content, "Missing btn-compare-all in layout."


def test_model_benchmark_table_layout(html_content):
    """Verify the benchmark table layout exists in Tab 4."""
    assert 'id="all-models-tbody"' in html_content, "Missing all-models-tbody container in benchmark layout."
    assert "Gradient Boosting" in html_content, "Benchmark layout should reference top model."
    assert "XGBoost" in html_content, "Benchmark layout should reference XGBoost."
