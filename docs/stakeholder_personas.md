# Stakeholder Personas — Bird Migration Collision Risk Prediction System

> This document defines the formal stakeholder personas identified during the requirements and design phase of the project. Each persona represents a real-world user group whose needs shaped the system's features, data choices, and interface design.

---

## Persona 1 — Airport Safety Officer

**Name:** Captain Meera Joshi  
**Role:** Wildlife Strike Hazard Manager, Regional Airport Authority  
**Location:** O'Hare International Airport, Chicago, IL  
**Age:** 44

### Background
Meera has 18 years of experience managing bird-strike incident reporting and runway safety at one of the USA's busiest airports. She coordinates with ATC, grounds crews, and the FAA's Wildlife Strike Database team. She files mandatory FAA Form 5200-7 reports after every confirmed bird strike.

### Goals
- Receive **advance warning** of high-risk periods (migration season, peak months) so she can schedule extra wildlife hazing operations
- Access **historical trend data** for her specific airport and state
- Produce compliance reports for the FAA with accurate risk classifications

### Pain Points
- Current alerts are generic national advisories — no airport-level or state-specific predictions
- Manual weather + season correlation is time-consuming
- No single tool combines migration data, weather, and flight-phase risk

### How She Uses This System
- Selects her airport (O'Hare) and enters current flight parameters
- Views the **risk dashboard** to check today's collision risk level
- Runs the **9-model consensus** to verify the prediction with high confidence
- Exports the risk output for her daily safety briefing report

### Key Features Used
- `/api/predict` with specific airport and month
- Model comparison card (Tab 3)
- `/api/stats` for historical context

---

## Persona 2 — Wind Farm Operations Manager

**Name:** David Reyes  
**Role:** Site Operations Manager, Rocky Mountain Wind Energy LLC  
**Location:** Centennial Airport area, Colorado (CO)  
**Age:** 38

### Background
David oversees a 120-turbine wind farm 6 km from Centennial Airport. During peak raptor migration (September–October), his site experiences a surge in bird fatality incidents, which triggers regulatory reviews under the Bald and Golden Eagle Protection Act (BGEPA). He needs to balance energy production targets with mandatory curtailment requirements.

### Goals
- Identify **exactly which days/weeks** require turbine curtailment to minimise raptor collision risk
- Justify curtailment decisions to investors with **data-backed risk scores**
- Receive early warning 24–48 hours ahead of high-risk migration events

### Pain Points
- Migration forecasts are only available at a coarse national or flyway level
- No tool links turbine proximity data with live weather and migration density
- Manual spreadsheet tracking of fatality incidents is error-prone

### How He Uses This System
- Enters current weather conditions and his turbine distance (6 km)
- Checks `migration_density` and `is_peak_migration` values in the API response
- Uses the **risk level badge** (Low/Medium/High) to make same-day curtailment decisions
- Reviews seasonal trends in the `/api/stats` endpoint

### Key Features Used
- `/api/predict` with `nearest_turbine_km` set to his farm distance
- `/api/predict/compare-all` for high-stakes curtailment decisions
- `/api/stats` for monthly migration density reference

---

## Persona 3 — Data Science Student / Academic Researcher

**Name:** Priya Nair  
**Role:** MSc Data Science Student, University of Illinois  
**Location:** Remote (working on thesis)  
**Age:** 24

### Background
Priya is writing her thesis on wildlife collision prediction using machine learning. She wants to compare multiple model families on a real ecological dataset and understand feature importance in high-accuracy ensemble models. She is comfortable with Python, pandas, and scikit-learn but new to API-based ML deployment.

### Goals
- Understand **which ML model performs best** for this prediction task and why
- Reproduce training results using the provided scripts
- Study the **feature engineering** choices (migration density, seasonal encoding, turbine proximity)
- Use the API as a reference implementation for her own deployment chapter

### Pain Points
- Most published bird-strike datasets are small or unbalanced
- No publicly available tool combines FAA strike data with GBIF migration and weather
- Needs reproducible code with documented random seeds and evaluation methodology

### How She Uses This System
- Reads `README.md` for data sources, preprocessing pipeline, and model methodology
- Runs `scripts/04_train_all_models.py` to reproduce training with logs
- Inspects `logs/training_run_*.log` for per-fold CV scores and confusion matrices
- Calls `/api/models` to benchmark all 9 models
- Studies `evaluation_results.json` and the two PNG visualisation artefacts

### Key Features Used
- Training scripts with structured logs (`logs/`)
- `/api/models` benchmark endpoint
- `models/trained_models/evaluation_results.json`
- `model_performance_comparison.png`, `confusion_matrices_all_9.png`

---

## Persona 4 — FAA Wildlife Strike Policy Analyst

**Name:** Robert Kamau  
**Role:** Senior Policy Analyst, FAA Wildlife Hazard Management Division  
**Location:** Washington, D.C.  
**Age:** 52

### Background
Robert uses aggregated strike data from the FAA Wildlife Strike Database to recommend policy updates for airports in high-risk states. He is not a programmer but is comfortable with dashboards, maps, and summary statistics. He needs evidence-based arguments for budget allocation for wildlife hazing equipment.

### Goals
- See **state-level risk comparisons** (CO vs IL vs NY) across years
- Identify **which airports** are highest risk during peak migration months
- Get simple, explainable risk summaries without ML jargon

### Pain Points
- Existing tools require database query skills he doesn't have
- Needs exportable summaries, not raw CSVs
- Cannot evaluate individual model internals — needs a consensus/majority view

### How He Uses This System
- Opens the **web dashboard** directly in his browser
- Uses the **Tab 1** data overview cards for state-level stats
- Requests a prediction for any airport scenario using the UI form
- Reviews **contributing factors** (plain-English bullets) to support policy arguments
- Uses the 9-model consensus percentage as a confidence proxy for reporting

### Key Features Used
- Frontend dashboard (Tab 1, Tab 2 prediction form, Tab 3 contributing factors)
- `contributing_factors` and `recommended_action` fields in API response
- 9-model consensus agreement percentage

---

## Summary Table

| Persona | Role | Primary Goal | Technical Level | Key Concern |
|---|---|---|---|---|
| **Meera Joshi** | Airport Safety Officer | Pre-flight risk warning | Low–Medium | Accuracy & FAA compliance |
| **David Reyes** | Wind Farm Manager | Curtailment scheduling | Low | Turbine-proximity risk |
| **Priya Nair** | MSc Researcher | Model benchmarking & reproduction | High | Reproducibility & methodology |
| **Robert Kamau** | FAA Policy Analyst | State-level trend analysis | Low | Plain-English summaries |

---

## How Personas Influenced Design Decisions

| Design Decision | Driven By |
|---|---|
| 3-class risk target (Low/Medium/High) instead of binary | Meera & Robert — needed nuanced severity levels |
| Airport name → auto-fill coordinates | Meera & David — no need to enter lat/lon manually |
| `nearest_turbine_km` as a first-class feature | David — wind farm distance is his primary lever |
| 9-model consensus endpoint | Robert & Meera — high-stakes decisions need consensus confidence |
| Plain-English `contributing_factors` and `recommended_action` | Robert — non-technical users need interpretable output |
| Fixed random seeds + structured training logs | Priya — reproducibility is a core academic requirement |
| Early stopping for MLP | Priya — prevents overfitting, documents convergence epoch |
| States limited to CO, IL, NY | All — dataset coverage aligned with real FAA strike concentration |
