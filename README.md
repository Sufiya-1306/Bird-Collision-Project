# Bird Migration Collision Risk Prediction for Wind Farms and Airports Using Data Science

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![TailwindCSS](https://img.shields.io/badge/UI-TailwindCSS%20%7C%20Chart.js-38B2AC.svg)](https://tailwindcss.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary & Problem Statement
Avian collisions with commercial aircraft and wind turbines present severe aviation safety hazards and critical ecological challenges. Annually, thousands of bird strikes occur globally, threatening passenger safety, causing billions of dollars in aircraft structural and engine damage, and exacerbating wildlife mortality near expanding wind energy developments.

Traditional mitigation techniques rely on post-incident reporting or isolated visual spotters. This project builds a predictive Data Science pipeline that models **pre-event environmental, geographic, seasonal, and flight conditions** to forecast bird collision risk categories (**Low, Medium, High**) across the United States, focusing on recent **2021–2025** data across three critical flyway states: **Colorado (CO), Illinois (IL), and New York (NY)**.

---

## 🎯 Project Objectives
1. **Recent USA Scoping (2021–2025)**: Modernize prior academic workflows by training on recent aviation and environmental data.
2. **Multi-Source Data Fusion**: Combine FAA wildlife strikes, USWTDB wind turbine locations, NOAA METAR station weather, and massive GBIF bird migration records.
3. **Out-of-Core Stream Processing**: Ingest and process a **55.3-million-record (11.6 GB)** GBIF bird occurrence dataset in chunk-wise streaming mode without memory crashes.
4. **Leakage-Free Feature Engineering**: Isolate pre-event conditions and eliminate post-impact leakage variables (e.g., damage costs, repair effects).
5. **Multi-Model Benchmark (9 Models)**: Systematically train and compare 9 machine learning classifiers across Person A, Person B, and Person C with 5-fold cross-validation.
6. **Interactive Real-Time Decision Support**: Deploy a modern web dashboard and FastAPI backend enabling live hazard predictions and operational mitigation recommendations.

---

## 📂 Repository Architecture & Structure

```
Bird-Collision-Project/
├── .gitignore                                    # Strict Git exclusion (11.6 GB file & raw archives)
├── README.md                                     # Full project documentation & reproduction guide
├── requirements.txt                              # Python environment dependencies
│
├── backend/
│   └── app.py                                   # FastAPI backend serving prediction API & static UI
│
├── frontend/
│   └── index.html                               # Modern AI/Data Science SPA (Dashboard, Predict, Results, Models)
│
├── data/
│   ├── raw/
│   │   └── 0014699-260903145123482.zip          # Raw GBIF data dump (60.8 GB, Git-ignored)
│   └── processed/
│       ├── bird_strikes_v2_NY_IL_CO_clean.csv   # Cleaned FAA strike records (11,954 rows, 2021-2025)
│       ├── airports_v2_NY_IL_CO_clean.csv       # Cleaned airport reference table (1,974 rows)
│       ├── wind_turbines_v2_NY_IL_CO_clean.csv  # Cleaned USWTDB wind turbines (8,533 rows)
│       ├── weather_daily_station_aggregates.csv # Daily NOAA weather aggregates (5,094 station-days)
│       ├── migration_US_NY_IL_CO_2021_2025.csv  # Full GBIF dataset (11.6 GB, Git-ignored)
│       ├── migration_monthly_density_2021_2025.csv # Monthly migration intensity indices (180 rows)
│       ├── migration_sample_clean.csv           # Clean occurrences sample (276,490 rows, 25.7 MB)
│       └── bird_collision_model_dataset_FINAL_2021_2025.csv # Master ML dataset (11,304 rows, 29 cols)
│
├── models/
│   ├── trained_models/                          # Saved production models & evaluation artifacts
│   │   ├── preprocessor.joblib                  # Fitted ColumnTransformer (scaling & one-hot encoding)
│   │   ├── best_model.joblib                    # Highest-scoring production classifier
│   │   ├── logistic_regression.joblib           # Person A: Logistic Regression
│   │   ├── decision_tree.joblib                 # Person A: Decision Tree
│   │   ├── k_nearest_neighbors.joblib           # Person A: KNN
│   │   ├── random_forest.joblib                 # Person B: Random Forest
│   │   ├── gaussian_naive_bayes.joblib          # Person B: Gaussian Naive Bayes
│   │   ├── support_vector_machine.joblib        # Person B: SVM
│   │   ├── gradient_boosting.joblib             # Person C: Gradient Boosting
│   │   ├── mlp_classifier.joblib                # Person C: Multi-Layer Perceptron (MLP)
│   │   ├── xgboost.joblib                       # Person C: XGBoost
│   │   ├── evaluation_results.json              # Full cross-validation and test metrics
│   │   ├── model_comparison_table.csv           # Tabular comparison of all 9 models
│   │   ├── model_performance_comparison.png     # Side-by-side performance chart
│   │   └── confusion_matrices_all_9.png         # 3x3 multi-panel confusion matrix grid
│   ├── person_a/                                # Person A standalone model artifacts
│   ├── person_b/                                # Person B standalone model artifacts
│   └── person_c/                                # Person C standalone model artifacts
│
├── scripts/
│   ├── 01_clean_data.py                         # Final cleaning of strikes, airports, turbines & weather
│   ├── 02_process_migration_streaming.py        # Stream processing of 11.6 GB migration CSV
│   ├── 03_build_final_ml_dataset.py             # Feature engineering, spatial joins & target creation
│   ├── 04_train_all_models.py                   # 5-fold CV & training of all 9 models
│   └── 05_evaluate_and_compare.py               # Evaluation metric calculation & visualization export
│
└── tests/
    └── test_pipeline.py                         # Automated test suite (data, models, API & Git safety)
```

---

## 📊 Datasets & Preprocessing Methodology

### 1. FAA Wildlife Strike Database (2021–2025)
- **Source**: Federal Aviation Administration (FAA) National Wildlife Strike Database.
- **Coverage**: 11,954 verified wildlife strike reports across Colorado (5,043), Illinois (3,590), and New York (3,321) from January 1, 2021 to December 30, 2025.
- **Preprocessing**: Normalized flight dates, imputed missing engine counts with median (2.0), cleaned precipitation string artifacts, standardized categorical labels (`flight_phase`, `wildlife_size`, `conditions_sky`).

### 2. USWTDB Wind Turbine Database
- **Source**: United States Wind Turbine Database (USGS / LBNL / NREL).
- **Coverage**: 8,533 operational wind turbines in Illinois (3,678), Colorado (3,391), and New York (1,464).
- **Preprocessing**: Verified geographic bounding boxes, extracted operational attributes (`turbine_capacity`, `t_hh`, `t_rd`, `t_ttlh`), and calculated great-circle Haversine distance (`nearest_turbine_km`) to flight departure/arrival airports.

### 3. NOAA ASOS/METAR Weather Data (2021–2025)
- **Source**: National Oceanic and Atmospheric Administration automated surface observing systems.
- **Coverage**: 187,365 hourly observations from 3 benchmark stations: **KDEN (Denver, CO)**, **KORD (Chicago, IL)**, and **KJFK (New York, NY)** from 2021-01-01 to 2025-08-25.
- **Preprocessing**: Imputed dry periods (`Precipitation_mm = 0.0`), removed non-physical sensor spikes, and aggregated hourly readings into daily station averages (`avg_temp_c`, `avg_humidity_pct`, `avg_wind_speed_kmh`, `avg_visibility_km`, `total_precip_mm`, `avg_pressure_hpa`).

### 4. GBIF Bird Migration Occurrences (11.6 GB Stream Processing)
- **Source**: Global Biodiversity Information Facility (GBIF) / iNaturalist research-grade bird observations.
- **Volume**: **55,344,878 records** (11.6 GB raw CSV).
- **Stream Processing**:
  - Chunked reading (`chunksize=500,000`) without loading into memory.
  - Filtered for target states (`New York`, `Illinois`, `Colorado`), valid years (`2021–2025`), non-null species, and coordinate bounding boxes.
  - Derived `data/processed/migration_monthly_density_2021_2025.csv` capturing relative avian migration activity indices ($0.0 - 1.0$) and peak migration flags per state and month.
  - Generated `data/processed/migration_sample_clean.csv` (276,490 clean observation records, 25.7 MB) for spatial analysis and species reference.

---

## ⚙️ Feature Engineering & Target Formulation

### Leakage Prevention
To ensure valid predictive modeling, all post-event outcome variables were strictly eliminated before model training:
- `damage`, `cost`, `effect`, `people_injured`, `remains_collected`, `remains_sent_to_smithsonian`, `remarks`, `record_id`, and `operator`.

### Target Variable: `risk_target`
A transparent, quantile-based composite risk score was derived from pre-event flight and environmental risk factors:
$$\begin{aligned}
\text{risk\_index} = & 0.20 \times \text{count\_score} + 0.10 \times \text{size\_score} + 0.15 \times \text{migration\_score} \\
& + 0.15 \times \text{altitude\_score} + 0.10 \times \text{phase\_score} + 0.08 \times \text{wind\_score} \\
& + 0.07 \times \text{visibility\_score} + 0.05 \times \text{precip\_score} + 0.10 \times \text{turbine\_score}
\end{aligned}$$

Quantile tertiles ($1/3$ and $2/3$) partition the dataset into balanced risk categories:
- **0 = Low Risk** (33.5% of dataset)
- **1 = Medium Risk** (33.9% of dataset)
- **2 = High Risk** (32.6% of dataset)

---

## 🤖 Machine Learning Models & Evaluation

The project implements and benchmarks the complete set of 9 intended classifiers across 3 team tracks:
- **Person A**: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN)
- **Person B**: Random Forest, Gaussian Naive Bayes, Support Vector Machine (SVM)
- **Person C**: Gradient Boosting, Multi-Layer Perceptron (MLP Classifier), XGBoost

### Validation Protocol
- **Split**: 80% Train ($N=9,043$), 20% Test ($N=2,261$), stratified on `risk_target` (`random_state=42`).
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training set.
- **Preprocessing**: `ColumnTransformer` with median imputation and `StandardScaler` for numeric variables, and mode imputation with `OneHotEncoder` for categorical variables.

### Benchmark Results Table

| Rank | Model Name | Author Track | 5-Fold CV Acc | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Random Forest** | Person B | **95.2%** | **95.8%** | **0.958** | **0.958** | **0.958** | **0.958** |
| 🥈 | **XGBoost** | Person C | 94.6% | 95.1% | 0.951 | 0.951 | 0.951 | 0.951 |
| 🥉 | **Gradient Boosting** | Person C | 93.9% | 94.4% | 0.944 | 0.944 | 0.944 | 0.944 |
| 4 | **Decision Tree** | Person A | 89.8% | 90.4% | 0.904 | 0.904 | 0.904 | 0.904 |
| 5 | **Support Vector Machine (SVM)** | Person B | 88.7% | 89.3% | 0.894 | 0.893 | 0.893 | 0.893 |
| 6 | **MLP Classifier (Neural Net)** | Person C | 88.1% | 88.7% | 0.887 | 0.887 | 0.887 | 0.887 |
| 7 | **K-Nearest Neighbors (KNN)** | Person A | 83.9% | 84.6% | 0.846 | 0.846 | 0.846 | 0.846 |
| 8 | **Logistic Regression** | Person A | 81.5% | 82.1% | 0.821 | 0.821 | 0.821 | 0.821 |
| 9 | **Gaussian Naive Bayes** | Person B | 74.9% | 75.6% | 0.761 | 0.756 | 0.755 | 0.756 |

---

## 🔒 Git Safety Rule: 11.6 GB Migration File Exclusion

> [!CRITICAL]
> **Large File Safety**:
> The 11.6 GB migration CSV (`data/processed/migration_US_NY_IL_CO_2021_2025.csv`) and the raw 60.8 GB archive are **strictly excluded from Git tracking** via `.gitignore`.
> 
> Verification check:
> ```bash
> git check-ignore data/processed/migration_US_NY_IL_CO_2021_2025.csv
> ```
> Returns matching ignore rule. The large file remains safely on disk for local pipeline execution and is never staged, committed, or pushed to GitHub.

---

## 🚀 How to Run Backend & User Interface

### 1. Install Dependencies
```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Launch FastAPI Server & Dashboard
```bash
py -3.11 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Open Web UI Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```
- Interactive Swagger API docs are available at `http://127.0.0.1:8000/docs`.

---

## 🔄 Reproduction Guide: End-to-End Pipeline

To regenerate all cleaned datasets, stream-process the 11.6 GB file, build the master ML dataset, train all 9 models, and export charts from scratch:

```bash
# Step 1: Clean 4 core processed datasets
py -3.11 scripts/01_clean_data.py

# Step 2: Stream process the 11.6 GB GBIF migration file in chunks
py -3.11 scripts/02_process_migration_streaming.py

# Step 3: Merge spatial, temporal, weather, and migration features to create Master ML dataset
py -3.11 scripts/03_build_final_ml_dataset.py

# Step 4: Train all 9 machine learning models with 5-fold cross-validation
py -3.11 scripts/04_train_all_models.py

# Step 5: Generate comparison visualizations and tables
py -3.11 scripts/05_evaluate_and_compare.py

# Step 6: Run automated test suite
py -3.11 -m pytest tests/test_pipeline.py -v
```

---

## 🧪 Automated Testing
Run the comprehensive test suite verifying data integrity, model loading, leakage avoidance, and prediction endpoints:
```bash
py -3.11 -m pytest tests/test_pipeline.py -v
```
All tests validate that:
- Cleaned and processed files exist and contain zero unexpected nulls.
- `risk_target` contains classes `{0, 1, 2}` with no post-event leakage features.
- Git safely ignores the 11.6 GB dataset.
- All 9 machine learning model checkpoints load and execute inference.
- The prediction API `/api/predict` returns valid risk categories and probabilities.
