"""
Script 04: Train All 9 Machine Learning Models (2021-2025)
----------------------------------------------------------
Trains all 9 intended classifiers on the unified 2021-2025 dataset:
  Person A:
    1. Logistic Regression
    2. Decision Tree
    3. K-Nearest Neighbors (KNN)
  Person B:
    4. Random Forest
    5. Gaussian Naive Bayes
    6. Support Vector Machine (SVM)
  Person C:
    7. Gradient Boosting
    8. Multi-Layer Perceptron (MLP Classifier)
    9. XGBoost

Features:
  - Numeric: altitude, coordinates, elevation, turbine distance, daily weather, migration density
  - Categorical: state, flight phase, wildlife size, sky conditions, precipitation, pilot warning, season
Target:
  - risk_target (0 = Low, 1 = Medium, 2 = High)

Validation:
  - 80/20 Stratified train/test split (random_state=42)
  - 5-fold Stratified Cross-Validation on training set
  - Strict leakage prevention: fit preprocessor on train only, transform test.
  - Full evaluation metrics: Accuracy, Precision (Macro/Weighted), Recall, F1-Score.
  - Saves all trained models, preprocessor, and metrics as joblib & JSON artifacts.

Reproducibility:
  - Global random seed: RANDOM_STATE = 42
  - numpy global seed set at startup via np.random.seed(42)
  - All models with a random_state parameter use RANDOM_STATE = 42
  - KNN and GaussianNB are inherently deterministic; numpy seed covers tie-breaking

Early Stopping (MLP):
  - early_stopping=True monitors validation loss each epoch
  - validation_fraction=0.1 reserves 10% of training data as internal validation set
  - n_iter_no_change=15 stops training after 15 consecutive epochs without improvement

Training Logs:
  - Detailed logs written to logs/training_run_<timestamp>.log (file) and console
  - File logs: DEBUG level (CV fold scores, confusion matrices, classification reports, elapsed time)
  - Console logs: INFO level (summary row per model, MLP early-stopping epoch)
"""

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL RANDOM SEED — fixed at 42 for full reproducibility across all runs
# =============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = Path("data/processed/bird_collision_model_dataset_FINAL_2021_2025.csv")
MODELS_DIR = Path("models/trained_models")
PERSON_A_DIR = Path("models/person_a")
PERSON_B_DIR = Path("models/person_b")
PERSON_C_DIR = Path("models/person_c")
LOGS_DIR = Path("logs")


# =============================================================================
# LOGGING SETUP
# Writes to BOTH:
#   - logs/training_run_<timestamp>.log  (DEBUG level — full detail)
#   - console / stdout                   (INFO level — summary per model)
# =============================================================================
def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_filename = LOGS_DIR / f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("BirdCollisionTraining")
    logger.setLevel(logging.DEBUG)
    # Prevent duplicate handlers if function called more than once
    if logger.handlers:
        logger.handlers.clear()

    # File handler — DEBUG level (all detail: fold scores, confusion matrices, reports)
    fh = logging.FileHandler(log_filename, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — INFO level (compact summary)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_filename


# =============================================================================
# DATA LOADING
# =============================================================================
def load_and_prepare_data(logger):
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"ML dataset not found at {DATA_PATH}. "
            "Run scripts/03_build_final_ml_dataset.py first."
        )

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset: {len(df):,} rows x {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")

    target_col = "risk_target"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset.")

    y = df[target_col].astype(int)
    class_dist = y.value_counts().sort_index().to_dict()
    logger.info(f"Target class distribution: {class_dist}")

    drop_cols = [
        target_col, "airport_name", "flight_date", "wildlife_species",
        "origin_state", "aircraft_type", "operator"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for c in categorical_cols:
        X[c] = X[c].astype(str)

    logger.info(f"Features: {X.shape[1]} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    logger.debug(f"Numeric:     {numeric_cols}")
    logger.debug(f"Categorical: {categorical_cols}")

    return X, y, numeric_cols, categorical_cols


# =============================================================================
# PREPROCESSOR
# =============================================================================
def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


# =============================================================================
# MODEL DEFINITIONS — all with RANDOM_STATE=42
# =============================================================================
def get_all_9_models():
    return {
        # ── Person A ──────────────────────────────────────────────────────────
        "Logistic Regression": {
            "author": "Person A",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        },
        "Decision Tree": {
            "author": "Person A",
            "model": DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE
            )
        },
        "K-Nearest Neighbors": {
            "author": "Person A",
            # KNN is deterministic by nature; np.random.seed(42) covers tie-breaking
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance")
        },

        # ── Person B ──────────────────────────────────────────────────────────
        "Random Forest": {
            "author": "Person B",
            "model": RandomForestClassifier(
                n_estimators=200, max_depth=15, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1
            )
        },
        "Gaussian Naive Bayes": {
            "author": "Person B",
            # GaussianNB is fully deterministic; np.random.seed(42) covers internals
            "model": GaussianNB()
        },
        "Support Vector Machine": {
            "author": "Person B",
            "model": SVC(kernel="rbf", C=1.5, probability=True, random_state=RANDOM_STATE)
        },

        # ── Person C ──────────────────────────────────────────────────────────
        "Gradient Boosting": {
            "author": "Person C",
            "model": GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                random_state=RANDOM_STATE
            )
        },
        "MLP Classifier": {
            "author": "Person C",
            # early_stopping=True: monitors val loss; stops when no improvement
            # validation_fraction=0.1: 10% of train data used as internal val set
            # n_iter_no_change=15: patience of 15 epochs before stopping
            "model": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=350,
                alpha=0.01,
                random_state=RANDOM_STATE,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                verbose=False
            )
        },
        "XGBoost": {
            "author": "Person C",
            "model": XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1,
                random_state=RANDOM_STATE, eval_metric="mlogloss"
            )
        },
    }


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def train_and_evaluate():
    logger, log_filename = setup_logging()

    logger.info("=" * 80)
    logger.info("STEP 3: TRAINING ALL 9 MACHINE LEARNING MODELS")
    logger.info(f"Run started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Random seed : RANDOM_STATE = {RANDOM_STATE}  |  np.random.seed({RANDOM_STATE})")
    logger.info(f"Log file    : {log_filename}")
    logger.info("=" * 80)

    for d in [MODELS_DIR, PERSON_A_DIR, PERSON_B_DIR, PERSON_C_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    X, y, numeric_cols, categorical_cols = load_and_prepare_data(logger)

    # Save feature metadata
    feature_meta = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_names": {"0": "Low", "1": "Medium", "2": "High"},
        "target_classes": [0, 1, 2]
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # 80/20 stratified split — seed=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(
        f"Train/Test split: {X_train.shape[0]:,} train | {X_test.shape[0]:,} test "
        f"(80/20 stratified, random_state={RANDOM_STATE})"
    )

    # Fit preprocessor on train only — strict no-leakage policy
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    logger.info("Preprocessor fitted on training data only (zero leakage). Saved.")

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    logger.debug(f"Transformed — Train: {X_train_proc.shape} | Test: {X_test_proc.shape}")

    # 5-fold stratified CV — seed=42 for reproducible fold assignments
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models_dict = get_all_9_models()
    evaluation_results = {}
    best_f1 = -1.0
    best_model_name = ""

    logger.info("\nStarting training & 5-fold cross-validation...\n")
    header = f"{'Model':<26} | {'Author':<9} | {'CV Acc':>7} | {'Test Acc':>8} | {'F1 Mac':>7} | {'F1 Wt':>7} | {'Time':>6}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, config in models_dict.items():
        clf = config["model"]
        author = config["author"]

        logger.debug(f"[{name}] BEGIN — random_state={RANDOM_STATE}")
        t_start = time.time()

        # 5-fold CV
        cv_scores = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_acc = cv_scores.mean()
        cv_std = cv_scores.std()
        logger.debug(f"[{name}] CV fold accuracies: {[round(s, 4) for s in cv_scores.tolist()]}")
        logger.debug(f"[{name}] CV mean={cv_acc:.4f}  std={cv_std:.4f}")

        # Full fit on training set
        clf.fit(X_train_proc, y_train)

        # MLP early-stopping detail
        if name == "MLP Classifier" and hasattr(clf, "n_iter_"):
            logger.debug(
                f"[MLP] Early stopping: converged at epoch {clf.n_iter_} / {clf.max_iter} "
                f"(n_iter_no_change=15, validation_fraction=0.10)"
            )
            logger.info(f"  [MLP] Early stopping at epoch {clf.n_iter_} (max={clf.max_iter})")

        # Test-set evaluation
        y_pred = clf.predict(X_test_proc)
        acc = accuracy_score(y_test, y_pred)
        prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        elapsed = round(time.time() - t_start, 2)

        logger.debug(f"[{name}] Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        logger.debug(f"[{name}] Confusion Matrix: {cm}")
        logger.debug(f"[{name}] Training time: {elapsed}s")

        evaluation_results[name] = {
            "author": author,
            "random_state": RANDOM_STATE,
            "cv_accuracy_mean": round(float(cv_acc), 4),
            "cv_accuracy_std": round(float(cv_std), 4),
            "test_accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_m), 4),
            "recall_macro": round(float(rec_m), 4),
            "f1_macro": round(float(f1_m), 4),
            "f1_weighted": round(float(f1_w), 4),
            "confusion_matrix": cm,
            "training_time_seconds": elapsed,
        }

        logger.info(
            f"{name:<26} | {author:<9} | {cv_acc:>7.4f} | {acc:>8.4f} | "
            f"{f1_m:>7.4f} | {f1_w:>7.4f} | {elapsed:>5}s"
        )

        # Save to trained_models/ and per-author folder
        slug = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(clf, MODELS_DIR / f"{slug}.joblib")
        logger.debug(f"[{name}] Saved: models/trained_models/{slug}.joblib")

        if author == "Person A":
            joblib.dump(clf, PERSON_A_DIR / f"{slug}.joblib")
        elif author == "Person B":
            joblib.dump(clf, PERSON_B_DIR / f"{slug}.joblib")
        elif author == "Person C":
            joblib.dump(clf, PERSON_C_DIR / f"{slug}.joblib")

        if f1_m > best_f1:
            best_f1 = f1_m
            best_model_name = name

    # Save best model as best_model.joblib
    best_slug = best_model_name.lower().replace(" ", "_").replace("-", "_")
    best_clf = joblib.load(MODELS_DIR / f"{best_slug}.joblib")
    joblib.dump(best_clf, MODELS_DIR / "best_model.joblib")
    logger.info(f"\nBest model: {best_model_name} (F1 Macro={best_f1:.4f}) -> best_model.joblib")

    # Save test set for evaluation/plotting
    test_eval_df = X_test.copy()
    test_eval_df["y_true"] = y_test.values
    test_eval_df.to_csv(MODELS_DIR / "test_set_with_labels.csv", index=False)

    # Save full evaluation results
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("ALL 9 MODELS TRAINED AND SAVED SUCCESSFULLY!")
    logger.info(f"Best Model   : {best_model_name} (Macro F1 = {best_f1:.4f})")
    logger.info(f"Metrics JSON : models/trained_models/evaluation_results.json")
    logger.info(f"Training log : {log_filename}")
    logger.info("=" * 80)


if __name__ == "__main__":
    train_and_evaluate()
