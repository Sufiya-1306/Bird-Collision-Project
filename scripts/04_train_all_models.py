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
"""

import os
import json
import warnings
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
RANDOM_STATE = 42

DATA_PATH = Path("data/processed/bird_collision_model_dataset_FINAL_2021_2025.csv")
MODELS_DIR = Path("models/trained_models")
PERSON_A_DIR = Path("models/person_a")
PERSON_B_DIR = Path("models/person_b")
PERSON_C_DIR = Path("models/person_c")

def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ML dataset not found at {DATA_PATH}. Run scripts/03_build_final_ml_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset with {len(df):,} rows and {len(df.columns)} columns.")

    target_col = "risk_target"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' missing from dataset.")

    y = df[target_col].astype(int)

    # Exclude non-predictive identifiers or dates if present
    drop_cols = [
        target_col, "airport_name", "flight_date", "wildlife_species",
        "origin_state", "aircraft_type", "operator"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    categorical_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Cast categorical columns to string
    for c in categorical_cols:
        X[c] = X[c].astype(str)

    print(f"Features: {X.shape[1]} total ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
    print(f"Numeric features:     {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    return X, y, numeric_cols, categorical_cols

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

def get_all_9_models():
    return {
        # Person A Models
        "Logistic Regression": {
            "author": "Person A",
            "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        },
        "Decision Tree": {
            "author": "Person A",
            "model": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=RANDOM_STATE)
        },
        "K-Nearest Neighbors": {
            "author": "Person A",
            "model": KNeighborsClassifier(n_neighbors=7, weights="distance")
        },

        # Person B Models
        "Random Forest": {
            "author": "Person B",
            "model": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
        },
        "Gaussian Naive Bayes": {
            "author": "Person B",
            "model": GaussianNB()
        },
        "Support Vector Machine": {
            "author": "Person B",
            "model": SVC(kernel="rbf", C=1.5, probability=True, random_state=RANDOM_STATE)
        },

        # Person C Models
        "Gradient Boosting": {
            "author": "Person C",
            "model": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
        },
        "MLP Classifier": {
            "author": "Person C",
            "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350, alpha=0.01, random_state=RANDOM_STATE)
        },
        "XGBoost": {
            "author": "Person C",
            "model": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="mlogloss")
        }
    }

def train_and_evaluate():
    print("=" * 80)
    print("STEP 3: TRAINING ALL 9 MACHINE LEARNING MODELS")
    print("=" * 80)

    for d in [MODELS_DIR, PERSON_A_DIR, PERSON_B_DIR, PERSON_C_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    X, y, numeric_cols, categorical_cols = load_and_prepare_data()

    # Save feature names metadata
    feature_meta = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target_names": {0: "Low", 1: "Medium", 2: "High"},
        "target_classes": [0, 1, 2]
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # 80/20 Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training split: {X_train.shape[0]:,} samples | Test split: {X_test.shape[0]:,} samples")

    # Fit preprocessor on training data only
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    print("Saved preprocessor to models/trained_models/preprocessor.joblib")

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # 5-fold CV setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    models_dict = get_all_9_models()
    evaluation_results = {}
    best_f1 = -1.0
    best_model_name = ""

    print("\nStarting Training & 5-Fold Cross Validation...\n")
    print(f"{'Model Name':<26} | {'Author':<9} | {'CV Acc':<8} | {'Test Acc':<9} | {'Macro F1':<9} | {'Weighted F1':<11}")
    print("-" * 85)

    for name, config in models_dict.items():
        clf = config["model"]
        author = config["author"]

        # Cross-validation on train split
        cv_scores = cross_val_score(clf, X_train_proc, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        cv_acc = cv_scores.mean()

        # Fit model on full training set
        clf.fit(X_train_proc, y_train)

        # Evaluate on held-out test set
        y_pred = clf.predict(X_test_proc)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_wt = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()

        evaluation_results[name] = {
            "author": author,
            "cv_accuracy_mean": round(float(cv_acc), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_macro": round(float(f1_mac), 4),
            "f1_weighted": round(float(f1_wt), 4),
            "confusion_matrix": cm,
        }

        print(f"{name:<26} | {author:<9} | {cv_acc:.4f}   | {acc:.4f}    | {f1_mac:.4f}    | {f1_wt:.4f}")

        # Save model artifact
        slug = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(clf, MODELS_DIR / f"{slug}.joblib")

        # Also save in respective author directory
        if author == "Person A":
            joblib.dump(clf, PERSON_A_DIR / f"{slug}.joblib")
        elif author == "Person B":
            joblib.dump(clf, PERSON_B_DIR / f"{slug}.joblib")
        elif author == "Person C":
            joblib.dump(clf, PERSON_C_DIR / f"{slug}.joblib")

        if f1_mac > best_f1:
            best_f1 = f1_mac
            best_model_name = name

    # Save best model reference
    best_slug = best_model_name.lower().replace(" ", "_").replace("-", "_")
    best_clf = joblib.load(MODELS_DIR / f"{best_slug}.joblib")
    joblib.dump(best_clf, MODELS_DIR / "best_model.joblib")

    # Save test dataset and predictions for evaluation plotting
    test_eval_df = X_test.copy()
    test_eval_df["y_true"] = y_test
    test_eval_df.to_csv(MODELS_DIR / "test_set_with_labels.csv", index=False)

    # Save evaluation results to JSON
    with open(MODELS_DIR / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL 9 MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print(f"Best Model Overall: {best_model_name} (Macro F1 = {best_f1:.4f})")
    print(f"Metrics saved to:   models/trained_models/evaluation_results.json")
    print("=" * 80)

if __name__ == "__main__":
    train_and_evaluate()
