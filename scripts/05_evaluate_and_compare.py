"""
Script 05: Model Evaluation, Visualization & Comparison
-------------------------------------------------------
Loads trained model evaluation results and generates:
  1. Comprehensive performance comparison table (CLI + CSV + Markdown)
  2. Multi-panel Confusion Matrices plot (3x3 grid for all 9 models)
  3. Model Performance Bar Chart (Accuracy & Macro F1 side-by-side)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path("models/trained_models")
RESULTS_FILE = MODELS_DIR / "evaluation_results.json"
OUT_CSV = MODELS_DIR / "model_comparison_table.csv"
OUT_CHART = MODELS_DIR / "model_performance_comparison.png"
OUT_CMS = MODELS_DIR / "confusion_matrices_all_9.png"

def evaluate_and_plot():
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"Evaluation results not found at {RESULTS_FILE}. Run scripts/04_train_all_models.py first.")

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    # 1. Build DataFrame
    records = []
    for model_name, m in results.items():
        records.append({
            "Model": model_name,
            "Author": m.get("author", "N/A"),
            "CV Accuracy": m["cv_accuracy_mean"],
            "Test Accuracy": m["test_accuracy"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "F1-Score (Macro)": m["f1_macro"],
            "F1-Score (Weighted)": m["f1_weighted"],
        })

    df = pd.DataFrame(records)
    df = df.sort_values(by="F1-Score (Macro)", ascending=False).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)

    print("=" * 95)
    print("ALL 9 MODELS PERFORMANCE COMPARISON")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)

    # 2. Performance Comparison Bar Chart
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")

    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(15, 7))
    bars1 = ax.bar(x - width/2, df["Test Accuracy"], width, label="Test Accuracy", color="#3b82f6", edgecolor="none")
    bars2 = ax.bar(x + width/2, df["F1-Score (Macro)"], width, label="Macro F1-Score", color="#10b981", edgecolor="none")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=13, fontweight="bold")
    ax.set_title("Performance Comparison of All 9 Machine Learning Models\n(Bird Collision Risk Prediction 2021-2025)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=25, ha="right", fontsize=11, fontweight="medium")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=11)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9, rotation=0)

    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(OUT_CHART, dpi=200)
    plt.close()
    print(f"Saved performance comparison chart: {OUT_CHART}")

    # 3. 3x3 Confusion Matrices Grid
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    axes = axes.flatten()
    classes = ["Low (0)", "Medium (1)", "High (2)"]

    palette_map = {
        "Person A": "Blues",
        "Person B": "Greens",
        "Person C": "Purples"
    }

    for idx, (model_name, m) in enumerate(results.items()):
        ax = axes[idx]
        cm = np.array(m["confusion_matrix"])
        cmap = palette_map.get(m.get("author", "Person A"), "Blues")

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, cbar=False,
            xticklabels=classes, yticklabels=classes, ax=ax,
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{model_name}\n({m.get('author')}) | Acc: {m['test_accuracy']:.3f} | F1: {m['f1_macro']:.3f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Confusion Matrices for All 9 Models (Held-out Test Set)", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(OUT_CMS, dpi=200)
    plt.close()
    print(f"Saved confusion matrices: {OUT_CMS}")

if __name__ == "__main__":
    evaluate_and_plot()
