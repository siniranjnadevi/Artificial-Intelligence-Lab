"""
Comprehensive Evaluation Orchestrator:
Runs all 6 algorithms on the exact same dataset splits,
collects actual empirical metrics, saves models, and generates publication-grade visualizations.
"""

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, classification_report

from .config import METRICS_DIR, RANDOM_SEED
from .data_loader import load_detection_data
from .train_baselines import train_and_evaluate_baselines
from .train_distilbert import train_and_evaluate_distilbert
from .train_hybrid import train_and_evaluate_hybrid


def run_full_evaluation():
    print(f"\n{'#'*70}")
    print("STARTING COMPREHENSIVE PHASE 3 MODEL BENCHMARKING (6 ALGORITHMS)")
    print(f"{'#'*70}\n")

    # Load test labels once for evaluation
    _, _, _, y_test = load_detection_data()

    all_results = {}

    # 1. Train and evaluate 4 baselines
    print("\n--- PHASE 3A: BASELINE ALGORITHMS ---")
    baseline_res = train_and_evaluate_baselines()
    all_results.update(baseline_res)

    # 2. Train and evaluate DistilBERT
    print("\n--- PHASE 3B: DEEP LEARNING TRANSFORMER (DistilBERT) ---")
    distilbert_res = train_and_evaluate_distilbert()
    all_results["DistilBERT"] = distilbert_res

    # 3. Train and evaluate Proposed Hybrid Model
    print("\n--- PHASE 3C: PROPOSED HYBRID CONTEXTUAL-LEXICAL ARCHITECTURE ---")
    hybrid_res = train_and_evaluate_hybrid()
    all_results["Proposed Hybrid Model"] = hybrid_res

    # 4. Construct Comparison DataFrame
    rows = []
    reports_dict = {}

    for name, res in all_results.items():
        rows.append({
            "Algorithm": name,
            "CV_F1_Macro_Mean": round(res["cv_f1_macro_mean"], 4),
            "CV_F1_Macro_Std": round(res["cv_f1_macro_std"], 4),
            "Test_Accuracy": round(res["test_accuracy"], 4),
            "Test_Precision_Macro": round(res["test_precision_macro"], 4),
            "Test_Recall_Macro": round(res["test_recall_macro"], 4),
            "Test_F1_Macro": round(res["test_f1_macro"], 4),
            "Test_Precision_Weighted": round(res["test_precision_weighted"], 4),
            "Test_Recall_Weighted": round(res["test_recall_weighted"], 4),
            "Test_F1_Weighted": round(res["test_f1_weighted"], 4),
            "Test_ROC_AUC": round(res["test_roc_auc"], 4),
            "True_Negatives": res["confusion_matrix"]["tn"],
            "False_Positives": res["confusion_matrix"]["fp"],
            "False_Negatives": res["confusion_matrix"]["fn"],
            "True_Positives": res["confusion_matrix"]["tp"],
            "False_Positive_Rate": round(res["false_positive_rate"], 4),
            "False_Negative_Rate": round(res["false_negative_rate"], 4),
            "Training_Time_Sec": round(res["training_time_seconds"], 2),
            "Prediction_Latency_ms": round(res["prediction_latency_ms"], 3)
        })

        y_pred = np.array(res["y_pred"])
        report = classification_report(y_test, y_pred, target_names=["Not Disaster (0)", "Disaster (1)"], output_dict=True)
        reports_dict[name] = report

    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values(by="Test_F1_Macro", ascending=False).reset_index(drop=True)

    # Save Comparison CSV and JSON
    csv_path = METRICS_DIR / "model_comparison.csv"
    json_path = METRICS_DIR / "classification_reports.json"
    comparison_df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f:
        json.dump(reports_dict, f, indent=2)

    print(f"\n[Evaluation] Saved comparison table to: {csv_path}")
    print(f"[Evaluation] Saved detailed classification reports to: {json_path}")

    # 5. Generate Visualizations (300 DPI)
    print("\n[Evaluation] Generating publication-grade visualizations...")
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Plot 1: 6-Panel Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (name, res) in enumerate(all_results.items()):
        cm = np.array([
            [res["confusion_matrix"]["tn"], res["confusion_matrix"]["fp"]],
            [res["confusion_matrix"]["fn"], res["confusion_matrix"]["tp"]]
        ])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
            xticklabels=['Non-Disaster', 'Disaster'],
            yticklabels=['Non-Disaster', 'Disaster']
        )
        axes[idx].set_title(f"{name}\nAcc: {res['test_accuracy']:.3f} | F1: {res['test_f1_macro']:.3f}", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Predicted Label", fontsize=10)
        axes[idx].set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    cm_path = METRICS_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Confusion Matrices figure to: {cm_path}")

    # Plot 2: Comparative ROC Curves
    plt.figure(figsize=(10, 7))
    palette = sns.color_palette("tab10", len(all_results))

    for idx, (name, res) in enumerate(all_results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['test_roc_auc']:.3f})", color=palette[idx], linewidth=2.0)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Chance (AUC = 0.500)')
    plt.title("Comparative ROC Curves (Independent Test Set)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = METRICS_DIR / "roc_curves.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved ROC Curves figure to: {roc_path}")

    # Plot 3: Metrics Comparison Bar Chart
    plt.figure(figsize=(13, 7))
    plot_df = comparison_df.melt(
        id_vars=["Algorithm"],
        value_vars=["Test_Accuracy", "Test_Precision_Macro", "Test_Recall_Macro", "Test_F1_Macro"],
        var_name="Metric",
        value_name="Score"
    )
    plot_df["Metric"] = plot_df["Metric"].str.replace("Test_", "").str.replace("_", " ")

    ax = sns.barplot(data=plot_df, x="Algorithm", y="Score", hue="Metric", palette="muted")
    plt.title("Algorithm Performance Comparison on Independent Test Set (N=1,034)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Machine Learning Algorithm", fontsize=11, fontweight='bold')
    plt.ylabel("Score [0.0 - 1.0]", fontsize=11, fontweight='bold')
    plt.ylim(0.60, 0.95)
    plt.xticks(rotation=15, ha='right')
    plt.legend(loc="lower right", frameon=True)

    # Annotate bar values
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            ax.annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=7, color='black', xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    bar_path = METRICS_DIR / "performance_comparison_bar.png"
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved Performance Comparison Bar chart to: {bar_path}")

    # Plot 4: F1 vs Latency Tradeoff
    plt.figure(figsize=(9, 6))
    for _, r in comparison_df.iterrows():
        plt.scatter(r["Prediction_Latency_ms"], r["Test_F1_Macro"], s=160, alpha=0.85, edgecolors='black', linewidth=1.2)
        plt.annotate(
            r["Algorithm"],
            (r["Prediction_Latency_ms"], r["Test_F1_Macro"]),
            xytext=(6, 4), textcoords='offset points', fontsize=9, fontweight='semibold'
        )

    plt.title("F1-Score vs. Prediction Latency Trade-Off for Real-Time Alerting", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Prediction Latency per Sample (Milliseconds)", fontsize=11)
    plt.ylabel("Test F1-Macro Score", fontsize=11)
    plt.tight_layout()
    tradeoff_path = METRICS_DIR / "tradeoff_latency_f1.png"
    plt.savefig(tradeoff_path, dpi=300)
    plt.close()
    print(f"[Evaluation] Saved F1-Latency Tradeoff figure to: {tradeoff_path}")

    print(f"\n{'='*70}")
    print("FINAL BENCHMARK COMPARISON TABLE (Ranked by Test F1-Macro):")
    print(f"{'='*70}")
    print(comparison_df[[
        "Algorithm", "Test_Accuracy", "Test_Precision_Macro", "Test_Recall_Macro",
        "Test_F1_Macro", "Test_ROC_AUC", "Training_Time_Sec", "Prediction_Latency_ms"
    ]].to_string(index=False))
    print(f"{'='*70}\n")

    return comparison_df


if __name__ == "__main__":
    run_full_evaluation()
