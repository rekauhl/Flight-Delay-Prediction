import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set design theme
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Setup Paths & Load Data
results_path = Path("results/final_benchmarks.json")
figures_dir = Path("figures/results")
figures_dir.mkdir(parents=True, exist_ok=True)

with open(results_path, "r") as f:
    benchmarks = json.load(f)

# ---------------------------------------------------------
# Plot 1: Benchmark Performance & Lift Comparison
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

models = ["Naive Baseline", "LR Model 1", "RF Model 1", "LR Model 2", "RF Model 2"]
pr_aucs = [
    benchmarks["Naive_Baseline"]["test"]["pr_auc"],
    benchmarks["LR_Model1"]["test"]["pr_auc"],
    benchmarks["RF_Model1"]["test"]["pr_auc"],
    benchmarks["LR_Model2"]["test"]["pr_auc"],
    benchmarks["RF_Model2"]["test"]["pr_auc"],
]

colors = ["#7f8c8d", "#3498db", "#3498db", "#2ecc71", "#2ecc71"]
bars = ax.bar(models, pr_aucs, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

ax.set_ylabel("Test PR-AUC", fontsize=12, fontweight="bold")
ax.set_title("PR-AUC Comparison: Pre-Pushback (M1) vs. Post-Pushback (M2)", fontsize=14, pad=15, fontweight="bold")
ax.set_ylim(0, 1.0)

# Annotate exact PR-AUC scores & baseline lift above bars
baseline_auc = pr_aucs[0]
for bar, score in zip(bars, pr_aucs):
    yval = bar.get_height()
    lift = yval / baseline_auc
    label = f"{score:.4f}\n({lift:.1f}x Lift)" if score != baseline_auc else f"{score:.4f}\n(Baseline)"
    ax.text(
        bar.get_x() + bar.get_width() / 2.0, 
        yval + 0.03, 
        label, 
        ha="center", 
        va="bottom", 
        fontsize=10, 
        fontweight="bold"
    )

ax.axhline(baseline_auc, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Naive Baseline Floor (0.2432)")
ax.legend(loc="upper left")

plt.tight_layout()
fig1_path = figures_dir / "01_pr_auc_comparison.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# ---------------------------------------------------------
# Plot 2: Top Feature Importances (RF Model 1 vs RF Model 2)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False)

rf_m1_feats = pd.Series(benchmarks["RF_Model1"]["top_features"]).sort_values(ascending=True)
rf_m2_feats = pd.Series(benchmarks["RF_Model2"]["top_features"]).sort_values(ascending=True)

# Plot M1 Importances
rf_m1_feats.plot(kind="barh", ax=axes[0], color="#3498db", edgecolor="black")
axes[0].set_title("Model 1 (Pre-Pushback) Top Features", fontweight="bold", fontsize=12)
axes[0].set_xlabel("Gini Importance", fontweight="bold")

# Plot M2 Importances
rf_m2_feats.plot(kind="barh", ax=axes[1], color="#2ecc71", edgecolor="black")
axes[1].set_title("Model 2 (Post-Pushback: +DepDelay) Top Features", fontweight="bold", fontsize=12)
axes[1].set_xlabel("Gini Importance", fontweight="bold")

# Highlight DepDelay dominance in M2
# axes[1].patches[-1].set_facecolor("#145a32")

plt.suptitle("Feature Importance Shift: Post-Pushback Feature Dominance", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig2_path = figures_dir / "02_feature_importance.png"
plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")

# ---------------------------------------------------------
# Plot 3: Operational Confusion Matrices (RF Model 1 vs. RF Model 2)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_m1 = benchmarks["RF_Model1"]["operational_eval"]["test_confusion_matrix"]
cm_m2 = benchmarks["RF_Model2"]["operational_eval"]["test_confusion_matrix"]

arr_m1 = np.array([[cm_m1["tn"], cm_m1["fp"]], [cm_m1["fn"], cm_m1["tp"]]])
arr_m2 = np.array([[cm_m2["tn"], cm_m2["fp"]], [cm_m2["fn"], cm_m2["tp"]]])

labels = ["On-Time", "Delayed"]

# Model 1 Heatmap (RF)
sns.heatmap(arr_m1, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
axes[0].set_title(
    f"RF Model 1 (Pre-Pushback)\nTest Recall: {benchmarks['RF_Model1']['operational_eval']['test_recall']:.2f} | Precision: {benchmarks['RF_Model1']['operational_eval']['test_precision']:.2f}",
    fontweight="bold"
)
axes[0].set_xlabel("Predicted Label", fontweight="bold")
axes[0].set_ylabel("Actual Label", fontweight="bold")

# Model 2 Heatmap (RF)
sns.heatmap(arr_m2, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False,
            xticklabels=labels, yticklabels=labels, annot_kws={"size": 12, "weight": "bold"})
axes[1].set_title(
    f"RF Model 2 (Post-Pushback)\nTest Recall: {benchmarks['RF_Model2']['operational_eval']['test_recall']:.2f} | Precision: {benchmarks['RF_Model2']['operational_eval']['test_precision']:.2f}",
    fontweight="bold"
)
axes[1].set_xlabel("Predicted Label", fontweight="bold")
axes[1].set_ylabel("Actual Label", fontweight="bold")

plt.suptitle("Test Set Confusion Matrix Comparison (Derived from Val ~80% Recall Threshold)", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
fig3_path = figures_dir / "03_confusion_matrices.png"
plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {fig3_path}")