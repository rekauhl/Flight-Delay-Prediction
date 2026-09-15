import json
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    precision_recall_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# =========================================================
# STEP 1: LOAD DATA & SPLIT INTO SETS
# =========================================================
data_path = Path("data/processed/bts_features.parquet")
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(data_path)

# Separate by our time splits
train_df = df[df["SPLIT"] == "train"].copy()
val_df = df[df["SPLIT"] == "val"].copy()
test_df = df[df["SPLIT"] == "test"].copy()

# Target column: 1 if delayed >15 mins, 0 if on-time
y_train = train_df["ArrDel15"].values
y_val = val_df["ArrDel15"].values
y_test = test_df["ArrDel15"].values

# Model 1 features: Pre-Pushback (Remove DepDelay)
X_train_m1 = train_df.drop(columns=["ArrDel15", "SPLIT", "DepDelay"])
X_val_m1 = val_df.drop(columns=["ArrDel15", "SPLIT", "DepDelay"])
X_test_m1 = test_df.drop(columns=["ArrDel15", "SPLIT", "DepDelay"])

# Model 2 features: Post-Pushback (Includes DepDelay)
X_train_m2 = train_df.drop(columns=["ArrDel15", "SPLIT"])
X_val_m2 = val_df.drop(columns=["ArrDel15", "SPLIT"])
X_test_m2 = test_df.drop(columns=["ArrDel15", "SPLIT"])

results = {}

# Helper function to get simple score pair
def get_ranking_scores(y_true, probabilities):
    return {
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4)
    }

# Helper function to find cutoff threshold and evaluate real-world performance
def evaluate_operational_cutoff(val_probs, test_probs, target_recall=0.80):
    # 1. Find probability cutoff on Validation set that catches ~80% of delays
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
    closest_index = np.argmin(np.abs(recalls - target_recall))
    cutoff_threshold = float(thresholds[min(closest_index, len(thresholds) - 1)])

    # 2. Check recall achieved on Validation set with this cutoff
    val_predictions = (val_probs >= cutoff_threshold).astype(int)
    val_recall_achieved = float(recall_score(y_val, val_predictions))

    # 3. Apply the exact same cutoff to Test predictions
    test_predictions = (test_probs >= cutoff_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_predictions).ravel()

    # 4. Calculate stats on Test set
    test_pr_auc = float(average_precision_score(y_test, test_probs))
    test_delay_rate = float(np.mean(y_test))  # Percentage of delayed flights on arrival (prevalence)
    multiplier_over_baseline = test_pr_auc / test_delay_rate  # How many times better than guessing (lift)

    return {
        "decision_threshold": round(cutoff_threshold, 4),
        "val_recall_achieved": round(val_recall_achieved, 4),
        "test_precision": round(float(precision_score(y_test, test_predictions)), 4),
        "test_recall": round(float(recall_score(y_test, test_predictions)), 4),
        "test_f1": round(float(f1_score(y_test, test_predictions)), 4),
        "test_prevalence": round(test_delay_rate, 4),
        "lift_over_baseline": round(multiplier_over_baseline, 4),
        "test_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

# =========================================================
# STEP 2: DUMMY BASELINE (GUESSING HISTORICAL AVERAGE)
# =========================================================
print("--- Computing Naive Baseline ---")
dummy = DummyClassifier(strategy="prior")
dummy.fit(X_train_m1, y_train)

dummy_val_probs = dummy.predict_proba(X_val_m1)[:, 1]
dummy_test_probs = dummy.predict_proba(X_test_m1)[:, 1]

results["Naive_Baseline"] = {
    "val": get_ranking_scores(y_val, dummy_val_probs),
    "test": get_ranking_scores(y_test, dummy_test_probs)
}

# =========================================================
# STEP 3: LOGISTIC REGRESSION (MODEL 1 & MODEL 2)
# =========================================================
def run_logistic_regression(X_tr, X_v, X_te):
    #  Standardize features to mean 0, std 1, to treat all features fairly regardless of scale
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_v_scaled = scaler.transform(X_v)
    X_te_scaled = scaler.transform(X_te)

    best_val_score = -1.0
    best_model = None
    best_weight_setting = None

    # Try two class weight settings
    for class_weight in [None, "balanced"]:
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight=class_weight)
        model.fit(X_tr_scaled, y_train)

        # Predict probability of delay on Validation set
        val_probs = model.predict_proba(X_v_scaled)[:, 1]
        val_score = average_precision_score(y_val, val_probs)

        # Keep the model with the highest validation score
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = model
            best_weight_setting = class_weight

    # Final predictions with best model
    final_val_probs = best_model.predict_proba(X_v_scaled)[:, 1]
    final_test_probs = best_model.predict_proba(X_te_scaled)[:, 1]

    return {
        "class_weight": best_weight_setting,
        "val": get_ranking_scores(y_val, final_val_probs),
        "test": get_ranking_scores(y_test, final_test_probs),
        "operational_eval": evaluate_operational_cutoff(final_val_probs, final_test_probs)
    }

# =========================================================
# STEP 4: RANDOM FOREST (MODEL 1 & MODEL 2)
# =========================================================
def run_random_forest(X_tr, X_v, X_te):
    best_val_score = -1.0
    best_model = None
    best_parameters = {}

    # 12 parameter combinations
    for depth in [10, 15, 20]:
        for min_samples in [10, 50]:
            for class_weight in [None, "balanced"]:
                model = RandomForestClassifier(
                    n_estimators=150,
                    max_depth=depth,
                    min_samples_leaf=min_samples,
                    class_weight=class_weight,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_tr, y_train)

                val_probs = model.predict_proba(X_v)[:, 1]
                val_score = average_precision_score(y_val, val_probs)

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model = model
                    best_parameters = {
                        "max_depth": depth,
                        "min_samples_leaf": min_samples,
                        "class_weight": class_weight
                    }

    # Final predictions with best model
    final_val_probs = best_model.predict_proba(X_v)[:, 1]
    final_test_probs = best_model.predict_proba(X_te)[:, 1]

    # Get top 10 most important features
    feature_importances = pd.Series(best_model.feature_importances_, index=X_tr.columns)
    top_10_features = feature_importances.sort_values(ascending=False).head(10).round(4).to_dict()

    return {
        "params": best_parameters,
        "val": get_ranking_scores(y_val, final_val_probs),
        "test": get_ranking_scores(y_test, final_test_probs),
        "operational_eval": evaluate_operational_cutoff(final_val_probs, final_test_probs),
        "top_features": top_10_features
    }

# =========================================================
# STEP 5: RUN ALL EXPERIMENTS AND SAVE RESULTS
# =========================================================
print("\n--- Running Model 1 (Pre-Pushback) ---")
results["LR_Model1"] = run_logistic_regression(X_train_m1, X_val_m1, X_test_m1)
print(f"LR M1 Winner | Val PR-AUC: {results['LR_Model1']['val']['pr_auc']} | Test PR-AUC: {results['LR_Model1']['test']['pr_auc']}")

results["RF_Model1"] = run_random_forest(X_train_m1, X_val_m1, X_test_m1)
print(f"RF M1 Winner | Val PR-AUC: {results['RF_Model1']['val']['pr_auc']} | Test PR-AUC: {results['RF_Model1']['test']['pr_auc']}")

print("\n--- Running Model 2 (Post-Pushback: +DepDelay) ---")
results["LR_Model2"] = run_logistic_regression(X_train_m2, X_val_m2, X_test_m2)
print(f"LR M2 Winner | Val PR-AUC: {results['LR_Model2']['val']['pr_auc']} | Test PR-AUC: {results['LR_Model2']['test']['pr_auc']}")

results["RF_Model2"] = run_random_forest(X_train_m2, X_val_m2, X_test_m2)
print(f"RF M2 Winner | Val PR-AUC: {results['RF_Model2']['val']['pr_auc']} | Test PR-AUC: {results['RF_Model2']['test']['pr_auc']}")

# Export JSON
export_path = results_dir / "final_benchmarks.json"
with open(export_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved results to {export_path}")