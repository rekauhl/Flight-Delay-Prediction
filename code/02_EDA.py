import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
df = pd.read_parquet("data/processed/bts_cleaned.parquet")
print(f"Loaded {len(df):,} total rows across 2024-2025.\n")

# ---------------------------------------------------------
# 1. DATA HEALTH CHECKS
# ---------------------------------------------------------
print("--- 1. Data Health Checks ---")
print("Missing values:")
print(df.isna().sum())

# Target definition verification
mismatches = ((df["ArrDelay"] >= 15).astype(int) != df["ArrDel15"]).sum()
print(f"\nTarget definition mismatches ((ArrDelay >= 15) != ArrDel15): {mismatches}")

# Hour calculation check (only in [0, 23]?)
df["ARR_HOUR"] = (df["CRSArrTime"] // 100) % 24
print(f"\nCRSDepTime == 2400: {(df['CRSDepTime'] == 2400).sum()} | DEP_HOUR == 24: {(df['DEP_HOUR'] == 24).sum()}")
print(f"CRSArrTime == 2400: {(df['CRSArrTime'] == 2400).sum()} | ARR_HOUR == 24: {(df['ARR_HOUR'] == 24).sum()}")

# Some sanity Checks
print(f"\nUnique Origin Airports ({len(df['Origin'].unique())}): {sorted(df['Origin'].unique())}")
print(f"Unique Dest Airports   ({len(df['Dest'].unique())}): {sorted(df['Dest'].unique())}")
print(f"Unique Carriers        ({len(df['Reporting_Airline'].unique())}): {sorted(df['Reporting_Airline'].unique())}")

# ---------------------------------------------------------
# 2. TARGET DISTRIBUTION & SEVERITY
# ---------------------------------------------------------
print("\n--- 2. Target Distribution & Severity ---")
print("Overall arrival delay rate (%):")
print((df["ArrDel15"].value_counts(normalize=True) * 100).round(2))

# Severity quantiles for delayed flights (ArrDel15 == 1)
delayed_mins = df[df["ArrDel15"] == 1]["ArrDelay"]
print("\nArrDelay severity quantiles (minutes delayed for positive class):")
print(delayed_mins.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).round(1))

# ---------------------------------------------------------
# 3. TEMPORAL SPLIT BREAKDOWN
# ---------------------------------------------------------
print("\n--- 3. Temporal Split Breakdown ---")
split_summary = df.groupby("SPLIT").agg(
    count=("ArrDel15", "count"),
    arr_delay_rate_pct=("ArrDel15", lambda x: (x.mean() * 100))
).reindex(["train", "val", "test"])
print(split_summary.round(2))

# ---------------------------------------------------------
# 4. FEATURE EXPLORATION
# ---------------------------------------------------------
print("\n--- 4. Feature Signal Exploration (Arrival Delay Rates %) ---")
print("\nBy Year:")
print((df.groupby("YEAR")["ArrDel15"].mean() * 100).round(2))

print("\nBy Carrier:")
print((df.groupby("Reporting_Airline")["ArrDel15"].mean() * 100).sort_values(ascending=False).round(2))

print("\nBy Hub:")
print((df.groupby("Origin")["ArrDel15"].mean() * 100).sort_values(ascending=False).round(2))

print("\nBy Departure Hour:")
print((df.groupby("DEP_HOUR")["ArrDel15"].mean() * 100).round(2))

print("\nBy Day of Week (0=Mon, 6=Sun):")
print((df.groupby("DAY_OF_WEEK")["ArrDel15"].mean() * 100).round(2))

print("\nBy Month:")
print((df.groupby("MONTH")["ArrDel15"].mean() * 100).round(2))

# ---------------------------------------------------------
# 5. EDA PLOTS
# ---------------------------------------------------------
plots_dir = Path("figures/eda")
plots_dir.mkdir(parents=True, exist_ok=True)

# Plot 1: Arrival Delay Rate by Departure Hour
plt.figure(figsize=(8, 4))
hourly = df.groupby("DEP_HOUR")["ArrDel15"].mean() * 100
plt.plot(hourly.index, hourly.values, marker="o", color="#1f77b4", linewidth=2)
plt.title("Arrival Delay Rate by Departure Hour", fontsize=12, fontweight="bold")
plt.xlabel("Scheduled Departure Hour (24-Hour)")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(0, 24))
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)  # Pushes grid lines behind bars
plt.tight_layout()
plt.savefig(plots_dir / "01_delay_rate_by_hour.png", dpi=300)
plt.close()

# Plot 2: Arrival Delay Rate by Month (2024–2025 Combined)
plt.figure(figsize=(8, 4))
monthly = df.groupby("MONTH")["ArrDel15"].mean() * 100
plt.bar(monthly.index, monthly.values, color="#1f77b4", width=0.6)
plt.title("Arrival Delay Rate by Month (2024–2025 Combined)", fontsize=12, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig(plots_dir / "02_delay_rate_by_month_combined.png", dpi=300)
plt.close()

# Plot 3: Arrival Delay Rate by Split (Train 2024 / Val H1-2025 / Test H2-2025)
plt.figure(figsize=(6, 4))
split_delays = df.groupby("SPLIT")["ArrDel15"].mean().reindex(["train", "val", "test"]) * 100
split_labels = ["Train (2024)", "Val (H1-2025)", "Test (H2-2025)"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
bars = plt.bar(split_labels, split_delays.values, color=colors, width=0.5)
plt.title("Arrival Delay Rate by Temporal Split", fontsize=12, fontweight="bold")
plt.ylabel("Delay Rate (%)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.tight_layout()
plt.savefig(plots_dir / "03_delay_rate_by_split.png", dpi=300)
plt.close()

# Plot 4: Seasonality (2024 Train vs 2025 Val/Test)
print("\n--- Year x Month Arrival Delay Rate (%) Crosstab ---")
year_month_ct = df.groupby(["YEAR", "MONTH"])["ArrDel15"].mean().unstack(level=0) * 100
print(year_month_ct.round(2))

plt.figure(figsize=(8, 4))
plt.plot(year_month_ct.index, year_month_ct[2024], marker="o", label="2024 (Train)", color="#1f77b4", linewidth=2, zorder=3)
plt.plot(year_month_ct.index, year_month_ct[2025], marker="s", label="2025 (Val/Test)", color="#ff7f0e", linewidth=2, linestyle="--", zorder=3)

plt.title("Arrival Delay Rate: 2024 vs. 2025", fontsize=12, fontweight="bold")
plt.xlabel("Month")
plt.ylabel("Delay Rate (%)")
plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_axisbelow(True)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(plots_dir / "04_seasonality_2024_vs_2025.png", dpi=300)
plt.close()

print(f"\nSaved all 4 plots to {plots_dir}/")