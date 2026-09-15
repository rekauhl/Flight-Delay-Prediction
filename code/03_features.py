from pathlib import Path
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# 1. Load multi-year cleaned dataset
input_path = Path("data/processed/bts_cleaned.parquet")
output_path = Path("data/processed/bts_features.parquet")

df = pd.read_parquet(input_path)
print(f"Loaded {len(df):,} rows across 2024–2025 for feature engineering.")

# 2. Derive Scheduled Arrival Hour
df["ARR_HOUR"] = (df["CRSArrTime"] // 100) % 24

# 3. Holiday Flag (US Federal Holidays +/- 1 day travel window across 2024-2025)
cal = USFederalHolidayCalendar()
holidays = cal.holidays(start="2024-01-01", end="2025-12-31")
expanded_holidays = set(holidays).union(
    set(holidays - pd.Timedelta(days=1)),
    set(holidays + pd.Timedelta(days=1))
)
df["IS_HOLIDAY"] = df["FlightDate"].isin(expanded_holidays).astype(int)

# 4. One-Hot Encoding for Categorical & Temporal Features
ohe_cols = [
    "Reporting_Airline", "Origin", "Dest",
    "MONTH", "DAY_OF_WEEK", "DEP_HOUR", "ARR_HOUR"
]
df = pd.get_dummies(df, columns=ohe_cols, dtype=int)

# 5. Drop raw timestamps, non-predictive metadata & leakage columns
unneeded_cols = [
    "FlightDate", "CRSDepTime", "CRSArrTime",
    "Cancelled", "Diverted", "ArrDelay", "YEAR"
]
df = df.drop(columns=unneeded_cols)

# 6. Save Clean Feature Matrix
df.to_parquet(output_path, index=False)

print("\nFeature engineering complete!")
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

feature_cols = [c for c in df.columns if c not in ["SPLIT", "ArrDel15", "DepDelay"]]
print(f"Total features created for modeling: {len(feature_cols)}")
print(f"Saved feature matrix to {output_path}")