import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path

# Paths and configuration
YEARS = [2024, 2025]
MONTHS = list(range(1, 13))

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_PARQUET = PROCESSED_DIR / "bts_cleaned.parquet"

# Columns to extract from raw CSVs
BTS_COLS = [
    'FlightDate', 'Reporting_Airline', 'Origin', 'Dest',
    'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'Distance',
    'Cancelled', 'Diverted', 'DepDelay', 'ArrDelay', 'ArrDel15'
]

CARRIERS = ['AA', 'DL', 'UA', 'WN']
HUBS = ['ATL', 'ORD', 'DFW', 'DEN', 'LAX', 'JFK', 'SFO', 'SEA', 'CLT', 'MSP']



RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

monthly_dfs = []

for year in YEARS:
    for month in MONTHS:
        zip_path = RAW_DIR / f"bts_{year}_{month}.zip"
        url = f"https://www.transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        
        # Download monthly zip if not downloaded yet
        if not zip_path.exists():
            print(f"Downloading {year}-{month:02d}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"Failed to download {year}-{month:02d}: {e}")
                continue
        else:
            print(f"Using cached raw file for {year}-{month:02d}.")

        # Extract and parse matching columns
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                csv_name = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
                with zip_ref.open(csv_name) as f:
                    df_m = pd.read_csv(f, usecols=BTS_COLS)

            # Drop cancelled, diverted, or invalid rows
            df_m = df_m[(df_m['Cancelled'] == 0) & (df_m['Diverted'] == 0)].copy()
            df_m = df_m.dropna(subset=['ArrDel15']).copy()
            df_m['ArrDel15'] = df_m['ArrDel15'].astype(int)

            # Filter for selected carriers and flights (Hub-to-Hub)
            df_m = df_m[
                (df_m['Reporting_Airline'].isin(CARRIERS)) &
                (df_m['Origin'].isin(HUBS)) &
                (df_m['Dest'].isin(HUBS))
            ].copy()

            monthly_dfs.append(df_m)
            print(f"Loaded {year}-{month:02d}: {len(df_m):,} flights")
        except Exception as e:
            print(f"Error processing {year}-{month:02d}: {e}")

# Combine into single DataFrame and format date/time columns
df = pd.concat(monthly_dfs, ignore_index=True)
df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['YEAR'] = df['FlightDate'].dt.year
df['MONTH'] = df['FlightDate'].dt.month
df['DAY_OF_WEEK'] = df['FlightDate'].dt.dayofweek
df['DEP_HOUR'] = (df['CRSDepTime'] // 100) % 24

# Assign chronological train / val / test splits
conditions = [
    (df['YEAR'] == 2024),                       # Train:  Full Year 2024 (Jan - Dec)
    (df['YEAR'] == 2025) & (df['MONTH'] <= 6),  # Val:    H1 2025 (Jan - Jun)
    (df['YEAR'] == 2025) & (df['MONTH'] >= 7)   # Test:   H2 2025 (Jul - Dec)
]
choices = ['train', 'val', 'test']
df['SPLIT'] = np.select(conditions, choices, default='train')

# Save processed data
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"\nDone! Saved {len(df):,} total rows to {OUTPUT_PARQUET}")
print("\nSplit distribution:")
print(df['SPLIT'].value_counts())