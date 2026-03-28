

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
INPUT_PATH = "data/processed/processed_hourly.csv"
OUTPUT_PATH = "data/processed/cleaned_hourly.csv"

POLLUTANTS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)

# ---------------- DATETIME VALIDATION ----------------
before_rows = df.shape[0]
df = df.dropna(subset=["datetime"])
after_rows = df.shape[0]

print(f"Dropped {before_rows - after_rows} rows due to missing datetime")

# ---------------- COLUMN SELECTION ----------------
df = df[["datetime", "station"] + POLLUTANTS]

print("After column selection:", df.shape)

# ---------------- MISSING VALUE HANDLING ----------------
print("\nHandling missing values (station-wise median)...")

for col in POLLUTANTS:
    df[col] = df.groupby("station")[col].transform(
        lambda x: x.fillna(x.median())
    )

# Fallback (if a station has all NaNs)
df[POLLUTANTS] = df[POLLUTANTS].fillna(df[POLLUTANTS].median())

print("Missing values after imputation:")
print(df[POLLUTANTS].isnull().sum())

# ---------------- OUTLIER CLIPPING (IQR) ----------------
print("\nApplying IQR-based outlier clipping...")

for col in POLLUTANTS:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

# ---------------- TIME FEATURE EXTRACTION ----------------
df["hour"] = df["datetime"].dt.hour
df["day_of_week"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month

# ---------------- FEATURE SCALING ----------------
scaler = StandardScaler()
df_scaled = df.copy()

df_scaled[POLLUTANTS] = scaler.fit_transform(df[POLLUTANTS])

# ---------------- SAVE CLEANED DATA ----------------
df_scaled.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned data saved to:", OUTPUT_PATH)
print("Final shape:", df_scaled.shape)
