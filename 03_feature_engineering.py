# src/03_feature_engineering.py

import pandas as pd
import numpy as np

# ---------------- CONFIG ----------------
INPUT_PATH = "data/processed/cleaned_hourly.csv"
OUTPUT_PATH = "data/processed/featured_hourly.csv"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(INPUT_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Initial shape:", df.shape)

# ---------------- POLLUTION RATIO FEATURES ----------------
df["PM_ratio"] = df["PM10"] / (df["PM2.5"] + 1e-6)
df["NO2_CO_ratio"] = df["NO2"] / (df["CO"] + 1e-6)
df["SO2_NO2_ratio"] = df["SO2"] / (df["NO2"] + 1e-6)

# ---------------- TEMPORAL FEATURES ----------------
df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ---------------- SEASON FEATURE (FIXED & CLEAN) ----------------
df["season"] = pd.cut(
    df["month"],
    bins=[0, 2, 5, 9, 11, 12],
    labels=["Winter_1", "Summer", "Monsoon", "Post_Monsoon", "Winter_2"],
    include_lowest=True,
    ordered=True
)

# Convert season to numeric codes
df["season_code"] = df["season"].astype("category").cat.codes

# Drop string season column ONCE
df.drop(columns=["season"], inplace=True)

# ---------------- INTERACTION FEATURES ----------------
df["PM2.5_x_NO2"] = df["PM2.5"] * df["NO2"]
df["PM10_x_SO2"] = df["PM10"] * df["SO2"]

# ---------------- SAVE FEATURED DATA ----------------
df.to_csv(OUTPUT_PATH, index=False)

print("Feature-engineered data saved to:", OUTPUT_PATH)
print("Final shape:", df.shape)
