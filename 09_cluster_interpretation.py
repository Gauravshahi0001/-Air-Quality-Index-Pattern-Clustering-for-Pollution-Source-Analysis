# src/09_cluster_interpretation.py

import pandas as pd

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/kmeans_clustered.csv"
OUTPUT_PATH = "results/cluster_profiles.csv"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# ---------------- FEATURES FOR INTERPRETATION ----------------
analysis_features = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
    "PM_ratio", "NO2_CO_ratio", "SO2_NO2_ratio",
    "is_rush_hour", "is_night", "is_weekend",
    "season_code"
]

# ---------------- CLUSTER PROFILES ----------------
cluster_profiles = (
    df
    .groupby("kmeans_cluster")[analysis_features]
    .mean()
    .round(3)
)

# Save cluster profiles
cluster_profiles.to_csv(OUTPUT_PATH)

print("Cluster profiles saved to:", OUTPUT_PATH)
print("\nCluster-wise mean values:\n")
print(cluster_profiles)
