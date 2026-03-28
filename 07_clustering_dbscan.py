# src/07_clustering_dbscan.py

import pandas as pd
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/featured_hourly.csv"
SAMPLE_SIZE = 100000

# ---------------- LOAD & SAMPLE ----------------
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(SAMPLE_SIZE, random_state=42)

cluster_features = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3"
]

X = df_sample[cluster_features]

# ---------------- DBSCAN MODEL ----------------
dbscan = DBSCAN(
    eps=0.5,
    min_samples=50
)

df_sample["dbscan_cluster"] = dbscan.fit_predict(X)

print("DBSCAN clustering completed.")
print(df_sample["dbscan_cluster"].value_counts())
