# src/06_clustering_hierarchical.py

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/featured_hourly.csv"
SAMPLE_SIZE = 50000
N_CLUSTERS = 5

# ---------------- LOAD & SAMPLE DATA ----------------
df = pd.read_csv(DATA_PATH)
df_sample = df.sample(SAMPLE_SIZE, random_state=42)

cluster_features = [
    "PM2.5", "PM10", "NO2", "CO", "SO2", "O3",
    "PM_ratio", "NO2_CO_ratio", "SO2_NO2_ratio",
    "PM2.5_x_NO2", "PM10_x_SO2"
]

X = df_sample[cluster_features]

# ---------------- HIERARCHICAL MODEL ----------------
hc = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    linkage="ward"
)

df_sample["hier_cluster"] = hc.fit_predict(X)

print("Hierarchical clustering completed on sample.")
print(df_sample["hier_cluster"].value_counts())
