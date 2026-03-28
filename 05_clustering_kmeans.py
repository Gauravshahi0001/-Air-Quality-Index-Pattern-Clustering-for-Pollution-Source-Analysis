# src/05_clustering_kmeans.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = "data/processed/featured_hourly.csv"
OUTPUT_PATH = "data/processed/kmeans_clustered.csv"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# ---------------- PATTERN FEATURES (NO INTERACTIONS) ----------------
PATTERN_FEATURES = [
    "PM_ratio",
    "NO2_CO_ratio",
    "SO2_NO2_ratio",
    "is_rush_hour",
    "is_night",
    "is_weekend",
    "season_code"
]

X = df[PATTERN_FEATURES]

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- DIMENSION REDUCTION ----------------
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance by PCA:", pca.explained_variance_ratio_)

# ---------------- SAMPLE FOR TRAINING ----------------
sample_idx = df.sample(200000, random_state=42).index
X_sample = X_pca[sample_idx]

# ---------------- K-MEANS ----------------
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=30
)

kmeans.fit(X_sample)

# ---------------- PREDICT FULL DATA ----------------
df["kmeans_cluster"] = kmeans.predict(X_pca)

df.to_csv(OUTPUT_PATH, index=False)

print("KMeans with PCA completed.")
print(df["kmeans_cluster"].value_counts())
print("Clustered data saved to:", OUTPUT_PATH)