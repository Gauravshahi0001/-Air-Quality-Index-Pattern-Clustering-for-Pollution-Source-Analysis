# src/08_evaluation.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/featured_hourly.csv"
FIG_DIR = "results/figures/"
SAMPLE_SIZE = 100000   # ⭐ CRITICAL FIX

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

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

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- PCA ----------------
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ---------------- SAMPLE DATA ----------------
sample_idx = df.sample(SAMPLE_SIZE, random_state=42).index
X_sample = X_pca[sample_idx]

# ---------------- ELBOW METHOD ----------------
inertia = []
K_RANGE = range(2, 9)

for k in K_RANGE:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_sample)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_RANGE, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.savefig(FIG_DIR + "elbow_curve.png")
plt.show()

# ---------------- SILHOUETTE & DB INDEX ----------------
print("\nCluster Evaluation Metrics (on sample):\n")

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_sample)

    sil = silhouette_score(X_sample, labels)
    db = davies_bouldin_score(X_sample, labels)

    print(f"K = {k}")
    print(f"  Silhouette Score      : {sil:.3f}")
    print(f"  Davies-Bouldin Index  : {db:.3f}\n")
