# src/10_cluster_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/kmeans_clustered.csv"
FIG_DIR = "results/figures/"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)

# Remove outlier cluster (cluster 4)
df = df[df["kmeans_cluster"] != 4]

# ---------------- FEATURES ----------------
pattern_features = [
    "PM_ratio",
    "NO2_CO_ratio",
    "SO2_NO2_ratio",
    "is_rush_hour",
    "is_night",
    "is_weekend",
    "season_code"
]

pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# ---------------- PCA SCATTER PLOT ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[pattern_features])

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df.sample(50000, random_state=42),
    x="PC1",
    y="PC2",
    hue="kmeans_cluster",
    palette="tab10",
    alpha=0.6
)
plt.title("PCA Scatter Plot of Pollution Clusters")
plt.savefig(FIG_DIR + "pca_cluster_scatter.png")
plt.show()

# ---------------- CLUSTER HEATMAP ----------------
cluster_pollutants = (
    df
    .groupby("kmeans_cluster")[pollutants]
    .mean()
)

plt.figure(figsize=(8, 5))
sns.heatmap(
    cluster_pollutants,
    annot=True,
    cmap="RdYlGn_r",
    fmt=".2f"
)
plt.title("Cluster-wise Average Pollutant Levels")
plt.savefig(FIG_DIR + "cluster_pollutant_heatmap.png")
plt.show()

# ---------------- HOURLY PATTERN PER CLUSTER ----------------
hourly_pattern = (
    df
    .groupby(["kmeans_cluster", "hour"])["PM2.5"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=hourly_pattern,
    x="hour",
    y="PM2.5",
    hue="kmeans_cluster"
)
plt.title("Hourly PM2.5 Pattern per Cluster")
plt.xlabel("Hour of Day")
plt.ylabel("Average PM2.5 (scaled)")
plt.grid(True)
plt.savefig(FIG_DIR + "hourly_cluster_pattern.png")
plt.show()

print("Cluster visualizations saved to:", FIG_DIR)

# ---------------- PM2.5 BOXPLOT PER CLUSTER ----------------
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df,
    x="kmeans_cluster",
    y="PM2.5",
    hue="kmeans_cluster",
    palette="Set2",
    legend=False
)

plt.title("PM2.5 Distribution Across Pollution Clusters")
plt.xlabel("Cluster (Pollution Source)")
plt.ylabel("PM2.5 (Scaled Concentration)")
plt.grid(True)

plt.savefig(FIG_DIR + "pm25_boxplot_per_cluster.png")
plt.show()
