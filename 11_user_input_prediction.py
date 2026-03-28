# src/11_user_input_prediction.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib

# ---------------- CONFIG ----------------
TRAINED_DATA_PATH = "data/processed/featured_hourly.csv"
USER_INPUT_PATH = "data/raw/user_input.csv"

# ---------------- LOAD TRAINING DATA ----------------
df_train = pd.read_csv(TRAINED_DATA_PATH)

PATTERN_FEATURES = [
    "PM_ratio",
    "NO2_CO_ratio",
    "SO2_NO2_ratio",
    "is_rush_hour",
    "is_night",
    "is_weekend",
    "season_code"
]

X_train = df_train[PATTERN_FEATURES]

# ---------------- TRAIN PIPELINE (ONCE) ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=30)
kmeans.fit(X_pca)

# ---------------- LOAD USER INPUT ----------------
df_user = pd.read_csv(USER_INPUT_PATH)

# -------- FEATURE ENGINEERING FOR USER DATA --------
df_user["PM_ratio"] = df_user["PM10"] / (df_user["PM2.5"] + 1e-6)
df_user["NO2_CO_ratio"] = df_user["NO2"] / (df_user["CO"] + 1e-6)
df_user["SO2_NO2_ratio"] = df_user["SO2"] / (df_user["NO2"] + 1e-6)

df_user["is_rush_hour"] = df_user["hour"].isin([7,8,9,17,18,19]).astype(int)
df_user["is_night"] = df_user["hour"].isin([22,23,0,1,2,3,4]).astype(int)
df_user["is_weekend"] = df_user["day_of_week"].isin([5,6]).astype(int)

# -------- SEASON CODE (RULE-BASED, SAFE FOR USER INPUT) --------
def get_season_code(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Summer
    elif month in [6, 7, 8, 9]:
        return 2  # Monsoon
    else:
        return 3  # Post-monsoon

df_user["season_code"] = df_user["month"].apply(get_season_code)
X_user = df_user[PATTERN_FEATURES]
# ---------------- PREDICT CLUSTER ----------------
X_user_scaled = scaler.transform(X_user)
X_user_pca = pca.transform(X_user_scaled)

df_user["predicted_cluster"] = kmeans.predict(X_user_pca)

# ---------------- MAP CLUSTER TO SOURCE ----------------
cluster_map = {
    0: "Traffic Emissions",
    1: "Construction & Dust",
    2: "Industrial Emissions",
    3: "Seasonal / Background"
}

df_user["pollution_source"] = df_user["predicted_cluster"].map(cluster_map)

print("\nUser Input Prediction Result:\n")
print(df_user[["predicted_cluster", "pollution_source"]])
