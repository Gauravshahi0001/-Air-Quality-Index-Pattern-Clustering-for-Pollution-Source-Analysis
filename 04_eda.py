# src/04_eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- CONFIG ----------------
DATA_PATH = "data/processed/featured_hourly.csv"
FIG_DIR = "results/figures/"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset loaded for EDA:", df.shape)

# Select main pollutant columns
pollutants = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# ---------------- 1. DISTRIBUTION PLOTS ----------------
print("Plotting pollutant distributions...")

plt.figure(figsize=(12, 8))
for i, col in enumerate(pollutants):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], bins=50, kde=True)
    plt.title(col)

plt.tight_layout()
plt.savefig(FIG_DIR + "pollutant_distributions.png")
plt.show()

# ---------------- 2. CORRELATION HEATMAP ----------------
print("Plotting correlation heatmap...")

corr = df[pollutants].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Pollutant Correlation Heatmap")
plt.savefig(FIG_DIR + "pollutant_correlation_heatmap.png")
plt.show()

# ---------------- 3. HOURLY POLLUTION TRENDS ----------------
print("Plotting hourly pollution trends...")

hourly_avg = df.groupby("hour")[pollutants].mean()

plt.figure(figsize=(10, 6))
for col in pollutants:
    plt.plot(hourly_avg.index, hourly_avg[col], label=col)

plt.xlabel("Hour of Day")
plt.ylabel("Average Scaled Concentration")
plt.title("Hourly Pollution Trends")
plt.legend()
plt.grid(True)
plt.savefig(FIG_DIR + "hourly_pollution_trends.png")
plt.show()

# ---------------- 4. SEASONAL POLLUTION TRENDS ----------------
print("Plotting seasonal pollution trends...")

seasonal_avg = df.groupby("season_code")[pollutants].mean()

plt.figure(figsize=(10, 6))
seasonal_avg.plot(kind="bar")
plt.xlabel("Season Code")
plt.ylabel("Average Scaled Concentration")
plt.title("Seasonal Pollution Trends")
plt.grid(True)
plt.savefig(FIG_DIR + "seasonal_pollution_trends.png")
plt.show()

print("EDA completed. Figures saved in:", FIG_DIR)
