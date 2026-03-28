

Air pollution is a major environmental and public health concern. Different pollution sources such as traffic, industrial activities, construction dust, and seasonal factors contribute differently to air quality deterioration.

This project presents an end-to-end unsupervised machine learning framework to analyze large-scale air quality data and automatically identify pollution source patterns using clustering techniques.

The system processes raw air quality data, engineers meaningful pollution features, applies clustering algorithms, evaluates cluster quality, and interprets clusters as real-world pollution sources.

🎯 Objectives

Process and clean large-scale air quality datasets

Engineer pollution pattern features (ratios, temporal indicators)

Apply unsupervised clustering (K-Means, Hierarchical, DBSCAN)

Determine the optimal number of clusters using evaluation metrics

Interpret clusters as pollution sources:

Traffic emissions

Industrial emissions

Construction & dust pollution

Seasonal/background pollution

Provide research-ready visualizations and insights

🧠 Why Clustering?

Pollution source labels are not available

Clustering helps discover hidden patterns without supervision

Enables data-driven source attribution

Suitable for large, real-world environmental datasets

🗂️ Project Folder Structure
Air-Quality-Index-Pattern-Clustering/
│
├── data/
│   ├── raw/                    # Raw input datasets
│   └── processed/              # Cleaned & feature-engineered datasets
│
├── src/
│   ├── 01_data_loading.py
│   ├── 02_data_cleaning.py
│   ├── 03_feature_engineering.py
│   ├── 04_eda.py
│   ├── 05_clustering_kmeans.py
│   ├── 06_clustering_hierarchical.py
│   ├── 07_clustering_dbscan.py
│   ├── 08_evaluation.py
│   ├── 09_cluster_interpretation.py
│   └── 10_cluster_visualization.py
│
├── results/
│   ├── figures/                # All generated plots
│   └── cluster_profiles.csv    # Cluster-wise statistics
│
├── notebooks/                  # (Optional) Jupyter notebooks
│
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies

📊 Dataset Description

Source: Public air quality datasets (India / Beijing)

Granularity: Hourly observations

Key Pollutants:

PM2.5, PM10

NO2, SO2, CO, O3

Additional Features:

AQI

Temporal features (hour, day, season)

Station ID

⚙️ Tech Stack & Libraries

Language: Python

Libraries:

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run Files in Sequence
python src/01_data_loading.py
python src/02_data_cleaning.py
python src/03_feature_engineering.py
python src/04_eda.py
python src/05_clustering_kmeans.py
python src/06_clustering_hierarchical.py
python src/07_clustering_dbscan.py
python src/08_evaluation.py
python src/09_cluster_interpretation.py
python src/10_cluster_visualization.py


📌 Important:
Evaluation and visualization steps use sampling for computational efficiency.

📈 Clustering Techniques Used
🔹 K-Means (Primary Model)

Pattern-based features

PCA for dimensionality reduction

Optimal clusters selected using evaluation metrics

🔹 Hierarchical Clustering

Used for structure validation on sampled data

🔹 DBSCAN

Used for anomaly and extreme pollution event detection

📐 Cluster Evaluation Metrics

Elbow Method

Silhouette Score

Davies–Bouldin Index

✅ Optimal number of clusters selected: K = 4

🏭 Identified Pollution Sources
Cluster	Pollution Source
0	Traffic Emissions
1	Construction & Dust
2	Industrial Emissions
3	Seasonal / Background
Outlier	Extreme Pollution Events
📊 Key Visualizations

PCA cluster scatter plot

Cluster-wise pollutant heatmap

Hourly pollution patterns

PM2.5 boxplot per cluster

All figures are saved in:

results/figures/

🌱 Key Findings

Traffic pollution peaks during rush hours

Industrial emissions dominate at night

Construction dust causes high PM10 variability

Seasonal effects significantly influence background pollution

⚠️ Limitations

Unsupervised source labeling

Limited meteorological modeling

Station-level spatial resolution

Sensitivity to extreme outliers

🚀 Future Scope

Integration with meteorological data

Deep learning for temporal modeling

GIS-based spatial analysis

Real-time pollution monitoring

Pollution source fingerprint generation (patent-ready)
