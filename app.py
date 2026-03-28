# ================= AIR POLLUTION INTELLIGENCE PLATFORM =================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import requests
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Air Pollution Intelligence Platform",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------------
st.markdown("""
<style>
.big-title {font-size:34px; font-weight:800;}
.subtitle {color:#9aa0a6;}
.metric-box {
    background:#0e1117;
    padding:20px;
    border-radius:16px;
    border:1px solid #2c2f33;
    text-align:center;
}
.metric-title {font-size:15px; color:#9aa0a6;}
.metric-value {font-size:26px; font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------
st.markdown("<div class='big-title'>🌍 Air Pollution Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-driven pollution source identification & smart-city analytics</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
st.sidebar.title("⚙️ Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload Air Quality CSV", type=["csv"])
run_clustering = st.sidebar.button("🚀 Run Clustering Engine")

# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    df["PM_ratio"] = df["PM10"] / (df["PM2.5"] + 1e-6)
    df["NO2_CO_ratio"] = df["NO2"] / (df["CO"] + 1e-6)
    df["SO2_NO2_ratio"] = df["SO2"] / (df["NO2"] + 1e-6)

    df["is_rush_hour"] = df["hour"].isin([7,8,9,17,18,19]).astype(int)
    df["is_night"] = df["hour"].isin([22,23,0,1,2,3,4]).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

    def season_code(m):
        if m in [12,1,2]: return 0
        elif m in [3,4,5]: return 1
        elif m in [6,7,8,9]: return 2
        else: return 3

    df["season_code"] = df["month"].apply(season_code)
    return df

FEATURES = [
    "PM_ratio","NO2_CO_ratio","SO2_NO2_ratio",
    "is_rush_hour","is_night","is_weekend","season_code"
]

cluster_map = {
    0: "🚗 Traffic Emissions",
    1: "🏗 Construction & Dust",
    2: "🏭 Industrial Emissions",
    3: "🌿 Seasonal / Background"
}

# ---------------------------------------------------------------------
# LIVE AQI API (OpenAQ)
# ---------------------------------------------------------------------
def fetch_live_aqi(city):
    url = "https://api.openaq.org/v2/latest"
    params = {"city": city, "limit": 1}
    r = requests.get(url, params=params, timeout=10)
    data = r.json()

    if "results" not in data or not data["results"]:
        return None

    measurements = data["results"][0]["measurements"]
    return {m["parameter"]: m["value"] for m in measurements}

# ---------------------------------------------------------------------
# POLICY REPORT GENERATOR
# ---------------------------------------------------------------------
def generate_policy_report(summary):
    path = "Air_Pollution_Policy_Report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Air Pollution Pattern Analysis Report</b>", styles["Title"]))
    story.append(Paragraph(
        "This report summarizes pollution source patterns identified using unsupervised machine learning.",
        styles["Normal"]
    ))

    for src, count in summary.items():
        story.append(Paragraph(f"<b>{src}</b>: {count} observations", styles["Normal"]))

    story.append(Paragraph(
        "Policy Recommendation: Apply source-specific interventions such as traffic regulation, industrial monitoring, and construction dust control.",
        styles["Normal"]
    ))

    doc.build(story)
    return path

# ---------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------
tabs = st.tabs([
    "🏠 Overview",
    "📂 Data Explorer",
    "🧠 Clustering",
    "🔍 Insights",
    "🧪 Scenario Simulator",
    "🗺 City AQI Map",
    "🌐 Live AQI",
    "🛰 Satellite Map",
    "📄 Policy Report"
])

# ================= TAB 1 =================
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("<div class='metric-box'><div class='metric-title'>Pollution Sources</div><div class='metric-value'>4</div></div>", unsafe_allow_html=True)
    col2.markdown("<div class='metric-box'><div class='metric-title'>ML Models</div><div class='metric-value'>3</div></div>", unsafe_allow_html=True)
    col3.markdown("<div class='metric-box'><div class='metric-title'>Manual Labels</div><div class='metric-value'>0</div></div>", unsafe_allow_html=True)
    col4.markdown("<div class='metric-box'><div class='metric-title'>Smart-City Ready</div><div class='metric-value'>Yes</div></div>", unsafe_allow_html=True)

# ================= TAB 2 =================
with tabs[1]:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        st.dataframe(df.describe())

# ================= TAB 3 =================
with tabs[2]:
    if uploaded_file and run_clustering:
        df_feat = engineer_features(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_feat[FEATURES])

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=25)
        df_feat["cluster"] = kmeans.fit_predict(X_pca)

        st.session_state["models"] = (scaler, pca, kmeans, df_feat)

        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_feat["cluster"], palette="tab10", ax=ax)
        st.pyplot(fig)

# ================= TAB 4 =================
with tabs[3]:
    if "models" in st.session_state:
        _, _, _, df_feat = st.session_state["models"]
        df_feat["source"] = df_feat["cluster"].map(cluster_map)
        st.dataframe(df_feat["source"].value_counts())

# ================= TAB 5 =================
with tabs[4]:
    if "models" in st.session_state:
        scaler, pca, kmeans, _ = st.session_state["models"]

        with st.form("predict"):
            pm25 = st.number_input("PM2.5", 120.0)
            pm10 = st.number_input("PM10", 180.0)
            no2 = st.number_input("NO2", 80.0)
            co = st.number_input("CO", 2.0)
            so2 = st.number_input("SO2", 10.0)
            o3 = st.number_input("O3", 30.0)
            hour = st.slider("Hour", 0, 23, 8)
            dow = st.slider("Day of Week", 0, 6, 1)
            month = st.slider("Month", 1, 12, 11)
            submit = st.form_submit_button("Predict")

        if submit:
            user = pd.DataFrame([{
                "PM2.5":pm25,"PM10":pm10,"NO2":no2,"CO":co,"SO2":so2,"O3":o3,
                "hour":hour,"day_of_week":dow,"month":month
            }])
            uf = engineer_features(user)
            c = kmeans.predict(pca.transform(scaler.transform(uf[FEATURES])))[0]
            st.success(cluster_map[c])

# ================= TAB 6 =================
with tabs[5]:
    city_data = pd.DataFrame({
        "lat":[28.61,19.07,22.57,13.08,12.97],
        "lon":[77.20,72.87,88.36,80.27,77.59],
        "PM2.5":[180,120,95,85,70]
    })
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("HeatmapLayer", city_data, get_position=["lon","lat"], get_weight="PM2.5")],
        initial_view_state=pdk.ViewState(latitude=22.5, longitude=78.9, zoom=4.5)
    ))

# ================= TAB 7 =================
with tabs[6]:
    city = st.selectbox("Select City", ["Delhi","Mumbai","Kolkata","Chennai"])
    if st.button("Fetch Live AQI"):
        data = fetch_live_aqi(city)
        if data:
            for k,v in data.items():
                st.metric(k.upper(), v)
        else:
            st.warning("No live data available.")

# ================= TAB 8 =================
with tabs[7]:
    st.components.v1.iframe(
        "https://earth.nullschool.net/#current/particulates/surface/level/overlay=pm2.5",
        height=600
    )

# ================= TAB 9 =================
with tabs[8]:
    if "models" in st.session_state:
        _, _, _, df_feat = st.session_state["models"]
        summary = df_feat["cluster"].map(cluster_map).value_counts()

        if st.button("Generate Policy Report"):
            path = generate_policy_report(summary)
            with open(path, "rb") as f:
                st.download_button("Download PDF", f, file_name=path)
            st.success("Report generated successfully.")        