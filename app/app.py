"""
Fitbit Calorie Burn Prediction & Workout Pattern Clustering — Streamlit App
Run with: streamlit run app.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

st.set_page_config(page_title="Fitbit ML App", page_icon="🏃", layout="wide")
st.title("🏃 Fitbit: Calorie Burn Prediction & Workout Pattern Clustering")

tab1, tab2 = st.tabs(["🔥 Calorie Prediction", "🧩 Workout Clustering"])

# ------------------------------------------------------------------
# TAB 1 — Regression
# ------------------------------------------------------------------
with tab1:
    st.header("Predict Calories Burned")

    @st.cache_resource
    def load_reg_artifacts():
        model = joblib.load(os.path.join(OUT, "best_regressor.pkl"))
        scaler = joblib.load(os.path.join(OUT, "reg_scaler.pkl"))
        columns = joblib.load(os.path.join(OUT, "reg_feature_columns.pkl"))
        needs_scaling = joblib.load(os.path.join(OUT, "reg_needs_scaling.pkl"))
        results = pd.read_csv(os.path.join(OUT, "regression_results.csv"))
        return model, scaler, columns, needs_scaling, results

    model, scaler, feature_cols, needs_scaling, results_df = load_reg_artifacts()

    with st.expander("📊 Model comparison (all 8 algorithms)"):
        st.dataframe(results_df, use_container_width=True)
        st.caption("Best model selected by highest R² is used for predictions below.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 15, 70, 30)
        weight = st.slider("Weight (kg)", 40.0, 150.0, 70.0)
        height = st.slider("Height (m)", 1.45, 2.05, 1.70)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        max_bpm = st.slider("Max BPM", 140, 210, 180)
        avg_bpm = st.slider("Avg BPM", 90, 200, 140)
        resting_bpm = st.slider("Resting BPM", 45, 90, 65)
        fat_pct = st.slider("Fat %", 5.0, 45.0, 22.0)
    with col3:
        duration = st.slider("Session Duration (hours)", 0.2, 2.5, 1.0)
        water = st.slider("Water Intake (liters)", 0.5, 5.0, 2.0)
        freq = st.slider("Workout Frequency (days/week)", 1, 7, 4)
        workout_type = st.selectbox("Workout Type", ["Cardio", "Strength", "HIIT", "Yoga", "Mixed"])
        experience = st.selectbox(
            "Experience Level", [0, 1, 2, 3],
            format_func=lambda x: {0: "0 - Beginner", 1: "1", 2: "2", 3: "3 - Advanced"}[x],
        )

    if st.button("Predict Calories Burned", type="primary"):
        bmi = weight / (height ** 2)

        row = {
            "Age": age, "Weight (kg)": weight, "Height (m)": height, "BMI": bmi,
            "Fat_Percentage": fat_pct, "Max_BPM": max_bpm, "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm, "Session_Duration (hours)": duration,
            "Water_Intake (liters)": water, "Workout_Frequency (days/week)": freq,
            "Experience_Level_enc": experience,
            "Gender_Male": 1 if gender == "Male" else 0,
            "Workout_Type_HIIT": 1 if workout_type == "HIIT" else 0,
            "Workout_Type_Mixed": 1 if workout_type == "Mixed" else 0,
            "Workout_Type_Strength": 1 if workout_type == "Strength" else 0,
            "Workout_Type_Yoga": 1 if workout_type == "Yoga" else 0,
        }
        input_df = pd.DataFrame([row])
        # align to training columns (fills any missing dummy cols with 0)
        input_df = input_df.reindex(columns=feature_cols, fill_value=0)

        if needs_scaling:
            input_scaled, _ = (input_df, None)
            input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_cols)
            pred = model.predict(input_scaled)[0]
        else:
            pred = model.predict(input_df)[0]

        st.success(f"### Predicted Calories Burned: **{pred:.0f} kcal**")

# ------------------------------------------------------------------
# TAB 2 — Clustering
# ------------------------------------------------------------------
with tab2:
    st.header("Workout Pattern Clusters")

    @st.cache_resource
    def load_cluster_artifacts():
        scaler = joblib.load(os.path.join(OUT, "cluster_scaler.pkl"))
        pca = joblib.load(os.path.join(OUT, "cluster_pca.pkl"))
        kmeans = joblib.load(os.path.join(OUT, "kmeans_model.pkl"))
        columns = joblib.load(os.path.join(OUT, "cluster_feature_columns.pkl"))
        clustered = pd.read_csv(os.path.join(OUT, "clustered_data.csv"))
        centroids = pd.read_csv(os.path.join(OUT, "cluster_centroids.csv"))
        return scaler, pca, kmeans, columns, clustered, centroids

    c_scaler, pca, kmeans, c_columns, clustered_df, centroids_df = load_cluster_artifacts()

    n_clusters = clustered_df["Cluster"].nunique()
    st.metric("Number of clusters found", n_clusters)

    st.subheader("Cluster centroids (avg physiological profile)")
    st.dataframe(centroids_df, use_container_width=True)

    st.subheader("PCA scatter (first 2 components, colored by cluster)")
    X = clustered_df[c_columns]
    X_scaled = c_scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clustered_df["Cluster"],
                          cmap="viridis", alpha=0.6, s=15)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    legend = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend)
    st.pyplot(fig)

    st.subheader("Cluster composition by Experience Level")
    ct = pd.crosstab(clustered_df["Cluster"], clustered_df["Experience_Level"], normalize="index") * 100
    st.bar_chart(ct)

    st.subheader("Cluster composition by Workout Type")
    ct2 = pd.crosstab(clustered_df["Cluster"], clustered_df["Workout_Type"], normalize="index") * 100
    st.bar_chart(ct2)

    st.caption(
        "Clusters are derived from BPM, session duration, hydration, body composition, "
        "and experience — not from Workout_Type — so the charts above show how those "
        "*emerge* per cluster rather than define it."
    )
