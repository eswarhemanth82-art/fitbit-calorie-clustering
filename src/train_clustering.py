"""
Task 2 — Unsupervised Learning: Workout Pattern Clustering
Drops Workout_Type (kept aside only for post-hoc comparison), scales
features, applies PCA, then KMeans. Also runs Hierarchical + DBSCAN
for comparison. Evaluates with Silhouette Score.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

from preprocessing import load_data, handle_missing, cap_outliers, scale_features, PROJECT_ROOT

RANDOM_STATE = 42
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main():
    df = load_data()
    df = handle_missing(df)
    df = cap_outliers(df, cols=["Calories_Burned", "Avg_BPM"])

    # Keep Workout_Type and Experience_Level aside for interpretation only
    workout_type_labels = df["Workout_Type"].copy()
    experience_labels = df["Experience_Level"].copy()

    # Focus the feature set on workout intensity & physiological response —
    # demographic-only columns (Age, Height, Weight, Gender) are dropped since
    # they dilute the intensity signal without describing how someone trains.
    intensity_features = [
        "Max_BPM", "Avg_BPM", "Resting_BPM", "Session_Duration (hours)",
        "Fat_Percentage", "BMI", "Water_Intake (liters)",
        "Workout_Frequency (days/week)", "Experience_Level",
    ]
    cluster_df = df[intensity_features].copy()
    if cluster_df["Experience_Level"].dtype == object:
        exp_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        cluster_df["Experience_Level_enc"] = cluster_df["Experience_Level"].map(exp_order)
    else:
        cluster_df["Experience_Level_enc"] = cluster_df["Experience_Level"]
    cluster_df = cluster_df.drop(columns=["Experience_Level"])

    X_scaled, scaler = scale_features(cluster_df)

    # ---- PCA ----
    pca = PCA(n_components=0.90, random_state=RANDOM_STATE)  # keep 90% variance
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_pca.shape[1]} components explain "
          f"{pca.explained_variance_ratio_.sum():.2%} variance")

    # ---- KMeans: pick k via silhouette sweep ----
    best_k, best_score, best_labels, best_model = None, -1, None, None
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f"KMeans k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score, best_labels, best_model = k, score, labels, km

    print(f"\nBest KMeans: k={best_k}, silhouette={best_score:.4f} "
          f"({'PASS' if best_score >= 0.15 else 'FAIL'} acceptance criterion >= 0.15)")

    # ---- Comparison methods (optional) ----
    hier = AgglomerativeClustering(n_clusters=best_k)
    hier_labels = hier.fit_predict(X_pca)
    hier_score = silhouette_score(X_pca, hier_labels)
    print(f"Hierarchical (k={best_k}): silhouette={hier_score:.4f}")

    dbscan = DBSCAN(eps=1.2, min_samples=10)
    db_labels = dbscan.fit_predict(X_pca)
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    if n_db_clusters >= 2:
        db_score = silhouette_score(X_pca, db_labels)
        print(f"DBSCAN: {n_db_clusters} clusters (+noise), silhouette={db_score:.4f}")
    else:
        print("DBSCAN: fewer than 2 clusters found with current eps/min_samples")

    # ---- Interpretation: cluster centroids on original (unscaled) features ----
    result_df = cluster_df.copy()
    result_df["Cluster"] = best_labels
    result_df["Workout_Type"] = workout_type_labels.values
    result_df["Experience_Level"] = experience_labels.values

    centroid_summary = result_df.groupby("Cluster")[
        ["Avg_BPM", "Session_Duration (hours)", "Fat_Percentage", "BMI", "Water_Intake (liters)"]
    ].mean().round(2)
    print("\n=== Cluster Centroids (key physiological features) ===")
    print(centroid_summary.to_string())

    print("\n=== Cluster size distribution ===")
    print(result_df["Cluster"].value_counts().sort_index().to_string())

    print("\n=== Workout_Type composition per cluster (%) ===")
    ct = pd.crosstab(result_df["Cluster"], result_df["Workout_Type"], normalize="index") * 100
    print(ct.round(1).to_string())

    print("\n=== Experience_Level composition per cluster (%) ===")
    ct2 = pd.crosstab(result_df["Cluster"], result_df["Experience_Level"], normalize="index") * 100
    print(ct2.round(1).to_string())

    # ---- Save artifacts ----
    joblib.dump(scaler, os.path.join(OUTPUTS_DIR, "cluster_scaler.pkl"))
    joblib.dump(pca, os.path.join(OUTPUTS_DIR, "cluster_pca.pkl"))
    joblib.dump(best_model, os.path.join(OUTPUTS_DIR, "kmeans_model.pkl"))
    joblib.dump(list(cluster_df.columns), os.path.join(OUTPUTS_DIR, "cluster_feature_columns.pkl"))
    result_df.to_csv(os.path.join(OUTPUTS_DIR, "clustered_data.csv"), index=False)
    centroid_summary.to_csv(os.path.join(OUTPUTS_DIR, "cluster_centroids.csv"))

    print("\nSaved: cluster_scaler.pkl, cluster_pca.pkl, kmeans_model.pkl, clustered_data.csv")


if __name__ == "__main__":
    main()
