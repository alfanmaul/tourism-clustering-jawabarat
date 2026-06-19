import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from tourism_clustering_jawabarat.config import (
    DEFAULT_N_CLUSTERS,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
)
from tourism_clustering_jawabarat.features import build_features, normalize_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURE_COLS = ["total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]

def evaluate_kmeans_range(max_k: int = 10) -> dict:
    """
    Evaluate K-Means for K from 2 to max_k.
    Calculates Inertia (Elbow Method) and Silhouette Score.
    """
    if not PROCESSED_DATA_PATH.exists():
        build_features()
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    scaled_df, _ = normalize_features(df, FEATURE_COLS)
    X = scaled_df.values
    
    k_range = list(range(2, max_k + 1))
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        
    return {
        "k_range": k_range,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores
    }

def train_and_save_kmeans(n_clusters: int = DEFAULT_N_CLUSTERS) -> tuple:
    """
    Train K-Means clustering model, label the clusters, and save model & scaler.
    Labels: Potensi Rendah, Potensi Sedang, Potensi Tinggi
    """
    if not PROCESSED_DATA_PATH.exists():
        build_features()
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    scaled_df, scaler = normalize_features(df, FEATURE_COLS)
    X = scaled_df.values
    
    logging.info(f"Training K-Means with K={n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X)
    
    df["cluster_raw"] = cluster_labels
    
    # Label mapping based on average "total_pengunjung" per cluster
    # We sort the cluster IDs by their mean visitors in ascending order
    cluster_means = df.groupby("cluster_raw")["total_pengunjung"].mean().sort_values()
    sorted_clusters = cluster_means.index.tolist()
    
    # Create potential labels dynamically based on n_clusters
    label_map = {}
    if n_clusters == 3:
        labels = ["Potensi Rendah", "Potensi Sedang", "Potensi Tinggi"]
        for idx, cluster_id in enumerate(sorted_clusters):
            label_map[cluster_id] = labels[idx]
    elif n_clusters == 2:
        labels = ["Potensi Rendah", "Potensi Tinggi"]
        for idx, cluster_id in enumerate(sorted_clusters):
            label_map[cluster_id] = labels[idx]
    else:
        # Generic ranking label if user chooses K != 2 or 3
        for idx, cluster_id in enumerate(sorted_clusters):
            label_map[cluster_id] = f"Potensi Tingkat {idx+1}"
            
    df["cluster_label"] = df["cluster_raw"].map(label_map)
    
    # Save the models
    logging.info(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    logging.info(f"Saving model to {MODEL_PATH}")
    joblib.dump(kmeans, MODEL_PATH)
    
    # Reorder columns slightly for presentation
    # Move clusters labels to the front or keep them clean
    cols = ["nama_kabupaten_kota", "cluster_label", "total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]
    result_df = df[cols]
    
    logging.info("Clustering completed successfully.")
    return result_df, kmeans, scaler, label_map

def benchmark_clustering_models(n_clusters: int = DEFAULT_N_CLUSTERS) -> tuple:
    """
    Benchmark multiple clustering algorithms (K-Means, Agglomerative, Birch, GMM)
    using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    Returns:
        metrics_df (pd.DataFrame): DataFrame of metrics for each model.
        labels_dict (dict): Dictionary mapping model name to its predicted labels.
    """
    if not PROCESSED_DATA_PATH.exists():
        build_features()
        
    df = pd.read_csv(PROCESSED_DATA_PATH)
    scaled_df, scaler = normalize_features(df, FEATURE_COLS)
    X = scaled_df.values
    
    # 1. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    # 2. Agglomerative Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    agg_labels = agg.fit_predict(X)
    
    # 3. Birch
    birch = Birch(n_clusters=n_clusters)
    birch_labels = birch.fit_predict(X)
    
    # 4. GMM
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    gmm.fit(X)
    gmm_labels = gmm.predict(X)
    
    labels_dict = {
        "K-Means": kmeans_labels,
        "Agglomerative": agg_labels,
        "Birch": birch_labels,
        "GMM": gmm_labels
    }
    
    metrics_list = []
    for model_name, labels in labels_dict.items():
        sil = silhouette_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        metrics_list.append({
            "Model": model_name,
            "Silhouette Score": sil,
            "Calinski-Harabasz Score": ch,
            "Davies-Bouldin Score": db
        })
        
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df, labels_dict

if __name__ == "__main__":
    # Test range evaluation
    evals = evaluate_kmeans_range()
    print("Inertias:", evals["inertias"])
    print("Silhouette Scores:", evals["silhouette_scores"])
    
    # Test training
    results, kmeans, scaler, lmap = train_and_save_kmeans(n_clusters=3)
    print("\nTraining Results Preview:")
    print(results.head())
    print("\nCluster Label Map:", lmap)
