import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tourism_clustering_jawabarat.config import COLOR_PALETTE, COORDINATES

# Set style for Matplotlib
plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

def get_color_map(labels: list) -> dict:
    """
    Generate color mapping: if the label is in COLOR_PALETTE, use it.
    Otherwise, assign dynamically from Plotly's Safe color sequence.
    """
    unique_labels = sorted(list(set(labels)))
    fallback_colors = px.colors.qualitative.Safe
    color_map = {}
    
    for i, label in enumerate(unique_labels):
        if label in COLOR_PALETTE:
            color_map[label] = COLOR_PALETTE[label]
        else:
            color_map[label] = fallback_colors[i % len(fallback_colors)]
    return color_map


def plot_elbow_curve_plotly(k_range: list, inertias: list) -> go.Figure:
    """
    Generate interactive Plotly Elbow Curve.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        marker=dict(size=10, color="#6366F1", line=dict(width=2, color="white")),
        line=dict(width=4, color="#6366F1"),
        name="Inersia"
    ))
    
    # Highlight potential optimal K (e.g., K=3)
    optimal_idx = k_range.index(3) if 3 in k_range else 1
    fig.add_trace(go.Scatter(
        x=[k_range[optimal_idx]],
        y=[inertias[optimal_idx]],
        mode="markers",
        marker=dict(size=18, color="#F59E0B", symbol="star", line=dict(width=2, color="white")),
        name="K Rekomendasi (3)"
    ))

    fig.update_layout(
        title="Metode Elbow (Elbow Method) untuk Mencari K Optimal",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Inersia (SSE)",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_silhouette_scores_plotly(k_range: list, scores: list) -> go.Figure:
    """
    Generate interactive Plotly Silhouette Scores chart.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=k_range,
        y=scores,
        marker_color=["#10B981" if s == max(scores) else "#93C5FD" for s in scores],
        name="Silhouette Score"
    ))
    
    fig.update_layout(
        title="Silhouette Score untuk Setiap K (Lebih Tinggi Lebih Baik)",
        xaxis_title="Jumlah Cluster (K)",
        yaxis_title="Silhouette Score",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_clusters_3d_plotly(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = "cluster_label") -> go.Figure:
    """
    Generate interactive 3D Scatter Plot for clusters.
    """
    # Mapping colors
    color_map = get_color_map(df[color_col])
    
    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        hover_name="nama_kabupaten_kota",
        color_discrete_map=color_map,
        labels={
            x_col: "Total Wisatawan Nusantara",
            y_col: "Total Wisatawan Mancanegara",
            z_col: "Rata-rata Pengunjung Tahunan",
            color_col: "Klaster Potensi"
        },
        title="Visualisasi Klasterisasi 3D Interaktif"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(240, 240, 240)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            yaxis=dict(backgroundcolor="rgb(230, 230, 230)", gridcolor="white", showbackground=True, zerolinecolor="white"),
            zaxis=dict(backgroundcolor="rgb(220, 220, 220)", gridcolor="white", showbackground=True, zerolinecolor="white"),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_clusters_2d_plotly(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = "cluster_label") -> go.Figure:
    """
    Generate interactive 2D Scatter Plot.
    """
    color_map = get_color_map(df[color_col])
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="nama_kabupaten_kota",
        size="total_pengunjung",
        color_discrete_map=color_map,
        labels={
            x_col: "Wisatawan Nusantara",
            y_col: "Wisatawan Mancanegara",
            color_col: "Klaster Potensi"
        },
        title=f"Sebaran Klaster: {x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}"
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
    return fig

def plot_comparison_bar_plotly(df: pd.DataFrame, feature_col: str, title: str, color_col: str = "cluster_label") -> go.Figure:
    """
    Generate an interactive bar chart ranking kabupaten/kota based on a feature, colored by cluster.
    """
    color_map = get_color_map(df[color_col])
    
    sorted_df = df.sort_values(by=feature_col, ascending=True)
    
    fig = px.bar(
        sorted_df,
        x=feature_col,
        y="nama_kabupaten_kota",
        color=color_col,
        orientation="h",
        color_discrete_map=color_map,
        labels={
            feature_col: feature_col.replace("_", " ").title(),
            "nama_kabupaten_kota": "Daerah",
            color_col: "Klaster Potensi"
        },
        title=title,
        height=600
    )
    fig.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
    return fig

def plot_spasial_map_plotly(df: pd.DataFrame, color_col: str = "cluster_label") -> go.Figure:
    """
    Generate interactive Mapbox Scatter map using geographical coordinates.
    """
    color_map = get_color_map(df[color_col])
    
    # Map coordinates to the dataframe
    map_df = df.copy()
    map_df["latitude"] = map_df["nama_kabupaten_kota"].map(lambda x: COORDINATES.get(x, [0.0, 0.0])[0])
    map_df["longitude"] = map_df["nama_kabupaten_kota"].map(lambda x: COORDINATES.get(x, [0.0, 0.0])[1])
    
    # Remove any counties with [0.0, 0.0] coordinates if not mapped
    map_df = map_df[(map_df["latitude"] != 0.0) & (map_df["longitude"] != 0.0)]
    
    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color=color_col,
        size="rata_rata_tahunan",
        hover_name="nama_kabupaten_kota",
        hover_data={
            "latitude": False,
            "longitude": False,
            "total_pengunjung": ":,",
            "rata_rata_tahunan": ":,"
        },
        color_discrete_map=color_map,
        zoom=7.5,
        center={"lat": -6.9175, "lon": 107.6191},  # Centered around Bandung
        mapbox_style="open-street-map",
        title="Distribusi Spasial Klaster Potensi Pariwisata Jawa Barat",
        height=550
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame, feature_cols: list) -> plt.Figure:
    """
    Matplotlib-based correlation heatmap.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[feature_cols].corr()
    sns.heatmap(
        corr, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f", 
        linewidths=0.5, 
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Matriks Korelasi Fitur", fontsize=12, pad=15)
    plt.tight_layout()
    return fig

def plot_model_comparison_bar(metrics_df: pd.DataFrame, metric_col: str, title: str, higher_is_better: bool = True) -> go.Figure:
    """
    Generate an interactive Plotly bar chart comparing a metric across different models.
    """
    # Sort for visual aesthetics
    sorted_df = metrics_df.sort_values(by=metric_col, ascending=higher_is_better)
    
    # Highlight color for best value
    best_idx = sorted_df[metric_col].idxmax() if higher_is_better else sorted_df[metric_col].idxmin()
    best_model = metrics_df.loc[best_idx, "Model"]
    
    colors = ["#4F46E5" if m == best_model else "#93C5FD" for m in sorted_df["Model"]]
    
    fig = px.bar(
        sorted_df,
        x="Model",
        y=metric_col,
        title=title,
        text_auto=".4f",
        labels={"Model": "Algoritma", metric_col: metric_col},
        color="Model",
        color_discrete_map={m: c for m, c in zip(sorted_df["Model"], colors)}
    )
    
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False
    )
    fig.update_traces(textposition="outside")
    return fig

def plot_pca_2d_plotly(df: pd.DataFrame, X_scaled: np.ndarray, labels: np.ndarray, model_name: str) -> go.Figure:
    """
    Apply PCA to reduce features to 2D and generate an interactive scatter plot of the clusters.
    """
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame({
        "PC 1": X_pca[:, 0],
        "PC 2": X_pca[:, 1],
        "nama_kabupaten_kota": df["nama_kabupaten_kota"],
        "Cluster": [f"Cluster {l + 1}" for l in labels]
    })
    
    # Ensure consistent sorting for colors
    pca_df = pca_df.sort_values(by="Cluster")
    
    fig = px.scatter(
        pca_df,
        x="PC 1",
        y="PC 2",
        color="Cluster",
        hover_name="nama_kabupaten_kota",
        title=f"Visualisasi Hasil Klasterisasi PCA 2D - {model_name}",
        labels={"PC 1": "Principal Component 1", "PC 2": "Principal Component 2", "Cluster": "Klaster"},
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    fig.update_traces(marker=dict(size=12, opacity=0.85, line=dict(width=1, color="white")))
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

