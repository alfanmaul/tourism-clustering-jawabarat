import logging
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from tourism_clustering_jawabarat.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from tourism_clustering_jawabarat.clustering import (
    benchmark_clustering_models,
    load_and_apply_kmeans,
)
from tourism_clustering_jawabarat.visualizations import (
    plot_clusters_3d_plotly,
    plot_clusters_2d_plotly,
    plot_comparison_bar_plotly,
    plot_spasial_map_plotly,
    plot_correlation_matrix,
    plot_model_comparison_bar,
    plot_pca_2d_plotly,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Startup Diagnostics ---
logging.info(f"[App] Current working directory: {os.getcwd()}")
logging.info(f"[App] RAW_DATA_PATH: {RAW_DATA_PATH} | exists: {RAW_DATA_PATH.exists()}")
logging.info(f"[App] PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH} | exists: {PROCESSED_DATA_PATH.exists()}")
logging.info(f"[App] MODEL_PATH: {MODEL_PATH} | exists: {MODEL_PATH.exists()}")
logging.info(f"[App] SCALER_PATH: {SCALER_PATH} | exists: {SCALER_PATH.exists()}")

# Page Configuration
st.set_page_config(
    page_title="Analisis Potensi Pariwisata Jawa Barat",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling CSS for premium look
st.markdown("""
<style>
    /* Main body adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Card design */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: white;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 8px;
    }
    
    /* Title Banner styling */
    .banner-title {
        background: linear-gradient(90deg, #6366F1 0%, #EC4899 50%, #F59E0B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 36px;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .banner-subtitle {
        color: #64748b;
        font-size: 18px;
        margin-bottom: 30px;
        text-align: center;
        font-weight: 400;
    }
    
    /* Sidebar Navigation */
    .sidebar-header {
        font-weight: 800;
        font-size: 20px;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# Helper functions to load data (NO API calls)
# ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_raw_dataset():
    """
    Load the raw dataset from the local JSON file.
    Does NOT call any API – the file must exist in the repository.
    """
    logging.info(f"[App] Loading raw dataset from {RAW_DATA_PATH}")

    if not RAW_DATA_PATH.exists():
        logging.error(f"[App] Raw data file NOT FOUND at {RAW_DATA_PATH}")
        st.error(
            f"❌ File dataset mentah tidak ditemukan di: `{RAW_DATA_PATH}`.\n\n"
            f"Pastikan file `data/raw/raw_tourism_data.json` sudah di-commit "
            f"ke repository dan tidak di-exclude oleh `.gitignore`."
        )
        return pd.DataFrame()

    try:
        import json
        with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "data" not in data or not data["data"]:
            st.error("❌ File JSON ditemukan tetapi key 'data' kosong atau tidak ada.")
            return pd.DataFrame()

        df = pd.DataFrame(data["data"])
        logging.info(f"[App] Raw dataset loaded: {len(df)} rows.")
        return df

    except Exception as e:
        logging.error(f"[App] Failed to read raw data: {e}")
        st.error(f"❌ Gagal membaca data lokal: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_processed_dataset():
    """
    Load the processed (feature-engineered) dataset from CSV.
    Does NOT call build_features() or any API – the file must exist.
    """
    logging.info(f"[App] Loading processed dataset from {PROCESSED_DATA_PATH}")

    if not PROCESSED_DATA_PATH.exists():
        logging.error(f"[App] Processed data file NOT FOUND at {PROCESSED_DATA_PATH}")
        st.error(
            f"❌ File dataset hasil preprocessing tidak ditemukan di: `{PROCESSED_DATA_PATH}`.\n\n"
            f"Pastikan file `data/processed/processed_tourism_data.csv` sudah di-commit "
            f"ke repository dan tidak di-exclude oleh `.gitignore`."
        )
        return pd.DataFrame()

    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logging.info(f"[App] Processed dataset loaded: {len(df)} rows.")

        if df.empty:
            st.warning("⚠️ File processed dataset ditemukan tetapi kosong (0 baris).")

        return df

    except Exception as e:
        logging.error(f"[App] Failed to read processed data: {e}")
        st.error(f"❌ Gagal membaca data hasil pemrosesan: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────
# Sidebar setup
# ──────────────────────────────────────────────────────────

st.sidebar.markdown('<div class="sidebar-header">MENU NAVIGASI</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "📊 Evaluasi & Perbandingan Model",
        "💾 Dataset",
        "📈 Exploratory Data Analysis (EDA)",
        "🗺️ Hasil Cluster"
    ]
)

# Sidebar parameters (hardcoded to optimal K = 3)
k_selected = 3
if page in ["🗺️ Hasil Cluster", "📊 Evaluasi & Perbandingan Model"]:
    st.sidebar.write("---")
    st.sidebar.subheader("Informasi Klasterisasi")
    st.sidebar.info("""
    **Model**: K-Means
    
    **Jumlah Cluster Optimal**: 3

    **Metode Penentuan**: Elbow Method + Silhouette Score
    """)

# Group Members list for display
GROUP_MEMBERS = [
    {"nim": "20124088", "name": "ALFAN MAULANA"},
    {"nim": "20124082", "name": "TIARA AULIA SEPTIANI"},
    {"nim": "20124089", "name": "DEDE RIFKI"},
    {"nim": "20124094", "name": "RIFA'I AHMAD"},
    {"nim": "20124098", "name": "ZIDAN ZAIN SHIBGHATULLAH"},
    {"nim": "20124074", "name": "MUHAMAD HASBI"},
]


# ──────────────────────────────────────────────────────────
# Page: DATASET
# ──────────────────────────────────────────────────────────

if page == "💾 Dataset":
    st.subheader("💾 Dataset Eksplorasi (Open Data Jabar)")
    raw_df = load_raw_dataset()

    # Guard: stop if dataset is empty
    if raw_df.empty:
        st.warning("⚠️ Dataset kosong. Halaman ini tidak dapat menampilkan data.")
        st.stop()
    
    # Pre-calculate metrics
    total_records = len(raw_df)
    min_year = raw_df["tahun"].min()
    max_year = raw_df["tahun"].max()
    unique_cities = raw_df["nama_kabupaten_kota"].nunique()
    total_visitors = raw_df["jumlah_pengunjung"].sum()
    
    # Clean raw dataset column for display
    display_df = raw_df.copy()
    display_df["nama_kabupaten_kota"] = display_df["nama_kabupaten_kota"].str.upper()
    
    # 4 Cards Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Rekaman Data</div><div class="metric-value">{total_records} Baris</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Rentang Tahun</div><div class="metric-value">{min_year} - {max_year}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Kabupaten / Kota</div><div class="metric-value">{unique_cities} Daerah</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Pengunjung</div><div class="metric-value">{total_visitors:,.0f}</div></div>', unsafe_allow_html=True)
        
    st.write("---")
    
    # Interactive filters
    st.write("### Filter Pencarian Dataset")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        search_city = st.text_input("Cari Kabupaten/Kota:", "").upper()
    with col_f2:
        selected_year = st.multiselect("Pilih Tahun:", sorted(display_df["tahun"].unique().tolist()), default=None)
    with col_f3:
        selected_tourist = st.selectbox("Jenis Wisatawan:", ["SEMUA", "NUSANTARA", "MANCANEGARA"])
        
    # Filtering data
    filtered_df = display_df.copy()
    if search_city:
        filtered_df = filtered_df[filtered_df["nama_kabupaten_kota"].str.contains(search_city, na=False)]
    if selected_year:
        filtered_df = filtered_df[filtered_df["tahun"].isin(selected_year)]
    if selected_tourist != "SEMUA":
        filtered_df = filtered_df[filtered_df["jenis_wisatawan"] == selected_tourist]
        
    # Display table
    st.write(f"Menampilkan {len(filtered_df)} baris data hasil filter.")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Filtered (CSV)",
        data=csv_data,
        file_name="jabar_tourism_data_filtered.csv",
        mime="text/csv"
    )


# ──────────────────────────────────────────────────────────
# Page: EDA
# ──────────────────────────────────────────────────────────

elif page == "📈 Exploratory Data Analysis (EDA)":
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    st.write("Halaman ini menyajikan grafik interaktif untuk melihat distribusi dan tren wisatawan di Jawa Barat.")
    
    raw_df = load_raw_dataset()
    processed_df = load_processed_dataset()

    # Guard: stop if either dataset is empty
    if raw_df.empty or processed_df.empty:
        st.warning("⚠️ Dataset kosong. Halaman EDA tidak dapat ditampilkan.")
        st.stop()
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribusi & Komposisi Wisatawan", "📈 Tren Kunjungan Tahunan", "🔍 Korelasi Fitur"])
    
    with tab1:
        st.write("### Komposisi Wisatawan Nusantara vs Mancanegara")
        tourist_comp = raw_df.groupby("jenis_wisatawan")["jumlah_pengunjung"].sum().reset_index()
        fig_pie = px.pie(
            tourist_comp, 
            values="jumlah_pengunjung", 
            names="jenis_wisatawan",
            color="jenis_wisatawan",
            color_discrete_map={"NUSANTARA": "#4F46E5", "MANCANEGARA": "#F59E0B"},
            hole=0.4,
            title="Perbandingan Total Wisatawan Nusantara vs Mancanegara"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.write("### Daerah dengan Kunjungan Wisatawan Tertinggi (Semua Tahun)")
        top_cities = processed_df.sort_values(by="total_pengunjung", ascending=False)
        fig_bar = px.bar(
            top_cities,
            x="total_pengunjung",
            y="nama_kabupaten_kota",
            orientation="h",
            title="Peringkat Daerah Berdasarkan Total Pengunjung",
            color="total_pengunjung",
            color_continuous_scale="blues",
            height=650,
            labels={"total_pengunjung": "Total Pengunjung", "nama_kabupaten_kota": "Kabupaten / Kota"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with tab2:
        st.write("### Tren Pertumbuhan Kunjungan Wisatawan Tahunan")
        yearly_trend = raw_df.groupby(["tahun", "jenis_wisatawan"])["jumlah_pengunjung"].sum().reset_index()
        fig_line = px.line(
            yearly_trend,
            x="tahun",
            y="jumlah_pengunjung",
            color="jenis_wisatawan",
            markers=True,
            color_discrete_map={"NUSANTARA": "#4F46E5", "MANCANEGARA": "#EC4899"},
            title="Tren Kunjungan Wisatawan Nusantara & Mancanegara Per Tahun (2014-2024)",
            labels={"jumlah_pengunjung": "Total Pengunjung", "tahun": "Tahun", "jenis_wisatawan": "Jenis Wisatawan"}
        )
        fig_line.update_layout(xaxis=dict(tickmode="linear"))
        st.plotly_chart(fig_line, use_container_width=True)
        
        st.info("""
        **Analisis Tren:**
        Terlihat penurunan drastis pada tahun **2020 - 2021** untuk kedua jenis wisatawan. Hal ini disebabkan oleh dampak pandemi global COVID-19 yang membatasi mobilitas sosial dan pariwisata. Pemulihan ekonomi dan industri pariwisata mulai terlihat meningkat pesat kembali pada tahun **2022 - 2023**.
        """)
        
    with tab3:
        st.write("### Hubungan dan Korelasi Fitur")
        st.write("Menganalisis keeratan korelasi antara 4 fitur yang akan diumpankan ke algoritma K-Means.")
        
        cols = ["total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]
        fig_corr = plot_correlation_matrix(processed_df, cols)
        st.pyplot(fig_corr)
        
        st.write("""
        **Temuan Korelasi:**
        - Korelasi antara `total_nusantara` dan `total_pengunjung` bernilai mendekati **1.00**, menandakan bahwa mayoritas pariwisata di Jawa Barat didominasi oleh wisatawan domestik (Nusantara).
        - Korelasi `total_mancanegara` dengan fitur lainnya bernilai positif sedang, mengindikasikan bahwa persebaran wisatawan mancanegara tidak selalu terdistribusi sama dengan wisatawan Nusantara (mancanegara cenderung hanya menumpuk di beberapa daerah tertentu).
        """)


# ──────────────────────────────────────────────────────────
# Page: EVALUASI & PERBANDINGAN MODEL
# ──────────────────────────────────────────────────────────

elif page == "📊 Evaluasi & Perbandingan Model":

    # Load data and normalise features
    processed_df = load_processed_dataset()

    # Guard: stop if dataset is empty
    if processed_df.empty:
        st.error("❌ Dataset processed kosong. Tidak dapat melakukan evaluasi model.")
        st.stop()

    cols = ["total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]
    from tourism_clustering_jawabarat.features import normalize_features
    scaled_df, scaler_obj = normalize_features(processed_df, cols)
    X_scaled = scaled_df.values
    
    # Run evaluation
    metrics_df, labels_dict = benchmark_clustering_models(n_clusters=k_selected)

    # Section 4 - Ranking Model
    st.write("### 🏆 Peringkat Agregat Model Terbaik")
    df_rank = metrics_df.copy()
    df_rank["rank_sil"] = df_rank["Silhouette Score"].rank(ascending=False)
    df_rank["rank_ch"] = df_rank["Calinski-Harabasz Score"].rank(ascending=False)
    df_rank["rank_db"] = df_rank["Davies-Bouldin Score"].rank(ascending=True)
    df_rank["total_rank"] = df_rank["rank_sil"] + df_rank["rank_ch"] + df_rank["rank_db"]
    df_rank = df_rank.sort_values(by=["total_rank", "Silhouette Score"], ascending=[True, False]).reset_index(drop=True)
    
    r_cols = st.columns(4)
    medals = ["🥇 Model Terbaik", "🥈 Peringkat Kedua", "🥉 Peringkat Ketiga", "4️⃣ Peringkat Keempat"]
    for idx, col in enumerate(r_cols):
        with col:
            model_name = df_rank.loc[idx, "Model"]
            sil_val = df_rank.loc[idx, "Silhouette Score"]
            col.metric(label=medals[idx], value=model_name, delta=f"Sil Score: {sil_val:.4f}")
            
    st.write("---")

    # Section 2 - Tabel Perbandingan Model
    st.write("### 📋 Tabel Evaluasi Metrik Komparasi")
    
    # Formatting helper for pandas styled dataframe
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #047857; color: white' if v else '' for v in is_max]
        
    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #047857; color: white' if v else '' for v in is_min]
        
    # Sort by Silhouette Score descending
    sorted_metrics_df = metrics_df.sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
    
    styled_df = sorted_metrics_df.style.apply(highlight_max, subset=["Silhouette Score", "Calinski-Harabasz Score"])\
                                     .apply(highlight_min, subset=["Davies-Bouldin Score"])
    
    st.dataframe(styled_df, use_container_width=True)
   
    st.write("---")
    
    # Section 3 - Visualisasi Perbandingan Metrik
    st.write("### 📈 Grafik Perbandingan Metrik Evaluasi")
    v_col1, v_col2, v_col3 = st.columns(3)
    with v_col1:
        fig_sil_comp = plot_model_comparison_bar(metrics_df, "Silhouette Score", "Silhouette Score", higher_is_better=True)
        st.plotly_chart(fig_sil_comp, use_container_width=True)
    with v_col2:
        fig_ch_comp = plot_model_comparison_bar(metrics_df, "Calinski-Harabasz Score", "Calinski-Harabasz Score", higher_is_better=True)
        st.plotly_chart(fig_ch_comp, use_container_width=True)
    with v_col3:
        fig_db_comp = plot_model_comparison_bar(metrics_df, "Davies-Bouldin Score", "Davies-Bouldin Score", higher_is_better=False)
        st.plotly_chart(fig_db_comp, use_container_width=True)
        
    st.write("---")
    
    # Section 6 - Visualisasi PCA 2 Dimensi
    st.write("### 🌌 Visualisasi PCA 2 Dimensi per Algoritma")
    tabs_list = st.tabs(["K-Means", "Agglomerative", "Birch", "GMM"])
    
    for t_idx, t_name in enumerate(["K-Means", "Agglomerative", "Birch", "GMM"]):
        with tabs_list[t_idx]:
            labels = labels_dict[t_name]
            fig_pca = plot_pca_2d_plotly(processed_df, X_scaled, labels, t_name)
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Members per cluster count
            unique_labels, counts = np.unique(labels, return_counts=True)
            c_df = pd.DataFrame({
                "Klaster": [f"Cluster {u + 1}" for u in unique_labels],
                "Jumlah Anggota (Daerah)": counts
            })
            st.dataframe(c_df, use_container_width=True)
            
    st.write("---")


# ──────────────────────────────────────────────────────────
# Page: HASIL CLUSTER
# ──────────────────────────────────────────────────────────

elif page == "🗺️ Hasil Cluster":
    st.subheader("🗺️ Hasil Clustering K-Means")
    st.write("Halaman ini melatih model K-Means secara dinamis menggunakan parameter cluster pilihan Anda.")

    # Check that model files exist before proceeding
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error(
            "❌ File model K-Means atau Scaler tidak ditemukan.\n\n"
            f"- Model: `{MODEL_PATH}` → {'✅ Ada' if MODEL_PATH.exists() else '❌ Tidak ada'}\n"
            f"- Scaler: `{SCALER_PATH}` → {'✅ Ada' if SCALER_PATH.exists() else '❌ Tidak ada'}\n\n"
            "Pastikan file `models/kmeans_model.joblib` dan `models/scaler.joblib` "
            "sudah di-commit ke repository."
        )
        st.stop()

    # Check that processed dataset exists
    if not PROCESSED_DATA_PATH.exists():
        st.error(
            f"❌ File dataset processed tidak ditemukan di: `{PROCESSED_DATA_PATH}`.\n\n"
            f"Pastikan file `data/processed/processed_tourism_data.csv` sudah di-commit."
        )
        st.stop()

    try:
        # Load model and scaler offline
        df_clustered, kmeans_model, scaler_model, label_map = load_and_apply_kmeans()
    except Exception as e:
        st.error(f"❌ Gagal menjalankan clustering: {e}")
        st.stop()
    
    # Display Stats
    st.markdown("### 🏆 Profil Ringkasan Klaster")
    
    # Calculate stats per cluster
    cluster_stats = df_clustered.groupby("cluster_label").agg(
        total_daerah=("nama_kabupaten_kota", "count"),
        avg_nusantara=("total_nusantara", "mean"),
        avg_mancanegara=("total_mancanegara", "mean"),
        avg_tahunan=("rata_rata_tahunan", "mean")
    ).reset_index()
    
    # Style the column names for display
    cluster_stats.columns = ["Klaster Potensi", "Jumlah Daerah", "Rerata Wis. Nusantara", "Rerata Wis. Mancanegara", "Rerata Pengunjung Tahunan"]
    st.dataframe(cluster_stats.style.background_gradient(cmap="Greens", subset=["Rerata Pengunjung Tahunan"]), use_container_width=True)
    
    # Layout tabs for visualization
    tab_map, tab_3d, tab_2d, tab_rank = st.tabs([
        "🌍 Peta Spasial Cluster", 
        "🪐 Visualisasi 3D", 
        "📊 Perbandingan Fitur 2D", 
        "🏆 Peringkat Daerah"
    ])
    
    with tab_map:
        st.write("### Peta Distribusi Geografis Jawa Barat")
        fig_map = plot_spasial_map_plotly(df_clustered, color_col="cluster_label")
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Peta menggunakan koordinat daerah untuk mengidentifikasi konsentrasi klaster potensi.")
        
    with tab_3d:
        st.write("### Visualisasi Cluster Interaktif 3D")
        fig_3d = plot_clusters_3d_plotly(df_clustered, "total_nusantara", "total_mancanegara", "rata_rata_tahunan", "cluster_label")
        st.plotly_chart(fig_3d, use_container_width=True)
        st.caption("Gunakan mouse untuk memutar, memperbesar, dan mengarahkan pointer untuk melihat detail daerah di ruang 3 dimensi.")
        
    with tab_2d:
        st.write("### Scatter Plot 2D Hubungan Wisatawan Nusantara vs Mancanegara")
        fig_2d = plot_clusters_2d_plotly(df_clustered, "total_nusantara", "total_mancanegara", "cluster_label")
        st.plotly_chart(fig_2d, use_container_width=True)
        
    with tab_rank:
        st.write("### Peringkat Kunjungan Rata-rata Tahunan per Daerah")
        fig_rank = plot_comparison_bar_plotly(df_clustered, "rata_rata_tahunan", "Peringkat Rerata Pengunjung Tahunan per Daerah Berdasarkan Klaster", "cluster_label")
        st.plotly_chart(fig_rank, use_container_width=True)
        
    # Table showing full data list
    st.write("### 📋 Detail Anggota Klaster Kabupaten/Kota")
    st.dataframe(df_clustered, use_container_width=True)
    
    # Download results button
    csv_results = df_clustered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Hasil Klasterisasi (CSV)",
        data=csv_results,
        file_name=f"hasil_clustering_kmeans_k_{k_selected}.csv",
        mime="text/csv"
    )
