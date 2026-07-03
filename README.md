# Identifikasi Potensi Pariwisata Daerah di Jawa Barat Menggunakan Algoritma K-Means Clustering Berbasis Streamlit

Proyek machine learning tingkat profesional ini bertujuan untuk mengelompokkan 27 Kabupaten/Kota di Jawa Barat berdasarkan potensi pariwisata mereka menggunakan algoritma **K-Means Clustering**. Dashboard interaktif dibangun menggunakan **Streamlit** untuk memvisualisasikan persebaran potensi secara real-time.

---

## 📌 Anggota Kelompok (Kelompok 2)

| NIM | Nama | Role |
|-----|------|------|
| 20124088 | ALFAN MAULANA | Project Manager |
| 20124082 | TIARA AULIA SEPTIANI | Data Scientist |
| 20124089 | DEDE RIFKI | UI/UX Designer |
| 20124094 | RIFA'I AHMAD | Data Analyst |
| 20124098 | ZIDAN ZAIN SHIBGHATULLAH | Software Engineer |
| 20124074 | MUHAMAD HASBI | UI/UX Designer |

---

## 🚀 Alur Kerja Proyek

1. **Ingestion Data**: Mengambil data dinamis dari API Open Data Jawa Barat dengan limit 1000 record (dilakukan secara offline/terpisah sebelum runtime).
2. **Data Cleaning**: Membersihkan data kosong/anomali, normalisasi tipe data, dan standardisasi penulisan nama daerah.
3. **Exploratory Data Analysis (EDA)**: Menganalisis sebaran kunjungan wisatawan Nusantara/Mancanegara dan tren tahunan (2014-2024).
4. **Feature Engineering**: Mengagregasikan data transaksional tahunan menjadi 4 fitur utama tingkat daerah.
5. **Normalisasi**: Transformasi fitur menggunakan `StandardScaler`.
6. **Optimasi Hyperparameter & Benchmarking**: Evaluasi jumlah klaster optimal ($K$) menggunakan **Elbow Method** (Inersia) dan **Silhouette Score** serta komparasi model (K-Means, Agglomerative, Birch, GMM).
7. **Modelling & Labeling**: Implementasi algoritma K-Means, lalu cluster diurutkan berdasarkan volume rata-rata kunjungan tahunan untuk dilabeli menjadi **Potensi Rendah**, **Potensi Sedang**, dan **Potensi Tinggi**.
8. **Persistensi Model**: Menyimpan model terlatih dan scaler ke folder `models/` menggunakan `joblib`.
9. **Dashboard Visualization**: Menyajikan hasil analisis, benchmarking model, dan visualisasi spasial/3D dalam antarmuka Streamlit.

---

## 📁 Struktur Folder 

```
tourism-clustering-jawabarat/
├── data/
│   ├── raw/
│   │   └── raw_tourism_data.json       <- File JSON hasil unduhan dari API Jabarprov
│   └── processed/
│       └── processed_tourism_data.csv <- Dataset teragregasi dengan 4 fitur utama
├── models/
│   ├── kmeans_model.joblib            <- Model K-Means terlatih
│   └── scaler.joblib                  <- Scaler normalisasi Standardizer
├── notebooks/                         <- Jupyter notebooks untuk eksperimen awal
├── references/                        <- Dokumentasi API dan panduan referensi
├── reports/                           <- Hasil analisis statis dan visualisasi
├── tests/                             <- Unit testing untuk dataset dan pipeline model
│   └── test_data.py
├── tourism_clustering_jawabarat/      <- Modul utama Python (Source Code)
│   ├── __init__.py
│   ├── config.py                      <- Parameter konfigurasi, palet warna, dan koordinat peta
│   ├── dataset.py                     <- Script pengambilan data dari API Jabar
│   ├── preprocessing.py               <- Script pembersihan & standardisasi data
│   ├── features.py                    <- Script rekayasa & normalisasi fitur
│   ├── clustering.py                  <- Script evaluasi range K & pelatihan K-Means
│   └── visualizations.py              <- Script visualisasi grafik Plotly & Matplotlib
├── app.py                             <- Script utama dashboard Streamlit
├── requirements.txt                   <- Daftar dependensi library Python
├── pyproject.toml                     <- Konfigurasi packaging proyek
├── Makefile                           <- Otomatisasi perintah command line
├── LICENSE                            <- Lisensi proyek (MIT License)
└── README.md                          <- Dokumentasi proyek (File ini)
```

---

## 🔧 Daftar Library yang Diperlukan

Dependensi utama tercantum dalam `requirements.txt`:
- **Streamlit**: Kerangka kerja utama dashboard web interaktif.
- **Pandas & NumPy**: Pengolahan data tabular dan operasi matriks.
- **Scikit-Learn**: Pelatihan algoritma K-Means dan StandardScaler.
- **Plotly**: Visualisasi interaktif (Scatter 3D, Bar chart horizontal, Mapbox).
- **Matplotlib & Seaborn**: Visualisasi statis untuk korelasi fitur dan EDA.
- **Joblib**: Serialisasi model machine learning.
- **Requests**: Fetching data dari API Open Data Jabar.

---

## 💻 Cara Instalasi dan Menjalankan

### 1. Kloning Proyek & Masuk ke Direktori
```bash
cd tourism-clustering-jawabarat
```

### 2. Buat Virtual Environment (Sangat Direkomendasikan)
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Pipeline Modeling (Opsional - Dataset dan model terlatih sudah tersedia di repositori)
Untuk menjalankan pipeline secara modular dari CLI:
```bash
# Pengambilan data API
python -m tourism_clustering_jawabarat.dataset

# Pembersihan data
python -m tourism_clustering_jawabarat.preprocessing

# Rekayasa fitur
python -m tourism_clustering_jawabarat.features

# Pelatihan & evaluasi model
python -m tourism_clustering_jawabarat.clustering
```

### 5. Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Aplikasi akan secara otomatis terbuka di peramban (browser) default Anda di alamat `http://localhost:8501`.

---

## 📊 Penjelasan Fitur Dashboard Streamlit

Aplikasi telah disederhanakan menjadi **4 Halaman Utama** pada menu navigasi:
1. **Evaluasi & Perbandingan Model (Halaman Pertama)**: Peringkat agregat model terbaik (K-Means, Agglomerative, Birch, GMM), tabel evaluasi metrik komparasi, grafik perbandingan metrik evaluasi (Silhouette, Calinski-Harabasz, Davies-Bouldin), dan tab visualisasi PCA 2D per algoritma.
2. **Dataset**: Metrik ringkasan dataset (total rekaman, rentang tahun, jumlah daerah, total pengunjung), tabel interaktif dengan filter pencarian kabupaten/kota, tahun, jenis wisatawan, serta fitur unduh CSV.
3. **Exploratory Data Analysis (EDA)**: Visualisasi komposisi wisatawan (pie chart), tren pertumbuhan kunjungan tahunan (line chart), dan heatmap matriks korelasi fitur.
4. **Hasil Cluster**: Profil ringkasan klaster potensi pariwisata K-Means (K=3), peta spasial distribusi Jawa Barat, visualisasi cluster 3D dan 2D, peringkat daerah, detail anggota klaster, dan tombol download CSV.
