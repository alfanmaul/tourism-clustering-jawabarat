import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Files
RAW_DATA_PATH = RAW_DATA_DIR / "raw_tourism_data.json"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "processed_tourism_data.csv"

# Models Path
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "kmeans_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"

# API endpoint (kept for reference, no longer called at runtime)
API_URL = "https://data.jabarprov.go.id/api-backend/bigdata/disparbud/od_15367_jml_pengunjung_ke_objek_wisata__jenis_wisatawan_ka_v2?limit=1000"

# --- Diagnostic Logging ---
logging.info(f"[Config] BASE_DIR: {BASE_DIR}")
logging.info(f"[Config] RAW_DATA_PATH: {RAW_DATA_PATH} | exists: {RAW_DATA_PATH.exists()}")
logging.info(f"[Config] PROCESSED_DATA_PATH: {PROCESSED_DATA_PATH} | exists: {PROCESSED_DATA_PATH.exists()}")
logging.info(f"[Config] MODEL_PATH: {MODEL_PATH} | exists: {MODEL_PATH.exists()}")
logging.info(f"[Config] SCALER_PATH: {SCALER_PATH} | exists: {SCALER_PATH.exists()}")

# Clustering configurations
RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 3

# UI Styling Palettes (HSL tailored)
COLOR_PALETTE = {
    "Potensi Tinggi": "#10B981",  # Emerald Green
    "Potensi Sedang": "#F59E0B",  # Amber Yellow
    "Potensi Rendah": "#EF4444",  # Coral Red
}

# Coordinate Mapping for West Java Cities/Regencies (Latitude, Longitude)
# Used for geographical visualization (Scatter Mapbox) in the dashboard
COORDINATES = {
    "KABUPATEN BOGOR": [-6.5976, 106.7996],
    "KABUPATEN SUKABUMI": [-6.9181, 106.9266],
    "KABUPATEN CIANJUR": [-6.8222, 107.1394],
    "KABUPATEN BANDUNG": [-7.0253, 107.5198],
    "KABUPATEN GARUT": [-7.2279, 107.9087],
    "KABUPATEN TASIKMALAYA": [-7.3503, 108.1186],
    "KABUPATEN CIAMIS": [-7.3274, 108.3541],
    "KABUPATEN KUNINGAN": [-6.9772, 108.4831],
    "KABUPATEN CIREBON": [-6.7621, 108.5323],
    "KABUPATEN MAJALENGKA": [-6.8373, 108.2259],
    "KABUPATEN SUMEDANG": [-6.8617, 107.9209],
    "KABUPATEN INDRAMAYU": [-6.3276, 108.3249],
    "KABUPATEN SUBANG": [-6.5716, 107.7587],
    "KABUPATEN PURWAKARTA": [-6.5569, 107.4433],
    "KABUPATEN KARAWANG": [-6.3007, 107.2889],
    "KABUPATEN BEKASI": [-6.2349, 107.0718],
    "KABUPATEN BANDUNG BARAT": [-6.8439, 107.4988],
    "KOTA BOGOR": [-6.5971, 106.7986],
    "KOTA SUKABUMI": [-6.9278, 106.9300],
    "KOTA BANDUNG": [-6.9175, 107.6191],
    "KOTA CIREBON": [-6.7320, 108.5555],
    "KOTA BEKASI": [-6.2383, 106.9756],
    "KOTA DEPOK": [-6.4025, 106.7942],
    "KOTA CIMAHI": [-6.8723, 107.5414],
    "KOTA TASIKMALAYA": [-7.3274, 108.2207],
    "KOTA BANJAR": [-7.3694, 108.5348],
    "KABUPATEN PANGANDARAN": [-7.6804, 108.4907],
}
