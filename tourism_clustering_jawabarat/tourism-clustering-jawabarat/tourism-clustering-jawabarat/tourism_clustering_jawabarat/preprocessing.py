import json
import logging

import pandas as pd

from tourism_clustering_jawabarat.config import RAW_DATA_PATH
from tourism_clustering_jawabarat.dataset import fetch_raw_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def clean_data() -> pd.DataFrame:
    """
    Load raw JSON data, clean and normalize it, and return a clean DataFrame.
    """
    if not RAW_DATA_PATH.exists():
        logging.info("Raw data file not found, fetching from API first...")
        fetch_raw_data()

    logging.info(f"Loading raw data from {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
        
    df = pd.DataFrame(raw_json["data"])
    
    # 1. Column normalization
    logging.info("Normalizing columns and types...")
    df["jumlah_pengunjung"] = pd.to_numeric(df["jumlah_pengunjung"], errors="coerce").fillna(0).astype(int)
    df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce").astype(int)
    df["nama_kabupaten_kota"] = df["nama_kabupaten_kota"].str.upper().str.strip()
    df["jenis_wisatawan"] = df["jenis_wisatawan"].str.upper().str.strip()
    
    # 2. Standarization of names (if any variations exist)
    # The API names are clean, but let's ensure consistency.
    name_map = {
        "KAB. BOGOR": "KABUPATEN BOGOR",
        "KAB. SUKABUMI": "KABUPATEN SUKABUMI",
        "KAB. CIANJUR": "KABUPATEN CIANJUR",
        "KAB. BANDUNG": "KABUPATEN BANDUNG",
        "KAB. GARUT": "KABUPATEN GARUT",
        "KAB. TASIKMALAYA": "KABUPATEN TASIKMALAYA",
        "KAB. CIAMIS": "KABUPATEN CIAMIS",
        "KAB. KUNINGAN": "KABUPATEN KUNINGAN",
        "KAB. CIREBON": "KABUPATEN CIREBON",
        "KAB. MAJALENGKA": "KABUPATEN MAJALENGKA",
        "KAB. SUMEDANG": "KABUPATEN SUMEDANG",
        "KAB. INDRAMAYU": "KABUPATEN INDRAMAYU",
        "KAB. SUBANG": "KABUPATEN SUBANG",
        "KAB. PURWAKARTA": "KABUPATEN PURWAKARTA",
        "KAB. KARAWANG": "KABUPATEN KARAWANG",
        "KAB. BEKASI": "KABUPATEN BEKASI",
        "KAB. BANDUNG BARAT": "KABUPATEN BANDUNG BARAT",
        "KAB. PANGANDARAN": "KABUPATEN PANGANDARAN",
    }
    df["nama_kabupaten_kota"] = df["nama_kabupaten_kota"].replace(name_map)
    
    # 3. Clean anomalous values (negative values are replaced with 0)
    negative_mask = df["jumlah_pengunjung"] < 0
    if negative_mask.any():
        logging.warning(f"Found {negative_mask.sum()} negative values in 'jumlah_pengunjung'. Setting them to 0.")
        df.loc[negative_mask, "jumlah_pengunjung"] = 0
        
    logging.info(f"Pembersihan data selesai. Dataframe bersih: {df.shape[0]} baris, {df.shape[1]} kolom.")
    return df

if __name__ == "__main__":
    clean_df = clean_data()
    print(clean_df.head())
