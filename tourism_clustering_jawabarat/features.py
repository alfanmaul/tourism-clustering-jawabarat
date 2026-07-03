import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler

from tourism_clustering_jawabarat.config import PROCESSED_DATA_PATH
from tourism_clustering_jawabarat.preprocessing import clean_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def build_features() -> pd.DataFrame:
    """
    Perform feature engineering: aggregate cleaned data by kabupaten/kota 
    to create the four required features for clustering:
      1. Total Wisatawan Nusantara
      2. Total Wisatawan Mancanegara
      3. Total Pengunjung
      4. Rata-rata Pengunjung Tahunan
    Saves the result to PROCESSED_DATA_PATH.
    """
    df = clean_data()
    
    logging.info("Calculating Total Wisatawan Nusantara per kabupaten/kota...")
    nusantara_df = df[df["jenis_wisatawan"] == "NUSANTARA"]
    tot_nusantara = nusantara_df.groupby("nama_kabupaten_kota")["jumlah_pengunjung"].sum()
    
    logging.info("Calculating Total Wisatawan Mancanegara per kabupaten/kota...")
    mancanegara_df = df[df["jenis_wisatawan"] == "MANCANEGARA"]
    tot_mancanegara = mancanegara_df.groupby("nama_kabupaten_kota")["jumlah_pengunjung"].sum()
    
    logging.info("Calculating Total Pengunjung per kabupaten/kota...")
    tot_pengunjung = df.groupby("nama_kabupaten_kota")["jumlah_pengunjung"].sum()
    
    logging.info("Calculating Rata-rata Pengunjung Tahunan per kabupaten/kota...")
    # First get the total visitors per year for each county
    yearly_totals = df.groupby(["nama_kabupaten_kota", "tahun"])["jumlah_pengunjung"].sum().reset_index()
    # Then take the mean of these yearly totals
    avg_tahunan = yearly_totals.groupby("nama_kabupaten_kota")["jumlah_pengunjung"].mean()
    
    # Combine all features into a single DataFrame
    features_df = pd.DataFrame({
        "total_nusantara": tot_nusantara,
        "total_mancanegara": tot_mancanegara,
        "total_pengunjung": tot_pengunjung,
        "rata_rata_tahunan": avg_tahunan
    }).fillna(0)  # Handle any missing aggregates (e.g. if a county has 0 foreign visitors)
    
    # Clean index to keep county names as column and reset index
    features_df = features_df.reset_index()
    
    # Save features to CSV
    features_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logging.info(f"Engineered features saved to {PROCESSED_DATA_PATH}")
    
    return features_df

def normalize_features(features_df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Normalize the chosen feature columns using StandardScaler.
    
    Raises ValueError if the DataFrame is empty to prevent
    StandardScaler from receiving a 0-sample array.
    
    Returns:
        scaled_df (pd.DataFrame): DataFrame with scaled features.
        scaler (StandardScaler): Fitted scaler object.
    """
    logging.info(f"[Features] Rows before scaling: {len(features_df)}")

    if features_df.empty:
        raise ValueError(
            "Cannot normalize features: the input DataFrame is empty (0 rows). "
            "Check dataset loading or preprocessing steps."
        )

    # Verify that required columns exist
    missing_cols = [col for col in feature_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            f"Cannot normalize features: missing columns {missing_cols} "
            f"in DataFrame with columns {list(features_df.columns)}."
        )

    logging.info("Normalizing features using StandardScaler...")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(features_df[feature_cols])
    
    scaled_df = pd.DataFrame(
        scaled_array, 
        columns=[f"{col}_scaled" for col in feature_cols],
        index=features_df.index
    )
    
    return scaled_df, scaler

if __name__ == "__main__":
    features = build_features()
    print("Engineered features preview:")
    print(features.head())
    
    cols = ["total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]
    scaled_df, scaler = normalize_features(features, cols)
    print("\nScaled features preview:")
    print(scaled_df.head())
