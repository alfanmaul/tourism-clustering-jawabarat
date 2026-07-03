import unittest
import pandas as pd
from pathlib import Path

from tourism_clustering_jawabarat.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from tourism_clustering_jawabarat.preprocessing import clean_data
from tourism_clustering_jawabarat.features import build_features
from tourism_clustering_jawabarat.clustering import load_and_apply_kmeans


class TestTourismClustering(unittest.TestCase):

    def test_raw_data_exists(self):
        """Memastikan file data mentah (raw data) terunduh dan ada di data/raw/."""
        self.assertTrue(RAW_DATA_PATH.exists(), f"File data mentah tidak ditemukan di {RAW_DATA_PATH}")

    def test_preprocessing(self):
        """Memastikan fungsi clean_data() berjalan dan menghasilkan DataFrame yang valid."""
        df = clean_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty, "DataFrame hasil cleaning kosong")
        self.assertIn("jumlah_pengunjung", df.columns)
        self.assertIn("nama_kabupaten_kota", df.columns)

    def test_feature_engineering(self):
        """Memastikan rekayasa fitur menghasilkan 4 fitur agregasi yang benar."""
        df_features = build_features()
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertFalse(df_features.empty, "DataFrame hasil rekayasa fitur kosong")
        
        required_cols = ["nama_kabupaten_kota", "total_nusantara", "total_mancanegara", "total_pengunjung", "rata_rata_tahunan"]
        for col in required_cols:
            self.assertIn(col, df_features.columns)

    def test_clustering_inference(self):
        """Memastikan proses loading model (inference offline) berjalan lancar dan memberikan label klaster."""
        df_clustered, model, scaler, label_map = load_and_apply_kmeans()
        self.assertIsInstance(df_clustered, pd.DataFrame)
        self.assertFalse(df_clustered.empty, "DataFrame hasil klasterisasi kosong")
        self.assertIn("cluster_label", df_clustered.columns)
        
        # Cek keselarasan jumlah daerah (harus 27 daerah di Jawa Barat)
        self.assertEqual(len(df_clustered), 27, "Jumlah daerah hasil klasterisasi tidak sama dengan 27")


if __name__ == '__main__':
    unittest.main()
