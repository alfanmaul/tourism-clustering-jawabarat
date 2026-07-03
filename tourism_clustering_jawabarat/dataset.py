import json
import logging

import requests

from tourism_clustering_jawabarat.config import API_URL, RAW_DATA_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_raw_data() -> dict:
    """
    Fetch the raw tourism dataset from the Open Data Jawa Barat API
    and save it locally to RAW_DATA_PATH.
    """
    logging.info(f"Fetching data from API: {API_URL}")
    try:
        response = requests.get(API_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # Validation
        if "data" not in data or not data["data"]:
            raise ValueError("API returned response but 'data' key is empty or missing.")
            
        logging.info(f"Successfully fetched {len(data['data'])} records.")
        
        # Save to local file
        with open(RAW_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            
        logging.info(f"Saved raw data to {RAW_DATA_PATH}")
        return data
        
    except requests.exceptions.Timeout as e:
        logging.error(f"API request timed out (limit 15s): {e}")
        raise RuntimeError("Koneksi ke API Open Data Jabar mengalami timeout. Silakan coba beberapa saat lagi.") from e
    except requests.exceptions.ConnectionError as e:
        logging.error(f"API connection failed: {e}")
        raise RuntimeError("Gagal terhubung ke API Open Data Jabar. Periksa koneksi internet Anda.") from e
    except requests.exceptions.HTTPError as e:
        logging.error(f"API returned HTTP error status: {e}")
        raise RuntimeError(f"API Open Data Jabar mengembalikan error HTTP: {response.status_code}.") from e
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise RuntimeError("Terjadi kesalahan jaringan saat mengambil data dari API disparbud.") from e
    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching data: {e}")
        raise


if __name__ == "__main__":
    fetch_raw_data()
