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
        response = requests.get(API_URL)
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
        
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        raise

if __name__ == "__main__":
    fetch_raw_data()
