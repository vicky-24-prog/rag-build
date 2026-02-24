"""Data loading and preprocessing - simplified."""

import pandas as pd
import pickle
from pathlib import Path
import yaml
import re


def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_and_clean_data(config_path="config/config.yaml", force_reload=False):
    """Load CSV and clean text in one step."""
    config = load_config(config_path)
    
    csv_path = config['data']['input_path']
    cache_path = "data/clean_products.pkl"
    
    if not force_reload and Path(cache_path).exists():
        return pickle.load(open(cache_path, 'rb'))
    
    # Load CSV with string product_id
    df = pd.read_csv(csv_path, dtype={'product_id': str})
    
    # Clean text
    df['clean_text'] = df.apply(
        lambda row: f"{row['product_name']} {row['description']}".lower().strip(),
        axis=1
    )
    
    # Remove URLs and extra whitespace
    df['clean_text'] = df['clean_text'].apply(
        lambda x: re.sub(r'http\S+|www\S+', '', x)
    ).apply(
        lambda x: re.sub(r'\s+', ' ', x).strip()
    )
    
    # Cache
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(cache_path, 'wb'))
    
    print(f"Loaded and cleaned {len(df)} products")
    return df
