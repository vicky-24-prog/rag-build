"""
DATA INGESTION LAYER

Load and validate raw product data with caching for reproducibility.
"""

import logging
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class DataIngestionLayer:
    """Load and validate raw e-commerce product data with caching."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize ingestion layer with configuration."""
        self.config = self._load_config(config_path)
        self.data_config = self.config.get("data", {})
        self.raw_df = None
        self.quality_report = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def ingest(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load product data from CSV with caching.
        
        Args:
            force_reload: If True, ignore cache and reload from CSV
            
        Returns:
            Raw product dataframe
        """
        cache_path = self.data_config.get("raw_data_cache")
        
        if not force_reload and cache_path and Path(cache_path).exists():
            logger.info(f"Loading raw data from cache: {cache_path}")
            self.raw_df = pickle.load(open(cache_path, 'rb'))
            logger.info(f"✓ Loaded {len(self.raw_df)} products from cache")
            return self.raw_df
        
        # Load from CSV
        input_path = self.data_config.get("input_path")
        encoding = self.data_config.get("encoding", "utf-8")
        
        logger.info(f"Loading raw data from CSV: {input_path}")
        try:
            self.raw_df = pd.read_csv(input_path, encoding=encoding, dtype={'product_id': str})
            logger.info(f"✓ Loaded {len(self.raw_df)} products from CSV")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {input_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
        
        # Validate and report
        self._validate_data()
        
        # Save cache
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.raw_df, open(cache_path, 'wb'))
            logger.info(f"✓ Cached raw data to {cache_path}")
        
        return self.raw_df
    
    def _validate_data(self) -> None:
        """Validate data integrity and generate quality report."""
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY REPORT")
        logger.info("="*60)
        
        required_columns = self.data_config.get("validation", {}).get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in self.raw_df.columns]
        
        if missing_columns:
            logger.error(f"✗ Missing required columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")
        else:
            logger.info(f"✓ All required columns present: {required_columns}")
        
        self.quality_report["columns_validated"] = True
        
        logger.info("\n--- Missing Values Analysis ---")
        missing_stats = self.raw_df.isnull().sum()
        
        if missing_stats.sum() > 0:
            logger.warning("Missing values detected:")
            for col, count in missing_stats[missing_stats > 0].items():
                pct = (count / len(self.raw_df)) * 100
                logger.warning(f"  {col}: {count} ({pct:.1f}%)")
        else:
            logger.info("✓ No missing values detected")
        
        self.quality_report["missing_values"] = missing_stats.to_dict()
        
        strategy = self.data_config.get("validation", {}).get("missing_value_strategy", "drop")
        
        if strategy == "drop":
            before_count = len(self.raw_df)
            self.raw_df = self.raw_df.dropna()
            after_count = len(self.raw_df)
            dropped = before_count - after_count
            if dropped > 0:
                logger.info(f"✓ Dropped {dropped} rows with missing values")
            self.quality_report["rows_dropped_missing"] = dropped
        
        logger.info("\n--- Duplicate Detection ---")
        duplicates = self.raw_df.duplicated(subset=['product_id']).sum()
        
        if duplicates > 0:
            logger.warning(f"✗ Found {duplicates} duplicate product_ids")
            dup_strategy = self.data_config.get("validation", {}).get("duplicate_strategy", "first")
            self.raw_df = self.raw_df.drop_duplicates(subset=['product_id'], keep=dup_strategy)
            logger.info(f"✓ Kept '{dup_strategy}' occurrence of duplicates")
        else:
            logger.info("✓ No duplicate product_ids found")
        
        self.quality_report["duplicates_found"] = duplicates
        
        logger.info("\n--- Data Type Summary ---")
        for col in required_columns:
            if col in self.raw_df.columns:
                logger.info(f"  {col}: {self.raw_df[col].dtype}")
        
        logger.info("\n--- Final Data Statistics ---")
        logger.info(f"Total products: {len(self.raw_df)}")
        logger.info(f"Total features: {len(self.raw_df.columns)}")
        logger.info(f"Features: {list(self.raw_df.columns)}")
        
        self.quality_report["final_product_count"] = len(self.raw_df)
        self.quality_report["final_feature_count"] = len(self.raw_df.columns)
        
        logger.info("="*60 + "\n")
    
    def get_quality_report(self) -> Dict:
        """Get data quality metrics.
        
        Returns:
            Dict with quality metrics (missing values, duplicates, etc.)
        """
        return self.quality_report
    
    def get_raw_data(self) -> pd.DataFrame:
        """Get raw data (no cache, returns current state)."""
        return self.raw_df
    
    def get_product_count(self) -> int:
        """Get number of products after ingestion."""
        return len(self.raw_df) if self.raw_df is not None else 0
    
    def get_columns(self) -> List[str]:
        """Get list of columns in dataset."""
        return list(self.raw_df.columns) if self.raw_df is not None else []


def main():
    """Test the ingestion layer independently."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Data Ingestion Layer Test")
    logger.info("="*60)
    
    # Initialize and ingest
    ingestion = DataIngestionLayer(config_path="config/config.yaml")
    df = ingestion.ingest()
    
    # Display sample
    logger.info("\nSample Products (First 3):")
    logger.info(df.head(3).to_string())
    
    # Display quality report
    logger.info("\nQuality Report:")
    import json
    logger.info(json.dumps(ingestion.get_quality_report(), indent=2))


if __name__ == "__main__":
    main()
