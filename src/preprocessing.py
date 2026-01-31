"""
TEXT PREPROCESSING LAYER

Responsibility: Clean and normalize product descriptions while preserving semantic meaning

Key Design Decision: MINIMAL preprocessing because:
- Modern embedding models (SentenceTransformers) work better with natural text
- Aggressive stemming/lemmatization HURTS semantic embeddings
- Stopwords carry semantic value for transformers
- We preserve what the model needs to understand intent

Preprocessing Steps (in order):
1. Whitespace normalization - Handle extra spaces, newlines
2. Lowercase conversion - Normalize casing
3. Remove URLs and emails - Remove noise
4. Remove special characters - But preserve structure
5. Length validation - Filter very short texts
6. Duplicate handling - Remove duplicate descriptions

What we DO NOT do:
- Stemming (porter_stem, snowball) - Breaks embeddings
- Lemmatization - Unnecessary, models handle variations
- Aggressive stopword removal - Stopwords carry meaning
"""

import logging
import re
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from collections import Counter

logger = logging.getLogger(__name__)


class PreprocessingLayer:
    """
    Clean and normalize text data while preserving semantic meaning.
    
    Design Principles:
    - Light touch: Don't over-clean (embeddings handle it better)
    - Explainability: Log each preprocessing decision
    - Semantic preservation: Keep content meaningful
    - Reproducibility: Cache results
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize preprocessing layer with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.preproc_config = self.config.get("preprocessing", {})
        self.preprocessed_df = None
        self.preprocessing_stats = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def preprocess(self, df: pd.DataFrame, force_reprocess: bool = False) -> pd.DataFrame:
        """
        Preprocess product descriptions.
        
        Strategy:
        1. Check cache first (if enabled)
        2. Apply preprocessing pipeline to descriptions
        3. Apply to other text fields
        4. Save cache
        
        Args:
            df: Raw dataframe from ingestion layer
            force_reprocess: If True, skip cache and reprocess
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        cache_path = self.preproc_config.get("caching", {}).get("path")
        
        # Try cache first
        if not force_reprocess and cache_path and Path(cache_path).exists():
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            self.preprocessed_df = pickle.load(open(cache_path, 'rb'))
            logger.info(f"✓ Loaded {len(self.preprocessed_df)} preprocessed products from cache")
            return self.preprocessed_df
        
        logger.info("\n" + "="*60)
        logger.info("TEXT PREPROCESSING PIPELINE")
        logger.info("="*60)
        
        # Create working copy
        self.preprocessed_df = df.copy()
        
        # Apply preprocessing to text fields
        text_fields = ['product_name', 'description']
        
        for field in text_fields:
            if field in self.preprocessed_df.columns:
                logger.info(f"\n--- Preprocessing field: {field} ---")
                
                before_state = self.preprocessed_df[field].copy()
                
                # Apply preprocessing pipeline
                self.preprocessed_df[field] = self.preprocessed_df[field].apply(
                    self._preprocess_text
                )
                
                # Log changes
                self._log_field_changes(field, before_state, self.preprocessed_df[field])
        
        # Handle duplicates at the description level
        self._handle_duplicates()
        
        # Summary statistics
        self._log_preprocessing_summary()
        
        # Save cache
        if cache_path and self.preproc_config.get("caching", {}).get("enabled", True):
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(self.preprocessed_df, open(cache_path, 'wb'))
            logger.info(f"✓ Cached preprocessed data to {cache_path}")
        
        logger.info("="*60 + "\n")
        
        return self.preprocessed_df
    
    def _preprocess_text(self, text: str) -> str:
        """
        Apply preprocessing pipeline to a single text string.
        
        Pipeline:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove emails
        4. Normalize whitespace
        5. Remove special characters (but preserve structure)
        6. Truncate if too long
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return ""
        
        # 1. Lowercase
        if self.preproc_config.get("lowercase", True):
            text = text.lower()
        
        # 2. Remove URLs
        if self.preproc_config.get("remove_urls", True):
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Remove email addresses
        if self.preproc_config.get("remove_emails", True):
            text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Normalize whitespace (remove extra spaces, tabs, newlines)
        if self.preproc_config.get("remove_extra_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # 5. Remove special characters but keep some structure
        # Keep: alphanumeric, spaces, hyphens, dots, commas (for readability)
        if self.preproc_config.get("remove_special_chars", True):
            text = re.sub(r'[^a-z0-9\s\-.,()]', '', text)
        
        # 6. Truncate very long descriptions (for efficiency)
        max_tokens_estimate = self.preproc_config.get("description_max_tokens", 512)
        # Rough estimate: ~4 characters per token
        max_chars = max_tokens_estimate * 4
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(' ', 1)[0]  # Truncate at word boundary
        
        return text
    
    def _log_field_changes(self, field: str, before: pd.Series, after: pd.Series) -> None:
        """
        Log what changed during preprocessing of a field.
        
        Args:
            field: Field name
            before: Series before preprocessing
            after: Series after preprocessing
        """
        # Count changes
        changed_count = (before != after).sum()
        avg_before = before.str.len().mean()
        avg_after = after.str.len().mean()
        
        logger.info(f"  • {changed_count} texts modified")
        logger.info(f"  • Average length before: {avg_before:.0f} chars")
        logger.info(f"  • Average length after: {avg_after:.0f} chars")
        logger.info(f"  • Compression ratio: {(avg_after/avg_before)*100:.1f}%")
        
        # Show example
        for i, (b, a) in enumerate(zip(before[:2], after[:2])):
            if b != a:
                logger.debug(f"\n    Example {i+1}:")
                logger.debug(f"    Before: {b[:100]}")
                logger.debug(f"    After: {a[:100]}")
    
    def _handle_duplicates(self) -> None:
        """
        Detect and handle duplicate descriptions.
        
        Duplicates could indicate:
        - Multiple SKUs of same product
        - Data entry errors
        
        For RAG, duplicates might slightly inflate retrieval relevance,
        but we keep them as they represent real products.
        """
        logger.info("\n--- Duplicate Description Detection ---")
        
        before_count = len(self.preprocessed_df)
        
        # Find exact duplicate descriptions
        duplicate_descriptions = (
            self.preprocessed_df['description'].duplicated().sum()
        )
        
        if duplicate_descriptions > 0:
            logger.warning(f"Found {duplicate_descriptions} duplicate descriptions")
            logger.info("  → Keeping duplicates (different products may have similar descriptions)")
        else:
            logger.info("✓ No duplicate descriptions found")
        
        self.preprocessing_stats['duplicate_descriptions_found'] = duplicate_descriptions
        self.preprocessing_stats['final_row_count'] = len(self.preprocessed_df)
    
    def _log_preprocessing_summary(self) -> None:
        """Log summary statistics of preprocessing."""
        logger.info("\n--- Preprocessing Summary ---")
        logger.info(f"Total products processed: {len(self.preprocessed_df)}")
        logger.info(f"Total features: {len(self.preprocessed_df.columns)}")
        
        if 'description' in self.preprocessed_df.columns:
            desc_lengths = self.preprocessed_df['description'].str.len()
            logger.info(f"\nDescription Statistics:")
            logger.info(f"  • Min length: {desc_lengths.min()} chars")
            logger.info(f"  • Max length: {desc_lengths.max()} chars")
            logger.info(f"  • Mean length: {desc_lengths.mean():.0f} chars")
            logger.info(f"  • Median length: {desc_lengths.median():.0f} chars")
        
        self.preprocessing_stats['products_processed'] = len(self.preprocessed_df)
    
    def get_preprocessing_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.preprocessing_stats
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """Get preprocessed dataframe."""
        return self.preprocessed_df


def main():
    """Test the preprocessing layer independently."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Text Preprocessing Layer Test")
    logger.info("="*60)
    
    # Load raw data first (using ingestion layer)
    from ingestion import DataIngestionLayer
    
    ingestion = DataIngestionLayer(config_path="config/config.yaml")
    raw_df = ingestion.ingest()
    
    # Apply preprocessing
    preprocessor = PreprocessingLayer(config_path="config/config.yaml")
    cleaned_df = preprocessor.preprocess(raw_df)
    
    # Show comparison
    logger.info("\n" + "="*60)
    logger.info("BEFORE vs AFTER PREPROCESSING")
    logger.info("="*60)
    
    comparison = pd.DataFrame({
        'Product': cleaned_df['product_name'],
        'Description_Before': raw_df['description'],
        'Description_After': cleaned_df['description']
    })
    
    logger.info(comparison.head(3).to_string())
    
    logger.info("\nPreprocessing Stats:")
    import json
    logger.info(json.dumps(preprocessor.get_preprocessing_stats(), indent=2))


if __name__ == "__main__":
    main()
