"""EMBEDDING GENERATION LAYER

Generate semantic embeddings using Gemini API.
Production-ready, API-based embeddings with no local ML dependencies.
"""

import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Tuple
import yaml
from tqdm import tqdm
import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmbeddingLayer:
    """Generate semantic embeddings for product descriptions using Gemini API."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.embed_config = self.config.get("embeddings", {})
        self.model_name = self.embed_config.get(
            "model_name", "models/gemini-embedding-001"
        )

        self.embeddings = None
        self.product_ids = None

        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def embed(
        self,
        df: pd.DataFrame,
        text_field: str = "description",
        force_regenerate: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for product descriptions with caching.
        
        Args:
            df: Preprocessed dataframe
            text_field: Column to embed
            force_regenerate: Skip cache
            
        Returns:
            Tuple of (embeddings, product_ids)
        """

        cache_path = self.embed_config.get("embedding_cache_path")

        if not force_regenerate and cache_path and Path(cache_path).exists():
            logger.info(f"Loading embeddings from cache: {cache_path}")
            self.embeddings = np.load(cache_path)
            self.product_ids = df["product_id"].values
            logger.info(f"✓ Loaded embeddings: shape {self.embeddings.shape}")
            return self.embeddings, self.product_ids

        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING GENERATION LAYER (GEMINI)")
        logger.info("=" * 60)

        texts = df[text_field].fillna("").tolist()
        self.product_ids = df["product_id"].values

        logger.info(f"Generating embeddings for {len(texts)} products")

        embeddings = []
        for text in tqdm(texts, desc="Embedding products"):
            response = genai.embed_content(
                model=self.model_name,
                content=text
            )
            embeddings.append(response["embedding"])

        embeddings = np.array(embeddings)

        logger.info(f"✓ Generated embeddings: shape {embeddings.shape}")


        if self.embed_config.get("normalize", True):
            embeddings = embeddings / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            )
            logger.info("✓ Embeddings normalized")

        self.embeddings = embeddings

        if cache_path and self.embed_config.get("cache_embeddings", True):
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"✓ Cached embeddings to {cache_path}")

        logger.info("=" * 60 + "\n")
        return embeddings, self.product_ids

    def get_embedding_for_query(self, query: str) -> np.ndarray:
        """Generate embedding for a user query."""
        response = genai.embed_content(
            model=self.model_name,
            content=query
        )
        query_embedding = np.array(response["embedding"])

        if self.embed_config.get("normalize", True):
            query_embedding = query_embedding / (
                np.linalg.norm(query_embedding) + 1e-8
            )

        return query_embedding

    def get_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return embeddings and product IDs."""
        if self.embeddings is None:
            raise ValueError("Embeddings not generated yet")

        return self.embeddings, self.product_ids
