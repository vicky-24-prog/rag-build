"""
VECTOR STORE & INDEXING LAYER (FAISS)

Responsibility: Store embeddings in searchable index and enable fast retrieval

Why FAISS?
- Industry-standard for semantic search (used at Meta, Google, etc.)
- Handles millions of vectors efficiently
- GPU-accelerated available
- Open-source and production-proven
- Supports exact and approximate search

Index Types:
- Flat: Exact KNN search. O(n) complexity. Best for up to ~100k vectors.
  * Every query searches entire index
  * Perfect accuracy
  * Good for this project (30 products)

- IVF (Inverted File): Approximate KNN. O(log n) complexity.
  * Clusters vectors, searches only relevant clusters
  * ~99% accuracy with 100x speedup
  * For millions of products

Normalization & Cosine Similarity:
- Normalized vectors: dot product = cosine similarity
- FAISS optimized for this with IndexFlatIP (Inner Product)
- No need for distance computation, just fast dot product
"""

import logging
import numpy as np
import pickle
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class VectorStoreLayer:
    """
    FAISS-based vector store for efficient similarity search.
    
    Design Principles:
    - Normalized vectors (for cosine similarity via dot product)
    - Persistent storage (reproducible results)
    - Metadata tracking (product_id → embedding index)
    - Production-ready (error handling, logging)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize vector store.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.vector_config = self.config.get("vector_store", {})
        self.index = None
        self.metadata = {}  # product_id → index mapping
        self.dimension = self.vector_config.get("dimension", 384)
        self.index_type = self.vector_config.get("index_type", "Flat")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def build(self, embeddings: np.ndarray, product_ids: np.ndarray) -> None:
        """
        Build FAISS index from embeddings.
        
        Strategy:
        1. Validate embeddings (must be normalized)
        2. Create appropriate FAISS index
        3. Add embeddings to index
        4. Create metadata mapping
        5. Save to disk
        
        Args:
            embeddings: Array of shape (n_products, embedding_dim)
            product_ids: Array of product IDs corresponding to embeddings
        """
        logger.info("\n" + "="*60)
        logger.info("VECTOR STORE BUILDING")
        logger.info("="*60)
        
        # Validate inputs
        if len(embeddings) != len(product_ids):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and product_ids ({len(product_ids)}) "
                "must have same length"
            )
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"configured dimension {self.dimension}"
            )
        
        # Validate normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Embeddings not normalized! Normalizing now...")
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        logger.info(f"\nBuilding {self.index_type} FAISS index")
        logger.info(f"  • Total vectors: {len(embeddings)}")
        logger.info(f"  • Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  • Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")
        
        # Create index
        if self.index_type == "Flat":
            self._build_flat_index(embeddings)
        elif self.index_type == "IVF":
            self._build_ivf_index(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Create metadata mapping
        logger.info("\nCreating metadata mapping...")
        self.metadata = {
            "product_ids": product_ids.tolist(),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "total_vectors": len(embeddings)
        }
        logger.info(f"✓ Metadata: {len(product_ids)} products mapped")
        
        # Persist
        self._save_index()
        
        logger.info("="*60 + "\n")
    
    def _build_flat_index(self, embeddings: np.ndarray) -> None:
        """
        Build exact KNN index using FAISS Flat.
        
        IndexFlatIP: Inner Product (dot product) distance
        - Perfect for normalized vectors (dot product = cosine similarity)
        - O(n) search complexity but very fast in practice
        - ~0.1ms per query for 1M vectors on CPU
        
        Args:
            embeddings: Normalized embedding matrix
        """
        logger.info("Creating IndexFlatIP (exact inner product search)...")
        
        embeddings_float32 = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings_float32)
        
        logger.info(f"✓ Index built successfully")
        logger.info(f"  • Index ntotal: {self.index.ntotal}")
        logger.info(f"  • Search mode: Exact KNN")
        logger.info(f"  • Complexity: O(n) per query")
    
    def _build_ivf_index(self, embeddings: np.ndarray) -> None:
        """
        Build approximate KNN index using FAISS IVF.
        
        IndexIVFFlat: Inverted File with flat quantizer
        - Approximate search: O(log n) complexity
        - Trades small accuracy loss for massive speedup
        - Good for millions of vectors
        
        Args:
            embeddings: Normalized embedding matrix
        """
        logger.info("Creating IndexIVFFlat (approximate search)...")
        
        embeddings_float32 = embeddings.astype(np.float32)
        
        # Parameters
        nlist = self.vector_config.get("nlist", 100)  # Number of clusters
        nprobe = self.vector_config.get("nprobe", 10)  # Cells to search
        
        logger.info(f"  • nlist (clusters): {nlist}")
        logger.info(f"  • nprobe (search cells): {nprobe}")
        
        # Create quantizer and index
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train on embeddings
        logger.info("Training index (creating clusters)...")
        self.index.train(embeddings_float32)
        
        # Add vectors
        self.index.add(embeddings_float32)
        
        # Set search parameters
        self.index.nprobe = nprobe
        
        logger.info(f"✓ Index built successfully")
        logger.info(f"  • Index ntotal: {self.index.ntotal}")
        logger.info(f"  • Search mode: Approximate KNN (IVF)")
    
    def _save_index(self) -> None:
        """Save index and metadata to disk."""
        index_path = self.vector_config.get("index_path", "models/faiss_index.bin")
        metadata_path = self.vector_config.get("metadata_path", "models/metadata.pkl")
        
        # Create directories
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        logger.info(f"Saving FAISS index to {index_path}")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        logger.info(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"✓ Index and metadata saved")
    
    def load_index(self) -> None:
        """Load persisted index and metadata from disk."""
        index_path = self.vector_config.get("index_path", "models/faiss_index.bin")
        metadata_path = self.vector_config.get("metadata_path", "models/metadata.pkl")
        
        logger.info(f"Loading FAISS index from {index_path}")
        
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"✓ Index loaded successfully")
        logger.info(f"  • Total vectors: {self.index.ntotal}")
        logger.info(f"  • Index type: {self.metadata.get('index_type')}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5
    ) -> Tuple[List[str], np.ndarray]:
        """
        Search index for most similar vectors.
        
        Args:
            query_embedding: Query vector (must be normalized)
            top_k: Number of results to return
            
        Returns:
            Tuple of (product_ids, distances)
                - product_ids: List of top-K product IDs
                - distances: Similarity scores (cosine similarity: 0 to 2)
        """
        if self.index is None:
            raise ValueError("Index not built or loaded. Call build() or load_index() first.")
        
        # Ensure query is float32 and has correct shape
        query_vector = query_embedding.astype(np.float32).reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_vector, top_k)
        
        # Extract results
        distances = distances[0]  # Remove batch dimension
        indices = indices[0]
        
        # Map indices to product IDs
        product_ids = self.metadata.get("product_ids", [])
        result_product_ids = [product_ids[idx] for idx in indices if idx >= 0]
        result_distances = distances[:len(result_product_ids)]
        
        return result_product_ids, result_distances
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the index."""
        if self.index is None:
            raise ValueError("No index loaded or built")
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metadata": self.metadata
        }


def main():
    """Test vector store layer independently."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Vector Store Layer Test")
    logger.info("="*60)
    
    # Load data and embeddings
    from ingestion import DataIngestionLayer
    from preprocessing import PreprocessingLayer
    from embeddings import EmbeddingLayer
    
    ingestion = DataIngestionLayer(config_path="config/config.yaml")
    raw_df = ingestion.ingest()
    
    preprocessor = PreprocessingLayer(config_path="config/config.yaml")
    clean_df = preprocessor.preprocess(raw_df)
    
    embedder = EmbeddingLayer(config_path="config/config.yaml")
    embeddings, product_ids = embedder.embed(clean_df)
    
    # Build vector store
    vector_store = VectorStoreLayer(config_path="config/config.yaml")
    vector_store.build(embeddings, product_ids)
    
    # Test search
    logger.info("\nTesting search...")
    query = "Budget running shoes"
    query_embedding = embedder.get_embedding_for_query(query)
    
    result_ids, distances = vector_store.search(query_embedding, top_k=5)
    
    logger.info(f"\nQuery: '{query}'")
    logger.info("Top-5 Results:")
    for i, (pid, dist) in enumerate(zip(result_ids, distances), 1):
        # Convert inner product distance to cosine similarity (0 to 2, or -1 to 1)
        logger.info(f"  {i}. Product ID: {pid}, Similarity Score: {dist:.4f}")


if __name__ == "__main__":
    main()
