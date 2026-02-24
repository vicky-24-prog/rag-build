"""
RETRIEVAL & RANKING LAYER

Responsibility: Retrieve and rank most relevant products for user queries

Key Responsibilities:
1. Convert user query to embedding
2. Search vector store for Top-K candidates
3. Return products with similarity scores
4. Optional: Re-rank based on other signals (price, rating, etc.)
5. Filter by similarity threshold

Why Top-K Matters:
- Recall vs Precision tradeoff:
  * K=1: Maximum precision, might miss relevant products
  * K=5: Good balance (standard in e-commerce)
  * K=100: High recall, but too many results (information overload)

Similarity Threshold:
- Cosine similarity range: 0 to 2 (or -1 to 1 after centering)
- For normalized vectors: > 0.3 usually means relevant
- Filters out low-relevance results automatically
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class RetrieverLayer:
    """Retrieve and rank products based on semantic similarity with OOD detection."""
    
    def __init__(
        self,
        vector_store,
        embedder,
        df_products: pd.DataFrame,
        config_path: str = "config/config.yaml"
    ):
        """Initialize retriever with vector store, embedder, and product dataframe."""
        self.config = self._load_config(config_path)
        self.retrieval_config = self.config.get("retrieval", {})
        
        self.vector_store = vector_store
        self.embedder = embedder
        self.df_products = df_products
        
        self.product_id_to_idx = {
            pid: idx for idx, pid in enumerate(df_products['product_id'])
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Retrieve most relevant products for a query with OOD detection.
        
        Args:
            query: User query string
            top_k: Number of results (uses config default if None)
            
        Returns:
            Dict with results, scores, OOD detection, and decision (ACCEPT/REJECT)
        """
        import time
        start_time = time.time()
        
        top_k = top_k or self.retrieval_config.get("top_k", 5)
        similarity_threshold = self.retrieval_config.get("similarity_threshold", 0.3)
        domain_threshold = self.retrieval_config.get("domain_threshold", 0.65)
        
        logger.info(f"\nRetrieving products for query: '{query}'")
        logger.info(f"  • Top-K: {top_k}")
        logger.info(f"  • Similarity threshold: {similarity_threshold}")
        logger.info(f"  • Domain threshold (OOD detection): {domain_threshold}")
        
        try:
            query_embedding = self.embedder.get_embedding_for_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise
        
        try:
            product_ids, similarity_scores = self.vector_store.search(query_embedding, top_k=top_k)
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
        
        logger.info(f"Retrieved {len(product_ids)} candidates from vector store")
        
        max_similarity = float(similarity_scores[0]) if len(similarity_scores) > 0 else 0.0
        avg_top_k_similarity = float(np.mean(similarity_scores)) if len(similarity_scores) > 0 else 0.0
        
        logger.info(f"\n[OOD DETECTION]")
        logger.info(f"  • Max similarity: {max_similarity:.4f}")
        logger.info(f"  • Avg top-K similarity: {avg_top_k_similarity:.4f}")
        logger.info(f"  • Domain threshold: {domain_threshold}")
        
        is_out_of_domain = (max_similarity < domain_threshold) or (avg_top_k_similarity < domain_threshold)
        
        if is_out_of_domain:
            logger.warning(f"⚠️ OUT-OF-DOMAIN QUERY DETECTED!")
            logger.warning(f"  Max similarity ({max_similarity:.4f}) and/or avg similarity ({avg_top_k_similarity:.4f})")
            logger.warning(f"  are below domain threshold ({domain_threshold})")
        else:
            logger.info(f"✓ Query appears to be within domain")
        
        results = []
        for rank, (product_id, score) in enumerate(zip(product_ids, similarity_scores), 1):
            if score < similarity_threshold:
                logger.info(f"  Filtering product {product_id} (score {score:.4f} < threshold {similarity_threshold})")
                continue
            
            try:
                product_idx = self.product_id_to_idx[product_id]
                product_row = self.df_products.iloc[product_idx]
            except (KeyError, IndexError) as e:
                logger.warning(f"Product {product_id} not found in dataframe: {e}")
                continue
            
            result_entry = {
                'rank': rank,
                'product_id': product_id,
                'similarity_score': float(score),
                'product_name': product_row.get('product_name', 'N/A'),
                'category': product_row.get('category', 'N/A'),
                'description': product_row.get('description', 'N/A'),
                'price': product_row.get('price', 'N/A'),
                'rating': product_row.get('rating', 'N/A')
            }
            results.append(result_entry)
        
        domain_confidence = self._calculate_domain_confidence(max_similarity, domain_threshold)
        decision = self._make_retrieval_decision(
            is_out_of_domain,
            len(results),
            max_similarity,
            domain_threshold,
            domain_confidence
        )
        
        retrieval_time = time.time() - start_time
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"RETRIEVAL RESULTS WITH EXPLAINABILITY")
        logger.info(f"{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info(f"Decision: {decision} | Domain Confidence: {domain_confidence}")
        logger.info(f"Results returned: {len(results)} (retrieved {len(product_ids)}, after filtering: {len(results)})")
        logger.info(f"Retrieval time: {retrieval_time:.3f} seconds\n")
        
        if decision == "REJECT":
            if is_out_of_domain:
                logger.warning("⚠️ REJECTION REASON: Out-of-Domain Query")
                logger.warning(f"  No confident recommendations found.")
                logger.warning(f"  This query appears to be outside the current product domain.")
                logger.warning(f"  Retrieved results show low semantic similarity and mixed categories.")
            else:
                logger.warning("⚠️ REJECTION REASON: Low Confidence")
                logger.warning(f"  Similarity scores are below acceptable threshold.")
        else:
            for result in results:
                logger.info(f"#{result['rank']}. {result['product_name']} (ID: {result['product_id']})")
                logger.info(f"    Similarity: {result['similarity_score']:.4f} | Price: ₹{result['price']} | Rating: {result['rating']}⭐")
                logger.info(f"    Category: {result['category']}")
                logger.info(f"    Description: {result['description'][:80]}...")
                logger.info("")
        logger.info("="*60)
        
        return {
            'query': query,
            'query_embedding': query_embedding,
            'results': results,
            'total_retrieved': len(results),
            'threshold_used': similarity_threshold,
            'retrieval_time': retrieval_time,
            'max_similarity': max_similarity,
            'avg_top_k_similarity': avg_top_k_similarity,
            'domain_threshold': domain_threshold,
            'is_out_of_domain': is_out_of_domain,
            'domain_confidence': domain_confidence,
            'decision': decision,
            'explainability': {
                'max_similarity_score': f"{max_similarity:.4f}",
                'avg_top_k_similarity': f"{avg_top_k_similarity:.4f}",
                'domain_confidence_label': domain_confidence,
                'final_decision': decision,
                'ood_detection_enabled': True
            }
        }
    
    def _calculate_domain_confidence(self, max_similarity: float, domain_threshold: float) -> str:
        """Calculate domain confidence: HIGH, MEDIUM, or LOW based on similarity scores."""
        high_threshold = domain_threshold
        medium_threshold = domain_threshold * 0.8
        
        if max_similarity >= high_threshold:
            return "HIGH"
        elif max_similarity >= medium_threshold:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _make_retrieval_decision(
        self,
        is_out_of_domain: bool,
        num_results: int,
        max_similarity: float,
        domain_threshold: float,
        domain_confidence: str
    ) -> str:
        """Make final decision: ACCEPT or REJECT based on OOD detection and confidence."""
        if is_out_of_domain:
            logger.warning("Decision: REJECT (out-of-domain query)")
            return "REJECT"
        
        if num_results == 0:
            logger.warning("Decision: REJECT (no results after filtering)")
            return "REJECT"
        
        if domain_confidence == "LOW":
            logger.warning("Decision: REJECT (low domain confidence)")
            return "REJECT"
        
        logger.info(f"Decision: ACCEPT ({domain_confidence} confidence, {num_results} results)")
        return "ACCEPT"
    
    def explain_retrieval(self) -> Dict:
        """
        Provide educational explanation of retrieval process.
        
        Returns:
            Dict with explanations
        """
        return {
            "why_top_k_matters": {
                "K=1": "Only best match - high precision, low recall. Miss alternatives.",
                "K=5": "Industry standard for e-commerce - good balance.",
                "K=100": "High recall but overwhelming for user - information overload."
            },
            "similarity_threshold": {
                "concept": "Minimum cosine similarity to consider a result relevant",
                "range": "0.3 to 0.5 typical for e-commerce",
                "too_low": "Returns irrelevant results (hallucination-like)",
                "too_high": "Filters out legitimate results"
            },
            "retrieval_pipeline": [
                "1. Embed user query using same model as products",
                "2. Search FAISS index for Top-K most similar vectors",
                "3. Convert similarity scores (0-2) back to product records",
                "4. Filter by similarity threshold",
                "5. Return ranked results with metadata"
            ],
            "why_this_prevents_hallucination": [
                "Retrieval ONLY returns existing products",
                "Similarity scores show relevance confidence",
                "Products not in index can NEVER be returned",
                "Threshold ensures minimum quality"
            ],
            "precision_recall_tradeoff": {
                "high_precision": "Only return best matches - fewer false positives",
                "high_recall": "Return all possible matches - catch edge cases",
                "balanced_approach": "K=5 with threshold=0.3 catches most relevant while staying efficient"
            }
        }
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about retrieval layer."""
        return {
            "total_products_indexed": len(self.df_products),
            "config_top_k": self.retrieval_config.get("top_k", 5),
            "config_threshold": self.retrieval_config.get("similarity_threshold", 0.3),
            "use_mmr": self.retrieval_config.get("ranking", {}).get("use_mmr", False),
            "use_reranker": self.retrieval_config.get("ranking", {}).get("use_reranker", False)
        }


def main():
    """Test retrieval layer independently."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Retrieval Layer Test")
    logger.info("="*60)
    
    # Build full pipeline
    from ingestion import DataIngestionLayer
    from preprocessing import PreprocessingLayer
    from embeddings import EmbeddingLayer
    from vector_store import VectorStoreLayer
    
    ingestion = DataIngestionLayer(config_path="config/config.yaml")
    raw_df = ingestion.ingest()
    
    preprocessor = PreprocessingLayer(config_path="config/config.yaml")
    clean_df = preprocessor.preprocess(raw_df)
    
    embedder = EmbeddingLayer(config_path="config/config.yaml")
    embeddings, product_ids = embedder.embed(clean_df)
    
    vector_store = VectorStoreLayer(config_path="config/config.yaml")
    vector_store.build(embeddings, product_ids)
    
    # Initialize retriever
    retriever = RetrieverLayer(
        vector_store=vector_store,
        embedder=embedder,
        df_products=raw_df,
        config_path="config/config.yaml"
    )
    
    # Test retrievals
    test_queries = [
        "Budget running shoes for beginners",
        "Wireless headphones with strong bass",
        "Laptop for video editing"
    ]
    
    logger.info("\nTesting retrieval with sample queries...")
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        logger.info(f"✓ Retrieved {results['total_retrieved']} products in {results['retrieval_time']:.3f}s\n")
    
    # Show explanations
    logger.info("\n" + "="*60)
    logger.info("RETRIEVAL EXPLAINED")
    logger.info("="*60)
    explanations = retriever.explain_retrieval()
    import json
    logger.info(json.dumps(explanations, indent=2))


if __name__ == "__main__":
    main()
