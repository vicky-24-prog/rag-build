"""
MAIN APPLICATION

Orchestrates all layers to demonstrate the complete system.

Usage:
    python app.py                    # Run interactive mode
    python app.py --eval            # Run evaluation
    python app.py --rebuild-index   # Rebuild vector index from scratch
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class SemanticSearchRAG:
    """Complete semantic search RAG system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize all layers."""
        self.config_path = config_path
        self.ingestion = None
        self.preprocessor = None
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.rag_pipeline = None
        self.evaluator = None
        self.raw_df = None
        self.clean_df = None
        self.api_key = os.getenv('GOOGLE_API_KEY')  # or GEMINI_API_KEY depending on your choice
    
    def initialize(self, rebuild_index: bool = False):
        """Initialize all layers in order."""
        logger.info("\n" + "="*70)
        logger.info("SEMANTIC SEARCH RAG SYSTEM - INITIALIZATION")
        logger.info("="*70)
        
        try:
            # Layer 1: Ingestion
            logger.info("\n[1/6] Loading Data Ingestion Layer...")
            from ingestion import DataIngestionLayer
            self.ingestion = DataIngestionLayer(config_path=self.config_path)
            self.raw_df = self.ingestion.ingest(force_reload=rebuild_index)
            logger.info("✓ Ingestion complete")
            
            # Layer 2: Preprocessing
            logger.info("\n[2/6] Loading Text Preprocessing Layer...")
            from preprocessing import PreprocessingLayer
            self.preprocessor = PreprocessingLayer(config_path=self.config_path)
            self.clean_df = self.preprocessor.preprocess(self.raw_df, force_reprocess=rebuild_index)
            logger.info("✓ Preprocessing complete")
            
            # Layer 3: Embeddings
            logger.info("\n[3/6] Loading Embedding Generation Layer...")
            from embeddings import EmbeddingLayer
            self.embedder = EmbeddingLayer(config_path=self.config_path)
            embeddings, product_ids = self.embedder.embed(self.clean_df, force_regenerate=rebuild_index)
            logger.info("✓ Embedding generation complete")
            
            # Layer 4: Vector Store
            logger.info("\n[4/6] Loading Vector Store (FAISS)...")
            from vector_store import VectorStoreLayer
            self.vector_store = VectorStoreLayer(config_path=self.config_path)
            
            if rebuild_index or not Path(self.vector_store.vector_config.get("index_path", "models/faiss_index.bin")).exists():
                logger.info("Building new FAISS index...")
                self.vector_store.build(embeddings, product_ids)
            else:
                logger.info("Loading existing FAISS index...")
                self.vector_store.load_index()
            
            logger.info("✓ Vector store ready")
            
            # Layer 5: Retriever
            logger.info("\n[5/6] Loading Retrieval Layer...")
            from retriever import RetrieverLayer
            self.retriever = RetrieverLayer(
                vector_store=self.vector_store,
                embedder=self.embedder,
                df_products=self.raw_df,
                config_path=self.config_path
            )
            logger.info("✓ Retriever ready")
            
            # Layer 6: RAG Pipeline
            logger.info("\n[6/6] Loading RAG Generation Layer...")
            from rag_pipeline import RAGPipeline
            self.rag_pipeline = RAGPipeline(config_path=self.config_path)
            logger.info("✓ RAG pipeline ready")
            
            # Evaluation
            logger.info("\nLoading Evaluation Layer...")
            from evaluation import EvaluationLayer
            self.evaluator = EvaluationLayer(config_path=self.config_path)
            logger.info("✓ Evaluation layer ready")
            
            logger.info("\n" + "="*70)
            logger.info("✓ SYSTEM FULLY INITIALIZED AND READY")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"✗ Initialization failed: {e}")
            raise
    
    def search(self, query: str) -> Dict:
        """
        Execute semantic search with RAG.
        
        Args:
            query: User query string
            
        Returns:
            Dict with complete pipeline results
        """
        logger.info("\n" + "╔" + "="*68 + "╗")
        logger.info("║ " + " "*66 + " ║")
        logger.info("║ " + f"USER QUERY: {query}".ljust(66) + " ║")
        logger.info("║ " + " "*66 + " ║")
        logger.info("╚" + "="*68 + "╝\n")
        
        # Retrieve
        retrieval_result = self.retriever.retrieve(query, top_k=5)
        
        # Generate with RAG
        rag_result = self.rag_pipeline.generate(retrieval_result)
        
        # Combine results
        complete_result = {
            'query': query,
            'retrieval': retrieval_result,
            'rag': rag_result
        }
        
        # Display results
        self._display_results(complete_result)
        
        return complete_result
    
    def _display_results(self, result: Dict) -> None:
        """Display formatted results with full explainability."""
        logger.info("\n" + "="*70)
        logger.info("EXPLAINABILITY & TRANSPARENCY LAYER")
        logger.info("="*70)
        
        # Extract key metrics
        query = result.get('query', '')
        decision = result['retrieval'].get('decision', 'UNKNOWN')
        domain_confidence = result['retrieval'].get('domain_confidence', 'UNKNOWN')
        max_similarity = result['retrieval'].get('max_similarity', 0.0)
        avg_similarity = result['retrieval'].get('avg_top_k_similarity', 0.0)
        
        logger.info(f"\n📊 DECISION METRICS:")
        logger.info(f"  • Query: '{query}'")
        logger.info(f"  • Max Similarity: {max_similarity:.4f}")
        logger.info(f"  • Avg Top-K Similarity: {avg_similarity:.4f}")
        logger.info(f"  • Domain Confidence: {domain_confidence}")
        logger.info(f"  • Final Decision: {decision}")
        
        # If rejected, show rejection reasoning
        if decision == "REJECT":
            logger.info("\n" + "="*70)
            logger.info("⚠️ QUERY REJECTED - HONEST FEEDBACK")
            logger.info("="*70)
            logger.info(f"\n{result['rag']['recommendation']}\n")
            logger.info("="*70)
            logger.info("WHY WAS THIS REJECTED?")
            logger.info("="*70)
            logger.info(
                "This is production safety behavior, not a failure.\n"
                "Out-of-domain detection prevents hallucinations by:\n"
                "1. Detecting when queries fall outside the product domain\n"
                "2. Refusing to generate confident recommendations\n"
                "3. Providing honest, trustworthy feedback\n\n"
                "This is how real-world AI systems should behave."
            )
            logger.info("="*70 + "\n")
            return
        
        # If accepted, show recommendation and products
        logger.info("\n" + "="*70)
        logger.info("✓ RECOMMENDATION")
        logger.info("="*70)
        logger.info(f"\n{result['rag']['recommendation']}\n")
        
        logger.info("="*70)
        logger.info("RETRIEVED PRODUCTS (Context Used)")
        logger.info("="*70)
        
        for product in result['retrieval']['results'][:5]:
            logger.info(f"\n#{product['rank']} {product['product_name']} (ID: {product['product_id']})")
            logger.info(f"  Category: {product['category']}")
            logger.info(f"  Price: ₹{product['price']}")
            logger.info(f"  Rating: {product['rating']}/5 ⭐")
            logger.info(f"  Similarity: {product['similarity_score']:.1%}")
            logger.info(f"  Description: {product['description'][:100]}...")
        
        logger.info("\n" + "="*70)
    
    def run_interactive(self):
        """Run interactive query mode."""
        logger.info("\n" + "="*70)
        logger.info("INTERACTIVE MODE")
        logger.info("="*70)
        logger.info("\nType queries to search products. Type 'exit' to quit.\n")
        
        while True:
            try:
                query = input("📝 Enter query (or 'exit'): ").strip()
                
                if query.lower() in ['exit', 'quit', 'q']:
                    logger.info("\nGoodbye!")
                    break
                
                if not query:
                    logger.warning("Please enter a query")
                    continue
                
                self.search(query)
                
            except KeyboardInterrupt:
                logger.info("\nInterrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
    
    def run_evaluation(self) -> Dict:
        """Run system evaluation."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING SYSTEM EVALUATION")
        logger.info("="*70)
        
        return self.evaluator.run_evaluation(
            self.retriever,
            self.rag_pipeline,
            self.raw_df
        )
    
    def demo_queries(self):
        """Run demonstration queries."""
        logger.info("\n" + "="*70)
        logger.info("RUNNING DEMONSTRATION QUERIES")
        logger.info("="*70)
        
        demo_queries = [
            "Budget running shoes for beginners",
            "Wireless headphones with strong bass and long battery life",
            "Laptop suitable for video editing under 80k",
            "Comfortable office chair for long hours",
            "Yoga mat for home workouts"
        ]
        
        results = []
        for i, query in enumerate(demo_queries, 1):
            logger.info(f"\n{'─'*70}")
            logger.info(f"Demo {i}/{len(demo_queries)}")
            logger.info(f"{'─'*70}")
            result = self.search(query)
            results.append(result)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Production-Grade E-Commerce Semantic Search RAG System"
    )
    
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Run system evaluation'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo queries'
    )
    
    parser.add_argument(
        '--rebuild-index',
        action='store_true',
        help='Rebuild vector index from scratch'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to search'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = SemanticSearchRAG(config_path=args.config)
        system.initialize(rebuild_index=args.rebuild_index)
        
        # Run requested operation
        if args.eval:
            # Run evaluation
            report = system.run_evaluation()
            
            logger.info("\n" + "="*70)
            logger.info("EVALUATION COMPLETE")
            logger.info("="*70)
            
            if report['hallucinations_detected'] > 0:
                logger.warning(f"⚠️ Hallucinations detected: {report['hallucinations_detected']}")
            else:
                logger.info("✓ No hallucinations detected")
            
            logger.info(f"✓ Average relevance: {report['average_relevance_score']}")
            logger.info(f"✓ Average explainability: {report['average_explainability_score']}")
        
        elif args.demo:
            # Run demo queries
            system.demo_queries()
        
        elif args.query:
            # Run single query
            system.search(args.query)
        
        else:
            # Interactive mode
            system.run_interactive()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
