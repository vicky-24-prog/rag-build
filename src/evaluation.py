"""
EVALUATION LAYER

Responsibility: Validate system quality through qualitative and quantitative checks

What Makes Evaluation "Production-Grade"?
- Manual test queries with KNOWN good answers
- Error analysis (why did retrieval fail?)
- Hallucination detection (is LLM inventing?)
- Explainability verification (can we explain choices?)
- Precision@K metrics (standard in IR)

This is NOT automated accuracy metrics (those require labeled data).
This IS manual validation + error analysis (what data scientists do in practice).
"""

import logging
import json
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationLayer:
    """
    Evaluate system quality through manual and quantitative checks.
    
    Design Principles:
    - Grounded in reality: Manual test queries with expectations
    - Error analysis: Understand why systems succeed or fail
    - Transparent: Show scores and reasoning
    - Actionable: Provide insights for improvement
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize evaluation layer.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.eval_config = self.config.get("evaluation", {})
        self.test_queries = self.eval_config.get("test_queries", [])
        self.results = []
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def run_evaluation(
        self,
        retriever,
        rag_pipeline,
        product_df
    ) -> Dict:
        """
        Run comprehensive evaluation.
        
        Pipeline:
        1. For each test query:
           a. Retrieve products
           b. Generate recommendations
           c. Check relevance
           d. Check for hallucinations
           e. Verify explainability
        2. Aggregate results
        3. Generate report
        
        Args:
            retriever: RetrieverLayer instance
            rag_pipeline: RAGPipeline instance
            product_df: Product dataframe (for ground truth)
            
        Returns:
            Dict with evaluation results
        """
        logger.info("\n" + "="*60)
        logger.info("SYSTEM EVALUATION")
        logger.info("="*60)
        
        logger.info(f"\nRunning evaluation on {len(self.test_queries)} test queries...")
        
        self.results = []
        hallucination_count = 0
        total_tests = len(self.test_queries)
        
        for i, test_case in enumerate(self.test_queries, 1):
            logger.info(f"\n{'─'*60}")
            logger.info(f"Test {i}/{total_tests}: '{test_case['query']}'")
            logger.info(f"{'─'*60}")
            
            result = self._evaluate_query(
                test_case,
                retriever,
                rag_pipeline,
                product_df
            )
            
            self.results.append(result)
            
            if result['hallucination_detected']:
                hallucination_count += 1
        
        # Aggregate report
        report = self._generate_report(hallucination_count, product_df)
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(json.dumps(report, indent=2))
        
        # Save results if configured
        if self.eval_config.get("save_results", False):
            self._save_results(report)
        
        logger.info("="*60 + "\n")
        
        return report
    
    def _evaluate_query(
        self,
        test_case: Dict,
        retriever,
        rag_pipeline,
        product_df
    ) -> Dict:
        """
        Evaluate single test query.
        
        Args:
            test_case: Test query case with expected categories
            retriever: Retriever instance
            rag_pipeline: RAG instance
            product_df: Product dataframe
            
        Returns:
            Dict with evaluation results for this query
        """
        query = test_case['query']
        expected_categories = test_case.get('expected_categories', [])
        
        try:
            # Retrieve
            retrieval_result = retriever.retrieve(query, top_k=5)
            retrieved_products = retrieval_result.get('results', [])
            
            # Generate
            rag_result = rag_pipeline.generate(retrieval_result)
            recommendation = rag_result.get('recommendation', '')
            
            # Evaluate
            checks = {
                'relevance': self._check_relevance(retrieved_products, expected_categories, product_df),
                'hallucination': self._check_hallucination(recommendation, retrieved_products, product_df),
                'explainability': self._check_explainability(recommendation, retrieved_products),
                'product_count': len(retrieved_products)
            }
            
            # Log results
            logger.info(f"\n✓ Retrieved {checks['product_count']} products")
            logger.info(f"  Relevance Score: {checks['relevance']['score']:.1%}")
            logger.info(f"  Hallucination Risk: {'HIGH ⚠️' if checks['hallucination']['detected'] else 'LOW ✓'}")
            logger.info(f"  Explainability: {checks['explainability']['score']:.0%}")
            
            if checks['relevance']['explanation']:
                logger.info(f"  Relevance Check: {checks['relevance']['explanation']}")
            
            return {
                'query': query,
                'expected_categories': expected_categories,
                'retrieved_products': [
                    {
                        'id': p['product_id'],
                        'name': p['product_name'],
                        'category': p['category'],
                        'similarity': f"{p['similarity_score']:.1%}"
                    }
                    for p in retrieved_products
                ],
                'relevance_score': checks['relevance']['score'],
                'relevance_explanation': checks['relevance']['explanation'],
                'hallucination_detected': checks['hallucination']['detected'],
                'hallucination_details': checks['hallucination']['details'],
                'explainability_score': checks['explainability']['score'],
                'explainability_feedback': checks['explainability']['feedback'],
                'recommendation_preview': recommendation[:200] + "..." if len(recommendation) > 200 else recommendation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating query: {e}")
            return {
                'query': query,
                'error': str(e),
                'hallucination_detected': False
            }
    
    def _check_relevance(
        self,
        retrieved_products: List[Dict],
        expected_categories: List[str],
        product_df
    ) -> Dict:
        """
        Check if retrieved products match expected categories.
        
        Args:
            retrieved_products: Products from retriever
            expected_categories: Categories we expect
            product_df: Full product dataframe for lookup
            
        Returns:
            Dict with relevance score and explanation
        """
        if not retrieved_products or not expected_categories:
            return {
                'score': 0.5,  # Neutral
                'explanation': "No products or categories to evaluate"
            }
        
        # Count matches
        matches = sum(
            1 for p in retrieved_products
            if any(cat.lower() in p['category'].lower() for cat in expected_categories)
        )
        
        score = matches / len(retrieved_products)
        
        # Explanation
        if score == 1.0:
            explanation = "✓ All retrieved products in expected categories"
        elif score >= 0.6:
            explanation = "✓ Most products match expected categories"
        elif score >= 0.3:
            explanation = "⚠️ Only some products match expected categories"
        else:
            explanation = "✗ Retrieved products don't match expected categories"
        
        return {
            'score': score,
            'explanation': explanation,
            'matches': matches,
            'total': len(retrieved_products)
        }
    
    def _check_hallucination(
        self,
        recommendation: str,
        retrieved_products: List[Dict],
        product_df
    ) -> Dict:
        """
        Check if LLM recommended products not in retrieval.
        
        Args:
            recommendation: Generated recommendation text
            retrieved_products: Products from retriever
            product_df: Full product dataframe
            
        Returns:
            Dict with hallucination risk
        """
        # Get product IDs and names from retrieval
        retrieved_ids = {p['product_id'] for p in retrieved_products}
        retrieved_names = {p['product_name'].lower() for p in retrieved_products}
        
        # Check if recommendation mentions products not in retrieval
        hallucination_detected = False
        suspicious_phrases = []
        
        # Check against all products in store
        for _, product in product_df.iterrows():
            product_name = str(product['product_name']).lower()
            product_id = str(product['product_id'])
            
            # Only flag if product NOT in retrieval but mentioned in recommendation
            if (product_id not in retrieved_ids and 
                product_name in recommendation.lower()):
                hallucination_detected = True
                suspicious_phrases.append(f"Product '{product['product_name']}' not in retrieval but mentioned in recommendation")
        
        # Check for invented product patterns (e.g., "XYZ Pro 3000")
        if any(phrase in recommendation.lower() for phrase in 
               ['new model', 'just released', 'upcoming', 'exclusive', 'limited edition']):
            suspicious_phrases.append("Recommendation contains temporal claims (new/upcoming) - potential hallucination")
        
        return {
            'detected': hallucination_detected,
            'details': suspicious_phrases if suspicious_phrases else ["No hallucinations detected"]
        }
    
    def _check_explainability(
        self,
        recommendation: str,
        retrieved_products: List[Dict]
    ) -> Dict:
        """
        Check if recommendation explains reasoning.
        
        Args:
            recommendation: Generated recommendation text
            retrieved_products: Products from retriever
            
        Returns:
            Dict with explainability score
        """
        score = 0.0
        feedback = []
        
        # Check for explanation words
        explanation_keywords = [
            'because', 'reason', 'match', 'fits', 'suitable',
            'budget', 'price', 'feature', 'rating', 'spec'
        ]
        
        recommendation_lower = recommendation.lower()
        found_keywords = sum(
            1 for keyword in explanation_keywords
            if keyword in recommendation_lower
        )
        
        # Calculate score based on keyword frequency
        keyword_score = min(found_keywords / 3, 1.0)  # Normalize by ~3 keywords expected
        
        # Check if mentions specific product features
        feature_score = 0.3 if any(
            p['price'] in recommendation or p['rating'] in recommendation
            for p in retrieved_products
        ) else 0.0
        
        score = (keyword_score * 0.7) + (feature_score * 0.3)
        
        if score >= 0.8:
            feedback.append("✓ Clear reasoning with specific details")
        elif score >= 0.5:
            feedback.append("⚠️ Some reasoning provided but could be more detailed")
        else:
            feedback.append("✗ Limited explanation of why products match")
        
        if len(recommendation) < 100:
            feedback.append("✓ Concise response")
        elif len(recommendation) < 500:
            feedback.append("✓ Well-detailed response")
        else:
            feedback.append("⚠️ Response is quite long")
        
        return {
            'score': score,
            'feedback': feedback
        }
    
    def _generate_report(self, hallucination_count: int, product_df) -> Dict:
        """
        Generate evaluation summary report.
        
        Args:
            hallucination_count: Number of hallucinations detected
            product_df: Product dataframe
            
        Returns:
            Dict with summary report
        """
        total = len(self.results)
        
        # Aggregate metrics
        avg_relevance = sum(
            r.get('relevance_score', 0) for r in self.results
        ) / total if total > 0 else 0
        
        avg_explainability = sum(
            r.get('explainability_score', 0) for r in self.results
        ) / total if total > 0 else 0
        
        avg_products_retrieved = sum(
            r.get('product_count', 0) for r in self.results
        ) / total if total > 0 else 0
        
        return {
            'total_tests': total,
            'hallucinations_detected': hallucination_count,
            'hallucination_rate': f"{(hallucination_count/total)*100:.1f}%" if total > 0 else "N/A",
            'average_relevance_score': f"{avg_relevance:.1%}",
            'average_explainability_score': f"{avg_explainability:.1%}",
            'average_products_retrieved': f"{avg_products_retrieved:.1f}",
            'status': self._get_status(hallucination_count, avg_relevance),
            'recommendations_for_improvement': self._get_improvement_recommendations(
                avg_relevance,
                hallucination_count,
                total
            ),
            'detailed_results': self.results
        }
    
    def _get_status(self, hallucination_count: int, avg_relevance: float) -> str:
        """Determine overall system status."""
        if hallucination_count > 0:
            return "⚠️ NEEDS ATTENTION - Hallucinations detected"
        elif avg_relevance < 0.6:
            return "⚠️ NEEDS IMPROVEMENT - Low relevance scores"
        elif avg_relevance < 0.8:
            return "✓ ACCEPTABLE - Room for improvement"
        else:
            return "✓ GOOD - System performing well"
    
    def _get_improvement_recommendations(
        self,
        avg_relevance: float,
        hallucination_count: int,
        total: int
    ) -> List[str]:
        """
        Get suggestions for system improvement.
        
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        if hallucination_count > 0:
            recommendations.append(
                "Hallucinations detected: Review LLM instructions and temperature settings"
            )
        
        if avg_relevance < 0.6:
            recommendations.append(
                "Low relevance: Consider refining similarity threshold or embedding model"
            )
        
        if avg_relevance < 0.8:
            recommendations.append(
                "Partial relevance issues: Expand product dataset or improve descriptions"
            )
        
        recommendations.append(
            "Increase product dataset size for better coverage"
        )
        
        recommendations.append(
            "Add more diverse product descriptions (e.g., pros/cons, use cases)"
        )
        
        recommendations.append(
            "Monitor LLM responses for consistent hallucination patterns"
        )
        
        return recommendations
    
    def _save_results(self, report: Dict) -> None:
        """Save evaluation results to file."""
        output_path = self.eval_config.get("results_path", "evaluation_results.json")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Results saved to {output_path}")


def main():
    """Test evaluation layer independently."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting Evaluation Layer Test")
    logger.info("="*60)
    
    # Build full pipeline
    from ingestion import DataIngestionLayer
    from preprocessing import PreprocessingLayer
    from embeddings import EmbeddingLayer
    from vector_store import VectorStoreLayer
    from retriever import RetrieverLayer
    from rag_pipeline import RAGPipeline
    
    ingestion = DataIngestionLayer(config_path="config/config.yaml")
    raw_df = ingestion.ingest()
    
    preprocessor = PreprocessingLayer(config_path="config/config.yaml")
    clean_df = preprocessor.preprocess(raw_df)
    
    embedder = EmbeddingLayer(config_path="config/config.yaml")
    embeddings, product_ids = embedder.embed(clean_df)
    
    vector_store = VectorStoreLayer(config_path="config/config.yaml")
    vector_store.build(embeddings, product_ids)
    
    retriever = RetrieverLayer(
        vector_store=vector_store,
        embedder=embedder,
        df_products=raw_df,
        config_path="config/config.yaml"
    )
    
    rag = RAGPipeline(config_path="config/config.yaml")
    
    # Run evaluation
    evaluator = EvaluationLayer(config_path="config/config.yaml")
    report = evaluator.run_evaluation(retriever, rag, raw_df)


if __name__ == "__main__":
    main()
