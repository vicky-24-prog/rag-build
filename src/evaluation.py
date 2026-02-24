"""
EVALUATION LAYER

Validates system quality through qualitative and quantitative checks:
- Manual test queries with expected answers
- Error analysis and hallucination detection
- Explainability verification
- IR metrics: Recall@K, MRR (standard in search engines and RAG systems)
"""

import logging
import json
from typing import Dict, List, Tuple, Optional, Set
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class EvaluationLayer:
    """Evaluate system quality through qualitative and quantitative checks."""
    
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
        
        # Build labeled evaluation dataset
        # Format: query -> set of relevant product IDs
        # This provides ground truth for computing Recall@K and MRR metrics
        self.evaluation_dataset = self._build_evaluation_dataset()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _build_evaluation_dataset(self) -> Dict[str, Set[str]]:
        """
        Build labeled evaluation dataset for IR metrics (Recall@K, MRR).
        Maps queries to relevant product IDs based on expected categories.
        
        Returns:
            Dict mapping query text to expected categories and product IDs
        """
        dataset = {}
        
        for test_case in self.test_queries:
            query = test_case.get('query', '')
            expected_categories = test_case.get('expected_categories', [])
            
            if not query or not expected_categories:
                continue
            
            # Mark as key for retrieval in dataset
            dataset[query] = {
                'expected_categories': expected_categories,
                'expected_price_range': test_case.get('expected_price_range', 'any'),
                'relevant_product_ids': set()  # Will be populated during evaluation
            }
        
        logger.info(f"\n[EVALUATION DATASET]")
        logger.info(f"Loaded {len(dataset)} labeled queries with expected categories")
        logger.info(f"Relevant product IDs will be matched by category during retrieval\n")
        
        return dataset
    
    def run_evaluation(
        self,
        retriever,
        rag_pipeline,
        product_df
    ) -> Dict:
        """
        Run comprehensive evaluation on all test queries.
        
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
        Evaluate single test query using qualitative and quantitative metrics.
        
        Metrics: Relevance, Hallucination, Explainability, Recall@K, MRR, OOD Detection
        
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
            retrieved_product_ids = [str(p['product_id']) for p in retrieved_products]
            
            # Build set of relevant products based on expected categories
            relevant_product_ids = set()
            for _, product in product_df.iterrows():
                product_cat = str(product.get('category', '')).lower()
                if any(cat.lower() in product_cat for cat in expected_categories):
                    relevant_product_ids.add(str(product['product_id']))
            
            logger.info(f"\n[GROUND TRUTH]")
            logger.info(f"Expected categories: {expected_categories}")
            logger.info(f"Relevant products in dataset: {len(relevant_product_ids)}")
            
            # Generate
            rag_result = rag_pipeline.generate(retrieval_result)
            recommendation = rag_result.get('recommendation', '')
            
            # Qualitative checks
            checks = {
                'relevance': self._check_relevance(retrieved_products, expected_categories, product_df),
                'hallucination': self._check_hallucination(recommendation, retrieved_products, product_df),
                'explainability': self._check_explainability(recommendation, retrieved_products),
                'product_count': len(retrieved_products)
            }
            
            # Quantitative retrieval metrics
            recall_at_k = self._calculate_recall_at_k(
                retrieved_product_ids,
                relevant_product_ids,
                k=5
            )
            
            mrr = self._calculate_mean_reciprocal_rank(
                retrieved_product_ids,
                relevant_product_ids
            )
            
            # Log results
            logger.info(f"\n✓ Retrieved {checks['product_count']} products")
            logger.info(f"  Relevance Score: {checks['relevance']['score']:.1%}")
            logger.info(f"  Recall@5: {recall_at_k['recall_at_k']:.1%} ({recall_at_k['matches']}/{recall_at_k['total_relevant']})")
            logger.info(f"  MRR: {mrr['mrr']:.3f} - {mrr['explanation']}")
            logger.info(f"  Domain Confidence: {retrieval_result.get('domain_confidence', 'N/A')}")
            logger.info(f"  Decision: {retrieval_result.get('decision', 'N/A')}")
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
                'product_count': checks['product_count'],
                'relevance_score': checks['relevance']['score'],
                'relevance_explanation': checks['relevance']['explanation'],
                'hallucination_detected': checks['hallucination']['detected'],
                'hallucination_details': checks['hallucination']['details'],
                'explainability_score': checks['explainability']['score'],
                'explainability_feedback': checks['explainability']['feedback'],
                'ood_detection': {
                    'is_out_of_domain': retrieval_result.get('is_out_of_domain', False),
                    'domain_confidence': retrieval_result.get('domain_confidence', 'UNKNOWN'),
                    'decision': retrieval_result.get('decision', 'UNKNOWN'),
                    'max_similarity': retrieval_result.get('max_similarity', 0.0),
                    'avg_top_k_similarity': retrieval_result.get('avg_top_k_similarity', 0.0)
                },
                'retrieval_metrics': {
                    'recall_at_k': {
                        'value': recall_at_k['recall_at_k'],
                        'explanation': recall_at_k['explanation'],
                        'matches': recall_at_k['matches'],
                        'total_relevant': recall_at_k['total_relevant']
                    },
                    'mrr': {
                        'value': mrr['mrr'],
                        'explanation': mrr['explanation'],
                        'first_relevant_rank': mrr.get('first_relevant_rank')
                    }
                },
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
                'score': 0.5,
                'explanation': "No products or categories to evaluate"
            }
        
        matches = sum(
            1 for p in retrieved_products
            if any(cat.lower() in p['category'].lower() for cat in expected_categories)
        )
        
        score = matches / len(retrieved_products)
        
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
        """Check if LLM recommended products not in retrieval results."""
        # Get product IDs and names from retrieval
        retrieved_ids = {str(p['product_id']) for p in retrieved_products}
        retrieved_names = {p['product_name'].lower() for p in retrieved_products}
        
        hallucination_detected = False
        suspicious_phrases = []
        
        # Check if recommendation mentions products not in retrieval
        for _, product in product_df.iterrows():
            product_name = str(product['product_name']).lower()
            product_id = str(product['product_id'])
            
            if (product_id not in retrieved_ids and 
                product_name in recommendation.lower()):
                hallucination_detected = True
                suspicious_phrases.append(f"Product '{product['product_name']}' not in retrieval but mentioned in recommendation")
        
        # Check for invented product patterns
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
        """Check if recommendation explains its reasoning."""
        score = 0.0
        feedback = []
        
        # Check for explanation keywords
        explanation_keywords = [
            'because', 'reason', 'match', 'fits', 'suitable',
            'budget', 'price', 'feature', 'rating', 'spec'
        ]
        
        recommendation_lower = recommendation.lower()
        found_keywords = sum(
            1 for keyword in explanation_keywords
            if keyword in recommendation_lower
        )
        
        keyword_score = min(found_keywords / 3, 1.0)
        
        # Check if mentions specific product features
        feature_keywords = ['price', '₹', 'rs', 'rating', 'star', '⭐', 'budget']
        feature_score = 0.3 if any(
            kw in recommendation.lower()
            for kw in feature_keywords
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
    
    def _calculate_recall_at_k(
        self,
        retrieved_product_ids: List[str],
        relevant_product_ids: Set[str],
        k: int
    ) -> Dict:
        """
        Calculate Recall@K: Of all relevant items, how many did we retrieve in top-K?
        
        Formula: Recall@K = (# relevant items in top-K) / (total # relevant items)
        
        Args:
            retrieved_product_ids: Products returned by retriever (in order)
            relevant_product_ids: Ground truth set of relevant products
            k: Top-K parameter
            
        Returns:
            Dict with recall score and explanation
        """
        if not relevant_product_ids:
            return {
                'recall_at_k': 1.0,  # No relevant items means perfect recall
                'explanation': "No relevant items defined",
                'matches': 0,
                'total_relevant': 0
            }
        
        # Get top-K retrieved items
        top_k_retrieved = set(retrieved_product_ids[:k])
        
        # Find matches with relevant set
        matches = len(top_k_retrieved & relevant_product_ids)
        
        # Calculate recall
        recall = matches / len(relevant_product_ids)
        
        return {
            'recall_at_k': recall,
            'explanation': f"Retrieved {matches} out of {len(relevant_product_ids)} relevant items in top-{k}",
            'matches': matches,
            'total_relevant': len(relevant_product_ids)
        }
    
    def _calculate_mean_reciprocal_rank(
        self,
        retrieved_product_ids: List[str],
        relevant_product_ids: Set[str]
    ) -> Dict:
        """
        Calculate Mean Reciprocal Rank (MRR): At what position is the first correct result?
        
        Formula: MRR = 1 / (rank of first relevant item), or 0 if no relevant item found
        
        Args:
            retrieved_product_ids: Products returned by retriever (in order)
            relevant_product_ids: Ground truth set of relevant products
            
        Returns:
            Dict with MRR score and explanation
        """
        # Find first relevant item
        for rank, product_id in enumerate(retrieved_product_ids, 1):
            if product_id in relevant_product_ids:
                mrr = 1.0 / rank
                return {
                    'mrr': mrr,
                    'first_relevant_rank': rank,
                    'explanation': f"First relevant item found at rank {rank} (MRR = 1/{rank} = {mrr:.3f})",
                    'found': True
                }
        
        # No relevant item found
        return {
            'mrr': 0.0,
            'first_relevant_rank': None,
            'explanation': "No relevant item found in retrieved results",
            'found': False
        }
    
    def _generate_report(self, hallucination_count: int, product_df) -> Dict:
        """
        Generate comprehensive evaluation summary report.
        
        Returns:
            Dict with aggregated metrics and recommendations
        """
        total = len(self.results)
        
        # Aggregate qualitative metrics
        avg_relevance = sum(
            r.get('relevance_score', 0) for r in self.results
        ) / total if total > 0 else 0
        
        avg_explainability = sum(
            r.get('explainability_score', 0) for r in self.results
        ) / total if total > 0 else 0
        
        avg_products_retrieved = sum(
            r.get('product_count', 0) for r in self.results
        ) / total if total > 0 else 0
        
        # Aggregate IR metrics
        avg_recall_at_k = sum(
            r.get('retrieval_metrics', {}).get('recall_at_k', {}).get('value', 0) for r in self.results
        ) / total if total > 0 else 0
        
        avg_mrr = sum(
            r.get('retrieval_metrics', {}).get('mrr', {}).get('value', 0) for r in self.results
        ) / total if total > 0 else 0
        
        # Count OOD queries and decisions
        ood_queries = sum(
            1 for r in self.results if r.get('ood_detection', {}).get('is_out_of_domain', False)
        )
        
        rejected_queries = sum(
            1 for r in self.results if r.get('ood_detection', {}).get('decision') == 'REJECT'
        )
        
        return {
            'total_tests': total,
            'hallucinations_detected': hallucination_count,
            'hallucination_rate': f"{(hallucination_count/total)*100:.1f}%" if total > 0 else "N/A",
            'average_relevance_score': f"{avg_relevance:.1%}",
            'average_explainability_score': f"{avg_explainability:.1%}",
            'average_products_retrieved': f"{avg_products_retrieved:.1f}",
            'retrieval_metrics': {
                'average_recall_at_k': f"{avg_recall_at_k:.1%}",
                'explanation_recall': "Of all relevant items in the dataset, what % did we retrieve in top-K? Higher is better.",
                'average_mrr': f"{avg_mrr:.3f}",
                'explanation_mrr': "On average, at what position was the first relevant item? Higher is better (1.0 = rank 1)."
            },
            'safety_metrics': {
                'ood_queries_detected': ood_queries,
                'queries_rejected': rejected_queries,
                'acceptance_rate': f"{((total-rejected_queries)/total)*100:.1f}%" if total > 0 else "N/A",
                'explanation': "Rejected queries are marked OOD or low-confidence to prevent hallucinations"
            },
            'status': self._get_status(hallucination_count, avg_relevance),
            'recommendations_for_improvement': self._get_improvement_recommendations(
                avg_relevance,
                hallucination_count,
                total,
                avg_recall_at_k,
                avg_mrr
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
        total: int,
        avg_recall_at_k: float = 0.0,
        avg_mrr: float = 0.0
    ) -> List[str]:
        """Get actionable improvement suggestions based on evaluation metrics."""
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
        
        if avg_recall_at_k < 0.5:
            recommendations.append(
                "Low Recall@K: Increasing top_k or lowering similarity_threshold may help retrieve more relevant items"
            )
        
        if avg_mrr < 0.5:
            recommendations.append(
                "Low MRR: First results are ranked poorly. Consider using a re-ranker or improving embedding model"
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
        
        recommendations.append(
            "Regularly collect user feedback to refine the evaluation dataset and track metrics over time"
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
