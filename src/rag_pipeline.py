"""
RAG GENERATION LAYER

Responsibility: Use LLM to generate recommendations based on RETRIEVED products only

Why RAG (Retrieval-Augmented Generation)?
- LLMs are amazing at language but terrible at facts
- LLMs hallucinate: They confidently generate false products/prices
- Solution: Retrieve first, then generate with context
- Model never sees data outside the retrieved set

Hallucination Prevention Strategy:
1. Retrieve Top-K products from vector store (FACTUAL)
2. Format products into context string
3. Give LLM ONLY this context + strict instructions
4. LLM can ONLY recommend what's in context
5. Results are grounded in real data

Why NOT raw LLM:
- User: "Laptop under 50k"
- Raw LLM might generate: "Try the XYZ Laptop Pro 3000 - only 45k! (HALLUCINATED)"
- RAG LLM: "Available options are [retrieved products only]"
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Generate recommendations using retrieved products as context.
    
    Design Principles:
    - Context-first: LLM only sees retrieved products
    - Instruction-led: Clear constraints prevent hallucination
    - Explainable: Show reasoning
    - Grounded: Every recommendation is traceable to source
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize RAG pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.rag_config = self.config.get("rag", {})
        self.llm = None
        self.llm_provider = self.rag_config.get("llm_provider", "gemini")
        
        # Initialize LLM
        self._init_llm()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _init_llm(self) -> None:
        """Initialize LLM client."""
        provider = self.llm_provider
        
        if provider == "gemini":
            self._init_gemini()
        else:
            logger.warning(f"Unknown LLM provider: {provider}. Will attempt generic initialization.")
    
    def _init_gemini(self) -> None:
        """Initialize Google Gemini API."""
        import os
        
        api_key = os.getenv(self.rag_config.get("api_key_env", "GEMINI_API_KEY"))
        
        if not api_key:
            logger.warning(
                f"API key not found in environment variable "
                f"'{self.rag_config.get('api_key_env', 'GEMINI_API_KEY')}'. "
                "RAG generation will not work. Please set the API key."
            )
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.llm = genai.GenerativeModel(
                model_name=self.rag_config.get("model_name", "gemini-pro")
            )
            logger.info(f"✓ Initialized Gemini LLM: {self.rag_config.get('model_name', 'gemini-pro')}")
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            raise
    
    def generate(self, retrieval_result: Dict) -> Dict:
        """
        Generate recommendation based on retrieved products.
        
        Pipeline:
        1. Extract products from retrieval result
        2. Format context string
        3. Build prompt with strict instructions
        4. Call LLM with temperature=low (factual mode)
        5. Parse response
        
        Args:
            retrieval_result: Output from RetrieverLayer.retrieve()
            
        Returns:
            Dict with:
            {
                'query': original query,
                'recommendation': LLM's recommendation text,
                'reasoning': Why these products match,
                'retrieved_products': Products from retrieval,
                'generated_with_llm': whether LLM was actually used
            }
        """
        logger.info("\n" + "="*60)
        logger.info("RAG GENERATION LAYER")
        logger.info("="*60)
        
        query = retrieval_result.get('query')
        products = retrieval_result.get('results', [])
        
        logger.info(f"Query: '{query}'")
        logger.info(f"Context products: {len(products)}")
        
        # If no LLM, return retrieval results with explanation
        if self.llm is None:
            logger.warning("LLM not initialized. Returning retrieval-only results.")
            return self._generate_without_llm(query, products)
        
        # Build context
        context = self._build_context(products)
        
        # Build prompt
        prompt = self._build_prompt(query, context)
        
        logger.info(f"\nContext window size: {len(context)} characters")
        logger.info("Calling LLM for generation...")
        
        # Generate using LLM
        try:
            response = self.llm.generate_content(
                prompt,
                generation_config=self._get_generation_config()
            )
            
            recommendation = response.text
            logger.info("✓ LLM generation successful")
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            logger.info("Falling back to retrieval-only results")
            return self._generate_without_llm(query, products)
        
        logger.info("="*60 + "\n")
        
        return {
            'query': query,
            'recommendation': recommendation,
            'retrieved_products': products,
            'context_used': context,
            'generated_with_llm': True,
            'llm_model': self.rag_config.get('model_name'),
            'hallucination_risk': 'LOW (retrieval-grounded)'
        }
    
    def _generate_without_llm(self, query: str, products: List[Dict]) -> Dict:
        """
        Generate response without LLM (fallback).
        
        Simply formats retrieval results into readable text.
        """
        recommendation = self._format_product_list(query, products)
        
        return {
            'query': query,
            'recommendation': recommendation,
            'retrieved_products': products,
            'generated_with_llm': False,
            'hallucination_risk': 'NONE (retrieval only, no generation)'
        }
    
    def _build_context(self, products: List[Dict]) -> str:
        """
        Build formatted context string from products.
        
        Args:
            products: List of product dicts from retriever
            
        Returns:
            Formatted string with product details
        """
        max_products = self.rag_config.get("max_products_in_context", 5)
        products_to_include = products[:max_products]
        
        context_lines = []
        for i, product in enumerate(products_to_include, 1):
            context_lines.append(
                f"{i}. {product['product_name']} (ID: {product['product_id']})\n"
                f"   Category: {product['category']}\n"
                f"   Price: ₹{product['price']}\n"
                f"   Rating: {product['rating']}/5\n"
                f"   Description: {product['description']}\n"
                f"   Relevance Score: {product['similarity_score']:.2%}\n"
            )
        
        return "\n".join(context_lines)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """
        Build prompt for LLM with strict instructions.
        
        Args:
            query: User query
            context: Formatted products context
            
        Returns:
            Formatted prompt string
        """
        instruction = self.rag_config.get("instruction", 
            "You are an e-commerce assistant. ONLY recommend from provided products. "
            "Never invent products or features."
        )
        
        prompt = f"""
{instruction}

USER QUERY:
{query}

AVAILABLE PRODUCTS (ONLY THESE CAN BE RECOMMENDED):
{context}

TASK:
1. Identify which product(s) best match the user's needs
2. Explain WHY each product is relevant (use specific details from above)
3. If multiple products fit, explain the tradeoffs
4. If NO product in the list matches well, say so clearly - don't recommend suboptimal options

CRITICAL: Do NOT recommend any product not in the list above. Do NOT invent features or prices.

RESPONSE:
"""
        return prompt
    
    def _format_product_list(self, query: str, products: List[Dict]) -> str:
        """Format products as readable text."""
        lines = [
            f"Query: {query}\n",
            f"Found {len(products)} matching product(s):\n"
        ]
        
        for i, product in enumerate(products, 1):
            lines.append(
                f"\n{i}. {product['product_name']} (ID: {product['product_id']})\n"
                f"   Category: {product['category']}\n"
                f"   Price: ₹{product['price']}\n"
                f"   Rating: {product['rating']}/5\n"
                f"   Relevance: {product['similarity_score']:.1%}\n"
                f"   {product['description']}"
            )
        
        return "\n".join(lines)
    
    def _get_generation_config(self):
        """Get generation configuration for LLM."""
        # For Gemini API
        try:
            from google.generativeai.types.generation_types import GenerationConfig
            return GenerationConfig(
                temperature=self.rag_config.get("temperature", 0.3),
                top_p=self.rag_config.get("top_p", 0.9),
                top_k=self.rag_config.get("top_k", 40),
                max_output_tokens=self.rag_config.get("max_output_tokens", 200)
            )
        except ImportError:
            # Fallback: return dict
            return {
                "temperature": self.rag_config.get("temperature", 0.3),
                "top_p": self.rag_config.get("top_p", 0.9),
                "max_output_tokens": self.rag_config.get("max_output_tokens", 200)
            }
    
    def explain_rag(self) -> Dict:
        """
        Provide educational explanation of RAG.
        
        Returns:
            Dict with explanations
        """
        return {
            "why_rag_prevents_hallucination": [
                "LLMs generate plausible-sounding text, even when false",
                "Example hallucination: User asks for laptop under 50k, LLM invents 'XYZ Pro 3000'",
                "RAG solution: Products come from retrieval, not generation",
                "LLM can ONLY recommend what's in the vector store"
            ],
            "rag_pipeline": [
                "1. User query",
                "2. Semantic search retrieves relevant products (GROUNDED IN DATA)",
                "3. Retrieved products formatted into context",
                "4. LLM receives context + strict instructions",
                "5. LLM generates explanation using only provided products",
                "6. Response is factually accurate (bounded by retrieval)"
            ],
            "why_temperature_low": [
                "Temperature controls randomness in LLM output",
                "High (0.7+): Creative but less factual",
                "Low (0.3): Deterministic and factual - ideal for RAG",
                "We use 0.3 to minimize hallucination risk"
            ],
            "why_max_products_in_context": [
                "LLM has context window limit (~2k tokens for lite models)",
                "Including too many products wastes tokens",
                "Top-5 products usually sufficient for good recommendations",
                "Keeps response focused and fast"
            ],
            "grounding_vs_pure_llm": {
                "pure_llm": {
                    "pros": "Creative, no retrieval needed, fast initial response",
                    "cons": "Hallucinates, invents products, not reliable",
                    "use_case": "Creative writing, brainstorming (NOT e-commerce)"
                },
                "rag": {
                    "pros": "Factually grounded, no hallucinations, explainable",
                    "cons": "Only knows products in store, need good retrieval",
                    "use_case": "E-commerce, customer support (BEST for this)"
                }
            }
        }


def main():
    """Test RAG pipeline independently."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("Starting RAG Pipeline Test")
    logger.info("="*60)
    
    # Build full pipeline
    from ingestion import DataIngestionLayer
    from preprocessing import PreprocessingLayer
    from embeddings import EmbeddingLayer
    from vector_store import VectorStoreLayer
    from retriever import RetrieverLayer
    
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
    
    # Initialize RAG
    rag = RAGPipeline(config_path="config/config.yaml")
    
    # Test with sample query
    query = "Budget running shoes for beginners"
    logger.info(f"\nTest Query: '{query}'")
    
    # Retrieve
    retrieval_result = retriever.retrieve(query, top_k=3)
    
    # Generate
    rag_result = rag.generate(retrieval_result)
    
    logger.info("\nGenerated Recommendation:")
    logger.info(rag_result['recommendation'])
    
    # Show explanations
    logger.info("\n" + "="*60)
    logger.info("RAG EXPLAINED")
    logger.info("="*60)
    explanations = rag.explain_rag()
    import json
    logger.info(json.dumps(explanations, indent=2))


if __name__ == "__main__":
    main()
