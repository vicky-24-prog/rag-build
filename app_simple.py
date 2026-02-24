"""Main application - simplified RAG system."""

import argparse
import sys
from pathlib import Path

from src import data, embed, vector, retrieve, generate, eval


def build_system(force_rebuild=False):
    """Initialize the RAG system."""
    
    print("="*60)
    print("SEMANTIC SEARCH RAG SYSTEM")
    print("="*60)
    
    # Load and clean data
    print("\n[1/4] Loading data...")
    df = data.load_and_clean_data(force_reload=force_rebuild)
    
    # Generate embeddings
    print("\n[2/4] Generating embeddings...")
    embeddings, product_ids = embed.get_embeddings(df, force_rebuild=force_rebuild)
    
    # Build/load FAISS index
    print("\n[3/4] Building vector index...")
    index_path = "models/faiss_index.bin"
    metadata_path = "models/metadata.pkl"
    
    if force_rebuild or not Path(index_path).exists():
        index, metadata = vector.build_index(embeddings, product_ids)
    else:
        index, metadata = vector.load_index()
    
    # Setup LLM
    print("\n[4/4] Initializing LLM...")
    model = generate.setup_gemini()
    
    print("\n" + "="*60)
    print("SYSTEM READY")
    print("="*60 + "\n")
    
    return df, index, metadata, model


def process_query(query, df, index, metadata, model):
    """Process a single query."""
    
    print(f"\nQuery: {query}")
    print("-"*60)
    
    # Retrieve
    result = retrieve.retrieve(query, index, metadata, df)
    
    # Generate
    recommendation = generate.generate_recommendation(query, result, model)
    
    # Display
    if result['decision'] == 'ACCEPT':
        print(f"\nFound {len(result['results'])} products")
        print(f"  Confidence: {result['confidence']}")
        print(f"  Max Similarity: {result['max_similarity']:.3f}")
        
        print("\nTop Results:")
        for i, p in enumerate(result['results'][:3], 1):
            print(f"  {i}. {p['product_name']} ({p['category']})")
            print(f"     Price: ₹{p['price']} | Rating: {p['rating']}/5 | Match: {p['similarity_score']:.1%}")
        
        if recommendation['generated']:
            print(f"\nRecommendation:\n{recommendation['recommendation']}")
    else:
        print(f"\n[REJECTED] {result['reason']}")
        print(recommendation['recommendation'])
    
    print("-"*60)
    
    return result, recommendation


def main():
    parser = argparse.ArgumentParser(description="Semantic Search RAG System")
    parser.add_argument('--query', type=str, help='Single query to process')
    parser.add_argument('--rebuild-index', action='store_true', help='Force rebuild index from scratch')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    args = parser.parse_args()
    
    # Build system
    df, index, metadata, model = build_system(force_rebuild=args.rebuild_index)
    
    # Evaluation mode
    if args.eval:
        def retrieve_fn(query):
            return retrieve.retrieve(query, index, metadata, df)
        eval.run_evaluation(retrieve_fn, df)
        return
    
    # Single query mode
    if args.query:
        process_query(args.query, df, index, metadata, model)
        return
    
    # Interactive mode
    print("Type your query (or 'exit' to quit):\n")
    while True:
        try:
            query = input("> ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            process_query(query, df, index, metadata, model)
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
