"""Retrieval with OOD detection - simplified."""

import numpy as np
from src.embed import embed_query
from src import vector


def retrieve(query, index, metadata, df, top_k=5, sim_threshold=0.3, domain_threshold=0.65):
    """Retrieve products with OOD detection."""
    
    # Embed query
    query_emb = embed_query(query)
    
    # Search
    results = vector.search(index, metadata, query_emb, top_k)
    
    if not results:
        return {
            'decision': 'REJECT',
            'confidence': 'LOW',
            'reason': 'No results found',
            'results': []
        }
    
    # Filter by similarity threshold
    filtered = [(pid, score) for pid, score in results if score >= sim_threshold]
    
    if not filtered:
        return {
            'decision': 'REJECT',
            'confidence': 'LOW',
            'reason': f'All results below threshold {sim_threshold}',
            'results': []
        }
    
    # OOD detection
    max_sim = filtered[0][1]
    avg_sim = np.mean([score for _, score in filtered])
    
    if max_sim < domain_threshold:
        return {
            'decision': 'REJECT',
            'confidence': 'LOW',
            'reason': f'Max similarity {max_sim:.3f} < domain threshold {domain_threshold}',
            'max_similarity': max_sim,
            'avg_similarity': avg_sim,
            'results': []
        }
    
    # Format results
    products = []
    for pid, score in filtered:
        product = df[df['product_id'] == pid].iloc[0]
        products.append({
            'product_id': str(pid),
            'product_name': product['product_name'],
            'category': product['category'],
            'description': product['description'],
            'price': int(product['price']),
            'rating': float(product['rating']),
            'similarity_score': float(score)
        })
    
    return {
        'decision': 'ACCEPT',
        'confidence': 'HIGH' if max_sim >= 0.75 else 'MEDIUM',
        'max_similarity': max_sim,
        'avg_similarity': avg_sim,
        'results': products
    }
