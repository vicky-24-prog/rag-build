"""Evaluation metrics - simplified."""

import yaml


def compute_recall(retrieved_ids, relevant_ids, k):
    """Recall@K: How many relevant items did we retrieve?"""
    if not relevant_ids:
        return 1.0, 0, 0
    
    retrieved_set = set(retrieved_ids[:k])
    matches = len(retrieved_set & relevant_ids)
    recall = matches / len(relevant_ids)
    return recall, matches, len(relevant_ids)


def compute_mrr(retrieved_ids, relevant_ids):
    """MRR: At what rank is the first relevant item?"""
    for rank, pid in enumerate(retrieved_ids, 1):
        if pid in relevant_ids:
            return 1.0 / rank, rank
    return 0.0, None


def get_relevant_products(df, categories):
    """Get products matching expected categories."""
    relevant = set()
    for _, product in df.iterrows():
        cat = str(product['category']).lower()
        if any(exp_cat.lower() in cat for exp_cat in categories):
            relevant.add(str(product['product_id']))
    return relevant


def run_evaluation(retrieve_fn, df, config_path="config/config.yaml"):
    """Run evaluation on test queries."""
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    test_queries = config.get('evaluation', {}).get('test_queries', [])
    
    results = []
    total_recall = 0
    total_mrr = 0
    
    print(f"\nRunning evaluation on {len(test_queries)} queries...\n")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        expected_cats = test_case.get('expected_categories', [])
        
        print(f"[{i}/{len(test_queries)}] {query}")
        
        # Retrieve
        result = retrieve_fn(query)
        retrieved_ids = [str(p['product_id']) for p in result.get('results', [])]
        
        # Ground truth
        relevant_ids = get_relevant_products(df, expected_cats)
        
        # Metrics
        recall, matches, total_rel = compute_recall(retrieved_ids, relevant_ids, k=5)
        mrr, rank = compute_mrr(retrieved_ids, relevant_ids)
        
        print(f"  Retrieved: {len(retrieved_ids)} | Relevant in dataset: {len(relevant_ids)}")
        print(f"  Recall@5: {recall:.1%} ({matches}/{total_rel}) | MRR: {mrr:.3f}" + 
              (f" (rank {rank})" if rank else " (none found)"))
        print(f"  Decision: {result.get('decision')} | Confidence: {result.get('confidence', 'N/A')}\n")
        
        total_recall += recall
        total_mrr += mrr
        
        results.append({
            'query': query,
            'recall': recall,
            'mrr': mrr,
            'decision': result.get('decision')
        })
    
    avg_recall = total_recall / len(test_queries) if test_queries else 0
    avg_mrr = total_mrr / len(test_queries) if test_queries else 0
    
    print("="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Recall@5: {avg_recall:.1%}")
    print(f"Average MRR: {avg_mrr:.3f}")
    print(f"Queries evaluated: {len(test_queries)}")
    print("="*60)
    
    return {
        'avg_recall': avg_recall,
        'avg_mrr': avg_mrr,
        'results': results
    }
