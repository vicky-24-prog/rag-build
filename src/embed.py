"""Embedding generation using Gemini - simplified."""

import numpy as np
import pickle
from pathlib import Path
import google.generativeai as genai
import os
from tqdm import tqdm


def get_embeddings(df, model_name="models/gemini-embedding-001", cache_path="data/embeddings.npy", force_rebuild=False):
    """Generate or load embeddings."""
    
    if not force_rebuild and Path(cache_path).exists():
        embeddings = np.load(cache_path)
        product_ids = df['product_id'].values
        print(f"Loaded embeddings: {embeddings.shape}")
        return embeddings, product_ids
    
    # Configure Gemini
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    genai.configure(api_key=api_key)
    
    # Generate embeddings
    texts = df['clean_text'].tolist()
    product_ids = df['product_id'].values
    embeddings = []
    
    print(f"Generating embeddings for {len(texts)} products...")
    for text in tqdm(texts):
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Cache
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    
    print(f"Generated embeddings: {embeddings.shape}")
    return embeddings, product_ids


def embed_query(query, model_name="models/gemini-embedding-001"):
    """Embed a single query."""
    # Ensure genai is configured
    api_key = os.environ.get('GOOGLE_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
    
    result = genai.embed_content(
        model=model_name,
        content=query.lower().strip(),
        task_type="retrieval_query"
    )
    embedding = np.array(result['embedding'], dtype=np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding
