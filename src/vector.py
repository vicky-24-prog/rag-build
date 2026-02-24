"""FAISS vector store - simplified."""

import numpy as np
import faiss
import pickle
from pathlib import Path


def build_index(embeddings, product_ids, index_path="models/faiss_index.bin", metadata_path="models/metadata.pkl"):
    """Build FAISS index."""
    dim = embeddings.shape[1]
    
    # Create flat index (exact search)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    
    # Save
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    
    metadata = {
        'product_ids': [str(pid) for pid in product_ids],
        'dimension': dim,
        'total': len(product_ids)
    }
    pickle.dump(metadata, open(metadata_path, 'wb'))
    
    print(f"Built FAISS index: {len(product_ids)} vectors, dim={dim}")
    return index, metadata


def load_index(index_path="models/faiss_index.bin", metadata_path="models/metadata.pkl"):
    """Load FAISS index."""
    index = faiss.read_index(index_path)
    metadata = pickle.load(open(metadata_path, 'rb'))
    total = metadata.get('total', metadata.get('total_vectors', len(metadata.get('product_ids', []))))
    print(f"Loaded FAISS index: {total} vectors")
    return index, metadata


def search(index, metadata, query_embedding, top_k=5):
    """Search index."""
    query_vec = query_embedding.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    
    product_ids = metadata['product_ids']
    results = [
        (product_ids[idx], float(distances[0][i]))
        for i, idx in enumerate(indices[0])
        if idx >= 0
    ]
    return results
