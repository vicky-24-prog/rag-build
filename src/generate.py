"""RAG generation with Gemini - simplified."""

import google.generativeai as genai
import os


def setup_gemini(model_name="gemini-2.0-flash"):
    """Initialize Gemini model."""
    api_key = os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_recommendation(query, retrieval_result, model, max_products=5):
    """Generate recommendation from retrieved products."""
    
    if retrieval_result['decision'] == 'REJECT':
        return {
            'recommendation': f"I couldn't find relevant products for '{query}'. Please try a different search term.",
            'generated': False
        }
    
    # Format context
    products = retrieval_result['results'][:max_products]
    context = "\n\n".join([
        f"Product {i+1}: {p['product_name']}\n"
        f"Category: {p['category']} | Price: ₹{p['price']} | Rating: {p['rating']}/5\n"
        f"Description: {p['description']}"
        for i, p in enumerate(products)
    ])
    
    # Build prompt
    prompt = f"""You are an e-commerce product assistant. Based on the user's query and the retrieved products, provide a helpful recommendation.

IMPORTANT: Only recommend products from the list below. Do not invent or mention products not in this list.

User Query: {query}

Retrieved Products:
{context}

Provide a concise, helpful recommendation (2-3 sentences max)."""
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 200
            }
        )
        return {
            'recommendation': response.text,
            'generated': True,
            'products_used': len(products)
        }
    except Exception as e:
        # Fallback to retrieval-only
        return {
            'recommendation': f"Found {len(products)} matching products for '{query}'.",
            'generated': False,
            'error': str(e)
        }
