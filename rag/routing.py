"""
QUERY ROUTING SYSTEM  (Upgrade 2)

QueryRouter classifies every incoming query into one of three routes before
the RAG pipeline decides what to do with it:

  DIRECT_LLM    – the LLM can answer from parametric knowledge alone
  RAG_RETRIEVAL – the answer requires document retrieval
  HYBRID        – needs both retrieval AND LLM reasoning

Two classification strategies are implemented and can be combined:

  A) LLM-based routing  (high accuracy, ~200 ms, costs ~0.0001 USD / call)
     Uses Gemini-Flash (cheap, fast) to classify the query with a
     chain-of-thought-lite prompt.

  B) Embedding-based routing  (fast, ~5 ms, zero API cost)
     Pre-embeds 20 example queries per category, stores the centroid of
     each category, then routes by highest cosine-similarity to centroids.
     Falls back to LLM routing when max cosine similarity < CONFIDENCE_THRESHOLD.

Decision logic:
  1. Compute cosine similarity to centroids (Embedding approach).
  2. If best similarity >= CONFIDENCE_THRESHOLD → use embedding route.
  3. Otherwise → call LLM router.

All routing decisions are logged for accuracy auditing.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.75        # min cosine similarity to trust embedding routing
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Route(str, Enum):
    DIRECT_LLM    = "DIRECT_LLM"
    RAG_RETRIEVAL = "RAG_RETRIEVAL"
    HYBRID        = "HYBRID"


@dataclass
class RoutingDecision:
    route: Route
    confidence: float           # 0–1
    method: str                 # "embedding" | "llm"
    latency_ms: float
    query: str
    decision_id: str


# ---------------------------------------------------------------------------
# Example queries used to build embedding centroids
# ---------------------------------------------------------------------------

ROUTE_EXAMPLES: Dict[Route, List[str]] = {
    Route.DIRECT_LLM: [
        "What is machine learning?",
        "Explain gradient descent.",
        "What are the main types of neural networks?",
        "Define precision and recall.",
        "What is the difference between supervised and unsupervised learning?",
        "What is natural language processing?",
        "What is the capital of France?",
        "How does TCP/IP work?",
        "What is a transformer architecture?",
        "What is the Big-O notation?",
        "What does API stand for?",
        "What is a hash table?",
        "Explain the CAP theorem.",
        "What is REST?",
        "Describe the travelling salesman problem.",
        "What is a decision tree?",
        "What is logistic regression?",
        "What does GPU stand for?",
        "What is recursion?",
        "What is overfitting?",
    ],
    Route.RAG_RETRIEVAL: [
        "What is our return policy?",
        "What products are available under 500 rupees?",
        "Show me running shoes with rating above 4.5.",
        "What is the price of the Canon DSLR in stock?",
        "Which brands do we carry in the laptop category?",
        "What is the warranty on the Sony headphones?",
        "List all products in the electronics category.",
        "What sizes are available for the Nike sneakers?",
        "Do we have wireless keyboards in inventory?",
        "What are the top-rated products this month?",
        "How many blue shirts are in stock?",
        "What is the product description for item P0042?",
        "Show me all discounted items.",
        "What is the SKU for the Adidas running shoes?",
        "Do we stock Samsung Galaxy phones?",
        "What are the shipping options for international orders?",
        "What is the refund process for damaged goods?",
        "Show me the newest arrivals in the furniture category.",
        "What is the price range for gaming laptops in catalogue?",
        "Do you have noise-cancelling headphones under 3000?",
    ],
    Route.HYBRID: [
        "How does our return policy compare to Amazon's?",
        "Is our laptop warranty better than the industry standard?",
        "Are our shoe prices competitive with market rates?",
        "How do our headphone specs compare to best-in-class?",
        "Why would someone prefer our camera over a Sony Alpha?",
        "Is the CPU in our gaming laptop good for deep learning?",
        "How does our product quality compare to international brands?",
        "What makes our wireless earbuds unique versus AirPods?",
        "Is the RAM in our budget laptops sufficient for coding?",
        "How do our running shoes compare to Nike Pegasus?",
        "Would a photographer prefer our DSLR or a mirrorless camera?",
        "Is our 4K TV better value than industry alternatives?",
        "How does our furniture quality compare to IKEA standards?",
        "Is our phone cover compatible with the latest iPhone?",
        "Which of our products would an ML engineer recommend?",
        "How does our pricing strategy compare to competitors?",
        "Is our GPU good enough for stable diffusion on our laptops?",
        "How do our keyboard switches compare to mechanical standards?",
        "What is the trade-off between our two smartwatch models?",
        "Should a student buy our budget laptop or a refurbished MacBook?",
    ],
}


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Classifies queries into DIRECT_LLM / RAG_RETRIEVAL / HYBRID.

    Parameters
    ----------
    llm_model    : Gemini-Flash model for LLM-based routing.
    embedder     : A sentence-transformers model name (or None to skip
                   embedding routing and always use LLM routing).
    api_key_env  : Env var holding the Gemini API key.
    log_path     : Optional path to a JSONL file for decision logging.
    """

    def __init__(
        self,
        llm_model: str = "gemini-2.0-flash",
        embedder: Optional[str] = "all-mpnet-base-v2",
        api_key_env: str = "GEMINI_API_KEY",
        log_path: Optional[str] = "routing_log.jsonl",
    ) -> None:
        self.llm_model = llm_model
        self.api_key_env = api_key_env
        self.log_path = log_path
        self._llm = None
        self._embed_model = None
        self._centroids: Optional[Dict[Route, np.ndarray]] = None

        self._init_llm()
        if embedder:
            self._init_embedding_router(embedder)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            logger.warning(
                f"[QueryRouter] No API key in '{self.api_key_env}'. "
                "LLM routing will fail at runtime."
            )
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._llm = genai.GenerativeModel(model_name=self.llm_model)
            logger.info(f"[QueryRouter] LLM router ready: {self.llm_model}")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

    def _init_embedding_router(self, model_name: str) -> None:
        """Load sentence-transformer and pre-compute per-route centroids."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[QueryRouter] Loading embedding model: {model_name}")
            self._embed_model = SentenceTransformer(model_name)
            self._centroids = self._build_centroids()
            logger.info("[QueryRouter] Embedding centroids built for all routes.")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Embedding routing disabled. Falling back to LLM routing only."
            )
            self._embed_model = None

    def _build_centroids(self) -> Dict[Route, np.ndarray]:
        """Embed all example queries and compute per-route mean vectors."""
        centroids: Dict[Route, np.ndarray] = {}
        for route, examples in ROUTE_EXAMPLES.items():
            embeddings = self._embed_model.encode(examples, normalize_embeddings=True)
            centroid = embeddings.mean(axis=0)
            # Renormalise centroid so cosine sim stays in [0,1]
            norm = np.linalg.norm(centroid)
            centroids[route] = centroid / norm if norm > 0 else centroid
        return centroids

    # ------------------------------------------------------------------
    # Approach A – LLM-based routing
    # ------------------------------------------------------------------

    def _route_via_llm(self, query: str) -> Tuple[Route, float]:
        """
        Call Gemini-Flash to classify the query.
        Returns (Route, confidence).
        """
        prompt = f"""You are a query routing system for a RAG (Retrieval-Augmented Generation) pipeline.

Classify this query into exactly ONE of the three categories below:

DIRECT_LLM    - answerable from general world knowledge alone
                (e.g., "What is machine learning?", "Define precision")
RAG_RETRIEVAL - requires looking up specific documents/catalogue/policy data
                (e.g., "What is our return policy?", "List products under 500 INR")
HYBRID        - requires BOTH document retrieval AND general reasoning/comparison
                (e.g., "Is our laptop better than a MacBook?", "How do we compare to Amazon?")

Query: {query}

Return ONLY a valid JSON object with no extra text:
{{"route": "DIRECT_LLM" | "RAG_RETRIEVAL" | "HYBRID", "confidence": <float 0.0-1.0>}}"""

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES):
            wait = 2 ** attempt
            try:
                resp = self._llm.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                    },
                )
                text = resp.text.strip()
                # Strip markdown fences
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
                data = json.loads(text)
                route_str = data.get("route", "RAG_RETRIEVAL")
                confidence = float(data.get("confidence", 0.5))
                route = Route(route_str)
                return route, min(max(confidence, 0.0), 1.0)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[QueryRouter] LLM routing attempt {attempt+1}/{MAX_RETRIES} "
                    f"failed: {exc}. Retrying in {wait}s…"
                )
                time.sleep(wait)

        logger.error(
            f"[QueryRouter] All LLM routing attempts failed. "
            f"Defaulting to RAG_RETRIEVAL. Error: {last_exc}"
        )
        return Route.RAG_RETRIEVAL, 0.0

    # ------------------------------------------------------------------
    # Approach B – Embedding-based routing
    # ------------------------------------------------------------------

    def _route_via_embeddings(self, query: str) -> Tuple[Optional[Route], float]:
        """
        Returns (Route, cosine_similarity) or (None, 0.0) if unavailable.
        """
        if self._embed_model is None or self._centroids is None:
            return None, 0.0

        q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0]

        best_route: Optional[Route] = None
        best_sim: float = -1.0
        for route, centroid in self._centroids.items():
            sim = float(np.dot(q_emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_route = route

        return best_route, best_sim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, query: str) -> RoutingDecision:
        """
        Classify a query and return a RoutingDecision.

        Strategy:
          1. Try embedding-based routing.
          2. If confidence >= CONFIDENCE_THRESHOLD → use embedding result.
          3. Otherwise call LLM router.

        Parameters
        ----------
        query : The raw user query string.

        Returns
        -------
        RoutingDecision dataclass.
        """
        t0 = time.perf_counter()
        decision_id = str(uuid.uuid4())

        # Step 1: embedding routing
        emb_route, emb_sim = self._route_via_embeddings(query)

        if emb_route is not None and emb_sim >= CONFIDENCE_THRESHOLD:
            route = emb_route
            confidence = emb_sim
            method = "embedding"
        else:
            # Step 2: LLM routing
            if self._llm is None:
                # No LLM available, use embedding even if below threshold
                route = emb_route or Route.RAG_RETRIEVAL
                confidence = emb_sim
                method = "embedding_fallback"
            else:
                route, confidence = self._route_via_llm(query)
                method = "llm"

        latency_ms = (time.perf_counter() - t0) * 1000

        decision = RoutingDecision(
            route=route,
            confidence=confidence,
            method=method,
            latency_ms=latency_ms,
            query=query,
            decision_id=decision_id,
        )
        self._log_decision(decision)
        return decision

    def evaluate_routing_accuracy(
        self,
        labeled_queries: List[Dict],
    ) -> Dict:
        """
        Measure routing accuracy on a labeled test set.

        Parameters
        ----------
        labeled_queries : list of {"query": str, "expected_route": str}

        Returns
        -------
        Dict with accuracy, per-route precision/recall, and confusion matrix.
        """
        correct = 0
        total = len(labeled_queries)
        confusion: Dict[str, Dict[str, int]] = {
            r.value: {r2.value: 0 for r2 in Route} for r in Route
        }

        for item in labeled_queries:
            q = item["query"]
            expected = Route(item["expected_route"])
            decision = self.route(q)
            predicted = decision.route

            if predicted == expected:
                correct += 1
            confusion[expected.value][predicted.value] += 1

        accuracy = correct / total if total > 0 else 0.0

        # Per-route precision & recall
        per_route: Dict[str, Dict[str, float]] = {}
        for r in Route:
            tp = confusion[r.value][r.value]
            fp = sum(confusion[other.value][r.value] for other in Route if other != r)
            fn = sum(confusion[r.value][other.value] for other in Route if other != r)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            per_route[r.value] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(confusion[r.value].values()),
            }

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "per_route": per_route,
            "confusion_matrix": confusion,
        }

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_decision(self, d: RoutingDecision) -> None:
        """Append routing decision to JSONL log file."""
        if not self.log_path:
            return
        record = {
            "decision_id": d.decision_id,
            "query": d.query,
            "route": d.route.value,
            "confidence": d.confidence,
            "method": d.method,
            "latency_ms": d.latency_ms,
            "timestamp": time.time(),
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"[QueryRouter] Log write failed: {exc}")

    def load_log(self) -> List[Dict]:
        """Read all decisions from the JSONL routing log."""
        if not self.log_path:
            return []
        records = []
        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"[QueryRouter] Log read failed: {exc}")
        return records
