"""
CLI EVALUATION SCRIPT

Usage examples
--------------
# Run evaluation with default settings
python evaluate.py --dataset qa_pairs.json --experiment-name "baseline"

# Specify retrieval k and judge model
python evaluate.py --dataset qa_pairs.json --k 5 --judge-model gemini-1.5-pro

# Compare two experiments
python evaluate.py --compare baseline v2_reranking

# List all saved experiments
python evaluate.py --list

# Skip routing (retrieval only)
python evaluate.py --dataset qa_pairs.json --no-routing

Dataset format (qa_pairs.json)
-------------------------------
[
  {
    "query": "Recommend running shoes under 3000 rupees",
    "ground_truth_answer": "Nike Revolution 6 at 2499 is a good option.",
    "relevant_doc_ids": ["P001", "P023"]
  },
  ...
]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Helpers to wire existing RAG stack
# ---------------------------------------------------------------------------

def _build_rag_callables(config_path: str, k: int):
    """
    Return (retriever_fn, retrieval_ids_fn, generator_fn) by lazily importing
    the existing RetrieverLayer and RAGPipeline from src/.

    If the existing stack is not importable, returns lightweight stubs so
    the evaluation framework can still run with mock data.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        import yaml
        import numpy as np
        import pandas as pd

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        from embeddings import EmbeddingLayer
        from vector_store import VectorStoreLayer
        from retriever import RetrieverLayer
        from rag_pipeline import RAGPipeline

        # Load data
        df = pd.read_csv(cfg["data"]["input_path"])

        # Build embedding + vector store
        embedder = EmbeddingLayer(config_path=config_path)
        embeddings = np.load(cfg["embeddings"]["cache_path"])
        vs = VectorStoreLayer(config_path=config_path)
        vs.build_index(embeddings)

        retriever = RetrieverLayer(
            vector_store=vs, embedder=embedder, df_products=df,
            config_path=config_path
        )
        generator = RAGPipeline(config_path=config_path)

        def retriever_fn(query: str, top_k: int) -> list:
            result = retriever.retrieve(query, top_k=top_k)
            return [
                f"{p['product_name']}: {p['description']}"
                for p in result.get("results", [])
            ]

        def retrieval_ids_fn(query: str, top_k: int) -> list:
            result = retriever.retrieve(query, top_k=top_k)
            return [p["product_id"] for p in result.get("results", [])]

        def generator_fn(query: str, chunks: list) -> str:
            mock_result = {
                "query": query,
                "decision": "ACCEPT",
                "results": [],
                "accepted": True,
            }
            gen_out = generator.generate(mock_result)
            return gen_out.get("response", "")

        logger.info("RAG stack loaded from src/.")
        return retriever_fn, retrieval_ids_fn, generator_fn

    except Exception as exc:
        logger.warning(
            f"Could not load existing RAG stack ({exc}). "
            "Using stub callables — responses will be '[STUB]'."
        )

        # Stub callables (useful for testing the evaluation framework itself)
        def retriever_fn(query: str, top_k: int) -> list:
            return [f"[Stub chunk {i} for: {query}]" for i in range(top_k)]

        def retrieval_ids_fn(query: str, top_k: int) -> list:
            return [f"P{i:03d}" for i in range(top_k)]

        def generator_fn(query: str, chunks: list) -> str:
            return f"[STUB RESPONSE for: {query}]"

        return retriever_fn, retrieval_ids_fn, generator_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Evaluation CLI – run LLM-as-Judge + retrieval metrics"
    )

    # Dataset
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Path to qa_pairs.json evaluation dataset.",
    )

    # Experiment
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        default=None,
        help="Human-readable experiment name (used as directory name).",
    )
    parser.add_argument(
        "--experiments-dir",
        type=str,
        default="experiments",
        help="Base directory for experiment artefacts.",
    )

    # Retrieval
    parser.add_argument(
        "--k", "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="k cutoffs for Recall@k / NDCG@k (default: 1 3 5 10).",
    )

    # Models
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gemini-1.5-pro",
        help="LLM model name for judging (default: gemini-1.5-pro).",
    )
    parser.add_argument(
        "--router-model",
        type=str,
        default="gemini-1.5-flash",
        help="LLM model name for routing (default: gemini-1.5-flash).",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence-transformer model for embedding routing.",
    )

    # Switches
    parser.add_argument(
        "--no-routing",
        action="store_true",
        help="Disable query routing (always use RAG retrieval).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config.yaml for existing RAG stack.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="evaluations.db",
        help="SQLite file for judge results (default: evaluations.db).",
    )

    # Utility commands
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all saved experiments and exit.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs=2,
        metavar=("EXP_A", "EXP_B"),
        help="Compare two saved experiments and exit.",
    )

    # Config knobs saved into ExperimentConfig
    parser.add_argument("--chunk-size",       type=int,   default=512)
    parser.add_argument("--chunk-overlap",    type=int,   default=64)
    parser.add_argument("--reranking",        action="store_true", default=False)
    parser.add_argument("--description",      type=str,   default="")
    parser.add_argument("--tags",             type=str,   nargs="*", default=[])

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Lazy imports (keep startup fast when using --list / --compare)
    # ------------------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).parent))
    from rag.evaluation.judge import EvaluationJudge
    from rag.evaluation.pipeline import RAGEvaluationPipeline, QAPair
    from rag.routing import QueryRouter
    from rag.experiments import ExperimentTracker, ExperimentConfig

    tracker = ExperimentTracker(base_dir=args.experiments_dir)

    # ------------------------------------------------------------------
    # --list
    # ------------------------------------------------------------------
    if args.list:
        tracker.print_experiments()
        return

    # ------------------------------------------------------------------
    # --compare
    # ------------------------------------------------------------------
    if args.compare:
        tracker.compare_experiments(args.compare[0], args.compare[1])
        return

    # ------------------------------------------------------------------
    # Normal evaluation run
    # ------------------------------------------------------------------
    if not args.dataset:
        parser.error("--dataset is required for evaluation runs.")

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    qa_pairs = [
        QAPair(
            query=item["query"],
            ground_truth_answer=item.get("ground_truth_answer", ""),
            relevant_doc_ids=item.get("relevant_doc_ids", []),
        )
        for item in raw
    ]
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from {dataset_path}")

    # --- Build callables ---
    retriever_fn, retrieval_ids_fn, generator_fn = _build_rag_callables(
        args.config, args.k
    )

    # --- Judge ---
    judge = EvaluationJudge(
        judge_model=args.judge_model,
        db_path=args.db_path,
    )

    # --- Router ---
    router = None
    if not args.no_routing:
        router = QueryRouter(
            llm_model=args.router_model,
            embedder=args.embedding_model,
        )

    # --- Pipeline ---
    pipeline = RAGEvaluationPipeline(
        retriever_fn=retriever_fn,
        retrieval_ids_fn=retrieval_ids_fn,
        generator_fn=generator_fn,
        judge=judge,
        router=router,
        k_values=args.k_values,
    )

    exp_name = args.experiment_name or f"exp_{int(time.time())}"

    logger.info(f"Starting evaluation run: '{exp_name}' on {len(qa_pairs)} queries")
    t0 = time.time()
    report = pipeline.evaluate_dataset(
        qa_pairs=qa_pairs,
        top_k=args.k,
        experiment_id=exp_name,
        verbose=True,
    )
    elapsed = time.time() - t0
    logger.info(f"Evaluation complete in {elapsed:.1f}s")

    # --- Print report ---
    pipeline.print_report(report)

    # --- Save experiment ---
    config = ExperimentConfig(
        embedding_model=args.embedding_model,
        judge_model=args.judge_model,
        generator_model=args.router_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        k_retrieved=args.k,
        k_values=args.k_values,
        reranking_enabled=args.reranking,
        query_routing_enabled=not args.no_routing,
        description=args.description,
        tags=args.tags or [],
    )
    saved_id = tracker.save(config, report, pipeline=pipeline, experiment_name=exp_name)
    logger.info(f"Experiment saved as '{saved_id}'")


if __name__ == "__main__":
    main()
