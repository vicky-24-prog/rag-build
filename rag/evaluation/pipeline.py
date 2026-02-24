"""
RAGAS-STYLE FULL EVALUATION PIPELINE  (Upgrade 3)

RAGEvaluationPipeline orchestrates the complete evaluation loop:

  For every (query, ground_truth_answer, relevant_doc_ids) triple:
    1. Route the query  (QueryRouter)
    2. Retrieve + generate response  (existing RAG stack)
    3. Run all four LLM-judge metrics  (EvaluationJudge)
    4. Compute retrieval metrics: Recall@k, MRR, NDCG@k
    5. Persist everything

Then aggregate into an EvaluationReport containing:
  - Overall metrics dict
  - Failure analysis: top-10 worst queries by faithfulness
  - Routing breakdown
  - Cost report
  - Latency breakdown

The pipeline is designed to plug into the EXISTING project structure:
  - Expects a retriever callable: (query, top_k) → List[str] (chunk texts)
    and a retrieval_ids callable: (query, top_k) → List[str] (doc IDs)
  - Expects a generator callable: (query, chunks) → str
  - Both can be lambda-wrapped from the existing RetrieverLayer / RAGPipeline
"""

from __future__ import annotations

import csv
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .judge import EvaluationJudge, JudgeResult
from .metrics import compute_all_retrieval_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """One labelled evaluation example."""
    query: str
    ground_truth_answer: str
    relevant_doc_ids: List[str]                # ground-truth relevant doc/product IDs
    query_id: Optional[str] = None

    def __post_init__(self):
        if self.query_id is None:
            self.query_id = str(uuid.uuid4())


@dataclass
class PerQueryResult:
    """Full result record for a single query evaluation."""
    query_id: str
    query: str
    ground_truth_answer: str
    relevant_doc_ids: List[str]
    route: str
    routing_method: str
    routing_confidence: float
    routing_latency_ms: float
    retrieved_doc_ids: List[str]
    retrieved_chunks: List[str]
    response: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    recall_at_k: Dict[int, float]              # {1: 0.0, 3: 0.5, 5: 1.0, 10: 1.0}
    ndcg_at_k: Dict[int, float]
    rr: float                                   # reciprocal rank
    retrieval_latency_ms: float
    generation_latency_ms: float
    evaluation_latency_ms: float
    total_latency_ms: float
    estimated_cost_usd: float
    parse_failures: int
    judge_model: str


@dataclass
class EvaluationReport:
    """Aggregated report for a full evaluation run."""
    experiment_id: str
    total_queries: int

    # Aggregated retrieval metrics
    mean_recall: Dict[int, float]              # k → mean recall
    mean_ndcg: Dict[int, float]
    mrr: float

    # Aggregated judge metrics
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_precision: float
    mean_context_recall: float

    # Failure analysis
    worst_faithfulness: List[PerQueryResult]   # top-10 lowest faithfulness

    # Routing breakdown
    routing_distribution: Dict[str, int]       # route → count
    routing_method_distribution: Dict[str, int]

    # Cost report
    total_api_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float

    # Latency breakdown (mean ms)
    mean_retrieval_latency_ms: float
    mean_generation_latency_ms: float
    mean_evaluation_latency_ms: float

    # All per-query results for drill-down
    per_query_results: List[PerQueryResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGEvaluationPipeline:
    """
    Full RAGAS-style evaluation pipeline.

    Parameters
    ----------
    retriever_fn     : (query: str, top_k: int) → List[str]  (chunk texts)
    retrieval_ids_fn : (query: str, top_k: int) → List[str]  (doc/product IDs)
    generator_fn     : (query: str, chunks: List[str]) → str (generated response)
    judge            : EvaluationJudge instance
    router           : QueryRouter instance (optional; skip routing if None)
    k_values         : k cutoffs for Recall@k / NDCG@k
    """

    def __init__(
        self,
        retriever_fn: Callable[[str, int], List[str]],
        retrieval_ids_fn: Callable[[str, int], List[str]],
        generator_fn: Callable[[str, List[str]], str],
        judge: EvaluationJudge,
        router=None,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> None:
        self.retriever_fn = retriever_fn
        self.retrieval_ids_fn = retrieval_ids_fn
        self.generator_fn = generator_fn
        self.judge = judge
        self.router = router
        self.k_values = list(k_values)

    # ------------------------------------------------------------------
    # Single query evaluation
    # ------------------------------------------------------------------

    def _evaluate_single(self, qa: QAPair, top_k: int) -> PerQueryResult:
        """Run full evaluation for one QAPair."""
        t_total_start = time.perf_counter()

        # --- 1. Route ---
        route_str = "RAG_RETRIEVAL"
        routing_method = "none"
        routing_confidence = 1.0
        routing_latency_ms = 0.0

        if self.router is not None:
            decision = self.router.route(qa.query)
            route_str = decision.route.value
            routing_method = decision.method
            routing_confidence = decision.confidence
            routing_latency_ms = decision.latency_ms

        # --- 2. Retrieve ---
        t_ret = time.perf_counter()
        # For DIRECT_LLM route, skip retrieval to save cost
        if route_str == "DIRECT_LLM":
            chunks: List[str] = []
            retrieved_ids: List[str] = []
        else:
            chunks = self.retriever_fn(qa.query, top_k)
            retrieved_ids = self.retrieval_ids_fn(qa.query, top_k)
        retrieval_latency_ms = (time.perf_counter() - t_ret) * 1000

        # --- 3. Generate ---
        t_gen = time.perf_counter()
        response = self.generator_fn(qa.query, chunks)
        generation_latency_ms = (time.perf_counter() - t_gen) * 1000

        # --- 4. LLM-judge metrics ---
        t_eval = time.perf_counter()
        judge_result: JudgeResult = self.judge.evaluate(
            query=qa.query,
            response=response,
            chunks=chunks,
            ground_truth=qa.ground_truth_answer,
            query_id=qa.query_id,
        )
        evaluation_latency_ms = (time.perf_counter() - t_eval) * 1000

        # --- 5. Retrieval metrics ---
        relevant_set: Set[str] = set(qa.relevant_doc_ids)
        recall_at_k: Dict[int, float] = {}
        ndcg_at_k: Dict[int, float] = {}

        from .metrics import compute_recall_at_k, compute_ndcg_at_k, compute_rr
        for k in self.k_values:
            recall_at_k[k] = compute_recall_at_k(retrieved_ids, relevant_set, k)
            ndcg_at_k[k] = compute_ndcg_at_k(retrieved_ids, relevant_set, k)
        rr = compute_rr(retrieved_ids, relevant_set)

        total_latency_ms = (time.perf_counter() - t_total_start) * 1000

        return PerQueryResult(
            query_id=qa.query_id,
            query=qa.query,
            ground_truth_answer=qa.ground_truth_answer,
            relevant_doc_ids=qa.relevant_doc_ids,
            route=route_str,
            routing_method=routing_method,
            routing_confidence=routing_confidence,
            routing_latency_ms=routing_latency_ms,
            retrieved_doc_ids=retrieved_ids,
            retrieved_chunks=chunks,
            response=response,
            faithfulness=judge_result.faithfulness,
            answer_relevance=judge_result.answer_relevance,
            context_precision=judge_result.context_precision,
            context_recall=judge_result.context_recall or 0.0,
            recall_at_k=recall_at_k,
            ndcg_at_k=ndcg_at_k,
            rr=rr,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            evaluation_latency_ms=evaluation_latency_ms,
            total_latency_ms=total_latency_ms,
            estimated_cost_usd=judge_result.estimated_cost_usd,
            parse_failures=judge_result.parse_failures,
            judge_model=judge_result.judge_model,
        )

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_dataset(
        self,
        qa_pairs: List[QAPair],
        top_k: int = 5,
        experiment_id: Optional[str] = None,
        verbose: bool = True,
    ) -> EvaluationReport:
        """
        Evaluate the full dataset and return an aggregated EvaluationReport.

        Parameters
        ----------
        qa_pairs      : labelled evaluation examples.
        top_k         : number of chunks to retrieve per query.
        experiment_id : optional identifier; auto-generated if None.
        verbose       : print progress to stdout.

        Returns
        -------
        EvaluationReport
        """
        experiment_id = experiment_id or str(uuid.uuid4())[:8]
        results: List[PerQueryResult] = []

        for i, qa in enumerate(qa_pairs):
            if verbose:
                print(f"[{i+1}/{len(qa_pairs)}] Evaluating: {qa.query[:80]}")
            try:
                r = self._evaluate_single(qa, top_k=top_k)
                results.append(r)
            except Exception as exc:
                logger.error(f"[Pipeline] Failed on query '{qa.query[:60]}': {exc}")

        return self._build_report(results, experiment_id)

    # ------------------------------------------------------------------
    # Report aggregation
    # ------------------------------------------------------------------

    def _build_report(
        self, results: List[PerQueryResult], experiment_id: str
    ) -> EvaluationReport:
        n = len(results)
        if n == 0:
            raise ValueError("No results to aggregate.")

        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        mean_recall = {
            k: _mean([r.recall_at_k.get(k, 0.0) for r in results])
            for k in self.k_values
        }
        mean_ndcg = {
            k: _mean([r.ndcg_at_k.get(k, 0.0) for r in results])
            for k in self.k_values
        }
        mrr = _mean([r.rr for r in results])

        mean_faithfulness = _mean([r.faithfulness for r in results])
        mean_answer_relevance = _mean([r.answer_relevance for r in results])
        mean_context_precision = _mean([r.context_precision for r in results])
        mean_context_recall = _mean([r.context_recall for r in results])

        # Failure analysis: lowest faithfulness
        worst = sorted(results, key=lambda r: r.faithfulness)[:10]

        # Routing distribution
        routing_dist: Dict[str, int] = {}
        method_dist: Dict[str, int] = {}
        for r in results:
            routing_dist[r.route] = routing_dist.get(r.route, 0) + 1
            method_dist[r.routing_method] = method_dist.get(r.routing_method, 0) + 1

        # Cost / token totals
        total_cost = sum(r.estimated_cost_usd for r in results)

        judge_results = self.judge.load_results(limit=n * 5)
        total_input_tokens = sum(jr.get("input_tokens", 0) for jr in judge_results)
        total_output_tokens = sum(jr.get("output_tokens", 0) for jr in judge_results)

        # Latency means
        mean_ret_ms = _mean([r.retrieval_latency_ms for r in results])
        mean_gen_ms = _mean([r.generation_latency_ms for r in results])
        mean_eval_ms = _mean([r.evaluation_latency_ms for r in results])

        return EvaluationReport(
            experiment_id=experiment_id,
            total_queries=n,
            mean_recall=mean_recall,
            mean_ndcg=mean_ndcg,
            mrr=mrr,
            mean_faithfulness=mean_faithfulness,
            mean_answer_relevance=mean_answer_relevance,
            mean_context_precision=mean_context_precision,
            mean_context_recall=mean_context_recall,
            worst_faithfulness=worst,
            routing_distribution=routing_dist,
            routing_method_distribution=method_dist,
            total_api_calls=n,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost,
            mean_retrieval_latency_ms=mean_ret_ms,
            mean_generation_latency_ms=mean_gen_ms,
            mean_evaluation_latency_ms=mean_eval_ms,
            per_query_results=results,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_report(self, report: EvaluationReport, output_dir: str) -> None:
        """
        Persist a report to disk:
          {output_dir}/results.json       – full aggregated report
          {output_dir}/per_query_results.csv – one row per query
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # results.json  (exclude per_query_results to keep it small)
        summary = {k: v for k, v in asdict(report).items()
                   if k not in ("per_query_results", "worst_faithfulness")}
        summary["worst_faithfulness"] = [
            {k2: v2 for k2, v2 in asdict(r).items()
             if k2 not in ("retrieved_chunks",)}
            for r in report.worst_faithfulness
        ]
        (out / "results.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        # per_query_results.csv
        if report.per_query_results:
            fieldnames = [
                "query_id", "query", "route", "routing_method",
                "routing_confidence", "faithfulness", "answer_relevance",
                "context_precision", "context_recall", "rr",
                "retrieval_latency_ms", "generation_latency_ms",
                "evaluation_latency_ms", "estimated_cost_usd", "parse_failures",
            ] + [f"recall@{k}" for k in self.k_values] + \
              [f"ndcg@{k}" for k in self.k_values]

            csv_path = out / "per_query_results.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in report.per_query_results:
                    row = asdict(r)
                    for k in self.k_values:
                        row[f"recall@{k}"] = row["recall_at_k"].get(k, 0.0)
                        row[f"ndcg@{k}"] = row["ndcg_at_k"].get(k, 0.0)
                    writer.writerow(row)

        logger.info(f"[Pipeline] Report saved to {out}")

    def print_report(self, report: EvaluationReport) -> None:
        """Pretty-print the evaluation report to stdout."""
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  EVALUATION REPORT  |  Experiment: {report.experiment_id}")
        print(sep)
        print(f"  Total queries evaluated : {report.total_queries}")
        print()

        print("  RETRIEVAL METRICS")
        for k in self.k_values:
            r = report.mean_recall.get(k, 0)
            n = report.mean_ndcg.get(k, 0)
            print(f"    Recall@{k:<3} = {r:.4f}   NDCG@{k:<3} = {n:.4f}")
        print(f"    MRR       = {report.mrr:.4f}")
        print()

        print("  LLM-JUDGE METRICS")
        print(f"    Faithfulness       = {report.mean_faithfulness:.4f}")
        print(f"    Answer Relevance   = {report.mean_answer_relevance:.4f}")
        print(f"    Context Precision  = {report.mean_context_precision:.4f}")
        print(f"    Context Recall     = {report.mean_context_recall:.4f}")
        print()

        print("  ROUTING DISTRIBUTION")
        for route, cnt in report.routing_distribution.items():
            pct = cnt / report.total_queries * 100
            print(f"    {route:<20} {cnt:>4} ({pct:.1f}%)")
        print()

        print("  COST REPORT")
        print(f"    Total API calls    = {report.total_api_calls}")
        print(f"    Input tokens       = {report.total_input_tokens:,}")
        print(f"    Output tokens      = {report.total_output_tokens:,}")
        print(f"    Estimated USD cost = ${report.total_cost_usd:.4f}")
        print()

        print("  LATENCY (mean ms)")
        print(f"    Retrieval    = {report.mean_retrieval_latency_ms:.1f} ms")
        print(f"    Generation   = {report.mean_generation_latency_ms:.1f} ms")
        print(f"    Evaluation   = {report.mean_evaluation_latency_ms:.1f} ms")
        print()

        print("  TOP-10 WORST QUERIES (by faithfulness)")
        for i, r in enumerate(report.worst_faithfulness, 1):
            print(f"    {i:2}. [{r.faithfulness:.2f}] {r.query[:70]}")
        print(sep)
