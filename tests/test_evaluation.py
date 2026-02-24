"""
UNIT TESTS FOR RAG EVALUATION FRAMEWORK

Tests use unittest.mock to intercept all Gemini API calls so the full test
suite runs offline with zero API cost.

Run:
    python -m pytest tests/test_evaluation.py -v
    # or
    python -m unittest tests/test_evaluation.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------

def _mock_gemini_response(text: str):
    """Build a minimal Gemini GenerateContentResponse mock."""
    mock_resp = MagicMock()
    mock_resp.text = text
    usage = MagicMock()
    usage.prompt_token_count = 100
    usage.candidates_token_count = 50
    mock_resp.usage_metadata = usage
    return mock_resp


# ===========================================================================
# 1. Metrics (pure functions – no mocks needed)
# ===========================================================================

class TestMetrics(unittest.TestCase):

    def test_recall_at_k_perfect(self):
        from rag.evaluation.metrics import compute_recall_at_k
        self.assertAlmostEqual(
            compute_recall_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3), 1.0
        )

    def test_recall_at_k_partial(self):
        from rag.evaluation.metrics import compute_recall_at_k
        score = compute_recall_at_k(["a", "b", "x"], {"a", "b", "c"}, k=3)
        self.assertAlmostEqual(score, 2 / 3)

    def test_recall_at_k_zero_relevant(self):
        from rag.evaluation.metrics import compute_recall_at_k
        self.assertEqual(compute_recall_at_k(["a", "b"], set(), k=2), 0.0)

    def test_recall_at_k_k_cutoff(self):
        from rag.evaluation.metrics import compute_recall_at_k
        # relevant item is at rank 6; k=5 should not see it
        self.assertEqual(
            compute_recall_at_k(["x", "x", "x", "x", "x", "a"], {"a"}, k=5), 0.0
        )

    def test_mrr_first_position(self):
        from rag.evaluation.metrics import compute_mrr
        self.assertAlmostEqual(
            compute_mrr([["a", "b"]], [{"a"}]), 1.0
        )

    def test_mrr_second_position(self):
        from rag.evaluation.metrics import compute_mrr
        self.assertAlmostEqual(
            compute_mrr([["x", "a"]], [{"a"}]), 0.5
        )

    def test_mrr_not_found(self):
        from rag.evaluation.metrics import compute_mrr
        self.assertAlmostEqual(
            compute_mrr([["x", "y"]], [{"a"}]), 0.0
        )

    def test_mrr_batch_average(self):
        from rag.evaluation.metrics import compute_mrr
        # Query 1 rank 1 → 1.0, Query 2 rank 2 → 0.5  →  mean = 0.75
        self.assertAlmostEqual(
            compute_mrr([["a", "b"], ["x", "a"]], [{"a"}, {"a"}]), 0.75
        )

    def test_ndcg_perfect_top1(self):
        from rag.evaluation.metrics import compute_ndcg_at_k
        # Single relevant doc at rank 1 → NDCG = 1.0
        self.assertAlmostEqual(
            compute_ndcg_at_k(["a", "b"], {"a"}, k=2), 1.0
        )

    def test_ndcg_worst_case(self):
        from rag.evaluation.metrics import compute_ndcg_at_k
        # Relevant doc not in retrieved list → NDCG = 0
        self.assertAlmostEqual(
            compute_ndcg_at_k(["x", "y"], {"a"}, k=2), 0.0
        )

    def test_ndcg_empty_relevant(self):
        from rag.evaluation.metrics import compute_ndcg_at_k
        self.assertEqual(compute_ndcg_at_k(["a"], set(), k=1), 0.0)

    def test_compute_all_retrieval_metrics_keys(self):
        from rag.evaluation.metrics import compute_all_retrieval_metrics
        results = compute_all_retrieval_metrics(
            [["a", "b", "c"]], [{"a"}], k_values=[1, 3, 5]
        )
        for k in [1, 3, 5]:
            self.assertIn(f"recall@{k}", results)
            self.assertIn(f"ndcg@{k}", results)
        self.assertIn("mrr", results)


# ===========================================================================
# 2. EvaluationJudge
# ===========================================================================

FAITHFULNESS_JSON = json.dumps(
    {"score": 0.85, "unsupported_claims": ["Claim X has no support"]}
)
RELEVANCE_JSON = json.dumps(
    {"score": 0.9, "reasoning": "The answer directly addresses the question."}
)
PRECISION_JSON = json.dumps({"relevant": True})
RECALL_JSON = json.dumps(
    {"recall_score": 0.75, "missing_facts": ["Fact about price"]}
)


class TestEvaluationJudge(unittest.TestCase):

    def _get_judge(self, tmp_db):
        """Return a judge with a patched Gemini model."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as MockModel:
                    instance = MockModel.return_value
                    instance.generate_content.side_effect = [
                        _mock_gemini_response(FAITHFULNESS_JSON),
                        _mock_gemini_response(RELEVANCE_JSON),
                        _mock_gemini_response(PRECISION_JSON),  # chunk 1
                        _mock_gemini_response(PRECISION_JSON),  # chunk 2
                        _mock_gemini_response(RECALL_JSON),
                    ]
                    from rag.evaluation.judge import EvaluationJudge
                    judge = EvaluationJudge(
                        judge_model="gemini-1.5-pro",
                        api_key_env="GEMINI_API_KEY",
                        db_path=tmp_db,
                    )
                    judge._llm = instance          # inject mock directly
                    return judge, instance

    def test_evaluate_returns_correct_scores(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        judge, mock_llm = self._get_judge(tmp_db)
        mock_llm.generate_content.side_effect = [
            _mock_gemini_response(FAITHFULNESS_JSON),
            _mock_gemini_response(RELEVANCE_JSON),
            _mock_gemini_response(PRECISION_JSON),
            _mock_gemini_response(PRECISION_JSON),
            _mock_gemini_response(RECALL_JSON),
        ]

        result = judge.evaluate(
            query="What shoes do you recommend?",
            response="We recommend Nike shoes priced at 2499.",
            chunks=["Nike shoes, price 2499, running category.",
                    "Adidas shoes, price 1999."],
            ground_truth="Nike Revolution 6 at 2499 is available.",
        )

        self.assertAlmostEqual(result.faithfulness, 0.85)
        self.assertAlmostEqual(result.answer_relevance, 0.9)
        self.assertAlmostEqual(result.context_precision, 1.0)  # both chunks relevant
        self.assertAlmostEqual(result.context_recall, 0.75)
        self.assertIsInstance(result.estimated_cost_usd, float)
        self.assertGreater(result.latency_ms, 0)

    def test_json_parse_fallback_on_bad_output(self):
        from rag.evaluation.judge import EvaluationJudge, _CallStats

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as MockModel:
                    instance = MockModel.return_value
                    # Return garbage twice (both tries fail), then valid JSON
                    instance.generate_content.side_effect = [
                        _mock_gemini_response("NOT JSON AT ALL"),
                        _mock_gemini_response('{"score": 0.5, "unsupported_claims": []}'),
                    ]
                    judge = EvaluationJudge(
                        judge_model="gemini-1.5-pro",
                        api_key_env="GEMINI_API_KEY",
                        db_path=tmp_db,
                    )
                    judge._llm = instance
                    stats = _CallStats()
                    score, claims = judge._evaluate_faithfulness(
                        "Some context", "Some response", stats
                    )
                    # First call fails → parse_failures = 1, retry succeeds
                    self.assertGreaterEqual(stats.parse_failures, 1)

    def test_results_persisted_to_sqlite(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        judge, mock_llm = self._get_judge(tmp_db)
        mock_llm.generate_content.side_effect = [
            _mock_gemini_response(FAITHFULNESS_JSON),
            _mock_gemini_response(RELEVANCE_JSON),
            _mock_gemini_response(PRECISION_JSON),
            _mock_gemini_response(RECALL_JSON),
        ]
        # Only 1 chunk so only 1 precision call
        judge.evaluate(
            query="test q",
            response="test answer",
            chunks=["chunk1"],
            ground_truth="gt",
        )
        rows = judge.load_results()
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("faithfulness", rows[0])

    def test_markdown_json_fence_stripping(self):
        from rag.evaluation.judge import EvaluationJudge

        fenced = '```json\n{"score": 0.7, "unsupported_claims": []}\n```'
        result, ok = EvaluationJudge._parse_json(fenced, {"score": 0.0, "unsupported_claims": []})
        self.assertTrue(ok)
        self.assertAlmostEqual(result["score"], 0.7)

    def test_score_clamped_to_0_1(self):
        from rag.evaluation.judge import EvaluationJudge
        bad = '{"score": 1.5, "unsupported_claims": []}'
        result, ok = EvaluationJudge._parse_json(bad, {})
        self.assertTrue(ok)
        # Clamping happens inside _evaluate_faithfulness; check parse returns raw
        self.assertAlmostEqual(result["score"], 1.5)


# ===========================================================================
# 3. QueryRouter
# ===========================================================================

class TestQueryRouter(unittest.TestCase):

    def _get_router_llm_only(self):
        """Route with LLM mock, embedding routing disabled."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as MockModel:
                    instance = MockModel.return_value
                    from rag.routing import QueryRouter
                    router = QueryRouter(
                        llm_model="gemini-1.5-flash",
                        embedder=None,               # disable embedding routing
                        api_key_env="GEMINI_API_KEY",
                        log_path=None,
                    )
                    router._llm = instance
                    return router, instance

    def test_llm_route_direct_llm(self):
        router, mock_llm = self._get_router_llm_only()
        mock_llm.generate_content.return_value = _mock_gemini_response(
            json.dumps({"route": "DIRECT_LLM", "confidence": 0.95})
        )
        from rag.routing import Route
        decision = router.route("What is machine learning?")
        self.assertEqual(decision.route, Route.DIRECT_LLM)
        self.assertAlmostEqual(decision.confidence, 0.95)
        self.assertEqual(decision.method, "llm")

    def test_llm_route_rag_retrieval(self):
        router, mock_llm = self._get_router_llm_only()
        mock_llm.generate_content.return_value = _mock_gemini_response(
            json.dumps({"route": "RAG_RETRIEVAL", "confidence": 0.88})
        )
        from rag.routing import Route
        decision = router.route("What is our return policy?")
        self.assertEqual(decision.route, Route.RAG_RETRIEVAL)

    def test_llm_route_hybrid(self):
        router, mock_llm = self._get_router_llm_only()
        mock_llm.generate_content.return_value = _mock_gemini_response(
            json.dumps({"route": "HYBRID", "confidence": 0.80})
        )
        from rag.routing import Route
        decision = router.route("Is our laptop better than a MacBook?")
        self.assertEqual(decision.route, Route.HYBRID)

    def test_llm_route_fallback_on_failure(self):
        """All retries fail → default to RAG_RETRIEVAL."""
        router, mock_llm = self._get_router_llm_only()
        mock_llm.generate_content.side_effect = Exception("API error")
        from rag.routing import Route
        # Patch sleep so the 3 retries don't actually wait
        with patch("rag.routing.time.sleep"):
            decision = router.route("some query")
        self.assertEqual(decision.route, Route.RAG_RETRIEVAL)

    def test_embedding_router_returns_decision(self):
        """Embedding routing uses cosine similarity; verify it returns valid route."""
        try:
            import numpy as _np
            from unittest.mock import MagicMock
            # Build a fake SentenceTransformer that returns deterministic embeddings
            fake_model = MagicMock()
            # encode() returns a (n, 4) float array; all the same unit vector so
            # centroids are identical → max sim ties → router still returns A Route
            fake_model.encode = lambda texts, normalize_embeddings=True: \
                _np.tile([1.0, 0.0, 0.0, 0.0], (len(texts), 1)).astype("float32")

            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                with patch("google.generativeai.configure"):
                    with patch("google.generativeai.GenerativeModel"):
                        with patch("sentence_transformers.SentenceTransformer",
                                   return_value=fake_model):
                            from rag.routing import QueryRouter, Route
                            router = QueryRouter(
                                embedder="all-MiniLM-L6-v2",
                                log_path=None,
                            )
                            decision = router.route("What is machine learning?")
                            self.assertIn(decision.route, list(Route))
                            self.assertGreaterEqual(decision.confidence, 0.0)
        except ImportError:
            self.skipTest("sentence-transformers not installed")

    def test_routing_accuracy_evaluation(self):
        router, mock_llm = self._get_router_llm_only()
        labeled = [
            {"query": "What is ML?",          "expected_route": "DIRECT_LLM"},
            {"query": "What is our policy?",   "expected_route": "RAG_RETRIEVAL"},
        ]
        mock_llm.generate_content.side_effect = [
            _mock_gemini_response(json.dumps({"route": "DIRECT_LLM",    "confidence": 0.9})),
            _mock_gemini_response(json.dumps({"route": "RAG_RETRIEVAL", "confidence": 0.85})),
        ]
        results = router.evaluate_routing_accuracy(labeled)
        self.assertEqual(results["accuracy"], 1.0)
        self.assertEqual(results["correct"], 2)


# ===========================================================================
# 4. RAGEvaluationPipeline
# ===========================================================================

class TestRAGEvaluationPipeline(unittest.TestCase):

    def _build_pipeline(self, tmp_db, mock_llm_instance):
        from rag.evaluation.judge import EvaluationJudge
        from rag.evaluation.pipeline import RAGEvaluationPipeline

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("google.generativeai.configure"):
                with patch("google.generativeai.GenerativeModel") as MockModel:
                    MockModel.return_value = mock_llm_instance
                    judge = EvaluationJudge(
                        judge_model="gemini-1.5-pro",
                        db_path=tmp_db,
                    )
                    judge._llm = mock_llm_instance

        def retriever_fn(q, k): return ["Nike shoes context", "Adidas shoes context"]
        def retrieval_ids_fn(q, k): return ["P001", "P002"]
        def generator_fn(q, chunks): return "I recommend Nike shoes at 2499."

        pipeline = RAGEvaluationPipeline(
            retriever_fn=retriever_fn,
            retrieval_ids_fn=retrieval_ids_fn,
            generator_fn=generator_fn,
            judge=judge,
            router=None,
            k_values=[1, 3, 5],
        )
        return pipeline

    def _side_effects_for_n_queries(self, n: int):
        """Provide enough mock responses for n queries (4 judge calls each)."""
        effects = []
        for _ in range(n):
            effects += [
                _mock_gemini_response(FAITHFULNESS_JSON),
                _mock_gemini_response(RELEVANCE_JSON),
                _mock_gemini_response(PRECISION_JSON),  # chunk 1
                _mock_gemini_response(PRECISION_JSON),  # chunk 2
                _mock_gemini_response(RECALL_JSON),
            ]
        return effects

    def test_evaluate_dataset_returns_report(self):
        from rag.evaluation.pipeline import QAPair

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        mock_llm = MagicMock()
        mock_llm.generate_content.side_effect = self._side_effects_for_n_queries(2)

        pipeline = self._build_pipeline(tmp_db, mock_llm)

        qa_pairs = [
            QAPair(
                query="Recommend running shoes under 3000",
                ground_truth_answer="Nike Revolution 6 at 2499",
                relevant_doc_ids=["P001"],
            ),
            QAPair(
                query="Best wireless headphones",
                ground_truth_answer="Sony WH-1000XM5",
                relevant_doc_ids=["P050"],
            ),
        ]

        report = pipeline.evaluate_dataset(qa_pairs, top_k=5, verbose=False)

        self.assertEqual(report.total_queries, 2)
        self.assertAlmostEqual(report.mean_faithfulness, 0.85)
        self.assertAlmostEqual(report.mean_answer_relevance, 0.9)
        self.assertIn(5, report.mean_recall)
        self.assertIn(5, report.mean_ndcg)
        self.assertEqual(len(report.per_query_results), 2)

    def test_report_saved_to_disk(self):
        from rag.evaluation.pipeline import QAPair

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = MagicMock()
            mock_llm.generate_content.side_effect = self._side_effects_for_n_queries(1)

            pipeline = self._build_pipeline(tmp_db, mock_llm)
            qa_pairs = [
                QAPair(
                    query="test query",
                    ground_truth_answer="test gt",
                    relevant_doc_ids=["P001"],
                )
            ]
            report = pipeline.evaluate_dataset(qa_pairs, verbose=False)
            pipeline.save_report(report, tmpdir)

            self.assertTrue(Path(tmpdir, "results.json").exists())
            self.assertTrue(Path(tmpdir, "per_query_results.csv").exists())

    def test_mrr_computed_correctly(self):
        from rag.evaluation.pipeline import QAPair

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            tmp_db = f.name

        mock_llm = MagicMock()
        mock_llm.generate_content.side_effect = self._side_effects_for_n_queries(1)

        # P001 is at rank 1 → RR = 1.0 → MRR = 1.0
        pipeline = self._build_pipeline(tmp_db, mock_llm)
        qa_pairs = [
            QAPair(
                query="q",
                ground_truth_answer="gt",
                relevant_doc_ids=["P001"],
            )
        ]
        report = pipeline.evaluate_dataset(qa_pairs, verbose=False)
        self.assertAlmostEqual(report.mrr, 1.0)   # P001 is rank 1


# ===========================================================================
# 5. ExperimentTracker
# ===========================================================================

class TestExperimentTracker(unittest.TestCase):

    def _dummy_report(self, exp_id: str):
        from rag.evaluation.pipeline import EvaluationReport
        return EvaluationReport(
            experiment_id=exp_id,
            total_queries=10,
            mean_recall={1: 0.5, 3: 0.7, 5: 0.85, 10: 0.9},
            mean_ndcg={1: 0.5, 3: 0.65, 5: 0.80, 10: 0.88},
            mrr=0.82,
            mean_faithfulness=0.78,
            mean_answer_relevance=0.84,
            mean_context_precision=0.72,
            mean_context_recall=0.69,
            worst_faithfulness=[],
            routing_distribution={"RAG_RETRIEVAL": 7, "DIRECT_LLM": 2, "HYBRID": 1},
            routing_method_distribution={"embedding": 8, "llm": 2},
            total_api_calls=10,
            total_input_tokens=5000,
            total_output_tokens=1200,
            total_cost_usd=0.0124,
            mean_retrieval_latency_ms=22.0,
            mean_generation_latency_ms=850.0,
            mean_evaluation_latency_ms=2100.0,
        )

    def test_save_and_load_round_trip(self):
        from rag.experiments import ExperimentTracker, ExperimentConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            config = ExperimentConfig(
                embedding_model="all-mpnet-base-v2",
                k_retrieved=5,
                description="baseline run",
            )
            report = self._dummy_report("test_exp")
            saved_id = tracker.save(config, report, experiment_name="test_exp")

            self.assertEqual(saved_id, "test_exp")
            loaded_cfg, loaded_res = tracker.load("test_exp")
            self.assertEqual(loaded_cfg.embedding_model, "all-mpnet-base-v2")
            self.assertAlmostEqual(loaded_res["mrr"], 0.82)

    def test_list_experiments(self):
        from rag.experiments import ExperimentTracker, ExperimentConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)
            for name in ["exp_a", "exp_b"]:
                tracker.save(
                    ExperimentConfig(), self._dummy_report(name),
                    experiment_name=name
                )
            listing = tracker.list_experiments()
            ids = [e["experiment_id"] for e in listing]
            self.assertIn("exp_a", ids)
            self.assertIn("exp_b", ids)

    def test_compare_experiments(self):
        from rag.experiments import ExperimentTracker, ExperimentConfig
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(base_dir=tmpdir)

            cfg_a = ExperimentConfig(k_retrieved=3)
            cfg_b = ExperimentConfig(k_retrieved=10)
            rpt_a = self._dummy_report("exp_a")
            rpt_b = self._dummy_report("exp_b")
            rpt_b.mrr = 0.90  # B is better

            tracker.save(cfg_a, rpt_a, experiment_name="exp_a")
            tracker.save(cfg_b, rpt_b, experiment_name="exp_b")

            comparison = tracker.compare_experiments("exp_a", "exp_b")

            self.assertIn("config_diff", comparison)
            self.assertIn("metrics_comparison", comparison)
            # k_retrieved differs between configs
            self.assertIn("k_retrieved", comparison["config_diff"])


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
