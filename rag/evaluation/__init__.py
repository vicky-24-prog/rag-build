"""
Evaluation sub-package: LLM-as-Judge, retrieval metrics, full pipeline.
"""
from .judge import EvaluationJudge, JudgeResult
from .metrics import compute_recall_at_k, compute_mrr, compute_ndcg_at_k
from .pipeline import RAGEvaluationPipeline, QAPair, EvaluationReport
