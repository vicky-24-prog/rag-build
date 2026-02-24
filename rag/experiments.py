"""
EXPERIMENT TRACKING  (Upgrade 4)

Manages versioned RAG experiment runs so different configurations can be
compared side-by-side.

Directory layout
----------------
experiments/
  {experiment_id}/
    config.json           – ExperimentConfig (hyperparameters)
    results.json          – EvaluationReport (aggregated metrics)
    per_query_results.csv – one row per query

Public API
----------
  ExperimentConfig    – dataclass capturing all tuneable knobs
  ExperimentTracker   – CRUD + comparison for experiment runs
    .save(config, report)         → experiment_id
    .load(experiment_id)          → (config, report_dict)
    .list_experiments()           → List[Dict]
    .compare_experiments(id1, id2)→ comparison table (printed + returned)
"""

from __future__ import annotations

import csv
import json
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evaluation.pipeline import EvaluationReport, RAGEvaluationPipeline


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """
    Captures every hyperparameter that can change between RAG experiments.
    Adding a new knob here automatically shows up in comparison tables.
    """
    # Model configuration
    embedding_model: str = "all-mpnet-base-v2"
    judge_model: str = "gemini-2.0-flash"
    generator_model: str = "gemini-2.0-flash"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    k_retrieved: int = 5
    similarity_threshold: float = 0.3

    # Pipeline switches
    reranking_enabled: bool = False
    query_routing_enabled: bool = True
    hybrid_search_enabled: bool = False

    # Evaluation k-values
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    # Free-form notes
    description: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """
    Save, load, list, and compare RAG experiment runs.

    Parameters
    ----------
    base_dir : root folder where experiment sub-directories are created.
    """

    def __init__(self, base_dir: str = "experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "index.json"
        self._index: Dict[str, Dict] = self._load_index()

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> Dict[str, Dict]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2, default=str), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        config: ExperimentConfig,
        report: EvaluationReport,
        pipeline: Optional[RAGEvaluationPipeline] = None,
        experiment_name: Optional[str] = None,
    ) -> str:
        """
        Persist an experiment to disk.

        Parameters
        ----------
        config          : ExperimentConfig with all hyperparameters.
        report          : EvaluationReport returned by RAGEvaluationPipeline.
        pipeline        : optional – used to trigger save_report for the CSV.
        experiment_name : human-readable label (used as experiment_id if set).

        Returns
        -------
        experiment_id (str)
        """
        exp_id = experiment_name or report.experiment_id or str(uuid.uuid4())[:8]
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # config.json
        (exp_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2), encoding="utf-8"
        )

        # results.json + per_query_results.csv via pipeline helper
        if pipeline is not None:
            pipeline.save_report(report, str(exp_dir))
        else:
            # Minimal results.json without CSV
            summary = {
                k: v for k, v in asdict(report).items()
                if k not in ("per_query_results", "worst_faithfulness")
            }
            (exp_dir / "results.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )

        # Update index
        self._index[exp_id] = {
            "experiment_id": exp_id,
            "timestamp": time.time(),
            "description": config.description,
            "tags": config.tags,
            "embedding_model": config.embedding_model,
            "k_retrieved": config.k_retrieved,
            "mrr": report.mrr,
            "mean_faithfulness": report.mean_faithfulness,
            "mean_recall_at_5": report.mean_recall.get(5, 0.0),
            "total_queries": report.total_queries,
            "total_cost_usd": report.total_cost_usd,
        }
        self._save_index()

        print(f"[ExperimentTracker] Saved experiment '{exp_id}' → {exp_dir}")
        return exp_id

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, experiment_id: str) -> Tuple[ExperimentConfig, Dict]:
        """
        Load a saved experiment.

        Returns
        -------
        (ExperimentConfig, results_dict)
        """
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            raise FileNotFoundError(
                f"Experiment '{experiment_id}' not found in {self.base_dir}"
            )

        config_path = exp_dir / "config.json"
        results_path = exp_dir / "results.json"

        config = ExperimentConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        results = json.loads(results_path.read_text(encoding="utf-8"))
        return config, results

    def load_per_query_csv(self, experiment_id: str) -> List[Dict]:
        """Load per_query_results.csv as a list of dicts."""
        csv_path = self.base_dir / experiment_id / "per_query_results.csv"
        if not csv_path.exists():
            return []
        with open(csv_path, encoding="utf-8") as f:
            return list(csv.DictReader(f))

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Dict]:
        """Return all indexed experiments sorted by timestamp (newest first)."""
        return sorted(
            self._index.values(),
            key=lambda x: x.get("timestamp", 0),
            reverse=True,
        )

    def print_experiments(self) -> None:
        """Pretty-print the experiment index."""
        experiments = self.list_experiments()
        if not experiments:
            print("[ExperimentTracker] No experiments found.")
            return

        col = "{:<20} {:>6} {:>8} {:>8} {:>10} {:>12}"
        header = col.format(
            "Experiment ID", "Qs", "MRR", "Faith.", "Recall@5", "Cost USD"
        )
        sep = "-" * len(header)
        print(f"\n{sep}\n{header}\n{sep}")
        for exp in experiments:
            print(col.format(
                exp["experiment_id"][:20],
                exp.get("total_queries", "?"),
                f"{exp.get('mrr', 0):.4f}",
                f"{exp.get('mean_faithfulness', 0):.4f}",
                f"{exp.get('mean_recall_at_5', 0):.4f}",
                f"${exp.get('total_cost_usd', 0):.4f}",
            ))
        print(sep)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def compare_experiments(
        self, exp_id_1: str, exp_id_2: str
    ) -> Dict[str, Any]:
        """
        Generate a side-by-side comparison table for two experiments.

        Returns
        -------
        Dict with keys "config_diff", "metrics_comparison", and prints table.
        """
        cfg1, res1 = self.load(exp_id_1)
        cfg2, res2 = self.load(exp_id_2)

        # --- Config diff ---
        d1 = cfg1.to_dict()
        d2 = cfg2.to_dict()
        config_diff = {
            k: {"experiment_1": d1.get(k), "experiment_2": d2.get(k)}
            for k in set(d1) | set(d2)
            if d1.get(k) != d2.get(k)
        }

        # --- Metrics comparison ---
        metric_keys = [
            "mrr", "mean_faithfulness", "mean_answer_relevance",
            "mean_context_precision", "mean_context_recall",
            "total_cost_usd",
            "mean_retrieval_latency_ms", "mean_generation_latency_ms",
            "mean_evaluation_latency_ms",
        ]
        metrics_cmp: Dict[str, Dict] = {}
        for mk in metric_keys:
            v1 = res1.get(mk)
            v2 = res2.get(mk)
            delta = None
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                delta = v2 - v1
            metrics_cmp[mk] = {
                exp_id_1: v1,
                exp_id_2: v2,
                "delta (2-1)": delta,
            }

        # Recall/NDCG at each k
        for prefix in ("mean_recall", "mean_ndcg"):
            for k in [1, 3, 5, 10]:
                mk = f"{prefix}@{k}"
                v1 = (res1.get(prefix) or {}).get(str(k)) or (res1.get(prefix) or {}).get(k)
                v2 = (res2.get(prefix) or {}).get(str(k)) or (res2.get(prefix) or {}).get(k)
                delta = (v2 - v1) if (v1 is not None and v2 is not None) else None
                metrics_cmp[mk] = {exp_id_1: v1, exp_id_2: v2, "delta (2-1)": delta}

        # --- Print table ---
        self._print_comparison(
            exp_id_1, exp_id_2, config_diff, metrics_cmp
        )

        return {
            "config_diff": config_diff,
            "metrics_comparison": metrics_cmp,
        }

    @staticmethod
    def _print_comparison(
        id1: str,
        id2: str,
        config_diff: Dict,
        metrics_cmp: Dict,
    ) -> None:
        sep = "=" * 72
        col = "{:<35} {:>15} {:>15}"
        print(f"\n{sep}")
        print(f"  EXPERIMENT COMPARISON")
        print(f"    A = {id1}")
        print(f"    B = {id2}")
        print(sep)

        if config_diff:
            print("\n  CONFIG DIFFERENCES")
            header = col.format("Parameter", "A", "B")
            print(f"  {header}")
            print(f"  {'-' * 65}")
            for k, v in config_diff.items():
                print(col.format(
                    f"  {k}"[:35],
                    str(v['experiment_1'])[:15],
                    str(v['experiment_2'])[:15],
                ))
        else:
            print("\n  Configs are identical.")

        print("\n  METRICS COMPARISON")
        col2 = "{:<35} {:>13} {:>13} {:>13}"
        header2 = col2.format("Metric", "A", "B", "Δ (B−A)")
        print(f"  {header2}")
        print(f"  {'-' * 74}")
        for metric, vals in metrics_cmp.items():
            a = vals.get(id1)
            b = vals.get(id2)
            delta = vals.get("delta (2-1)")

            def _fmt(v):
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            delta_str = _fmt(delta)
            if isinstance(delta, float):
                sign = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
                delta_str = f"{sign}{abs(delta):.4f}"

            print(col2.format(
                f"  {metric}"[:35],
                _fmt(a)[:13],
                _fmt(b)[:13],
                delta_str[:13],
            ))
        print(sep)
