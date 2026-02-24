"""
LLM-AS-JUDGE EVALUATION LAYER  (Upgrade 1)

EvaluationJudge evaluates every (query, response, context) triple using
Gemini-Pro (or any configured judge model) across four RAG quality metrics:

  1. Faithfulness       – are response claims supported by retrieved context?
  2. Answer Relevance   – does the answer fully address the question?
  3. Context Precision  – what fraction of retrieved chunks actually matter?
  4. Context Recall     – what fraction of ground-truth facts are in context?

All results are persisted to SQLite for offline analysis.

Design decisions:
  - Judge model is fully configurable (swap Gemini for GPT-4o / Claude easily)
  - Every Gemini call has exponential-backoff retry (max 3 attempts)
  - LLM outputs are parsed defensively; parse failures are logged and counted
  - Token counts are tracked per call for cost estimation
  - JSON-only output is enforced in every prompt
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost constants (USD per 1 000 tokens, as of early 2025)
# ---------------------------------------------------------------------------
COST_PER_1K_INPUT = {
    "gemini-1.5-pro":   0.00125,
    "gemini-1.5-flash": 0.000075,
    "gpt-4o":           0.005,
    "claude-3-5-sonnet-20241022": 0.003,
}
COST_PER_1K_OUTPUT = {
    "gemini-1.5-pro":   0.005,
    "gemini-1.5-flash": 0.0003,
    "gpt-4o":           0.015,
    "claude-3-5-sonnet-20241022": 0.015,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """All metrics for one (query, response, context) triple."""
    query_id: str
    query: str
    response: str
    faithfulness: float                    # 0-1
    unsupported_claims: List[str]          # from faithfulness judge
    answer_relevance: float                # 0-1
    relevance_reasoning: str
    context_precision: float               # 0-1
    context_recall: Optional[float]        # 0-1, None if no ground truth
    missing_facts: List[str]
    judge_model: str
    timestamp: float
    latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    parse_failures: int                    # degraded-mode counter


@dataclass
class _CallStats:
    """Tracks token usage across one evaluation session."""
    input_tokens: int = 0
    output_tokens: int = 0
    parse_failures: int = 0

    def add(self, inp: int, out: int) -> None:
        self.input_tokens += inp
        self.output_tokens += out


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class EvaluationJudge:
    """
    LLM-as-Judge evaluation layer.

    Parameters
    ----------
    judge_model : str
        Model name used for judging, e.g. "gemini-1.5-pro".
    api_key_env : str
        Environment variable that holds the Gemini API key.
    db_path : str
        Path to the SQLite database for result persistence.
    max_retries : int
        Max retry attempts per LLM call (exponential backoff).
    """

    def __init__(
        self,
        judge_model: str = "gemini-2.0-flash",
        api_key_env: str = "GEMINI_API_KEY",
        db_path: str = "evaluations.db",
        max_retries: int = 3,
    ) -> None:
        self.judge_model = judge_model
        self.api_key_env = api_key_env
        self.db_path = db_path
        self.max_retries = max_retries
        self._llm = None

        self._init_llm()
        self._init_db()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        """Initialise the judge LLM client based on model name."""
        model = self.judge_model
        api_key = os.getenv(self.api_key_env)

        if "gemini" in model:
            if not api_key:
                logger.warning(
                    f"[EvaluationJudge] No API key in env '{self.api_key_env}'. "
                    "Calls will fail at runtime."
                )
                return
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._llm = genai.GenerativeModel(model_name=model)
                logger.info(f"[EvaluationJudge] Initialised Gemini judge: {model}")
            except ImportError:
                raise ImportError(
                    "google-generativeai not installed. "
                    "Run: pip install google-generativeai"
                )
        elif "gpt" in model:
            try:
                from openai import OpenAI
                self._llm = OpenAI(api_key=api_key)
                logger.info(f"[EvaluationJudge] Initialised OpenAI judge: {model}")
            except ImportError:
                raise ImportError("openai not installed. Run: pip install openai")
        elif "claude" in model:
            try:
                import anthropic
                self._llm = anthropic.Anthropic(api_key=api_key)
                logger.info(f"[EvaluationJudge] Initialised Anthropic judge: {model}")
            except ImportError:
                raise ImportError("anthropic not installed. Run: pip install anthropic")
        else:
            raise ValueError(
                f"Unsupported judge model: {model}. "
                "Use a gemini-*, gpt-*, or claude-* model name."
            )

    def _init_db(self) -> None:
        """Create SQLite evaluations table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    query_id           TEXT PRIMARY KEY,
                    query              TEXT,
                    response           TEXT,
                    faithfulness       REAL,
                    answer_relevance   REAL,
                    context_precision  REAL,
                    context_recall     REAL,
                    judge_model        TEXT,
                    timestamp          REAL,
                    latency_ms         REAL,
                    input_tokens       INTEGER,
                    output_tokens      INTEGER,
                    estimated_cost_usd REAL,
                    parse_failures     INTEGER,
                    unsupported_claims TEXT,
                    missing_facts      TEXT,
                    relevance_reasoning TEXT
                )
            """)
            conn.commit()
        logger.info(f"[EvaluationJudge] SQLite DB ready at: {self.db_path}")

    # ------------------------------------------------------------------
    # Core LLM call with retry
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        prompt: str,
        stats: _CallStats,
    ) -> Tuple[str, int, int]:
        """
        Call the configured judge LLM with exponential-backoff retry.

        Returns
        -------
        (raw_text, input_tokens, output_tokens)
        """
        model = self.judge_model
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            wait = 2 ** attempt          # 1 s, 2 s, 4 s
            try:
                if "gemini" in model:
                    resp = self._llm.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.0,
                            "response_mime_type": "application/json",
                        },
                    )
                    text = resp.text
                    # Gemini usage metadata (tokens)
                    usage = getattr(resp, "usage_metadata", None)
                    inp = getattr(usage, "prompt_token_count", len(prompt) // 4)
                    out = getattr(usage, "candidates_token_count", len(text) // 4)
                    return text, inp, out

                elif "gpt" in model:
                    from openai import OpenAI
                    client: OpenAI = self._llm
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    text = completion.choices[0].message.content
                    inp = completion.usage.prompt_tokens
                    out = completion.usage.completion_tokens
                    return text, inp, out

                elif "claude" in model:
                    import anthropic
                    client: anthropic.Anthropic = self._llm
                    message = client.messages.create(
                        model=model,
                        max_tokens=512,
                        temperature=0.0,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = message.content[0].text
                    inp = message.usage.input_tokens
                    out = message.usage.output_tokens
                    return text, inp, out

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    f"[EvaluationJudge] LLM call failed (attempt {attempt+1}/"
                    f"{self.max_retries}): {exc}. Retrying in {wait}s…"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[EvaluationJudge] All {self.max_retries} attempts failed. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # JSON parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, fallback: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Parse JSON from LLM output. Returns (dict, success_bool).
        Strips markdown code fences if present.
        """
        text = raw.strip()
        # Strip ```json ... ``` fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        try:
            return json.loads(text), True
        except json.JSONDecodeError:
            # Second attempt: find first { … } substring
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end]), True
                except json.JSONDecodeError:
                    pass
        logger.warning("[EvaluationJudge] JSON parse failed; using fallback values.")
        return fallback, False

    # ------------------------------------------------------------------
    # Metric 1 – Faithfulness
    # ------------------------------------------------------------------

    def _evaluate_faithfulness(
        self, context: str, response: str, stats: _CallStats
    ) -> Tuple[float, List[str]]:
        """
        Return (faithfulness_score, unsupported_claims).
        Score = proportion of response claims directly supported by context.
        """
        prompt = f"""You are a strict factual auditor evaluating an AI-generated response.

Given this context:
{context}

And this response:
{response}

Score the faithfulness of the response to the context on a scale of 0.0 to 1.0.
Faithfulness = proportion of response claims that are directly supported by the context.
Claims not in the context should be flagged as unsupported.

Return ONLY a valid JSON object with no extra text:
{{"score": <float between 0.0 and 1.0>, "unsupported_claims": [<list of unsupported claim strings>]}}"""

        fallback = {"score": 0.0, "unsupported_claims": ["[parse error]"]}
        raw, inp, out = self._call_llm(prompt, stats)
        stats.add(inp, out)
        data, ok = self._parse_json(raw, fallback)
        if not ok:
            stats.parse_failures += 1
            # Retry once
            try:
                raw2, inp2, out2 = self._call_llm(prompt, stats)
                stats.add(inp2, out2)
                data, ok2 = self._parse_json(raw2, fallback)
                if not ok2:
                    stats.parse_failures += 1
                    data = fallback
            except Exception:
                data = fallback

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        claims = data.get("unsupported_claims", [])
        if not isinstance(claims, list):
            claims = [str(claims)]
        return score, claims

    # ------------------------------------------------------------------
    # Metric 2 – Answer Relevance
    # ------------------------------------------------------------------

    def _evaluate_answer_relevance(
        self, query: str, response: str, stats: _CallStats
    ) -> Tuple[float, str]:
        """Return (relevance_score, reasoning_sentence)."""
        prompt = f"""You are evaluating how well an AI answer addresses a user question.

Given this question:
{query}

And this answer:
{response}

Score how completely and directly the answer addresses the question on a scale of 0.0 to 1.0.
1.0 = fully and directly answers the question.
0.0 = completely irrelevant or missing the point.

Return ONLY a valid JSON object with no extra text:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one concise sentence>"}}"""

        fallback = {"score": 0.0, "reasoning": "[parse error]"}
        raw, inp, out = self._call_llm(prompt, stats)
        stats.add(inp, out)
        data, ok = self._parse_json(raw, fallback)
        if not ok:
            stats.parse_failures += 1
            data = fallback

        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        reasoning = str(data.get("reasoning", ""))
        return score, reasoning

    # ------------------------------------------------------------------
    # Metric 3 – Context Precision
    # ------------------------------------------------------------------

    def _evaluate_context_precision(
        self, query: str, chunks: List[str], stats: _CallStats
    ) -> float:
        """
        For each retrieved chunk, ask whether it contributes to answering the query.
        context_precision = relevant_chunks / total_chunks
        """
        if not chunks:
            return 0.0

        relevant = 0
        for chunk in chunks:
            prompt = f"""You are evaluating retrieved document chunks for a search query.

Given this query:
{query}

And this context chunk:
{chunk}

Is this chunk relevant to answering the query? Answer with true or false.

Return ONLY a valid JSON object with no extra text:
{{"relevant": <true or false>}}"""

            fallback = {"relevant": False}
            try:
                raw, inp, out = self._call_llm(prompt, stats)
                stats.add(inp, out)
                data, ok = self._parse_json(raw, fallback)
                if not ok:
                    stats.parse_failures += 1
                    data = fallback
                if data.get("relevant", False):
                    relevant += 1
            except Exception as exc:
                logger.warning(f"[EvaluationJudge] Context precision chunk error: {exc}")

        return relevant / len(chunks)

    # ------------------------------------------------------------------
    # Metric 4 – Context Recall (requires ground truth)
    # ------------------------------------------------------------------

    def _evaluate_context_recall(
        self,
        ground_truth: str,
        context: str,
        stats: _CallStats,
    ) -> Tuple[float, List[str]]:
        """Return (recall_score, missing_facts)."""
        prompt = f"""You are evaluating how well retrieved context covers known facts.

Given this ground truth answer:
{ground_truth}

And this retrieved context:
{context}

What fraction of the key facts stated in the ground truth appear in the retrieved context?
Identify any important facts from the ground truth that are missing from the context.

Return ONLY a valid JSON object with no extra text:
{{"recall_score": <float between 0.0 and 1.0>, "missing_facts": [<list of missing fact strings>]}}"""

        fallback = {"recall_score": 0.0, "missing_facts": ["[parse error]"]}
        raw, inp, out = self._call_llm(prompt, stats)
        stats.add(inp, out)
        data, ok = self._parse_json(raw, fallback)
        if not ok:
            stats.parse_failures += 1
            data = fallback

        score = float(data.get("recall_score", 0.0))
        score = max(0.0, min(1.0, score))
        missing = data.get("missing_facts", [])
        if not isinstance(missing, list):
            missing = [str(missing)]
        return score, missing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        response: str,
        chunks: List[str],
        ground_truth: Optional[str] = None,
        query_id: Optional[str] = None,
    ) -> JudgeResult:
        """
        Run all four evaluation metrics for one RAG response.

        Parameters
        ----------
        query        : The user question.
        response     : The generated answer from the RAG system.
        chunks       : List of retrieved text chunks used as context.
        ground_truth : (Optional) Known correct answer for Context Recall.
        query_id     : (Optional) Unique ID; generated if omitted.

        Returns
        -------
        JudgeResult dataclass with all scores.
        """
        if self._llm is None:
            raise RuntimeError(
                "[EvaluationJudge] LLM not initialised. Check API key."
            )

        query_id = query_id or str(uuid.uuid4())
        stats = _CallStats()
        t0 = time.perf_counter()

        context = "\n\n---\n\n".join(chunks)

        # --- Run all metrics ---
        faithfulness, unsupported_claims = self._evaluate_faithfulness(
            context, response, stats
        )
        answer_relevance, relevance_reasoning = self._evaluate_answer_relevance(
            query, response, stats
        )
        context_precision = self._evaluate_context_precision(query, chunks, stats)

        context_recall: Optional[float] = None
        missing_facts: List[str] = []
        if ground_truth:
            context_recall, missing_facts = self._evaluate_context_recall(
                ground_truth, context, stats
            )

        latency_ms = (time.perf_counter() - t0) * 1000
        timestamp = time.time()

        # --- Cost estimation ---
        model = self.judge_model
        cost_in = (stats.input_tokens / 1000) * COST_PER_1K_INPUT.get(model, 0.002)
        cost_out = (stats.output_tokens / 1000) * COST_PER_1K_OUTPUT.get(model, 0.006)
        estimated_cost = cost_in + cost_out

        result = JudgeResult(
            query_id=query_id,
            query=query,
            response=response,
            faithfulness=faithfulness,
            unsupported_claims=unsupported_claims,
            answer_relevance=answer_relevance,
            relevance_reasoning=relevance_reasoning,
            context_precision=context_precision,
            context_recall=context_recall,
            missing_facts=missing_facts,
            judge_model=model,
            timestamp=timestamp,
            latency_ms=latency_ms,
            total_input_tokens=stats.input_tokens,
            total_output_tokens=stats.output_tokens,
            estimated_cost_usd=estimated_cost,
            parse_failures=stats.parse_failures,
        )

        self._save_result(result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_result(self, r: JudgeResult) -> None:
        """Upsert a JudgeResult into the SQLite evaluations table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO evaluations VALUES
                    (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        r.query_id,
                        r.query,
                        r.response,
                        r.faithfulness,
                        r.answer_relevance,
                        r.context_precision,
                        r.context_recall,
                        r.judge_model,
                        r.timestamp,
                        r.latency_ms,
                        r.total_input_tokens,
                        r.total_output_tokens,
                        r.estimated_cost_usd,
                        r.parse_failures,
                        json.dumps(r.unsupported_claims),
                        json.dumps(r.missing_facts),
                        r.relevance_reasoning,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"[EvaluationJudge] DB write failed: {exc}")

    def load_results(self, limit: int = 1000) -> List[Dict]:
        """Fetch recent evaluation results from SQLite as a list of dicts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM evaluations ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["unsupported_claims"] = json.loads(d.get("unsupported_claims") or "[]")
            d["missing_facts"] = json.loads(d.get("missing_facts") or "[]")
            results.append(d)
        return results
