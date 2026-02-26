"""
Interview turn evaluation engine.

Scores each candidate answer against a structured rubric using a dedicated
reasoning model (gpt-5.2 by default). Runs entirely off the critical path —
all scoring is fire-and-forget via asyncio.create_task(). TTS latency is
never affected.

Token consumption is the primary cost driver on a busy session. Every
mechanism here exists to cut that cost without degrading signal quality:

  Adaptive sampling     — full evaluation for the first N turns, then 1-in-K
                          sampling. Chatty candidates don't burn 10× the budget.
  Per-session budget    — hard token cap stored in Redis. Once a session
                          exhausts its budget, evaluation stops silently.
  Minimum answer gate   — answers under EVAL_MIN_ANSWER_CHARS are skipped.
                          "Yes", "I don't know", and one-liners don't need scoring.
  Prompt truncation     — transcripts and LLM responses are capped before the
                          API call. The scoring model never sees more than it needs.
  Tight completion cap  — max_tokens on the scoring chain constrains output size.
                          Structured JSON scores don't need paragraphs.
  Token-bucket limiter  — separate from the main LLM rate limiter so evaluation
                          can't consume the interview quota.
  Dedicated bulkhead    — max EVAL_MAX_CONCURRENT parallel scoring calls.
                          A burst of simultaneous PTT presses won't spawn 20 API calls.
  Circuit breaker       — three consecutive failures open the breaker and stop
                          evaluation until the model recovers.
  Content-hash cache    — identical (question, answer) pairs return cached scores.
                          Useful when candidates repeat or rephrase the same answer.

Redis key layout
────────────────
  eval:score:v1:{session_id}:{turn_index}    JSON blob, TTL = SESSION_TTL_S + GRACE
  eval:budget:v1:{session_id}                int tokens consumed, same TTL
  eval:dedup:v1:{session_id}:{turn_index}    NX sentinel — prevents double-scoring
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import weakref
from dataclasses import dataclass, field, fields
from typing import Any

import redis.asyncio as aioredis
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from opentelemetry.trace import StatusCode
from pydantic import SecretStr

from app.common.shared import (
    CircuitBreaker,
    CircuitBreakerOpen,
    InMemoryLRU,
    RateLimiter,
    ServiceHealthState,
    backoff_retry,
    bulkheads,
    degraded_mode,
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
    make_versioned_cache_key,  # noqa
    new_request_id,
)
from app.monitoring.observability import get_logger
from dotenv import load_dotenv

load_dotenv()

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

EVAL_MODEL: str = os.getenv("EVAL_MODEL", "gpt-5.2")
EVAL_FALLBACK: str = os.getenv("EVAL_FALLBACK", "gpt-5-mini")

EVAL_TEMPERATURE: float = float(os.getenv("EVAL_TEMPERATURE", "0.1"))

# Per-session hard cap on tokens consumed by evaluation calls.
# Once exceeded, evaluation stops and the session report uses whatever
# scores were collected. ~700 tokens per eval → ~14 full evaluations at 10k.
EVAL_SESSION_TOKEN_BUDGET: int = int(os.getenv("EVAL_SESSION_TOKEN_BUDGET", "10000"))

# Answers shorter than this are not worth scoring — they're fillers.
EVAL_MIN_ANSWER_CHARS: int = int(os.getenv("EVAL_MIN_ANSWER_CHARS", "40"))

# Input truncation — keeps prompt tokens bounded regardless of answer length.
EVAL_MAX_QUESTION_CHARS: int = int(os.getenv("EVAL_MAX_QUESTION_CHARS", "400"))
EVAL_MAX_TRANSCRIPT_CHARS: int = int(os.getenv("EVAL_MAX_TRANSCRIPT_CHARS", "900"))
EVAL_MAX_COMPLETION_TOKENS: int = int(os.getenv("EVAL_MAX_COMPLETION_TOKENS", "420"))

# Adaptive sampling: evaluate every turn up to EVAL_FULL_TURNS, then
# evaluate 1 in EVAL_SAMPLE_EVERY_N turns. Turn index is zero-based.
EVAL_FULL_TURNS: int = int(os.getenv("EVAL_FULL_TURNS", "4"))
EVAL_SAMPLE_EVERY_N: int = int(os.getenv("EVAL_SAMPLE_EVERY_N", "2"))

# Concurrency + rate limiting (separate from main LLM limits)
EVAL_MAX_CONCURRENT: int = int(os.getenv("EVAL_MAX_CONCURRENT", "4"))
EVAL_RATE_PER_SEC: float = float(os.getenv("EVAL_RATE_PER_SEC", "3.0"))
EVAL_RATE_BURST: float = float(os.getenv("EVAL_RATE_BURST", "6.0"))

# Redis (reuses the same instance as session_store and LLM_service.py)
REDIS_URL: str | None = os.getenv("REDIS_URL")
REDIS_MAX_CONN: int = int(os.getenv("REDIS_MAX_CONN", "200"))

# Evaluation TTL mirrors session TTL so scores expire with the session.
EVAL_TTL_S: int = int(os.getenv("SESSION_TTL_S", "3600")) + int(
    os.getenv("SESSION_GRACE_S", "60")
)

EVAL_LRU_SIZE: int = int(os.getenv("EVAL_LRU_SIZE", "256"))

# Cache prefix — bump version if scoring schema changes
_SCORE_PREFIX = "eval:score:v1:"
_BUDGET_PREFIX = "eval:budget:v1:"
_DEDUP_PREFIX = "eval:dedup:v1:"
_CACHE_PREFIX = "eval:cache:v1:"

# Register the evaluation bulkhead if not already present
bulkheads.register("eval.batch", limit=EVAL_MAX_CONCURRENT)

# ── scoring rubric ─────────────────────────────────────────────────────────────
# Kept terse to minimise input tokens. The model is instructed to respond
# with JSON only — response_format=json_object enforces this at the API level.

_EVAL_SYSTEM_PROMPT = """\
You are a strict technical interview evaluator. Your only job is to score a \
candidate's spoken answer and justify the score.

You MUST respond with a single valid JSON object — no prose, no markdown, no \
explanation outside the JSON.

Scoring dimensions (each 0–10):
  technical_accuracy  — correctness of facts, algorithms, and concepts stated
  reasoning_depth     — quality of analysis, trade-off discussion, edge cases
  clarity             — how well-structured and understandable the answer was
  completeness        — whether the answer fully addressed what was asked
  overall             — holistic score weighted across the above

Additional fields:
  key_points   — list of 2–4 exact short phrases quoted verbatim from the \
candidate's answer that most influenced the score (positive or negative)
  reasoning    — 1–2 sentences explaining the scores, citing the key_points
  flags        — list, zero or more of: \
"shallow" | "off_topic" | "strong_reasoning" | "needs_followup" | "excellent" | "contradictory"

JSON schema (strict):
{
  "technical_accuracy": <int 0-10>,
  "reasoning_depth":    <int 0-10>,
  "clarity":            <int 0-10>,
  "completeness":       <int 0-10>,
  "overall":            <int 0-10>,
  "key_points":         [<str>, ...],
  "reasoning":          <str>,
  "flags":              [<str>, ...]
}
"""

_EVAL_PROMPT_HASH: str = hashlib.sha256(_EVAL_SYSTEM_PROMPT.encode()).hexdigest()[:16]

# ── Prometheus metrics ─────────────────────────────────────────────────────────

_evals_total = make_counter("eval_turns_total", "Evaluation attempts", ["status"])
_evals_skipped = make_counter(
    "eval_turns_skipped_total", "Evaluations skipped", ["reason"]
)
_budget_exhausted = make_counter(
    "eval_budget_exhausted_total", "Sessions over token cap"
)
_cache_hits = make_counter("eval_cache_hits_total", "Score cache hits")
_circuit_trips = make_counter("eval_circuit_trips_total", "Circuit breaker trips")
_tokens_consumed = make_counter(
    "eval_tokens_consumed_total", "Total tokens consumed by evaluation"
)

_eval_latency = make_histogram(
    "eval_turn_latency_seconds",
    "Time from score_turn() call to Redis write",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30),
)
_score_distribution = make_histogram(
    "eval_overall_score",
    "Distribution of overall scores",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)
_active_evals = make_gauge("eval_active", "Evaluations currently in flight")


# ── data model ─────────────────────────────────────────────────────────────────


@dataclass
class TurnScore:
    """
    Structured result of a single turn evaluation.

    All numeric fields default to -1 to distinguish "not evaluated" from
    a score of zero. Callers must check evaluated=True before using scores.
    """

    session_id: str
    turn_index: int
    technical_accuracy: int = -1
    reasoning_depth: int = -1
    clarity: int = -1
    completeness: int = -1
    overall: int = -1
    key_points: list = field(default_factory=list)
    reasoning: str = ""
    flags: list = field(default_factory=list)
    evaluated: bool = False
    skipped: bool = False
    skip_reason: str = ""
    tokens_used: int = 0
    eval_latency_s: float = 0.0
    evaluated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "technical_accuracy": self.technical_accuracy,
            "reasoning_depth": self.reasoning_depth,
            "clarity": self.clarity,
            "completeness": self.completeness,
            "overall": self.overall,
            "key_points": self.key_points,
            "reasoning": self.reasoning,
            "flags": self.flags,
            "evaluated": self.evaluated,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "tokens_used": self.tokens_used,
            "eval_latency_s": self.eval_latency_s,
            "evaluated_at": self.evaluated_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "TurnScore":
        valid_fields = {f.name for f in fields(TurnScore)}
        return TurnScore(**{k: v for k, v in d.items() if k in valid_fields})


@dataclass
class SessionReport:
    """Aggregated evaluation report for a completed session."""

    session_id: str
    turns_evaluated: int = 0
    turns_skipped: int = 0
    avg_technical_accuracy: float = 0.0
    avg_reasoning_depth: float = 0.0
    avg_clarity: float = 0.0
    avg_completeness: float = 0.0
    avg_overall: float = 0.0
    strongest_turn: int = -1
    weakest_turn: int = -1
    strongest_score: int = -1
    weakest_score: int = 10
    all_flags: list = field(default_factory=list)
    all_key_points: list = field(default_factory=list)
    tokens_consumed: int = 0
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turns_evaluated": self.turns_evaluated,
            "turns_skipped": self.turns_skipped,
            "avg_technical_accuracy": round(self.avg_technical_accuracy, 2),
            "avg_reasoning_depth": round(self.avg_reasoning_depth, 2),
            "avg_clarity": round(self.avg_clarity, 2),
            "avg_completeness": round(self.avg_completeness, 2),
            "avg_overall": round(self.avg_overall, 2),
            "strongest_turn": self.strongest_turn,
            "weakest_turn": self.weakest_turn,
            "strongest_score": self.strongest_score,
            "weakest_score": self.weakest_score,
            "all_flags": self.all_flags,
            "all_key_points": self.all_key_points,
            "tokens_consumed": self.tokens_consumed,
            "generated_at": self.generated_at,
        }


# ── evaluation engine ──────────────────────────────────────────────────────────


class EvaluationEngine:
    """
    Async evaluation engine — call schedule_turn() from _node_llm after
    appending the turn. Everything else happens off the critical path.

    Thread-safety
    ─────────────
    All state mutations go through asyncio primitives. The engine is safe
    to use from multiple concurrent pipeline tasks within the same event loop.
    The _tasks weakref set prevents fire-and-forget tasks from being garbage-
    collected before they complete while also not holding strong references
    that would delay process shutdown.

    Usage
    ─────
        # In _node_llm, after append_turn():
        question = session.turns[-2].assistant if len(session.turns) >= 2 else ""
        evaluation_engine.schedule_turn(
            session_id    = session_id,
            question      = question,
            candidate_ans = transcript,
            turn_index    = len(session.turns) - 1,
            request_id    = rid,
        )
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._conn_lock = asyncio.Lock()

        self._lru = InMemoryLRU(max_size=EVAL_LRU_SIZE)
        self._breaker = CircuitBreaker(
            name="eval_engine",
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=2,
        )
        self._rate_limiter = RateLimiter(EVAL_RATE_PER_SEC, EVAL_RATE_BURST)

        self._chain_primary: Any = None
        self._chain_fallback: Any = None
        self._inflight: int = 0

        # weakref set — tasks are referenced here until done, then GC'd
        self._tasks: weakref.WeakSet[asyncio.Task] = weakref.WeakSet()

    # ── Redis connection ───────────────────────────────────────────────────────

    async def _get_redis(self) -> aioredis.Redis | None:
        if not REDIS_URL:
            return None
        async with self._conn_lock:
            if self._redis is None:
                self._redis = await aioredis.from_url(
                    REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=REDIS_MAX_CONN,
                    socket_connect_timeout=0.15,
                    socket_timeout=0.5,
                )
        return self._redis

    # ── LangChain chain factory ────────────────────────────────────────────────

    def _get_chain(self, model: str):  # noqa
        """
        Build a JSON-mode ChatOpenAI chain for the given model.
        Chains are constructed lazily and cached per model name.
        response_format=json_object guarantees the output is always
        parseable JSON — no backtick stripping needed.
        """
        llm = ChatOpenAI(
            model=model,
            api_key=SecretStr(OPENAI_API_KEY),
            temperature=EVAL_TEMPERATURE,
            max_retries=0,
            timeout=45,
            max_tokens=EVAL_MAX_COMPLETION_TOKENS,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        return llm | StrOutputParser()

    def _chain(self, primary: bool = True):
        if primary:
            if self._chain_primary is None:
                self._chain_primary = self._get_chain(EVAL_MODEL)
            return self._chain_primary
        if self._chain_fallback is None:
            self._chain_fallback = self._get_chain(EVAL_FALLBACK)
        return self._chain_fallback

    # ── Redis helpers with LRU fallback ───────────────────────────────────────

    async def _rget(self, key: str) -> str | None:
        try:
            r = await self._get_redis()
            if r:
                val = await self._breaker.call(r.get, key)
                if val:
                    return val
                if degraded_mode.active:
                    await degraded_mode.exit("redis_readable_eval")
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("eval_redis_read_failed", key=key, error=str(exc))
            await degraded_mode.enter(f"eval_redis_error: {exc}")

        val = await self._lru.get(key)
        if val and val != "__tombstone__":
            return val
        return None

    async def _rset(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._lru.set(key, value)
        try:
            r = await self._get_redis()
            if r:
                if ttl and ttl > 0:
                    await self._breaker.call(r.setex, key, ttl, value)
                else:
                    await self._breaker.call(r.set, key, value)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("eval_redis_write_failed", key=key, error=str(exc))

    async def _rincr(self, key: str, amount: int, ttl: int) -> int:
        """Atomic increment with TTL on first write — used for budget tracking."""
        try:
            r = await self._get_redis()
            if r:
                pipe = r.pipeline(transaction=True)
                await pipe.incrby(key, amount)
                await pipe.expire(key, ttl)
                results = await self._breaker.call(pipe.execute)
                return int(results[0])
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("eval_redis_incr_failed", key=key, error=str(exc))

        # LRU fallback — approximate, non-atomic
        raw = await self._lru.get(key)
        current = int(raw) if raw and raw.isdigit() else 0
        new_val = current + amount
        await self._lru.set(key, str(new_val))
        return new_val

    async def _rsetnx(self, key: str, value: str, ttl: int) -> bool:
        """Return True if key was set (acquired), False if already exists."""
        try:
            r = await self._get_redis()
            if r:
                result = await self._breaker.call(r.set, key, value, nx=True, ex=ttl)
                return bool(result)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("eval_redis_setnx_failed", key=key, error=str(exc))

        # LRU fallback
        existing = await self._lru.get(key)
        if existing:
            return False
        await self._lru.set(key, value)
        return True

    # ── budget management ──────────────────────────────────────────────────────

    async def _get_tokens_consumed(self, session_id: str) -> int:
        key = f"{_BUDGET_PREFIX}{session_id}"
        raw = await self._rget(key)
        if raw and raw.isdigit():
            return int(raw)
        return 0

    async def _deduct_budget(self, session_id: str, tokens: int) -> int:
        """Increment budget counter and return the new total."""
        key = f"{_BUDGET_PREFIX}{session_id}"
        new_total = await self._rincr(key, tokens, ttl=EVAL_TTL_S)
        _tokens_consumed.inc(tokens)
        return new_total

    # ── skip / sampling logic ──────────────────────────────────────────────────

    async def _should_skip(
        self,
        session_id: str,
        candidate_ans: str,
        turn_index: int,
    ) -> tuple[bool, str]:
        """
        Return (should_skip, reason). Called before every rate-limit acquire
        so we never consume a rate-limit token on an answer we'll discard.
        """

        # 1. Answer too short to be meaningful
        if len(candidate_ans.strip()) < EVAL_MIN_ANSWER_CHARS:
            return True, "answer_too_short"

        # 2. Adaptive sampling — past the full-eval window, sample every N turns
        if turn_index >= EVAL_FULL_TURNS and (turn_index % EVAL_SAMPLE_EVERY_N) != 0:
            return True, "adaptive_sample"

        # 3. Session token budget exhausted
        consumed = await self._get_tokens_consumed(session_id)
        if consumed >= EVAL_SESSION_TOKEN_BUDGET:
            _budget_exhausted.inc()
            return True, "budget_exhausted"

        # 4. Circuit breaker is open — no point queuing
        if self._breaker.state == "OPEN":
            _circuit_trips.inc()
            return True, "circuit_open"

        return False, ""

    # ── prompt construction ────────────────────────────────────────────────────

    def _build_eval_prompt(  # noqa
        self,
        question: str,
        candidate_ans: str,
    ) -> str:
        """
        Build the user-turn prompt. Both inputs are truncated to their
        configured char limits before inclusion so token counts stay bounded
        regardless of how verbose the interview session was.
        """
        q = (question.strip() or "(no prior question context)")[
            :EVAL_MAX_QUESTION_CHARS
        ]
        a = candidate_ans.strip()[:EVAL_MAX_TRANSCRIPT_CHARS]

        # Mark truncation so the model knows the answer may be incomplete
        q_suffix = (
            "…[truncated]" if len(question.strip()) > EVAL_MAX_QUESTION_CHARS else ""
        )
        a_suffix = (
            "…[truncated]"
            if len(candidate_ans.strip()) > EVAL_MAX_TRANSCRIPT_CHARS
            else ""
        )

        return (
            f"INTERVIEWER QUESTION:\n{q}{q_suffix}\n\n"
            f"CANDIDATE ANSWER (verbatim transcript):\n{a}{a_suffix}\n\n"
            "Score the candidate answer according to the JSON schema."
        )

    def _content_hash(self, question: str, candidate_ans: str) -> str:  # noqa
        """Deterministic cache key based on the (question, answer) content."""
        payload = f"{question[:200]}|{candidate_ans[:400]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    # ── model call ─────────────────────────────────────────────────────────────

    async def _call_model(self, messages: list, request_id: str) -> tuple[str, int]:
        """
        Invoke primary model with backoff, fall back to EVAL_FALLBACK on failure.
        Returns (raw_json_text, estimated_tokens).
        Token count is estimated as len(raw_text) // 4 when the API does
        not return usage data (streaming mode does not include it).
        """
        primary_err: Exception | None = None  # noqa

        async def _invoke_primary() -> str:
            return await self._chain(primary=True).ainvoke(messages)

        try:
            text: str = await self._breaker.call(
                backoff_retry,
                _invoke_primary,
                attempts=2,
                base_delay=0.8,
                exceptions=(Exception,),
            )
            return text, max(1, len(text) // 4)

        except (CircuitBreakerOpen, Exception) as exc:
            primary_err = exc
            log.warning(
                "eval_primary_failed_fallback",
                request_id=request_id,
                error=str(exc),
            )

        # Fallback
        async def _invoke_fallback() -> str:
            return await self._chain(primary=False).ainvoke(messages)

        try:
            text = await backoff_retry(
                _invoke_fallback, attempts=2, base_delay=0.8, exceptions=(Exception,)
            )
            log.info("eval_fallback_used", request_id=request_id)
            return text, max(1, len(text) // 4)

        except Exception as fallback_err:
            raise RuntimeError(
                f"Both eval models failed. Primary: {primary_err}. Fallback: {fallback_err}"
            ) from fallback_err

    # ── JSON parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_score(raw: str, session_id: str, turn_index: int) -> dict | None:
        """
        Parse the model's JSON output into a validated score dict.
        Returns None on any parse or validation failure — the caller
        records a skipped score rather than crashing.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Attempt to extract the first JSON object from surrounding text
            start = raw.find("{")
            end = raw.rfind("}") + 1

            if 0 <= start < end:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    log.warning(
                        "eval_json_parse_failed",
                        session_id=session_id,
                        turn_index=turn_index,
                        raw_len=len(raw),
                    )
                    return None

            else:
                log.warning(
                    "eval_json_no_object", session_id=session_id, turn_index=turn_index
                )
                return None

        required_int_fields = (
            "technical_accuracy",
            "reasoning_depth",
            "clarity",
            "completeness",
            "overall",
        )
        for f in required_int_fields:
            val = data.get(f)
            if not isinstance(val, int) or not (0 <= val <= 10):
                log.warning(
                    "eval_field_invalid",
                    session_id=session_id,
                    turn_index=turn_index,
                    field=f,
                    value=val,
                )
                # Clamp rather than discard — partial data is better than none
                data[f] = (
                    max(0, min(10, int(val))) if isinstance(val, (int, float)) else 0
                )

        if not isinstance(data.get("key_points"), list):
            data["key_points"] = []
        if not isinstance(data.get("flags"), list):
            data["flags"] = []
        if not isinstance(data.get("reasoning"), str):
            data["reasoning"] = ""

        return data

    # ── core scoring ───────────────────────────────────────────────────────────

    async def _score_turn_impl(
        self,
        session_id: str,
        question: str,
        candidate_ans: str,
        turn_index: int,
        request_id: str,
    ) -> TurnScore:
        """
        Full scoring path. Handles dedup guard, cache, rate limit, model call,
        JSON parse, Redis persistence, and budget deduction in sequence.

        All failures are caught and returned as a TurnScore with evaluated=False
        so the report aggregator can handle gaps gracefully.
        """
        t0 = time.monotonic()
        result = TurnScore(session_id=session_id, turn_index=turn_index)

        with tracer.start_as_current_span("eval.score_turn") as span:
            span.set_attribute("request_id", request_id)
            span.set_attribute("session_id", session_id)
            span.set_attribute("turn_index", turn_index)

            # ── dedup guard ────────────────────────────────────────────────────
            # Prevents double-scoring if schedule_turn() is called twice for
            # the same turn (e.g. retry after a crash mid-turn).
            dedup_key = f"{_DEDUP_PREFIX}{session_id}:{turn_index}"
            if not await self._rsetnx(dedup_key, request_id, ttl=EVAL_TTL_S):
                log.info(
                    "eval_dedup_skip", session_id=session_id, turn_index=turn_index
                )
                result.skipped = True
                result.skip_reason = "duplicate"
                _evals_skipped.labels(reason="duplicate").inc()
                return result

            # ── content-hash cache ─────────────────────────────────────────────
            content_hash = self._content_hash(question, candidate_ans)
            cache_key = f"{_CACHE_PREFIX}{content_hash}"
            cached_raw = await self._rget(cache_key)
            if cached_raw:
                parsed = self._parse_score(cached_raw, session_id, turn_index)
                if parsed:
                    _cache_hits.inc()
                    span.set_attribute("cache_hit", True)
                    log.info(
                        "eval_cache_hit",
                        session_id=session_id,
                        turn_index=turn_index,
                        request_id=request_id,
                    )
                    result.technical_accuracy = parsed["technical_accuracy"]  # noqa
                    result.reasoning_depth = parsed["reasoning_depth"]
                    result.clarity = parsed["clarity"]
                    result.completeness = parsed["completeness"]
                    result.overall = parsed["overall"]
                    result.key_points = parsed["key_points"]
                    result.reasoning = parsed["reasoning"]
                    result.flags = parsed["flags"]
                    result.evaluated = True
                    result.eval_latency_s = round(time.monotonic() - t0, 3)
                    await self._persist_score(session_id, turn_index, result)
                    _score_distribution.observe(result.overall)
                    return result

            # ── rate limit ─────────────────────────────────────────────────────
            await self._rate_limiter.acquire()

            # ── bulkhead ───────────────────────────────────────────────────────
            async with bulkheads.acquire("eval.batch"):
                _active_evals.inc()
                self._inflight += 1
                span.set_attribute("cache_hit", False)

                try:
                    messages = [
                        SystemMessage(content=_EVAL_SYSTEM_PROMPT),
                        HumanMessage(
                            content=self._build_eval_prompt(question, candidate_ans)
                        ),
                    ]

                    raw_json, est_tokens = await self._call_model(messages, request_id)

                    parsed = self._parse_score(raw_json, session_id, turn_index)
                    if not parsed:
                        result.skipped = True
                        result.skip_reason = "parse_failed"
                        _evals_total.labels(status="parse_error").inc()
                        return result

                    result.technical_accuracy = parsed["technical_accuracy"]  # noqa
                    result.reasoning_depth = parsed["reasoning_depth"]
                    result.clarity = parsed["clarity"]
                    result.completeness = parsed["completeness"]
                    result.overall = parsed["overall"]
                    result.key_points = parsed["key_points"]
                    result.reasoning = parsed["reasoning"]
                    result.flags = parsed["flags"]
                    result.tokens_used = est_tokens
                    result.evaluated = True
                    result.eval_latency_s = round(time.monotonic() - t0, 3)

                    latency = time.monotonic() - t0
                    _eval_latency.observe(latency)
                    _score_distribution.observe(result.overall)
                    _evals_total.labels(status="ok").inc()
                    span.set_attribute("overall_score", result.overall)
                    span.set_attribute("tokens_est", est_tokens)
                    span.set_attribute("latency_s", round(latency, 3))

                    # Persist score and warm content cache
                    await self._persist_score(session_id, turn_index, result)
                    await self._rset(cache_key, raw_json, ttl=EVAL_TTL_S)

                    # Deduct from session budget
                    new_budget = await self._deduct_budget(session_id, est_tokens)

                    log.info(
                        "eval_turn_scored",
                        request_id=request_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        overall=result.overall,
                        flags=result.flags,
                        tokens=est_tokens,
                        budget_consumed=new_budget,
                        latency_s=round(latency, 3),
                    )

                    return result

                except asyncio.CancelledError:
                    log.warning(
                        "eval_cancelled", session_id=session_id, turn_index=turn_index
                    )
                    _evals_total.labels(status="cancelled").inc()
                    raise

                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    _evals_total.labels(status="error").inc()
                    log.error(
                        "eval_turn_failed",
                        request_id=request_id,
                        session_id=session_id,
                        turn_index=turn_index,
                        error=str(exc),
                        latency_s=round(time.monotonic() - t0, 3),
                    )
                    result.skipped = True
                    result.skip_reason = f"model_error: {type(exc).__name__}"
                    return result

                finally:
                    _active_evals.dec()
                    self._inflight -= 1

    async def _persist_score(
        self, session_id: str, turn_index: int, score: TurnScore
    ) -> None:
        key = f"{_SCORE_PREFIX}{session_id}:{turn_index}"
        await self._rset(key, json.dumps(score.to_dict()), ttl=EVAL_TTL_S)

    # ── safe wrapper for create_task ───────────────────────────────────────────

    async def _score_turn_safe(
        self,
        session_id: str,
        question: str,
        candidate_ans: str,
        turn_index: int,
        request_id: str,
    ) -> None:
        """
        Exception-safe coroutine wrapper for asyncio.create_task().
        All exceptions are logged and swallowed — a failing evaluation
        must never propagate to the event loop's unhandled-exception handler
        and must never affect the voice pipeline in any way.
        """
        try:
            skip, reason = await self._should_skip(
                session_id, candidate_ans, turn_index
            )
            if skip:
                _evals_skipped.labels(reason=reason).inc()
                log.info(
                    "eval_skipped",
                    session_id=session_id,
                    turn_index=turn_index,
                    reason=reason,
                )
                return

            await self._score_turn_impl(
                session_id=session_id,
                question=question,
                candidate_ans=candidate_ans,
                turn_index=turn_index,
                request_id=request_id,
            )

        except asyncio.CancelledError:
            pass  # Task was cancelled — clean exit, not an error

        except Exception as exc:
            # Last-resort catch — should not normally fire since
            # _score_turn_impl already catches all exceptions internally
            log.error(
                "eval_unhandled",
                session_id=session_id,
                turn_index=turn_index,
                error=str(exc),
            )

    # ── public API ─────────────────────────────────────────────────────────────

    def schedule_turn(
            self,
            session_id: str,
            question: str,
            candidate_ans: str,
            turn_index: int,
            request_id: str | None = None,
    ) -> asyncio.Task:
        """
        Schedule a single-turn evaluation off the critical path.

        Spawns an asyncio.Task and returns immediately — the caller must NOT
        await this. The task is held in a weakref set (_tasks) so it survives
        long enough to complete without keeping a hard reference that would
        block shutdown or prevent GC.

        The task runs _score_turn_safe(), which internally:
          1. Checks the per-session token budget (Redis).
          2. Checks the dedup sentinel (Redis SET NX) to prevent double-scoring
             the same turn if this method is called twice for the same index.
          3. Calls the scoring LLM (adaptive sampling may skip it).
          4. Persists the TurnScore to Redis under eval:score:v1:{sid}:{turn_index}.

        Calling convention — invoke from conversation_memory.route_committed_turn_to_audit()
        immediately after qa_controller.commit_turn() returns a CommittedTurn:
        ─────────────────────────────────────────────────────────────────────────
            evaluation_engine.schedule_turn(
                session_id    = committed.session_id,
                question      = committed.question,       # the interviewer's question
                candidate_ans = committed.candidate_answer,
                turn_index    = committed.turn_index,
                request_id    = request_id,
            )

        NOTE: Do NOT call this from node_llm directly. All eval scheduling must
        flow through conversation_memory (QAAuditBus) so the dead-letter queue,
        domain-batch tracking, and transcript fan-out stay consistent.
        """
        rid = request_id or new_request_id()

        # Create the scoring task. The name is structured for easy grep in logs
        # and async debuggers: "eval:{session_id}:{turn_index}".
        task = asyncio.create_task(
            self._score_turn_safe(
                session_id=session_id,
                question=question,
                candidate_ans=candidate_ans,
                turn_index=turn_index,
                request_id=rid,
            ),
            name=f"eval:{session_id}:{turn_index}",
        )

        # Weak-reference the task so we can drain it on shutdown (via self._tasks)
        # without holding a strong reference that would keep it alive past its
        # natural completion. Tasks that finish remove themselves from the set
        # via the done-callback registered in _score_turn_safe.
        self._tasks.add(task)
        return task

    async def schedule_domain_eval(
            self,
            session_id: str,
            domain: str,
            eval_pairs: list[dict],
            candidate_meta: dict, # noqa
            request_id: str | None = None,
    ) -> None:
        """
        Schedule evaluation for all Q/A pairs in a completed domain batch.

        Called by conversation_memory._call_eval_engine() when a domain rotation
        occurs (CommittedTurn.domain_rotated=True) or at session close via
        finalize_session_eval(). This is the bridge that makes the QAAuditBus's
        domain-batch dispatch actually reach the evaluation engine.

        Each pair in eval_pairs is expected to have the shape:
            {
                "turn_index":     int,   # zero-based global turn counter
                "question":       str,   # interviewer's question text
                "answer":         str,   # candidate's answer text
            }

        Pairs that are too short (< EVAL_MIN_ANSWER_CHARS) are skipped silently —
        the individual _score_turn_safe() call handles the gate, so no filtering
        is needed here. Scheduling all pairs is safe and cheap.

        candidate_meta is forwarded to the scoring prompt for context (level,
        domain, name). It is not persisted here — the QADocument owns that.

        This method does NOT await the scoring tasks. Each pair becomes an
        independent fire-and-forget task via schedule_turn(). The domain batch
        is therefore evaluated concurrently up to EVAL_MAX_CONCURRENT (the
        bulkhead configured on the engine), not serially.

        Idempotency: the Redis SET NX dedup sentinel inside _score_turn_safe()
        ensures that if this method is called twice for the same (session, domain)
        — e.g. due to a dead-letter retry in QAAuditBus — no turn is double-scored.
        """
        rid = request_id or new_request_id()

        if not eval_pairs:
            log.info(
                "eval_domain_batch_empty",
                session_id=session_id[:8],
                domain=domain,
                request_id=rid,
            )
            return

        log.info(
            "eval_domain_batch_scheduled",
            session_id=session_id[:8],
            domain=domain,
            pair_count=len(eval_pairs),
            request_id=rid,
        )

        for pair in eval_pairs:
            turn_index = pair.get("turn_index", -1)
            question = pair.get("question", "")
            answer = pair.get("answer", "")

            if turn_index < 0 or not question:
                # Corrupted pair — log and skip rather than poisoning the task.
                log.warning(
                    "eval_domain_pair_malformed",
                    session_id=session_id[:8],
                    domain=domain,
                    pair=pair,
                )
                continue

            # schedule_turn() is fire-and-forget — do not await.
            # Each task runs independently through the bulkhead, circuit breaker,
            # and token-budget gate already wired inside _score_turn_safe().
            self.schedule_turn(
                session_id=session_id,
                question=question,
                candidate_ans=answer,
                turn_index=turn_index,
                request_id=f"{rid}:{turn_index}",  # unique rid per sub-task for tracing
            )

    async def get_session_report(self, session_id: str) -> SessionReport:
        """
        Aggregate all persisted turn scores into a session-level report.

        Designed to be called at session end (POST /session/end) or on
        demand via GET /session/report. Safe to call mid-session — partial
        reports reflect turns scored so far.
        """
        with tracer.start_as_current_span("eval.session_report") as span:
            span.set_attribute("session_id", session_id)

            report = SessionReport(session_id=session_id)

            # Scan all score keys for this session
            score_keys: list[str] = []
            try:
                r = await self._get_redis()
                if r:
                    pattern = f"{_SCORE_PREFIX}{session_id}:*"
                    async for key in r.scan_iter(pattern, count=100):
                        score_keys.append(key)
            except Exception as exc:
                log.warning(
                    "eval_report_scan_failed", session_id=session_id, error=str(exc)
                )

            # Load all score blobs (Redis first, LRU fallback per key)
            scores: list[TurnScore] = []
            for key in sorted(score_keys):
                raw = await self._rget(key)
                if raw:
                    try:
                        scores.append(TurnScore.from_dict(json.loads(raw)))
                    except Exception:  # noqa
                        pass

            evaluated = [s for s in scores if s.evaluated]
            skipped = [s for s in scores if s.skipped]

            report.turns_evaluated = len(evaluated)
            report.turns_skipped = len(skipped)

            if evaluated:
                report.avg_technical_accuracy = sum(
                    s.technical_accuracy for s in evaluated
                ) / len(evaluated)
                report.avg_reasoning_depth = sum(
                    s.reasoning_depth for s in evaluated
                ) / len(evaluated)
                report.avg_clarity = sum(s.clarity for s in evaluated) / len(evaluated)
                report.avg_completeness = sum(s.completeness for s in evaluated) / len(
                    evaluated
                )
                report.avg_overall = sum(s.overall for s in evaluated) / len(evaluated)

                best = max(evaluated, key=lambda s: s.overall)
                worst = min(evaluated, key=lambda s: s.overall)

                report.strongest_turn = best.turn_index
                report.weakest_turn = worst.turn_index
                report.strongest_score = best.overall
                report.weakest_score = worst.overall

                report.all_flags = list({f for s in evaluated for f in s.flags})
                report.all_key_points = [kp for s in evaluated for kp in s.key_points]

            report.tokens_consumed = await self._get_tokens_consumed(session_id)
            span.set_attribute("turns_evaluated", report.turns_evaluated)
            span.set_attribute("avg_overall", round(report.avg_overall, 2))

            log.info(
                "eval_report_generated",
                session_id=session_id,
                turns_evaluated=report.turns_evaluated,
                avg_overall=round(report.avg_overall, 2),
                tokens_consumed=report.tokens_consumed,
            )

            return report

    async def get_turn_score(
        self, session_id: str, turn_index: int
    ) -> TurnScore | None:
        """Return the persisted score for a specific turn, or None if not yet scored."""
        key = f"{_SCORE_PREFIX}{session_id}:{turn_index}"
        raw = await self._rget(key)
        if not raw:
            return None
        try:
            return TurnScore.from_dict(json.loads(raw))
        except Exception:  # noqa
            return None

    async def health(self) -> ServiceHealthState:
        try:
            r = await self._get_redis()
            healthy = bool(r and await r.ping())
        except Exception:  # noqa
            healthy = False

        return ServiceHealthState(
            service="eval_engine",
            healthy=healthy and self._breaker.state != "OPEN",
            circuit_state=self._breaker.state,
            inflight=self._inflight,
            degraded=degraded_mode.active,
        )

    async def close(self) -> None:
        # Cancel any still-running eval tasks cleanly
        for task in list(self._tasks):
            if not task.done():
                task.cancel()
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        log.info("eval_engine_closed")


# ── module-level singleton ─────────────────────────────────────────────────────

evaluation_engine = EvaluationEngine()

from fastapi import APIRouter, Depends, Request  # noqa
from app.user_tracking.session_service.session_store import (
    require_session,
    SessionData,
)  # noqa: E402

eval_router = APIRouter()


@eval_router.get("/report", summary="Session evaluation report")
async def session_report(
    session: SessionData = Depends(require_session),
) -> dict:
    """
    Return the aggregated evaluation report for the current session.

    Each scored turn's overall and per-dimension averages are included.
    Call this at any point during or after the interview — partial reports
    reflect turns scored so far. Call again at session end for the full picture.
    """
    report = await evaluation_engine.get_session_report(session.session_id)
    return report.to_dict()


@eval_router.get("/report/turn/{turn_index}", summary="Single turn score")
async def turn_score(
    turn_index: int,
    session: SessionData = Depends(require_session),
) -> dict:
    """
    Return the evaluation score for a specific turn by index (zero-based).

    If the turn has not yet been scored (evaluation is async and may still
    be in flight), returns HTTP 202 to indicate the score is pending.
    """
    from fastapi import HTTPException, status  # noqa: F401

    score = await evaluation_engine.get_turn_score(session.session_id, turn_index)
    if score is None:
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail={
                "status": "pending",
                "message": f"Turn {turn_index} has not been scored yet. Check back shortly.",
            },
        )
    return score.to_dict()


@eval_router.get("/eval/health", include_in_schema=False)
async def eval_health() -> dict:
    state = await evaluation_engine.health()
    return {
        "service": state.service,
        "healthy": state.healthy,
        "circuit_state": state.circuit_state,
        "inflight": state.inflight,
        "degraded": state.degraded,
    }
