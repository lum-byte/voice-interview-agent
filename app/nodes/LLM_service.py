"""
LLM_service.py — Multi-mode LLM node for the structured interview pipeline.

OPERATIONAL MODES:
──────────────────
  InterviewerMode   — Question engine. Accepts LLMInterviewInput from
                      qa_controller. Receives ONLY:
                        domain | level | last_q | last_a | domain_switch_flag
                      Outputs exactly one question with a brief acknowledgment.
                      Enforced by mode-level prompt guards and response validators.

  ATSMode           — Intro extractor. Accepts raw intro transcript.
                      Returns strictly validated JSON only.
                      Separate chain with json_object response_format.

  EvalMode          — Reserved for evaluation_engine.py internal use only.
                      Not callable from voice_graph.

MODE ROUTING:
─────────────
  voice_graph.node_llm determines the current stage from qa_controller and
  routes to the correct mode:
    stage=="greeting"  → returns GREETING_TEXT directly (no LLM call)
    stage=="intro"     → ATSMode.extract(intro_text)
    stage=="interview" → InterviewerMode.stream_question(llm_input)
    stage=="complete"  → returns CLOSING_TEXT directly (no LLM call)

GUARDRAILS:
───────────
  1. InterviewerMode enforces a strict 80-word response cap via
     _ResponseValidator. Responses exceeding this are truncated to the
     first sentence that ends with "?". If no valid question is found,
     the turn is flagged and a safe fallback question is generated.

  2. ATSMode uses json_object response_format at the API level.
     The response is validated against the expected schema before returning.
     Invalid responses trigger rule-based fallback via qa_controller.

  3. Neither InterviewerMode nor ATSMode has access to session history,
     full QA state, or evaluation _signals. These are structurally excluded
     from the message lists they receive.

  4. All modes operate through the same resilience stack:
     Redis + LRU dual-layer cache, circuit breakers, rate limiting,
     bulkhead concurrency control, token counting, OTel tracing,
     Prometheus metrics, stampede protection.

  5. The SYSTEM_PROMPT constant from the previous version is REMOVED.
     Each mode carries its own prompt, imported from qa_controller.

ARCHITECTURE:
─────────────
  LLMNode
    ├── InterviewerMode   (stream_question → AsyncIterator[str])
    ├── ATSMode           (extract → str  [JSON])
    └── _SharedInfra      (cache, circuit, rate, bulkhead, tracing)

  LLMNodeV2(LLMNode)
    ├── PromptInjectionDetector
    ├── ContentPolicyFilter
    ├── QuestionDiversityEnforcer
    ├── StaticQuestionBank
    ├── StreamingWordGuard
    ├── EvalModeLLM
    └── BatchATSProcessor

  RemoteLLMClient         (satisfies LLMNodeProtocol for distributed mode)
    ├── stream_question
    └── extract_ats

  get_llm_node()          Factory — returns RemoteLLMClient or LLMNodeV2.
"""

from __future__ import annotations

import asyncio
import hashlib  # noqa
import json
import math
import os
import random
import re
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import AsyncIterator
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, replace as dc_replace  # noqa
from enum import Enum
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import httpx
import redis.asyncio as aioredis
from dotenv import load_dotenv
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_openai import ChatOpenAI

# Exceptions worth retrying for local LLM calls (langchain/openai transport errors).
# OutputParserException / length-limit parse failures are NOT retryable — they are
# deterministic: the same prompt + same max_tokens will always produce the same failure.
try:
    from langchain_core.exceptions import OutputParserException as _LCOutputParserException
except ImportError:
    _LCOutputParserException = None  # type: ignore[assignment,misc]

_LLM_RETRYABLE_LOCAL: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    TimeoutError,
    ConnectionError,
)
# Remote HTTP client: only retry on network/transient errors, not 4xx.
_LLM_RETRYABLE_REMOTE: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
    TimeoutError,
    ConnectionError,
)
from opentelemetry.trace import StatusCode
from pydantic import SecretStr  # noqa

try:
    from pydantic import BaseModel, ValidationError, field_validator # noqa
    _PYDANTIC_AVAILABLE = True
except ImportError:
    _PYDANTIC_AVAILABLE = False

from app.common.shared import (
    CircuitBreaker,
    CircuitBreakerOpen,
    InMemoryLRU,
    LatencyBudget,
    LatencyBudgetExceeded,
    RateLimiter,
    ServiceHealthState,
    backoff_retry,
    bulkheads,
    current_request_id,  # noqa
    degraded_mode,
    get_meter,
    get_tracer,
    inject_trace_headers,
    make_counter,
    make_gauge,
    make_histogram,
    make_versioned_cache_key,
    new_request_id,
)
from app.interview.qa_controller import (
    LLMInterviewInput,
    INTERVIEWER_SYSTEM_PROMPT,  # noqa
    DOMAIN_SWITCH_SYSTEM_PROMPT,  # noqa
    ATS_SYSTEM_PROMPT,
    ATS_USER_TEMPLATE,
    DOMAIN_REGISTRY,  # noqa
)
from app.monitoring.observability import get_logger
load_dotenv()

log    = get_logger(__name__)
tracer = get_tracer(__name__)
meter  = get_meter(__name__)


# ── config ─────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# Question engine model — fast and instruction-following
PRIMARY_MODEL:  str = os.getenv("LLM_MODEL",          "gpt-5-mini")
FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "gpt-5-nano")

# ATS extraction model — json_object mode required
ATS_MODEL:    str = os.getenv("ATS_MODEL",          "gpt-5-mini")
ATS_FALLBACK: str = os.getenv("ATS_FALLBACK_MODEL", "gpt-5-nano")

# Eval model — wider token budget, higher temperature, internal only
EVAL_MODEL:       str   = os.getenv("LLM_EVAL_MODEL",        "gpt-5.2")
EVAL_TEMPERATURE: float = float(os.getenv("LLM_EVAL_TEMPERATURE", "0.2"))
EVAL_MAX_TOKENS:  int   = int(os.getenv("LLM_EVAL_MAX_TOKENS",  "1500"))

# Temperature: very low for interviewer (deterministic), slightly higher for ATS
INTERVIEWER_TEMPERATURE: float = float(os.getenv("LLM_INTERVIEWER_TEMPERATURE", "0.3"))
ATS_TEMPERATURE:         float = float(os.getenv("LLM_ATS_TEMPERATURE",         "0.1"))

# Max tokens per mode — interviewer is tightly capped
INTERVIEWER_MAX_TOKENS: int = int(os.getenv("LLM_INTERVIEWER_MAX_TOKENS", "120"))
ATS_MAX_TOKENS:         int = int(os.getenv("LLM_ATS_MAX_TOKENS",         "1500"))

# Redis
REDIS_URL:      str | None = os.getenv("REDIS_URL")
REDIS_MAX_CONN: int        = int(os.getenv("REDIS_MAX_CONN", "200"))
CACHE_TTL:      int        = int(os.getenv("LLM_CACHE_TTL", "1800"))   # 30 min
CACHE_PREFIX                = "llm_v5:"   # bump on schema changes

# Stampede protection
STAMPEDE_LOCK_TTL:      int   = 25
STAMPEDE_POLL_INTERVAL: float = 0.35
STAMPEDE_POLL_LIMIT:    int   = 60

# Rate limiting (per mode)
INTERVIEWER_RATE:  float = float(os.getenv("LLM_INTERVIEWER_RATE",  "15.0"))
INTERVIEWER_BURST: float = float(os.getenv("LLM_INTERVIEWER_BURST", "30.0"))
ATS_RATE:          float = float(os.getenv("LLM_ATS_RATE",          "5.0"))
ATS_BURST:         float = float(os.getenv("LLM_ATS_BURST",         "10.0"))

LRU_SIZE:         int = int(os.getenv("LLM_LRU_SIZE",         "512"))
STREAM_CACHE_MIN: int = int(os.getenv("LLM_STREAM_CACHE_MIN", "10"))

# Remote service config
LLM_SERVICE_URL:     str   = os.getenv("LLM_SERVICE_URL",     "")
LLM_SERVICE_API_KEY: str   = os.getenv("LLM_SERVICE_API_KEY", "")
LLM_SERVICE_TIMEOUT: float = float(os.getenv("LLM_SERVICE_TIMEOUT", "60.0"))

# Response validation
MAX_RESPONSE_WORDS: int = int(os.getenv("LLM_MAX_RESPONSE_WORDS", "80"))
MIN_RESPONSE_WORDS: int = int(os.getenv("LLM_MIN_RESPONSE_WORDS", "8"))

# Session token budget
SESSION_TOKEN_BUDGET: int = int(os.getenv("SESSION_TOKEN_BUDGET", "25000"))

# Diversity enforcer
DIVERSITY_THRESHOLD:   float = float(os.getenv("LLM_DIVERSITY_THRESHOLD",   "0.65"))
MAX_DIVERSITY_RETRIES: int   = int(os.getenv("LLM_MAX_DIVERSITY_RETRIES",   "3"))
DIVERSITY_NGRAM_N:     int   = int(os.getenv("LLM_DIVERSITY_NGRAM_N",       "3"))
DIVERSITY_WINDOW:      int   = int(os.getenv("LLM_DIVERSITY_WINDOW",        "5"))

# Feature flags
INJECTION_BLOCK:      bool = os.getenv("LLM_INJECTION_BLOCK",      "true").lower() == "true"
CONTENT_POLICY_BLOCK: bool = os.getenv("LLM_CONTENT_POLICY_BLOCK", "true").lower() == "true"
BANK_FALLBACK_ENABLED: bool = os.getenv("LLM_BANK_FALLBACK_ENABLED", "true").lower() == "true"

# Batch ATS
BATCH_ATS_CONCURRENCY: int   = int(os.getenv("LLM_BATCH_ATS_CONCURRENCY", "5"))
BATCH_ATS_TIMEOUT:     float = float(os.getenv("LLM_BATCH_ATS_TIMEOUT_S", "30.0"))


# ── operational mode enum ──────────────────────────────────────────────────────

class LLMMode(str, Enum):
    INTERVIEWER = "interviewer"
    ATS         = "ats"
    EVAL        = "eval"   # eval_engine.py only — never call from voice_graph


# ── Prometheus metrics ─────────────────────────────────────────────────────────

_req_total     = make_counter("llm_requests_total",            "Total LLM requests",             ["model", "status", "mode"])
_cache_hits    = make_counter("llm_cache_hits_total",          "Cache hits",                     ["source"])
_lru_hits      = make_counter("llm_lru_hits_total",            "LRU hits")
_budget_exc    = make_counter("llm_budget_exceeded_total",     "Budget exceeded")
_validator_fix = make_counter("llm_response_validated_total",  "Validator corrections",          ["action"])
_fallback_used = make_counter("llm_fallback_used_total",       "Primary→fallback switches",      ["mode"])
_injection_blocks  = make_counter("llm_injection_blocks_total",    "Prompt injection blocks")
_policy_blocks     = make_counter("llm_policy_blocks_total",       "Content policy blocks",       ["rule"])
_diversity_retries = make_counter("llm_diversity_retries_total",   "Diversity retry attempts",    ["result"])
_bank_fallbacks    = make_counter("llm_bank_fallbacks_total",      "Static bank fallback uses",   ["domain", "level"])
_ats_schema_fails  = make_counter("llm_ats_schema_failures_total", "ATS JSON schema fails")
_eval_calls        = make_counter("llm_eval_calls_total",          "Eval mode LLM calls",         ["status"])
_batch_ats_calls   = make_counter("llm_batch_ats_calls_total",     "Batch ATS processing calls",  ["status"])
_stream_word_cuts  = make_counter("llm_stream_word_cuts_total",    "Streaming word-count cutoffs")

_latency = make_histogram(
    "llm_latency_seconds", "LLM call latency", ["model", "mode"],
    buckets=(0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0),
)
_ttft = make_histogram(
    "llm_time_to_first_token_seconds", "Time to first token", ["model"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
_tokens = make_histogram(
    "llm_tokens_used", "Token counts", ["model", "kind"],
    buckets=(20, 50, 100, 200, 400, 800, 1500),
)
_diversity_similarity = make_histogram(
    "llm_question_similarity_score", "N-gram cosine similarity vs recent questions",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
_eval_latency = make_histogram(
    "llm_eval_latency_seconds", "Eval mode LLM call latency",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0),
)
_batch_ats_throughput = make_histogram(
    "llm_batch_ats_throughput_per_minute", "Batch ATS processing throughput",
    buckets=(1, 2, 5, 10, 20, 50),
)

_active  = make_gauge("llm_active_requests",      "In-flight requests",    ["mode"])
_cb_open = make_gauge("llm_circuit_breaker_open", "Circuit breaker state", ["model", "path"])


# ── response validator ─────────────────────────────────────────────────────────

class _ResponseValidator:
    """
    Enforces the interviewer response contract:
      - Must contain exactly one question (ends with ?)
      - Must be under MAX_RESPONSE_WORDS words
      - Must be over MIN_RESPONSE_WORDS words (not empty/trivial)
      - Must not contain multiple question marks (multiple questions)
      - Must not contain bullets, numbered lists, or headers

    On violation, attempts to repair by extracting the first valid sentence.
    If repair fails, raises ValueError so the caller can fall back gracefully.
    """

    _BULLET_PATTERN   = re.compile(r"^[\s]*[-•*]\s",  re.MULTILINE) # noqa
    _NUMBERED_PATTERN = re.compile(r"^[\s]*\d+[.)]\s", re.MULTILINE) # noqa
    _HEADER_PATTERN   = re.compile(r"^#{1,6}\s",       re.MULTILINE)
    _MULTI_Q_PATTERN  = re.compile(r"\?.*\?",          re.DOTALL)

    def validate_and_repair(self, response: str, mode: LLMMode) -> str:
        if mode != LLMMode.INTERVIEWER:
            return response

        text = response.strip()

        if not text:
            _validator_fix.labels(action="empty_reject").inc()
            raise ValueError("LLM returned empty interviewer response")

        text = self._strip_markdown(text)

        words = text.split()
        if len(words) < MIN_RESPONSE_WORDS:
            _validator_fix.labels(action="too_short").inc()
            raise ValueError(f"Response too short ({len(words)} words): {text[:60]}")

        if self._MULTI_Q_PATTERN.search(text):
            text = self._extract_last_question(text)
            _validator_fix.labels(action="multi_q_trimmed").inc()

        words = text.split()
        if len(words) > MAX_RESPONSE_WORDS:
            text = self._truncate_to_question(text)
            _validator_fix.labels(action="truncated").inc()

        if not text.rstrip().endswith("?"):
            _validator_fix.labels(action="no_question_mark").inc()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            questions = [s for s in sentences if s.strip().endswith("?")]
            if questions:
                text = " ".join(sentences[:sentences.index(questions[-1]) + 1])
            else:
                raise ValueError(f"Response contains no question: {text[:80]}")

        return text.strip()

    def _strip_markdown(self, text: str) -> str:
        text = self._BULLET_PATTERN.sub("", text)
        text = self._NUMBERED_PATTERN.sub("", text)
        text = self._HEADER_PATTERN.sub("", text)
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        text = re.sub(r"`[^`]+`", "", text)
        return text.strip()

    def _extract_last_question(self, text: str) -> str: # noqa
        sentences = re.split(r"(?<=[.!?])\s+", text)
        questions = [i for i, s in enumerate(sentences) if s.strip().endswith("?")]
        if questions:
            keep_indices = {0, questions[-1]}
            return " ".join(sentences[i] for i in sorted(keep_indices) if i < len(sentences))
        return text

    def _truncate_to_question(self, text: str) -> str: # noqa
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result    = []
        for s in sentences:
            result.append(s)
            if s.strip().endswith("?"):
                break
        return " ".join(result)


# ── token counter callback ─────────────────────────────────────────────────────

class _TokenCounter(AsyncCallbackHandler):
    def __init__(self, model: str) -> None:
        self._model           = model
        self.prompt_tokens:     int = 0
        self.completion_tokens: int = 0

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        self.prompt_tokens     = usage.get("prompt_tokens",     0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        _tokens.labels(model=self._model, kind="prompt").observe(self.prompt_tokens)
        _tokens.labels(model=self._model, kind="completion").observe(self.completion_tokens)


# ── cache key helpers ──────────────────────────────────────────────────────────

def _interviewer_cache_key(messages: list, model: str) -> str:
    serialised = "|".join(
        f"{type(m).__name__}:{getattr(m, 'content', '')}" for m in messages
    )
    return make_versioned_cache_key(CACHE_PREFIX + "iv:", serialised, model, INTERVIEWER_TEMPERATURE, "")


def _ats_cache_key(intro_text: str, model: str) -> str:
    return make_versioned_cache_key(CACHE_PREFIX + "ats:", intro_text, model, ATS_TEMPERATURE, "")


def _lock_key(cache_key: str) -> str:
    return f"lock:{cache_key}"


# ── shared Redis infrastructure ────────────────────────────────────────────────

class _SharedRedis:
    """
    Shared Redis + LRU cache infrastructure used by all modes.
    One instance per LLMNode — modes do not have independent connections.
    """

    def __init__(self, redis_url: str | None, lru_size: int, cache_ttl: int) -> None:
        self._redis_url = redis_url
        self._cache_ttl = cache_ttl
        self._redis:    aioredis.Redis | None = None
        self._lru       = InMemoryLRU(max_size=lru_size)
        self._breaker   = CircuitBreaker(name="llm_redis", failure_threshold=4, recovery_timeout=15.0)

    async def _conn(self) -> aioredis.Redis | None:
        if not self._redis_url:
            return None
        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding               = "utf-8",
                decode_responses       = True,
                max_connections        = REDIS_MAX_CONN,
                socket_connect_timeout = 0.1,
                socket_timeout         = 0.1,
            )
        return self._redis

    async def cache_get(self, key: str) -> str | None:
        if not degraded_mode.active:
            try:
                r = await self._conn()
                if r:
                    val = await self._breaker.call(r.get, key)
                    if val is not None:
                        _cache_hits.labels(source="redis").inc()
                        return val
                await degraded_mode.exit("redis_ok_llm")
            except Exception as exc:
                log.warning("llm_redis_read_failed", error=str(exc))
                await degraded_mode.enter(f"llm_redis_error: {exc}")

        val = await self._lru.get(key)
        if val is not None:
            _lru_hits.inc()
        return val

    async def cache_set(self, key: str, value: str) -> None:
        await self._lru.set(key, value)
        try:
            r = await self._conn()
            if r:
                if self._cache_ttl > 0:
                    await self._breaker.call(r.setex, key, self._cache_ttl, value)
                else:
                    await self._breaker.call(r.set, key, value)
        except Exception as exc:
            log.warning("llm_cache_write_skipped", error=str(exc))

    async def acquire_lock(self, lock_key: str) -> bool:
        try:
            r = await self._conn()
            if r is None:
                return True
            return bool(await self._breaker.call(r.set, lock_key, "1", nx=True, ex=STAMPEDE_LOCK_TTL))
        except Exception: # noqa
            return True

    async def release_lock(self, lock_key: str) -> None:
        try:
            r = await self._conn()
            if r:
                await self._breaker.call(r.delete, lock_key)
        except Exception: # noqa
            pass

    async def poll_for_cached(self, key: str) -> str | None:
        for _ in range(STAMPEDE_POLL_LIMIT):
            await asyncio.sleep(STAMPEDE_POLL_INTERVAL)
            val = await self.cache_get(key)
            if val is not None:
                return val
        return None

    async def close(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception: # noqa
                pass
            self._redis = None


# ── LangChain chain factory ────────────────────────────────────────────────────

class _ChainFactory:
    """
    Builds and caches LangChain ChatOpenAI chains per (model, mode, streaming).
    Separates interviewer and ATS chains so their settings never bleed together.
    """

    def __init__(self) -> None:
        self._chains: dict[str, Any] = {}

    def get(self, model: str, mode: LLMMode, streaming: bool = False) -> Any:
        key = f"{model}:{mode.value}:{'s' if streaming else 'b'}"
        if key not in self._chains:
            self._chains[key] = self._build(model, mode, streaming)
        return self._chains[key]

    def _build(self, model: str, mode: LLMMode, streaming: bool) -> Any: # noqa
        common = dict(
            model       = model,
            api_key     = OPENAI_API_KEY,
            max_retries = 0,
            timeout     = 60,
        )
        if mode == LLMMode.INTERVIEWER:
            llm = ChatOpenAI(
                **common,
                temperature = INTERVIEWER_TEMPERATURE,
                max_tokens  = INTERVIEWER_MAX_TOKENS,
                streaming   = streaming,
            )
        elif mode == LLMMode.ATS:
            llm = ChatOpenAI(
                **common,
                temperature  = ATS_TEMPERATURE,
                max_tokens   = ATS_MAX_TOKENS,
                streaming    = False,   # ATS never streams
                model_kwargs = {"response_format": {"type": "json_object"}},
            )
        elif mode == LLMMode.EVAL:
            llm = ChatOpenAI(
                **common,
                temperature = EVAL_TEMPERATURE,
                max_tokens  = EVAL_MAX_TOKENS,
                streaming   = streaming,
            )
        else:
            raise ValueError(f"Unknown LLMMode: {mode}") # noqa static checker

        return llm | StrOutputParser()


# ── LLMNodeProtocol ───────────────────────────────────────────────────────────

@runtime_checkable
class LLMNodeProtocol(Protocol):
    """
    Structural protocol satisfied by LLMNode (local) and RemoteLLMClient.
    voice_graph.py and qa_controller depend ONLY on this protocol.
    """

    def stream_question(
        self,
        llm_input:  LLMInterviewInput,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the next interview question tokens."""
        ...

    async def extract_ats(
        self,
        intro_text:  str,
        request_id:  str | None = None,
    ) -> str:
        """Run ATS extraction on intro_text. Returns raw JSON string."""
        ...

    def stream_messages(
        self,
        messages:   list,
        request_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Raw message list streaming. ONLY for evaluation_engine internal use.
        Must NOT be called from voice_graph.node_llm for interview turns.
        """
        ...

    async def health(self) -> ServiceHealthState: ...
    async def close(self)  -> None: ...

    def evict_session(self, session_id: str) -> None: ...


# ── session token budget ───────────────────────────────────────────────────────

@dataclass
class SessionTokenBudget:
    """
    Tracks token usage per session across LLM calls.
    Advisory for interview turns (warn only). Enforced for eval turns.
    """
    session_id:        str
    max_total_tokens:  int   = SESSION_TOKEN_BUDGET
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    warn_threshold:    float = 0.8

    class TokenBudgetExhausted(RuntimeError):
        pass

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_total_tokens - self.total_tokens)

    @property
    def usage_pct(self) -> float:
        return self.total_tokens / max(self.max_total_tokens, 1)

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens     += prompt
        self.completion_tokens += completion

    def check_warn(self, session_id: str) -> None:
        if self.usage_pct >= self.warn_threshold:
            log.warning(
                "session_token_budget_warn",
                session_id = session_id[:8],
                used       = self.total_tokens,
                remaining  = self.remaining,
                pct        = round(self.usage_pct * 100, 1),
            )

    def check_enforce(self, session_id: str) -> None:
        if self.remaining == 0:
            raise SessionTokenBudget.TokenBudgetExhausted(
                f"Session {session_id[:8]} has exhausted its token budget "
                f"({self.total_tokens}/{self.max_total_tokens})"
            )


class _SessionBudgetStore:
    """In-process store for per-session token budgets. LRU-evicted."""

    def __init__(self, max_sessions: int = 256) -> None:
        self._store: OrderedDict[str, SessionTokenBudget] = OrderedDict()
        self._max   = max_sessions

    def get_or_create(self, session_id: str) -> SessionTokenBudget:
        if session_id not in self._store:
            if len(self._store) >= self._max:
                self._store.popitem(last=False)
            self._store[session_id] = SessionTokenBudget(session_id=session_id)
        self._store.move_to_end(session_id)
        return self._store[session_id]

    def record(self, session_id: str, prompt: int, completion: int) -> None:
        budget = self.get_or_create(session_id)
        budget.add(prompt, completion)
        budget.check_warn(session_id)


# ── voice sanitizer ────────────────────────────────────────────────────────────

class _VoiceSanitizer:
    """
    Post-processes LLM text for TTS consumption.
    Called AFTER _ResponseValidator.validate_and_repair().
    """

    _TTS_EXPANSIONS: dict[str, str] = {
        r"\bAPI\b":        "A P I",
        r"\bAPIs\b":       "A P I s",
        r"\bSQL\b":        "S Q L",
        r"\bORM\b":        "O R M",
        r"\bJVM\b":        "J V M",
        r"\bGC\b":         "G C",
        r"\bO\(n\)\b":     "O of n",
        r"\bO\(1\)\b":     "O of one",
        r"\bO\(log n\)\b": "O log n",
        r"\bO\(n²\)\b":    "O of n squared",
        r"\bSTL\b":        "S T L",
        r"\bDSA\b":        "D S A",
        r"\bLRU\b":        "L R U",
        r"\bCPU\b":        "C P U",
        r"\bI/O\b":        "I O",
        r"\bHTTP\b":       "H T T P",
        r"\bHTTPS\b":      "H T T P S",
        r"\bREST\b":       "R E S T",
        r"\bJSON\b":       "J S O N",
        r"\bXML\b":        "X M L",
        r"\bOOP\b":        "O O P",
    }

    _STRIP_PATTERNS: list[tuple[str, str]] = [
        (r"`{1,3}[^`]*`{1,3}",        ""),        # backtick code spans
        (r"\*{1,3}([^*]+)\*{1,3}",    r"\1"),     # bold/italic → plain
        (r"_{1,3}([^_]+)_{1,3}",      r"\1"),     # underscore emphasis → plain
        (r"\.{3,}",                   "..."),     # collapse excessive ellipsis
        (r"-{2,}",                    "—"),       # double dash → em dash
        (r"~{2}([^~]+)~{2}",          r"\1"),     # ~~strikethrough~~ → plain
        (r"\[([^\]]+)\]\([^\)]+\)",   r"\1"),     # [link text](url) → link text
        (r"https?://\S+",             ""),        # strip bare URLs
        (r"^\s*#{1,6}\s+",            ""),        # strip markdown headers
    ]

    def sanitize(self, text: str) -> str:
        result = text
        for pattern, replacement in self._STRIP_PATTERNS:
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        for pattern, expansion in self._TTS_EXPANSIONS.items():
            result = re.sub(pattern, expansion, result)
        result = re.sub(r"  +", " ", result)
        result = re.sub(r"\n{2,}", "\n", result)
        return result.strip()


# ── fallback question bank (simple) ───────────────────────────────────────────

class _FallbackQuestionBank:
    """
    Pre-written fallback questions used when the LLM response fails validation
    AND repair also fails. Ensures the interview never stalls.
    """

    _BANK: dict[str, dict[str, list[str]]] = {
        "python": {
            "beginner":     [
                "Got it. What is the difference between a list and a tuple in Python?",
                "Understood. How does Python manage memory for variables?",
                "Okay. What does the 'self' keyword mean inside a Python class?",
                "Noted. What is the difference between '==' and 'is' in Python?",
            ],
            "intermediate": [
                "Got it. What is a Python generator and how does it differ from a regular function?",
                "Understood. How does the GIL affect multithreaded Python programs?",
                "Okay. Explain Python's descriptor protocol and give an example use case.",
                "Noted. What is the difference between deepcopy and shallow copy in Python?",
            ],
            "advanced": [
                "Got it. How does Python's memory allocator handle small object pooling?",
                "Understood. Explain metaclasses in Python and when you'd use one.",
                "Okay. What is the C3 linearization algorithm and how does Python use it for MRO?",
            ],
        },
        "java": {
            "beginner":     [
                "Got it. What is the difference between an interface and an abstract class in Java?",
                "Understood. How does Java handle memory management?",
                "Okay. What is the difference between '==' and '.equals()' in Java?",
            ],
            "intermediate": [
                "Got it. How does the Java memory model define happens-before relationships?",
                "Understood. What is the difference between HashMap and ConcurrentHashMap?",
                "Okay. Explain how Java's try-with-resources statement works at the bytecode level.",
            ],
            "advanced": [
                "Got it. How does the G1 garbage collector differ from the CMS collector in Java?",
                "Understood. Explain how Java class loading works and what the bootstrap classloader does.",
                "Okay. What is JIT compilation and how does tiered compilation work in the JVM?",
            ],
        },
        "dsa": {
            "beginner":     [
                "Got it. What is the time complexity of binary search and why?",
                "Understood. Explain the difference between a stack and a queue.",
                "Okay. What is a linked list and how does it differ from an array?",
            ],
            "intermediate": [
                "Got it. How does a hash table resolve collisions using chaining versus open addressing?",
                "Understood. What is the difference between BFS and DFS, and when would you use each?",
                "Okay. Explain the concept of dynamic programming with a simple example.",
            ],
            "advanced": [
                "Got it. How does a segment tree support range queries and point updates?",
                "Understood. Explain the amortized complexity of a splay tree's operations.",
                "Okay. What is the difference between Dijkstra's algorithm and A* search?",
            ],
        },
        "system_design": {
            "beginner":     [
                "Got it. What is the difference between horizontal and vertical scaling?",
                "Understood. What is a load balancer and why would you use one?",
                "Okay. What is the CAP theorem and what does it mean for distributed systems?",
            ],
            "intermediate": [
                "Got it. How would you design a rate limiter for a REST API at scale?",
                "Understood. What is eventual consistency and when would you accept it?",
                "Okay. Explain the difference between optimistic and pessimistic locking in databases.",
            ],
            "advanced": [
                "Got it. How would you design a globally distributed key-value store with low write latency?",
                "Understood. Explain how consistent hashing reduces remapping during node additions.",
                "Okay. What is the two-phase commit protocol and what are its failure modes?",
            ],
        },
        "databases": {
            "beginner":     [
                "Got it. What is the difference between a primary key and a foreign key?",
                "Understood. What does ACID stand for in database transactions?",
                "Okay. What is an index and why would you add one to a database table?",
            ],
            "intermediate": [
                "Got it. How does a B-tree index work and why is it preferred over a hash index for range queries?",
                "Understood. What is the difference between INNER JOIN and LEFT JOIN?",
                "Okay. Explain what a database transaction isolation level does and name two common levels.",
            ],
            "advanced": [
                "Got it. How does MVCC implement read consistency without locking in PostgreSQL?",
                "Understood. What is write-ahead logging and how does it support crash recovery?",
                "Okay. Explain the difference between row-store and column-store database architectures.",
            ],
        },
    }

    _DEFAULT: dict[str, list[str]] = {
        "beginner":     ["Got it. Can you explain a technical concept from your introduction that you feel most confident about?"],
        "intermediate": ["Understood. Can you walk me through the most complex technical problem you have solved recently?"],
        "advanced":     ["Noted. Can you describe a production incident you diagnosed and how you traced the root cause?"],
    }

    def get(self, domain: str, level: str) -> str:
        domain_bank = self._BANK.get(domain, self._DEFAULT)
        level_pool  = domain_bank.get(level) or domain_bank.get("intermediate") or list(self._DEFAULT.values())[0]
        return random.choice(level_pool)


# ── health probe ───────────────────────────────────────────────────────────────

class _LLMHealthProbe:
    """Periodic background health probe. Updates liveness gauge."""

    PROBE_INTERVAL_S: int = int(os.getenv("LLM_HEALTH_PROBE_INTERVAL", "60"))

    def __init__(self, node: "LLMNode") -> None:
        self._node    = node
        self._task:   asyncio.Task | None = None
        self._last_ok: float | None       = None

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._run())
            log.info("llm_health_probe_started", interval_s=self.PROBE_INTERVAL_S)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.PROBE_INTERVAL_S)
                health        = await self._node.health()
                self._last_ok = time.time() if health.healthy else None
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("llm_health_probe_error", error=str(exc))

    @property
    def last_ok(self) -> float | None:
        return self._last_ok

    @property
    def stale(self) -> bool:
        if self._last_ok is None:
            return True
        return (time.time() - self._last_ok) > (self.PROBE_INTERVAL_S * 2)


# ── warmup ─────────────────────────────────────────────────────────────────────

class _LLMWarmup:
    """Sends a cheap probe call to each model on startup to eliminate cold-start latency."""

    @staticmethod
    async def warm(node: "LLMNode") -> None:
        probe_messages = [
            SystemMessage(content="You are a terse assistant."),
            HumanMessage(content="Reply with exactly: ready"),
        ]
        models_to_warm = list({node._primary, node._fallback, node._ats_model}) # noqa
        for model in models_to_warm:
            t0 = time.monotonic()
            try:
                chain   = node._chains.get(model, LLMMode.INTERVIEWER, streaming=False) # noqa
                result: str = await asyncio.wait_for(chain.ainvoke(probe_messages), timeout=8.0)
                latency = time.monotonic() - t0
                _latency.labels(model=model, mode="warmup").observe(latency)
                log.info("llm_warmup_ok", model=model, latency=round(latency, 3), reply=result.strip()[:20])
            except asyncio.TimeoutError:
                log.warning("llm_warmup_timeout", model=model)
            except Exception as exc:
                log.warning("llm_warmup_failed", model=model, error=str(exc))


# ── LLMNode ────────────────────────────────────────────────────────────────────

class LLMNode:
    """
    Production LLM node implementing LLMNodeProtocol.

    Three operational modes, each with independent configuration:
      InterviewerMode  — stream_question()  — low temp, tight token cap, validated
      ATSMode          — extract_ats()      — json_object, batch, schema-validated
      EvalMode         — stream_messages()  — used only by evaluation_engine

    Shared infrastructure:
      Redis + LRU dual-layer cache, stampede protection, per-model/per-mode
      circuit breakers, token-bucket rate limiters, bulkhead semaphores,
      OTel spans, Prometheus metrics.

    Built-in post-processing:
      _VoiceSanitizer          — cleans output for TTS
      _FallbackQuestionBank    — used when validator repair fails
      _SessionBudgetStore      — advisory token budget per session
      _LLMHealthProbe          — background liveness probe
    """

    def __init__(
        self,
        primary_model:  str        = PRIMARY_MODEL,
        fallback_model: str        = FALLBACK_MODEL,
        ats_model:      str        = ATS_MODEL,
        ats_fallback:   str        = ATS_FALLBACK,
        redis_url:      str | None = REDIS_URL,
        cache_ttl:      int        = CACHE_TTL,
    ) -> None:
        self._primary      = primary_model
        self._fallback     = fallback_model
        self._ats_model    = ats_model
        self._ats_fallback = ats_fallback

        self._redis         = _SharedRedis(redis_url, LRU_SIZE, cache_ttl)
        self._chains        = _ChainFactory()
        self._validator     = _ResponseValidator()
        self._sanitizer     = _VoiceSanitizer()
        self._fallback_bank = _FallbackQuestionBank()
        self._budget_store  = _SessionBudgetStore()
        self._health_probe  = _LLMHealthProbe(self)

        self._rate_iv  = RateLimiter(INTERVIEWER_RATE,  INTERVIEWER_BURST)
        self._rate_ats = RateLimiter(ATS_RATE,          ATS_BURST)

        self._breakers: dict[str, CircuitBreaker] = {
            f"{primary_model}.interviewer.stream":  CircuitBreaker(name=f"llm:{primary_model}:iv:stream"),
            f"{primary_model}.interviewer.batch":   CircuitBreaker(name=f"llm:{primary_model}:iv:batch"),
            f"{fallback_model}.interviewer.stream": CircuitBreaker(name=f"llm:{fallback_model}:iv:stream"),
            f"{fallback_model}.interviewer.batch":  CircuitBreaker(name=f"llm:{fallback_model}:iv:batch"),
            f"{ats_model}.ats.batch":               CircuitBreaker(name=f"llm:{ats_model}:ats:batch"),
            f"{ats_fallback}.ats.batch":            CircuitBreaker(name=f"llm:{ats_fallback}:ats:batch"),
            f"{primary_model}.eval.stream":         CircuitBreaker(name=f"llm:{primary_model}:eval:stream"),
            f"{fallback_model}.eval.stream":        CircuitBreaker(name=f"llm:{fallback_model}:eval:stream"),
        }

        self._inflight: dict[str, int] = {"interviewer": 0, "ats": 0, "eval": 0}

    # ── breaker helper ─────────────────────────────────────────────────────────

    def _breaker(self, model: str, mode: LLMMode, path: str = "stream") -> CircuitBreaker:
        key = f"{model}.{mode.value}.{path}"
        if key not in self._breakers:
            self._breakers[key] = CircuitBreaker(name=f"llm:{model}:{mode.value}:{path}")
        cb = self._breakers[key]
        _cb_open.labels(model=model, path=f"{mode.value}.{path}").set(1 if cb.state == "OPEN" else 0)
        return cb

    # ── stream question (InterviewerMode) ──────────────────────────────────────

    async def stream_question(
        self,
        llm_input:  LLMInterviewInput,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        INTERVIEWER MODE. The ONLY entry point for interview-turn LLM calls
        from voice_graph.node_llm.

        Flow:
          budget advisory → rate limit → bulkhead → cache → stampede lock →
          primary stream → fallback → accumulate → validate → sanitize →
          fallback bank if needed → cache write → yield final string
        """
        if not llm_input or not llm_input.messages:
            raise ValueError("stream_question: LLMInterviewInput.messages is empty")

        rid      = request_id or new_request_id()
        messages = llm_input.messages

        self._validate_messages(messages, llm_input, rid, session_id)

        key      = _interviewer_cache_key(messages, self._primary)
        lk       = _lock_key(key)

        # ── prompt injection guard (candidate answer) ──
        if INJECTION_BLOCK and llm_input.last_answer:
            detector = PromptInjectionDetector()
            clean_answer, detected = detector.scan(llm_input.last_answer)

            if detected:
                log.warning(
                    "llm_injection_risk",
                    session_id=session_id[:8] if session_id else None,
                    domain=llm_input.domain,
                    level=llm_input.level,
                )

                # Replace candidate answer with sanitized version
                llm_input = dc_replace(llm_input, last_answer=clean_answer)

                # OPTIONAL: hard fail → fallback question
                if BANK_FALLBACK_ENABLED and clean_answer.startswith("[redacted"):
                    fallback = self._fallback_bank.get(llm_input.domain, llm_input.level)
                    yield self._sanitizer.sanitize(fallback)
                    return

        # Advisory budget check
        if session_id:
            self._budget_store.get_or_create(session_id).check_warn(session_id)

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="llm.interviewer")
            except LatencyBudgetExceeded:
                _budget_exc.inc()
                log.warning("llm_iv_budget_exceeded", request_id=rid)
                raise

        with tracer.start_as_current_span("llm.stream_question") as span:
            span.set_attribute("request_id",      rid)
            span.set_attribute("domain",          llm_input.domain)
            span.set_attribute("level",           llm_input.level)
            span.set_attribute("domain_switched", llm_input.domain_switched)

            await self._rate_iv.acquire()

            async with bulkheads.acquire("llm.interviewer"):
                _active.labels(mode="interviewer").inc()
                self._inflight["interviewer"] += 1
                t0          = time.monotonic()
                accumulated: list[str] = []
                model_used  = self._primary
                got_lock    = False

                try:
                    # Cache check
                    cached = await self._redis.cache_get(key)
                    if cached:
                        _cache_hits.labels(source="iv_stream").inc()
                        _req_total.labels(model="cache", status="hit", mode="interviewer").inc()
                        span.set_attribute("cache_hit", True)
                        yield cached
                        return

                    span.set_attribute("cache_hit", False)

                    got_lock = await self._redis.acquire_lock(lk)
                    if not got_lock:
                        polled = await self._redis.poll_for_cached(key)
                        if polled:
                            _cache_hits.labels(source="iv_stampede").inc()
                            yield polled
                            return
                        got_lock = True

                    # Primary stream
                    first_token    = True
                    primary_failed = False
                    primary_err: Exception | None = None

                    try:
                        async for chunk in self._stream_model(self._primary, messages, LLMMode.INTERVIEWER):
                            if budget:
                                budget.check(stage="llm.interviewer.stream")
                            if first_token:
                                _ttft.labels(model=self._primary).observe(time.monotonic() - t0)
                                first_token = False
                            accumulated.append(chunk)
                    except (CircuitBreakerOpen, Exception) as exc:
                        primary_failed = True
                        primary_err    = exc
                        _fallback_used.labels(mode="interviewer").inc()
                        log.warning("llm_iv_primary_failed_fallback", request_id=rid, error=str(exc))

                    # Fallback stream
                    if primary_failed:
                        accumulated.clear()
                        model_used = self._fallback
                        try:
                            async for chunk in self._stream_model(self._fallback, messages, LLMMode.INTERVIEWER):
                                if budget:
                                    budget.check(stage="llm.iv.fallback")
                                if first_token:
                                    _ttft.labels(model=self._fallback).observe(time.monotonic() - t0)
                                    first_token = False
                                accumulated.append(chunk)
                        except Exception as fallback_err:
                            _req_total.labels(model=model_used, status="error", mode="interviewer").inc()
                            span.set_status(StatusCode.ERROR, str(fallback_err))
                            raise RuntimeError(
                                f"Both interviewer models failed. "
                                f"Primary: {primary_err}. Fallback: {fallback_err}"
                            ) from fallback_err

                    full_text = "".join(accumulated)

                    if not full_text.strip():
                        fallback_text = self._fallback_bank.get(llm_input.domain, llm_input.level)
                        sanitized     = self._sanitizer.sanitize(fallback_text)
                        _validator_fix.labels(action="fallback_empty_stream").inc()
                        yield sanitized
                        return

                    # Validate and sanitize
                    try:
                        validated = self._validator.validate_and_repair(full_text, LLMMode.INTERVIEWER)
                    except ValueError as ve:
                        log.warning("llm_iv_validation_failed", error=str(ve), request_id=rid)
                        validated = self._fallback_bank.get(llm_input.domain, llm_input.level)
                        _validator_fix.labels(action="fallback_post_validate").inc()

                    sanitized = self._sanitizer.sanitize(validated)

                    if len(sanitized) >= STREAM_CACHE_MIN:
                        await self._redis.cache_set(key, sanitized)

                    # Approximate token budget tracking
                    if session_id:
                        approx_prompt     = sum(len(getattr(m, "content", "")) // 4 for m in messages)
                        approx_completion = len(sanitized) // 4
                        self._budget_store.record(session_id, approx_prompt, approx_completion)

                    latency = time.monotonic() - t0
                    _req_total.labels(model=model_used, status="ok", mode="interviewer").inc()
                    _latency.labels(model=model_used, mode="interviewer").observe(latency)
                    span.set_attribute("model",   model_used)
                    span.set_attribute("chars",   len(sanitized))
                    span.set_attribute("latency", round(latency, 3))
                    log.info(
                        "llm_iv_ok",
                        request_id = rid,
                        model      = model_used,
                        domain     = llm_input.domain,
                        level      = llm_input.level,
                        chars      = len(sanitized),
                        latency_s  = round(latency, 3),
                    )

                    yield sanitized

                except LatencyBudgetExceeded:
                    _budget_exc.inc()
                    log.warning("llm_iv_budget_mid_stream", request_id=rid)
                    raise
                except asyncio.CancelledError:
                    log.warning("llm_iv_cancelled", request_id=rid)
                    raise
                finally:
                    if got_lock:
                        await self._redis.release_lock(lk)
                    _active.labels(mode="interviewer").dec()
                    self._inflight["interviewer"] -= 1

    def _validate_messages( # noqa
            self,
            messages: list,
            llm_input: LLMInterviewInput,
            request_id: str,
            session_id: str | None,
    ) -> None:
        """
        Validate message list before passing to LLM.
        Raises ValueError if messages are malformed.

        Checks:
          1. List is non-empty
          2. Has at least SystemMessage + HumanMessage
          3. All messages have non-empty content
          4. Message types are valid (SystemMessage, HumanMessage, etc.)
        """
        # Check 1: Non-empty list
        if not messages or len(messages) == 0:
            raise ValueError(
                f"stream_question: messages list is empty (domain={llm_input.domain})"
            )

        # Check 2: Minimum required messages (System + User)
        if len(messages) < 2:
            log.error(
                "llm_invalid_messages_min",
                request_id=request_id,
                session_id=session_id[:8] if session_id else "unknown",
                domain=llm_input.domain,
                message_count=len(messages),
            )
            raise ValueError(
                f"stream_question: need at least 2 messages (System + User), got {len(messages)}"
            )

        # Check 3: All messages have content
        for idx, msg in enumerate(messages):
            # Handle both dict-like and object-like messages
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

            if content is None or not str(content).strip():
                log.error(
                    "llm_empty_message_content",
                    request_id=request_id,
                    session_id=session_id[:8] if session_id else "unknown",
                    domain=llm_input.domain,
                    message_index=idx,
                    message_type=type(msg).__name__,
                )
                raise ValueError(
                    f"stream_question: message[{idx}] has empty content "
                    f"(type={type(msg).__name__})"
                )

        # Check 4: First message should be SystemMessage
        first_msg_type = type(messages[0]).__name__
        if first_msg_type != "SystemMessage":
            log.warning(
                "llm_first_message_not_system",
                request_id=request_id,
                session_id=session_id[:8] if session_id else "unknown",
                domain=llm_input.domain,
                first_message_type=first_msg_type,
            )
            # Don't raise — just warn. Some models may handle this.

        log.debug(
            "llm_messages_valid",
            request_id=request_id,
            session_id=session_id[:8] if session_id else "unknown",
            domain=llm_input.domain,
            message_count=len(messages),
            first_type=first_msg_type,
        )

    # ── extract ATS (ATSMode) ──────────────────────────────────────────────────

    async def extract_ats(
        self,
        intro_text:  str,
        request_id:  str | None = None,
    ) -> str:
        """
        ATS MODE. Batch mode with json_object output.
        Returns raw JSON string — schema validation done by qa_controller.
        """
        if not intro_text or not intro_text.strip():
            raise ValueError("extract_ats: intro_text is empty")

        rid      = request_id or new_request_id()
        key      = _ats_cache_key(intro_text, self._ats_model)
        lk       = _lock_key(key)
        messages = [
            SystemMessage(content=ATS_SYSTEM_PROMPT),
            HumanMessage(content=ATS_USER_TEMPLATE.format(intro_text=intro_text.strip())),
        ]

        with tracer.start_as_current_span("llm.extract_ats") as span:
            span.set_attribute("request_id",  rid)
            span.set_attribute("intro_chars", len(intro_text))

            await self._rate_ats.acquire()

            async with bulkheads.acquire("llm.ats"):
                _active.labels(mode="ats").inc()
                self._inflight["ats"] += 1
                t0       = time.monotonic()
                got_lock = False

                try:
                    cached = await self._redis.cache_get(key)
                    if cached:
                        _cache_hits.labels(source="ats_batch").inc()
                        _req_total.labels(model="cache", status="hit", mode="ats").inc()
                        span.set_attribute("cache_hit", True)
                        return cached

                    span.set_attribute("cache_hit", False)

                    got_lock = await self._redis.acquire_lock(lk)
                    if not got_lock:
                        polled = await self._redis.poll_for_cached(key)
                        if polled:
                            _cache_hits.labels(source="ats_stampede").inc()
                            return polled
                        got_lock = True

                    model_used  = self._ats_model
                    primary_err: Exception | None = None # noqa
                    text        = "" # noqa

                    try:
                        text = await self._batch_model(self._ats_model, messages, LLMMode.ATS, rid)
                    except (CircuitBreakerOpen, Exception) as exc:
                        primary_err = exc
                        _fallback_used.labels(mode="ats").inc()
                        log.warning("llm_ats_primary_failed_fallback", request_id=rid, error=str(exc))
                        try:
                            text       = await self._batch_model(self._ats_fallback, messages, LLMMode.ATS, rid)
                            model_used = self._ats_fallback
                        except Exception as fe:
                            raise RuntimeError(
                                f"Both ATS models failed. Primary: {primary_err}. Fallback: {fe}"
                            ) from fe

                    if not text or not text.strip():
                        raise ValueError(f"ATS model returned empty response (model={model_used})")

                    try:
                        json.loads(text.strip())
                    except json.JSONDecodeError:
                        log.warning("llm_ats_invalid_json", request_id=rid, raw_len=len(text))

                    await self._redis.cache_set(key, text)

                    latency = time.monotonic() - t0
                    _req_total.labels(model=model_used, status="ok", mode="ats").inc()
                    _latency.labels(model=model_used, mode="ats").observe(latency)
                    span.set_attribute("model",   model_used)
                    span.set_attribute("latency", round(latency, 3))
                    log.info("llm_ats_ok", request_id=rid, model=model_used, json_chars=len(text), latency_s=round(latency, 3))
                    return text

                except asyncio.CancelledError:
                    log.warning("llm_ats_cancelled", request_id=rid)
                    raise
                except Exception as exc:
                    _req_total.labels(model=self._ats_model, status="error", mode="ats").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("llm_ats_failed", request_id=rid, error=str(exc))
                    raise
                finally:
                    if got_lock:
                        await self._redis.release_lock(lk)
                    _active.labels(mode="ats").dec()
                    self._inflight["ats"] -= 1

    # ── stream messages (eval + compat) ───────────────────────────────────────

    async def stream_messages(
        self,
        messages:   list,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Raw message list streaming for evaluation_engine.py and legacy paths.
        MUST NOT be called from voice_graph.node_llm for interview turns.
        """
        if not messages:
            raise ValueError("stream_messages: messages list is empty")

        rid        = request_id or new_request_id()
        serialised = "|".join(f"{type(m).__name__}:{getattr(m, 'content', '')}" for m in messages)
        key        = make_versioned_cache_key(CACHE_PREFIX + "ev:", serialised, self._primary, 0.1, "")
        lk         = _lock_key(key)

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="llm.eval.stream_messages")
            except LatencyBudgetExceeded:
                _budget_exc.inc()
                raise

        with tracer.start_as_current_span("llm.stream_messages_eval") as span:
            span.set_attribute("request_id",    rid)
            span.set_attribute("message_count", len(messages))

            await self._rate_iv.acquire()

            async with bulkheads.acquire("llm.eval"):
                _active.labels(mode="eval").inc()
                self._inflight["eval"] += 1
                t0          = time.monotonic()
                accumulated: list[str] = []
                model_used  = self._primary
                got_lock    = False

                try:
                    cached = await self._redis.cache_get(key)
                    if cached:
                        _cache_hits.labels(source="eval_stream").inc()
                        span.set_attribute("cache_hit", True)
                        yield cached
                        return

                    span.set_attribute("cache_hit", False)

                    got_lock = await self._redis.acquire_lock(lk)
                    if not got_lock:
                        polled = await self._redis.poll_for_cached(key)
                        if polled:
                            yield polled
                            return
                        got_lock = True

                    first_token    = True
                    primary_failed = False
                    primary_err: Exception | None = None

                    try:
                        async for chunk in self._stream_model(self._primary, messages, LLMMode.EVAL):
                            if budget:
                                budget.check(stage="llm.eval.before_yield")
                            if first_token:
                                _ttft.labels(model=self._primary).observe(time.monotonic() - t0)
                                first_token = False
                            accumulated.append(chunk)
                            yield chunk
                    except (CircuitBreakerOpen, Exception) as exc:
                        primary_failed = True
                        primary_err    = exc
                        log.warning("llm_eval_primary_failed", request_id=rid, error=str(exc))

                    if primary_failed:
                        accumulated.clear()
                        model_used = self._fallback
                        try:
                            async for chunk in self._stream_model(self._fallback, messages, LLMMode.EVAL):
                                if budget:
                                    budget.check(stage="llm.eval.fallback")
                                if first_token:
                                    _ttft.labels(model=self._fallback).observe(time.monotonic() - t0)
                                    first_token = False
                                accumulated.append(chunk)
                                yield chunk
                        except Exception as fe:
                            _req_total.labels(model=model_used, status="error", mode="eval").inc()
                            raise RuntimeError(
                                f"Both eval models failed. Primary: {primary_err}. Fallback: {fe}"
                            ) from fe

                    full_text = "".join(accumulated)
                    if not full_text.strip():
                        raise ValueError(f"Eval LLM returned empty (model={model_used})")

                    if len(full_text) >= STREAM_CACHE_MIN:
                        await self._redis.cache_set(key, full_text)

                    latency = time.monotonic() - t0
                    _req_total.labels(model=model_used, status="ok", mode="eval").inc()
                    _latency.labels(model=model_used, mode="eval").observe(latency)
                    span.set_attribute("latency", round(latency, 3))

                except LatencyBudgetExceeded:
                    _budget_exc.inc()
                    raise
                except asyncio.CancelledError:
                    raise
                finally:
                    if got_lock:
                        await self._redis.release_lock(lk)
                    _active.labels(mode="eval").dec()
                    self._inflight["eval"] -= 1

    # ── internal model call helpers ────────────────────────────────────────────

    async def _stream_model(
        self,
        model:    str,
        messages: list,
        mode:     LLMMode,
    ) -> AsyncIterator[str]:
        """Yield tokens via LangChain astream(). Applies per-model/per-mode circuit breaker."""
        chain   = self._chains.get(model, mode, streaming=True)
        breaker = self._breaker(model, mode, "stream")

        async def _open() -> AsyncIterator[str]:
            return chain.astream(messages)

        iterator: AsyncIterator[str] = await breaker.call(_open)
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            if hasattr(iterator, "aclose"):
                try:
                    await iterator.aclose()
                except Exception: # noqa
                    pass

    async def _batch_model(
        self,
        model:      str,
        messages:   list,
        mode:       LLMMode,
        request_id: str, # noqa
    ) -> str:
        """Batch (non-streaming) model call. Used ONLY by ATSMode."""
        counter = _TokenCounter(model)
        chain   = self._chains.get(model, mode, streaming=False)
        breaker = self._breaker(model, mode, "batch")

        async def _call() -> str:
            return await chain.ainvoke(messages, config={"callbacks": [counter]})

        return await breaker.call(backoff_retry, _call, attempts=2, base_delay=0.8, exceptions=_LLM_RETRYABLE_LOCAL)

    # ── warmup / probe / lifecycle ─────────────────────────────────────────────

    async def warmup(self) -> None:
        """Warm all configured models on application startup."""
        await _LLMWarmup.warm(self)

    def start_health_probe(self) -> None:
        """Start background health probe loop. Call from lifespan startup."""
        self._health_probe.start()

    def stop_health_probe(self) -> None:
        """Stop background health probe loop. Call from lifespan shutdown."""
        self._health_probe.stop()

    async def health(self) -> ServiceHealthState:
        iv_breaker = self._breakers.get(f"{self._primary}.interviewer.stream")
        cb_state   = iv_breaker.state if iv_breaker else "UNKNOWN"
        redis_ok   = False
        try:
            r = await self._redis._conn() # noqa
            if r:
                await r.ping()
                redis_ok = True
        except Exception: # noqa
            pass

        return ServiceHealthState(
            service       = "llm.local",
            healthy       = cb_state != "OPEN",
            circuit_state = cb_state,
            inflight      = sum(self._inflight.values()),
            degraded      = degraded_mode.active,
            extra         = {
                "primary_model":     self._primary,
                "fallback_model":    self._fallback,
                "ats_model":         self._ats_model,
                "redis_ok":          redis_ok,
                "inflight_by_mode":  dict(self._inflight),
                "health_probe_stale": self._health_probe.stale,
                "health_probe_last":  self._health_probe.last_ok,
            },
        )

    async def close(self) -> None:
        self.stop_health_probe()
        await self._redis.close()
        log.info("llm_node_closed")


# ── prompt injection detector ──────────────────────────────────────────────────

class PromptInjectionDetector:
    """
    Scans raw candidate answer text for prompt injection attempts BEFORE
    it is included in any LLM input. Strips or redacts detected phrases.
    """

    _PATTERNS: list[re.Pattern] = [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|context)", re.I),
        re.compile(r"(you\s+are\s+now|act\s+as|pretend\s+(you\s+are|to\s+be))\s+", re.I),
        re.compile(r"(repeat|reveal|print|output|show)\s+(your\s+)?(system\s+prompt|instructions?)", re.I),
        re.compile(r"\bDAN\b"),
        re.compile(r"jailbreak", re.I),
        re.compile(r"(disregard|forget|override)\s+(all|any)\s+(above|previous|prior|instructions?)", re.I),
        re.compile(r"your\s+new\s+(task|job|role|instruction)\s+is", re.I),
        re.compile(r"<\|im_start\|>|<\|im_end\|>|<system>|</system>|\[INST\]|\[/INST\]", re.I),  # noqa
        re.compile(r"(say|respond\s+with|output\s+only)\s+[\"']?(yes|no|I\s+will|I\s+am)", re.I),
    ]

    _MIN_CLEAN_CHARS = 10

    def scan(self, text: str) -> tuple[str, bool]:
        """
        Returns (clean_text, was_injected).
        Injection phrases are replaced with [REDACTED].
        """
        detected   = False
        clean_text = text

        for pattern in self._PATTERNS:
            if pattern.search(clean_text):
                clean_text = pattern.sub("[REDACTED]", clean_text)
                detected   = True

        if detected:
            _injection_blocks.inc()
            log.warning(
                "llm_injection_detected",
                original_len = len(text),
                clean_len    = len(clean_text),
                sample       = text[:60].replace("\n", " "),
            )
            if len(clean_text.strip()) < self._MIN_CLEAN_CHARS:
                clean_text = "[redacted due to policy violation]"

        return clean_text, detected


# ── content policy filter ──────────────────────────────────────────────────────

class ContentPolicyFilter:
    """
    Post-processing filter on LLM OUTPUTS (interviewer mode only).

    Catches cases where the LLM breaks the strict interviewer persona and
    starts behaving like a chatbot or tutor.
    """

    class ContentPolicyViolation(ValueError):
        """Raised when output violates content policy and cannot be repaired."""

    _WARN_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("filler", re.compile(
            r"^(great|awesome|excellent|wonderful|fantastic|certainly|absolutely|"
            r"of course|sure|ok|okay|no problem|got it|understood)[!.,]?\s*",
            re.I | re.MULTILINE,
        )),
        ("self_ref", re.compile(
            r"\b(I can|I'll|I will|I'd be|let me|I'm going to|I should|I think)\b",
            re.I,
        )),
    ]

    _BLOCK_PATTERNS: list[tuple[str, re.Pattern]] = [
        ("offer_help", re.compile(
            r"\b(I can (explain|help|show|guide|walk you through)|"
            r"let me (explain|help|show|clarify)|"
            r"feel free to ask|don'?t hesitate|happy to help|"
            r"hope that helps|does that make sense|does that answer)\b",
            re.I,
        )),
        ("tutorial", re.compile(
            r"\b(the (answer|solution|reason|explanation) is|"
            r"basically,|in other words,|what this means is|"
            r"to summarize,|to clarify,|in simple terms)\b",
            re.I,
        )),
    ]

    def filter(self, response: str, mode: str = "interviewer") -> str:
        if mode != "interviewer" or not CONTENT_POLICY_BLOCK:
            return response

        text = response.strip()

        for rule, pattern in self._WARN_PATTERNS:
            if pattern.search(text):
                text = pattern.sub("", text).strip()
                _policy_blocks.labels(rule=rule).inc()
                log.debug("llm_policy_warn_stripped", rule=rule)

        for rule, pattern in self._BLOCK_PATTERNS:
            if pattern.search(text):
                _policy_blocks.labels(rule=rule).inc()
                log.warning("llm_policy_block", rule=rule, response=text[:80])
                raise ContentPolicyFilter.ContentPolicyViolation(
                    f"Content policy violation ({rule}): {text[:60]}"
                )

        return text


# ── streaming word guard ───────────────────────────────────────────────────────

class StreamingWordGuard:
    """
    Wraps an async token stream and enforces a hard word count limit IN-STREAM,
    avoiding unnecessary token generation and cost after the cap is reached.
    """

    def __init__(self, max_words: int = MAX_RESPONSE_WORDS) -> None:
        self._max_words  = max_words
        self._word_count = 0
        self._buffer     = ""
        self._cut        = False

    async def wrap(self, source: AsyncIterator[str]) -> AsyncIterator[str]:
        async for chunk in source:
            if self._cut:
                break
            self._buffer += chunk
            if len(self._buffer.split()) >= self._max_words:
                truncated = self._truncate_to_last_question(self._buffer)
                if truncated:
                    yield truncated
                _stream_word_cuts.inc()
                self._cut = True
                break
            else:
                yield chunk
                self._word_count += len(chunk.split())

    @staticmethod
    def _truncate_to_last_question(text: str) -> str:
        idx = text.rfind("?")
        if idx >= 0:
            return text[: idx + 1].strip()
        return text.strip()


# ── question diversity enforcer ────────────────────────────────────────────────

class QuestionDiversityEnforcer:
    """
    Character-level N-gram cosine similarity check on generated questions
    against recent questions in the same session/domain. Forces regeneration
    if similarity exceeds DIVERSITY_THRESHOLD (up to MAX_DIVERSITY_RETRIES).
    """

    def __init__(self, n: int = DIVERSITY_NGRAM_N, window: int = DIVERSITY_WINDOW) -> None:
        self._n      = n
        self._window = window
        self._recent: dict[str, list[str]] = defaultdict(list)

    def _ngrams(self, text: str) -> Counter:
        text   = re.sub(r"\s+", " ", text.lower().strip())
        return Counter(text[i: i + self._n] for i in range(len(text) - self._n + 1))

    def _cosine(self, a: Counter, b: Counter) -> float: # noqa
        if not a or not b:
            return 0.0
        intersection = sum((a & b).values())
        mag_a        = math.sqrt(sum(v * v for v in a.values()))
        mag_b        = math.sqrt(sum(v * v for v in b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return intersection / (mag_a * mag_b)

    def is_too_similar(self, session_id: str, domain: str, question: str) -> tuple[bool, float]:
        key     = f"{session_id}:{domain}"
        recent  = self._recent.get(key, [])[-self._window:]
        q_ng    = self._ngrams(question)
        max_sim = max((self._cosine(q_ng, self._ngrams(p)) for p in recent), default=0.0)
        _diversity_similarity.observe(max_sim)
        return max_sim >= DIVERSITY_THRESHOLD, max_sim

    def register(self, session_id: str, domain: str, question: str) -> None:
        key = f"{session_id}:{domain}"
        self._recent[key].append(question)
        if len(self._recent[key]) > 50:
            self._recent[key] = self._recent[key][-50:]

    def evict_session(self, session_id: str) -> None:
        for k in [k for k in self._recent if k.startswith(f"{session_id}:")]:
            del self._recent[k]


# ── static question bank ───────────────────────────────────────────────────────

class StaticQuestionBank:
    """
    Pre-classified domain × level × difficulty question bank (~300+ questions).
    Priority 3 fallback when all LLM paths are exhausted.
    Tracks served questions per session to avoid repeats.
    """

    _BANK: dict[str, dict[str, dict[str, list[str]]]] = {
        "python": {
            "beginner": {
                "easy": [
                    "What is the difference between a list and a tuple in Python?",
                    "How do you define a function in Python?",
                    "What does the `len()` function do in Python?",
                    "What is an f-string and how do you use one?",
                    "How do you handle a divide-by-zero error in Python?",
                ],
                "medium": [
                    "What is the difference between `__init__` and `__new__` in Python?",
                    "How does Python's garbage collector work?",
                    "What are list comprehensions and when would you use one?",
                    "What is the purpose of `*args` and `**kwargs`?",
                ],
            },
            "intermediate": {
                "easy": [
                    "What is a Python decorator and what problem does it solve?",
                    "Explain the difference between `is` and `==` in Python.",
                    "What is the Global Interpreter Lock and why does it matter?",
                ],
                "medium": [
                    "How does Python's `asyncio` event loop work?",
                    "What is the difference between a generator and an iterator?",
                    "How would you implement a context manager using `__enter__` and `__exit__`?",
                    "What are metaclasses in Python and when would you use them?",
                ],
                "hard": [
                    "Explain how Python's `__slots__` affects memory layout and attribute access.",
                    "What is the MRO and how does C3 linearization resolve diamond inheritance?",
                    "How does `functools.lru_cache` work internally?",
                ],
            },
            "advanced": {
                "hard": [
                    "How would you implement a thread-safe singleton in Python?",
                    "Explain the tradeoffs between multiprocessing and asyncio for CPU-bound tasks.",
                    "How do Python descriptors work and when would you use `__get__`?",
                ],
                "expert": [
                    "Walk me through how CPython compiles source to bytecode.",
                    "How would you implement a coroutine scheduler from scratch?",
                    "What are the memory implications of circular references with weakrefs?",
                ],
            },
        },
        "java": {
            "beginner": {
                "easy": [
                    "What is the difference between `==` and `.equals()` in Java?",
                    "What is an interface in Java and how is it different from an abstract class?",
                    "What does the `final` keyword mean when applied to a variable?",
                    "What is autoboxing in Java?",
                ],
                "medium": [
                    "How does Java's garbage collector decide which objects to collect?",
                    "What is the difference between `ArrayList` and `LinkedList`?",
                    "How does exception handling work in Java — checked vs unchecked?",
                ],
            },
            "intermediate": {
                "medium": [
                    "What is the Java Memory Model and why does it matter for concurrency?",
                    "How do Java generics work and what is type erasure?",
                    "Explain the difference between `volatile`, `synchronized`, and `AtomicInteger`.",
                    "What is the difference between `Callable` and `Runnable`?",
                ],
                "hard": [
                    "How does the Fork/Join framework work in Java?",
                    "Explain the happens-before guarantee in the Java Memory Model.",
                    "How would you implement a lock-free stack using `compareAndSet`?",
                ],
            },
            "advanced": {
                "hard": [
                    "What are Java's virtual threads and how do they differ from platform threads?",
                    "How does G1GC differ from ZGC in Java garbage collection?",
                ],
                "expert": [
                    "Walk me through how the JIT compiler performs inlining and escape analysis.",
                    "How would you design a thread pool with work-stealing semantics?",
                ],
            },
        },
        "cpp": {
            "beginner": {
                "easy": [
                    "What is the difference between a pointer and a reference in C++?",
                    "What does the `const` keyword do in C++?",
                    "What is a destructor and when is it called?",
                ],
                "medium": [
                    "What is RAII and how does it prevent resource leaks?",
                    "What is the Rule of Three in C++?",
                    "What is the difference between `new`/`delete` and `malloc`/`free`?",
                ],
            },
            "intermediate": {
                "medium": [
                    "What are move semantics in C++ and what problem do they solve?",
                    "What is a virtual table (vtable) and how does it implement polymorphism?",
                    "What is template specialization and partial specialization?",
                ],
                "hard": [
                    "Explain the difference between `std::unique_ptr` and `std::shared_ptr`.",
                    "What is undefined behaviour in C++ and give three common examples.",
                    "How does `std::move` work and what does it actually do to the source object?",
                ],
            },
            "advanced": {
                "hard": [
                    "What is SFINAE and how is it used for compile-time dispatch?",
                    "How does `std::atomic` prevent data races without using a mutex?",
                ],
                "expert": [
                    "Walk me through how the compiler handles a virtual function call.",
                    "What are the memory ordering models in C++11 and when would you use `memory_order_acquire`?",
                ],
            },
        },
        "javascript": {
            "beginner": {
                "easy": [
                    "What is the difference between `let`, `const`, and `var`?",
                    "What is a closure in JavaScript?",
                    "What does `===` check that `==` does not?",
                    "What is the event loop in JavaScript?",
                ],
                "medium": [
                    "What is prototypal inheritance and how does it differ from class-based?",
                    "What is the difference between `null` and `undefined`?",
                ],
            },
            "intermediate": {
                "medium": [
                    "How does the JavaScript event loop handle microtasks vs macrotasks?",
                    "What is the difference between a Promise and async/await?",
                    "How does `this` binding work in arrow functions vs regular functions?",
                ],
                "hard": [
                    "What is a WeakMap and when would you use it instead of a Map?",
                    "How does V8's hidden class optimization work?",
                    "What are generators in JavaScript and how do they differ from async functions?",
                ],
            },
            "advanced": {
                "hard": [
                    "How does Node.js achieve non-blocking I/O with a single thread?",
                    "Explain the difference between `Object.freeze` and `const`.",
                ],
                "expert": [
                    "How would you implement a job queue with backpressure in Node.js?",
                    "Walk me through how V8 compiles JavaScript to machine code via Ignition and TurboFan.",
                ],
            },
        },
        "dsa": {
            "beginner": {
                "easy": [
                    "What is the time complexity of searching in a sorted array using binary search?",
                    "What is the difference between a stack and a queue?",
                    "What is a hash table and how does it achieve O(1) average lookup?",
                ],
                "medium": [
                    "What is the difference between BFS and DFS, and when would you use each?",
                    "How does a min-heap guarantee the minimum element is at the root?",
                ],
            },
            "intermediate": {
                "medium": [
                    "What is dynamic programming and how do you identify overlapping subproblems?",
                    "Explain the difference between a B-tree and a binary search tree.",
                    "How does quicksort choose a pivot and what is its worst-case complexity?",
                ],
                "hard": [
                    "What is the difference between Dijkstra's and Bellman-Ford algorithms?",
                    "How would you detect a cycle in a directed graph?",
                    "What is the time complexity of building a heap using heapify?",
                ],
            },
            "advanced": {
                "hard": [
                    "How does a segment tree support range queries in O(log n)?",
                    "What is amortized analysis and how does it apply to a dynamic array?",
                ],
                "expert": [
                    "Explain van Emde Boas trees and their advantage over balanced BSTs.",
                    "How would you design a data structure for LFU cache eviction in O(1)?",
                ],
            },
        },
        "system_design": {
            "intermediate": {
                "medium": [
                    "What is the CAP theorem and how does it apply to distributed databases?",
                    "What is the difference between horizontal and vertical scaling?",
                    "How does consistent hashing minimize reshuffling when nodes are added?",
                ],
                "hard": [
                    "How would you design a URL shortener that handles 10,000 requests per second?",
                    "What is event sourcing and how does it differ from CRUD?",
                    "How does a write-ahead log guarantee durability in databases?",
                ],
            },
            "advanced": {
                "hard": [
                    "How would you design a distributed rate limiter across multiple data centers?",
                    "What is the two-phase commit protocol and why is it slow?",
                ],
                "expert": [
                    "How does Google Spanner achieve external consistency without NTP?",
                    "Walk me through designing a globally distributed key-value store with strong consistency.",
                ],
            },
        },
        "databases": {
            "beginner": {
                "easy": [
                    "What is the difference between a primary key and a foreign key?",
                    "What is an index in a database and why does it speed up queries?",
                    "What does ACID stand for in database transactions?",
                ],
            },
            "intermediate": {
                "medium": [
                    "What is the difference between an inner join and an outer join?",
                    "What is database normalization and what problem does it solve?",
                    "How does a B+ tree index differ from a hash index?",
                ],
                "hard": [
                    "How does MVCC (multi-version concurrency control) work?",
                    "What is the difference between optimistic and pessimistic locking?",
                    "How would you diagnose and fix N+1 query problems in an ORM?",
                ],
            },
            "advanced": {
                "expert": [
                    "How does PostgreSQL's VACUUM process handle dead tuples?",
                    "Walk me through how a distributed SQL database handles a two-phase commit.",
                ],
            },
        },
        "os_concepts": {
            "intermediate": {
                "medium": [
                    "What is the difference between a process and a thread?",
                    "How does virtual memory allow a process to use more memory than physically available?",
                    "What is a deadlock and what are the four conditions required for one?",
                ],
                "hard": [
                    "How does the kernel handle a page fault?",
                    "What is the difference between a mutex and a semaphore?",
                    "How does copy-on-write work in the context of `fork()`?",
                ],
            },
            "advanced": {
                "expert": [
                    "How does the Linux CFS scheduler prioritize processes?",
                    "Walk me through what happens when a user process calls `read()` on a file.",
                ],
            },
        },
    }

    _GENERIC_QUESTIONS: list[str] = [
        "What is the most significant challenge you've faced in this domain and how did you solve it?",
        "Describe a production bug related to this technology and how you diagnosed it.",
        "How do you stay current with changes in this technology area?",
        "What is the most important thing a developer should understand before using this in production?",
        "Describe a tradeoff you made in a project using this technology.",
    ]

    def __init__(self) -> None:
        self._served: dict[str, set[tuple]] = defaultdict(set)

    def get_question(self, session_id: str, domain: str, level: str, difficulty: str) -> str:
        q = self._try_get(session_id, domain, level, difficulty)
        if q:
            _bank_fallbacks.labels(domain=domain, level=level).inc()
            return q
        for diff_alt in ("medium", "hard", "easy", "expert"):
            if diff_alt == difficulty:
                continue
            q = self._try_get(session_id, domain, level, diff_alt)
            if q:
                _bank_fallbacks.labels(domain=domain, level=level).inc()
                return q
        for level_alt in ("intermediate", "beginner", "advanced"):
            if level_alt == level:
                continue
            q = self._try_get(session_id, domain, level_alt, "medium")
            if q:
                _bank_fallbacks.labels(domain=domain, level=level).inc()
                return q
        # Count how many generic questions have already been served this session
        # so the index advances on each call rather than being stuck at the same slot.
        served_generics = sum(1 for k in self._served[session_id] if len(k) == 4 and k[0] == "__generic__")
        generic_idx = served_generics % len(self._GENERIC_QUESTIONS)
        self._served[session_id].add(("__generic__", "", "", served_generics))
        _bank_fallbacks.labels(domain="generic", level="any").inc()
        return self._GENERIC_QUESTIONS[generic_idx]

    def _try_get(self, session_id: str, domain: str, level: str, difficulty: str) -> str | None:
        questions = self._BANK.get(domain, {}).get(level, {}).get(difficulty, [])
        for i, q in enumerate(questions):
            idx_key = (domain, level, difficulty, i)
            if idx_key not in self._served[session_id]:
                self._served[session_id].add(idx_key)
                return q
        return None

    def evict_session(self, session_id: str) -> None:
        self._served.pop(session_id, None)


# ── ATS schema validator ───────────────────────────────────────────────────────

if _PYDANTIC_AVAILABLE:
    class ATSExtractionSchema(BaseModel):
        """Pydantic v2 schema for ATS LLM output validation."""

        name:      str
        domains:   list[str]
        level:     str
        languages: list[str]
        notes:     str

        _VALID_DOMAINS: ClassVar[frozenset[str]] = frozenset({
            "python", "java", "cpp", "csharp", "javascript", "php",
            "golang", "rust", "dsa", "system_design", "databases",
            "os_concepts", "behavioral", "ml",
        })
        _VALID_LEVELS: ClassVar[frozenset[str]] = frozenset({
            "beginner", "intermediate", "advanced"
        })

        @field_validator("domains") # noqa
        @classmethod
        def validate_domains(cls, v: list[str]) -> list[str]:
            valid = [d for d in v if d in cls._VALID_DOMAINS]
            if not valid:
                raise ValueError(f"No valid domains in: {v}")
            return valid

        @field_validator("level") # noqa
        @classmethod
        def validate_level(cls, v: str) -> str:
            return v if v in cls._VALID_LEVELS else "intermediate"

        @field_validator("name") # noqa
        @classmethod
        def truncate_name(cls, v: str) -> str:
            return v[:64]


def validate_ats_json(raw: str) -> dict | None:
    """
    Parse and validate ATS extraction JSON.
    Returns the validated dict on success, None on failure.
    """
    try:
        data = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        _ats_schema_fails.inc()
        log.warning("llm_ats_json_parse_failed", error=str(exc), raw=raw[:80])
        return None

    if not _PYDANTIC_AVAILABLE:
        return data

    try:
        return ATSExtractionSchema(**data).model_dump()
    except (ValidationError, Exception) as exc:
        _ats_schema_fails.inc()
        log.warning("llm_ats_schema_validation_failed", error=str(exc))
        return None


# ── eval mode LLM ──────────────────────────────────────────────────────────────

class EvalModeLLM:
    """
    Dedicated LLM wrapper for evaluation_engine.py internal use ONLY.

    Wider token budget, higher temperature, no interviewer guardrails.
    voice_graph.node_llm MUST NOT call this class under any circumstances.
    """

    def __init__(self, llm_node_ref: LLMNode) -> None:
        self._node    = llm_node_ref
        self._breaker = CircuitBreaker(name="llm_eval_mode", failure_threshold=5, recovery_timeout=30.0)
        self._sem     = asyncio.Semaphore(3)

    async def generate(self, messages: list, request_id: str | None = None) -> str:
        """Non-streaming eval call. Returns full response text."""
        rid = request_id or new_request_id()
        t0  = time.monotonic()

        with tracer.start_as_current_span("llm.eval_mode.generate") as span:
            span.set_attribute("request_id", rid)

            async with self._sem:
                _eval_calls.labels(status="attempt").inc()
                try:
                    chain  = self._node._chains.get(EVAL_MODEL, LLMMode.EVAL, streaming=False) # noqa
                    result = await self._breaker.call(chain.ainvoke, messages)

                    latency = time.monotonic() - t0
                    _eval_latency.observe(latency)
                    _eval_calls.labels(status="ok").inc()
                    span.set_attribute("latency_s", round(latency, 3))
                    log.info("llm_eval_generate_ok", request_id=rid, latency_ms=round(latency * 1000, 1), chars=len(result))
                    return result

                except Exception as exc:
                    _eval_calls.labels(status="error").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("llm_eval_generate_failed", request_id=rid, error=str(exc))
                    raise

    async def stream(self, messages: list, request_id: str | None = None) -> AsyncIterator[str]:
        """Streaming eval call for long-form evaluation responses."""
        rid = request_id or new_request_id()
        with tracer.start_as_current_span("llm.eval_mode.stream") as span:
            span.set_attribute("request_id", rid)
            async with self._sem:
                _eval_calls.labels(status="stream_attempt").inc()
                try:
                    result     = await self.generate(messages, request_id=rid)
                    words      = result.split()
                    chunk_size = 5
                    for i in range(0, len(words), chunk_size):
                        yield " ".join(words[i: i + chunk_size]) + " "
                    _eval_calls.labels(status="stream_ok").inc()
                except Exception as exc:
                    _eval_calls.labels(status="stream_error").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    raise


# ── batch ATS processor ────────────────────────────────────────────────────────

class BatchATSProcessor:
    """
    Offline ATS processing for bulk intro extraction.
    NOT used by the live voice graph.
    """

    def __init__(self, llm_node_ref: LLMNode) -> None:
        self._node = llm_node_ref
        self._sem  = asyncio.Semaphore(BATCH_ATS_CONCURRENCY)

    async def process_batch(
        self,
        items:       list[dict],
        on_progress: Any | None = None,
    ) -> list[dict]:
        t0      = time.monotonic()
        total   = len(items)
        results: list[dict] = [{} for _ in range(total)]
        done    = 0

        async def process_one(index: int, item: dict) -> None:
            nonlocal done
            cid        = item.get("candidate_id", str(index))
            intro_text = item.get("intro_text", "")

            async with self._sem:
                try:
                    raw_json  = await asyncio.wait_for(
                        self._node.extract_ats(intro_text), timeout=BATCH_ATS_TIMEOUT
                    )
                    validated = validate_ats_json(raw_json)
                    results[index] = {"candidate_id": cid, "result": validated, "error": None}
                    _batch_ats_calls.labels(status="ok").inc()
                except asyncio.TimeoutError:
                    results[index] = {"candidate_id": cid, "result": None, "error": "timeout"}
                    _batch_ats_calls.labels(status="timeout").inc()
                except Exception as exc:
                    results[index] = {"candidate_id": cid, "result": None, "error": str(exc)}
                    _batch_ats_calls.labels(status="error").inc()
                finally:
                    done += 1
                    if on_progress:
                        try:
                            on_progress(done, total)
                        except Exception: # noqa
                            pass

        await asyncio.gather(*(process_one(i, item) for i, item in enumerate(items)))

        elapsed_minutes = (time.monotonic() - t0) / 60
        if elapsed_minutes > 0:
            _batch_ats_throughput.observe(total / elapsed_minutes)

        ok_count    = sum(1 for r in results if r.get("error") is None)
        error_count = sum(1 for r in results if r.get("error") is not None)
        log.info("llm_batch_ats_complete", total=total, ok=ok_count, errors=error_count)
        return results


# ── LLMNodeV2 ─────────────────────────────────────────────────────────────────

class LLMNodeV2(LLMNode):
    """
    Production LLMNode v2 — extends LLMNode with all security and quality guards:
      - PromptInjectionDetector on all candidate answer text
      - ContentPolicyFilter on all interviewer responses
      - QuestionDiversityEnforcer with N-gram cosine similarity
      - StaticQuestionBank as Priority 3 fallback
      - ATSSchemaValidator replacing manual JSON checks
      - StreamingWordGuard cutting generation in-stream at the word cap
      - EvalModeLLM for evaluation_engine use
      - BatchATSProcessor for offline screening

    CRITICAL: Injection detection and content policy filtering are always active
    and cannot be bypassed per-request.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._injection = PromptInjectionDetector()
        self._policy    = ContentPolicyFilter()
        self._diversity = QuestionDiversityEnforcer()
        self._bank      = StaticQuestionBank()
        self.eval_mode  = EvalModeLLM(self)
        self.batch_ats  = BatchATSProcessor(self)

    async def stream_question(
        self,
        llm_input:  LLMInterviewInput,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Extended stream_question with ALL guardrails active:

          1. Injection scan on llm_input.last_answer
          2. Parent stream_question (rate limit → cache → LLM → validate → sanitize)
          3. ContentPolicyFilter on output
          4. QuestionDiversityEnforcer — retry up to MAX_DIVERSITY_RETRIES
          5. StaticQuestionBank if all LLM paths exhausted
          6. DiversityEnforcer registration on the final question
        """
        rid = request_id or new_request_id()

        # ── Step 1: Injection scan ─────────────────────────────────────────────
        if llm_input and llm_input.last_answer:
            clean_answer, was_injected = self._injection.scan(llm_input.last_answer)
            if was_injected:
                new_messages = []
                for msg in llm_input.messages:
                    if hasattr(msg, "content") and llm_input.last_answer in msg.content:
                        new_content = msg.content.replace(llm_input.last_answer, clean_answer)
                        new_messages.append(type(msg)(content=new_content))
                    else:
                        new_messages.append(msg)
                llm_input = dc_replace(llm_input, last_answer=clean_answer, messages=new_messages)

        domain   = getattr(llm_input, "domain",     "")             if llm_input else ""
        level    = getattr(llm_input, "level",       "intermediate") if llm_input else "intermediate"
        diff_val = getattr(llm_input, "difficulty",  "medium")       if llm_input else "medium"
        diff_str = diff_val.value if hasattr(diff_val, "value") else str(diff_val)

        final_question: str | None = None

        # ── Steps 2–4: Generate with diversity retries ─────────────────────────
        for attempt in range(MAX_DIVERSITY_RETRIES + 1):
            try:
                chunks: list[str] = []
                guard = StreamingWordGuard(max_words=MAX_RESPONSE_WORDS)

                async for chunk in guard.wrap(
                    super().stream_question(llm_input, request_id=rid, session_id=session_id)
                ):
                    chunks.append(chunk)

                candidate_q = "".join(chunks).strip()

                # Content policy filter
                try:
                    candidate_q = self._policy.filter(candidate_q, mode="interviewer")
                except ContentPolicyFilter.ContentPolicyViolation as exc:
                    log.warning("llm_content_policy_blocked", attempt=attempt, error=str(exc))
                    if attempt < MAX_DIVERSITY_RETRIES:
                        _diversity_retries.labels(result="policy_retry").inc()
                        continue
                    break

                if not candidate_q:
                    continue

                # Diversity check
                if session_id:
                    too_similar, sim_score = self._diversity.is_too_similar(session_id, domain, candidate_q)
                    if too_similar and attempt < MAX_DIVERSITY_RETRIES:
                        _diversity_retries.labels(result="diversity_retry").inc()
                        log.debug(
                            "llm_diversity_retry",
                            session_id = session_id[:8],
                            domain     = domain,
                            similarity = round(sim_score, 3),
                            attempt    = attempt,
                        )
                        continue

                final_question = candidate_q
                _diversity_retries.labels(result="ok").inc()
                break

            except (StopAsyncIteration, Exception) as exc:
                if attempt < MAX_DIVERSITY_RETRIES:
                    _diversity_retries.labels(result="error_retry").inc()
                    log.debug("llm_diversity_attempt_failed", attempt=attempt, error=str(exc))
                    continue
                break

        # ── Step 5: Static bank fallback ───────────────────────────────────────
        if not final_question:
            if BANK_FALLBACK_ENABLED:
                final_question = self._bank.get_question(
                    session_id = session_id or "",
                    domain     = domain,
                    level      = level,
                    difficulty = diff_str,
                )
                log.warning(
                    "llm_bank_fallback_used",
                    session_id = (session_id or "")[:8],
                    domain     = domain,
                    level      = level,
                )
            else:
                raise ValueError(
                    f"LLMNodeV2: all question generation paths exhausted for domain={domain}"
                )

        # ── Step 6: Register with diversity enforcer ───────────────────────────
        if session_id and final_question:
            self._diversity.register(session_id, domain, final_question)

        yield final_question

    async def extract_ats(self, intro_text: str, request_id: str | None = None) -> str:
        """
        Extended extract_ats with ATSSchemaValidator.
        Falls back to ATSRuleExtractor on schema failure.
        """
        raw       = await super().extract_ats(intro_text, request_id=request_id)
        validated = validate_ats_json(raw)

        if validated is None:
            log.warning("llm_ats_schema_fallback_triggered")
            from app.interview.qa_controller import ATSRuleExtractor
            rule_result = ATSRuleExtractor().extract(intro_text)
            return json.dumps({
                "name":      rule_result.name,
                "domains":   rule_result.domains,
                "level":     rule_result.level,
                "languages": rule_result.languages,
                "notes":     rule_result.notes,
                "_fallback": "rule_based",
            })

        return json.dumps(validated)

    def evict_session(self, session_id: str) -> None:
        """Clean up session-specific in-process state. Call on session close."""
        self._diversity.evict_session(session_id)
        self._bank.evict_session(session_id)

    async def health(self) -> ServiceHealthState:
        base = await super().health()
        base.extra.update({
            "injection_detection": INJECTION_BLOCK,
            "content_policy":      CONTENT_POLICY_BLOCK,
            "diversity_threshold": DIVERSITY_THRESHOLD,
            "bank_fallback":       BANK_FALLBACK_ENABLED,
            "eval_model":          EVAL_MODEL,
            "bank_domains":        list(self._bank._BANK.keys()), # noqa
        })
        return base

    async def close(self) -> None:
        await super().close()
        log.info("llm_node_v2_closed")


# ═══════════════════════════════════════════════════════════════════════════════
# REMOTE LLM CLIENT
# ═══════════════════════════════════════════════════════════════════════════════


class RemoteLLMClient:
    """
    HTTP client implementing LLMNodeProtocol against a remote LLM service.
    Drop-in replacement for LLMNodeV2 in any voice_graph configuration.

    Expected remote endpoints
    ─────────────────────────
      POST /stream_question  → SSE stream of interview question tokens
      POST /extract_ats      → JSON: {"result": "<json string>"}
      POST /stream_messages  → SSE stream of evaluation/raw tokens
      GET  /health           → {"healthy": bool, "circuit_state": str, ...}

    SSE framing
    ───────────
    All streaming endpoints use standard SSE framing:
      data: <token>\n\n   — yield this token
      data: [DONE]\n\n    — stream complete
      event: error\n      — remote signalled an error; logged and raised

    Observability
    ─────────────
    Every public method participates in the module-level Prometheus counters
    (_req_total, _latency, _ttft, _budget_exc, _active, _cb_open) so
    dashboards and alerts are identical whether the node is local or remote.
    OTel spans are opened per call and trace headers propagated outbound
    so the remote service joins the same distributed trace as voice_graph.

    Latency budget
    ──────────────
    LatencyBudget.current() is checked at entry to every streaming method.
    The remaining budget is forwarded as X-Latency-Budget-Ms so the remote
    can self-abort rather than producing a result voice_graph will discard.
    Per-token budget checks run inside every streaming loop so mid-stream
    SLA violations abort immediately rather than waiting for [DONE].

    Resilience
    ──────────
    Single CircuitBreaker ("llm.remote") wraps all three methods — a remote
    LLM outage trips the breaker regardless of which method surfaced the
    failure.  All streaming calls use backoff_retry (2 attempts, 1.0 s base)
    inside the breaker so transient 5xx errors are retried before tripping.
    extract_ats() uses 3 attempts with 1.5 s base (mirrors local ATSMode).

    Cancellation
    ────────────
    asyncio.CancelledError is always re-raised from all call sites.
    GeneratorExit is caught inside streaming yield sites and logged at
    DEBUG — not as an error — so caller drop-out cleans up silently.

    evict_session
    ─────────────
    No-op with a debug log — remote manages its own session state.
    Callers that call evict_session() on an LLMNodeProtocol reference work
    correctly regardless of which implementation is behind the protocol.

    warmup / health probe stubs
    ───────────────────────────
    warmup(), start_health_probe(), stop_health_probe() are provided as
    no-ops so the voice_graph on_startup / on_shutdown hooks work without
    isinstance checks when a RemoteLLMClient is injected.

    Env config (resolved at module load)
    ─────────────────────────────────────
      LLM_SERVICE_URL      required — base URL of the remote LLM service
      LLM_SERVICE_API_KEY  optional — Bearer token
      LLM_SERVICE_TIMEOUT  optional — per-request timeout in seconds (default 60)
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        base_url: str   = LLM_SERVICE_URL,
        api_key:  str   = LLM_SERVICE_API_KEY,
        timeout:  float = LLM_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("LLM_SERVICE_URL must be set to use RemoteLLMClient.")

        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        self._timeout  = timeout
        self._breaker  = CircuitBreaker(name="llm.remote")

        # Per-mode inflight counters — mirroring LLMNode._inflight dict so
        # health() can report the same granularity.
        self._inflight: dict[str, int] = {
            "interviewer": 0,
            "ats":         0,
            "eval":        0,
        }

        # Auth baked into the client so it is never omitted per-call.
        base_headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            base_headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=base_headers,
            http2=True,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _request_headers(self, rid: str, accept_sse: bool = False) -> dict[str, str]: # noqa
        """
        Build per-request headers.

        Injects:
          X-Request-Id          — propagated to remote for correlated logging
          X-Latency-Budget-Ms   — remaining SLA; remote uses this to self-abort
          Accept                — "text/event-stream" for SSE endpoints
          OTel trace headers    — remote joins the same distributed trace
        """
        headers: dict[str, str] = {"X-Request-Id": rid}
        if accept_sse:
            headers["Accept"] = "text/event-stream"
        inject_trace_headers(headers)

        budget = LatencyBudget.current()
        if budget:
            headers["X-Latency-Budget-Ms"] = budget.as_header_value()

        return headers

    def _check_budget(self, stage: str) -> None: # noqa
        """Raise LatencyBudgetExceeded if the pipeline SLA is already blown."""
        budget = LatencyBudget.current()
        if budget:
            budget.check(stage=stage)

    def _parse_sse_line(self, raw_line: str) -> str | None: # noqa
        """
        Parse a single SSE line.

        Returns:
          token string   — for "data: <token>" lines
          None           — for blank lines, "data: [DONE]", and event lines
        Raises:
          RuntimeError   — for "event: error" lines (remote signalled failure)
        """
        line = raw_line.strip()
        if not line:
            return None

        if line.startswith("event:"):
            event = line[len("event:"):].strip()
            if event == "error":
                raise RuntimeError("Remote LLM service signalled an error event")
            return None

        if not line.startswith("data:"):
            return None

        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return None

        # Unescape newlines encoded by SSEEncoder.encode()
        return data.replace("\\n", "\n")

    def _serialize_messages(self, messages: list) -> list[dict[str, str]]: # noqa
        """
        Serialize LangChain message objects into plain dicts for JSON transport.

        Handles SystemMessage, HumanMessage, AIMessage by type name.
        Unknown types fall back to role "user" with a warning so the stream
        is never silently dropped due to an unexpected message type.
        """
        role_map = {
            "SystemMessage": "system",
            "HumanMessage":  "user",
            "AIMessage":     "assistant",
        }
        serialised = []
        for m in messages:
            role    = role_map.get(type(m).__name__, "user")
            content = getattr(m, "content", "")
            if role == "user" and type(m).__name__ not in role_map:
                log.warning(
                    "remote_llm_unknown_message_type",
                    type=type(m).__name__,
                    content_preview=str(content)[:60],
                )
            serialised.append({"role": role, "content": content})
        return serialised

    # ── stream_question ───────────────────────────────────────────────────────

    async def stream_question(
        self,
        llm_input:  LLMInterviewInput,
        request_id: str | None = None,
        session_id: str | None = None,  # noqa — forwarded to remote
    ) -> AsyncIterator[str]:
        """
        POST /stream_question — SSE stream of interview question tokens.

        Mirrors LLMNode.stream_question() latency-budget behaviour:
          1. Budget check at entry — abort if SLA already blown
          2. Per-token budget check — abort mid-stream if SLA expires

        The full LLMInterviewInput is serialised to the remote so it can
        apply its own interviewer mode logic (validation, fallback bank,
        diversity enforcement).  Callers receive the same validated token
        stream they would get from LLMNode.

        Raises:
          LatencyBudgetExceeded  — SLA blown at entry or mid-stream
          asyncio.CancelledError — task was cancelled; always re-raised
          RuntimeError           — remote signalled an error event
          httpx.HTTPStatusError  — non-2xx from remote
        """
        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_llm.stream_question")
        except LatencyBudgetExceeded:
            _budget_exc.inc()
            log.warning("remote_llm_stream_question_budget_exceeded_entry", request_id=rid)
            raise

        headers = self._request_headers(rid, accept_sse=True)
        payload = {
            "domain":          llm_input.domain,
            "domain_label":    llm_input.domain_label,
            "level":           llm_input.level,
            "last_question":   llm_input.last_question,
            "last_answer":     llm_input.last_answer,
            "is_first":        llm_input.is_first_in_domain,
            "domain_switched": llm_input.domain_switched,
            "prev_domain":     llm_input.prev_domain_label,
            "q_index":         llm_input.q_index_in_domain,
            "session_id":      session_id,
            "request_id":      rid,
        }

        with tracer.start_as_current_span("llm.remote.stream_question") as span:
            span.set_attribute("request_id",      rid)
            span.set_attribute("domain",          llm_input.domain)
            span.set_attribute("level",           llm_input.level)
            span.set_attribute("domain_switched", llm_input.domain_switched)
            span.set_attribute("provider",        "remote")

            async with bulkheads.acquire("llm.interviewer"):
                _active.labels(mode="interviewer").inc()
                _cb_open.labels(model="remote", path="interviewer.stream").set(
                    1 if self._breaker.state == "OPEN" else 0
                )
                self._inflight["interviewer"] += 1
                t0            = time.monotonic()
                token_count   = 0
                first_token   = True

                try:
                    async def _open_stream() -> Any:
                        return self._http.stream(
                            "POST", "/stream_question",
                            json=payload,
                            headers=headers,
                        )

                    stream_ctx = await self._breaker.call(
                        backoff_retry,
                        _open_stream,
                        attempts=2,
                        base_delay=1.0,
                        exceptions=_LLM_RETRYABLE_REMOTE,
                    )

                    async with stream_ctx as resp:
                        resp.raise_for_status()

                        async for raw_line in resp.aiter_lines():
                            token = self._parse_sse_line(raw_line)
                            if token is None:
                                continue

                            try:
                                self._check_budget("remote_llm.stream_question.token")
                            except LatencyBudgetExceeded:
                                _budget_exc.inc()
                                log.warning(
                                    "remote_llm_stream_question_budget_mid",
                                    request_id  = rid,
                                    tokens_seen = token_count,
                                )
                                raise

                            if first_token:
                                _ttft.labels(model="remote").observe(
                                    time.monotonic() - t0
                                )
                                first_token = False

                            token_count += 1

                            try:
                                yield token
                            except (GeneratorExit, asyncio.CancelledError):
                                log.debug(
                                    "remote_llm_stream_question_generator_exit",
                                    request_id  = rid,
                                    tokens_seen = token_count,
                                )
                                return

                    elapsed = round(time.monotonic() - t0, 3)
                    _req_total.labels(
                        model="remote", status="ok", mode="interviewer"
                    ).inc()
                    _latency.labels(model="remote", mode="interviewer").observe(elapsed)

                    span.set_attribute("latency_s",   elapsed)
                    span.set_attribute("token_count", token_count)
                    log.info(
                        "remote_llm_stream_question_ok",
                        request_id  = rid,
                        domain      = llm_input.domain,
                        level       = llm_input.level,
                        token_count = token_count,
                        latency_s   = elapsed,
                    )

                except LatencyBudgetExceeded:
                    _req_total.labels(
                        model="remote", status="budget_exceeded", mode="interviewer"
                    ).inc()
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        model="remote", status="cancelled", mode="interviewer"
                    ).inc()
                    log.warning(
                        "remote_llm_stream_question_cancelled",
                        request_id  = rid,
                        tokens_seen = token_count,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(
                        model="remote", status="error", mode="interviewer"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "remote_llm_stream_question_error",
                        request_id  = rid,
                        error       = str(exc),
                        tokens_seen = token_count,
                    )
                    raise

                finally:
                    _active.labels(mode="interviewer").dec()
                    _cb_open.labels(model="remote", path="interviewer.stream").set(
                        1 if self._breaker.state == "OPEN" else 0
                    )
                    self._inflight["interviewer"] -= 1

    # ── extract_ats ───────────────────────────────────────────────────────────

    async def extract_ats(
        self,
        intro_text:  str,
        request_id:  str | None = None,
    ) -> str:
        """
        POST /extract_ats — returns raw JSON string.

        Mirrors LLMNode.extract_ats() input validation so callers get the
        same ValueError for empty intro_text regardless of local vs remote.
        Uses 3 attempts / 1.5 s base (matches local ATSMode retry config).
        Validates that the response is parseable JSON before returning,
        logging a warning on invalid JSON rather than raising — the schema
        validation is the caller's responsibility (validate_ats_json in
        extract_and_validate_intro()).

        Raises:
          ValueError             — empty intro_text
          LatencyBudgetExceeded  — SLA blown before the call
          asyncio.CancelledError — task was cancelled; always re-raised
          httpx.HTTPStatusError  — non-2xx from remote
          RuntimeError           — both retry attempts failed
        """
        if not intro_text or not intro_text.strip():
            raise ValueError("extract_ats: intro_text is empty")

        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_llm.extract_ats")
        except LatencyBudgetExceeded:
            _budget_exc.inc()
            log.warning("remote_llm_extract_ats_budget_exceeded", request_id=rid)
            raise

        headers = self._request_headers(rid)
        payload = {"intro_text": intro_text.strip(), "request_id": rid}

        with tracer.start_as_current_span("llm.remote.extract_ats") as span:
            span.set_attribute("request_id",  rid)
            span.set_attribute("intro_chars", len(intro_text))
            span.set_attribute("provider",    "remote")

            async with bulkheads.acquire("llm.ats"):
                _active.labels(mode="ats").inc()
                _cb_open.labels(model="remote", path="ats.batch").set(
                    1 if self._breaker.state == "OPEN" else 0
                )
                self._inflight["ats"] += 1
                t0 = time.monotonic()

                try:
                    async def _call() -> str:
                        resp = await self._http.post(
                            "/extract_ats",
                            json=payload,
                            headers=headers,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        return data.get("result", "{}")

                    raw: str = await self._breaker.call(
                        backoff_retry,
                        _call,
                        attempts=3,
                        base_delay=1.5,
                        exceptions=_LLM_RETRYABLE_REMOTE,
                    )

                    if not raw or not raw.strip():
                        raise ValueError(
                            f"Remote ATS returned empty result (request_id={rid})"
                        )

                    # Warn on invalid JSON — schema validation is caller's job
                    try:
                        json.loads(raw.strip())
                    except json.JSONDecodeError:
                        log.warning(
                            "remote_llm_extract_ats_invalid_json",
                            request_id=rid,
                            raw_len=len(raw),
                        )

                    elapsed = round(time.monotonic() - t0, 3)
                    _req_total.labels(model="remote", status="ok", mode="ats").inc()
                    _latency.labels(model="remote", mode="ats").observe(elapsed)
                    _cb_open.labels(model="remote", path="ats.batch").set(
                        1 if self._breaker.state == "OPEN" else 0
                    )

                    span.set_attribute("latency_s",  elapsed)
                    span.set_attribute("json_chars", len(raw))
                    log.info(
                        "remote_llm_extract_ats_ok",
                        request_id = rid,
                        json_chars = len(raw),
                        latency_s  = elapsed,
                    )
                    return raw

                except LatencyBudgetExceeded:
                    _req_total.labels(
                        model="remote", status="budget_exceeded", mode="ats"
                    ).inc()
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        model="remote", status="cancelled", mode="ats"
                    ).inc()
                    log.warning("remote_llm_extract_ats_cancelled", request_id=rid)
                    raise

                except Exception as exc:
                    _req_total.labels(
                        model="remote", status="error", mode="ats"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "remote_llm_extract_ats_error",
                        request_id=rid,
                        error=str(exc),
                    )
                    raise

                finally:
                    _active.labels(mode="ats").dec()
                    _cb_open.labels(model="remote", path="ats.batch").set(
                        1 if self._breaker.state == "OPEN" else 0
                    )
                    self._inflight["ats"] -= 1

    # ── stream_messages ───────────────────────────────────────────────────────

    async def stream_messages(
        self,
        messages:   list,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        POST /stream_messages — SSE stream of evaluation/raw tokens.

        For evaluation_engine.py and legacy paths only.
        MUST NOT be called from voice_graph.node_llm for interview turns —
        the same constraint as LLMNode.stream_messages().

        LangChain message objects are serialised via _serialize_messages()
        (SystemMessage → "system", HumanMessage → "user", AIMessage →
        "assistant") so the remote receives clean role/content dicts.

        Budget checks and GeneratorExit handling mirror stream_question()
        for consistency.

        Raises:
          ValueError             — empty messages list
          LatencyBudgetExceeded  — SLA blown at entry or mid-stream
          asyncio.CancelledError — task was cancelled; always re-raised
          RuntimeError           — remote signalled an error event
          httpx.HTTPStatusError  — non-2xx from remote
        """
        if not messages:
            raise ValueError("stream_messages: messages list is empty")

        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_llm.stream_messages")
        except LatencyBudgetExceeded:
            _budget_exc.inc()
            log.warning("remote_llm_stream_messages_budget_exceeded_entry", request_id=rid)
            raise

        headers = self._request_headers(rid, accept_sse=True)
        payload = {
            "messages":   self._serialize_messages(messages),
            "request_id": rid,
        }

        with tracer.start_as_current_span("llm.remote.stream_messages") as span:
            span.set_attribute("request_id",    rid)
            span.set_attribute("message_count", len(messages))
            span.set_attribute("provider",      "remote")

            async with bulkheads.acquire("llm.eval"):
                _active.labels(mode="eval").inc()
                _cb_open.labels(model="remote", path="eval.stream").set(
                    1 if self._breaker.state == "OPEN" else 0
                )
                self._inflight["eval"] += 1
                t0          = time.monotonic()
                token_count = 0
                first_token = True

                try:
                    async def _open_stream() -> Any:
                        return self._http.stream(
                            "POST", "/stream_messages",
                            json=payload,
                            headers=headers,
                        )

                    stream_ctx = await self._breaker.call(
                        backoff_retry,
                        _open_stream,
                        attempts=2,
                        base_delay=1.0,
                        exceptions=_LLM_RETRYABLE_REMOTE,
                    )

                    async with stream_ctx as resp:
                        resp.raise_for_status()

                        async for raw_line in resp.aiter_lines():
                            token = self._parse_sse_line(raw_line)
                            if token is None:
                                continue

                            try:
                                self._check_budget(
                                    "remote_llm.stream_messages.token"
                                )
                            except LatencyBudgetExceeded:
                                _budget_exc.inc()
                                log.warning(
                                    "remote_llm_stream_messages_budget_mid",
                                    request_id  = rid,
                                    tokens_seen = token_count,
                                )
                                raise

                            if first_token:
                                _ttft.labels(model="remote").observe(
                                    time.monotonic() - t0
                                )
                                first_token = False

                            token_count += 1

                            try:
                                yield token
                            except (GeneratorExit, asyncio.CancelledError):
                                log.debug(
                                    "remote_llm_stream_messages_generator_exit",
                                    request_id  = rid,
                                    tokens_seen = token_count,
                                )
                                return

                    elapsed = round(time.monotonic() - t0, 3)
                    _req_total.labels(
                        model="remote", status="ok", mode="eval"
                    ).inc()
                    _latency.labels(model="remote", mode="eval").observe(elapsed)

                    span.set_attribute("latency_s",   elapsed)
                    span.set_attribute("token_count", token_count)
                    log.info(
                        "remote_llm_stream_messages_ok",
                        request_id  = rid,
                        token_count = token_count,
                        latency_s   = elapsed,
                    )

                except LatencyBudgetExceeded:
                    _req_total.labels(
                        model="remote", status="budget_exceeded", mode="eval"
                    ).inc()
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        model="remote", status="cancelled", mode="eval"
                    ).inc()
                    log.warning(
                        "remote_llm_stream_messages_cancelled",
                        request_id  = rid,
                        tokens_seen = token_count,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(
                        model="remote", status="error", mode="eval"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "remote_llm_stream_messages_error",
                        request_id  = rid,
                        error       = str(exc),
                        tokens_seen = token_count,
                    )
                    raise

                finally:
                    _active.labels(mode="eval").dec()
                    _cb_open.labels(model="remote", path="eval.stream").set(
                        1 if self._breaker.state == "OPEN" else 0
                    )
                    self._inflight["eval"] -= 1

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        """
        GET /health — returns ServiceHealthState.

        Merges the remote service's own health flag with the local circuit
        breaker state.  A remote reporting healthy=True while the local
        breaker is OPEN is still considered unhealthy — consistent with
        LLMNode.health() logic.

        Reports per-mode inflight counts in the extra dict, matching
        the format returned by LLMNode.health() so operator dashboards
        work identically for both implementations.

        5-second timeout prevents a dead health probe from blocking the
        voice graph health check cycle.  Always returns (never raises).
        """
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            is_remote_healthy = bool(data.get("healthy", False))
            _cb_open.labels(model="remote", path="all").set(
                1 if self._breaker.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "llm.remote",
                healthy       = is_remote_healthy and self._breaker.state != "OPEN",
                circuit_state = self._breaker.state,
                inflight      = sum(self._inflight.values()),
                extra         = {
                    "inflight_by_mode": dict(self._inflight),
                    "base_url":         self._base_url,
                },
            )
        except Exception as exc:
            _cb_open.labels(model="remote", path="all").set(
                1 if self._breaker.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "llm.remote",
                healthy       = False,
                circuit_state = self._breaker.state,
                inflight      = sum(self._inflight.values()),
                error         = str(exc),
                extra         = {"inflight_by_mode": dict(self._inflight)},
            )

    # ── lifecycle stubs ───────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """
        No-op — the remote service manages its own warmup.
        Provided so on_startup() works without isinstance checks when
        RemoteLLMClient is injected instead of LLMNodeV2.
        """
        log.debug("remote_llm_warmup_noop", base_url=self._base_url)

    def start_health_probe(self) -> None:
        """No-op — the remote service runs its own probe loop."""
        log.debug("remote_llm_start_health_probe_noop", base_url=self._base_url)

    def stop_health_probe(self) -> None:
        """No-op — nothing to stop."""
        log.debug("remote_llm_stop_health_probe_noop", base_url=self._base_url)

    def evict_session(self, session_id: str) -> None: # noqa
        """
        No-op for the remote client — the remote service manages its own
        session state (diversity cache, budget store, etc.).  Satisfies
        LLMNodeProtocol.evict_session() so callers can call it unconditionally
        without isinstance checks.
        """
        log.debug(
            "remote_llm_evict_session_noop",
            session_id=session_id[:8] if session_id else "",
        )

    async def close(self) -> None:
        """Drain and close the underlying httpx connection pool."""
        await self._http.aclose()


# ── SSE streaming utilities ────────────────────────────────────────────────────


class SSEEncoder:
    """
    Encodes streaming token chunks into Server-Sent Events format.
    Used by RemoteLLMClient-serving endpoints, not by the local LLMNode.
    """

    @staticmethod
    def encode(chunk: str) -> str:
        return f"data: {chunk.replace(chr(10), chr(92) + 'n')}\n\n"

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"

    @staticmethod
    def error(message: str) -> str:
        return f"event: error\ndata: {json.dumps({'error': message})}\n\n"

    @staticmethod
    def decode(line: str) -> str | None:
        if not line.startswith("data:"):
            return None
        data = line[5:].strip()
        if data == "[DONE]":
            return None
        return data.replace("\\n", "\n")


class SSEAccumulator:
    """Accumulates SSE token stream chunks into a complete response string."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._lock = asyncio.Lock()

    async def push(self, chunk: str) -> None:
        async with self._lock:
            self._chunks.append(chunk)

    async def get_full(self) -> str:
        async with self._lock:
            return "".join(self._chunks)

    def get_full_sync(self) -> str:
        return "".join(self._chunks)

    def reset(self) -> None:
        self._chunks.clear()

    @property
    def char_count(self) -> int:
        return sum(len(c) for c in self._chunks)


# ── voice_graph convenience helpers ───────────────────────────────────────────


async def generate_interviewer_question(
    *,
    llm_input:  LLMInterviewInput,
    session_id: str,
    request_id: str = "",
) -> str:
    """
    Single entry point for voice_graph.node_llm to generate the next question.
    Returns the final validated question string (never a raw stream).

    Usage:
        question = await generate_interviewer_question(
            llm_input  = llm_input,
            session_id = state["session_id"],
            request_id = state["request_id"],
        )
    """
    rid    = request_id or new_request_id()
    chunks: list[str] = []
    async for chunk in llm_node.stream_question(
        llm_input, request_id=rid, session_id=session_id
    ):
        chunks.append(chunk)
    return "".join(chunks).strip()


async def extract_and_validate_intro(
    intro_text: str,
    request_id: str = "",
) -> dict | None:
    """
    Single entry point for voice_graph.node_ats to extract intro data.
    Returns validated dict or None if extraction failed completely.
    """
    try:
        raw = await llm_node.extract_ats(intro_text, request_id=request_id or None)
        return validate_ats_json(raw)
    except Exception as exc:
        log.error("llm_ats_extract_failed", error=str(exc))
        return None


# ── factory and module-level singletons ───────────────────────────────────────


def get_llm_node() -> LLMNodeProtocol:
    """
    Factory. Returns RemoteLLMClient for distributed deployments,
    LLMNodeV2 for local/monolith deployments.
    """
    if LLM_SERVICE_URL:
        log.info("llm_using_remote_client", url=LLM_SERVICE_URL)
        return cast(LLMNodeProtocol, RemoteLLMClient())
    log.info("llm_using_local_node_v2", primary=PRIMARY_MODEL, ats=ATS_MODEL)
    return cast(LLMNodeProtocol, LLMNodeV2())


llm_node: LLMNodeProtocol = get_llm_node()

# Convenience references for eval_engine.py and offline tooling
eval_llm  = getattr(llm_node, "eval_mode",  None)
batch_ats = getattr(llm_node, "batch_ats",  None)