"""
LLM node — OpenAI chat via LangChain, Redis cache in front.

Base version features:
  - Connects to locally running Ollama server
  - Uses configurable model (default: llama3.1:8b)
  - Startup connection check to verify model availability
  - Synchronous request-response generation (no streaming)
  - Basic latency measurement via decorator
  - Simple structured logging for connection + timing
  - Graceful failure handling with fallback message
  - 60s request timeout protection

Extra layers on top of the base version:
  - circuit breaker so one bad OpenAI outage doesn't queue everything up
  - Redis SETNX lock to stop cache stampedes under load
  - fallback model if the primary is rate-limited or down
  - streaming via astream() for low-latency voice UX
  - token-bucket rate limiter so we don't accidentally blow the API quota
  - full OTel spans + Prometheus counters/histograms
  - structured JSON logging with request_id and token counts
  - asyncio.Semaphore cap so the process doesn't spin up unlimited goroutines

Distributed-service additions:
  - RemoteLLMClient: satisfies the same LLMNodeProtocol as the local node
    over plain HTTPS (httpx). VoiceGraph only imports the protocol — it
    switches between local and remote via an env-driven factory function
    (get_llm_node()) without any graph-level code changes.
  - Versioned cache keys: the cache key now encodes the model name,
    temperature, and a sha256 of the system prompt so config changes
    automatically bust stale cache entries.
  - Dual circuit breakers: stream and batch paths have separate breakers.
    A slow streaming endpoint can degrade without tripping the batch breaker
    that serves synchronous /run requests.
  - InMemoryLRU fallback: if Redis is unreachable we serve from an in-process
    LRU and enter DegradedMode, which automatically reduces concurrency to
    avoid flooding the model APIs during the outage window.
  - LatencyBudget enforcement: check() at entry aborts immediately if the
    per-request SLA is already exceeded before we even start work.
  - Health endpoint data: health() returns a ServiceHealthState that
    VoiceGraph can use for routing decisions.
  - stream_with_metadata(): wraps stream() and emits a final metadata dict
    after the last token — useful for WebSocket handlers.
"""

from __future__ import annotations

import asyncio  # noqa
import hashlib
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable
from pydantic import SecretStr

import httpx
import redis.asyncio as aioredis
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from opentelemetry.trace import StatusCode

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
    get_logger,
    get_meter,
    get_tracer,
    inject_trace_headers,
    make_counter,
    make_gauge,
    make_histogram,
    make_versioned_cache_key,
    new_request_id,
)

from dotenv import load_dotenv

load_dotenv()

log = get_logger(__name__)
tracer = get_tracer(__name__)
meter = get_meter(__name__)

# ── config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

PRIMARY_MODEL: str = os.getenv("LLM_MODEL", "gpt-5-mini")
FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "gpt-5-nano")
TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

REDIS_URL: str | None = os.getenv("REDIS_URL")
REDIS_MAX_CONN: int = int(os.getenv("REDIS_MAX_CONN", "200"))

CACHE_TTL: int = int(os.getenv("LLM_CACHE_TTL", "3600"))
CACHE_PREFIX: str = "llm:v3:"  # bump when cache structure changes
STAMPEDE_LOCK_TTL: int = 30
STAMPEDE_POLL_INTERVAL: float = 0.4
STAMPEDE_POLL_LIMIT: int = 75

RATE_PER_SEC: float = float(os.getenv("LLM_RATE_PER_SEC", "20.0"))
RATE_BURST: float = float(os.getenv("LLM_RATE_BURST", "40.0"))
LRU_SIZE: int = int(os.getenv("LLM_LRU_SIZE", "512"))

STREAM_CACHE_MIN_CHARS: int = int(os.getenv("LLM_STREAM_CACHE_MIN_CHARS", "10"))

# Remote service config (used by RemoteLLMClient)
LLM_SERVICE_URL: str = os.getenv("LLM_SERVICE_URL", "")
LLM_SERVICE_API_KEY: str = os.getenv("LLM_SERVICE_API_KEY", "")
LLM_SERVICE_TIMEOUT: float = float(os.getenv("LLM_SERVICE_TIMEOUT", "60.0"))

SYSTEM_PROMPT = """
You are a technical interviewer conducting real interview-style conversations.

Behavior:
- Ask structured, professional questions.
- Do NOT give full solutions immediately.
- Ask follow-up questions.
- Give hints before answers.
- Evaluate responses like an interviewer.
- Keep replies concise for voice output.

Interview domains:
- Coding / DSA
- System design
- Behavioral
- CS fundamentals

Rules:
- Avoid long essays.
- Speak in short, clear steps.
- Prioritize reasoning over final answers.
- Challenge shallow responses.
"""

# Stable hash of the system prompt included in cache keys so that
# editing SYSTEM_PROMPT automatically busts all cached responses.
_SYSTEM_PROMPT_HASH: str = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]

# ── Prometheus metrics ────────────────────────────────────────────────────────

_req_total = make_counter(
    "llm_requests_total", "Total LLM requests", ["model", "status", "mode"]
)
_cache_hits = make_counter("llm_cache_hits_total", "Cache hits", ["source"])
_latency = make_histogram(
    "llm_response_latency_seconds",
    "End-to-end LLM response latency",
    ["model"],
    buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 30),
)
_ttft = make_histogram(
    "llm_time_to_first_token_seconds",
    "Latency from stream() call to first yielded token",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
_tokens_used = make_histogram(
    "llm_tokens_used",
    "Token counts per request",
    ["model", "kind"],
    buckets=(50, 100, 200, 500, 1000, 2000, 4000),
)
_circuit_open = make_gauge(
    "llm_circuit_breaker_open", "1 when breaker is OPEN", ["model", "path"]
)
_active = make_gauge("llm_active_requests", "Requests currently in flight", ["mode"])
_budget_exceeded = make_counter(
    "llm_latency_budget_exceeded_total",
    "Requests aborted because the SLA budget was already blown",
)
_lru_hits = make_counter("llm_lru_hits_total", "In-memory LRU cache hits")


# ── LLMNodeProtocol ───────────────────────────────────────────────────────────


@runtime_checkable
class LLMNodeProtocol(Protocol):
    """
    The contract that both LLMNode (local) and RemoteLLMClient (distributed)
    must satisfy. VoiceGraph only ever depends on this protocol — never on
    the concrete implementations below.

    Any class that implements generate() and stream() with these exact
    signatures automatically satisfies the protocol (structural subtyping).
    """

    async def generate(
        self, prompt: str, request_id: str | None = None
    ) -> dict[str, Any]: ...

    def stream(
        self, prompt: str, request_id: str | None = None
    ) -> AsyncIterator[str]: ...

    async def health(self) -> ServiceHealthState: ...

    async def close(self) -> None: ...


# ── token counting callback ───────────────────────────────────────────────────


class _TokenCounter(AsyncCallbackHandler):
    """
    Hooked into LangChain so we get prompt/completion counts even when
    streaming, where the full text isn't available until the end.
    """

    def __init__(self, model: str) -> None:
        self._model = model
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        usage = (response.llm_output or {}).get("token_usage", {})
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        _tokens_used.labels(model=self._model, kind="prompt").observe(
            self.prompt_tokens
        )
        _tokens_used.labels(model=self._model, kind="completion").observe(
            self.completion_tokens
        )


# ── helpers ───────────────────────────────────────────────────────────────────


def _cache_key(prompt: str, model: str, temperature: float) -> str:
    return make_versioned_cache_key(
        CACHE_PREFIX, prompt, model, temperature, _SYSTEM_PROMPT_HASH
    )


def _lock_key(cache_key: str) -> str:
    return f"lock:{cache_key}"


# ── local LLM node ────────────────────────────────────────────────────────────


class LLMNode:
    """
    LangGraph-compatible async LLM node (local / in-process implementation).

    State contract
    ──────────────
    reads:  state["user_input"]       (str)
    writes: state["llm_response"]     (str)
            state["llm_tokens"]       (dict: prompt / completion counts)
            state["llm_model_used"]   (str)
            state["llm_cached"]       (bool)
            state["request_id"]       (str)

    Distributed usage
    ─────────────────
    Use get_llm_node() instead of instantiating directly. The factory
    returns a RemoteLLMClient when LLM_SERVICE_URL is configured, or this
    class otherwise. VoiceGraph imports only the factory — it never knows
    which implementation it's using.
    """

    def __init__(
        self,
        primary_model: str = PRIMARY_MODEL,
        fallback_model: str = FALLBACK_MODEL,
        temperature: float = TEMPERATURE,
        redis_url: str = REDIS_URL,
        cache_ttl: int = CACHE_TTL,
        rate_per_sec: float = RATE_PER_SEC,
        rate_burst: float = RATE_BURST,
    ) -> None:
        self._primary = primary_model
        self._fallback = fallback_model
        self._temperature = temperature
        self._redis_url = redis_url
        self._cache_ttl = cache_ttl

        self._redis: aioredis.Redis | None = None
        self._lru = InMemoryLRU(max_size=LRU_SIZE)
        self._chains: dict[str, Any] = {}
        self._rate_limiter = RateLimiter(rate_per_sec, rate_burst)

        # separate breakers per model AND per operation type (stream vs batch)
        # so a slow streaming endpoint never trips the batch breaker
        self._breakers: dict[str, CircuitBreaker] = {
            f"{primary_model}.batch": CircuitBreaker(name=f"llm:{primary_model}:batch"),
            f"{primary_model}.stream": CircuitBreaker(
                name=f"llm:{primary_model}:stream"
            ),
            f"{fallback_model}.batch": CircuitBreaker(
                name=f"llm:{fallback_model}:batch"
            ),
            f"{fallback_model}.stream": CircuitBreaker(
                name=f"llm:{fallback_model}:stream"
            ),
        }

        self._inflight_batch = 0
        self._inflight_stream = 0

    # ── internal setup ────────────────────────────────────────────────────────

    async def _get_redis(self) -> aioredis.Redis | None:
        if not self._redis_url:
            return None

        if self._redis is None:
            self._redis = await aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=REDIS_MAX_CONN,
                socket_connect_timeout=0.1,  # fail fast
                socket_timeout=0.1,
            )
        return self._redis

    def _get_chain(self, model: str, streaming: bool = False):
        # Keyed by model + mode so batch and stream never share a ChatOpenAI
        # instance. A single chain with streaming=True used for ainvoke() (batch
        # path) causes LangChain to route through its streaming aggregation
        # internally, which silently returns "" when the model response is empty
        # or the response format isn't handled — zero tokens, no exception raised.
        key = f"{model}:{'stream' if streaming else 'batch'}"
        if key not in self._chains:
            llm = ChatOpenAI(
                model=model,
                api_key=SecretStr(OPENAI_API_KEY),
                temperature=self._temperature,
                max_retries=0,
                timeout=60,
                max_tokens=200,
                streaming=streaming,
            )
            self._chains[key] = llm | StrOutputParser()
        return self._chains[key]

    # ── cache (Redis + LRU fallback) ──────────────────────────────────────────
    # Redis is the primary cache. On any Redis error we fall back to the
    # in-process LRU and enter DegradedMode which reduces concurrency.

    async def _cache_get(self, key: str) -> str | None:
        # try Redis first
        try:
            r = await self._get_redis()
            if r:
                val = await r.get(key)
                if val is not None:
                    return val
            # Redis was reachable but missed — exit degraded mode if active
            if degraded_mode.active:
                await degraded_mode.exit("redis_reachable")
        except Exception as exc:
            log.warning("cache_read_failed_using_lru", error=str(exc))
            await degraded_mode.enter(f"redis_error: {exc}")

        # LRU fallback
        val = await self._lru.get(key)
        if val is not None:
            _lru_hits.inc()
        return val

    async def _cache_set(self, key: str, value: str) -> None:
        # always write to LRU so it stays warm even if Redis works
        await self._lru.set(key, value)
        try:
            r = await self._get_redis()
            if r:
                if self._cache_ttl > 0:
                    await r.setex(key, self._cache_ttl, value)
                else:
                    await r.set(key, value)
        except Exception as exc:
            log.warning("cache_write_skipped", error=str(exc))

    # ── stampede protection ───────────────────────────────────────────────────
    # When hundreds of requests all miss the cache at the same instant,
    # only the one that wins the SETNX lock actually hits the model.
    # Everyone else waits and reads from cache once it's populated.

    async def _acquire_lock(self, lock_key: str) -> bool:
        try:
            r = await self._get_redis()
            return bool(await r.set(lock_key, "1", nx=True, ex=STAMPEDE_LOCK_TTL))
        except Exception:  # noqa
            # Redis down: fail-open BUT in degraded mode we allow only one
            # concurrent compute to reduce model pressure
            if degraded_mode.active:
                return False  # force everything to poll and serialize
            return True

    async def _release_lock(self, lock_key: str) -> None:
        try:
            r = await self._get_redis()
            await r.delete(lock_key)
        except Exception:  # noqa
            pass

    async def _poll_for_cached(self, key: str) -> str | None:
        for _ in range(STAMPEDE_POLL_LIMIT):
            await asyncio.sleep(STAMPEDE_POLL_INTERVAL)
            val = await self._cache_get(key)
            if val is not None:
                return val
        return None

    # ── model calls ───────────────────────────────────────────────────────────

    async def _invoke(
        self, model: str, messages: list, path: str = "batch"
    ) -> tuple[str, int, int]:
        counter = _TokenCounter(model)
        chain = self._get_chain(model, streaming=False)  # batch: streaming=False so ainvoke() uses the standard non-streaming path and token counts are reliable
        breaker_key = f"{model}.{path}"
        breaker = self._breakers[breaker_key]
        _circuit_open.labels(model=model, path=path).set(
            1 if breaker.state == "OPEN" else 0
        )

        async def _call() -> str:
            return await chain.ainvoke(messages, config={"callbacks": [counter]})

        text: str = await breaker.call(
            backoff_retry, _call, attempts=3, base_delay=1.0, exceptions=(Exception,)
        )
        return text, counter.prompt_tokens, counter.completion_tokens

    async def _stream_model(self, model: str, messages: list) -> AsyncIterator[str]:
        """
        Yield tokens from model; wraps iterator acquisition in the stream
        circuit breaker so repeated failures trip it independently of batch.
        """
        chain = self._get_chain(model, streaming=True)  # stream: dedicated chain with streaming=True
        breaker = self._breakers[f"{model}.stream"]
        _circuit_open.labels(model=model, path="stream").set(
            1 if breaker.state == "OPEN" else 0
        )

        async def _open() -> AsyncIterator[str]:
            return chain.astream(messages)

        iterator: AsyncIterator[str] = await breaker.call(_open)

        try:
            async for chunk in iterator:
                yield chunk
        finally:
            # Ensure underlying async iterator is closed on cancellation
            # This prevents leaking the HTTP stream or continuing token generation
            if hasattr(iterator, "aclose"):
                try:
                    await iterator.aclose()
                except Exception:  # noqa
                    pass

    # ── public generate ───────────────────────────────────────────────────────

    async def generate(
        self, prompt: str, request_id: str | None = None
    ) -> dict[str, Any]:
        """
        Main entry point.

        Flow: budget check → rate-limit → bulkhead → cache check →
              stampede lock → primary model → fallback → cache write.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        rid = request_id or new_request_id()
        key = _cache_key(prompt, self._primary, self._temperature)
        lk = _lock_key(key)

        # ── SLA budget check ──────────────────────────────────────────────────
        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="llm.generate")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                log.warning("llm_budget_exceeded_at_entry", request_id=rid)
                raise

        with tracer.start_as_current_span("llm.generate") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("prompt_length", len(prompt))

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("llm.batch"):
                _active.labels(mode="batch").inc()
                self._inflight_batch += 1
                t0 = time.monotonic()

                try:
                    cached_val = await self._cache_get(key)
                    if cached_val is not None:
                        _cache_hits.labels(source="redis").inc()
                        _req_total.labels(
                            model="cache", status="hit", mode="batch"
                        ).inc()
                        span.set_attribute("cache_hit", True)
                        return {
                            "response": cached_val,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "model_used": "cache",
                            "cached": True,
                        }

                    span.set_attribute("cache_hit", False)

                    got_lock = await self._acquire_lock(lk)
                    if not got_lock:
                        log.info("stampede_wait", request_id=rid)
                        polled = await self._poll_for_cached(key)
                        if polled is not None:
                            _cache_hits.labels(source="stampede_wait").inc()
                            return {
                                "response": polled,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "model_used": "cache",
                                "cached": True,
                            }
                        got_lock = True

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]

                    model_used = self._primary
                    pt = ct = 0  # noqa

                    try:
                        text, pt, ct = await self._invoke(self._primary, messages)
                    except (CircuitBreakerOpen, Exception) as primary_err:
                        log.warning(
                            "primary_failed_using_fallback",
                            request_id=rid,
                            error=str(primary_err),
                        )
                        try:
                            text, pt, ct = await self._invoke( # noqa
                                self._fallback, messages, path="batch"
                            )  # noqa
                            model_used = self._fallback
                        except Exception as fallback_err:
                            _req_total.labels(
                                model=model_used, status="error", mode="batch"
                            ).inc()
                            span.set_status(StatusCode.ERROR, str(fallback_err))
                            raise RuntimeError(
                                f"Both models failed. Primary: {primary_err}. Fallback: {fallback_err}"
                            ) from fallback_err
                    finally:
                        if got_lock:
                            await self._release_lock(lk)

                    # Guard: an empty response must raise so the retry/error path
                    # in VoiceGraph handles it cleanly instead of passing "" to
                    # sanitize/TTS and serving the user a pre-baked apology.
                    if not text or not text.strip():
                        _req_total.labels(
                            model=model_used, status="error", mode="batch"
                        ).inc()
                        raise ValueError(
                            f"LLM returned an empty response "
                            f"(model={model_used}, prompt_tokens={pt}, completion_tokens={ct}). "
                            f"Check model name, API key, and quota."
                        )

                    await self._cache_set(key, text)

                    latency = time.monotonic() - t0
                    _req_total.labels(model=model_used, status="ok", mode="batch").inc()
                    _latency.labels(model=model_used).observe(latency)
                    span.set_attribute("model", model_used)
                    span.set_attribute("latency_s", round(latency, 3))

                    log.info(
                        "llm_ok",
                        request_id=rid,
                        model=model_used,
                        latency_s=round(latency, 3),
                        prompt_tokens=pt,
                        completion_tokens=ct,
                    )

                    return {
                        "response": text,
                        "prompt_tokens": pt,
                        "completion_tokens": ct,
                        "model_used": model_used,
                        "cached": False,
                    }

                finally:
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── streaming ─────────────────────────────────────────────────────────────

    async def stream(
        self, prompt: str, request_id: str | None = None
    ) -> AsyncIterator[str]:
        """
        Yield tokens as they arrive — separate stream breaker from generate().

        Flow: budget check → rate-limit → stream bulkhead → cache hit yields
              immediately → stampede poll → primary stream → fallback stream
              → cache write on exhaustion.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        rid = request_id or new_request_id()
        key = _cache_key(prompt, self._primary, self._temperature)
        lk = _lock_key(key)

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="llm.stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        with tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("prompt_length", len(prompt))

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("llm.stream"):
                _active.labels(mode="stream").inc()
                self._inflight_stream += 1
                t0 = time.monotonic()
                accumulated: list[str] = []
                model_used = self._primary
                got_lock = False

                try:
                    cached_val = await self._cache_get(key)
                    if cached_val is not None:
                        _cache_hits.labels(source="redis_stream").inc()
                        _req_total.labels(
                            model="cache", status="hit", mode="stream"
                        ).inc()
                        _ttft.labels(model="cache").observe(0.0)
                        span.set_attribute("cache_hit", True)
                        yield cached_val
                        return

                    span.set_attribute("cache_hit", False)

                    got_lock = await self._acquire_lock(lk)
                    if not got_lock:
                        polled = await self._poll_for_cached(key)
                        if polled is not None:
                            _cache_hits.labels(source="stampede_wait_stream").inc()
                            _ttft.labels(model="cache").observe(time.monotonic() - t0)
                            yield polled
                            return
                        got_lock = True

                    messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]

                    first_token = True
                    primary_failed = False
                    primary_err: Exception | None = None

                    try:
                        async for chunk in self._stream_model( # noqa
                            self._primary, messages
                        ):  # noqa

                            # Enforce SLA before emitting token
                            if budget:
                                budget.check(stage="llm.stream.before_yield")

                            if first_token:
                                _ttft.labels(model=self._primary).observe(
                                    time.monotonic() - t0
                                )
                                first_token = False

                            accumulated.append(chunk)
                            yield chunk

                    except (CircuitBreakerOpen, Exception) as exc:
                        primary_failed = True
                        primary_err = exc
                        log.warning(
                            "stream_primary_failed_using_fallback",
                            request_id=rid,
                            error=str(exc),
                            chars_streamed=sum(len(c) for c in accumulated),
                        )

                    if primary_failed:
                        accumulated.clear()
                        model_used = self._fallback
                        try:
                            async for chunk in self._stream_model( # noqa
                                self._fallback, messages
                            ):  # noqa

                                # Enforce SLA before emitting token
                                if budget:
                                    budget.check(stage="llm.stream.before_yield")

                                if first_token:
                                    _ttft.labels(model=self._fallback).observe(
                                        time.monotonic() - t0
                                    )
                                    first_token = False

                                accumulated.append(chunk)
                                yield chunk

                        except Exception as fallback_err:
                            _req_total.labels(
                                model=model_used, status="error", mode="stream"
                            ).inc()
                            span.set_status(StatusCode.ERROR, str(fallback_err))
                            raise RuntimeError(
                                f"Both stream models failed. Primary: {primary_err}. Fallback: {fallback_err}"
                            ) from fallback_err

                    full_text = "".join(accumulated)
                    latency = time.monotonic() - t0

                    if len(full_text) >= STREAM_CACHE_MIN_CHARS:
                        await self._cache_set(key, full_text)

                    _req_total.labels(
                        model=model_used, status="ok", mode="stream"
                    ).inc()
                    _latency.labels(model=model_used).observe(latency)
                    span.set_attribute("model", model_used)
                    span.set_attribute("latency_s", round(latency, 3))

                    log.info(
                        "stream_ok",
                        request_id=rid,
                        model=model_used,
                        latency_s=round(latency, 3),
                        chars=len(full_text),
                    )

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning(
                        "stream_budget_exceeded",
                        request_id=rid,
                        chars_streamed=sum(len(c) for c in accumulated),
                    )
                    raise

                except asyncio.CancelledError:
                    log.warning(
                        "stream_cancelled",
                        request_id=rid,
                        chars_streamed=sum(len(c) for c in accumulated),
                    )
                    raise

                finally:
                    if got_lock:
                        await self._release_lock(lk)
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    # ── stream with metadata ──────────────────────────────────────────────────

    async def stream_with_metadata(
        self, prompt: str, request_id: str | None = None
    ) -> AsyncIterator[str | dict[str, Any]]:
        """
        Yield str tokens followed by a single final metadata dict.
        Detect metadata with isinstance(item, dict).
        """
        rid = request_id or new_request_id()
        t0 = time.monotonic()
        chars = 0
        first_token_t: float | None = None

        async for token in self.stream(prompt, request_id=rid):
            if first_token_t is None:
                first_token_t = time.monotonic()
            chars += len(token)
            yield token

        yield {
            "type": "metadata",
            "request_id": rid,
            "ttft_s": round((first_token_t - t0) if first_token_t else 0.0, 3),
            "total_s": round(time.monotonic() - t0, 3),
            "chars": chars,
        }

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        batch_breaker = self._breakers.get(f"{self._primary}.batch")
        return ServiceHealthState(
            service="llm.local",
            healthy=batch_breaker.state != "OPEN" if batch_breaker else True,
            circuit_state=batch_breaker.state if batch_breaker else "CLOSED",
            inflight=self._inflight_batch + self._inflight_stream,
            degraded=degraded_mode.active,
        )

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        rid = state.get("request_id") or new_request_id()
        result = await self.generate(state.get("user_input", ""), request_id=rid)
        return {
            **state,
            "request_id": rid,
            "llm_response": result["response"],
            "llm_tokens": {
                "prompt": result["prompt_tokens"],
                "completion": result["completion_tokens"],
            },
            "llm_model_used": result["model_used"],
            "llm_cached": result["cached"],
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None


# ── remote LLM client ─────────────────────────────────────────────────────────


class RemoteLLMClient:
    """
    Calls a remote LLM microservice over HTTPS. Satisfies LLMNodeProtocol
    so VoiceGraph can swap between local and remote with zero graph changes.

    The remote service is expected to expose:
      POST /generate              → {"response": str, "model_used": str, ...}
      GET  /stream?prompt=...     → text/event-stream of token chunks
      GET  /health                → {"healthy": bool, "circuit_state": str, ...}

    OTel TraceContext and LatencyBudget are injected as HTTP headers on every
    outbound request so distributed traces stay stitched and the remote service
    can honour the remaining SLA budget.
    """

    def __init__(
        self,
        base_url: str = LLM_SERVICE_URL,
        api_key: str = LLM_SERVICE_API_KEY,
        timeout: float = LLM_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("LLM_SERVICE_URL must be set to use RemoteLLMClient.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._breaker = CircuitBreaker(name="llm:remote")
        self._inflight = 0

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        )

    def _request_headers(self) -> dict[str, str]:  # noqa
        """Merge OTel trace headers + remaining latency budget."""
        headers: dict[str, str] = {}
        inject_trace_headers(headers)
        budget = LatencyBudget.current()
        if budget:
            headers["X-Latency-Budget-Ms"] = budget.as_header_value()
            headers["X-Request-Id"] = current_request_id()
        return headers

    async def generate(
        self, prompt: str, request_id: str | None = None
    ) -> dict[str, Any]:
        rid = request_id or new_request_id()
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_llm.generate")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid

        with tracer.start_as_current_span("llm.remote.generate") as span:
            span.set_attribute("request_id", rid)
            self._inflight += 1
            t0 = time.monotonic()
            try:

                async def _call() -> dict[str, Any]:
                    resp = await self._http.post(
                        "/generate",
                        json={"prompt": prompt, "request_id": rid},
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()

                result = await self._breaker.call(
                    backoff_retry,
                    _call,
                    attempts=3,
                    base_delay=1.0,
                    exceptions=(Exception,),
                )
                span.set_attribute("latency_s", round(time.monotonic() - t0, 3))
                log.info(
                    "remote_llm_generate_ok",
                    request_id=rid,
                    latency_s=round(time.monotonic() - t0, 3),
                )
                return result

            except Exception as exc:
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_llm_generate_error", request_id=rid, error=str(exc))
                raise
            finally:
                self._inflight -= 1

    async def stream(
        self, prompt: str, request_id: str | None = None
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the remote service via SSE (text/event-stream).

        Each SSE event contains one token chunk. The stream ends with a
        special [DONE] event that the client strips before yielding.
        """
        rid = request_id or new_request_id()
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        budget = LatencyBudget.current()
        if budget:
            budget.check(stage="remote_llm.stream")

        headers = self._request_headers()
        headers["X-Request-Id"] = rid
        headers["Accept"] = "text/event-stream"

        with tracer.start_as_current_span("llm.remote.stream") as span:
            span.set_attribute("request_id", rid)

            async with bulkheads.acquire("llm.stream"):
                self._inflight += 1
                t0 = time.monotonic()
                first_token = True
                try:
                    async with self._http.stream(
                        "POST",
                        "/stream",
                        json={"prompt": prompt, "request_id": rid},
                        headers=headers,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break

                            # Enforce SLA before emitting token
                            if budget:
                                try:
                                    budget.check(stage="remote_llm.stream.before_yield")
                                except LatencyBudgetExceeded:
                                    _budget_exceeded.inc()
                                    raise

                            if first_token:
                                _ttft.labels(model="remote").observe(
                                    time.monotonic() - t0
                                )
                                first_token = False

                            yield data

                    span.set_attribute("latency_s", round(time.monotonic() - t0, 3))

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning("remote_llm_stream_budget_exceeded", request_id=rid)
                    raise

                except asyncio.CancelledError:
                    log.warning("remote_llm_stream_cancelled", request_id=rid)
                    raise

                except Exception as exc:
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("remote_llm_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    self._inflight -= 1

    async def health(self) -> ServiceHealthState:
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            return ServiceHealthState(
                service="llm.remote",
                healthy=data.get("healthy", False),
                circuit_state=self._breaker.state,
                inflight=self._inflight,
            )
        except Exception as exc:
            return ServiceHealthState(
                service="llm.remote",
                healthy=False,
                circuit_state=self._breaker.state,
                inflight=self._inflight,
                error=str(exc),
            )

    async def close(self) -> None:
        await self._http.aclose()


# ── node factory ──────────────────────────────────────────────────────────────


def get_llm_node() -> LLMNodeProtocol:
    """
    Return a RemoteLLMClient if LLM_SERVICE_URL is configured,
    otherwise the local LLMNode.

    This is the only import VoiceGraph needs — it never references the
    concrete classes directly, so local ↔ distributed switching is a
    single env-var change with zero code changes.
    """
    if LLM_SERVICE_URL:
        log.info("llm_using_remote_client", url=LLM_SERVICE_URL)
        return RemoteLLMClient()
    log.info("llm_using_local_node", model=PRIMARY_MODEL)
    return LLMNode()


# ── module-level singleton (backward-compatible) ──────────────────────────────

llm_node: LLMNodeProtocol = get_llm_node()

if __name__ == "__main__":
    import asyncio

    async def _smoke():
        node = get_llm_node()
        r = await node.generate("Explain binary search in one sentence.")
        print("Response:", r["response"])
        print("Model:", r["model_used"], "| Cached:", r["cached"])

        print("\nStreaming:")
        async for tok in node.stream("What is a hash map?"):
            print(tok, end="", flush=True)
        print()

        print("\nHealth:", await node.health())
        await node.close()

    asyncio.run(_smoke())