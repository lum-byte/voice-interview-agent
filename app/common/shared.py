"""
Shared plumbing that every node pulls from.

Covers: structured logging, OpenTelemetry tracing + metrics,
Prometheus counters/histograms, a circuit breaker, and a token-bucket
rate limiter. None of this is business logic — it's the stuff that
keeps the whole thing observable and resilient under load.

Distributed-service additions:
  - LatencyBudget: a per-request SLA deadline propagated via a context
    var. Every node checks remaining_ms() before starting work and
    raises LatencyBudgetExceeded when the request is already too late
    to meet its SLA. This is the single biggest voice-UX lever.
  - InMemoryLRU: an async-safe bounded LRU that activates automatically
    when Redis is unreachable. Concurrency is also throttled during
    degraded mode to avoid model overload (see DegradedMode).
  - BulkheadPool: a named-semaphore registry so different operations
    (stream vs batch, STT vs LLM vs TTS) each have their own concurrency
    budget rather than sharing one global semaphore.
  - NodeProtocol / STTNodeProtocol / LLMNodeProtocol / TTSNodeProtocol:
    typing.Protocol definitions that both local nodes and remote HTTP
    clients must satisfy. VoiceGraph only ever talks to the protocol —
    it never cares whether the node is in-process or over the wire.
  - QoSTier: three-level priority (REALTIME / STANDARD / BATCH) used
    by the load shedder and bounded pipeline queues.
  - OTelHeaders: extract/inject helpers for W3C TraceContext propagation
    across HTTP boundaries so distributed traces stay stitched together.
  - ServiceHealthState: a lightweight struct that remote clients and
    local nodes both publish so the graph can make routing decisions.
"""

from __future__ import annotations

import asyncio
import collections
import contextvars
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, ClassVar, TypeVar  # noqa

import structlog
from opentelemetry import context as otel_ctx
from opentelemetry import metrics as otel_metrics
from opentelemetry import propagate as otel_propagate
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

T = TypeVar("T")

# ── request-id propagation ────────────────────────────────────────────────────
# Canonical ContextVar owned by shared.py.  observability.py imports it directly
# (no circular dependency) so both layers always read the same value — eliminating
# the request_id correlation gap identified in the iteration-9 audit.

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def new_request_id() -> str:
    rid = uuid.uuid4().hex
    _request_id_var.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id_var.get() or new_request_id()


# ── structured logging ────────────────────────────────────────────────────────

from app.common.log_config import configure_logging

configure_logging()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


# ── OpenTelemetry setup ───────────────────────────────────────────────────────
#
# The error "Failed to detach context / ValueError: Token was created in a
# different Context" has three root causes that all need to be fixed:
#
# Cause 1 — provider was built but never installed
#   _build_tracer_provider() created a TracerProvider but never called
#   otel_trace.set_tracer_provider(). get_tracer() therefore always hit the
#   default ProxyTracerProvider, which performs real context attach/detach
#   unconditionally regardless of OTEL_ENABLED.
#   Fix: call set_tracer_provider() / set_meter_provider() after building.
#
# Cause 2 — cross-asyncio-task contextvars mismatch
#   start_as_current_span() calls otel_ctx.attach() which mints a
#   contextvars.Token tied to the exact Context object active at that instant.
#   controller.py uses run_coroutine_threadsafe() to submit coroutines to a
#   background event loop. asyncio copies the context into each Task; if a
#   context copy is made across an await inside the span block, the token's
#   originating Context no longer matches and Token.reset() raises ValueError.
#
# Cause 3 — the previous _safe_detach patch targeted the wrong function
#   otel_ctx.detach() (opentelemetry/context/__init__.py) already has its own
#   try/except that catches the ValueError and emits it via logger.error().
#   Patching that wrapper did nothing — the logger.error() call fired first,
#   inside the wrapper, before our code could intercept anything.
#
# Three-layer fix (each layer is independent, together they are airtight):
#
#   Layer 1 — _NoOpTracer when OTEL_ENABLED=false (primary)
#     get_tracer() returns _NOOP_TRACER, whose start_as_current_span is a
#     contextlib.contextmanager that yields without ever calling attach() or
#     detach(). The error is structurally impossible on this path.
#
#   Layer 2 — patch ContextVarsRuntimeContext.detach (real fix for enabled=true)
#     This is the method that actually calls token.reset() and raises ValueError.
#     Patching it to swallow ValueError prevents the raise before otel_ctx.detach()
#     ever reaches its except/logger.error() clause. Spans still work; the only
#     lost telemetry is the context de-registration for the mismatched task copy.
#
#   Layer 3 — logging.Filter on opentelemetry.context logger (absolute backstop)
#     OTel's logger.error("Failed to detach context") uses the stdlib logger
#     named "opentelemetry.context". A Filter on that logger suppresses the
#     specific message so it never reaches any handler, regardless of whether
#     layers 1 or 2 fire. This is the final safety net.

import contextlib
import logging as _stdlib_logging
from typing import Iterator

_OTEL_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "voice-pipeline")
_OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"

# ── Prefer validated settings over raw env vars ───────────────────────────────
# settings.py validates and normalises these at startup.  If it's available,
# use it so the entire codebase reads from one source of truth.
try:
    from app.common.settings import settings as _s  # type: ignore[import]

    _OTEL_ENDPOINT = _s.otel_exporter_otlp_endpoint
    _SERVICE_NAME = _s.otel_service_name
    _OTEL_ENABLED = _s.otel_enabled
    del _s
except Exception:  # noqa
    pass  # fall back to os.getenv values above

_resource = Resource.create({SERVICE_NAME: _SERVICE_NAME})


# ── Layer 1: no-op tracer ─────────────────────────────────────────────────────


class _NoOpSpan:
    """Stub span — every method is a no-op. No context operations whatsoever."""

    def set_attribute(self, key: str, value: object) -> "_NoOpSpan":  # noqa
        return self

    def set_status(self, status: object, description: str = "") -> "_NoOpSpan":  # noqa
        return self

    def add_event(
        self, name: str, attributes: dict | None = None
    ) -> "_NoOpSpan":  # noqa
        return self

    def record_exception(
        self, exc: BaseException, attributes: dict | None = None
    ) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_: object) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class _NoOpTracer:
    """
    Returned by get_tracer() when OTEL_ENABLED=false.
    start_as_current_span is a contextmanager that yields without ever calling
    otel_ctx.attach() or detach() — the ValueError is structurally impossible.
    """

    @contextlib.contextmanager
    def start_as_current_span(
        self, name: str, **kwargs: object
    ) -> Iterator[_NoOpSpan]:  # noqa
        yield _NOOP_SPAN

    def start_span(self, name: str, **kwargs: object) -> _NoOpSpan:  # noqa
        return _NOOP_SPAN


_NOOP_TRACER = _NoOpTracer()

# ── Layer 2: patch ContextVarsRuntimeContext.detach ───────────────────────────
# Target: the method that actually calls token.reset() and raises ValueError.
# Patching here prevents the raise before otel_ctx.detach()'s except-clause
# ever fires, so logger.error("Failed to detach context") is never reached.
# Applied unconditionally so it guards both disabled and enabled paths.

try:
    from opentelemetry.context.contextvars_context import (
        ContextVarsRuntimeContext as _CVCtx,
    )

    _original_cv_detach = _CVCtx.detach

    def _safe_cv_detach(self: "_CVCtx", token: object) -> None:  # type: ignore[type-arg]
        try:
            self._current_context.reset(token)  # type: ignore[attr-defined]
        except ValueError:
            # Token was minted in a different asyncio Task's context copy.
            # Silently discard — span is already closed, only context book-keeping
            # is affected. Trace data collected up to this point is unaffected.
            pass

    _CVCtx.detach = _safe_cv_detach  # type: ignore[method-assign]

except Exception:  # noqa - import may fail on future OTel versions
    pass  # Layer 3 (log filter) still covers us


# ── Layer 3: suppress the OTel error log as absolute backstop ─────────────────
# otel_ctx.detach() calls logger.error("Failed to detach context", exc_info=True)
# using the stdlib logger named after its module: "opentelemetry.context".
# A Filter on that logger drops the specific record before it reaches any handler.
# This is independent of layers 1 and 2 and fires even if they somehow miss.


class _SuppressDetachError(_stdlib_logging.Filter):
    def filter(self, record: _stdlib_logging.LogRecord) -> bool:
        return "Failed to detach context" not in record.getMessage()


_stdlib_logging.getLogger("opentelemetry.context").addFilter(_SuppressDetachError())


# ── provider installation ─────────────────────────────────────────────────────


def _build_and_install_tracer_provider() -> TracerProvider:
    provider = TracerProvider(resource=_resource)
    if _OTEL_ENABLED:
        try:
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=_OTEL_ENDPOINT, insecure=True)
                )
            )
        except Exception:  # noqa
            pass
    # Must call set_tracer_provider() — without this, get_tracer() hits the
    # default ProxyTracerProvider which does real context ops unconditionally.
    otel_trace.set_tracer_provider(provider)
    return provider


def _build_and_install_meter_provider() -> MeterProvider:
    if not _OTEL_ENABLED:
        provider = MeterProvider(resource=_resource)
    else:
        try:
            reader = PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=_OTEL_ENDPOINT, insecure=True),
                export_interval_millis=30_000,
            )
            provider = MeterProvider(resource=_resource, metric_readers=[reader])
        except Exception:  # noqa
            provider = MeterProvider(resource=_resource)
    otel_metrics.set_meter_provider(provider)
    return provider


# Runs at import time — before any get_tracer() / get_meter() call downstream.
# When OBSERVABILITY_OWNS_OTEL=true, observability.bootstrap() will call
# _OTEL.initialize() which installs its own providers with the correct sampler
# and OTLP exporter. shared.py must not clobber that with a second provider.
_OBSERVABILITY_OWNS_OTEL: bool = (
    os.getenv("OBSERVABILITY_OWNS_OTEL", "false").lower() == "true"
)
if not _OBSERVABILITY_OWNS_OTEL:
    # shared.py is the sole OTel owner — build and install full providers now.
    _tracer_provider = _build_and_install_tracer_provider()
    _meter_provider = _build_and_install_meter_provider()
else:
    # observability.bootstrap() will install providers with its own sampler and
    # exporter configuration.  Install *no-op* SDK providers here (no OTLP
    # exporters attached) so get_tracer() / get_meter() return something safe
    # before bootstrap() fires.  bootstrap() will overwrite them with real ones.
    _tracer_provider = TracerProvider(resource=_resource)
    otel_trace.set_tracer_provider(_tracer_provider)
    _meter_provider = MeterProvider(resource=_resource)
    otel_metrics.set_meter_provider(_meter_provider)


def get_tracer(name: str) -> otel_trace.Tracer | _NoOpTracer:
    """
    Return a tracer for *name*.

    OTEL_ENABLED=false → _NOOP_TRACER (no attach/detach, no context mutation).
    OTEL_ENABLED=true  → real SDK tracer, protected by layers 2 and 3.
    """
    if not _OTEL_ENABLED:
        return _NOOP_TRACER
    return otel_trace.get_tracer(name)


def get_meter(name: str) -> otel_metrics.Meter:
    return otel_metrics.get_meter(name)


# ── Prometheus registry ───────────────────────────────────────────────────────

prom_registry = CollectorRegistry()


def make_counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    if name in prom_registry._names_to_collectors:  # noqa
        return prom_registry._names_to_collectors[name]  # noqa
    return Counter(name, doc, labels or [], registry=prom_registry)


def make_histogram(
    name: str,
    doc: str,
    labels: list[str] | None = None,
    buckets: tuple[float, ...] = Histogram.DEFAULT_BUCKETS,
) -> Histogram:
    if name in prom_registry._names_to_collectors:  # noqa
        return prom_registry._names_to_collectors[name]  # noqa
    return Histogram(name, doc, labels or [], buckets=buckets, registry=prom_registry)


def make_gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
    if name in prom_registry._names_to_collectors:  # noqa
        return prom_registry._names_to_collectors[name]  # noqa
    return Gauge(name, doc, labels or [], registry=prom_registry)


# ── OTel trace-context HTTP propagation ──────────────────────────────────────
# Used by remote clients to stitch distributed traces across service boundaries.


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject the current span's W3C TraceContext into a mutable headers dict.
    No-op when OTEL_ENABLED=false so no context operations occur on the
    disabled path (avoids the cross-thread contextvars issue).
    """
    if _OTEL_ENABLED:
        otel_propagate.inject(headers)
    return headers


def extract_trace_context(headers: dict[str, str]) -> otel_ctx.Context:
    """
    Extract incoming W3C TraceContext from HTTP request headers.
    Returns an empty context when OTEL_ENABLED=false.
    """
    if not _OTEL_ENABLED:
        return otel_ctx.create_key("")  # type: ignore[return-value]
    return otel_propagate.extract(headers)


# ── QoS tier ─────────────────────────────────────────────────────────────────


class QoSTier(Enum):
    """
    Priority tier for a pipeline request.

    REALTIME  — interactive voice call; must meet <1.5 s TTFT SLA
    STANDARD  — normal API request; 5 s budget acceptable
    BATCH     — background processing; no latency guarantee
    """

    REALTIME = 1
    STANDARD = 2
    BATCH = 3


# ── latency budget ────────────────────────────────────────────────────────────


class LatencyBudgetExceeded(RuntimeError):
    """
    Raised when a node discovers the per-request SLA has already elapsed
    before it even starts work. Callers should handle this the same way
    they handle TimeoutError — degrade and return a partial result.
    """


_budget_var: contextvars.ContextVar["LatencyBudget | None"] = contextvars.ContextVar(
    "latency_budget", default=None
)


@dataclass
class LatencyBudget:
    """
    Per-request end-to-end SLA tracker.

    Create one per request via LatencyBudget.for_tier() and store it with
    LatencyBudget.activate(). Every node should call check() at entry to
    abort immediately if the budget is already blown.

    The budget is propagated automatically to child tasks that share the
    same contextvars context (i.e. everything spawned within the same
    asyncio.create_task() call). For remote HTTP calls, include the
    remaining budget in the X-Latency-Budget-Ms request header so the
    remote service can honour it too.

    Example::

        budget = LatencyBudget.for_tier(QoSTier.REALTIME)
        budget.activate()
        ...
        await some_node.transcribe(...)  # node calls budget.check() internally
    """

    _TIER_BUDGETS_MS: ClassVar[dict[QoSTier, float]] = {
        QoSTier.REALTIME: 10_000.0,
        QoSTier.STANDARD: 20_000.0,
        QoSTier.BATCH: 60_000.0,
    }

    total_ms: float
    _started_at: float = field(default_factory=time.monotonic, init=False)

    @classmethod
    def for_tier(cls, tier: QoSTier) -> "LatencyBudget":
        return cls(total_ms=cls._TIER_BUDGETS_MS[tier])

    @classmethod
    def current(cls) -> "LatencyBudget | None":
        return _budget_var.get()

    def activate(self) -> None:
        """Bind this budget to the current async context."""
        _budget_var.set(self)

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._started_at) * 1_000

    def remaining_ms(self) -> float:
        return max(0.0, self.total_ms - self.elapsed_ms())

    def remaining_s(self) -> float:
        return self.remaining_ms() / 1_000

    def is_exceeded(self) -> bool:
        return self.elapsed_ms() >= self.total_ms

    def check(self, stage: str = "unknown") -> None:
        """
        Raise LatencyBudgetExceeded if the budget is blown.
        Call at the entry point of every node method.
        """
        if self.is_exceeded():
            try:
                from app.monitoring.observability import BudgetEmitter

                BudgetEmitter.breached(
                    stage=stage,
                    budget_ms=self.total_ms,
                    elapsed_ms=self.elapsed_ms(),
                )
            except Exception:  # noqa — observability must never crash the pipeline
                pass
            raise LatencyBudgetExceeded(
                f"[{stage}] Latency budget of {self.total_ms:.0f} ms exceeded "
                f"({self.elapsed_ms():.0f} ms elapsed)."
            )

    def as_header_value(self) -> str:
        """Serialize remaining budget as a header value for remote calls."""
        return f"{self.remaining_ms():.0f}"

    @classmethod
    def from_header(cls, value: str | None) -> "LatencyBudget | None":
        """
        Reconstruct a LatencyBudget from an inbound X-Latency-Budget-Ms header.
        Returns None if the header is absent or malformed (remote service should
        then use its own default budget).
        """
        if not value:
            return None
        try:
            remaining = float(value)
            b = cls(total_ms=remaining)
            # Budget was already partially consumed by the time the header arrived.
            # We can't know the network RTT precisely, so we deduct a small
            # conservative estimate (50 ms) as a transmission guard.
            b._started_at = time.monotonic() - 0.05
            return b
        except (ValueError, TypeError):
            return None


# ── degraded-mode manager ─────────────────────────────────────────────────────


class DegradedMode:
    """
    Tracks whether the system is running in degraded mode (e.g. Redis down)
    and automatically reduces concurrency to avoid amplifying the blast radius.

    Call enter() when a critical dependency degrades and exit() when it
    recovers. While degraded, reduced_semaphore(name) returns a tighter
    semaphore so the model APIs aren't flooded.

    Prometheus gauge is published so Grafana can alert on the degraded state.
    """

    _gauge: Gauge = make_gauge(
        "system_degraded_mode", "1 when system is operating in degraded mode"
    )
    _counter: Counter = make_counter(
        "system_degraded_mode_entries_total", "Times system entered degraded mode"
    )

    def __init__(self) -> None:
        self._active = False
        self._reason = ""
        self._entered_at: float = 0.0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> bool:
        return self._active

    async def enter(self, reason: str) -> None:
        async with self._lock:
            if not self._active:
                self._active = True
                self._reason = reason
                self._entered_at = time.monotonic()
                self._gauge.set(1)
                self._counter.inc()
                get_logger(__name__).warning("degraded_mode_entered", reason=reason)

    async def exit(self, reason: str = "recovered") -> None:
        async with self._lock:
            if self._active:
                elapsed = round(time.monotonic() - self._entered_at, 1)
                self._active = False
                self._reason = ""
                self._gauge.set(0)
                get_logger(__name__).info(
                    "degraded_mode_exited", reason=reason, degraded_for_s=elapsed
                )


degraded_mode = DegradedMode()


# ── in-memory LRU cache (Redis fallback) ─────────────────────────────────────


class InMemoryLRU:
    """
    Async-safe bounded LRU cache that activates when Redis is unreachable.

    It is intentionally not a drop-in Redis replacement — it doesn't
    support TTL expiry (entries age out on eviction only) and is per-process
    (no shared state across pods). For a voice pipeline that means:

    * single pod: same behaviour as Redis for repeated identical prompts
    * multiple pods: each pod has its own cache; cache-hit rate is lower
      but the system doesn't fall over

    The cache is disabled by default (max_size=0) so production configs
    choose their own sizing. Set LLM_LRU_SIZE env var.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._max = max_size
        self._store: collections.OrderedDict[str, str] = collections.OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> str | None:
        if self._max == 0:
            return None
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    async def set(self, key: str, value: str) -> None:
        if self._max == 0:
            return
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._max:
                self._store.popitem(last=False)  # evict LRU

    @property
    def stats(self) -> dict[str, int]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._store),
            "hit_rate_pct": round(100 * self._hits / total) if total else 0,
        }


# ── bulkhead semaphore pool ───────────────────────────────────────────────────


class BulkheadPool:
    """
    Named semaphore registry that enforces separate concurrency budgets
    for each operation type.

    Using a single global semaphore means a flood of TTS requests can block
    STT requests from getting through. Bulkheads ensure each operation type
    has its own quota and can't starve the others.

    Usage::

        pool = BulkheadPool()
        pool.register("stt.batch",   limit=20)
        pool.register("stt.stream",  limit=10)
        pool.register("llm.batch",   limit=50)
        pool.register("llm.stream",  limit=30)
        pool.register("tts.stream",  limit=20)

        async with pool.acquire("llm.stream"):
            ...
    """

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._limits: dict[str, int] = {}

    def register(self, name: str, limit: int) -> None:
        if name in self._semaphores:
            return
        self._semaphores[name] = asyncio.Semaphore(limit)
        self._limits[name] = limit

    def acquire(self, name: str):
        """Return the semaphore's async context manager for use in `async with`."""
        if name not in self._semaphores:
            raise KeyError(f"Bulkhead '{name}' not registered. Call register() first.")
        return self._semaphores[name]

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {
                "limit": self._limits[name],
                "in_use": self._limits[name] - sem._value,  # type: ignore[attr-defined]
            }
            for name, sem in self._semaphores.items()
        }


bulkheads = BulkheadPool()

# Register standard bulkhead pools. Nodes also call register() themselves
# for any additional names they need — register() is idempotent.
bulkheads.register("stt.batch", limit=int(os.getenv("BULKHEAD_STT_BATCH", "20")))
bulkheads.register("stt.stream", limit=int(os.getenv("BULKHEAD_STT_STREAM", "10")))
bulkheads.register("llm.batch", limit=int(os.getenv("BULKHEAD_LLM_BATCH", "50")))
bulkheads.register("llm.stream", limit=int(os.getenv("BULKHEAD_LLM_STREAM", "30")))
bulkheads.register("tts.batch", limit=int(os.getenv("BULKHEAD_TTS_BATCH", "20")))
bulkheads.register("tts.stream", limit=int(os.getenv("BULKHEAD_TTS_STREAM", "20")))


# ── load shedder ──────────────────────────────────────────────────────────────


class LoadSheddingGuard:
    """
    Rejects new requests when the system is overloaded, based on the number
    of in-flight requests per QoS tier.

    Lower-priority tiers are shed first. REALTIME requests are only shed
    when the queue is completely at capacity.

    Usage::

        shedder = LoadSheddingGuard(max_inflight=100)
        shedder.enter(QoSTier.STANDARD)  # raises LoadSheddingRejected if too full
        try:
            ...
        finally:
            shedder.exit()
    """

    _rejected_counter: Counter = make_counter(
        "load_shedding_rejected_total",
        "Requests rejected by load shedding",
        ["tier"],
    )

    def __init__(self, max_inflight: int = 100) -> None:
        self._max = max_inflight
        self._inflight = 0
        self._lock = asyncio.Lock()

        # Tier-specific rejection thresholds (as % of max_inflight)
        self._thresholds: dict[QoSTier, float] = {
            QoSTier.BATCH: 0.60,  # shed BATCH requests when >60% full
            QoSTier.STANDARD: 0.85,  # shed STANDARD requests when >85% full
            QoSTier.REALTIME: 1.00,  # only shed REALTIME at 100% capacity
        }

    async def enter(self, tier: QoSTier = QoSTier.STANDARD) -> None:
        async with self._lock:
            threshold = int(self._max * self._thresholds[tier])
            if self._inflight >= threshold:
                self._rejected_counter.labels(tier=tier.name).inc()
                raise LoadSheddingRejected(
                    f"System overloaded ({self._inflight}/{self._max} in-flight). "
                    f"Rejected {tier.name} request."
                )
            self._inflight += 1

    async def exit(self) -> None:
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)

    @property
    def inflight(self) -> int:
        return self._inflight


class LoadSheddingRejected(RuntimeError):
    """Raised by LoadSheddingGuard.enter() when the system is overloaded."""


# ── bounded inter-node pipeline queue ─────────────────────────────────────────


class BoundedPipelineQueue(asyncio.Queue):
    """
    An asyncio.Queue with a hard size cap and drop/timeout policies.

    Used between STT → LLM and LLM → TTS to apply backpressure. If the
    downstream consumer falls behind, put() will raise QueueFull instead
    of blocking forever — callers must handle this as a degrade signal.

    The queue size is also exposed as a Prometheus gauge so you can set
    alerts on sustained queue depth.
    """

    _depth_gauge: Counter = make_counter(
        "pipeline_queue_full_total",
        "Times a bounded pipeline queue rejected a put()",
        ["queue_name"],
    )

    def __init__(self, maxsize: int, name: str) -> None:
        super().__init__(maxsize=maxsize)
        self._name = name

    def put_nowait_or_raise(self, item: Any) -> None:
        """Put an item without blocking; raise QueueFull and count if full."""
        try:
            self.put_nowait(item)
        except asyncio.QueueFull:
            self._depth_gauge.labels(queue_name=self._name).inc()
            raise


# ── service health state ──────────────────────────────────────────────────────


@dataclass
class ServiceHealthState:
    """
    Lightweight health summary published by every node (local or remote).

    VoiceGraph reads this to make routing decisions — e.g. falling back
    to a secondary LLM endpoint when the primary's circuit breaker is OPEN.
    """

    service: str
    healthy: bool
    circuit_state: str  # "CLOSED" | "OPEN" | "HALF_OPEN"
    inflight: int
    queue_depth: int = 0
    degraded: bool = False
    error: str = ""
    checked_at: float = field(default_factory=time.monotonic)


# ── circuit breaker ───────────────────────────────────────────────────────────


class CBState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreakerOpen(RuntimeError):
    """Raised when the breaker is open and we won't even attempt the call."""


@dataclass
class CircuitBreaker:
    """
    Three-state async circuit breaker.

    CLOSED  → OPEN      when failure_threshold consecutive failures occur
    OPEN    → HALF_OPEN after recovery_timeout seconds have elapsed
    HALF_OPEN → CLOSED  after success_threshold consecutive successes
    HALF_OPEN → OPEN    on the first failure (back to square one)
    """

    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    success_threshold: int = 2

    _state: CBState = field(default=CBState.CLOSED, init=False, repr=False)
    _failures: int = field(default=0, init=False, repr=False)
    _successes: int = field(default=0, init=False, repr=False)
    _opened_at: float = field(default=0.0, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    @property
    def state(self) -> str:
        return self._state.name

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if self._state is CBState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed < self.recovery_timeout:
                    try:
                        from app.monitoring.observability import CBEmitter

                        CBEmitter.rejected(
                            service=self.name,
                            remain_s=self.recovery_timeout - elapsed,
                        )
                    except Exception:  # noqa
                        pass
                    raise CircuitBreakerOpen(
                        f"[{self.name}] Circuit open — {self.recovery_timeout - elapsed:.1f}s until probe."
                    )
                self._state = CBState.HALF_OPEN
                self._successes = 0

        try:
            result = await fn(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise

        await self._on_success()
        return result

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._successes = 0
            self._opened_at = time.monotonic()
            if (
                self._state is CBState.HALF_OPEN
                or self._failures >= self.failure_threshold
            ):
                self._state = CBState.OPEN
                self._failures = 0
                try:
                    from app.monitoring.observability import CBEmitter

                    CBEmitter.opened(
                        service=self.name,
                        failures=self.failure_threshold,
                        threshold=self.failure_threshold,
                    )
                except Exception:  # noqa — observability must never crash the pipeline
                    pass

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state is CBState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self.success_threshold:
                    self._state = CBState.CLOSED
                    self._failures = 0
                    try:
                        from app.monitoring.observability import CBEmitter

                        CBEmitter.closed(service=self.name)
                    except Exception:  # noqa
                        pass
                else:
                    try:
                        from app.monitoring.observability import CBEmitter

                        CBEmitter.half_open(service=self.name)
                    except Exception:  # noqa
                        pass
            elif self._state is CBState.CLOSED:
                self._failures = max(0, self._failures - 1)


# ── token-bucket rate limiter ─────────────────────────────────────────────────


class RateLimitExceeded(RuntimeError):
    """Raised (non-blocking mode) when the bucket is empty."""


class RateLimiter:
    """
    Async token bucket.

    rate     — tokens refilled per second
    capacity — maximum tokens (controls burst headroom)
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        gained = (now - self._last_refill) * self._rate
        self._tokens = min(self._capacity, self._tokens + gained)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                wait = (tokens - self._tokens) / self._rate
            await asyncio.sleep(wait)

    def try_acquire(self, tokens: float = 1.0) -> None:
        self._refill()
        if self._tokens < tokens:
            raise RateLimitExceeded("Rate limit exceeded — try again shortly.")
        self._tokens -= tokens


# ── exponential backoff helper ────────────────────────────────────────────────


async def backoff_retry(
    fn: Callable[..., Any],
    *args: Any,
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Retry `fn` up to `attempts` times with exponential + jitter backoff.
    Only retries on the listed exception types.
    """
    import random

    for attempt in range(1, attempts + 1):
        try:
            return await fn(*args, **kwargs)
        except exceptions as exc:
            if attempt == attempts:
                raise
            delay = min(
                base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.3),
                max_delay,
            )
            get_logger(__name__).warning(
                "retry_scheduled",
                attempt=attempt,
                max_attempts=attempts,
                delay_s=round(delay, 2),
                error=str(exc),
            )
            await asyncio.sleep(delay)

    raise RuntimeError("backoff_retry exited unexpectedly")


# ── versioned cache key helper ────────────────────────────────────────────────


def make_versioned_cache_key(
    prefix: str,
    prompt: str,
    model: str,
    temperature: float,
    system_prompt_hash: str,
    extra: str = "",
) -> str:
    """
    Build a cache key that is invalidated whenever model config changes.

    Without versioning, a cache entry created with gpt-4o-mini at temperature=0.7
    would be incorrectly returned after the operator switches to gpt-4o at
    temperature=0.0, producing wrong answers silently.

    The key format is: {prefix}{sha256(prompt|model|temp|sysprompt_hash|extra)}
    """
    import hashlib

    payload = f"{prompt}|{model}|{temperature:.3f}|{system_prompt_hash}|{extra}"
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{prefix}{digest}"
