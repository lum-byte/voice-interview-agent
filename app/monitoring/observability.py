"""
observability.py — AI Pipeline Observability & Performance Monitor

Dual-mode structured logging, comprehensive Prometheus metrics, MongoDB
structured event storage, and Grafana dashboard provisioning for every
component in the voice-interview pipeline.

Unique identifier: session_id produced by session_store. Every event,
metric label, and MongoDB document is correlated by session_id first,
then request_id (per-turn), giving you complete session-level and
turn-level observability simultaneously.

Architecture
────────────
  Three-layer observability stack:

  Layer 1  — Rich console + JSON file (human debugging, standard mode)
  Layer 2  — Prometheus counters / histograms / gauges (real-time dashboards)
  Layer 3  — MongoDB structured event documents (historical analysis, per-session)

  All three layers are driven by a single emit() call from each component.
  Nothing is logged twice; no duplicate work.

Scope
─────
  Covers every observable event across:
    STT   — transcription, language confidence, audio quality, remote fallback
    LLM   — token usage, cache hit/miss, model fallback, stream vs batch
    TTS   — synthesis, apology fallback, chunk errors, S3 upload
    Eval  — scoring, budget, adaptive sampling, dedup
    Sess  — registration, IP change, suspension, TTL expiry
    Pipe  — stage retries, aborts, load shedding, SLA breaches
    Mem   — history compression, turn overflow, LRU fallback
    San   — sanitization warnings, prompt injection, truncation
    CB    — circuit breaker state transitions (all services)
    RL    — rate limiter exhaustion events
    BH    — bulkhead saturation events
    Ctrl  — PTT press/release, empty recording, pipeline interrupt

Environment variables
─────────────────────
  LOG_MODE          standard | verbose   (default: verbose)
  LOG_FILE          path to JSON log file
  MONGO_URI         MongoDB connection string
  MONGO_DB          database name        (default: ai_observability)
  MONGO_COLLECTION  collection name      (default: pipeline_events)
  MONGO_ENABLED     true | false         (default: true)
  MONGO_QUEUE_DEPTH events buffered in the writer queue (default: 4096)
  MONGO_BATCH_SIZE  events per insert_many() call      (default: 64)
  MONGO_FLUSH_MS    writer flush interval in ms         (default: 500)
  MONGO_TTL_DAYS    TTL index expiry in days            (default: 90)
  MONGO_RECONNECT_INTERVAL_S  seconds between reconnect attempts (default: 30)
  PROMETHEUS_PORT   HTTP port for /metrics (default: 9090)
  GRAFANA_OUT_DIR   directory for dashboard JSON (default: grafana)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Iterator, Literal, Optional  # type: ignore[unused-import]

import contextlib
import logging as _stdlib_logging

import structlog
from prometheus_client import (  # type: ignore[unused-import]
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
)
from rich.console import Console
from rich.theme import Theme

# OpenTelemetry — all imports are guarded so the file loads cleanly even
# when the otel packages are absent (OTEL_ENABLED=false never needs them).
try:
    from opentelemetry import baggage as otel_baggage
    from opentelemetry import context as otel_ctx
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import propagate as otel_propagate
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME as _OTEL_SVC_NAME
    from opentelemetry.sdk.resources import Resource as _OtelResource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import NonRecordingSpan, SpanContext, StatusCode
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator,
    )

    _OTEL_PACKAGES_AVAILABLE = True
except ImportError:
    otel_baggage = None
    otel_ctx = None
    otel_metrics = None
    otel_propagate = None
    otel_trace = None
    OTLPMetricExporter = None
    OTLPSpanExporter = None
    MeterProvider = None
    PeriodicExportingMetricReader = None
    _OTEL_SVC_NAME = None
    _OtelResource = None
    TracerProvider = None
    BatchSpanProcessor = None
    StatusCode = None
    TraceContextTextMapPropagator = None

    _OTEL_PACKAGES_AVAILABLE = False

# ── environment ───────────────────────────────────────────────────────────────

LOG_MODE: str = os.getenv("LOG_MODE", "verbose").lower()
LOG_FILE: str = os.getenv("LOG_FILE", "logs/voice_assistant.log")

if LOG_MODE == "both":
    LOG_MODE = "standard"

MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB: str = os.getenv("MONGO_DB", "ai_observability")
MONGO_COLLECTION: str = os.getenv("MONGO_COLLECTION", "pipeline_events")
MONGO_ENABLED: bool = os.getenv("MONGO_ENABLED", "true").lower() == "true"

PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
PROMETHEUS_ENABLED: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

GRAFANA_OUT_DIR: str = os.getenv("GRAFANA_OUT_DIR", "grafana")

# ── OpenTelemetry environment ─────────────────────────────────────────────────
#
# OTEL_ENABLED           — master switch. false = _NoOpTracer everywhere, no
#                          context operations, zero overhead. Default: false.
# OTEL_EXPORTER_OTLP_ENDPOINT — gRPC endpoint for both traces and metrics.
#                          Tempo: http://localhost:4317
#                          Jaeger: http://localhost:4317
#                          Signoz: http://localhost:4317
#                          Default: http://localhost:4317
# OTEL_SERVICE_NAME      — service name embedded in every span and metric.
#                          Default: voice-pipeline
# OTEL_TRACE_SAMPLE_RATE — fraction of requests to sample (0.0–1.0).
#                          1.0 = record every span. Default: 1.0
# OTEL_METRIC_INTERVAL_MS — how often OTel metrics are pushed to the backend.
#                          Independent from Prometheus scrape interval.
#                          Default: 30000

OTEL_ENABLED: bool = os.getenv("OTEL_ENABLED", "false").lower() == "true"
OTEL_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "voice-pipeline")
OTEL_SAMPLE_RATE: float = float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0"))
OTEL_METRIC_MS: int = int(os.getenv("OTEL_METRIC_INTERVAL_MS", "30000"))

# ── Settings integration: override env-var defaults with validated settings ────
# settings.py is the canonical config source for OTel.  We import lazily so
# observability can still be used standalone (e.g. in tests or scripts) without
# requiring the full settings dependency graph.
try:
    from app.common.settings import settings as _app_settings  # type: ignore[import]

    OTEL_ENABLED = _app_settings.otel_enabled
    OTEL_ENDPOINT = _app_settings.otel_exporter_otlp_endpoint
    OTEL_SERVICE_NAME = _app_settings.otel_service_name
except Exception:  # noqa
    pass  # remain with env-var defaults above


# ── OTel no-op stubs (active when OTEL_ENABLED=false) ────────────────────────
#
# Design mirrors shared.py's three-layer approach but is self-contained:
#
#   Layer 1 — _NoOpTracer / _NoOpSpan returned by _OtelLayer.tracer() when
#             OTEL_ENABLED=false.  start_as_current_span() is a plain
#             contextmanager that never calls otel_ctx.attach() or detach(),
#             making the ValueError from token mismatches structurally impossible.
#
#   Layer 2 — ContextVarsRuntimeContext.detach() patched to swallow ValueError.
#             This covers the rare case where OTEL_ENABLED=true and a span is
#             created inside an asyncio task that was copied from another context.
#
#   Layer 3 — stdlib logging.Filter on "opentelemetry.context" as absolute
#             backstop in case layers 1 and 2 both miss somehow.


class _NoOpSpan:
    """Zero-overhead stub span returned when OTEL_ENABLED=false."""

    def set_attribute(self, key: str, value: object) -> "_NoOpSpan":  # noqa
        return self

    def set_status(self, status: object, description: str = "") -> "_NoOpSpan":  # noqa
        return self

    def add_event(
        self, name: str, attributes: dict | None = None # noqa
    ) -> "_NoOpSpan":  # noqa
        return self

    def record_exception(
        self, exc: BaseException, attributes: dict | None = None
    ) -> None:
        pass

    def end(self) -> None:
        pass

    def is_recording(self) -> bool:  # noqa
        return False

    def get_span_context(self) -> None:  # noqa
        return None

    def __enter__(self) -> "_NoOpSpan":
        return self

    def __exit__(self, *_: object) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


class _NoOpTracer:
    """
    Zero-overhead stub tracer returned by _OtelLayer.tracer() when OTel is disabled.
    start_as_current_span never calls attach()/detach(), making context-mismatch
    errors structurally impossible regardless of how asyncio copies task contexts.
    """

    @contextlib.contextmanager
    def start_as_current_span(self, name: str, **_kwargs: object):  # noqa
        yield _NOOP_SPAN

    @contextlib.contextmanager
    def start_span(self, name: str, **_kwargs: object):  # noqa
        yield _NOOP_SPAN


_NOOP_TRACER = _NoOpTracer()

# Layer 2 — patch ContextVarsRuntimeContext.detach
try:
    from opentelemetry.context.contextvars_context import (
        ContextVarsRuntimeContext as _CVCtx,
    )

    def _safe_cv_detach(self: "_CVCtx", token: object) -> None:  # type: ignore[type-arg]
        try:
            self._current_context.reset(token)  # type: ignore[attr-defined]
        except ValueError:
            pass

    _CVCtx.detach = _safe_cv_detach  # type: ignore[method-assign]
except Exception:  # noqa
    pass


# Layer 3 — log filter backstop
class _SuppressDetachError(_stdlib_logging.Filter):
    def filter(self, record: _stdlib_logging.LogRecord) -> bool:
        return "Failed to detach context" not in record.getMessage()


_stdlib_logging.getLogger("opentelemetry.context").addFilter(_SuppressDetachError())


# ── event kind taxonomy ───────────────────────────────────────────────────────


class EventKind(str, Enum):
    # Pipeline lifecycle
    PIPELINE_START = "pipeline_start"
    PIPELINE_DONE = "pipeline_done"
    PIPELINE_FAILED = "pipeline_failed"
    PIPELINE_DEGRADED = "pipeline_degraded"
    PIPELINE_CANCELLED = "pipeline_cancelled"
    PIPELINE_LOAD_SHED = "pipeline_load_shed"
    PIPELINE_RETRY = "pipeline_retry"
    PIPELINE_ABORT = "pipeline_abort"

    # STT
    STT_START = "stt_start"
    STT_OK = "stt_ok"
    STT_FAILED = "stt_failed"
    STT_EMPTY_AUDIO = "stt_empty_audio"
    STT_LOW_CONFIDENCE = "stt_low_confidence"
    STT_AUDIO_TOO_SHORT = "stt_audio_too_short"
    STT_PATH_REJECTED = "stt_path_rejected"
    STT_REMOTE_FALLBACK = "stt_remote_fallback"
    STT_TRANSCRIPT_TRUNCATED = "stt_transcript_truncated"
    STT_STREAM_CHUNK = "stt_stream_chunk"

    # LLM
    LLM_START = "llm_start"
    LLM_OK = "llm_ok"
    LLM_FAILED = "llm_failed"
    LLM_CACHE_HIT = "llm_cache_hit"
    LLM_CACHE_MISS = "llm_cache_miss"
    LLM_CACHE_STAMPEDE = "llm_cache_stampede"
    LLM_MODEL_FALLBACK = "llm_model_fallback"
    LLM_TOKEN_BUDGET_NEAR = "llm_token_budget_near"
    LLM_STREAM_START = "llm_stream_start"
    LLM_STREAM_DONE = "llm_stream_done"
    LLM_RESPONSE_TRUNCATED = "llm_response_truncated"

    # TTS
    TTS_START = "tts_start"
    TTS_OK = "tts_ok"
    TTS_FAILED = "tts_failed"
    TTS_APOLOGY_FALLBACK = "tts_apology_fallback"
    TTS_CHUNK_ERROR = "tts_chunk_error"
    TTS_S3_UPLOAD_OK = "tts_s3_upload_ok"
    TTS_S3_UPLOAD_FAILED = "tts_s3_upload_failed"
    TTS_STREAM_START = "tts_stream_start"
    TTS_STREAM_DONE = "tts_stream_done"
    TTS_FILE_CLEANUP = "tts_file_cleanup"

    # Evaluation engine
    EVAL_START = "eval_start"
    EVAL_OK = "eval_ok"
    EVAL_FAILED = "eval_failed"
    EVAL_SKIPPED_TOO_SHORT = "eval_skipped_too_short"
    EVAL_SKIPPED_SAMPLING = "eval_skipped_sampling"
    EVAL_BUDGET_EXHAUSTED = "eval_budget_exhausted"
    EVAL_DEDUP_HIT = "eval_dedup_hit"
    EVAL_MODEL_FALLBACK = "eval_model_fallback"

    # Session
    SESSION_REGISTERED = "session_registered"
    SESSION_ENDED = "session_ended"
    SESSION_REJECTED = "session_rejected"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_IP_CHANGE = "session_ip_change"
    SESSION_IP_LIMIT = "session_ip_limit"
    SESSION_SUSPENDED = "session_suspended"
    SESSION_TTL_EXPIRY = "session_ttl_expiry"
    SESSION_LRU_HIT = "session_lru_hit"
    SESSION_DEGRADED = "session_degraded"

    # Conversation memory
    MEMORY_RESOLVE = "memory_resolve"
    MEMORY_COMMIT = "memory_commit"
    MEMORY_COMPRESSION = "memory_compression"
    MEMORY_OVERFLOW = "memory_overflow"
    MEMORY_LRU_FALLBACK = "memory_lru_fallback"

    # Sanitizer
    SANITIZE_OK = "sanitize_ok"
    SANITIZE_INJECTION = "sanitize_injection_detected"
    SANITIZE_TRUNCATED = "sanitize_truncated"
    SANITIZE_EMPTY = "sanitize_empty_result"
    SANITIZE_PATH_TRAVERSAL = "sanitize_path_traversal"

    # Circuit breaker
    CB_OPENED = "circuit_breaker_opened"
    CB_HALF_OPEN = "circuit_breaker_half_open"
    CB_CLOSED = "circuit_breaker_closed"
    CB_REJECTED = "circuit_breaker_rejected"

    # Rate limiter
    RL_ACQUIRED = "rate_limit_acquired"
    RL_EXHAUSTED = "rate_limit_exhausted"
    RL_WAITING = "rate_limit_waiting"

    # Bulkhead
    BH_ACQUIRED = "bulkhead_acquired"
    BH_SATURATED = "bulkhead_saturated"
    BH_RELEASED = "bulkhead_released"

    # Latency budget
    BUDGET_OK = "latency_budget_ok"
    BUDGET_BREACHED = "latency_budget_breached"
    BUDGET_NEAR = "latency_budget_near"

    # Controller
    CTRL_PTT_PRESS = "controller_ptt_press"
    CTRL_PTT_RELEASE = "controller_ptt_release"
    CTRL_EMPTY_RECORDING = "controller_empty_recording"
    CTRL_INTERRUPT = "controller_interrupt"
    CTRL_PIPELINE_ERROR = "controller_pipeline_error"
    CTRL_SHUTDOWN = "controller_shutdown"
    CTRL_SIGNAL = "controller_signal"

    # Redis / infrastructure
    REDIS_CONNECTED = "redis_connected"
    REDIS_DISCONNECTED = "redis_disconnected"
    REDIS_DEGRADED = "redis_degraded"
    REDIS_RECOVERED = "redis_recovered"

    # Transcript
    TRANSCRIPT_TURN = "transcript_turn"
    TRANSCRIPT_QUEUE_DROP = "transcript_queue_drop"

    # Health
    HEALTH_CHECK = "health_check"
    HEALTH_DEGRADED = "health_degraded"


# ── structured event dataclass ────────────────────────────────────────────────


@dataclass
class ObsEvent:
    """
    Single structured observability event. Every field is nullable so
    component-specific emitters only populate what they know.

    MongoDB document = asdict(event) with _id injected by the writer.
    Prometheus labels are derived from (kind, service, session_id).
    Log line is rendered from (kind, session_id, request_id, + fields).
    """

    kind: str
    service: str  # stt | llm | tts | session | pipeline | eval | memory | sanitize | cb | rl | bh | controller

    # Correlation keys
    session_id: str = ""  # primary; from session_store
    request_id: str = ""  # per-turn; from shared.new_request_id()

    # Timing
    ts: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    wall_s: float = 0.0

    # Status
    ok: bool = True
    error: str = ""
    error_type: str = ""  # exception class name

    # Stage info
    stage: str = ""
    retry_attempt: int = 0
    abort_reason: str = ""

    # STT fields
    audio_path: str = ""
    audio_bytes: int = 0
    audio_duration_s: float = 0.0
    transcript: str = ""
    transcript_chars: int = 0
    language: str = ""
    lang_confidence: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    truncated: bool = False
    remote_fallback: bool = False
    segment_count: int = 0

    # LLM fields
    model: str = ""
    model_used: str = ""  # actual model after fallback
    fallback_model: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_hit: bool = False
    streaming: bool = False
    temperature: float = 0.0
    response_chars: int = 0
    response_truncated: bool = False
    history_turns: int = 0

    # TTS fields
    voice: str = ""
    tts_format: str = ""
    input_chars: int = 0
    audio_output: str = ""
    audio_size_bytes: int = 0
    s3_uri: str = ""
    s3_ok: bool = False
    chunk_count: int = 0
    apology_used: bool = False

    # Eval fields
    eval_turn_idx: int = 0
    eval_score: float = 0.0
    eval_model: str = ""
    eval_tokens: int = 0
    eval_budget_used: int = 0
    eval_budget_cap: int = 0
    eval_skipped: bool = False
    eval_skip_reason: str = ""
    eval_dedup: bool = False
    eval_sampled: bool = False
    rubric_keys: list = field(default_factory=list)

    # Session fields
    ip_masked: str = ""
    ip_changes: int = 0
    ip_changes_max: int = 0
    session_ttl_s: int = 0
    session_turns: int = 0
    session_duration_s: float = 0.0
    session_reason: str = ""

    # Memory fields
    history_depth: int = 0
    compression_ratio: float = 0.0
    turns_pruned: int = 0

    # Sanitizer fields
    warnings: list = field(default_factory=list)
    original_chars: int = 0
    sanitized_chars: int = 0
    injection_pattern: str = ""

    # Circuit breaker fields
    cb_name: str = ""
    cb_state: str = ""  # CLOSED | OPEN | HALF_OPEN
    cb_failures: int = 0
    cb_threshold: int = 0

    # Rate limiter / bulkhead
    rl_name: str = ""
    rl_tokens_remain: float = 0.0
    bh_name: str = ""
    bh_inflight: int = 0
    bh_capacity: int = 0

    # Latency budget
    budget_total_ms: float = 0.0
    budget_remain_ms: float = 0.0
    budget_used_pct: float = 0.0

    # Pipeline fields
    pipeline_version: str = ""
    qos_tier: str = ""
    execution_mode: str = ""
    stage_latencies: dict = field(default_factory=dict)
    degraded: bool = False

    # Controller fields
    recording_s: float = 0.0
    ptt_key: str = ""

    # Infrastructure
    backend: str = ""  # redis | lru | s3 | local
    host: str = ""

    # Extra arbitrary fields (for future-proofing)
    extra: dict = field(default_factory=dict)


# ── ISO timestamp helper ──────────────────────────────────────────────────────


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
#  PROMETHEUS METRICS
# ═════════════════════════════════════════════════════════════════════════════


class _Metrics:
    """
    Single namespace for all Prometheus metrics across the pipeline.

    All metrics are registered once. Each service uses the same registry
    so Grafana sees a single consistent label space.
    """

    def __init__(self) -> None:
        if not PROMETHEUS_ENABLED:
            return

        # ── pipeline ─────────────────────────────────────────────────────────
        self.pipeline_total = Counter(
            "ai_pipeline_requests_total",
            "Total pipeline executions",
            ["status", "tier", "version", "mode"],
        )
        self.pipeline_latency = Histogram(
            "ai_pipeline_latency_seconds",
            "End-to-end pipeline wall-clock latency",
            ["tier", "mode"],
            buckets=(0.5, 1, 2, 3, 5, 8, 12, 20, 40, 90),
        )
        self.pipeline_active = Gauge(
            "ai_pipeline_inflight",
            "Pipelines currently in flight",
            ["tier"],
        )
        self.pipeline_stage_latency = Histogram(
            "ai_pipeline_stage_latency_seconds",
            "Per-stage wall-clock latency",
            ["stage"],
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
        )
        self.pipeline_stage_errors = Counter(
            "ai_pipeline_stage_errors_total",
            "Errors per pipeline stage",
            ["stage", "error_type"],
        )
        self.pipeline_stage_retries = Counter(
            "ai_pipeline_stage_retries_total",
            "Retry attempts per stage",
            ["stage"],
        )
        self.pipeline_load_shed = Counter(
            "ai_pipeline_load_shed_total",
            "Requests rejected by load shedder",
            ["tier"],
        )
        self.pipeline_aborted = Counter(
            "ai_pipeline_aborted_total",
            "Pipelines aborted due to non-recoverable fault",
            ["reason"],
        )
        self.pipeline_degraded = Counter(
            "ai_pipeline_degraded_total",
            "Pipelines completed via fallback/apology path",
            ["stage"],
        )
        self.pipeline_cancelled = Counter(
            "ai_pipeline_cancelled_total",
            "Pipelines cancelled (e.g. PTT interrupt)",
            ["stage"],
        )

        # ── STT ──────────────────────────────────────────────────────────────
        self.stt_requests = Counter(
            "ai_stt_requests_total",
            "STT transcription requests",
            ["status"],
        )
        self.stt_latency = Histogram(
            "ai_stt_latency_seconds",
            "STT transcription latency",
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )
        self.stt_audio_duration = Histogram(
            "ai_stt_audio_duration_seconds",
            "Duration of audio submitted to STT",
            buckets=(0.5, 1, 2, 5, 10, 30, 60, 120),
        )
        self.stt_audio_bytes = Histogram(
            "ai_stt_audio_bytes",
            "File size of audio submitted to STT",
            buckets=(4096, 16384, 65536, 262144, 1048576, 4194304),
        )
        self.stt_lang_confidence = Histogram(
            "ai_stt_language_confidence",
            "Whisper language detection confidence score",
            ["language"],
            buckets=(0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0),
        )
        self.stt_transcript_chars = Histogram(
            "ai_stt_transcript_chars",
            "Character count of returned transcript",
            buckets=(10, 50, 100, 200, 500, 1000, 2000),
        )
        self.stt_empty_audio = Counter(
            "ai_stt_empty_audio_total",
            "STT requests with empty or silent audio",
        )
        self.stt_low_confidence = Counter(
            "ai_stt_low_confidence_total",
            "STT results with language confidence below threshold",
        )
        self.stt_remote_fallback = Counter(
            "ai_stt_remote_fallback_total",
            "STT requests routed to remote fallback endpoint",
        )
        self.stt_path_rejected = Counter(
            "ai_stt_path_rejected_total",
            "Audio paths rejected due to security validation",
        )

        # ── LLM ──────────────────────────────────────────────────────────────
        self.llm_requests = Counter(
            "ai_llm_requests_total",
            "LLM generation requests",
            ["status", "model", "mode"],
        )
        self.llm_latency = Histogram(
            "ai_llm_latency_seconds",
            "LLM generation latency (first-token for stream)",
            ["model", "mode"],
            buckets=(0.1, 0.5, 1, 2, 5, 10, 20, 45),
        )
        self.llm_tokens_prompt = Histogram(
            "ai_llm_tokens_prompt",
            "Prompt token count per LLM request",
            ["model"],
            buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000),
        )
        self.llm_tokens_completion = Histogram(
            "ai_llm_tokens_completion",
            "Completion token count per LLM request",
            ["model"],
            buckets=(10, 25, 50, 100, 200, 500, 1000, 2000),
        )
        self.llm_tokens_total = Counter(
            "ai_llm_tokens_total",
            "Cumulative tokens consumed by LLM",
            ["model", "kind"],  # kind: prompt | completion
        )
        self.llm_cache_hits = Counter(
            "ai_llm_cache_hits_total",
            "LLM requests served from Redis/LRU cache",
            ["backend"],
        )
        self.llm_cache_misses = Counter(
            "ai_llm_cache_misses_total",
            "LLM requests that hit the model (not cached)",
        )
        self.llm_cache_stampede = Counter(
            "ai_llm_cache_stampede_total",
            "Redis SETNX lock collisions (cache stampede prevented)",
        )
        self.llm_model_fallback = Counter(
            "ai_llm_model_fallback_total",
            "LLM requests that fell back from primary to secondary model",
            ["primary", "fallback"],
        )
        self.llm_response_truncated = Counter(
            "ai_llm_response_truncated_total",
            "LLM responses truncated before reaching TTS",
        )
        self.llm_history_turns = Histogram(
            "ai_llm_history_turns",
            "Conversation history depth at time of LLM call",
            buckets=(0, 1, 2, 5, 10, 15, 20),
        )

        # ── TTS ──────────────────────────────────────────────────────────────
        self.tts_requests = Counter(
            "ai_tts_requests_total",
            "TTS synthesis requests",
            ["status", "voice"],
        )
        self.tts_latency = Histogram(
            "ai_tts_latency_seconds",
            "TTS synthesis latency",
            ["voice"],
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )
        self.tts_input_chars = Histogram(
            "ai_tts_input_chars",
            "Character count fed to TTS",
            buckets=(10, 50, 100, 200, 500, 1000, 2000, 4000),
        )
        self.tts_chunk_errors = Counter(
            "ai_tts_chunk_errors_total",
            "TTS synthesis chunk stitching errors",
        )
        self.tts_apology_fallback = Counter(
            "ai_tts_apology_fallback_total",
            "TTS requests that returned pre-rendered apology audio",
        )
        self.tts_s3_uploads = Counter(
            "ai_tts_s3_uploads_total",
            "TTS audio S3 upload attempts",
            ["status"],
        )
        self.tts_file_cleanups = Counter(
            "ai_tts_file_cleanups_total",
            "Local TTS cache files removed by background cleanup",
        )

        # ── Eval ─────────────────────────────────────────────────────────────
        self.eval_requests = Counter(
            "ai_eval_requests_total",
            "Evaluation engine scoring attempts",
            ["status"],
        )
        self.eval_latency = Histogram(
            "ai_eval_latency_seconds",
            "Evaluation engine scoring latency",
            buckets=(0.5, 1, 2, 5, 10, 20, 45),
        )
        self.eval_scores = Histogram(
            "ai_eval_scores",
            "Distribution of evaluation scores (0-10)",
            buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        )
        self.eval_tokens = Counter(
            "ai_eval_tokens_total",
            "Cumulative tokens consumed by evaluation engine",
        )
        self.eval_budget_exhausted = Counter(
            "ai_eval_budget_exhausted_total",
            "Sessions that exhausted the evaluation token budget",
        )
        self.eval_skipped = Counter(
            "ai_eval_skipped_total",
            "Evaluation calls skipped (sampling, short answer, dedup)",
            ["reason"],
        )
        self.eval_model_fallback = Counter(
            "ai_eval_model_fallback_total",
            "Evaluation requests that fell back to secondary model",
        )

        # ── Session ───────────────────────────────────────────────────────────
        self.session_active = Gauge(
            "ai_session_active",
            "Sessions currently holding an IP lock",
        )
        self.session_created = Counter(
            "ai_session_created_total",
            "New sessions registered",
        )
        self.session_ended = Counter(
            "ai_session_ended_total",
            "Sessions ended",
            ["reason"],
        )
        self.session_rejected = Counter(
            "ai_session_rejected_total",
            "Session registration rejections",
            ["reason"],
        )
        self.session_turns = Histogram(
            "ai_session_turns",
            "Number of conversation turns per completed session",
            buckets=(1, 2, 5, 10, 15, 20, 30),
        )
        self.session_duration = Histogram(
            "ai_session_duration_seconds",
            "Wall-clock duration of completed sessions",
            buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600),
        )
        self.session_ip_changes = Counter(
            "ai_session_ip_changes_total",
            "IP address changes accepted within sessions",
        )
        self.session_ip_limit = Counter(
            "ai_session_ip_limit_total",
            "Sessions terminated due to IP change limit",
        )
        self.session_lru_hits = Counter(
            "ai_session_lru_hits_total",
            "Session reads served from in-process LRU (Redis down)",
        )

        # ── Memory ────────────────────────────────────────────────────────────
        self.memory_compressions = Counter(
            "ai_memory_compressions_total",
            "Conversation history compression events",
        )
        self.memory_compression_ratio = Histogram(
            "ai_memory_compression_ratio",
            "Ratio of turns preserved after compression (0-1)",
            buckets=(0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
        )
        self.memory_lru_fallbacks = Counter(
            "ai_memory_lru_fallback_total",
            "Memory reads served from LRU (session_store degraded)",
        )

        # ── Sanitizer ─────────────────────────────────────────────────────────
        self.sanitize_total = Counter(
            "ai_sanitize_total",
            "Sanitization passes",
            ["status"],
        )
        self.sanitize_injection = Counter(
            "ai_sanitize_injection_total",
            "Prompt injection patterns detected in LLM output",
        )
        self.sanitize_truncated = Counter(
            "ai_sanitize_truncated_total",
            "Texts truncated by sanitizer char cap",
        )
        self.sanitize_reduction = Histogram(
            "ai_sanitize_char_reduction",
            "Character reduction ratio after sanitization",
            buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0),
        )

        # ── Circuit Breaker ───────────────────────────────────────────────────
        self.cb_opened = Counter(
            "ai_circuit_breaker_opened_total",
            "Circuit breaker OPEN transitions",
            ["service"],
        )
        self.cb_half_open = Counter(
            "ai_circuit_breaker_half_open_total",
            "Circuit breaker HALF_OPEN probe attempts",
            ["service"],
        )
        self.cb_closed = Counter(
            "ai_circuit_breaker_closed_total",
            "Circuit breaker CLOSED (recovery) transitions",
            ["service"],
        )
        self.cb_rejected = Counter(
            "ai_circuit_breaker_rejected_total",
            "Calls rejected because breaker is OPEN",
            ["service"],
        )
        self.cb_state = Gauge(
            "ai_circuit_breaker_state",
            "Circuit breaker state: 0=CLOSED 1=HALF_OPEN 2=OPEN",
            ["service"],
        )

        # ── Rate Limiter ──────────────────────────────────────────────────────
        self.rl_acquired = Counter(
            "ai_rate_limit_acquired_total",
            "Rate limiter tokens successfully acquired",
            ["service"],
        )
        self.rl_exhausted = Counter(
            "ai_rate_limit_exhausted_total",
            "Rate limiter rejections (bucket empty)",
            ["service"],
        )
        self.rl_wait_seconds = Histogram(
            "ai_rate_limit_wait_seconds",
            "Time spent waiting for rate limiter token",
            ["service"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
        )

        # ── Bulkhead ──────────────────────────────────────────────────────────
        self.bh_inflight = Gauge(
            "ai_bulkhead_inflight",
            "Concurrent operations currently holding a bulkhead slot",
            ["name"],
        )
        self.bh_saturated = Counter(
            "ai_bulkhead_saturated_total",
            "Attempts that found the bulkhead at full capacity",
            ["name"],
        )
        self.bh_wait_seconds = Histogram(
            "ai_bulkhead_wait_seconds",
            "Time spent waiting for bulkhead slot",
            ["name"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
        )

        # ── Latency Budget ────────────────────────────────────────────────────
        self.budget_breached = Counter(
            "ai_latency_budget_breached_total",
            "Stage executions aborted because SLA budget was blown",
            ["stage"],
        )
        self.budget_remain_pct = Histogram(
            "ai_latency_budget_remaining_pct",
            "Remaining SLA budget at stage start (0-100)",
            ["stage"],
            buckets=(0, 5, 10, 20, 30, 50, 70, 90, 100),
        )

        # ── Controller ────────────────────────────────────────────────────────
        self.ctrl_ptt_presses = Counter(
            "ai_controller_ptt_presses_total",
            "Total PTT key press-and-release cycles",
        )
        self.ctrl_interrupts = Counter(
            "ai_controller_interrupts_total",
            "Pipelines interrupted by PTT press while active",
        )
        self.ctrl_empty_recordings = Counter(
            "ai_controller_empty_recordings_total",
            "PTT releases where audio was too short or empty",
        )
        self.ctrl_recording_duration = Histogram(
            "ai_controller_recording_duration_seconds",
            "Microphone hold duration per PTT press",
            buckets=(0.5, 1, 2, 5, 10, 30),
        )
        self.ctrl_pipeline_errors = Counter(
            "ai_controller_pipeline_errors_total",
            "Unhandled pipeline exceptions surfaced to controller",
        )

        # ── Redis / Infrastructure ────────────────────────────────────────────
        self.redis_disconnects = Counter(
            "ai_redis_disconnects_total",
            "Redis connection failures",
        )
        self.redis_recovered = Counter(
            "ai_redis_recovered_total",
            "Redis reconnections after outage",
        )
        self.redis_degraded_mode = Gauge(
            "ai_redis_degraded_mode",
            "1 when operating without Redis (LRU fallback active)",
        )

        # ── Transcript ────────────────────────────────────────────────────────
        self.transcript_turns = Counter(
            "ai_transcript_turns_total",
            "Conversation turns written to transcript sinks",
        )
        self.transcript_drops = Counter(
            "ai_transcript_drops_total",
            "Transcript entries dropped due to queue overflow",
        )

        # ── Observability infrastructure ──────────────────────────────────────
        # Visibility into the observability pipeline itself. If Mongo is struggling,
        # these counters surface the problem in Grafana before events silently vanish.
        self.mongo_event_drops = Counter(
            "ai_mongo_event_drops_total",
            "Structured events dropped because the MongoDB write queue was full",
        )
        self.mongo_reconnects = Counter(
            "ai_mongo_reconnects_total",
            "MongoDB reconnection attempts made by the writer thread after a write failure",
            ["status"],  # "success" | "failure"
        )


_METRICS: Optional[_Metrics] = None


def get_metrics() -> _Metrics:
    global _METRICS
    if _METRICS is None:
        _METRICS = _Metrics()
    return _METRICS


# ═════════════════════════════════════════════════════════════════════════════
#  MONGODB WRITER
# ═════════════════════════════════════════════════════════════════════════════


class _MongoWriter:
    """
    Non-blocking MongoDB event writer.

    Events are pushed onto a bounded queue. A single daemon thread drains
    the queue and performs bulk inserts. The main event loop never blocks
    on I/O. If MongoDB is unavailable the queue accumulates up to
    MONGO_QUEUE_DEPTH events; older events are dropped with a warning.

    Documents are structured as:

        {
            "_id":        <ObjectId>,
            "kind":       "stt_ok",
            "service":    "stt",
            "session_id": "<session_id>",
            "request_id": "<request_id>",
            "ts_iso":     "2025-01-01T12:00:00.000Z",
            ... (all ObsEvent fields)
        }

    Indexing recommendations (run once against your MongoDB):

        db.pipeline_events.createIndex({ "session_id": 1, "ts": -1 })
        db.pipeline_events.createIndex({ "kind": 1, "ts": -1 })
        db.pipeline_events.createIndex({ "service": 1, "ts": -1 })
        db.pipeline_events.createIndex({ "request_id": 1 })
        db.pipeline_events.createIndex({ "ts": -1 }, { expireAfterSeconds: 7776000 })
    """

    MONGO_QUEUE_DEPTH: int = int(os.getenv("MONGO_QUEUE_DEPTH", "4096"))
    MONGO_BATCH_SIZE: int = int(os.getenv("MONGO_BATCH_SIZE", "64"))
    MONGO_FLUSH_MS: int = int(os.getenv("MONGO_FLUSH_MS", "500"))

    def __init__(self) -> None:
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=self.MONGO_QUEUE_DEPTH)
        self._thread: threading.Thread | None = None
        self._client: Any = None
        self._col: Any = None
        self._drops: int = 0
        self._started: bool = False
        self._lock: threading.Lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            try:
                from pymongo import MongoClient
                from pymongo.errors import ConnectionFailure

                self._client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
                self._client.admin.command("ping")
                db = self._client[MONGO_DB]
                self._col = db[MONGO_COLLECTION]
                self._ensure_indexes()
                self._thread = threading.Thread(
                    target=self._drain_loop, daemon=True, name="obs-mongo-writer"
                )
                self._thread.start()
                self._started = True
            except Exception as exc:
                # MongoDB is optional — pipeline never halts on its absence
                import sys

                print(f"[observability] MongoDB unavailable: {exc}", file=sys.stderr)

    def _ensure_indexes(self) -> None:
        try:
            self._col.create_index([("session_id", 1), ("ts", -1)])
            self._col.create_index([("kind", 1), ("ts", -1)])
            self._col.create_index([("service", 1), ("ts", -1)])
            self._col.create_index([("request_id", 1)])
            self._col.create_index(
                [("ts", -1)],
                expireAfterSeconds=int(os.getenv("MONGO_TTL_DAYS", "90")) * 86400,
            )
        except Exception:  # noqa
            pass

    # ── Reconnect (called from writer thread on insert failure) ───────────────

    _RECONNECT_INTERVAL_S: float = float(os.getenv("MONGO_RECONNECT_INTERVAL_S", "30"))

    def _try_reconnect(self) -> bool:
        """
        Attempt to re-establish the MongoDB connection from the writer thread.

        Called automatically by _drain_loop when an insert_many() fails.
        Throttled by _RECONNECT_INTERVAL_S to avoid hammering a dead server.

        Returns True on success, False if the reconnect attempt failed.
        Updates ai_mongo_reconnects_total Prometheus counter (status=success|failure).
        """
        import sys

        try:
            from pymongo import MongoClient

            client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            old_client = self._client
            self._client = client
            self._col = client[MONGO_DB][MONGO_COLLECTION]
            self._ensure_indexes()
            try:
                old_client.close()
            except Exception:  # noqa
                pass
            print("[observability] MongoDB reconnected successfully.", file=sys.stderr)
            try:
                get_metrics().mongo_reconnects.labels(status="success").inc()
            except Exception:  # noqa
                pass
            return True
        except Exception as exc:
            print(f"[observability] MongoDB reconnect failed: {exc}", file=sys.stderr)
            self._col = None  # mark as unreachable so the drain loop skips inserts
            try:
                get_metrics().mongo_reconnects.labels(status="failure").inc()
            except Exception:  # noqa
                pass
            return False

    def push(self, doc: dict) -> None:
        if not self._started:
            return
        try:
            self._queue.put_nowait(doc)
        except queue.Full:
            self._drops += 1
            # Increment the Prometheus counter so Grafana can alert on drops.
            try:
                get_metrics().mongo_event_drops.inc()
            except Exception:  # noqa
                pass
            if self._drops % 100 == 1:
                import sys

                print(
                    f"[observability] Mongo queue full — {self._drops} events dropped",
                    file=sys.stderr,
                )

    def _drain_loop(self) -> None:
        flush_interval = self.MONGO_FLUSH_MS / 1000.0
        batch: list[dict] = []
        deadline = time.monotonic() + flush_interval
        last_reconnect_s = 0.0  # monotonic timestamp of last reconnect attempt

        while True:
            try:
                timeout = max(0.001, deadline - time.monotonic())
                doc = self._queue.get(timeout=timeout)
                batch.append(doc)
                self._queue.task_done()
            except queue.Empty:
                pass

            if time.monotonic() >= deadline or len(batch) >= self.MONGO_BATCH_SIZE:
                if batch and self._col is not None:
                    try:
                        self._col.insert_many(batch, ordered=False)
                        batch = []  # only clear on success so we can retry
                    except Exception:  # noqa
                        # Insert failed — Mongo may be down or restarted.
                        now = time.monotonic()
                        if now - last_reconnect_s >= self._RECONNECT_INTERVAL_S:
                            last_reconnect_s = now
                            if self._try_reconnect():
                                # Retry the batch immediately on a fresh connection.
                                try:
                                    self._col.insert_many(batch, ordered=False)
                                    batch = []
                                except Exception:  # noqa — will retry next cycle
                                    pass
                            else:
                                # Reconnect failed; drop the batch to bound memory.
                                batch = []
                        else:
                            # Reconnect was tried recently; drop batch to avoid OOM.
                            batch = []
                elif not batch or self._col is None:
                    batch = []
                deadline = time.monotonic() + flush_interval


_MONGO: _MongoWriter = _MongoWriter()


# ═════════════════════════════════════════════════════════════════════════════
#  OPENTELEMETRY LAYER
# ═════════════════════════════════════════════════════════════════════════════


class _OtelLayer:
    """
    Owns the TracerProvider, MeterProvider, and per-service tracer registry.

    Architecture
    ────────────
    Tracing  — BatchSpanProcessor → OTLPSpanExporter (gRPC) → Tempo / Jaeger
    Metrics  — PeriodicExportingMetricReader → OTLPMetricExporter → Prometheus
               (OTel metrics run alongside Prometheus, not instead of it)

    Each pipeline service gets its own named tracer so Tempo's service map
    shows STT, LLM, TTS, Eval, Session as distinct instrumented libraries
    under the same service.name root.

    Session correlation
    ───────────────────
    session_id is stored in W3C Baggage on every pipeline span so it
    propagates across HTTP boundaries to remote STT / LLM / TTS services
    automatically.  extract_context() at remote ingress restores it.

    Log-trace correlation
    ─────────────────────
    _inject_trace_fields() returns {"trace_id": ..., "span_id": ...} from
    the currently active span. configure_logging() injects these into every
    structlog record so Loki / Datadog can pivot from a log line to its
    parent trace with one click.

    OTel metrics vs Prometheus
    ──────────────────────────
    Both are active simultaneously.  Prometheus is the pull-based scrape
    target for Grafana dashboards.  OTel metrics are pushed via OTLP to
    the same backend (or a separate one like Signoz) for exemplar-linked
    histogram queries that tie metric anomalies to specific trace IDs.

    Span naming convention
    ──────────────────────
    Pipeline root span:   "ai.pipeline.run" / "ai.pipeline.stream"
    STT span:             "ai.stt.transcribe"
    LLM span:             "ai.llm.generate" / "ai.llm.stream"
    TTS span:             "ai.tts.synthesize" / "ai.tts.stream"
    Eval span:            "ai.eval.score"
    Session ops:          "ai.session.<operation>"
    Memory ops:           "ai.memory.<operation>"
    Sanitizer:            "ai.sanitize"

    Attribute namespacing follows OpenTelemetry semantic conventions where
    applicable (gen_ai.*, http.*, db.*) and uses "ai.*" for domain-specific
    attributes not yet covered by the OTel spec.
    """

    # Semantic-convention attribute keys used on spans
    ATTR_SESSION_ID = "ai.session.id"
    ATTR_REQUEST_ID = "ai.request.id"
    ATTR_QOS_TIER = "ai.pipeline.qos_tier"
    ATTR_EXEC_MODE = "ai.pipeline.mode"
    ATTR_GRAPH_VERSION = "ai.pipeline.version"
    ATTR_STAGE = "ai.pipeline.stage"
    ATTR_RETRY = "ai.pipeline.retry_attempt"
    ATTR_DEGRADED = "ai.pipeline.degraded"
    ATTR_ABORT_REASON = "ai.pipeline.abort_reason"
    ATTR_AUDIO_PATH = "ai.stt.audio_path"
    ATTR_AUDIO_BYTES = "ai.stt.audio_bytes"
    ATTR_AUDIO_DURATION = "ai.stt.audio_duration_s"
    ATTR_LANGUAGE = "ai.stt.language"
    ATTR_LANG_CONF = "ai.stt.language_confidence"
    ATTR_AVG_LOGPROB = "ai.stt.avg_logprob"
    ATTR_NO_SPEECH = "ai.stt.no_speech_prob"
    ATTR_TRANSCRIPT_CHARS = "ai.stt.transcript_chars"
    ATTR_REMOTE_FALLBACK = "ai.stt.remote_fallback"
    ATTR_GEN_AI_SYSTEM = "gen_ai.system"
    ATTR_GEN_AI_MODEL = "gen_ai.request.model"
    ATTR_GEN_AI_MODEL_USED = "gen_ai.response.model"
    ATTR_GEN_AI_STREAMING = "gen_ai.streaming"
    ATTR_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    ATTR_COMPL_TOKENS = "gen_ai.usage.completion_tokens"
    ATTR_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
    ATTR_CACHE_HIT = "ai.llm.cache_hit"
    ATTR_FALLBACK_MODEL = "ai.llm.fallback_model"
    ATTR_HISTORY_TURNS = "ai.llm.history_turns"
    ATTR_RESP_CHARS = "ai.llm.response_chars"
    ATTR_TTS_VOICE = "ai.tts.voice"
    ATTR_TTS_FORMAT = "ai.tts.format"
    ATTR_TTS_INPUT_CHARS = "ai.tts.input_chars"
    ATTR_TTS_CHUNK_COUNT = "ai.tts.chunk_count"
    ATTR_TTS_APOLOGY = "ai.tts.apology_fallback"
    ATTR_S3_URI = "ai.tts.s3_uri"
    ATTR_EVAL_TURN_IDX = "ai.eval.turn_index"
    ATTR_EVAL_MODEL = "ai.eval.model"
    ATTR_EVAL_SCORE = "ai.eval.score"
    ATTR_EVAL_TOKENS = "ai.eval.tokens_used"
    ATTR_EVAL_SKIPPED = "ai.eval.skipped"
    ATTR_EVAL_SKIP_REASON = "ai.eval.skip_reason"
    ATTR_EVAL_BUDGET_USED = "ai.eval.budget_used"
    ATTR_IP_MASKED = "ai.session.ip_masked"
    ATTR_IP_CHANGES = "ai.session.ip_changes"
    ATTR_SESSION_TURNS = "ai.session.turns"
    ATTR_SESSION_REASON = "ai.session.reason"
    ATTR_HISTORY_DEPTH = "ai.memory.history_depth"
    ATTR_COMPR_RATIO = "ai.memory.compression_ratio"
    ATTR_ORIG_CHARS = "ai.sanitize.original_chars"
    ATTR_SANI_CHARS = "ai.sanitize.sanitized_chars"
    ATTR_INJECTION_PAT = "ai.sanitize.injection_pattern"
    ATTR_CB_NAME = "ai.cb.name"
    ATTR_CB_STATE = "ai.cb.state"
    ATTR_CB_FAILURES = "ai.cb.failures"
    ATTR_BH_NAME = "ai.bulkhead.name"
    ATTR_BH_INFLIGHT = "ai.bulkhead.inflight"
    ATTR_RL_NAME = "ai.ratelimit.name"
    ATTR_BUDGET_REMAIN = "ai.latency_budget.remaining_ms"
    ATTR_BUDGET_USED_PCT = "ai.latency_budget.used_pct"
    ATTR_ERROR_TYPE = "exception.type"
    ATTR_BACKEND = "ai.backend"

    # W3C Baggage key used to propagate session_id across service boundaries
    BAGGAGE_SESSION_ID = "ai.session.id"
    BAGGAGE_REQUEST_ID = "ai.request.id"

    def __init__(self) -> None:
        self._enabled = OTEL_ENABLED and _OTEL_PACKAGES_AVAILABLE
        self._tracer_prov: Any = None
        self._meter_prov: Any = None
        self._tracers: dict[str, Any] = {}
        self._meters: dict[str, Any] = {}
        self._propagator: Any = None
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """
        Build and install OTel providers. Called once from bootstrap().
        Safe to call multiple times — subsequent calls are no-ops.
        """
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            if not self._enabled:
                return

            resource = _OtelResource.create({_OTEL_SVC_NAME: OTEL_SERVICE_NAME})

            # ── TracerProvider ────────────────────────────────────────────────
            try:
                from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

                existing = otel_trace.get_tracer_provider()
                # noinspection PyUnreachableCode
                if isinstance(existing, TracerProvider):
                    # shared.py installs an SDK TracerProvider unconditionally at module-level
                    # (via _build_and_install_tracer_provider()) the instant it is first imported.
                    # Since initialize() is only reachable when self._enabled=True, which requires
                    # _OTEL_PACKAGES_AVAILABLE=True, which requires shared.py to have imported
                    # successfully, the SDK provider is always already installed by the time this
                    # isinstance check runs. Adopt the existing provider rather than building a
                    # new one — set_tracer_provider() would be silently ignored by the OTel SDK
                    # but would still print "Overriding of current TracerProvider is not allowed".
                    # More critically, self._tracer_prov must reference the provider that is
                    # actually registered globally, not a freshly constructed one that was never
                    # passed to set_tracer_provider() and will therefore never serve any tracers.
                    self._tracer_prov = existing
                else:
                    # DEAD BRANCH — not reachable in any normal process startup.
                    # The only way to reach here is if isinstance(existing, TracerProvider) is
                    # False, meaning the global is still OTel's default ProxyTracerProvider.
                    # That is only possible if shared.py's module-level block did not run, which
                    # is only possible if shared.py was not imported — but self._enabled=True
                    # requires _OTEL_PACKAGES_AVAILABLE=True which requires shared.py to have
                    # imported successfully. The precondition for entering this method and the
                    # precondition for the ProxyTracerProvider still being active are mutually
                    # exclusive at runtime. Kept solely as a safety net for isolated unit tests
                    # that construct _OtelLayer() directly without going through shared.py.
                    sampler = TraceIdRatioBased(OTEL_SAMPLE_RATE)
                    provider = TracerProvider(resource=resource, sampler=sampler)
                    provider.add_span_processor(
                        BatchSpanProcessor(
                            OTLPSpanExporter(endpoint=OTEL_ENDPOINT, insecure=True)
                        )
                    )
                    otel_trace.set_tracer_provider(provider)
                    self._tracer_prov = provider
            except Exception as exc:
                import sys
                print(
                    f"[observability] OTel TracerProvider init failed: {exc}",
                    file=sys.stderr,
                )
                self._enabled = False
                return

            # ── MeterProvider (OTLP push, independent from Prometheus) ────────
            try:
                existing_meter = otel_metrics.get_meter_provider()
                # noinspection PyUnreachableCode
                if isinstance(existing_meter, MeterProvider):
                    # Same reasoning as the TracerProvider block above. shared.py installs an
                    # SDK MeterProvider at module-level via _build_and_install_meter_provider()
                    # before any call to initialize() is possible. Adopt the existing provider
                    # so self._meter_prov references what is actually registered globally.
                    self._meter_prov = existing_meter
                else:
                    # DEAD BRANCH — same mutual exclusion as the TracerProvider else above.
                    # Reaching here requires isinstance(existing_meter, MeterProvider) to be
                    # False, i.e. the global is still OTel's default ProxyMeterProvider, which
                    # is only possible if shared.py did not fully load. That contradicts the
                    # _OTEL_PACKAGES_AVAILABLE=True precondition required to reach this code.
                    # Kept for the same isolated unit-test scenario described above.
                    reader = PeriodicExportingMetricReader(
                        OTLPMetricExporter(endpoint=OTEL_ENDPOINT, insecure=True),
                        export_interval_millis=OTEL_METRIC_MS,
                    )
                    meter_prov = MeterProvider(resource=resource, metric_readers=[reader])
                    otel_metrics.set_meter_provider(meter_prov)
                    self._meter_prov = meter_prov

            except Exception:  # noqa
                pass  # meter failure is non-fatal; traces still work

            # ── Propagator ────────────────────────────────────────────────────
            #
            # W3C TraceContext is used as the sole propagation format.
            # CompositePropagator is intentionally kept for future extension
            # (e.g. adding B3 via opentelemetry-propagator-b3 if your infra
            # requires it — just append B3Format() to the list below).
            #
            # NOTE: B3Format was previously imported here but never added to
            # the propagator list, so it has been removed.  If you need B3:
            #
            #   pip install opentelemetry-propagator-b3
            #
            # Then append it:
            #
            #   from opentelemetry.propagators.b3 import B3Format
            #   propagators = [TraceContextTextMapPropagator(), B3Format()]
            try:
                from opentelemetry.propagators.composite import CompositePropagator

                self._propagator = CompositePropagator(
                    [
                        TraceContextTextMapPropagator(),
                    ]
                )
                otel_propagate.set_global_textmap(self._propagator)
            except Exception:  # noqa
                self._propagator = TraceContextTextMapPropagator()
                otel_propagate.set_global_textmap(self._propagator)

    # ── tracer / meter accessors ──────────────────────────────────────────────

    def tracer(self, service: str) -> Any:
        """
        Return the OTel tracer for *service*.
        Returns _NoOpTracer when OTEL_ENABLED=false — zero context overhead.
        """
        if not self._enabled:
            return _NOOP_TRACER
        if service not in self._tracers:
            self._tracers[service] = otel_trace.get_tracer(
                f"ai.{service}", schema_url="https://opentelemetry.io/schemas/1.23.0"
            )
        return self._tracers[service]

    def meter(self, service: str) -> Any:
        """Return the OTel meter for *service*, or None when disabled."""
        if not self._enabled:
            return None
        if service not in self._meters:
            self._meters[service] = otel_metrics.get_meter(f"ai.{service}")
        return self._meters[service]

    # ── current span helpers ──────────────────────────────────────────────────

    def current_span(self) -> Any:
        """Return the currently active span, or _NOOP_SPAN if none."""
        if not self._enabled:
            return _NOOP_SPAN
        try:
            span = otel_trace.get_current_span()
            return span if span is not None else _NOOP_SPAN
        except Exception:  # noqa
            return _NOOP_SPAN

    def trace_id(self) -> str:
        """
        Return the current trace ID as a 32-char hex string, or "" if no span.
        Used by configure_logging() to inject trace_id into structlog records
        for Grafana/Loki log-to-trace correlation.
        """
        if not self._enabled:
            return ""
        try:
            ctx = otel_trace.get_current_span().get_span_context()
            if ctx and ctx.is_valid:
                return format(ctx.trace_id, "032x")
        except Exception:  # noqa
            pass
        return ""

    def span_id(self) -> str:
        """Return the current span ID as a 16-char hex string, or ""."""
        if not self._enabled:
            return ""
        try:
            ctx = otel_trace.get_current_span().get_span_context()
            if ctx and ctx.is_valid:
                return format(ctx.span_id, "016x")
        except Exception:  # noqa
            pass
        return ""

    # ── W3C TraceContext + Baggage propagation ────────────────────────────────

    def inject_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """
        Inject the current span's W3C traceparent + tracestate headers into
        *headers*.  Also injects any active W3C Baggage (including session_id).
        No-op when OTEL_ENABLED=false.

        Usage in remote HTTP clients (STT, LLM, TTS remote nodes)::

            headers = _OTEL.inject_headers({})
            response = await httpx_client.post(url, headers=headers, ...)
        """
        if not self._enabled:
            return headers
        try:
            otel_propagate.inject(headers)
        except Exception:  # noqa
            pass
        return headers

    def extract_context(self, headers: dict[str, str]) -> Any:
        """
        Extract W3C TraceContext + Baggage from inbound HTTP request headers.
        Returns an OTel Context object that should be passed as *context=* to
        start_as_current_span() at the remote service ingress.

        Usage in remote service request handlers::

            ctx = _OTEL.extract_context(dict(request.headers))
            with tracer.start_as_current_span("ai.stt.transcribe", context=ctx) as span:
                ...
        """
        if not self._enabled:
            return None
        try:
            return otel_propagate.extract(headers)
        except Exception:  # noqa
            return None

    def set_baggage(self, key: str, value: str) -> None:
        """
        Set a W3C Baggage key in the current context.
        Used by session_context() to propagate session_id across HTTP hops.
        No-op when OTEL_ENABLED=false.
        """
        if not self._enabled:
            return
        try:
            token = otel_baggage.set_baggage(key, value)  # type: ignore[attr-defined]
            # baggage.set_baggage returns a new context; activate it
            otel_ctx.attach(token)  # type: ignore[arg-type]
        except Exception:  # noqa
            pass

    def get_baggage(self, key: str) -> str:
        """Retrieve a W3C Baggage value from the current context."""
        if not self._enabled:
            return ""
        try:
            return otel_baggage.get_baggage(key) or ""  # type: ignore[attr-defined]
        except Exception:  # noqa
            return ""

    # ── span event recording from ObsEvent ───────────────────────────────────

    def record_event(self, event: "ObsEvent") -> None:
        """
        Record *event* as a span event on the currently active span.

        This is called inside emit() as the 4th sink. It annotates whatever
        span is currently active (pipeline, stage, or root) with a timestamped
        event and a curated set of attributes derived from the ObsEvent fields.

        Design: spans are created by context managers, not by emit().
        emit() only annotates an existing span. This keeps emit() cheap
        (no context manipulation) and lets callers compose spans freely.
        """
        if not self._enabled:
            return
        span = self.current_span()
        if not span.is_recording():
            return
        try:
            attrs = self._event_to_attrs(event)
            span.add_event(event.kind, attributes=attrs)
            if not event.ok:
                if event.error_type:
                    span.set_attribute(self.ATTR_ERROR_TYPE, event.error_type)
                # Only set ERROR status on terminal failures, not warnings
                _terminal_failures = {
                    EventKind.PIPELINE_FAILED,
                    EventKind.STT_FAILED,
                    EventKind.LLM_FAILED,
                    EventKind.TTS_FAILED,
                    EventKind.EVAL_FAILED,
                    EventKind.BUDGET_BREACHED,
                }
                if event.kind in _terminal_failures:
                    span.set_status(
                        StatusCode.ERROR, event.error[:200] if event.error else ""
                    )
        except Exception:  # noqa
            pass

    def _event_to_attrs(self, e: "ObsEvent") -> dict[str, Any]:
        """
        Map an ObsEvent to a flat OTel attribute dict.
        Only non-empty / non-zero fields are included.
        Attribute values must be str | bool | int | float — no dicts/lists.
        """
        a: dict[str, Any] = {}

        def _set(key: str, val: Any) -> None:
            if val not in (None, "", 0, 0.0, False, [], {}):
                a[key] = val

        _set(self.ATTR_SESSION_ID, e.session_id)
        _set(self.ATTR_REQUEST_ID, e.request_id)
        _set(self.ATTR_STAGE, e.stage)
        _set(self.ATTR_QOS_TIER, e.qos_tier)
        _set(self.ATTR_EXEC_MODE, e.execution_mode)
        _set(self.ATTR_GRAPH_VERSION, e.pipeline_version)
        _set(self.ATTR_RETRY, e.retry_attempt)
        _set(self.ATTR_ABORT_REASON, e.abort_reason)
        _set(self.ATTR_DEGRADED, e.degraded)
        _set(self.ATTR_AUDIO_PATH, e.audio_path)
        _set(self.ATTR_AUDIO_BYTES, e.audio_bytes)
        _set(self.ATTR_AUDIO_DURATION, e.audio_duration_s)
        _set(self.ATTR_LANGUAGE, e.language)
        _set(self.ATTR_LANG_CONF, e.lang_confidence)
        _set(self.ATTR_AVG_LOGPROB, e.avg_logprob)
        _set(self.ATTR_NO_SPEECH, e.no_speech_prob)
        _set(self.ATTR_TRANSCRIPT_CHARS, e.transcript_chars)
        _set(self.ATTR_REMOTE_FALLBACK, e.remote_fallback)
        _set(self.ATTR_GEN_AI_SYSTEM, "openai")
        _set(self.ATTR_GEN_AI_MODEL, e.model)
        _set(self.ATTR_GEN_AI_MODEL_USED, e.model_used)
        _set(self.ATTR_GEN_AI_STREAMING, e.streaming)
        _set(self.ATTR_PROMPT_TOKENS, e.prompt_tokens)
        _set(self.ATTR_COMPL_TOKENS, e.completion_tokens)
        _set(self.ATTR_TOTAL_TOKENS, e.total_tokens)
        _set(self.ATTR_CACHE_HIT, e.cache_hit)
        _set(self.ATTR_FALLBACK_MODEL, e.fallback_model)
        _set(self.ATTR_HISTORY_TURNS, e.history_turns)
        _set(self.ATTR_RESP_CHARS, e.response_chars)
        _set(self.ATTR_TTS_VOICE, e.voice)
        _set(self.ATTR_TTS_FORMAT, e.tts_format)
        _set(self.ATTR_TTS_INPUT_CHARS, e.input_chars)
        _set(self.ATTR_TTS_CHUNK_COUNT, e.chunk_count)
        _set(self.ATTR_TTS_APOLOGY, e.apology_used)
        _set(self.ATTR_S3_URI, e.s3_uri)
        _set(self.ATTR_EVAL_TURN_IDX, e.eval_turn_idx)
        _set(self.ATTR_EVAL_MODEL, e.eval_model)
        _set(self.ATTR_EVAL_SCORE, e.eval_score)
        _set(self.ATTR_EVAL_TOKENS, e.eval_tokens)
        _set(self.ATTR_EVAL_SKIPPED, e.eval_skipped)
        _set(self.ATTR_EVAL_SKIP_REASON, e.eval_skip_reason)
        _set(self.ATTR_EVAL_BUDGET_USED, e.eval_budget_used)
        _set(self.ATTR_IP_MASKED, e.ip_masked)
        _set(self.ATTR_IP_CHANGES, e.ip_changes)
        _set(self.ATTR_SESSION_TURNS, e.session_turns)
        _set(self.ATTR_SESSION_REASON, e.session_reason)
        _set(self.ATTR_HISTORY_DEPTH, e.history_depth)
        _set(self.ATTR_COMPR_RATIO, e.compression_ratio)
        _set(self.ATTR_ORIG_CHARS, e.original_chars)
        _set(self.ATTR_SANI_CHARS, e.sanitized_chars)
        _set(self.ATTR_INJECTION_PAT, e.injection_pattern)
        _set(self.ATTR_CB_NAME, e.cb_name)
        _set(self.ATTR_CB_STATE, e.cb_state)
        _set(self.ATTR_CB_FAILURES, e.cb_failures)
        _set(self.ATTR_BH_NAME, e.bh_name)
        _set(self.ATTR_BH_INFLIGHT, e.bh_inflight)
        _set(self.ATTR_RL_NAME, e.rl_name)
        _set(self.ATTR_BUDGET_REMAIN, e.budget_remain_ms)
        _set(self.ATTR_BUDGET_USED_PCT, e.budget_used_pct)
        _set(self.ATTR_ERROR_TYPE, e.error_type)
        _set(self.ATTR_BACKEND, e.backend)
        if e.latency_ms:
            a["ai.latency_ms"] = round(e.latency_ms, 2)
        return a


# Module-level OTel singleton — used everywhere in this file
_OTEL: _OtelLayer = _OtelLayer()

# ═════════════════════════════════════════════════════════════════════════════
#  RICH CONSOLE RENDERER
# ═════════════════════════════════════════════════════════════════════════════

_THEME = Theme(
    {
        "ts": "dim white",
        "lvl.info": "bold bright_green",
        "lvl.warn": "bold yellow",
        "lvl.error": "bold bright_red",
        "lvl.debug": "dim cyan",
        "evt": "white",
        "kv.key": "dim white",
        "kv.val": "bright_white",
        "svc.stt": "cyan",
        "svc.llm": "magenta",
        "svc.tts": "blue",
        "svc.session": "yellow",
        "svc.eval": "bright_yellow",
        "svc.pipeline": "white",
        "svc.cb": "bright_red",
        "svc.rl": "red",
        "svc.bh": "bright_magenta",
        "svc.ctrl": "bright_green",
        "svc.memory": "bright_cyan",
        "svc.sanitize": "dim green",
        "svc.redis": "bright_blue",
        "ok": "bright_green",
        "warn": "yellow",
        "err": "bright_red",
    }
)

_CONSOLE = Console(theme=_THEME, highlight=False)

_SVC_STYLE: dict[str, tuple[str, str]] = {
    "stt": ("◈", "svc.stt"),
    "llm": ("◈", "svc.llm"),
    "tts": ("♪", "svc.tts"),
    "session": ("⬡", "svc.session"),
    "eval": ("⊛", "svc.eval"),
    "pipeline": ("⚙", "svc.pipeline"),
    "cb": ("⊗", "svc.cb"),
    "rl": ("⊘", "svc.rl"),
    "bh": ("⊕", "svc.bh"),
    "controller": ("●", "svc.ctrl"),
    "memory": ("⌘", "svc.memory"),
    "sanitize": ("⌖", "svc.sanitize"),
    "redis": ("⬡", "svc.redis"),
    "transcript": ("≡", "svc.session"),
}

_LEVEL_STYLE: dict[str, str] = {
    "info": "lvl.info",
    "warning": "lvl.warn",
    "warn": "lvl.warn",
    "error": "lvl.error",
    "debug": "lvl.debug",
}

_KV_SUPPRESS = {"request_id", "level", "timestamp", "event", "logger"}
_KV_ALIAS: dict[str, str] = {
    "session_id": "sid",
    "latency_ms": "ms",
    "latency_s": "lat",
    "total_tokens": "tok",
    "audio_path": "path",
    "lang_confidence": "conf",
    "circuit_state": "cb",
    "error_type": "etype",
    "model_used": "model",
}


def _kv_string(event_dict: dict) -> str:
    parts: list[str] = []
    for raw_key, val in event_dict.items():
        if raw_key in _KV_SUPPRESS:
            continue
        if (
            val is None
            or val == ""
            or val is False
            or val == 0
            or val == {}
            or val == []
        ):
            continue
        key = _KV_ALIAS.get(raw_key, raw_key)
        if isinstance(val, float):
            val = f"{val:.3f}"
        parts.append(f"[kv.key]{key}=[/kv.key][kv.val]{val}[/kv.val]")
    return "  ".join(parts)


_file_write_lock = threading.Lock()


def _write_json_line(doc: dict) -> None:
    try:
        path = Path(LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(doc, default=str)
        with _file_write_lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:  # noqa
        pass


class _DualSinkRenderer:
    """
    Forks every structlog record into Rich console and a JSON file.
    MongoDB writes happen via the ObsEvent path, not here.
    """

    def __call__(self, _logger: object, _method: str, event_dict: dict) -> str:
        json_dict = copy.deepcopy(event_dict)

        ts = event_dict.pop("timestamp", "")
        level = event_dict.pop("level", "info").lower()
        event = event_dict.pop("event", "")
        service = event_dict.get("service", "")

        time_str = ts[11:19] if len(ts) >= 19 else ts
        icon, style = _SVC_STYLE.get(service, ("·", "svc.pipeline"))
        lvl_style = _LEVEL_STYLE.get(level, "lvl.info")
        level_label = ("WARN" if level == "warning" else level.upper())[:5]
        kv = _kv_string(event_dict)

        line = (
            f"[ts]\\[{time_str}][/ts] "
            f"[{lvl_style}][{level_label:<4}][/{lvl_style}]  "
            f"[{style}]{icon}[/{style}]  "
            f"[evt]{event:<42}[/evt]  "
            f"{kv}"
        )
        _CONSOLE.print(line)
        _write_json_line(json_dict)

        raise structlog.DropEvent()


# ═════════════════════════════════════════════════════════════════════════════
#  STRUCTLOG CONFIGURATION  (replaces log_config.configure_logging)
# ═════════════════════════════════════════════════════════════════════════════

import contextvars as _cv

# ── ContextVar ownership ──────────────────────────────────────────────────────
# _request_id_var is imported from shared.py so both layers read and write the
# same ContextVar instance.  This eliminates the correlation gap where calling
# shared.new_request_id() wouldn't propagate into observability log/Mongo records.
#
# _session_id_var is owned exclusively here because session context is an
# observability concern — shared.py has no session concept.
#
# Fallback: if shared is unavailable (standalone testing), we create a local var.
try:
    from app.common.shared import _request_id_var  # noqa
except Exception:  # noqa
    _request_id_var: _cv.ContextVar[str] = _cv.ContextVar("request_id", default="")

_session_id_var: _cv.ContextVar[str] = _cv.ContextVar("obs_session_id", default="")


def set_request_context(*, session_id: str = "", request_id: str = "") -> None:
    """Bind session_id and request_id into the current async context."""
    if session_id:
        _session_id_var.set(session_id)
    if request_id:
        _request_id_var.set(request_id)


def get_session_id() -> str:
    return _session_id_var.get()


def get_request_id() -> str:
    return _request_id_var.get()


def _inject_context(_logger: object, _method: str, event_dict: dict) -> dict:
    sid = _session_id_var.get()
    rid = _request_id_var.get()
    if sid:
        event_dict.setdefault("session_id", sid)
    if rid:
        event_dict.setdefault("request_id", rid)
    # Inject OTel trace/span IDs for log-to-trace correlation in Grafana/Loki.
    # These fields allow Grafana's "Derived fields" to hyperlink a log line
    # directly to its parent trace in Tempo.
    trace_id = _OTEL.trace_id()
    span_id = _OTEL.span_id()
    if trace_id:
        event_dict["trace_id"] = trace_id
    if span_id:
        event_dict["span_id"] = span_id
    return event_dict


def configure_logging() -> None:
    """
    Drop-in replacement for log_config.configure_logging().

    Call once at process startup. Configures structlog with context injection,
    dual-sink rendering (standard mode) or JSON-only (verbose mode).
    """
    base_processors = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    final = (
        _DualSinkRenderer()
        if LOG_MODE == "standard"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*base_processors, final],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str):
    return structlog.get_logger(name)


# ═════════════════════════════════════════════════════════════════════════════
#  CORE EMIT FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

_log = structlog.get_logger("observability")


def emit(event: ObsEvent) -> None:
    """
    Emit one observability event to all four sinks simultaneously:

      1. structlog → Rich console (standard mode) + JSON file (always)
      2. Prometheus → kind-specific counter/histogram updates
      3. MongoDB   → full structured document via background queue
      4. OpenTelemetry → span event annotated on the currently active span

    This is the single callsite every component uses. Nothing writes to
    structlog, Prometheus, MongoDB, or OTel directly.

    OTel sink behaviour
    ───────────────────
    Spans are created by context managers (pipeline_span, stt_span, etc.).
    emit() only annotates whatever span is already active — it never
    creates spans or performs context operations itself. This means:

      - emit() is always O(1) overhead regardless of OTel state
      - context managers can be composed freely (pipeline_span wrapping
        stt_span wrapping the actual transcribe call)
      - OTEL_ENABLED=false → _OTEL.record_event() is a single bool check
        that returns immediately, zero context overhead
    """
    m = get_metrics()

    # ── structlog ─────────────────────────────────────────────────────────────
    log_fields: dict[str, Any] = {
        "service": event.service,
        "session_id": event.session_id or get_session_id(),
        "request_id": event.request_id or get_request_id(),
    }
    if event.latency_ms:
        log_fields["latency_ms"] = round(event.latency_ms, 2)
    if event.model_used:
        log_fields["model"] = event.model_used
    if event.error:
        log_fields["error"] = event.error[:200]
    if event.transcript:
        log_fields["transcript_chars"] = event.transcript_chars
    if event.total_tokens:
        log_fields["tokens"] = event.total_tokens
    if event.cb_state:
        log_fields["cb_state"] = event.cb_state
    if event.stage:
        log_fields["stage"] = event.stage
    if event.abort_reason:
        log_fields["abort"] = event.abort_reason

    log_level = (
        "error" if not event.ok else "warning" if _is_warning(event.kind) else "info"
    )
    getattr(_log, log_level)(event.kind, **log_fields)

    # ── Prometheus ────────────────────────────────────────────────────────────
    if PROMETHEUS_ENABLED:
        _update_prometheus(event, m)

    # ── MongoDB ───────────────────────────────────────────────────────────────
    if MONGO_ENABLED and _MONGO._started:  # noqa
        doc = {
            k: v
            for k, v in asdict(event).items()
            if v not in (None, "", 0, 0.0, False, [], {})
        }
        doc["ts_iso"] = _iso(event.ts)
        # Attach current trace/span IDs to MongoDB document so stored events
        # can be cross-referenced with traces even outside of Grafana.
        tid = _OTEL.trace_id()
        sid_otel = _OTEL.span_id()
        if tid:
            doc["trace_id"] = tid
        if sid_otel:
            doc["span_id"] = sid_otel
        _MONGO.push(doc)

    # ── OpenTelemetry (4th sink) ──────────────────────────────────────────────
    _OTEL.record_event(event)


def _is_warning(kind: str) -> bool:
    _warn_kinds = {
        EventKind.CB_OPENED,
        EventKind.CB_REJECTED,
        EventKind.RL_EXHAUSTED,
        EventKind.BH_SATURATED,
        EventKind.BUDGET_BREACHED,
        EventKind.BUDGET_NEAR,
        EventKind.SESSION_IP_CHANGE,
        EventKind.SESSION_IP_LIMIT,
        EventKind.SESSION_SUSPENDED,
        EventKind.PIPELINE_DEGRADED,
        EventKind.PIPELINE_LOAD_SHED,
        EventKind.PIPELINE_ABORT,
        EventKind.STT_LOW_CONFIDENCE,
        EventKind.STT_EMPTY_AUDIO,
        EventKind.LLM_MODEL_FALLBACK,
        EventKind.TTS_APOLOGY_FALLBACK,
        EventKind.EVAL_BUDGET_EXHAUSTED,
        EventKind.REDIS_DISCONNECTED,
        EventKind.REDIS_DEGRADED,
        EventKind.TRANSCRIPT_QUEUE_DROP,
        EventKind.SANITIZE_INJECTION,
        EventKind.MEMORY_COMPRESSION,
        EventKind.MEMORY_OVERFLOW,
    }
    return kind in _warn_kinds


def _cb_state_int(state: str) -> int:
    return {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}.get(state.upper(), 0)


def _update_prometheus(e: ObsEvent, m: _Metrics) -> None:  # noqa: C901
    k = e.kind

    # Pipeline
    if k == EventKind.PIPELINE_DONE:
        m.pipeline_total.labels(
            status="ok",
            tier=e.qos_tier,
            version=e.pipeline_version,
            mode=e.execution_mode,
        ).inc()
        m.pipeline_latency.labels(tier=e.qos_tier, mode=e.execution_mode).observe(
            e.wall_s
        )
        for stage, lat in e.stage_latencies.items():
            m.pipeline_stage_latency.labels(stage=stage).observe(lat)
    elif k == EventKind.PIPELINE_FAILED:
        m.pipeline_total.labels(
            status="error",
            tier=e.qos_tier,
            version=e.pipeline_version,
            mode=e.execution_mode,
        ).inc()
        m.pipeline_stage_errors.labels(stage=e.stage, error_type=e.error_type).inc()
    elif k == EventKind.PIPELINE_DEGRADED:
        m.pipeline_degraded.labels(stage=e.stage).inc()
    elif k == EventKind.PIPELINE_CANCELLED:
        m.pipeline_cancelled.labels(stage=e.stage).inc()
    elif k == EventKind.PIPELINE_LOAD_SHED:
        m.pipeline_load_shed.labels(tier=e.qos_tier).inc()
    elif k == EventKind.PIPELINE_RETRY:
        m.pipeline_stage_retries.labels(stage=e.stage).inc()
    elif k == EventKind.PIPELINE_ABORT:
        m.pipeline_aborted.labels(reason=e.abort_reason[:64]).inc()
    elif k == EventKind.PIPELINE_START:
        m.pipeline_active.labels(tier=e.qos_tier or "standard").inc()
    if k in (
        EventKind.PIPELINE_DONE,
        EventKind.PIPELINE_FAILED,
        EventKind.PIPELINE_CANCELLED,
        EventKind.PIPELINE_DEGRADED,
        EventKind.PIPELINE_ABORT,
    ):
        m.pipeline_active.labels(tier=e.qos_tier or "standard").dec()

    # STT
    if k == EventKind.STT_OK:
        m.stt_requests.labels(status="ok").inc()
        m.stt_latency.observe(e.latency_ms / 1000)
        if e.audio_duration_s:
            m.stt_audio_duration.observe(e.audio_duration_s)
        if e.audio_bytes:
            m.stt_audio_bytes.observe(e.audio_bytes)
        if e.lang_confidence and e.language:
            m.stt_lang_confidence.labels(language=e.language).observe(e.lang_confidence)
        if e.transcript_chars:
            m.stt_transcript_chars.observe(e.transcript_chars)
    elif k == EventKind.STT_FAILED:
        m.stt_requests.labels(status="error").inc()
    elif k == EventKind.STT_EMPTY_AUDIO:
        m.stt_empty_audio.inc()
    elif k == EventKind.STT_LOW_CONFIDENCE:
        m.stt_low_confidence.inc()
    elif k == EventKind.STT_REMOTE_FALLBACK:
        m.stt_remote_fallback.inc()
    elif k == EventKind.STT_PATH_REJECTED:
        m.stt_path_rejected.inc()

    # LLM
    if k == EventKind.LLM_OK:
        mode = "stream" if e.streaming else "batch"
        m.llm_requests.labels(status="ok", model=e.model_used, mode=mode).inc()
        m.llm_latency.labels(model=e.model_used, mode=mode).observe(e.latency_ms / 1000)
        if e.prompt_tokens:
            m.llm_tokens_prompt.labels(model=e.model_used).observe(e.prompt_tokens)
            m.llm_tokens_total.labels(model=e.model_used, kind="prompt").inc(
                e.prompt_tokens
            )
        if e.completion_tokens:
            m.llm_tokens_completion.labels(model=e.model_used).observe(
                e.completion_tokens
            )
            m.llm_tokens_total.labels(model=e.model_used, kind="completion").inc(
                e.completion_tokens
            )
        if e.history_turns:
            m.llm_history_turns.observe(e.history_turns)
    elif k == EventKind.LLM_FAILED:
        mode = "stream" if e.streaming else "batch"
        m.llm_requests.labels(
            status="error", model=e.model_used or e.model, mode=mode
        ).inc()
    elif k == EventKind.LLM_CACHE_HIT:
        m.llm_cache_hits.labels(backend=e.backend or "redis").inc()
        m.llm_requests.labels(status="cache", model=e.model, mode="batch").inc()
    elif k == EventKind.LLM_CACHE_MISS:
        m.llm_cache_misses.inc()
    elif k == EventKind.LLM_CACHE_STAMPEDE:
        m.llm_cache_stampede.inc()
    elif k == EventKind.LLM_MODEL_FALLBACK:
        m.llm_model_fallback.labels(primary=e.model, fallback=e.model_used).inc()
    elif k == EventKind.LLM_RESPONSE_TRUNCATED:
        m.llm_response_truncated.inc()

    # TTS
    if k == EventKind.TTS_OK:
        m.tts_requests.labels(status="ok", voice=e.voice).inc()
        m.tts_latency.labels(voice=e.voice).observe(e.latency_ms / 1000)
        if e.input_chars:
            m.tts_input_chars.observe(e.input_chars)
    elif k == EventKind.TTS_FAILED:
        m.tts_requests.labels(status="error", voice=e.voice or "unknown").inc()
    elif k == EventKind.TTS_APOLOGY_FALLBACK:
        m.tts_apology_fallback.inc()
    elif k == EventKind.TTS_CHUNK_ERROR:
        m.tts_chunk_errors.inc()
    elif k == EventKind.TTS_S3_UPLOAD_OK:
        m.tts_s3_uploads.labels(status="ok").inc()
    elif k == EventKind.TTS_S3_UPLOAD_FAILED:
        m.tts_s3_uploads.labels(status="error").inc()
    elif k == EventKind.TTS_FILE_CLEANUP:
        m.tts_file_cleanups.inc()

    # Eval
    if k == EventKind.EVAL_OK:
        m.eval_requests.labels(status="ok").inc()
        m.eval_latency.observe(e.latency_ms / 1000)
        m.eval_scores.observe(e.eval_score)
        if e.eval_tokens:
            m.eval_tokens.inc(e.eval_tokens)
    elif k == EventKind.EVAL_FAILED:
        m.eval_requests.labels(status="error").inc()
    elif k in (EventKind.EVAL_SKIPPED_TOO_SHORT, EventKind.EVAL_SKIPPED_SAMPLING):
        reason = "too_short" if k == EventKind.EVAL_SKIPPED_TOO_SHORT else "sampling"
        m.eval_skipped.labels(reason=reason).inc()
    elif k == EventKind.EVAL_BUDGET_EXHAUSTED:
        m.eval_budget_exhausted.inc()
    elif k == EventKind.EVAL_MODEL_FALLBACK:
        m.eval_model_fallback.inc()

    # Session
    if k == EventKind.SESSION_REGISTERED:
        m.session_created.inc()
        m.session_active.inc()
    elif k == EventKind.SESSION_ENDED:
        m.session_ended.labels(reason=e.session_reason).inc()
        m.session_active.dec()
        if e.session_turns:
            m.session_turns.observe(e.session_turns)
        if e.session_duration_s:
            m.session_duration.observe(e.session_duration_s)
    elif k == EventKind.SESSION_REJECTED:
        m.session_rejected.labels(reason=e.session_reason).inc()
    elif k == EventKind.SESSION_IP_CHANGE:
        m.session_ip_changes.inc()
    elif k == EventKind.SESSION_IP_LIMIT:
        m.session_ip_limit.inc()
    elif k == EventKind.SESSION_LRU_HIT:
        m.session_lru_hits.inc()

    # Memory
    if k == EventKind.MEMORY_COMPRESSION:
        m.memory_compressions.inc()
        if e.compression_ratio:
            m.memory_compression_ratio.observe(e.compression_ratio)
    elif k == EventKind.MEMORY_LRU_FALLBACK:
        m.memory_lru_fallbacks.inc()

    # Sanitize
    if k == EventKind.SANITIZE_OK:
        m.sanitize_total.labels(status="ok").inc()
        if e.original_chars and e.sanitized_chars:
            reduction = 1.0 - (e.sanitized_chars / max(1, e.original_chars))
            m.sanitize_reduction.observe(reduction)
    elif k == EventKind.SANITIZE_INJECTION:
        m.sanitize_injection.inc()
        m.sanitize_total.labels(status="injection").inc()
    elif k == EventKind.SANITIZE_TRUNCATED:
        m.sanitize_truncated.inc()
    elif k == EventKind.SANITIZE_EMPTY:
        m.sanitize_total.labels(status="empty").inc()

    # Circuit breaker
    if k == EventKind.CB_OPENED:
        m.cb_opened.labels(service=e.service).inc()
        m.cb_state.labels(service=e.service).set(2)
    elif k == EventKind.CB_HALF_OPEN:
        m.cb_half_open.labels(service=e.service).inc()
        m.cb_state.labels(service=e.service).set(1)
    elif k == EventKind.CB_CLOSED:
        m.cb_closed.labels(service=e.service).inc()
        m.cb_state.labels(service=e.service).set(0)
    elif k == EventKind.CB_REJECTED:
        m.cb_rejected.labels(service=e.service).inc()

    # Rate limiter
    if k == EventKind.RL_ACQUIRED:
        m.rl_acquired.labels(service=e.rl_name or e.service).inc()
    elif k == EventKind.RL_EXHAUSTED:
        m.rl_exhausted.labels(service=e.rl_name or e.service).inc()
    elif k == EventKind.RL_WAITING:
        m.rl_wait_seconds.labels(service=e.rl_name or e.service).observe(
            e.latency_ms / 1000
        )

    # Bulkhead
    if k == EventKind.BH_ACQUIRED:
        m.bh_inflight.labels(name=e.bh_name).inc()
    elif k == EventKind.BH_RELEASED:
        m.bh_inflight.labels(name=e.bh_name).dec()
    elif k == EventKind.BH_SATURATED:
        m.bh_saturated.labels(name=e.bh_name).inc()

    # Budget
    if k == EventKind.BUDGET_BREACHED:
        m.budget_breached.labels(stage=e.stage).inc()
    elif k in (EventKind.BUDGET_OK, EventKind.BUDGET_NEAR):
        if e.budget_used_pct and e.stage:
            m.budget_remain_pct.labels(stage=e.stage).observe(100 - e.budget_used_pct)

    # Controller
    if k == EventKind.CTRL_PTT_RELEASE:
        m.ctrl_ptt_presses.inc()
        if e.recording_s:
            m.ctrl_recording_duration.observe(e.recording_s)
    elif k == EventKind.CTRL_INTERRUPT:
        m.ctrl_interrupts.inc()
    elif k == EventKind.CTRL_EMPTY_RECORDING:
        m.ctrl_empty_recordings.inc()
    elif k == EventKind.CTRL_PIPELINE_ERROR:
        m.ctrl_pipeline_errors.inc()

    # Redis
    if k == EventKind.REDIS_DISCONNECTED:
        m.redis_disconnects.inc()
        m.redis_degraded_mode.set(1)
    elif k == EventKind.REDIS_RECOVERED:
        m.redis_recovered.inc()
        m.redis_degraded_mode.set(0)

    # Transcript
    if k == EventKind.TRANSCRIPT_TURN:
        m.transcript_turns.inc()
    elif k == EventKind.TRANSCRIPT_QUEUE_DROP:
        m.transcript_drops.inc()


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT EMITTER HELPERS
# ═════════════════════════════════════════════════════════════════════════════
#  These are the callsites each module uses. They construct typed ObsEvent
#  objects and call emit(). Callers never build ObsEvents directly.


class PipelineEmitter:
    """Emitters for voice_graph.py pipeline lifecycle."""

    @staticmethod
    def start(
        *,
        session_id: str,
        request_id: str,
        audio_path: str,
        qos_tier: str,
        mode: str,
        version: str,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_START,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                audio_path=audio_path,
                qos_tier=qos_tier,
                execution_mode=mode,
                pipeline_version=version,
            )
        )

    @staticmethod
    def done(
        *,
        session_id: str,
        request_id: str,
        wall_s: float,
        qos_tier: str,
        mode: str,
        version: str,
        stage_latencies: dict,
        degraded: bool = False,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_DONE,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                wall_s=wall_s,
                latency_ms=wall_s * 1000,
                qos_tier=qos_tier,
                execution_mode=mode,
                pipeline_version=version,
                stage_latencies=stage_latencies,
                degraded=degraded,
            )
        )

    @staticmethod
    def failed(
        *,
        session_id: str,
        request_id: str,
        stage: str,
        error: str,
        error_type: str,
        qos_tier: str,
        mode: str,
        version: str,
        abort_reason: str = "",
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_FAILED,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                stage=stage,
                error=error,
                error_type=error_type,
                qos_tier=qos_tier,
                execution_mode=mode,
                pipeline_version=version,
                abort_reason=abort_reason,
                ok=False,
            )
        )

    @staticmethod
    def degraded(*, session_id: str, request_id: str, stage: str, version: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_DEGRADED,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                stage=stage,
                pipeline_version=version,
                ok=True,
            )
        )

    @staticmethod
    def cancelled(
        *, session_id: str, request_id: str, stage: str, version: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_CANCELLED,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                stage=stage,
                pipeline_version=version,
            )
        )

    @staticmethod
    def load_shed(
        *, session_id: str, qos_tier: str, inflight: int, capacity: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_LOAD_SHED,
                service="pipeline",
                session_id=session_id,
                qos_tier=qos_tier,
                bh_inflight=inflight,
                bh_capacity=capacity,
                ok=False,
            )
        )

    @staticmethod
    def retry(
        *, session_id: str, request_id: str, stage: str, attempt: int, error: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_RETRY,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                stage=stage,
                retry_attempt=attempt,
                error=error,
            )
        )

    @staticmethod
    def abort(
        *,
        session_id: str,
        request_id: str,
        stage: str,
        abort_reason: str,
        error: str = "",
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.PIPELINE_ABORT,
                service="pipeline",
                session_id=session_id,
                request_id=request_id,
                stage=stage,
                abort_reason=abort_reason,
                error=error,
                ok=False,
            )
        )


class STTEmitter:
    """Emitters for STT_service.py."""

    @staticmethod
    def start(
        *, session_id: str, request_id: str, audio_path: str, audio_bytes: int = 0
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_START,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                audio_path=audio_path,
                audio_bytes=audio_bytes,
            )
        )

    @staticmethod
    def ok(
        *,
        session_id: str,
        request_id: str,
        latency_ms: float,
        transcript: str,
        language: str,
        lang_confidence: float,
        avg_logprob: float,
        no_speech_prob: float,
        audio_duration_s: float = 0.0,
        audio_bytes: int = 0,
        segment_count: int = 0,
        truncated: bool = False,
        remote_fallback: bool = False,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_OK,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                latency_ms=latency_ms,
                transcript=transcript,
                transcript_chars=len(transcript),
                language=language,
                lang_confidence=lang_confidence,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                audio_duration_s=audio_duration_s,
                audio_bytes=audio_bytes,
                segment_count=segment_count,
                truncated=truncated,
                remote_fallback=remote_fallback,
            )
        )

    @staticmethod
    def failed(
        *, session_id: str, request_id: str, error: str, error_type: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_FAILED,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                error=error,
                error_type=error_type,
                ok=False,
            )
        )

    @staticmethod
    def empty_audio(*, session_id: str, request_id: str, audio_path: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_EMPTY_AUDIO,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                audio_path=audio_path,
            )
        )

    @staticmethod
    def low_confidence(
        *,
        session_id: str,
        request_id: str,
        lang_confidence: float,
        language: str,
        threshold: float,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_LOW_CONFIDENCE,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                lang_confidence=lang_confidence,
                language=language,
                extra={"threshold": threshold},
            )
        )

    @staticmethod
    def audio_too_short(
        *, session_id: str, request_id: str, duration_s: float, minimum_s: float
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_AUDIO_TOO_SHORT,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                audio_duration_s=duration_s,
                extra={"minimum_s": minimum_s},
            )
        )

    @staticmethod
    def path_rejected(
        *, session_id: str, request_id: str, audio_path: str, reason: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_PATH_REJECTED,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                audio_path=audio_path,
                error=reason,
                ok=False,
            )
        )

    @staticmethod
    def remote_fallback(*, session_id: str, request_id: str, endpoint: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_REMOTE_FALLBACK,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                host=endpoint,
                remote_fallback=True,
            )
        )

    @staticmethod
    def transcript_truncated(
        *,
        session_id: str,
        request_id: str,
        original_chars: int,
        capped_chars: int,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.STT_TRANSCRIPT_TRUNCATED,
                service="stt",
                session_id=session_id,
                request_id=request_id,
                original_chars=original_chars,
                sanitized_chars=capped_chars,
                truncated=True,
            )
        )


class LLMEmitter:
    """Emitters for LLM_service.py."""

    @staticmethod
    def start(
        *,
        session_id: str,
        request_id: str,
        model: str,
        streaming: bool,
        history_turns: int,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_START,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                model=model,
                streaming=streaming,
                history_turns=history_turns,
            )
        )

    @staticmethod
    def ok(
        *,
        session_id: str,
        request_id: str,
        latency_ms: float,
        model_used: str,
        prompt_tokens: int,
        completion_tokens: int,
        streaming: bool,
        history_turns: int,
        response_chars: int,
        cache_hit: bool = False,
        fallback_model: bool = False,
        response_truncated: bool = False,
        temperature: float = 0.0,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_OK,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                latency_ms=latency_ms,
                model_used=model_used,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                streaming=streaming,
                history_turns=history_turns,
                response_chars=response_chars,
                cache_hit=cache_hit,
                fallback_model=fallback_model,
                temperature=temperature,
                response_truncated=response_truncated,
            )
        )

    @staticmethod
    def failed(
        *,
        session_id: str,
        request_id: str,
        error: str,
        error_type: str,
        model: str,
        streaming: bool,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_FAILED,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                error=error,
                error_type=error_type,
                model=model,
                model_used=model,
                streaming=streaming,
                ok=False,
            )
        )

    @staticmethod
    def cache_hit(
        *, session_id: str, request_id: str, model: str, backend: str = "redis"
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_CACHE_HIT,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                model=model,
                backend=backend,
                cache_hit=True,
            )
        )

    @staticmethod
    def cache_miss(*, session_id: str, request_id: str, model: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_CACHE_MISS,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                model=model,
            )
        )

    @staticmethod
    def cache_stampede(*, session_id: str, request_id: str, model: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_CACHE_STAMPEDE,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                model=model,
            )
        )

    @staticmethod
    def model_fallback(
        *, session_id: str, request_id: str, primary: str, fallback: str, reason: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_MODEL_FALLBACK,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                model=primary,
                model_used=fallback,
                fallback_model=True,
                error=reason,
            )
        )

    @staticmethod
    def response_truncated(
        *, session_id: str, request_id: str, original: int, capped: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.LLM_RESPONSE_TRUNCATED,
                service="llm",
                session_id=session_id,
                request_id=request_id,
                original_chars=original,
                sanitized_chars=capped,
                response_truncated=True,
            )
        )


class TTSEmitter:
    """Emitters for TTS_service.py."""

    @staticmethod
    def start(
        *,
        session_id: str,
        request_id: str,
        voice: str,
        input_chars: int,
        tts_format: str = "",
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_START,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                voice=voice,
                input_chars=input_chars,
                tts_format=tts_format,
            )
        )

    @staticmethod
    def ok(
            *,
            session_id: str,
            request_id: str,
            latency_ms: float,
            voice: str,
            input_chars: int,
            audio_output: str,
            audio_duration_s: float = 0.0,
            audio_size_bytes: int = 0,
            s3_uri: str = "",
            chunk_count: int = 0,
            tts_format: str = "",
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_OK,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                latency_ms=latency_ms,
                voice=voice,
                input_chars=input_chars,
                audio_output=audio_output,
                audio_duration_s=audio_duration_s,
                audio_size_bytes=audio_size_bytes,
                s3_uri=s3_uri,
                chunk_count=chunk_count,
                tts_format=tts_format,
            )
        )

    @staticmethod
    def failed(
        *, session_id: str, request_id: str, error: str, error_type: str, voice: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_FAILED,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                error=error,
                error_type=error_type,
                voice=voice,
                ok=False,
            )
        )

    @staticmethod
    def apology_fallback(
        *, session_id: str, request_id: str, apology_path: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_APOLOGY_FALLBACK,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                audio_output=apology_path,
                apology_used=True,
            )
        )

    @staticmethod
    def chunk_error(
        *, session_id: str, request_id: str, chunk_idx: int, error: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_CHUNK_ERROR,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                error=error,
                extra={"chunk_idx": chunk_idx},
                ok=False,
            )
        )

    @staticmethod
    def s3_upload_ok(
        *, session_id: str, request_id: str, s3_uri: str, bytes_uploaded: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_S3_UPLOAD_OK,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                s3_uri=s3_uri,
                s3_ok=True,
                audio_bytes=bytes_uploaded,
            )
        )

    @staticmethod
    def s3_upload_failed(*, session_id: str, request_id: str, error: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_S3_UPLOAD_FAILED,
                service="tts",
                session_id=session_id,
                request_id=request_id,
                error=error,
                s3_ok=False,
                ok=False,
            )
        )

    @staticmethod
    def file_cleanup(*, files_removed: int, bytes_freed: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TTS_FILE_CLEANUP,
                service="tts",
                extra={"files_removed": files_removed, "bytes_freed": bytes_freed},
            )
        )


class EvalEmitter:
    """Emitters for evaluation_engine.py."""

    @staticmethod
    def start(*, session_id: str, request_id: str, turn_idx: int, model: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_START,
                service="eval",
                session_id=session_id,
                request_id=request_id,
                eval_turn_idx=turn_idx,
                eval_model=model,
            )
        )

    @staticmethod
    def ok(
        *,
        session_id: str,
        request_id: str,
        turn_idx: int,
        latency_ms: float,
        score: float,
        model_used: str,
        tokens: int,
        budget_used: int,
        budget_cap: int,
        rubric_keys: list | None = None,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_OK,
                service="eval",
                session_id=session_id,
                request_id=request_id,
                eval_turn_idx=turn_idx,
                latency_ms=latency_ms,
                eval_score=score,
                eval_model=model_used,
                eval_tokens=tokens,
                eval_budget_used=budget_used,
                eval_budget_cap=budget_cap,
                rubric_keys=rubric_keys or [],
            )
        )

    @staticmethod
    def failed(
        *, session_id: str, request_id: str, turn_idx: int, error: str, error_type: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_FAILED,
                service="eval",
                session_id=session_id,
                request_id=request_id,
                eval_turn_idx=turn_idx,
                error=error,
                error_type=error_type,
                ok=False,
            )
        )

    @staticmethod
    def skipped_too_short(
        *, session_id: str, turn_idx: int, answer_chars: int, minimum: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_SKIPPED_TOO_SHORT,
                service="eval",
                session_id=session_id,
                eval_turn_idx=turn_idx,
                eval_skipped=True,
                eval_skip_reason="too_short",
                extra={"answer_chars": answer_chars, "minimum": minimum},
            )
        )

    @staticmethod
    def skipped_sampling(*, session_id: str, turn_idx: int, sample_rate: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_SKIPPED_SAMPLING,
                service="eval",
                session_id=session_id,
                eval_turn_idx=turn_idx,
                eval_skipped=True,
                eval_skip_reason="adaptive_sampling",
                eval_sampled=True,
                extra={"sample_rate": sample_rate},
            )
        )

    @staticmethod
    def budget_exhausted(*, session_id: str, budget_used: int, budget_cap: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_BUDGET_EXHAUSTED,
                service="eval",
                session_id=session_id,
                eval_budget_used=budget_used,
                eval_budget_cap=budget_cap,
            )
        )

    @staticmethod
    def dedup_hit(*, session_id: str, turn_idx: int, content_hash: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_DEDUP_HIT,
                service="eval",
                session_id=session_id,
                eval_turn_idx=turn_idx,
                eval_dedup=True,
                extra={"content_hash": content_hash[:16]},
            )
        )

    @staticmethod
    def model_fallback(
        *, session_id: str, primary: str, fallback: str, reason: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.EVAL_MODEL_FALLBACK,
                service="eval",
                session_id=session_id,
                model=primary,
                model_used=fallback,
                error=reason,
                fallback_model=True,
            )
        )


class SessionEmitter:
    """Emitters for session_store.py."""

    @staticmethod
    def registered(*, session_id: str, ip_masked: str, ttl_s: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_REGISTERED,
                service="session",
                session_id=session_id,
                ip_masked=ip_masked,
                session_ttl_s=ttl_s,
            )
        )

    @staticmethod
    def ended(*, session_id: str, reason: str, turns: int, duration_s: float) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_ENDED,
                service="session",
                session_id=session_id,
                session_reason=reason,
                session_turns=turns,
                session_duration_s=duration_s,
            )
        )

    @staticmethod
    def rejected(*, ip_masked: str, reason: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_REJECTED,
                service="session",
                ip_masked=ip_masked,
                session_reason=reason,
                ok=False,
            )
        )

    @staticmethod
    def not_found(*, session_id: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_NOT_FOUND,
                service="session",
                session_id=session_id,
                ok=False,
            )
        )

    @staticmethod
    def ip_change(
        *, session_id: str, ip_masked: str, change_count: int, change_max: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_IP_CHANGE,
                service="session",
                session_id=session_id,
                ip_masked=ip_masked,
                ip_changes=change_count,
                ip_changes_max=change_max,
            )
        )

    @staticmethod
    def ip_limit_exceeded(
        *, session_id: str, ip_masked: str, changes: int, limit: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_IP_LIMIT,
                service="session",
                session_id=session_id,
                ip_masked=ip_masked,
                ip_changes=changes,
                ip_changes_max=limit,
                ok=False,
            )
        )

    @staticmethod
    def suspended(*, session_id: str, reason: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_SUSPENDED,
                service="session",
                session_id=session_id,
                session_reason=reason,
                ok=False,
            )
        )

    @staticmethod
    def lru_hit(*, session_id: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_LRU_HIT,
                service="session",
                session_id=session_id,
                backend="lru",
            )
        )

    @staticmethod
    def degraded(*, reason: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SESSION_DEGRADED,
                service="session",
                error=reason,
                backend="lru",
            )
        )


class MemoryEmitter:
    """Emitters for conversation_memory.py."""

    @staticmethod
    def resolve(*, session_id: str, request_id: str, history_depth: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.MEMORY_RESOLVE,
                service="memory",
                session_id=session_id,
                request_id=request_id,
                history_depth=history_depth,
            )
        )

    @staticmethod
    def commit(*, session_id: str, request_id: str, history_depth: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.MEMORY_COMMIT,
                service="memory",
                session_id=session_id,
                request_id=request_id,
                history_depth=history_depth,
            )
        )

    @staticmethod
    def compression(
        *,
        session_id: str,
        turns_before: int,
        turns_after: int,
        summary_chars: int,
    ) -> None:
        ratio = turns_after / max(1, turns_before)
        emit(
            ObsEvent(
                kind=EventKind.MEMORY_COMPRESSION,
                service="memory",
                session_id=session_id,
                history_depth=turns_after,
                turns_pruned=turns_before - turns_after,
                compression_ratio=ratio,
                extra={"summary_chars": summary_chars, "turns_before": turns_before},
            )
        )

    @staticmethod
    def overflow(*, session_id: str, depth: int, max_depth: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.MEMORY_OVERFLOW,
                service="memory",
                session_id=session_id,
                history_depth=depth,
                extra={"max_depth": max_depth},
            )
        )

    @staticmethod
    def lru_fallback(*, session_id: str, reason: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.MEMORY_LRU_FALLBACK,
                service="memory",
                session_id=session_id,
                error=reason,
                backend="lru",
            )
        )


class SanitizeEmitter:
    """Emitters for sanitize.py."""

    @staticmethod
    def ok(
        *,
        session_id: str,
        request_id: str,
        original_chars: int,
        sanitized_chars: int,
        truncated: bool,
        warnings: list[str],
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SANITIZE_OK,
                service="sanitize",
                session_id=session_id,
                request_id=request_id,
                original_chars=original_chars,
                sanitized_chars=sanitized_chars,
                truncated=truncated,
                warnings=warnings,
            )
        )

    @staticmethod
    def injection_detected(
        *,
        session_id: str,
        request_id: str,
        pattern: str,
        original_chars: int,
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SANITIZE_INJECTION,
                service="sanitize",
                session_id=session_id,
                request_id=request_id,
                injection_pattern=pattern[:128],
                original_chars=original_chars,
                ok=False,
            )
        )

    @staticmethod
    def truncated(
        *, session_id: str, request_id: str, original: int, capped: int
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SANITIZE_TRUNCATED,
                service="sanitize",
                session_id=session_id,
                request_id=request_id,
                original_chars=original,
                sanitized_chars=capped,
                truncated=True,
            )
        )

    @staticmethod
    def empty_result(*, session_id: str, request_id: str, original_chars: int) -> None:
        emit(
            ObsEvent(
                kind=EventKind.SANITIZE_EMPTY,
                service="sanitize",
                session_id=session_id,
                request_id=request_id,
                original_chars=original_chars,
                sanitized_chars=0,
            )
        )


class CBEmitter:
    """
    Emitters for circuit breaker state transitions.

    Usage in shared.py CircuitBreaker._on_failure / _on_success:

        from app.common.observability import CBEmitter
        CBEmitter.opened(service="llm", failures=5, threshold=5)
    """

    @staticmethod
    def opened(
        *, service: str, failures: int, threshold: int, session_id: str = ""
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CB_OPENED,
                service="cb",
                session_id=session_id,
                cb_name=service,
                cb_state="OPEN",
                cb_failures=failures,
                cb_threshold=threshold,
                ok=False,
            )
        )

    @staticmethod
    def half_open(*, service: str, session_id: str = "") -> None:
        emit(
            ObsEvent(
                kind=EventKind.CB_HALF_OPEN,
                service="cb",
                session_id=session_id,
                cb_name=service,
                cb_state="HALF_OPEN",
            )
        )

    @staticmethod
    def closed(*, service: str, session_id: str = "") -> None:
        emit(
            ObsEvent(
                kind=EventKind.CB_CLOSED,
                service="cb",
                session_id=session_id,
                cb_name=service,
                cb_state="CLOSED",
            )
        )

    @staticmethod
    def rejected(*, service: str, remain_s: float, session_id: str = "") -> None:
        emit(
            ObsEvent(
                kind=EventKind.CB_REJECTED,
                service="cb",
                session_id=session_id,
                cb_name=service,
                cb_state="OPEN",
                extra={"recovery_remain_s": round(remain_s, 1)},
                ok=False,
            )
        )


class RateLimitEmitter:
    """Emitters for RateLimiter (shared.py)."""

    @staticmethod
    def acquired(*, service: str, session_id: str = "", tokens: float = 1.0) -> None:
        emit(
            ObsEvent(
                kind=EventKind.RL_ACQUIRED,
                service="rl",
                session_id=session_id,
                rl_name=service,
                extra={"tokens": tokens},
            )
        )

    @staticmethod
    def exhausted(
        *, service: str, session_id: str = "", tokens_needed: float = 1.0
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.RL_EXHAUSTED,
                service="rl",
                session_id=session_id,
                rl_name=service,
                extra={"tokens_needed": tokens_needed},
                ok=False,
            )
        )

    @staticmethod
    def waiting(*, service: str, wait_s: float, session_id: str = "") -> None:
        emit(
            ObsEvent(
                kind=EventKind.RL_WAITING,
                service="rl",
                session_id=session_id,
                rl_name=service,
                latency_ms=wait_s * 1000,
            )
        )


class BulkheadEmitter:
    """Emitters for BulkheadPool (shared.py)."""

    @staticmethod
    def acquired(
        *, name: str, inflight: int, capacity: int, session_id: str = ""
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.BH_ACQUIRED,
                service="bh",
                session_id=session_id,
                bh_name=name,
                bh_inflight=inflight,
                bh_capacity=capacity,
            )
        )

    @staticmethod
    def saturated(
        *, name: str, inflight: int, capacity: int, session_id: str = ""
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.BH_SATURATED,
                service="bh",
                session_id=session_id,
                bh_name=name,
                bh_inflight=inflight,
                bh_capacity=capacity,
                ok=False,
            )
        )

    @staticmethod
    def released(*, name: str, session_id: str = "") -> None:
        emit(
            ObsEvent(
                kind=EventKind.BH_RELEASED,
                service="bh",
                session_id=session_id,
                bh_name=name,
            )
        )


class BudgetEmitter:
    """Emitters for LatencyBudget (shared.py)."""

    @staticmethod
    def ok(
        *, stage: str, budget_ms: float, remain_ms: float, session_id: str = ""
    ) -> None:
        used_pct: float = 100.0 * (1.0 - remain_ms / max(1.0, float(budget_ms)))
        emit(
            ObsEvent(
                kind=EventKind.BUDGET_OK,
                service="pipeline",
                session_id=session_id,
                stage=stage,
                budget_total_ms=budget_ms,
                budget_remain_ms=remain_ms,
                budget_used_pct=used_pct,
            )
        )

    @staticmethod
    def near(
        *, stage: str, budget_ms: float, remain_ms: float, session_id: str = ""
    ) -> None:
        used_pct: float = 100.0 * (1.0 - remain_ms / max(1.0, float(budget_ms)))
        emit(
            ObsEvent(
                kind=EventKind.BUDGET_NEAR,
                service="pipeline",
                session_id=session_id,
                stage=stage,
                budget_total_ms=budget_ms,
                budget_remain_ms=remain_ms,
                budget_used_pct=used_pct,
            )
        )

    @staticmethod
    def breached(
        *, stage: str, budget_ms: float, elapsed_ms: float, session_id: str = ""
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.BUDGET_BREACHED,
                service="pipeline",
                session_id=session_id,
                stage=stage,
                budget_total_ms=budget_ms,
                budget_remain_ms=0,
                budget_used_pct=100.0,
                latency_ms=elapsed_ms,
                ok=False,
            )
        )


class ControllerEmitter:
    """Emitters for controller.py desktop PTT controller."""

    @staticmethod
    def ptt_press(*, session_id: str, ptt_key: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_PTT_PRESS,
                service="controller",
                session_id=session_id,
                ptt_key=ptt_key,
            )
        )

    @staticmethod
    def ptt_release(*, session_id: str, recording_s: float, ptt_key: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_PTT_RELEASE,
                service="controller",
                session_id=session_id,
                recording_s=recording_s,
                ptt_key=ptt_key,
            )
        )

    @staticmethod
    def empty_recording(*, session_id: str, recording_s: float) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_EMPTY_RECORDING,
                service="controller",
                session_id=session_id,
                recording_s=recording_s,
            )
        )

    @staticmethod
    def interrupt(*, session_id: str, request_id: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_INTERRUPT,
                service="controller",
                session_id=session_id,
                request_id=request_id,
            )
        )

    @staticmethod
    def pipeline_error(
        *, session_id: str, request_id: str, error: str, error_type: str
    ) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_PIPELINE_ERROR,
                service="controller",
                session_id=session_id,
                request_id=request_id,
                error=error,
                error_type=error_type,
                ok=False,
            )
        )

    @staticmethod
    def shutdown(*, reason: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_SHUTDOWN,
                service="controller",
                session_reason=reason,
            )
        )

    @staticmethod
    def signal(*, signum: int, signame: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.CTRL_SIGNAL,
                service="controller",
                extra={"signum": signum, "signame": signame},
            )
        )


class RedisEmitter:
    """Emitters for Redis connectivity events (session_store, LLM cache, eval)."""

    @staticmethod
    def disconnected(*, service: str, error: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.REDIS_DISCONNECTED,
                service="redis",
                error=error,
                extra={"consumer": service},
                ok=False,
            )
        )

    @staticmethod
    def recovered(*, service: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.REDIS_RECOVERED,
                service="redis",
                extra={"consumer": service},
            )
        )

    @staticmethod
    def degraded(*, service: str, backend: str = "lru") -> None:
        emit(
            ObsEvent(
                kind=EventKind.REDIS_DEGRADED,
                service="redis",
                extra={"consumer": service},
                backend=backend,
            )
        )


class TranscriptEmitter:
    """Emitters for transcription.py."""

    @staticmethod
    def turn(*, session_id: str, request_id: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TRANSCRIPT_TURN,
                service="transcript",
                session_id=session_id,
                request_id=request_id,
            )
        )

    @staticmethod
    def queue_drop(*, session_id: str) -> None:
        emit(
            ObsEvent(
                kind=EventKind.TRANSCRIPT_QUEUE_DROP,
                service="transcript",
                session_id=session_id,
            )
        )


# ═════════════════════════════════════════════════════════════════════════════
#  SESSION-SCOPED OBSERVABILITY CONTEXT
# ═════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def session_context(session_id: str, request_id: str = ""):
    """
    Async context manager that binds session_id and request_id into
    the current async task's context vars for automatic injection.

        async with session_context(session_id=sid, request_id=rid):
            # all emit() and structlog calls inherit sid/rid
            STTEmitter.ok(...)   # session_id injected automatically

    Also sets session_id in W3C Baggage so it propagates across HTTP
    boundaries to remote STT / LLM / TTS microservices automatically.
    Use in voice_graph node functions, controller dispatch, and any
    async handler that processes a request scoped to a session.
    """
    rid = request_id or str(uuid.uuid4().hex)
    sid_token = _session_id_var.set(session_id)
    rid_token = _request_id_var.set(rid)

    # W3C Baggage — propagates session_id to downstream HTTP services
    # so remote spans carry the same session correlation ID.
    _OTEL.set_baggage(_OtelLayer.BAGGAGE_SESSION_ID, session_id)
    _OTEL.set_baggage(_OtelLayer.BAGGAGE_REQUEST_ID, rid)

    try:
        yield rid
    finally:
        _session_id_var.reset(sid_token)
        _request_id_var.reset(rid_token)


@contextmanager
def sync_session_context(
    session_id: str, request_id: str = ""
) -> Generator[str, None, None]:
    """
    Synchronous version for controller.py and thread-pool callbacks.
    Also sets W3C Baggage for the synchronous execution path.
    """
    rid = request_id or str(uuid.uuid4().hex)
    sid_token = _session_id_var.set(session_id)
    rid_token = _request_id_var.set(rid)
    _OTEL.set_baggage(_OtelLayer.BAGGAGE_SESSION_ID, session_id)
    _OTEL.set_baggage(_OtelLayer.BAGGAGE_REQUEST_ID, rid)
    try:
        yield rid
    finally:
        _session_id_var.reset(sid_token)
        _request_id_var.reset(rid_token)


# ═════════════════════════════════════════════════════════════════════════════
#  OPENTELEMETRY SPAN CONTEXT MANAGERS
# ═════════════════════════════════════════════════════════════════════════════
#
# Spans are created here — emit() only annotates them. Callers compose them:
#
#   async with pipeline_span(...) as root_span:
#       async with stt_span(...) as stt_span:
#           result = await stt_node.transcribe(...)
#           STTEmitter.ok(...)   ← annotates stt_span, also root_span indirectly
#       async with llm_span(...) as llm_span:
#           ...
#
# Every context manager:
#   1. Opens the span with a canonical name and baseline attributes
#   2. On normal exit, sets status OK
#   3. On exception exit, records the exception and sets status ERROR
#   4. Yields the raw span so callers can add extra attributes mid-flight
#
# When OTEL_ENABLED=false, the tracer returns _NoOpTracer which yields
# _NOOP_SPAN — a zero-overhead object with no context operations.


@asynccontextmanager
async def pipeline_span(
    *,
    session_id: str,
    request_id: str,
    audio_path: str = "",
    qos_tier: str = "STANDARD",
    mode: str = "api",
    version: str = "",
    parent_context: Any = None,
):
    """
    Root span for a complete pipeline execution (STT → LLM → TTS).

    Always the outermost span. All stage spans (stt_span, llm_span, etc.)
    are children of this span — they are started inside this context.

    parent_context: pass _OTEL.extract_context(request.headers) at HTTP
    ingress to stitch this trace into a distributed trace from the client.

    Yields the raw OTel span. Set extra attributes via span.set_attribute()
    if needed, e.g. after the graph resolves which model was used.
    """
    tracer = _OTEL.tracer("pipeline")
    span_name = f"ai.pipeline.{mode}"
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_QOS_TIER: qos_tier,
        _OtelLayer.ATTR_EXEC_MODE: mode,
        _OtelLayer.ATTR_GRAPH_VERSION: version,
    }
    if audio_path:
        attrs[_OtelLayer.ATTR_AUDIO_PATH] = audio_path

    kwargs: dict[str, Any] = {"attributes": attrs}
    if parent_context is not None:
        kwargs["context"] = parent_context

    with tracer.start_as_current_span(span_name, **kwargs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except asyncio.CancelledError:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.add_event("pipeline_cancelled")
                span.set_status(StatusCode.ERROR, "cancelled")
            raise
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def stt_span(
    *,
    session_id: str,
    request_id: str,
    audio_path: str = "",
    audio_bytes: int = 0,
):
    """
    Child span for a STT transcription call.
    Start this inside pipeline_span() so it appears as a child in Tempo.
    """
    tracer = _OTEL.tracer("stt")
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_AUDIO_PATH: audio_path,
        _OtelLayer.ATTR_AUDIO_BYTES: audio_bytes,
    }
    with tracer.start_as_current_span("ai.stt.transcribe", attributes=attrs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except asyncio.CancelledError:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.add_event("stt_cancelled")
                span.set_status(StatusCode.ERROR, "cancelled")
            raise
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def llm_span(
    *,
    session_id: str,
    request_id: str,
    model: str = "",
    streaming: bool = False,
    history_turns: int = 0,
    temperature: float = 0.0,
):
    """
    Child span for a LLM generation call (batch or streaming).

    Sets gen_ai.* semantic convention attributes on open. After the call
    completes, callers can set token counts directly on the yielded span:

        async with llm_span(...) as span:
            response = await llm.generate(prompt)
            span.set_attribute(_OtelLayer.ATTR_COMPL_TOKENS, response.completion_tokens)
    """
    tracer = _OTEL.tracer("llm")
    span_name = "ai.llm.stream" if streaming else "ai.llm.generate"
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_GEN_AI_SYSTEM: "openai",
        _OtelLayer.ATTR_GEN_AI_MODEL: model,
        _OtelLayer.ATTR_GEN_AI_STREAMING: streaming,
        _OtelLayer.ATTR_HISTORY_TURNS: history_turns,
    }
    if temperature:
        attrs["gen_ai.request.temperature"] = temperature
    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except asyncio.CancelledError:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.add_event("llm_cancelled")
                span.set_status(StatusCode.ERROR, "cancelled")
            raise
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def tts_span(
    *,
    session_id: str,
    request_id: str,
    voice: str = "",
    input_chars: int = 0,
    tts_format: str = "",
    streaming: bool = False,
):
    """
    Child span for a TTS synthesis call.

    Stream mode (synthesize_stream) uses span_name "ai.tts.stream" so
    Tempo's service graph differentiates batch vs streaming audio generation.
    """
    tracer = _OTEL.tracer("tts")
    span_name = "ai.tts.stream" if streaming else "ai.tts.synthesize"
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_TTS_VOICE: voice,
        _OtelLayer.ATTR_TTS_INPUT_CHARS: input_chars,
        _OtelLayer.ATTR_TTS_FORMAT: tts_format,
    }
    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except asyncio.CancelledError:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.add_event("tts_cancelled")
                span.set_status(StatusCode.ERROR, "cancelled")
            raise
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def eval_span(
    *,
    session_id: str,
    request_id: str,
    turn_idx: int = 0,
    model: str = "",
):
    """
    Child span for an evaluation engine scoring call.

    Eval runs off the critical path (fire-and-forget via asyncio.create_task)
    so this span will typically NOT be a child of the pipeline span — it has
    its own root context. This is correct: the trace shows eval as a separate
    async operation that starts after the pipeline turn completes.
    """
    tracer = _OTEL.tracer("eval")
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_EVAL_TURN_IDX: turn_idx,
        _OtelLayer.ATTR_EVAL_MODEL: model,
    }
    with tracer.start_as_current_span("ai.eval.score", attributes=attrs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def session_op_span(
    operation: str,
    session_id: str = "",
    ip_masked: str = "",
):
    """
    Span for a session store operation (register, load, end, etc.).

    operation examples: "register", "load", "end", "ip_check", "ip_rebind"

    These spans are cheap and should be used around every Redis interaction
    in session_store.py so Tempo shows Redis latency contribution in context.
    """
    tracer = _OTEL.tracer("session")
    attrs: dict[str, Any] = {
        "ai.session.operation": operation,
        _OtelLayer.ATTR_SESSION_ID: session_id,
    }
    if ip_masked:
        attrs[_OtelLayer.ATTR_IP_MASKED] = ip_masked
    with tracer.start_as_current_span(
        f"ai.session.{operation}", attributes=attrs
    ) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def memory_span(
    operation: str,
    *,
    session_id: str = "",
    request_id: str = "",
    history_depth: int = 0,
):
    """
    Span for a conversation memory operation (resolve or commit).

    operation: "resolve" | "commit" | "compress"

    Sits inside the LLM span so Tempo shows memory latency as a sub-cost
    of the LLM stage, not as a separate pipeline stage.
    """
    tracer = _OTEL.tracer("memory")
    attrs: dict[str, Any] = {
        "ai.memory.operation": operation,
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_HISTORY_DEPTH: history_depth,
    }
    with tracer.start_as_current_span(
        f"ai.memory.{operation}", attributes=attrs
    ) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


@asynccontextmanager
async def sanitize_span(
    *,
    session_id: str = "",
    request_id: str = "",
    input_chars: int = 0,
):
    """
    Span for the TTS pre-processing sanitizer.
    Sits inside the sanitize node of voice_graph, which is between LLM and TTS.
    """
    tracer = _OTEL.tracer("sanitize")
    attrs: dict[str, Any] = {
        _OtelLayer.ATTR_SESSION_ID: session_id,
        _OtelLayer.ATTR_REQUEST_ID: request_id,
        _OtelLayer.ATTR_ORIG_CHARS: input_chars,
    }
    with tracer.start_as_current_span("ai.sanitize", attributes=attrs) as span:
        try:
            yield span
            if _OTEL._enabled and span.is_recording():  # noqa
                span.set_status(StatusCode.OK)
        except Exception as exc:
            if _OTEL._enabled and span.is_recording():  # noqa
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc)[:200])
            raise


# ── Trace propagation helpers (public API) ────────────────────────────────────


def inject_trace_headers(headers: dict[str, str]) -> dict[str, str]:
    """
    Inject W3C traceparent + tracestate + baggage into *headers*.
    Call before any outbound HTTP request in remote node clients.

    Usage in RemoteSTTClient, RemoteLLMClient, RemoteTTSClient::

        headers = inject_trace_headers({"Content-Type": "application/json"})
        resp = await client.post(url, headers=headers, ...)
    """
    return _OTEL.inject_headers(headers)


def extract_trace_context(headers: dict[str, str]) -> Any:
    """
    Extract OTel context from inbound HTTP request headers.
    Pass the result as context= to the appropriate span context manager.

    Usage in remote service FastAPI request handlers::

        ctx = extract_trace_context(dict(request.headers))
        async with pipeline_span(..., parent_context=ctx):
            ...
    """
    return _OTEL.extract_context(headers)


def get_trace_id() -> str:
    """
    Return the current trace ID as a 32-char hex string.
    Empty string when OTEL_ENABLED=false or no span is active.
    Useful for embedding in API responses so clients can reference traces.
    """
    return _OTEL.trace_id()


def get_span_id() -> str:
    """Return the current span ID as a 16-char hex string."""
    return _OTEL.span_id()


# ═════════════════════════════════════════════════════════════════════════════
#  TIMING HELPERS
# ═════════════════════════════════════════════════════════════════════════════


class StageTimer:
    """
    Monotonic wall-clock timer for a single pipeline stage.

        t = StageTimer()
        # ... do work ...
        latency_ms = t.stop()

    Or:

        with StageTimer() as t:
            # ... do work ...
        print(t.elapsed_ms)

    elapsed_ms is also available without stopping.
    """

    def __init__(self) -> None:
        self._start: float = time.monotonic()
        self._end: Optional[float] = None

    def stop(self) -> float:
        """Return elapsed milliseconds and freeze the timer."""
        if self._end is None:
            self._end = time.monotonic()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        end = self._end if self._end is not None else time.monotonic()
        return (end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0

    def __enter__(self) -> "StageTimer":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ═════════════════════════════════════════════════════════════════════════════
#  GRAFANA DASHBOARD PROVISIONING
# ═════════════════════════════════════════════════════════════════════════════


def _panel(
    id_: int,
    title: str,
    targets: list[dict],
    *,
    kind: str = "timeseries",
    unit: str = "",
    x: int = 0,
    y: int = 0,
    w: int = 12,
    h: int = 8,
    fill: int = 0,
    stack: bool = False,
    thresholds: list | None = None,
) -> dict:
    panel: dict[str, Any] = {
        "id": id_,
        "title": title,
        "type": kind,
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "targets": targets,
        "options": {},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "fillOpacity": fill,
                    "stacking": {"mode": "normal" if stack else "none"},
                    "lineWidth": 2,
                },
                "unit": unit,
            },
            "overrides": [],
        },
    }
    if thresholds:
        panel["fieldConfig"]["defaults"]["thresholds"] = {
            "mode": "absolute",
            "steps": thresholds,
        }
    return panel


def _stat(
    id_: int, title: str, expr: str, *, unit: str = "", x: int = 0, y: int = 0
) -> dict:
    return {
        "id": id_,
        "title": title,
        "type": "stat",
        "datasource": {"type": "prometheus", "uid": "prometheus"},
        "gridPos": {"x": x, "y": y, "w": 4, "h": 4},
        "targets": [{"expr": expr, "legendFormat": title}],
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"]},
            "colorMode": "background",
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "color": {"mode": "thresholds"},
                "thresholds": {
                    "steps": [
                        {"color": "green", "value": None},
                        {"color": "yellow", "value": 0.5},
                        {"color": "red", "value": 0.8},
                    ]
                },
            },
            "overrides": [],
        },
    }


def _target(expr: str, legend: str = "") -> dict:
    return {"expr": expr, "legendFormat": legend or expr, "interval": "30s"}


def build_grafana_dashboard() -> dict:
    """
    Build and return the full Grafana dashboard JSON for the AI pipeline.

    Every metric emitted by this module has a corresponding panel.
    """
    panels: list[dict] = []
    pid = 1

    def row(title: str, y: int) -> dict:
        nonlocal pid
        r = {
            "id": pid,
            "title": title,
            "type": "row",
            "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1},
        }
        pid += 1
        return r

    y = 0

    # ── Row 0: KPI stat strip ────────────────────────────────────────────────
    panels.append(row("Overview", y))
    y += 1
    kpis = [
        ("Active Pipelines", "ai_pipeline_inflight", "short"),
        ("Active Sessions", "ai_session_active", "short"),
        (
            "Pipeline P99 (s)",
            "histogram_quantile(0.99, sum(rate(ai_pipeline_latency_seconds_bucket[5m])) by (le))",
            "s",
        ),
        ("Redis Degraded", "ai_redis_degraded_mode", "short"),
        ("CB Open (any)", "sum(ai_circuit_breaker_state == 2)", "short"),
        (
            "Eval Budget Exhaust",
            "sum(increase(ai_eval_budget_exhausted_total[1h]))",
            "short",
        ),
    ]
    for i, (title, expr, unit) in enumerate(kpis):
        panels.append(_stat(pid, title, expr, unit=unit, x=(i % 6) * 4, y=y))
        pid += 1
    y += 4

    # ── Row 1: Pipeline throughput & latency ──────────────────────────────────
    panels.append(row("Pipeline", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "Pipeline Requests/min",
            [
                _target(
                    'sum(rate(ai_pipeline_requests_total{status="ok"}[1m])) * 60', "ok"
                ),
                _target(
                    'sum(rate(ai_pipeline_requests_total{status="error"}[1m])) * 60',
                    "error",
                ),
                _target(
                    'sum(rate(ai_pipeline_requests_total{status="cache"}[1m])) * 60',
                    "cache",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=8,
            fill=5,
            stack=True,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Pipeline Latency Percentiles",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_pipeline_latency_seconds_bucket[5m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_pipeline_latency_seconds_bucket[5m])) by (le))",
                    "p90",
                ),
                _target(
                    "histogram_quantile(0.99, sum(rate(ai_pipeline_latency_seconds_bucket[5m])) by (le))",
                    "p99",
                ),
            ],
            unit="s",
            x=12,
            y=y,
            w=12,
            h=8,
        )
    )
    pid += 1
    y += 8

    panels.append(
        _panel(
            pid,
            "Stage Latency P95 (s)",
            [
                _target(
                    f'histogram_quantile(0.95, sum(rate(ai_pipeline_stage_latency_seconds_bucket{{stage="{s}"}}[5m])) by (le))',
                    s,
                )
                for s in ("stt", "llm", "tts", "sanitize")
            ],
            unit="s",
            x=0,
            y=y,
            w=12,
            h=8,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Pipeline Errors by Stage",
            [
                _target(
                    "sum(rate(ai_pipeline_stage_errors_total[1m])) by (stage)",
                    "{{stage}}",
                ),
            ],
            x=12,
            y=y,
            w=8,
            h=8,
            fill=4,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Retries & Aborts",
            [
                _target(
                    "sum(rate(ai_pipeline_stage_retries_total[5m])) by (stage)",
                    "retry:{{stage}}",
                ),
                _target(
                    "sum(rate(ai_pipeline_aborted_total[5m])) by (reason)",
                    "abort:{{reason}}",
                ),
            ],
            x=20,
            y=y,
            w=4,
            h=8,
        )
    )
    pid += 1
    y += 8

    panels.append(
        _panel(
            pid,
            "Load Shedding & Degraded",
            [
                _target(
                    "sum(rate(ai_pipeline_load_shed_total[5m])) by (tier)",
                    "shed:{{tier}}",
                ),
                _target(
                    "sum(rate(ai_pipeline_degraded_total[5m])) by (stage)",
                    "degraded:{{stage}}",
                ),
                _target(
                    "sum(rate(ai_pipeline_cancelled_total[5m])) by (stage)",
                    "cancelled:{{stage}}",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Inflight by QoS Tier",
            [
                _target("ai_pipeline_inflight", "{{tier}}"),
            ],
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 2: STT ────────────────────────────────────────────────────────────
    panels.append(row("STT — Speech-to-Text", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "STT Requests/min",
            [
                _target(
                    "sum(rate(ai_stt_requests_total[1m])) by (status) * 60",
                    "{{status}}",
                ),
            ],
            x=0,
            y=y,
            w=8,
            h=7,
            fill=3,
            stack=True,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "STT Latency Percentiles",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_stt_latency_seconds_bucket[5m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_stt_latency_seconds_bucket[5m])) by (le))",
                    "p90",
                ),
                _target(
                    "histogram_quantile(0.99, sum(rate(ai_stt_latency_seconds_bucket[5m])) by (le))",
                    "p99",
                ),
            ],
            unit="s",
            x=8,
            y=y,
            w=8,
            h=7,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Language Confidence Distribution",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_stt_language_confidence_bucket[10m])) by (le, language))",
                    "{{language}} p50",
                ),
            ],
            x=16,
            y=y,
            w=8,
            h=7,
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.5},
                {"color": "green", "value": 0.8},
            ],
        )
    )
    pid += 1
    y += 7

    panels.append(
        _panel(
            pid,
            "STT Edge Cases",
            [
                _target("sum(rate(ai_stt_empty_audio_total[5m]))", "empty_audio"),
                _target("sum(rate(ai_stt_low_confidence_total[5m]))", "low_confidence"),
                _target(
                    "sum(rate(ai_stt_remote_fallback_total[5m]))", "remote_fallback"
                ),
                _target("sum(rate(ai_stt_path_rejected_total[5m]))", "path_rejected"),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Audio Duration Distribution (p50/p90)",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_stt_audio_duration_seconds_bucket[10m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_stt_audio_duration_seconds_bucket[10m])) by (le))",
                    "p90",
                ),
            ],
            unit="s",
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 3: LLM ────────────────────────────────────────────────────────────
    panels.append(row("LLM — Language Model", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "LLM Requests/min by Status & Model",
            [
                _target(
                    "sum(rate(ai_llm_requests_total[1m])) by (status, model) * 60",
                    "{{status}}:{{model}}",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=8,
            fill=4,
            stack=True,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "LLM Latency Percentiles",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_llm_latency_seconds_bucket[5m])) by (le, model))",
                    "p50 {{model}}",
                ),
                _target(
                    "histogram_quantile(0.95, sum(rate(ai_llm_latency_seconds_bucket[5m])) by (le, model))",
                    "p95 {{model}}",
                ),
            ],
            unit="s",
            x=12,
            y=y,
            w=12,
            h=8,
        )
    )
    pid += 1
    y += 8

    panels.append(
        _panel(
            pid,
            "Token Consumption Rate (tokens/min)",
            [
                _target(
                    "sum(rate(ai_llm_tokens_total[1m])) by (model, kind) * 60",
                    "{{model}}:{{kind}}",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=7,
            fill=5,
            stack=True,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Cache Hit Rate",
            [
                _target(
                    "sum(rate(ai_llm_cache_hits_total[5m])) / (sum(rate(ai_llm_cache_hits_total[5m])) + sum(rate(ai_llm_cache_misses_total[5m])))",
                    "hit_rate",
                ),
            ],
            unit="percentunit",
            x=12,
            y=y,
            w=6,
            h=7,
            thresholds=[
                {"color": "red", "value": None},
                {"color": "yellow", "value": 0.2},
                {"color": "green", "value": 0.5},
            ],
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "LLM Fallback & Stampede",
            [
                _target(
                    "sum(rate(ai_llm_model_fallback_total[5m])) by (primary, fallback)",
                    "fallback {{primary}}→{{fallback}}",
                ),
                _target("sum(rate(ai_llm_cache_stampede_total[5m]))", "cache_stampede"),
            ],
            x=18,
            y=y,
            w=6,
            h=7,
        )
    )
    pid += 1
    y += 7

    panels.append(
        _panel(
            pid,
            "Prompt Token Distribution (p50/p90/p99)",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_llm_tokens_prompt_bucket[10m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_llm_tokens_prompt_bucket[10m])) by (le))",
                    "p90",
                ),
                _target(
                    "histogram_quantile(0.99, sum(rate(ai_llm_tokens_prompt_bucket[10m])) by (le))",
                    "p99",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "History Depth at LLM Call",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_llm_history_turns_bucket[10m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_llm_history_turns_bucket[10m])) by (le))",
                    "p90",
                ),
            ],
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 4: TTS ────────────────────────────────────────────────────────────
    panels.append(row("TTS — Text-to-Speech", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "TTS Requests/min",
            [
                _target(
                    "sum(rate(ai_tts_requests_total[1m])) by (status, voice) * 60",
                    "{{status}}:{{voice}}",
                ),
            ],
            x=0,
            y=y,
            w=8,
            h=7,
            fill=3,
            stack=True,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "TTS Latency Percentiles",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_tts_latency_seconds_bucket[5m])) by (le, voice))",
                    "p50 {{voice}}",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_tts_latency_seconds_bucket[5m])) by (le, voice))",
                    "p90 {{voice}}",
                ),
            ],
            unit="s",
            x=8,
            y=y,
            w=8,
            h=7,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "TTS Edge Cases",
            [
                _target(
                    "sum(rate(ai_tts_apology_fallback_total[5m]))", "apology_fallback"
                ),
                _target("sum(rate(ai_tts_chunk_errors_total[5m]))", "chunk_errors"),
                _target(
                    'sum(rate(ai_tts_s3_uploads_total{status="error"}[5m]))',
                    "s3_errors",
                ),
                _target("sum(rate(ai_tts_file_cleanups_total[5m]))", "file_cleanups"),
            ],
            x=16,
            y=y,
            w=8,
            h=7,
        )
    )
    pid += 1
    y += 7

    # ── Row 5: Evaluation ─────────────────────────────────────────────────────
    panels.append(row("Evaluation Engine", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "Eval Score Distribution",
            [
                _target(
                    "histogram_quantile(0.25, sum(rate(ai_eval_scores_bucket[30m])) by (le))",
                    "p25",
                ),
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_eval_scores_bucket[30m])) by (le))",
                    "median",
                ),
                _target(
                    "histogram_quantile(0.75, sum(rate(ai_eval_scores_bucket[30m])) by (le))",
                    "p75",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=7,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Eval Skips & Budget",
            [
                _target(
                    "sum(rate(ai_eval_skipped_total[5m])) by (reason)",
                    "skip:{{reason}}",
                ),
                _target(
                    "sum(rate(ai_eval_budget_exhausted_total[5m]))", "budget_exhausted"
                ),
                _target(
                    "sum(rate(ai_eval_model_fallback_total[5m]))", "model_fallback"
                ),
            ],
            x=12,
            y=y,
            w=12,
            h=7,
        )
    )
    pid += 1
    y += 7

    # ── Row 6: Sessions ───────────────────────────────────────────────────────
    panels.append(row("Session Management", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "Active Sessions",
            [
                _target("ai_session_active", "active"),
            ],
            x=0,
            y=y,
            w=6,
            h=7,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 50},
                {"color": "red", "value": 100},
            ],
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Session Lifecycle Events/min",
            [
                _target("sum(rate(ai_session_created_total[1m])) * 60", "created"),
                _target(
                    "sum(rate(ai_session_ended_total[1m])) by (reason) * 60",
                    "ended:{{reason}}",
                ),
                _target(
                    "sum(rate(ai_session_rejected_total[1m])) by (reason) * 60",
                    "rejected:{{reason}}",
                ),
            ],
            x=6,
            y=y,
            w=12,
            h=7,
            fill=2,
            stack=False,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "IP Security Events",
            [
                _target("sum(rate(ai_session_ip_changes_total[5m]))", "ip_changes"),
                _target(
                    "sum(rate(ai_session_ip_limit_total[5m]))", "ip_limit_exceeded"
                ),
                _target(
                    "sum(rate(ai_session_lru_hits_total[5m]))", "lru_hits_degraded"
                ),
            ],
            x=18,
            y=y,
            w=6,
            h=7,
        )
    )
    pid += 1
    y += 7

    panels.append(
        _panel(
            pid,
            "Session Duration Distribution",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_session_duration_seconds_bucket[30m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_session_duration_seconds_bucket[30m])) by (le))",
                    "p90",
                ),
            ],
            unit="s",
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Turns per Session Distribution",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_session_turns_bucket[30m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_session_turns_bucket[30m])) by (le))",
                    "p90",
                ),
            ],
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 7: Resilience ─────────────────────────────────────────────────────
    panels.append(
        row("Resilience — Circuit Breakers, Rate Limits, Bulkheads, Budget", y)
    )
    y += 1
    panels.append(
        _panel(
            pid,
            "Circuit Breaker State (0=Closed 1=HalfOpen 2=Open)",
            [
                _target("ai_circuit_breaker_state", "{{service}}"),
            ],
            x=0,
            y=y,
            w=12,
            h=7,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 2},
            ],
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "CB Events/min",
            [
                _target(
                    "sum(rate(ai_circuit_breaker_opened_total[1m])) by (service) * 60",
                    "opened:{{service}}",
                ),
                _target(
                    "sum(rate(ai_circuit_breaker_rejected_total[1m])) by (service) * 60",
                    "rejected:{{service}}",
                ),
                _target(
                    "sum(rate(ai_circuit_breaker_closed_total[1m])) by (service) * 60",
                    "closed:{{service}}",
                ),
            ],
            x=12,
            y=y,
            w=12,
            h=7,
        )
    )
    pid += 1
    y += 7

    panels.append(
        _panel(
            pid,
            "Rate Limiter Exhaustions/min",
            [
                _target(
                    "sum(rate(ai_rate_limit_exhausted_total[1m])) by (service) * 60",
                    "{{service}}",
                ),
            ],
            x=0,
            y=y,
            w=8,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Bulkhead Inflight & Saturation",
            [
                _target("ai_bulkhead_inflight", "inflight:{{name}}"),
                _target(
                    "sum(rate(ai_bulkhead_saturated_total[5m])) by (name)",
                    "saturated:{{name}}",
                ),
            ],
            x=8,
            y=y,
            w=8,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Latency Budget Breaches/min",
            [
                _target(
                    "sum(rate(ai_latency_budget_breached_total[1m])) by (stage) * 60",
                    "{{stage}}",
                ),
            ],
            x=16,
            y=y,
            w=8,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 8: Sanitizer & Memory ─────────────────────────────────────────────
    panels.append(row("Sanitizer & Conversation Memory", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "Sanitizer Events/min",
            [
                _target(
                    "sum(rate(ai_sanitize_total[1m])) by (status) * 60", "{{status}}"
                ),
                _target(
                    "sum(rate(ai_sanitize_injection_total[1m])) * 60",
                    "injection_detected",
                ),
                _target("sum(rate(ai_sanitize_truncated_total[1m])) * 60", "truncated"),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Memory Compressions & LRU Fallbacks",
            [
                _target("sum(rate(ai_memory_compressions_total[5m]))", "compressions"),
                _target("sum(rate(ai_memory_lru_fallback_total[5m]))", "lru_fallbacks"),
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_memory_compression_ratio_bucket[30m])) by (le))",
                    "ratio_p50",
                ),
            ],
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 9: Controller & Infrastructure ───────────────────────────────────
    panels.append(row("Controller & Infrastructure", y))
    y += 1
    panels.append(
        _panel(
            pid,
            "PTT Activity",
            [
                _target(
                    "sum(rate(ai_controller_ptt_presses_total[1m])) * 60", "presses/min"
                ),
                _target(
                    "sum(rate(ai_controller_interrupts_total[1m])) * 60",
                    "interrupts/min",
                ),
                _target(
                    "sum(rate(ai_controller_empty_recordings_total[1m])) * 60",
                    "empty/min",
                ),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Recording Duration Distribution",
            [
                _target(
                    "histogram_quantile(0.50, sum(rate(ai_controller_recording_duration_seconds_bucket[10m])) by (le))",
                    "p50",
                ),
                _target(
                    "histogram_quantile(0.90, sum(rate(ai_controller_recording_duration_seconds_bucket[10m])) by (le))",
                    "p90",
                ),
            ],
            unit="s",
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    panels.append(
        _panel(
            pid,
            "Redis Health",
            [
                _target("ai_redis_degraded_mode", "degraded"),
                _target("sum(rate(ai_redis_disconnects_total[5m]))", "disconnects/5m"),
                _target("sum(rate(ai_redis_recovered_total[5m]))", "recoveries/5m"),
            ],
            x=0,
            y=y,
            w=12,
            h=6,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "red", "value": 1},
            ],
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Transcript Queue",
            [
                _target("sum(rate(ai_transcript_turns_total[1m])) * 60", "turns/min"),
                _target("sum(rate(ai_transcript_drops_total[5m]))", "queue_drops/5m"),
            ],
            x=12,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1
    y += 6

    # ── Row 10: OpenTelemetry Tracing ─────────────────────────────────────────
    panels.append(row("OpenTelemetry Traces (Tempo)", y))
    y += 1

    # Tempo service map (derived from tempo_service_graph_request_total)
    panels.append(
        {
            "id": pid,
            "title": "Service Dependency Graph",
            "type": "nodeGraph",
            "datasource": {"type": "tempo", "uid": "tempo"},
            "gridPos": {"x": 0, "y": y, "w": 12, "h": 10},
            "targets": [
                {
                    "datasource": {"type": "tempo", "uid": "tempo"},
                    "queryType": "serviceMap",
                }
            ],
            "options": {},
            "fieldConfig": {"defaults": {}, "overrides": []},
        }
    )
    pid += 1

    # Trace search — pre-filtered for errors on the voice-pipeline service
    panels.append(
        {
            "id": pid,
            "title": "Error Traces — Voice Pipeline",
            "type": "traces",
            "datasource": {"type": "tempo", "uid": "tempo"},
            "gridPos": {"x": 12, "y": y, "w": 12, "h": 10},
            "targets": [
                {
                    "datasource": {"type": "tempo", "uid": "tempo"},
                    "queryType": "traceql",
                    "query": (
                        '{resource.service.name="voice-pipeline" '
                        "&& status=error} | select(ai.session.id, ai.pipeline.stage, ai.latency_ms)"
                    ),
                    "limit": 20,
                }
            ],
            "options": {"frameType": "trace"},
            "fieldConfig": {"defaults": {}, "overrides": []},
        }
    )
    pid += 1
    y += 10

    # Tempo RED metrics derived from span data (request, error, duration)
    panels.append(
        _panel(
            pid,
            "Trace Request Rate by Service (from Tempo)",
            [
                _target(
                    'sum(rate(tempo_service_graph_request_total{server=~"ai.*"}[1m])) by (server) * 60',
                    "{{server}}",
                ),
            ],
            x=0,
            y=y,
            w=8,
            h=7,
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Trace Error Rate by Service (from Tempo)",
            [
                _target(
                    'sum(rate(tempo_service_graph_request_failed_total{server=~"ai.*"}[5m])) by (server) '
                    '/ sum(rate(tempo_service_graph_request_total{server=~"ai.*"}[5m])) by (server)',
                    "{{server}}",
                ),
            ],
            unit="percentunit",
            x=8,
            y=y,
            w=8,
            h=7,
            thresholds=[
                {"color": "green", "value": None},
                {"color": "yellow", "value": 0.01},
                {"color": "red", "value": 0.05},
            ],
        )
    )
    pid += 1

    panels.append(
        _panel(
            pid,
            "Trace Duration P99 by Service (from Tempo)",
            [
                _target(
                    'histogram_quantile(0.99, sum(rate(tempo_service_graph_request_duration_seconds_bucket{server=~"ai.*"}[5m])) by (le, server))',
                    "p99 {{server}}",
                ),
            ],
            unit="s",
            x=16,
            y=y,
            w=8,
            h=7,
        )
    )
    pid += 1
    y += 7

    # Span-level latency from OTel metrics (pushed via OTLP when OTEL_ENABLED=true)
    panels.append(
        _panel(
            pid,
            "Stage Span Duration P95 (OTel histogram)",
            [
                _target(
                    'histogram_quantile(0.95, sum(rate(duration_milliseconds_bucket{span_name=~"ai\\..*"}[5m])) by (le, span_name))',
                    "{{span_name}}",
                ),
            ],
            unit="ms",
            x=0,
            y=y,
            w=12,
            h=6,
        )
    )
    pid += 1

    panels.append(
        {
            "id": pid,
            "title": "Trace Search by Session ID",
            "type": "traces",
            "datasource": {"type": "tempo", "uid": "tempo"},
            "description": (
                "Search for all traces belonging to a specific session. "
                "Enter a session_id in the $session_id template variable above."
            ),
            "gridPos": {"x": 12, "y": y, "w": 12, "h": 6},
            "targets": [
                {
                    "datasource": {"type": "tempo", "uid": "tempo"},
                    "queryType": "traceql",
                    "query": '{span.ai.session.id="${session_id}"}',
                    "limit": 50,
                }
            ],
            "options": {"frameType": "trace"},
            "fieldConfig": {"defaults": {}, "overrides": []},
        }
    )
    pid += 1
    y += 6

    return {
        "title": "AI Voice Pipeline — Observability",
        "uid": "ai-voice-pipeline",
        "schemaVersion": 38,
        "version": 1,
        "refresh": "30s",
        "time": {"from": "now-1h", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "tags": ["ai", "llm", "stt", "tts", "voice", "pipeline"],
        "templating": {
            "list": [
                {
                    "name": "session_id",
                    "type": "textbox",
                    "label": "Session ID",
                    "current": {"value": ""},
                    "hide": 0,
                },
                {
                    "name": "tier",
                    "type": "custom",
                    "label": "QoS Tier",
                    "options": [
                        {"selected": True, "text": "All", "value": ""},
                        {"text": "REALTIME", "value": "REALTIME"},
                        {"text": "STANDARD", "value": "STANDARD"},
                        {"text": "BATCH", "value": "BATCH"},
                    ],
                    "current": {"text": "All", "value": ""},
                },
            ]
        },
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {"type": "grafana"},
                    "enable": True,
                    "hide": True,
                    "name": "Annotations & Alerts",
                    "type": "dashboard",
                }
            ]
        },
        "panels": panels,
    }


def write_grafana_dashboard(out_dir: str = GRAFANA_OUT_DIR) -> Path:
    """
    Write the Grafana dashboard JSON and provisioning YAML to out_dir.

    File layout written:
        {out_dir}/dashboards/ai_pipeline.json
        {out_dir}/provisioning/datasources/prometheus.yaml
        {out_dir}/provisioning/datasources/tempo.yaml
        {out_dir}/provisioning/dashboards/ai_pipeline.yaml

    Tempo datasource
    ────────────────
    Configured with:
      - serviceMap.datasource = Prometheus (draws service dependency graph
        from tempo_service_graph_request_total metrics generated by Tempo)
      - nodeGraph.enabled = true (visual call graph in trace view)
      - tracesToLogsV2: links from a trace span to the matching Loki log
        lines using trace_id as the filter — requires Loki datasource
      - tracesToMetrics: links from a span to Prometheus metric charts
        filtered by session_id and stage labels

    Prometheus datasource
    ─────────────────────
    Configured with exemplarTraceIdDestinations pointing to Tempo so that
    Prometheus histogram exemplars (automatically attached by OTel SDK when
    OTEL_ENABLED=true) link directly to their trace in Tempo.
    """
    root = Path(out_dir)
    dash_dir = root / "dashboards"
    ds_dir = root / "provisioning" / "datasources"
    prov_dir = root / "provisioning" / "dashboards"

    for d in (dash_dir, ds_dir, prov_dir):
        d.mkdir(parents=True, exist_ok=True)

    dashboard = build_grafana_dashboard()
    dash_path = dash_dir / "ai_pipeline.json"
    dash_path.write_text(json.dumps(dashboard, indent=2))

    # ── Prometheus datasource (with exemplar → Tempo link) ───────────────────
    ds_path = ds_dir / "prometheus.yaml"
    ds_path.write_text(
        "apiVersion: 1\n"
        "datasources:\n"
        "  - name: Prometheus\n"
        "    type: prometheus\n"
        "    uid: prometheus\n"
        "    access: proxy\n"
        f"    url: {os.getenv('PROMETHEUS_URL', 'http://prometheus:9090')}\n"
        "    isDefault: true\n"
        "    editable: false\n"
        "    jsonData:\n"
        "      exemplarTraceIdDestinations:\n"
        "        - name: trace_id\n"
        "          datasourceUid: tempo\n"
    )

    # ── Tempo datasource (trace backend) ─────────────────────────────────────
    tempo_url = os.getenv("TEMPO_URL", "http://localhost:3200")
    loki_url = os.getenv("LOKI_URL", "http://localhost:3100")
    tempo_path = ds_dir / "tempo.yaml"
    tempo_path.write_text(
        "apiVersion: 1\n"
        "datasources:\n"
        "  - name: Tempo\n"
        "    type: tempo\n"
        "    uid: tempo\n"
        "    access: proxy\n"
        f"    url: {tempo_url}\n"
        "    isDefault: false\n"
        "    editable: false\n"
        "    jsonData:\n"
        "      httpMethod: GET\n"
        "      serviceMap:\n"
        "        datasourceUid: prometheus\n"
        "      nodeGraph:\n"
        "        enabled: true\n"
        "      search:\n"
        "        hide: false\n"
        "      lokiSearch:\n"
        "        datasourceUid: loki\n"
        "      tracesToLogsV2:\n"
        "        datasourceUid: loki\n"
        "        spanStartTimeShift: -1m\n"
        "        spanEndTimeShift: 1m\n"
        "        filterByTraceID: true\n"
        "        filterBySpanID: false\n"
        "        customQuery: true\n"
        '        query: \'{service="voice-pipeline"} | json | trace_id="$${__span.traceId}"\'\n'
        "      tracesToMetrics:\n"
        "        datasourceUid: prometheus\n"
        "        spanStartTimeShift: -1m\n"
        "        spanEndTimeShift: 1m\n"
        "        tags:\n"
        "          - key: ai.session.id\n"
        "            value: session_id\n"
        "          - key: ai.pipeline.stage\n"
        "            value: stage\n"
        "        queries:\n"
        "          - name: Pipeline latency\n"
        "            query: histogram_quantile(0.95, sum(rate(ai_pipeline_latency_seconds_bucket{$$__tags}[5m])) by (le))\n"
        "          - name: Stage errors\n"
        "            query: sum(rate(ai_pipeline_stage_errors_total{$$__tags}[5m])) by (stage)\n"
    )

    # ── Loki datasource (for log-to-trace correlation) ───────────────────────
    loki_path = ds_dir / "loki.yaml"
    loki_path.write_text(
        "apiVersion: 1\n"
        "datasources:\n"
        "  - name: Loki\n"
        "    type: loki\n"
        "    uid: loki\n"
        "    access: proxy\n"
        f"    url: {loki_url}\n"
        "    isDefault: false\n"
        "    editable: false\n"
        "    jsonData:\n"
        "      derivedFields:\n"
        "        - datasourceUid: tempo\n"
        '          matcherRegex: \'"trace_id":"([a-f0-9]{32})"\'\n'
        "          name: TraceID\n"
        "          url: '$$${__value.raw}'\n"
        "          urlDisplayLabel: Open in Tempo\n"
        "        - datasourceUid: tempo\n"
        "          matcherRegex: 'trace_id=([a-f0-9]{32})'\n"
        "          name: TraceID\n"
        "          url: '$$${__value.raw}'\n"
        "          urlDisplayLabel: Open in Tempo\n"
    )

    # ── Dashboard provisioning ────────────────────────────────────────────────
    prov_path = prov_dir / "ai_pipeline.yaml"
    prov_path.write_text(
        "apiVersion: 1\n"
        "providers:\n"
        "  - name: AI Pipeline\n"
        "    orgId: 1\n"
        "    type: file\n"
        "    disableDeletion: false\n"
        "    updateIntervalSeconds: 30\n"
        "    allowUiUpdates: true\n"
        f"    options:\n"
        f"      path: {dash_dir.resolve()}\n"
    )

    return dash_path


# ═════════════════════════════════════════════════════════════════════════════
#  STARTUP / BOOTSTRAP
# ═════════════════════════════════════════════════════════════════════════════

_bootstrap_lock = threading.Lock()
_bootstrapped = False


def bootstrap(
    *,
    start_prometheus: bool = PROMETHEUS_ENABLED,
    start_mongo: bool = MONGO_ENABLED,
    write_grafana: bool = True,
    start_otel: bool = OTEL_ENABLED,
) -> None:
    """
    Idempotent bootstrap. Call once at process startup before any
    component emits events.

        from app.common.observability import bootstrap
        bootstrap()

    Performs (in order):
      1. configure_logging()       — structlog setup with trace_id injection
      2. get_metrics()             — Prometheus metric registration
      3. start_http_server()       — Prometheus /metrics endpoint
      4. _OTEL.initialize()        — OTel TracerProvider + MeterProvider + Propagator
      5. _MONGO.start()            — MongoDB writer thread
      6. write_grafana_dashboard() — provisioning JSON + YAML files (incl. Tempo)
    """

    global _bootstrapped
    with _bootstrap_lock:
        if _bootstrapped:
            return

        configure_logging()
        get_metrics()

        if start_prometheus:
            try:
                start_http_server(PROMETHEUS_PORT)
                _log.info(
                    "prometheus_metrics_server_started",
                    port=PROMETHEUS_PORT,
                    pid=os.getpid(),
                )
            except OSError as _prom_err:
                # Port already bound — most commonly caused by uvicorn's reloader
                # process importing the app before the worker process does.
                # The reloader owns the socket but never records real metrics.
                # Fix: pass reload=False to uvicorn.run(), or use --no-reload
                # and ensure uvicorn doesn't spawn a StatReload watcher process.
                _log.error(
                    "prometheus_metrics_server_failed",
                    port=PROMETHEUS_PORT,
                    pid=os.getpid(),
                    error=str(_prom_err),
                    hint=(
                        "Port already bound — if uvicorn reloader is running, "
                        "the reloader process grabbed the port first. "
                        "Set reload=False in uvicorn.run() or use PROMETHEUS_PORT "
                        "that doesn't conflict with an existing process."
                    ),
                )

        if start_otel:
            _OTEL.initialize()
            if OTEL_ENABLED and _OTEL._initialized:  # noqa
                _log.info(
                    "otel_initialized",
                    endpoint=OTEL_ENDPOINT,
                    service=OTEL_SERVICE_NAME,
                    sample_rate=OTEL_SAMPLE_RATE,
                )

        if start_mongo:
            _MONGO.start()

        if write_grafana:
            try:
                path = write_grafana_dashboard()
                _log.info("grafana_dashboard_written", path=str(path))
            except Exception as exc:
                _log.warning("grafana_dashboard_write_failed", error=str(exc))

        _bootstrapped = True
        _log.info(
            "observability_ready",
            log_mode=LOG_MODE,
            prometheus=start_prometheus,
            prometheus_port=PROMETHEUS_PORT if start_prometheus else None,
            prometheus_pid=os.getpid() if start_prometheus else None,
            otel=start_otel and OTEL_ENABLED,
            mongo=start_mongo,
            grafana=write_grafana,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC RE-EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Bootstrap
    "bootstrap",
    "configure_logging",
    "get_logger",
    "get_metrics",

    # Event taxonomy
    "EventKind",
    "ObsEvent",
    "emit",

    # Context
    "session_context",
    "sync_session_context",
    "set_request_context",
    "get_session_id",
    "get_request_id",

    # Timing
    "StageTimer",

    # Component emitters
    "PipelineEmitter",
    "STTEmitter",
    "LLMEmitter",
    "TTSEmitter",
    "EvalEmitter",
    "SessionEmitter",
    "MemoryEmitter",
    "SanitizeEmitter",
    "CBEmitter",
    "RateLimitEmitter",
    "BulkheadEmitter",
    "BudgetEmitter",
    "ControllerEmitter",
    "RedisEmitter",
    "TranscriptEmitter",

    # OpenTelemetry — span context managers
    "pipeline_span",
    "stt_span",
    "llm_span",
    "tts_span",
    "eval_span",
    "session_op_span",
    "memory_span",
    "sanitize_span",

    # OpenTelemetry — trace propagation + correlation
    "inject_trace_headers",
    "extract_trace_context",
    "get_trace_id",
    "get_span_id",

    # OpenTelemetry — attribute constants (for callers that add custom attrs)
    "_OtelLayer",

    # Grafana
    "build_grafana_dashboard",
    "write_grafana_dashboard",
]
