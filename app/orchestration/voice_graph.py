"""
Voice pipeline orchestration engine.

Purely orchestration — wires STT → LLM → TTS together, manages request
lifecycle, failure routing, per-stage retries, timeouts, cancellation,
streaming, and observability. No business logic lives here.

Execution modes
───────────────
  api      — await voice_graph.run(state)                      → VoicePipelineResult
  stream   — async for token in voice_graph.stream(state)      → str tokens
  realtime — async for chunk in voice_graph.stream_full(state) → audio bytes

Graph topology (run / api mode)
────────────────────────────────
  START
    └─► stt
          ├── [ok]     → llm
          └── [error]  → stt_error
                ├── [retries remain, no abort] → stt  ← retry loop
                └── [exhausted / abort]        → error_terminal
                                                         └─► END

        llm
          ├── [ok]     → sanitize → tts
          │                          ├── [ok, IS_DEV]  → audio_sink_dev → END
          │                          ├── [ok, !IS_DEV] → END
          │                          └── [error]       → tts_error → error_terminal → END
          └── [error]  → llm_error
                ├── [retries remain, no abort] → llm  ← retry loop
                └── [exhausted / abort]        → error_terminal → END

Key architectural decisions
────────────────────────────
1. Per-instance compiled graph
   _build_graph_for_instance() is called once in VoiceGraph.__init__(). Every
   node function inside it closes over the instance's stt/llm/tts and cfg —
   not module-level globals. This means run() → self._graph.ainvoke() actually
   uses the injected nodes, so test doubles and custom implementations work
   correctly in all three execution paths (run, stream, stream_full).

2. Settings-sourced config
   All timeout and capacity values come from the validated settings singleton,
   not raw os.getenv() calls. Fat-fingered env vars are caught at startup
   (settings raises on import) rather than silently misparsed at first request.
   Per-instance overrides are expressed in VoiceGraphConfig.

3. Per-instance LoadSheddingGuard
   Each VoiceGraph owns its own shedder with its own max_inflight cap. The
   three public singletons therefore cannot steal capacity from each other —
   a surge in standard requests never exhausts realtime slots.

4. Genuinely differentiated singletons
   voice_graph_realtime uses REALTIME tier by default, zero retries (a retry
   under a blown SLA makes things worse), and tight timeouts. voice_graph uses
   balanced defaults with one retry per stage. voice_graph_low_latency sits
   between them. The version string is no longer just a log label — it
   corresponds to real differences in concurrency, timeouts, and routing.

Distributed wiring
──────────────────
The constructor accepts any object implementing STTNodeProtocol,
LLMNodeProtocol, or TTSNodeProtocol — satisfied by both local node classes
and remote HTTP clients. Setting *_SERVICE_URL env vars before module load
switches a stage to a remote service with zero graph-level code changes.
"""

from __future__ import annotations

import asyncio  # noqa
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Required, TypedDict, cast

from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableSerializable
from opentelemetry.trace import StatusCode

from app.common.settings import settings
from app.common.shared import (
    BoundedPipelineQueue,
    LatencyBudget,
    LatencyBudgetExceeded,
    LoadSheddingGuard,
    LoadSheddingRejected,
    QoSTier,
    ServiceHealthState,
    current_request_id,
    get_logger,
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)
from app.monitoring.observability import (
    session_context,
    pipeline_span,
    stt_span,
    llm_span,
    tts_span,
    sanitize_span,  # noqa
    PipelineEmitter,
    STTEmitter,
    LLMEmitter,
    TTSEmitter,
    SanitizeEmitter,
)
from app.nodes.sanitize import sanitize
from app.nodes.STT_service import STTNodeProtocol, get_stt_node
from app.nodes.LLM_service import LLMNodeProtocol, get_llm_node
from app.nodes.TTS_service import TTSNodeProtocol, get_tts_node
from app.audio_essentials.player import play_audio

# ── dev flag ───────────────────────────────────────────────────────────────────
# Controls whether the audio_sink_dev node is compiled into the graph at all.
# Evaluated once at module load so the graph structure is fixed per process —
# there's no cost to checking IS_DEV in routing functions at request time.
IS_DEV: bool = os.getenv("ENV", "").lower() == "development"

# ── graph version ──────────────────────────────────────────────────────────────
# Used as a metric label and embedded in result payloads so deployed graph
# versions can be tracked in dashboards across rolling deployments.
GRAPH_VERSION: str = getattr(settings, "voice_graph_version", "v2")

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── module-level default node singletons ───────────────────────────────────────
# Resolved once via the factory functions. VoiceGraph.__init__() falls back to
# these when no explicit nodes are injected. The compiled graph never references
# these names — it always closes over the per-instance nodes supplied to
# _build_graph_for_instance(). These are purely for the "no injection" default.
_default_stt_node: STTNodeProtocol = get_stt_node()
_default_llm_node: LLMNodeProtocol = get_llm_node()
_default_tts_node: TTSNodeProtocol = get_tts_node()

# ── QoS / execution mode ───────────────────────────────────────────────────────

ExecutionMode = Literal["api", "stream", "realtime"]

_MODE_TO_TIER: dict[str, QoSTier] = {
    "realtime": QoSTier.REALTIME,
    "api": QoSTier.STANDARD,
    "stream": QoSTier.STANDARD,
}

# Apology strings used when a stage cannot produce a valid output and the
# pipeline degrades gracefully rather than surfacing a raw exception to the user.
APOLOGY_STT = "I couldn't catch that. Could you try again?"
APOLOGY_LLM = "I'm having trouble thinking right now. Please try again in a moment."
APOLOGY_TTS = "I have a response but couldn't convert it to audio right now."

# Maps error_stage values to the apology string the terminal node should use.
_STAGE_APOLOGY: dict[str, str] = {
    "stt": APOLOGY_STT,
    "llm": APOLOGY_LLM,
    "tts": APOLOGY_TTS,
    "unknown": APOLOGY_LLM,
}


# ── per-instance configuration ─────────────────────────────────────────────────


@dataclass
class VoiceGraphConfig:
    """
    All tunable knobs for one VoiceGraph instance.

    Defaults pull from the validated settings singleton so malformed env vars
    are caught at startup rather than silently misparsed at first request.
    Override individual fields when constructing specialised graph variants.

    For example, the realtime variant overrides stt/llm/tts_timeout to tight
    values, sets max_inflight low so realtime capacity is reserved, sets
    default_tier=REALTIME so callers don't need to pass mode="realtime", and
    zeroes out retries because a retry under a blown SLA only makes things worse.
    """

    # Per-stage wall-clock timeout guards. The outer LatencyBudget may cancel
    # earlier — these are the last resort if the budget check is missed.
    stt_timeout: float = field(
        default_factory=lambda: getattr(settings, "graph_stt_timeout", 30.0)
    )
    llm_timeout: float = field(
        default_factory=lambda: getattr(settings, "graph_llm_timeout", 45.0)
    )
    tts_timeout: float = field(
        default_factory=lambda: getattr(settings, "graph_tts_timeout", 30.0)
    )

    # Max concurrent requests this instance will admit.
    # Each VoiceGraph instance owns its own LoadSheddingGuard initialised with
    # this value, so variants cannot starve each other's capacity pools.
    max_inflight: int = field(
        default_factory=lambda: getattr(settings, "graph_max_inflight", 20)
    )

    # Default QoS tier applied when the caller does not specify a mode.
    # voice_graph_realtime sets this to REALTIME so every request through that
    # instance is treated as high-priority even without an explicit mode field.
    default_tier: QoSTier = QoSTier.STANDARD

    # Number of retry attempts allowed per stage after the first failure.
    # 0 means fail-fast: the first failure routes directly to error_terminal.
    # 1 means the graph will attempt the stage a second time before giving up.
    max_stt_retries: int = 1
    max_llm_retries: int = 1

    # Transcript and response size limits
    min_prompt_chars: int = field(
        default_factory=lambda: getattr(settings, "graph_min_prompt_chars", 25)
    )
    max_transcript_chars: int = field(
        default_factory=lambda: getattr(settings, "graph_max_transcript_chars", 2000)
    )
    max_llm_response_chars: int = field(
        default_factory=lambda: getattr(settings, "graph_max_llm_response_chars", 4096)
    )
    max_tts_chars: int = field(
        default_factory=lambda: getattr(settings, "graph_max_tts_chars", 4000)
    )

    # Bounded queue depths for the concurrent stream_full() pipeline.
    # Larger values absorb token bursts but increase memory per active stream.
    # These are per-call (not shared across requests).
    stt_llm_queue_depth: int = field(
        default_factory=lambda: getattr(settings, "graph_stt_llm_queue_depth", 8)
    )
    llm_tts_queue_depth: int = field(
        default_factory=lambda: getattr(settings, "graph_llm_tts_queue_depth", 16)
    )


# ── pipeline stage enum ────────────────────────────────────────────────────────


class PipelineStage(str, Enum):
    PENDING = "pending"
    STT = "stt"
    LLM = "llm"
    SANITIZE = "sanitize"
    TTS = "tts"
    DONE = "done"
    FAILED = "failed"


# ── shared state schema ────────────────────────────────────────────────────────


class VoiceState(TypedDict, total=False):
    # ── required input ────────────────────────────────────────────────────────
    audio_path: Required[str]

    # ── caller-supplied context ───────────────────────────────────────────────
    request_id: str
    session_id: str
    client_ip: str
    history: list[dict]
    mode: str  # ExecutionMode — normalised by _prepare_state
    language: str  # ISO-639-1 hint forwarded to STT
    stt_prompt: str  # whisper-style context hint for STT
    tts_voice: str
    tts_speed: float
    qos_tier: str  # QoSTier.name, injected by _prepare_state

    # ── STT outputs ───────────────────────────────────────────────────────────
    user_input: str
    stt_result: dict
    transcript_truncated: bool

    # ── LLM outputs ───────────────────────────────────────────────────────────
    llm_response: str
    llm_tokens: dict
    llm_model_used: str
    llm_cached: bool
    response_truncated: bool

    # ── sanitize output ───────────────────────────────────────────────────────
    cleaned_response: str

    # ── TTS outputs ───────────────────────────────────────────────────────────
    audio_output: str
    audio_local_path: str
    audio_s3_uri: str

    # ── retry bookkeeping ─────────────────────────────────────────────────────
    # These are internal graph state fields — they appear in metadata but are
    # never part of the user-facing result contract.

    # Counts retry attempts made at each stage. Incremented by the error handler
    # node; the routing function after the error handler checks this counter.
    stt_retries: int
    llm_retries: int

    # When non-empty, all retry routers skip the retry loop and go straight to
    # error_terminal. Set by nodes for non-recoverable faults like budget overrun,
    # path traversal, or validation errors where retrying cannot help.
    abort_reason: str

    # Guards against double-writing the session turn if node_llm is retried
    # after a failure that occurred after generate() but before turn commit.
    session_turn_appended: bool

    # ── pipeline metadata ─────────────────────────────────────────────────────
    stage: str
    error: str
    error_stage: str
    degraded: bool
    stage_latencies: dict
    pipeline_latency_s: float
    graph_version: str


class VoicePipelineResult(TypedDict):
    """
    Stable output contract. All fields are always present — missing values
    default to empty string, zero, or False so callers never need to guard
    against KeyError.
    """

    request_id: str
    transcript: str
    llm_response: str
    cleaned_response: str
    audio_output: str
    audio_s3_uri: str
    error: str
    error_stage: str
    degraded: bool
    stage_latencies: dict
    pipeline_latency_s: float
    graph_version: str
    metadata: dict


# ── Prometheus metrics ─────────────────────────────────────────────────────────

_pipeline_total = make_counter(
    "voice_pipeline_total",
    "Total pipeline executions",
    ["version", "status", "tier"],
)
_stage_errors = make_counter(
    "voice_pipeline_stage_errors_total",
    "Errors per stage",
    ["stage"],
)
_stage_retries = make_counter(
    "voice_pipeline_stage_retries_total",
    "Retry attempts per stage (excludes first attempt)",
    ["stage"],
)
_stage_latency = make_histogram(
    "voice_pipeline_stage_latency_seconds",
    "Per-stage wall-clock latency",
    ["stage"],
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30, 60),
)
_pipeline_latency = make_histogram(
    "voice_pipeline_latency_seconds",
    "Total pipeline wall-clock latency",
    buckets=(1, 2, 3, 5, 8, 12, 20, 40, 90),
)
_cancellations = make_counter(
    "voice_pipeline_cancellations_total",
    "Cancelled pipeline runs",
    ["stage"],
)
_active_pipelines = make_gauge(
    "voice_pipeline_active",
    "Pipelines currently in flight (run() only)",
)
_degraded_total = make_counter(
    "voice_pipeline_degraded_total",
    "Pipelines that completed via a fallback or apology path",
)
_load_shed_total = make_counter(
    "voice_pipeline_load_shed_total",
    "Requests rejected by the instance load shedder",
    ["tier"],
)
_budget_breached = make_counter(
    "voice_pipeline_budget_breached_total",
    "Stage executions aborted because the SLA budget was already blown",
    ["stage"],
)
_stream_full_active = make_gauge(
    "voice_pipeline_stream_full_active",
    "Concurrent stream_full() sessions currently producing audio",
)

# ── helpers ────────────────────────────────────────────────────────────────────

_PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\")


def _validate_audio_path(path: str) -> None:
    """Reject obviously unsafe or malformed audio paths before sending to STT."""
    if not path or not path.strip():
        raise ValueError("audio_path must not be empty.")
    if _PATH_TRAVERSAL.search(path):
        raise ValueError(f"Potentially unsafe audio path rejected: {path!r}")
    if (
        not path.startswith("s3://")
        and not path.startswith("/")
        and path[1:3] != ":\\"
        and path.startswith("..")
    ):
        raise ValueError(f"Relative path traversal rejected: {path!r}")


async def _with_timeout(coro: Any, timeout: float, stage: str) -> Any:
    """Wrap coro in a hard wall-clock limit and surface a labelled TimeoutError."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"{stage} exceeded {timeout:.0f}s timeout.")


def _state_update(state: VoiceState, updates: dict) -> VoiceState:
    """Return a new state dict merging updates over the current state."""
    return cast(VoiceState, {"audio_path": state["audio_path"], **state, **updates})


def _record_stage_latency(state: VoiceState, stage: str, latency: float) -> VoiceState:
    """Append a measured stage latency into the running stage_latencies dict."""
    latencies = dict(state.get("stage_latencies") or {})
    latencies[stage] = round(latency, 3)
    return _state_update(state, {"stage_latencies": latencies})


def _has_stage_error(state: VoiceState, stage: PipelineStage) -> bool:
    """
    True when the given stage has written an error into the state.
    Used by all routing functions to decide which branch to take.
    Checks both error_stage (which stage failed) and error (non-empty message)
    so an accidental empty-string error never triggers the error path.
    """
    return state.get("error_stage") == stage.value and bool(
        state.get("error", "").strip()
    )


# ── result builder ─────────────────────────────────────────────────────────────


def _build_result(state: VoiceState, pipeline_latency: float) -> VoicePipelineResult:
    """Project VoiceState → VoicePipelineResult with safe defaults for every field."""
    return VoicePipelineResult(
        request_id=state.get("request_id", ""),
        transcript=state.get("user_input", ""),
        llm_response=state.get("llm_response", ""),
        cleaned_response=state.get("cleaned_response", ""),
        audio_output=state.get("audio_output", ""),
        audio_s3_uri=state.get("audio_s3_uri", ""),
        error=state.get("error", ""),
        error_stage=state.get("error_stage", ""),
        degraded=state.get("degraded", False),
        stage_latencies=state.get("stage_latencies", {}),
        pipeline_latency_s=round(pipeline_latency, 3),
        graph_version=state.get("graph_version", GRAPH_VERSION),
        metadata={
            "llm_model_used": state.get("llm_model_used", ""),
            "llm_cached": state.get("llm_cached", False),
            "llm_tokens": state.get("llm_tokens", {}),
            "transcript_truncated": state.get("transcript_truncated", False),
            "response_truncated": state.get("response_truncated", False),
            "audio_local_path": state.get("audio_local_path", ""),
            "mode": state.get("mode", "api"),
            "qos_tier": state.get("qos_tier", QoSTier.STANDARD.name),
            "stt_retries": state.get("stt_retries", 0),
            "llm_retries": state.get("llm_retries", 0),
        },
    )


# ── per-instance graph builder ─────────────────────────────────────────────────


def _build_graph_for_instance(
    stt: STTNodeProtocol,
    llm: LLMNodeProtocol,
    tts: TTSNodeProtocol,
    cfg: VoiceGraphConfig,
    is_dev: bool,
) -> RunnableSerializable:
    """
    Compile a fresh LangGraph StateGraph whose every node function closes over
    the supplied node instances and VoiceGraphConfig — not module-level globals.

    This is called once per VoiceGraph.__init__(). The resulting compiled graph
    is stored as self._graph, so self._graph.ainvoke() in run() uses the injected
    stt/llm/tts rather than the module-level singletons. Test doubles, remote
    clients, and custom implementations all work correctly in run() as a result.

    Node topology
    ─────────────
    stt → [route_after_stt]
      "stt_error" → stt_error → [route_after_stt_error]
                     "stt"          → stt  (retry)
                     "error_terminal" → error_terminal → END
      "llm"       → llm → [route_after_llm]
                     "llm_error" → llm_error → [route_after_llm_error]
                                    "llm"            → llm  (retry)
                                    "error_terminal" → error_terminal → END
                     "sanitize"  → sanitize → tts → [route_after_tts]
                                                "tts_error"     → tts_error → error_terminal → END
                                                "audio_sink_dev" → audio_sink_dev → END  (IS_DEV)
                                                END                                        (!IS_DEV)
    """

    # ═══════════════════════════════════════════════════════════════════════════
    # NODE IMPLEMENTATIONS
    # Each node is an async function that receives the full VoiceState, does its
    # work, and returns an updated VoiceState. Errors are written into the state
    # and surfaced to the routing functions — nodes never raise to the graph.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── node: STT ─────────────────────────────────────────────────────────────
    async def node_stt(state: VoiceState) -> VoiceState:
        """
        Transcribe the audio file to text using the instance's STT implementation.

        On success: user_input carries the cleaned transcript; error/error_stage
        are cleared so a previous failed attempt leaves no residue.

        On LatencyBudgetExceeded: sets abort_reason so the downstream router
        skips the retry loop and routes directly to error_terminal. Retrying
        a stage after the SLA budget is already blown can only make things worse.

        On any other exception: sets error/error_stage so route_after_stt routes
        to stt_error, which will increment the retry counter and decide whether
        to retry or give up.
        """
        rid = state.get("request_id", current_request_id())
        t0 = time.monotonic()

        STTEmitter.start(
            session_id=state.get("session_id", ""),
            request_id=rid,
            audio_path=state.get("audio_path", ""),
        )

        async with stt_span(
            session_id=state.get("session_id", ""),
            request_id=rid,
            audio_path=state.get("audio_path", ""),
        ) as span:
            try:
                _validate_audio_path(state["audio_path"])

                result = await _with_timeout(
                    stt.transcribe(
                        audio_path=state["audio_path"],
                        language=state.get("language"),
                        prompt=state.get("stt_prompt"),
                        request_id=rid,
                    ),
                    timeout=cfg.stt_timeout,
                    stage="STT",
                )

                raw_transcript = result["text"]
                san = sanitize(
                    raw_transcript,
                    max_chars=cfg.max_transcript_chars,
                    request_id=rid,
                )
                cleaned = san.text
                truncated = san.truncated

                latency = time.monotonic() - t0
                _stage_latency.labels(stage="stt").observe(latency)

                log.info(
                    "graph_stt_ok",
                    request_id=rid,
                    transcript_len=len(cleaned),
                    truncated=truncated,
                    language=result.get("language"),
                    latency_s=round(latency, 3),
                )
                span.set_attribute("transcript_len", len(cleaned))
                span.set_attribute("latency_s", round(latency, 3))

                STTEmitter.ok(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    latency_ms=latency * 1000,
                    transcript=cleaned,
                    language=result.get("language", ""),
                    lang_confidence=result.get("lang_confidence", 0.0),
                    avg_logprob=result.get("avg_logprob", 0.0),
                    no_speech_prob=result.get("no_speech_prob", 0.0),
                    truncated=truncated,
                )

                # Clear any error residue left from a previous failed attempt
                # before this retry succeeded.
                return _state_update(
                    _record_stage_latency(state, "stt", latency),
                    {
                        "user_input": cleaned,
                        "stt_result": dict(result),
                        "transcript_truncated": truncated,
                        "stage": PipelineStage.STT.value,
                        "error": "",
                        "error_stage": "",
                        "abort_reason": "",
                    },
                )

            except asyncio.CancelledError:
                _cancellations.labels(stage="stt").inc()
                log.warning("graph_stt_cancelled", request_id=rid)
                raise

            except LatencyBudgetExceeded as exc:
                # Abort immediately — retry cannot recover from a blown SLA budget.
                _budget_breached.labels(stage="stt").inc()
                _stage_errors.labels(stage="stt").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.warning(
                    "graph_stt_budget_exceeded",
                    request_id=rid,
                    latency_s=round(latency, 3),
                )
                STTEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type="LatencyBudgetExceeded",
                )
                return _state_update(
                    _record_stage_latency(state, "stt", latency),
                    {
                        "user_input": "",
                        "stt_result": {},
                        "stage": PipelineStage.STT.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.STT.value,
                        "abort_reason": "budget_exceeded",
                    },
                )

            except Exception as exc:
                # Transient failure — the retry router will decide whether to retry.
                _stage_errors.labels(stage="stt").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "graph_stt_failed",
                    request_id=rid,
                    error=str(exc),
                    latency_s=round(latency, 3),
                )
                STTEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                return _state_update(
                    _record_stage_latency(state, "stt", latency),
                    {
                        "user_input": "",
                        "stt_result": {},
                        "stage": PipelineStage.STT.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.STT.value,
                    },
                )

    # ── node: STT error handler ────────────────────────────────────────────────
    async def node_stt_error(state: VoiceState) -> VoiceState:
        """
        Increments the STT retry counter and emits a metric, then returns.

        This node does not make a routing decision — that is left to
        route_after_stt_error. Its only job is to prepare the state for that
        router by incrementing stt_retries so the router can compare against
        cfg.max_stt_retries.

        If abort_reason is already set (e.g. budget_exceeded), route_after_stt_error
        will still pick error_terminal regardless of the counter value, so
        incrementing is harmless in that case.
        """
        rid = state.get("request_id", current_request_id())
        retries = state.get("stt_retries", 0) + 1
        _stage_retries.labels(stage="stt").inc()

        log.warning(
            "graph_stt_error_handler",
            request_id=rid,
            attempt=retries,
            max_retries=cfg.max_stt_retries,
            error=state.get("error", ""),
            abort_reason=state.get("abort_reason", ""),
        )

        PipelineEmitter.retry(
            session_id=state.get("session_id", ""),
            request_id=rid,
            stage="stt",
            attempt=retries,
            error=state.get("error", ""),
        )

        return _state_update(state, {"stt_retries": retries})

    # ── node: LLM ─────────────────────────────────────────────────────────────
    async def node_llm(state: VoiceState) -> VoiceState:
        """
        Generate an LLM response for the transcribed prompt.

        Session context injection:
            When session_id is present, the conversation history from session_store
            is serialised as a "User: … / Assistant: …" block prepended to the
            current turn. generate() wraps the whole thing in a single HumanMessage
            and internally prepends SYSTEM_PROMPT as a SystemMessage — we must
            never inject SYSTEM_PROMPT here.

        Turn persistence:
            After a successful generate(), the new turn is appended to session_store
            and a fire-and-forget evaluation is scheduled. session_turn_appended
            prevents double-writes if node_llm is retried after a failure that
            happened after generate() but before some later step.

        Empty-transcript fallback:
            If STT produced no usable transcript (e.g. silence or complete failure),
            a polite apology prompt is used so the LLM can produce a human-readable
            "please try again" response rather than generating from an empty string.
        """
        rid = state.get("request_id", current_request_id())
        t0 = time.monotonic()

        # Lazy imports avoid circular-import issues at module load time.
        # These modules depend on settings/db/etc. that may not be ready
        # when voice_graph is first imported.
        from app.user_tracking.session_service.session_store import session_store
        from app.eval.evaluation_engine import evaluation_engine

        from app.user_tracking.session_service.conversation_memory import (
            conversation_memory,
        )
        from app.user_tracking.transcript.transcription import transcript_writer

        LLMEmitter.start(
            session_id=state.get("session_id", ""),
            request_id=rid,
            model=getattr(settings, "llm_model", ""),
            streaming=False,
            history_turns=0,
        )

        async with llm_span(
            session_id=state.get("session_id", ""),
            request_id=rid,
            model=getattr(settings, "llm_model", ""),
            streaming=False,
        ) as span:
            try:
                transcript = state.get("user_input", "").strip()
                if not transcript:
                    # STT failed or returned silence. The LLM will produce an apology
                    # that TTS can synthesise into a graceful spoken error response.
                    log.info("graph_llm_using_apology_prompt", request_id=rid)
                    transcript = (
                        "The user's audio could not be transcribed. "
                        "Apologise briefly and ask them to try again."
                    )

                session_id = state.get("session_id")
                client_ip = state.get("client_ip", "")

                # Resolve conversation memory — injects rolling history + interview context.
                # Falls back gracefully if session_store is unreachable.
                memory_ctx = await conversation_memory.resolve(
                    session_id=session_id,
                    user_text=transcript,
                )

                # Build the prompt string from the resolved history.
                # generate() wraps this in [SystemMessage, HumanMessage] internally —
                # the interview context prefix is already embedded by conversation_memory.
                if session_id and memory_ctx.turn_index > 0:
                    history_lines: list[str] = []
                    from app.user_tracking.session_service.session_store import (
                        SessionNotFound,
                    )

                    try:
                        session = await session_store.load(session_id, client_ip)
                        for turn in session.turns:
                            history_lines.append(f"User: {turn.user}")
                            history_lines.append(f"Assistant: {turn.assistant}")
                    except SessionNotFound:
                        pass
                    history_block = "\n".join(history_lines)
                    ctx_prefix = ""
                    if (
                        memory_ctx.interview_state.topics_covered
                        or memory_ctx.interview_state.current_topic
                    ):
                        from app.user_tracking.session_service.conversation_memory import (
                            _build_context_prefix, # noqa
                        )  # noqa

                        ctx_prefix = _build_context_prefix(memory_ctx.interview_state)
                    prompt_parts = []
                    if ctx_prefix:
                        prompt_parts.append(ctx_prefix)
                    if history_block:
                        prompt_parts.append(f"Conversation so far:\n{history_block}")
                    prompt_parts.append(f"User:\n{transcript}")
                    prompt = "\n\n".join(prompt_parts).strip()
                    log.info(
                        "graph_llm_history_injected",
                        request_id=rid,
                        session_id=session_id,
                        history_turns=memory_ctx.turn_index,
                        topic=memory_ctx.interview_state.current_topic or "—",
                    )
                    span.set_attribute("history_turns", memory_ctx.turn_index)
                else:
                    prompt = transcript

                result = await _with_timeout(
                    llm.generate(prompt, request_id=rid),
                    timeout=cfg.llm_timeout,
                    stage="LLM",
                )
                raw_response = result["response"]

                # Persist the turn only once: skip if session_turn_appended is True,
                # which happens when node_llm is retried after generate() succeeded
                # but something else further down failed.
                if session_id and not state.get("session_turn_appended"):
                    await session_store.append_turn(
                        session_id, transcript, raw_response
                    )

                    await conversation_memory.commit(
                        session_id, transcript, raw_response
                    )
                    await transcript_writer.write_turn(
                        session_id=session_id,
                        user_text=transcript,
                        assistant_text=raw_response,
                        request_id=rid,
                    )

                    # Reload session so we get the accurate turn index and can derive
                    # the question (the previous assistant turn) for evaluation.
                    _reloaded = await session_store.load(session_id, client_ip)
                    _question = (
                        _reloaded.turns[-2].assistant
                        if len(_reloaded.turns) >= 2
                        else ""
                    )
                    evaluation_engine.schedule_turn(
                        session_id=session_id,
                        question=_question,
                        candidate_ans=transcript,
                        turn_index=len(_reloaded.turns) - 1,
                        request_id=rid,
                    )

                # Cap response length before handing to sanitize node.
                san = sanitize(
                    raw_response,
                    max_chars=cfg.max_llm_response_chars,
                    request_id=rid,
                )
                capped = san.text
                truncated = san.truncated

                latency = time.monotonic() - t0
                _stage_latency.labels(stage="llm").observe(latency)

                log.info(
                    "graph_llm_ok",
                    request_id=rid,
                    model=result.get("model_used"),
                    cached=result.get("cached"),
                    response_len=len(capped),
                    truncated=truncated,
                    latency_s=round(latency, 3),
                )
                span.set_attribute("model", result.get("model_used", ""))
                span.set_attribute("cached", result.get("cached", False))
                span.set_attribute("latency_s", round(latency, 3))

                LLMEmitter.ok(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    latency_ms=latency * 1000,
                    model_used=result.get("model_used", ""),
                    prompt_tokens=result.get("prompt_tokens", 0),
                    completion_tokens=result.get("completion_tokens", 0),
                    streaming=False,
                    history_turns=memory_ctx.turn_index,
                    response_chars=len(capped),
                    cache_hit=result.get("cached", False),
                    response_truncated=truncated,
                )

                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "llm_response": capped,
                        "llm_tokens": {
                            "prompt": result.get("prompt_tokens", 0),
                            "completion": result.get("completion_tokens", 0),
                        },
                        "llm_model_used": result.get("model_used", ""),
                        "llm_cached": result.get("cached", False),
                        "response_truncated": truncated,
                        "session_turn_appended": True,
                        "stage": PipelineStage.LLM.value,
                        # Preserve any STT-stage error info in the result without
                        # letting it look like the LLM stage failed.
                        "error": state.get("error", ""),
                        "error_stage": state.get("error_stage", ""),
                        "abort_reason": "",
                    },
                )

            except asyncio.CancelledError:
                _cancellations.labels(stage="llm").inc()
                log.warning("graph_llm_cancelled", request_id=rid)
                raise

            except LatencyBudgetExceeded as exc:
                _budget_breached.labels(stage="llm").inc()
                _stage_errors.labels(stage="llm").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.warning(
                    "graph_llm_budget_exceeded",
                    request_id=rid,
                    latency_s=round(latency, 3),
                )
                LLMEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type="LatencyBudgetExceeded",
                    model=getattr(settings, "llm_model", ""),
                    streaming=False,
                )
                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "stage": PipelineStage.LLM.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.LLM.value,
                        "abort_reason": "budget_exceeded",
                    },
                )

            except Exception as exc:
                _stage_errors.labels(stage="llm").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "graph_llm_failed",
                    request_id=rid,
                    error=str(exc),
                    latency_s=round(latency, 3),
                )
                LLMEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    model=getattr(settings, "llm_model", ""),
                    streaming=False,
                )
                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "stage": PipelineStage.LLM.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.LLM.value,
                    },
                )

    # ── node: LLM error handler ────────────────────────────────────────────────
    async def node_llm_error(state: VoiceState) -> VoiceState:
        """
        Mirror of node_stt_error for the LLM stage. Increments llm_retries so
        route_after_llm_error can compare against cfg.max_llm_retries and decide
        whether to route back to node_llm for another attempt or to error_terminal.
        """
        rid = state.get("request_id", current_request_id())
        retries = state.get("llm_retries", 0) + 1
        _stage_retries.labels(stage="llm").inc()

        log.warning(
            "graph_llm_error_handler",
            request_id=rid,
            attempt=retries,
            max_retries=cfg.max_llm_retries,
            error=state.get("error", ""),
            abort_reason=state.get("abort_reason", ""),
        )

        PipelineEmitter.retry(
            session_id=state.get("session_id", ""),
            request_id=rid,
            stage="llm",
            attempt=retries,
            error=state.get("error", ""),
        )

        return _state_update(state, {"llm_retries": retries})

    # ── node: sanitize ─────────────────────────────────────────────────────────
    async def node_sanitize(state: VoiceState) -> VoiceState:
        """
        Clean the LLM response before it reaches TTS synthesis.

        Applies TTS-safe character limits, strips markdown and problematic Unicode,
        and collapses whitespace. Uses the canonical sanitize() module as the single
        source of truth for text cleaning rules — no duplicate cleaning logic here.

        This node never hard-fails: the worst case is the apology string being
        passed to TTS, which is always clean. No conditional edge branching needed.
        """
        rid = state.get("request_id", current_request_id())

        raw = state.get("llm_response") or ""
        if not raw.strip():
            # LLM returned an empty response — shouldn't happen after a successful
            # node_llm, but guard here so TTS never receives silence by accident.
            raw = APOLOGY_LLM

        san = sanitize(raw, max_chars=cfg.max_tts_chars, request_id=rid)
        cleaned = san.text if san else APOLOGY_LLM

        log.info(
            "graph_sanitize_ok",
            request_id=rid,
            input_len=san.original_len,
            output_len=san.sanitized_len,
            truncated=san.truncated,
            warnings=list(san.warnings),
        )

        SanitizeEmitter.ok(
            session_id=state.get("session_id", ""),
            request_id=rid,
            original_chars=san.original_len,
            sanitized_chars=san.sanitized_len,
            truncated=san.truncated,
            warnings=list(san.warnings),
        )

        return _state_update(
            state,
            {
                "cleaned_response": cleaned,
                "stage": PipelineStage.SANITIZE.value,
            },
        )

    # ── node: TTS ─────────────────────────────────────────────────────────────
    async def node_tts(state: VoiceState) -> VoiceState:
        """
        Synthesise the cleaned response text into audio.

        On success: audio_output is the best available URI (S3 if present,
        local path otherwise). On failure: error/error_stage are written and
        route_after_tts sends the graph to node_tts_error → error_terminal.
        TTS failures are not retried because by this point the latency budget
        is typically exhausted and there is no audio fallback to offer.
        """
        rid = state.get("request_id", current_request_id())
        t0 = time.monotonic()

        _tts_voice = state.get("tts_voice") or ""
        _tts_input_chars = len((state.get("cleaned_response") or "").strip())

        TTSEmitter.start(
            session_id=state.get("session_id", ""),
            request_id=rid,
            voice=_tts_voice,
            input_chars=_tts_input_chars,
        )

        async with tts_span(
            session_id=state.get("session_id", ""),
            request_id=rid,
            voice=_tts_voice,
            input_chars=_tts_input_chars,
        ) as span:
            try:
                text = state.get("cleaned_response", "").strip() or APOLOGY_TTS

                local_path, s3_uri = await _with_timeout(
                    tts.synthesize(
                        text=text,
                        voice=state.get("tts_voice"),  # type: ignore[arg-type]
                        speed=float(state.get("tts_speed", 1.0)),
                        request_id=rid,
                    ),
                    timeout=cfg.tts_timeout,
                    stage="TTS",
                )

                latency = time.monotonic() - t0
                _stage_latency.labels(stage="tts").observe(latency)

                log.info(
                    "graph_tts_ok",
                    request_id=rid,
                    local_path=local_path,
                    s3_uri=s3_uri or "n/a",
                    latency_s=round(latency, 3),
                )
                span.set_attribute("latency_s", round(latency, 3))
                span.set_attribute("s3_uri", s3_uri or "")

                TTSEmitter.ok(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    latency_ms=latency * 1000,
                    voice=_tts_voice,
                    input_chars=len(text),
                    audio_output=s3_uri if s3_uri else local_path,
                    s3_uri=s3_uri or "",
                )

                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "audio_output": s3_uri if s3_uri else local_path,
                        "audio_local_path": local_path,
                        "audio_s3_uri": s3_uri or "",
                        "stage": PipelineStage.TTS.value,
                    },
                )

            except asyncio.CancelledError:
                _cancellations.labels(stage="tts").inc()
                log.warning("graph_tts_cancelled", request_id=rid)
                raise

            except LatencyBudgetExceeded as exc:
                _budget_breached.labels(stage="tts").inc()
                _stage_errors.labels(stage="tts").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.warning(
                    "graph_tts_budget_exceeded",
                    request_id=rid,
                    latency_s=round(latency, 3),
                )
                TTSEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type="LatencyBudgetExceeded",
                    voice=_tts_voice,
                )
                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "audio_output": "",
                        "audio_local_path": "",
                        "audio_s3_uri": "",
                        "stage": PipelineStage.TTS.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.TTS.value,
                        "abort_reason": "budget_exceeded",
                        "degraded": True,
                    },
                )

            except Exception as exc:
                _stage_errors.labels(stage="tts").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "graph_tts_failed",
                    request_id=rid,
                    error=str(exc),
                    latency_s=round(latency, 3),
                )
                TTSEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    voice=_tts_voice,
                )
                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "audio_output": "",
                        "audio_local_path": "",
                        "audio_s3_uri": "",
                        "stage": PipelineStage.TTS.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.TTS.value,
                        "degraded": True,
                    },
                )

    # ── node: TTS error handler ────────────────────────────────────────────────
    async def node_tts_error(state: VoiceState) -> VoiceState:
        """
        TTS failures are not retried — this node exists to emit a log entry
        and increment the retries counter (which here counts terminal failures,
        not actual retries) before the graph unconditionally routes to
        node_error_terminal. State already has error/error_stage set by node_tts.
        """
        rid = state.get("request_id", current_request_id())
        _stage_retries.labels(stage="tts").inc()

        log.error(
            "graph_tts_terminal_failure",
            request_id=rid,
            error=state.get("error", ""),
            abort_reason=state.get("abort_reason", ""),
        )
        return state

    # ── node: error terminal ───────────────────────────────────────────────────
    async def node_error_terminal(state: VoiceState) -> VoiceState:
        """
        Final stop for all unrecoverable failures regardless of which stage
        caused them. Ensures the result always carries a human-readable apology
        in llm_response and cleaned_response so callers never see empty strings.

        Sets stage=FAILED and degraded=True unconditionally so _build_result
        and all downstream consumers know the pipeline did not complete normally.

        Reachable from:
          - route_after_stt_error  when stt_retries are exhausted or abort_reason is set
          - route_after_llm_error  when llm_retries are exhausted or abort_reason is set
          - node_tts_error         always (TTS failures are never retried)
        """
        rid = state.get("request_id", current_request_id())
        error_stage = state.get("error_stage", "unknown")

        log.error(
            "graph_pipeline_terminal_failure",
            request_id=rid,
            error_stage=error_stage,
            error=state.get("error", ""),
            stt_retries=state.get("stt_retries", 0),
            llm_retries=state.get("llm_retries", 0),
            abort_reason=state.get("abort_reason", ""),
        )

        apology = _STAGE_APOLOGY.get(error_stage, APOLOGY_LLM)

        return _state_update(
            state,
            {
                "stage": PipelineStage.FAILED.value,
                "degraded": True,
                # Prefer any apology already written by a partial-success path;
                # only set the stage-appropriate apology if the field is empty.
                "llm_response": state.get("llm_response") or apology,
                "cleaned_response": state.get("cleaned_response") or apology,
            },
        )

    # ── node: dev audio playback ───────────────────────────────────────────────
    async def node_playback_dev(state: VoiceState) -> VoiceState:
        """
        Local audio playback for development only.
        Never compiled into the graph in non-dev environments — the node does
        not exist in the graph object at all, so there's zero overhead.
        Failures are swallowed so a missing file never crashes a dev run.
        """
        path = state.get("audio_local_path")
        if not path:
            return state
        try:
            play_audio(path)
            log.info(
                "dev_playback_started",
                request_id=state.get("request_id"),
                path=path,
            )
        except Exception as exc:
            log.warning(
                "dev_playback_failed",
                request_id=state.get("request_id"),
                error=str(exc),
            )
        return state

    # ═══════════════════════════════════════════════════════════════════════════
    # ROUTING FUNCTIONS
    # Each function receives the full state and returns a string that maps to
    # a node name in the path_map passed to add_conditional_edges(). Routers
    # are pure (no side effects) so they're cheap and easy to test in isolation.
    # ═══════════════════════════════════════════════════════════════════════════

    def route_after_stt(state: VoiceState) -> str:
        """Send failing STT runs to the error handler; successful runs to the LLM."""
        if _has_stage_error(state, PipelineStage.STT):
            return "stt_error"
        return "llm"

    def route_after_stt_error(state: VoiceState) -> str:
        """
        Decide whether to retry STT or abandon.

        abort_reason being set (budget_exceeded, validation, etc.) always wins
        and skips the retry loop. Otherwise, compare the incremented counter
        against cfg.max_stt_retries: if retries still available, arc back to
        node_stt; if exhausted, go to error_terminal.

        The comparison is (retries <= max_retries) because node_stt_error
        has already incremented before this router runs:
          1st failure → stt_retries becomes 1, max_stt_retries=1 → 1 <= 1 → retry
          2nd failure → stt_retries becomes 2, max_stt_retries=1 → 2 > 1  → terminal
        """
        if state.get("abort_reason"):
            return "error_terminal"
        retries = state.get("stt_retries", 0)
        if retries <= cfg.max_stt_retries:
            log.info(
                "graph_stt_retrying",
                request_id=state.get("request_id"),
                attempt=retries,
                max=cfg.max_stt_retries,
            )
            return "stt"
        return "error_terminal"

    def route_after_llm(state: VoiceState) -> str:
        """Send failing LLM runs to the error handler; successful runs to sanitize."""
        if _has_stage_error(state, PipelineStage.LLM):
            return "llm_error"
        return "sanitize"

    def route_after_llm_error(state: VoiceState) -> str:
        """
        Mirror of route_after_stt_error for the LLM stage.
        Same counter logic applies: node_llm_error has already incremented.
        """
        if state.get("abort_reason"):
            return "error_terminal"
        retries = state.get("llm_retries", 0)
        if retries <= cfg.max_llm_retries:
            log.info(
                "graph_llm_retrying",
                request_id=state.get("request_id"),
                attempt=retries,
                max=cfg.max_llm_retries,
            )
            return "llm"
        return "error_terminal"

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH ASSEMBLY
    # All nodes are registered and all edges — both unconditional and
    # conditional — are wired here. The resulting compiled graph is a closed,
    # immutable object that references only the local function objects above.
    # ═══════════════════════════════════════════════════════════════════════════

    builder = StateGraph(VoiceState)  # type: ignore[arg-type]

    # ── register nodes ─────────────────────────────────────────────────────────
    builder.add_node("stt", node_stt)  # type: ignore[arg-type]
    builder.add_node("stt_error", node_stt_error)  # type: ignore[arg-type]
    builder.add_node("llm", node_llm)  # type: ignore[arg-type]
    builder.add_node("llm_error", node_llm_error)  # type: ignore[arg-type]
    builder.add_node("sanitize", node_sanitize)  # type: ignore[arg-type]
    builder.add_node("tts", node_tts)  # type: ignore[arg-type]
    builder.add_node("tts_error", node_tts_error)  # type: ignore[arg-type]
    builder.add_node("error_terminal", node_error_terminal)  # type: ignore[arg-type]

    if is_dev:
        # Only wire the playback node into dev graphs. The node object doesn't
        # even exist in the compiled graph for non-dev deployments.
        builder.add_node("audio_sink_dev", node_playback_dev)  # type: ignore[arg-type]

    builder.set_entry_point("stt")

    # ── STT → conditional branch ───────────────────────────────────────────────
    builder.add_conditional_edges(
        "stt",
        route_after_stt,
        {"stt_error": "stt_error", "llm": "llm"},
    )

    # ── STT error → retry loop or terminal ────────────────────────────────────
    # This is one of two graph cycles: stt → stt_error → stt → stt_error → …
    # LangGraph handles cycles natively; the cycle terminates when the router
    # returns "error_terminal" (retries exhausted) or "llm" (success on retry).
    builder.add_conditional_edges(
        "stt_error",
        route_after_stt_error,
        {"stt": "stt", "error_terminal": "error_terminal"},
    )

    # ── LLM → conditional branch ───────────────────────────────────────────────
    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {"llm_error": "llm_error", "sanitize": "sanitize"},
    )

    # ── LLM error → retry loop or terminal ────────────────────────────────────
    # Second graph cycle: llm → llm_error → llm → llm_error → …
    builder.add_conditional_edges(
        "llm_error",
        route_after_llm_error,
        {"llm": "llm", "error_terminal": "error_terminal"},
    )

    # ── sanitize → tts (unconditional) ────────────────────────────────────────
    # sanitize() never hard-fails — worst case it returns the apology string —
    # so no conditional branching is needed after it.
    builder.add_edge("sanitize", "tts")

    # ── TTS → conditional branch ───────────────────────────────────────────────
    # The IS_DEV check here is a build-time decision, not a per-request one.
    # In dev: route to audio_sink_dev on success; in all other environments: END.
    # tts_error always routes to error_terminal (no retry for TTS).
    if is_dev:

        def route_after_tts(state: VoiceState) -> str:
            if _has_stage_error(state, PipelineStage.TTS):
                return "tts_error"
            return "audio_sink_dev"

        builder.add_conditional_edges(
            "tts",
            route_after_tts,
            {"tts_error": "tts_error", "audio_sink_dev": "audio_sink_dev"},
        )
        builder.add_edge("audio_sink_dev", END)
    else:

        def route_after_tts(state: VoiceState) -> str:  # type: ignore[misc]
            if _has_stage_error(state, PipelineStage.TTS):
                return "tts_error"
            return END

        builder.add_conditional_edges(
            "tts",
            route_after_tts,
            {"tts_error": "tts_error", END: END},
        )

    # ── tts_error → error_terminal (unconditional) ────────────────────────────
    builder.add_edge("tts_error", "error_terminal")

    # ── error_terminal → END (unconditional) ──────────────────────────────────
    builder.add_edge("error_terminal", END)

    return builder.compile()  # type: ignore[return-value]


# ── public orchestration engine ────────────────────────────────────────────────


class VoiceGraph:
    """
    Public face of the pipeline. Manages request lifecycle, load shedding,
    QoS budget, tracing, metrics, and task cancellation.

    Each instance owns:
    - A compiled LangGraph (built by _build_graph_for_instance) whose nodes
      close over the instance's stt/llm/tts implementations. This means
      injected test doubles, remote clients, or specialised nodes are actually
      used in run() — not just in stream() and stream_full().
    - A VoiceGraphConfig that drives timeouts, retry limits, the concurrency
      cap, and the default QoS tier for callers that omit an explicit mode.
    - Its own LoadSheddingGuard so the three public variants cannot steal
      capacity from each other's pools.

    The class carries no inter-request state — it is safe to call concurrently
    from any number of asyncio tasks and is horizontally scalable.
    """

    def __init__(
        self,
        stt: STTNodeProtocol | None = None,
        llm: LLMNodeProtocol | None = None,
        tts: TTSNodeProtocol | None = None,
        version: str = GRAPH_VERSION,
        config: VoiceGraphConfig | None = None,
    ) -> None:
        self._stt: STTNodeProtocol = stt or _default_stt_node
        self._llm: LLMNodeProtocol = llm or _default_llm_node
        self._tts: TTSNodeProtocol = tts or _default_tts_node
        self._version = version
        self._cfg: VoiceGraphConfig = config or VoiceGraphConfig()

        # Compile a fresh graph that closes over THIS instance's node implementations
        # and config. self._graph.ainvoke() in run() therefore uses self._stt/llm/tts,
        # not the module-level singletons. Each call to __init__ produces a separate
        # compiled graph object.
        self._graph: RunnableSerializable = _build_graph_for_instance(
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            cfg=self._cfg,
            is_dev=IS_DEV,
        )

        # Per-instance shedder with the cap from this instance's config.
        # The three public singletons each get different caps so a flood of
        # standard-tier requests cannot crowd out the realtime pool.
        self._shedder: LoadSheddingGuard = LoadSheddingGuard(
            max_inflight=self._cfg.max_inflight
        )

        # Maps request_id → asyncio.Task for external cancellation support.
        # Populated in run(); entries are removed in the finally block.
        self._active_tasks: dict[str, asyncio.Task] = {}

    def _prepare_state(self, state: dict[str, Any]) -> VoiceState:
        """
        Inject defaults and normalise caller-supplied state before graph execution.

        Guarantees:
        - request_id is always present and unique.
        - mode is always one of the three valid ExecutionMode values.
        - qos_tier reflects the instance's default_tier when mode is omitted,
          so voice_graph_realtime treats all calls as REALTIME without the caller
          having to explicitly set mode="realtime".
        - Retry counters and internal bookkeeping fields are initialised to zero.
        """
        rid = state.get("request_id") or new_request_id()
        raw_mode = state.get("mode", "api")

        if raw_mode not in ("api", "stream", "realtime"):
            raw_mode = "api"

        mode: ExecutionMode = cast(ExecutionMode, raw_mode)
        # Use the instance's default_tier as the fallback, not a hardcoded constant.
        tier = _MODE_TO_TIER.get(mode, self._cfg.default_tier)

        return cast(
            VoiceState,
            {
                # Defaults that the caller can always override
                "mode": "api",
                "degraded": False,
                "error": "",
                "error_stage": "",
                "abort_reason": "",
                "stage": PipelineStage.PENDING.value,
                "stage_latencies": {},
                "stt_retries": 0,
                "llm_retries": 0,
                "session_turn_appended": False,
                "graph_version": self._version,
                "qos_tier": tier.name,
                # Caller values override the defaults above
                **state,
                # These two always win regardless of what the caller passed
                "request_id": rid,
            },
        )

    # ── full run ───────────────────────────────────────────────────────────────

    async def run(
        self,
        state: dict[str, Any],
        timeout: float | None = None,
    ) -> VoicePipelineResult:
        """
        Execute the full STT → LLM → sanitize → TTS pipeline synchronously.

        Never raises. All errors — including load shedding, SLA timeout, and
        unexpected exceptions — are captured and returned in result.error and
        result.error_stage. If an explicit timeout is given, the pipeline is
        cancelled after that many seconds; otherwise the SLA budget for the
        QoS tier sets the deadline.

        Node injection is honoured here because self._graph was built by
        _build_graph_for_instance() with closures over self._stt/llm/tts.
        """
        prepared = self._prepare_state(state)
        rid = prepared["request_id"]
        mode: ExecutionMode = cast(ExecutionMode, prepared.get("mode", "api"))
        tier = _MODE_TO_TIER.get(mode, self._cfg.default_tier)
        t0 = time.monotonic()

        # ── load shedding ──────────────────────────────────────────────────────
        try:
            await self._shedder.enter(tier)
        except LoadSheddingRejected as exc:
            _load_shed_total.labels(tier=tier.name).inc()
            log.warning("pipeline_load_shed", request_id=rid, reason=str(exc))
            return _build_result(
                cast(
                    VoiceState,
                    {
                        **prepared,
                        "error": str(exc),
                        "error_stage": "load_shedding",
                        "degraded": True,
                        "stage": PipelineStage.FAILED.value,
                    },
                ),
                time.monotonic() - t0,
            )

        # ── SLA budget ─────────────────────────────────────────────────────────
        budget = LatencyBudget.for_tier(tier)
        budget.activate()

        _active_pipelines.inc()

        async with session_context(
            session_id=prepared.get("session_id", ""),
            request_id=rid,
        ):
            async with pipeline_span(
                session_id=prepared.get("session_id", ""),
                request_id=rid,
                audio_path=prepared.get("audio_path", ""),
                qos_tier=tier.name,
                mode=prepared.get("mode", "api"),
                version=self._version,
            ):
                try:
                    log.info(
                        "pipeline_start",
                        request_id=rid,
                        audio_path=prepared.get("audio_path", ""),
                        mode=prepared.get("mode", "api"),
                        version=self._version,
                        tier=tier.name,
                    )

                    PipelineEmitter.start(
                        session_id=prepared.get("session_id", ""),
                        request_id=rid,
                        audio_path=prepared.get("audio_path", ""),
                        qos_tier=tier.name,
                        mode=prepared.get("mode", "api"),
                        version=self._version,
                    )

                    # Register this task for external cancellation via cancel().
                    task = asyncio.current_task()
                    if task:
                        self._active_tasks[rid] = task

                    # Honour both the caller's explicit timeout and the SLA budget.
                    effective_timeout = timeout or budget.remaining_s()
                    final_state: VoiceState = await asyncio.wait_for(
                        self._graph.ainvoke(prepared),
                        timeout=effective_timeout,
                    )

                    pipeline_latency = time.monotonic() - t0
                    status = "degraded" if final_state.get("degraded") else "ok"
                    _pipeline_total.labels(
                        version=self._version, status=status, tier=tier.name
                    ).inc()
                    _pipeline_latency.observe(pipeline_latency)

                    if final_state.get("degraded"):
                        _degraded_total.inc()

                    result = _build_result(final_state, pipeline_latency)

                    log.info(
                        "pipeline_done",
                        request_id=rid,
                        status=status,
                        latency_s=round(pipeline_latency, 3),
                        stage_latencies=result["stage_latencies"],
                        error=result["error"] or None,
                        stt_retries=final_state.get("stt_retries", 0),
                        llm_retries=final_state.get("llm_retries", 0),
                    )

                    PipelineEmitter.done(
                        session_id=prepared.get("session_id", ""),
                        request_id=rid,
                        wall_s=pipeline_latency,
                        qos_tier=tier.name,
                        mode=prepared.get("mode", "api"),
                        version=self._version,
                        stage_latencies=result.get("stage_latencies", {}),
                        degraded=final_state.get("degraded", False),
                    )

                    return result

                except asyncio.CancelledError:
                    pipeline_latency = time.monotonic() - t0
                    _cancellations.labels(stage="pipeline").inc()
                    _pipeline_total.labels(
                        version=self._version, status="cancelled", tier=tier.name
                    ).inc()
                    log.warning(
                        "pipeline_cancelled",
                        request_id=rid,
                        latency_s=round(pipeline_latency, 3),
                    )
                    PipelineEmitter.cancelled(
                        session_id=prepared.get("session_id", ""),
                        request_id=rid,
                        stage="pipeline",
                        version=self._version,
                    )
                    return _build_result(
                        cast(
                            VoiceState,
                            {
                                **prepared,
                                "error": "Pipeline cancelled.",
                                "error_stage": "pipeline",
                                "degraded": True,
                                "stage": PipelineStage.FAILED.value,
                            },
                        ),
                        pipeline_latency,
                    )

                except asyncio.TimeoutError:
                    pipeline_latency = time.monotonic() - t0
                    _pipeline_total.labels(
                        version=self._version, status="timeout", tier=tier.name
                    ).inc()
                    log.error(
                        "pipeline_timeout",
                        request_id=rid,
                        latency_s=round(pipeline_latency, 3),
                    )
                    PipelineEmitter.failed(
                        session_id=prepared.get("session_id", ""),
                        request_id=rid,
                        stage="pipeline",
                        error=f"Pipeline timed out after {effective_timeout:.0f}s.",
                        error_type="TimeoutError",
                        qos_tier=tier.name,
                        mode=prepared.get("mode", "api"),
                        version=self._version,
                    )
                    return _build_result(
                        cast(
                            VoiceState,
                            {
                                **prepared,
                                "error": f"Pipeline timed out after {effective_timeout:.0f}s.",
                                "error_stage": "pipeline",
                                "degraded": True,
                                "stage": PipelineStage.FAILED.value,
                            },
                        ),
                        pipeline_latency,
                    )

                except Exception as exc:
                    pipeline_latency = time.monotonic() - t0
                    _pipeline_total.labels(
                        version=self._version, status="error", tier=tier.name
                    ).inc()
                    log.error(
                        "pipeline_unexpected_error", request_id=rid, error=str(exc)
                    )
                    PipelineEmitter.failed(
                        session_id=prepared.get("session_id", ""),
                        request_id=rid,
                        stage="pipeline",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        qos_tier=tier.name,
                        mode=prepared.get("mode", "api"),
                        version=self._version,
                    )
                    return _build_result(
                        cast(
                            VoiceState,
                            {
                                **prepared,
                                "error": str(exc),
                                "error_stage": "pipeline",
                                "degraded": True,
                                "stage": PipelineStage.FAILED.value,
                            },
                        ),
                        pipeline_latency,
                    )

                finally:
                    _active_pipelines.dec()
                    await self._shedder.exit()
                    self._active_tasks.pop(rid, None)  # noqa

    # ── token streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        state: dict[str, Any],
        stt_first: bool = True,
    ) -> AsyncIterator[str]:
        """
        Yield LLM tokens as they arrive (text streaming path).

        1. If stt_first=True, STT runs synchronously using this instance's
           self._stt — honoring any injected implementation.
        2. LLM tokens stream via self._llm.stream().
        3. The caller pipes the token stream into TTS if needed.

        Uses this instance's _shedder and LatencyBudget, same as run().
        """
        prepared = self._prepare_state(state)
        rid = prepared["request_id"]
        mode: ExecutionMode = cast(ExecutionMode, prepared.get("mode", "api"))
        tier = _MODE_TO_TIER.get(mode, self._cfg.default_tier)

        try:
            await self._shedder.enter(tier)
        except LoadSheddingRejected as exc:
            _load_shed_total.labels(tier=tier.name).inc()
            log.warning("stream_load_shed", request_id=rid, reason=str(exc))
            yield APOLOGY_LLM
            return

        budget = LatencyBudget.for_tier(tier)
        budget.activate()

        with tracer.start_as_current_span("voice_pipeline.stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("qos_tier", tier.name)

            try:
                if stt_first:
                    # Run STT synchronously using this instance's self._stt so
                    # injection works here exactly as it does in run().
                    _validate_audio_path(prepared["audio_path"])
                    stt_result = await _with_timeout(
                        self._stt.transcribe(
                            audio_path=prepared["audio_path"],
                            language=prepared.get("language"),
                            prompt=prepared.get("stt_prompt"),
                            request_id=rid,
                        ),
                        timeout=self._cfg.stt_timeout,
                        stage="STT",
                    )
                    san = sanitize(
                        stt_result["text"],
                        max_chars=self._cfg.max_transcript_chars,
                        request_id=rid,
                    )
                    prompt = san.text.strip()
                else:
                    # Caller already set user_input or is passing a raw prompt.
                    prompt = prepared.get("user_input", "").strip()

                if not prompt:
                    prompt = (
                        "The user's audio could not be transcribed. "
                        "Apologise briefly and ask them to try again."
                    )

                log.info("stream_start", request_id=rid, prompt_len=len(prompt))

                async for token in self._llm.stream(prompt, request_id=rid):
                    yield token

            except asyncio.CancelledError:
                _cancellations.labels(stage="stream").inc()
                log.warning("stream_cancelled", request_id=rid)
                span.set_status(StatusCode.ERROR, "Cancelled")
                raise

            except LatencyBudgetExceeded:
                _budget_breached.labels(stage="stream").inc()
                log.warning("stream_budget_exceeded", request_id=rid)
                yield APOLOGY_LLM

            except Exception as exc:
                log.error("stream_error", request_id=rid, error=str(exc))
                span.set_status(StatusCode.ERROR, str(exc))
                yield APOLOGY_LLM

            finally:
                await self._shedder.exit()

    # ── full streaming pipeline ────────────────────────────────────────────────

    async def stream_full(
        self,
        state: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """
        End-to-end streaming audio pipeline with STT, LLM, and TTS running
        concurrently as asyncio.Tasks connected by bounded queues.

        Audio bytes are yielded as each sentence is synthesised — the caller
        can start playback before STT has finished transcribing and before the
        LLM has finished generating. Each stage has independent error isolation:
        a failure puts a sentinel into the downstream queue so the next stage
        exits cleanly without hanging.

        Stage dataflow
        ──────────────
        STT (wav) → [stt_llm_q: BoundedPipelineQueue] →
        LLM (tokens) → [llm_tts_q: BoundedPipelineQueue] →
        TTS (bytes) → caller

        Backpressure
        ────────────
        When TTS is slow, llm_tts_q fills and the LLM task backs off with a
        brief sleep before each put, naturally slowing token production without
        dropping content or growing queues unboundedly. The same mechanism
        applies between STT and LLM via stt_llm_q.

        Queue depths are taken from this instance's cfg rather than module-level
        os.getenv reads, so the three public singletons can be tuned independently.

        Early LLM start
        ───────────────
        The LLM worker fires as soon as the rolling STT transcript crosses
        cfg.min_prompt_chars — it does not wait for STT to finish. This reduces
        perceived latency by overlapping generation with transcription of the
        tail of the audio.
        """
        prepared = self._prepare_state(state)
        rid = prepared["request_id"]
        raw_mode = prepared.get("mode", "realtime")
        if raw_mode not in ("api", "stream", "realtime"):
            raw_mode = "realtime"
        mode: ExecutionMode = cast(ExecutionMode, raw_mode)
        tier = _MODE_TO_TIER.get(mode, self._cfg.default_tier)

        try:
            await self._shedder.enter(tier)
        except LoadSheddingRejected as exc:
            _load_shed_total.labels(tier=tier.name).inc()
            log.warning("stream_full_load_shed", request_id=rid, reason=str(exc))
            return

        budget = LatencyBudget.for_tier(tier)
        budget.activate()

        # Per-call queues — never shared across requests. Bounded to avoid
        # unbounded memory growth if any stage produces faster than downstream
        # can consume.
        stt_llm_q: BoundedPipelineQueue = BoundedPipelineQueue(
            maxsize=self._cfg.stt_llm_queue_depth, name="stt_to_llm"
        )
        llm_tts_q: BoundedPipelineQueue = BoundedPipelineQueue(
            maxsize=self._cfg.llm_tts_queue_depth, name="llm_to_tts"
        )

        _stream_full_active.inc()

        with tracer.start_as_current_span("voice_pipeline.stream_full") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("qos_tier", tier.name)
            t0 = time.monotonic()

            # ── Stage 1: STT → stt_llm_q ──────────────────────────────────────
            async def _stt_worker() -> None:
                """
                Stream audio segments into stt_llm_q as they become available.
                The None sentinel is always sent in the finally block so the LLM
                worker exits even if STT fails partway through.
                """
                try:
                    _validate_audio_path(prepared["audio_path"])
                    async for seg in self._stt.transcribe_stream(
                        audio_path=prepared["audio_path"],
                        language=prepared.get("language"),
                        prompt=prepared.get("stt_prompt"),
                        request_id=rid,
                    ):
                        text = seg.get("text", "").strip()
                        if text:
                            await stt_llm_q.put(text)
                        if seg.get("is_final"):
                            break
                except Exception as exc:
                    log.error("stream_full_stt_error", request_id=rid, error=str(exc))
                finally:
                    await stt_llm_q.put(None)  # sentinel — always sent

            # ── Stage 2: stt_llm_q → LLM → llm_tts_q ─────────────────────────
            async def _llm_worker() -> None:
                """
                Drain STT segments as they arrive, accumulating a transcript.
                Fires the LLM in a sub-task as soon as min_prompt_chars is crossed
                so token generation overlaps with ongoing transcription.

                The LLM sub-task applies backpressure on llm_tts_q via a brief
                sleep when the queue is full, throttling token production to match
                TTS consumption rate.

                The None sentinel is always put into llm_tts_q in the finally block
                so the TTS drain exits cleanly even if this worker fails.
                """
                llm_task: asyncio.Task | None = None

                try:
                    buffer = ""
                    llm_started = False

                    while True:
                        segment = await asyncio.wait_for(
                            stt_llm_q.get(),
                            timeout=self._cfg.stt_timeout,
                        )
                        if segment is None:
                            break
                        stt_llm_q.task_done()
                        buffer += " " + segment

                        # Start LLM early once we can infer the user's intent.
                        # Fires exactly once per request — the flag prevents re-triggering
                        # as additional STT segments arrive after the task is created.
                        if (
                            not llm_started
                            and len(buffer.strip()) >= self._cfg.min_prompt_chars
                        ):
                            llm_started = True

                            async def _run_llm(initial_prompt: str) -> None:
                                from app.user_tracking.session_service.conversation_memory import (
                                    conversation_memory,
                                )
                                from app.user_tracking.transcript.transcription import (
                                    transcript_writer,
                                )

                                accumulated: list[str] = []
                                try:
                                    async for token in self._llm.stream(
                                        initial_prompt, request_id=rid
                                    ):
                                        accumulated.append(token)
                                        while llm_tts_q.full():
                                            await asyncio.sleep(0.02)
                                        await llm_tts_q.put(token)
                                finally:
                                    await llm_tts_q.put(None)  # noqa
                                    if accumulated:
                                        full_response = "".join(accumulated)
                                        sid = prepared.get("session_id")
                                        transcript = initial_prompt.split("User:")[
                                            -1
                                        ].strip()
                                        await conversation_memory.commit(
                                            sid, transcript, full_response
                                        )
                                        await transcript_writer.write_turn(
                                            session_id=sid,
                                            user_text=transcript,
                                            assistant_text=full_response,
                                            request_id=rid,
                                        )

                            llm_task = asyncio.create_task(_run_llm(buffer.strip()))

                    # Fallback path: the entire utterance finished before crossing
                    # min_prompt_chars (very short speech, or STT produced little text).
                    # Fire the LLM now with whatever we have.
                    if not llm_started:
                        prompt = buffer.strip() or (
                            "The user's audio could not be transcribed. "
                            "Apologise briefly and ask them to try again."
                        )

                        async def _run_llm_fallback(p: str) -> None:
                            from app.user_tracking.session_service.conversation_memory import (
                                conversation_memory,
                            )
                            from app.user_tracking.transcript.transcription import (
                                transcript_writer,
                            )

                            accumulated: list[str] = []
                            try:
                                async for token in self._llm.stream(p, request_id=rid):
                                    accumulated.append(token)
                                    await llm_tts_q.put(token)
                            finally:
                                await llm_tts_q.put(None)  # noqa
                                if accumulated:
                                    full_response = "".join(accumulated)
                                    sid = prepared.get("session_id")
                                    transcript = p.split("User:")[-1].strip()
                                    await conversation_memory.commit(
                                        sid, transcript, full_response
                                    )
                                    await transcript_writer.write_turn(
                                        session_id=sid,
                                        user_text=transcript,
                                        assistant_text=full_response,
                                        request_id=rid,
                                    )

                        llm_task = asyncio.create_task(_run_llm_fallback(prompt))

                    if llm_task:
                        await llm_task

                except asyncio.TimeoutError:
                    log.warning("llm_worker_stt_timeout", request_id=rid)

                except Exception as exc:
                    log.error("llm_worker_error", request_id=rid, error=str(exc))

                finally:
                    # Safety net: ensure TTS drain never hangs even if the LLM
                    # task was never created or exited abnormally.
                    await llm_tts_q.put(None)

            # ── Stage 3: llm_tts_q → TTS → audio bytes ────────────────────────
            async def _token_drain() -> AsyncIterator[str]:
                """
                Yield tokens from llm_tts_q until the sentinel arrives.
                Used as the token_stream argument to tts.synthesize_stream().
                """
                while True:
                    token = await asyncio.wait_for(
                        llm_tts_q.get(),
                        timeout=self._cfg.llm_timeout,
                    )
                    if token is None:
                        return
                    llm_tts_q.task_done()
                    yield token

            # ── launch and coordinate ──────────────────────────────────────────
            stt_task = asyncio.create_task(_stt_worker())
            llm_task = asyncio.create_task(_llm_worker())

            try:
                voice = prepared.get("tts_voice")  # type: ignore[assignment]
                speed = float(prepared.get("tts_speed", 1.0))

                # TTS runs in the foreground so we can yield audio bytes directly.
                # STT and LLM run as background tasks so all three stages overlap.
                async for audio_bytes in self._tts.synthesize_stream(
                    token_stream=_token_drain(),
                    voice=voice,
                    speed=speed,
                    request_id=rid,
                ):
                    yield audio_bytes

                # Drain background tasks before exiting. return_exceptions=True
                # prevents a background exception from masking the audio stream.
                await asyncio.gather(stt_task, llm_task, return_exceptions=True)

                pipeline_latency = time.monotonic() - t0
                log.info(
                    "stream_full_ok",
                    request_id=rid,
                    latency_s=round(pipeline_latency, 3),
                    tier=tier.name,
                )
                span.set_attribute("latency_s", round(pipeline_latency, 3))

            except asyncio.CancelledError:
                _cancellations.labels(stage="stream_full").inc()
                log.warning("stream_full_cancelled", request_id=rid)
                # Propagate cancellation to the background tasks so they don't
                # linger after the caller has already given up.
                stt_task.cancel()
                llm_task.cancel()
                raise

            except Exception as exc:
                log.error("stream_full_error", request_id=rid, error=str(exc))
                span.set_status(StatusCode.ERROR, str(exc))
                raise

            finally:
                _stream_full_active.dec()
                await self._shedder.exit()

    # ── health aggregation ─────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """
        Poll all three nodes concurrently and return a unified health snapshot.

        Used before routing REALTIME requests to decide whether to reject them
        early rather than letting them hit an OPEN circuit breaker mid-flight.
        The result can also be exposed directly as a /health endpoint body.
        """
        stt_h, llm_h, tts_h = await asyncio.gather(
            self._stt.health(),
            self._llm.health(),
            self._tts.health(),
            return_exceptions=True,
        )

        def _safe(h: ServiceHealthState | BaseException) -> dict:
            if isinstance(h, BaseException):
                return {"healthy": False, "error": str(h)}
            return {
                "healthy": h.healthy,
                "circuit_state": h.circuit_state,
                "inflight": h.inflight,
                "degraded": h.degraded,
                "service": h.service,
            }

        overall_healthy = all(
            isinstance(h, ServiceHealthState) and h.healthy
            for h in [stt_h, llm_h, tts_h]
        )

        return {
            "healthy": overall_healthy,
            "inflight": self._shedder.inflight,
            "max_inflight": self._cfg.max_inflight,
            "version": self._version,
            "tier_default": self._cfg.default_tier.name,
            "retries": {
                "max_stt": self._cfg.max_stt_retries,
                "max_llm": self._cfg.max_llm_retries,
            },
            "nodes": {
                "stt": _safe(stt_h),
                "llm": _safe(llm_h),
                "tts": _safe(tts_h),
            },
        }

    # ── external cancellation ──────────────────────────────────────────────────

    def cancel(
        self,
        request_id: str,
        *,
        reason: str = "interrupt",
        source: str = "unknown",
    ) -> bool:
        """
        Cancel an in-flight run() call by request_id.

        Returns True if a running task was found and cancelled, False if the
        request is unknown or already completed. The cancellation context is
        attached to the task object so any code that catches CancelledError
        downstream can inspect why the task was cancelled.
        """
        task = self._active_tasks.get(request_id)

        if not task:
            log.info(
                "pipeline_cancel_not_found",
                request_id=request_id,
                reason=reason,
                source=source,
            )
            return False

        if task.done():
            log.info(
                "pipeline_cancel_ignored_task_done",
                request_id=request_id,
                reason=reason,
            )
            return False

        setattr(
            task,
            "__cancellation_context__",
            {
                "reason": reason,
                "source": source,
                "requested_at": time.time(),
            },
        )
        task.cancel()
        _cancellations.labels(stage="external").inc()

        log.warning(
            "pipeline_cancellation_initiated",
            request_id=request_id,
            reason=reason,
            source=source,
        )
        return True

    # ── graceful shutdown ──────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Cancel all in-flight tasks and release node resources.
        Call from a SIGTERM handler or FastAPI lifespan event.
        """
        log.info(
            "pipeline_shutdown_start",
            active_tasks=len(self._active_tasks),
            version=self._version,
        )

        for rid, task in list(self._active_tasks.items()):
            if not task.done():
                task.cancel()
                log.info("pipeline_shutdown_cancel", request_id=rid)

        if self._active_tasks:
            await asyncio.gather(*self._active_tasks.values(), return_exceptions=True)

        await asyncio.gather(
            self._stt.close(),
            self._llm.close(),
            self._tts.close(),
            return_exceptions=True,
        )

        log.info("pipeline_shutdown_complete", version=self._version)


# ── versioned graph singletons ─────────────────────────────────────────────────
#
# Each singleton has a genuinely distinct VoiceGraphConfig, its own compiled
# LangGraph, and its own LoadSheddingGuard. "Distinct" means different timeouts,
# different retry policies, different concurrency caps, and a different default
# QoS tier — not just a different version label in metrics.
#
# Callers choose the appropriate singleton for the latency/reliability trade-off
# they need. The three variants do not share capacity: a burst of standard
# requests cannot crowd out realtime slots.

# Balanced defaults: one retry per stage before giving up. Uses the settings
# singleton for all timeout and capacity values.
voice_graph = VoiceGraph(version="v2")

# Tight timeouts; REALTIME tier by default; zero retries.
# A retry after a blown SLA budget only pushes the user further past the deadline.
# Concurrency cap is low to keep this pool reserved for truly latency-sensitive callers.
voice_graph_realtime = VoiceGraph(
    version="realtime",
    config=VoiceGraphConfig(
        stt_timeout=10.0,
        llm_timeout=15.0,
        tts_timeout=8.0,
        max_inflight=30,
        default_tier=QoSTier.REALTIME,
        max_stt_retries=0,
        max_llm_retries=0,
    ),
)

# Medium timeouts: tighter than standard to hit a lower p50 latency, but still
# allows one retry per stage before giving up. Aimed at callers that want snappy
# responses without the zero-tolerance latency budget of the realtime pool.
voice_graph_low_latency = VoiceGraph(
    version="low_latency",
    config=VoiceGraphConfig(
        stt_timeout=15.0,
        llm_timeout=20.0,
        tts_timeout=12.0,
        max_inflight=50,
        default_tier=QoSTier.STANDARD,
        max_stt_retries=1,
        max_llm_retries=1,
    ),
)

# ── smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio

    async def _smoke():
        print("── health check ──")
        h = await voice_graph.health()
        print("Overall healthy:", h["healthy"])
        for name, node_h in h["nodes"].items():
            print(f"  {name}: {node_h}")
        print("Realtime health:", (await voice_graph_realtime.health())["healthy"])
        print(
            "Low-latency health:", (await voice_graph_low_latency.health())["healthy"]
        )

        print("\n── full run ──")
        result = await voice_graph.run(
            {
                "audio_path": "audio/temp_IN/test.wav",
                "mode": "api",
            }
        )
        print("Request ID  :", result["request_id"])
        print("Transcript  :", result["transcript"])
        print("LLM Response:", result["llm_response"])
        print("Audio output:", result["audio_output"])
        print("Error       :", result["error"] or "none")
        print("Degraded    :", result["degraded"])
        print("Latencies   :", result["stage_latencies"])
        print("Total time  :", result["pipeline_latency_s"], "s")
        print("Version     :", result["graph_version"])
        print("STT retries :", result["metadata"]["stt_retries"])
        print("LLM retries :", result["metadata"]["llm_retries"])

        print("\n── token streaming ──")
        async for token in voice_graph.stream(
            {
                "audio_path": "audio/temp_IN/test.wav",
            }
        ):
            print(token, end="", flush=True)
        print()

        print("\n── full audio streaming (stream_full) ──")
        chunk_count = 0
        async for audio_bytes in voice_graph_realtime.stream_full(
            {
                "audio_path": "audio/temp_IN/test.wav",
                "mode": "realtime",
            }
        ):
            chunk_count += 1
            print(f"  audio chunk {chunk_count}: {len(audio_bytes)} bytes")

        await voice_graph.shutdown()
        await voice_graph_realtime.shutdown()
        await voice_graph_low_latency.shutdown()

    asyncio.run(_smoke())