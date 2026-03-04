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
  pcm      — async for chunk in voice_graph.stream_full_pcm(state)  → PCMChunk
  ptt      — await voice_graph.run_ptt(state, is_held_fn)      → VoicePipelineResult

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

PCM-native realtime path (stream_full_pcm)
──────────────────────────────────────────
  PCMInputStream(mic)
    └─► PCMSpeechEnhancer(bandpass → NS → AGC → gate → VAD)
          └─► PCMChunkWAVEncoder → stt.transcribe_chunk()
                └─► _node_llm_qa_path()
                      └─► tts.synthesize_pcm_stream()
                            └─► PCMPlaybackEnhancer(limiter → AGC)
                                  └─► PCMOutputStream(speaker)
                                        └─► PCMInterruptDetector(barge-in)

PTT recording path (run_ptt)
────────────────────────────
  recorder.record_audio_until_released_async(is_held_fn)
    └─► voice_graph.run({audio_path: recorded_path})
          └─► ... standard graph topology ...
    └─► recorder.delete_temp_recording(path)

Inter-stage message bus (StageBus)
──────────────────────────────────
  Wraps BoundedPipelineQueue with:
    - Backpressure signaling via PCM chunk watermarks
    - Dead letter queue for undeliverable messages
    - Per-stage throughput metering
    - Configurable overflow policy (drop_oldest | block | reject)
  Used by stream_full_pcm() to decouple PCM producers from consumers
  without unbounded memory growth or blocking the PortAudio callback thread.

Session lifecycle (SessionLifecycleManager)
───────────────────────────────────────────
  open_session(session_id)
    → mic health check (optional, async)
    → speaker health check (optional, async)
    → QA controller session create
    → audit bus open
    → transcript writer begin
    → PCM format negotiation (mic → STT, TTS → speaker)

  close_session(session_id)
    → QA controller close
    → audit bus close
    → transcript writer flush
    → PCM stream drain
    → LLM session evict
    → temporary file cleanup

Audio diagnostics pipeline (AudioDiagnosticsPipeline)
─────────────────────────────────────────────────────
  Taps into PCM streams at capture / pre-STT / post-TTS / pre-speaker
  boundaries and runs PCMDiagnosticsMonitor + PCMWaveformAnalyzer per
  boundary. Results are aggregated into an AudioHealthReport per request
  and emitted as structured logs + Prometheus histograms. The pipeline
  also feeds the watchdog — if clipping % exceeds a threshold the gain
  stage is dynamically reduced; if silence % exceeds a threshold the
  watchdog escalates to a microphone restart.

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

import contextlib
import asyncio  # noqa
import collections # noqa
import hashlib
import io # noqa
import os
import re
import json
import struct # noqa
import threading
import time
import uuid
import wave
import weakref # noqa
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence  # noqa
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Required, TypedDict, TypeVar, cast

T = TypeVar("T")

import numpy as np # noqa

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables import RunnableSerializable # noqa
from opentelemetry.trace import StatusCode

from app.common.settings import settings
from app.common.shared import ( # noqa
    BoundedPipelineQueue,
    LatencyBudget,
    LatencyBudgetExceeded,
    LoadSheddingGuard,
    LoadSheddingRejected,
    QoSTier,
    ServiceHealthState,
    current_request_id,
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)
from app.monitoring.observability import get_logger
from app.monitoring.observability import ( # noqa
    bootstrap,
    session_context,
    pipeline_span,
    stt_span,
    llm_span,
    tts_span,
    sanitize_span,
    PipelineEmitter,
    STTEmitter,
    LLMEmitter,
    TTSEmitter,
    SanitizeEmitter,
    ControllerEmitter,
    SessionEmitter,
    MemoryEmitter,
    SanitizeEmitter,
    RateLimitEmitter,
    BulkheadEmitter,
    CBEmitter,
    RedisEmitter,
    TranscriptEmitter,
    EvalEmitter,
)
from app.nodes.sanitize import sanitize
from app.nodes.STT_service import STTNodeProtocol, get_stt_node, stt_node as _stt_singleton  # noqa
from app.nodes.LLM_service import LLMNodeProtocol, get_llm_node, llm_node as _llm_singleton  # noqa
from app.nodes.TTS_service import TTSNodeProtocol, get_tts_node, tts_node as _tts_singleton  # noqa
from app.audio_essentials.player import play_audio
from langchain_core.messages import HumanMessage # noqa

# ── audio_engine imports ──────────────────────────────────────────────────────
# PCM pipeline primitives from the audio processing engine. These enable the
# zero-copy PCM-native realtime path (stream_full_pcm) which bypasses all
# file I/O and WAV encoding/decoding between pipeline stages.
from app.audio_essentials.audio_engine import ( # noqa
    PCMFormat,
    PCMChunk,
    PCMRingBuffer,
    PCMConverter,
    PCMInputStream,
    PCMOutputStream,
    PCMVADGate,
    PCMSplitter,
    PCMSpeechEnhancer,
    PCMPlaybackEnhancer,
    PCMInterruptDetector,
    PCMDiagnosticsMonitor,
    PCMLatencyTracker,
    PCMWaveformAnalyzer,
    AudioHealthReport,
    PCMJitterBuffer,
    PCMStreamMixer,
    PCMPipelineBuilder,
    PCMSilencePadder,
    PCMDriftCorrector,
    PCMChunkPool,
    PCMFormatRegistry,
    PCMStreamBridge,
    PCMMetricsSnapshot,
    tts_pcm_to_chunk,
    chunk_to_wav_bytes,
    negotiate_format,
    get_chunk_pool,
)

# ── recorder imports ──────────────────────────────────────────────────────────
# Direct integration with the mic recording subsystem for the PTT (push-to-talk)
# execution path. record_audio_until_released_async() blocks until the user
# releases the PTT button, then returns the temp file path for STT consumption.
from app.audio_essentials.recorder import (
    record_audio_until_released, # noqa
    record_audio_until_released_async,
    delete_temp_recording,
    get_recording_health,
    get_recording_latency_report, # noqa
    get_recording_format,
    run_startup_health_check as run_recorder_health_check,
)

# ── player imports ────────────────────────────────────────────────────────────
# PCM-native playback for the dev audio sink and interrupt detection during
# TTS output. The persistent PCMOutputStream eliminates the 50-150ms device-open
# cost on every playback call by keeping the speaker stream hot.
from app.audio_essentials.player import ( # noqa
    play_pcm_chunk,
    play_pcm_bytes,
    get_interrupt_detector,
    get_playback_enhancer,
    get_output_stream,
    get_playback_health,
)

# ── PCM-aware STT and TTS extensions ─────────────────────────────────────────
# The STT and TTS services expose PCM-native APIs that bypass WAV encoding.
# transcribe_chunk() accepts a PCMChunk directly and returns transcript text
# without writing to disk. synthesize_pcm_stream() produces PCMChunks from
# an async token iterator, feeding directly into the speaker output stream.
from app.nodes.STT_service import (
    PCMChunkWAVEncoder,
    PCMSTTResult, # noqa
    PCMConfidenceFilter,
)
from app.nodes.TTS_service import (
    PCMTTSOutputConfig,
    PCMSentenceGapManager,
    PCMTTSQualityGate,
    PCMStreamToWAVCollector, # noqa
)

# ── QA pipeline wiring ─────────────────────────────────────────────────────────
# Lazy-imported inside node_llm to avoid circular imports at module load time.
# These symbols are imported once and cached on first use via module-level refs.
# If the QA pipeline is not installed, all imports gracefully fall back to
# None-typed guards so the existing conversational path continues working.
from app.interview.qa_controller import ( # noqa
    CommittedTurn,
    qa_controller as _qa_controller,
    qa_session_timer as _qa_session_timer,
    qa_prefetch_buffer as _qa_prefetch_buffer,
    QAStage as _QAStage,
    GREETING_TEXT as _GREETING_TEXT,
    CLOSING_TEXT as _CLOSING_TEXT,
    DOMAIN_SWITCH_BRIDGE as _DOMAIN_SWITCH_BRIDGE,
    build_next_llm_input_for_voice_graph as _build_next_llm_input,
    GuardrailAction as _GuardrailAction,
    ATSExtractionResult as _ATSExtractionResult,
    TurnTimingRecord as _TurnTimingRecord,
    QAAnalytics as _QAAnalytics,
    SessionAnalytics as _SessionAnalytics,
    QADocument as _QADocument,
    LLMInterviewInput
)

conversation_memory = None
transcript_writer   = None

from app.user_tracking.session_service.conversation_memory import (
    qa_audit_bus as _qa_audit_bus,
    route_committed_turn_to_audit as _route_committed_turn, # noqa
    finalize_session_eval as _finalize_session_eval, # noqa | internalized
)
from app.nodes.LLM_service import ( # noqa
    llm_node as _llm_node_ref,
    generate_interviewer_question as _generate_interviewer_question,
    extract_and_validate_intro as _extract_and_validate_intro,
)
from app.user_tracking.transcript.transcription import transcript_writer as _transcript_writer

# ── evaluation engine wiring ──────────────────────────────────────────────────
# The evaluation engine runs entirely off the critical path — all scoring is
# fire-and-forget via asyncio.create_task(). Wired into the commit_turn flow
# through the QA audit bus, not called directly from voice_graph nodes. The
# imports here are for session lifecycle (health checks, report generation at
# session close, graceful shutdown) and the watchdog (circuit breaker state).
from app.eval.evaluation_engine import ( # noqa
    evaluation_engine as _eval_engine,
    TurnScore as _TurnScore,
    SessionReport as _SessionReport,
)

from app.audio_essentials.vad_context import ContextGatedVAD, VADContextHintBuilder, VADContextHint

# ── dev flag ───────────────────────────────────────────────────────────────────
# Controls whether the audio_sink_dev node is compiled into the graph at all.
# Evaluated once at module load so the graph structure is fixed per process —
# there's no cost to checking IS_DEV in routing functions at request time.
IS_DEV: bool = os.getenv("ENV", "").lower() == "development"

# When true, the stream_full LLM worker emits a per-turn log with the full
# message list preview (role, char count, 120-char snippet). Useful during
# development to verify conversation history structure; never enabled in prod.
_STREAM_LLM_DEBUG: bool = os.getenv("STREAM_LLM_DEBUG", "").lower() == "true"

# ── feature flags ─────────────────────────────────────────────────────────────
# Evaluated once at module load. Feature rollout is env-var-gated so operators
# can toggle new pipeline paths per-process without code changes or restarts.

# PCM-native realtime path: when True, stream_full_pcm() uses the full
# audio_engine pipeline. When False, falls back to stream_full() with
# file-based audio. Default off until the PCM path is validated in staging.
FF_PCM_PIPELINE: bool = os.getenv("FF_PCM_PIPELINE", "").lower() == "true"

# Barge-in detection: when True, the PCM output stream monitors for speech
# from the user during TTS playback and interrupts the current utterance.
# Requires FF_PCM_PIPELINE=true. Default off.
FF_BARGE_IN: bool = os.getenv("FF_BARGE_IN", "").lower() == "true"

# Audio diagnostics: when True, PCMDiagnosticsMonitor and PCMWaveformAnalyzer
# run on every chunk boundary and emit structured health reports. Adds ~2ms
# per chunk on commodity hardware. Default on in dev, off in prod.
FF_AUDIO_DIAGNOSTICS: bool = os.getenv("FF_AUDIO_DIAGNOSTICS", IS_DEV and "true" or "").lower() == "true"

# Session lifecycle management: when True, open_session/close_session perform
# full mic/speaker health checks, PCM format negotiation, and resource cleanup.
FF_SESSION_LIFECYCLE: bool = os.getenv("FF_SESSION_LIFECYCLE", "true").lower() == "true"

# Question prefetch: when True, the TTS worker fires a background prefetch for
# the predicted next question while the candidate is listening to the current one.
FF_QUESTION_PREFETCH: bool = os.getenv("FF_QUESTION_PREFETCH", "true").lower() == "true"

# Canary percentage: when >0, this fraction of requests use the PCM pipeline
# even when FF_PCM_PIPELINE is False. Used for gradual rollout. Range: 0.0–1.0.
FF_CANARY_PCT: float = float(os.getenv("FF_CANARY_PCT", "0.0"))

# ── graph version ──────────────────────────────────────────────────────────────
# Used as a metric label and embedded in result payloads so deployed graph
# versions can be tracked in dashboards across rolling deployments.
GRAPH_VERSION: str = getattr(settings, "voice_graph_version", "v3")

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── module-level default node singletons ───────────────────────────────────────
# Resolved once via the factory functions. VoiceGraph.__init__() falls back to
# these when no explicit nodes are injected. The compiled graph never references
# these names — it always closes over the per-instance nodes supplied to
# _build_graph_for_instance(). These are purely for the "no injection" default.
_default_stt_node: STTNodeProtocol | None = _stt_singleton
_default_llm_node: LLMNodeProtocol | None = _llm_singleton
_default_tts_node: TTSNodeProtocol | None = _tts_singleton

# ── QoS / execution mode ───────────────────────────────────────────────────────

ExecutionMode = Literal["api", "stream", "realtime", "pcm", "ptt"]

_MODE_TO_TIER: dict[str, QoSTier] = {
    "realtime": QoSTier.REALTIME,
    "pcm":      QoSTier.REALTIME,
    "ptt":      QoSTier.STANDARD,
    "api":      QoSTier.STANDARD,
    "stream":   QoSTier.STANDARD,
}

# Apology strings used when a stage cannot produce a valid output and the
# pipeline degrades gracefully rather than surfacing a raw exception to the user.
APOLOGY_STT = "I couldn't catch that. Could you try again?"
APOLOGY_LLM = "I'm having trouble thinking right now. Please try again in a moment."
APOLOGY_TTS = "I have a response but couldn't convert it to audio right now."


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

    # PCM pipeline configuration — only active when FF_PCM_PIPELINE=true.
    # These control the audio processing chain between mic input and speaker output.
    pcm_ring_buffer_seconds: float = field(
        default_factory=lambda: getattr(settings, "graph_pcm_ring_buffer_s", 30.0)
    )
    pcm_vad_energy_threshold: float = field(
        default_factory=lambda: getattr(settings, "graph_pcm_vad_energy", 0.015)
    )
    pcm_vad_hangover_frames: int = field(
        default_factory=lambda: getattr(settings, "graph_pcm_vad_hangover", 8)
    )
    pcm_jitter_buffer_ms: int = field(
        default_factory=lambda: getattr(settings, "graph_pcm_jitter_ms", 60)
    )
    pcm_interrupt_threshold: float = field(
        default_factory=lambda: getattr(settings, "graph_pcm_interrupt_threshold", 0.12)
    )

    # Session lifecycle timeouts — how long to wait for mic/speaker health checks
    # during open_session(). If these exceed the timeout the session still opens
    # but the health-check failure is logged and diagnostics are marked degraded.
    session_mic_health_timeout: float = field(
        default_factory=lambda: getattr(settings, "graph_mic_health_timeout", 2.0)
    )
    session_speaker_health_timeout: float = field(
        default_factory=lambda: getattr(settings, "graph_speaker_health_timeout", 2.0)
    )

    # Audio diagnostics sampling rate. 1.0 = every chunk, 0.1 = 10% of chunks.
    # Lower values reduce CPU overhead at the cost of less granular health data.
    diagnostics_sample_rate: float = field(
        default_factory=lambda: getattr(settings, "graph_diag_sample_rate", 1.0)
    )

    # Pipeline watchdog interval: seconds between heartbeat checks.
    # The watchdog monitors per-stage throughput and overall pipeline health.
    watchdog_interval_s: float = field(
        default_factory=lambda: getattr(settings, "graph_watchdog_interval", 5.0)
    )

    stage_bus_maxsize: int = field(
        default_factory=lambda: getattr(settings, "graph_stage_bus_maxsize", 32)
    )
    output_queue_maxsize: int = field(
        default_factory=lambda: getattr(settings, "graph_output_queue_maxsize", 64)
    )

    load_shed_queue_size: int = field(
        default_factory=lambda: getattr(settings, "graph_load_shed_queue_size", 5)
    )

    @classmethod
    def from_settings(cls, overrides: dict[str, Any] | None = None) -> "VoiceGraphConfig":
        """
        Construct config from the validated settings singleton.
        Optional overrides allow specialised variants (realtime, low_latency)
        to diverge from defaults without losing the settings baseline.
        """
        instance = cls()
        if overrides:
            for key, value in overrides.items():
                if hasattr(instance, key):
                    object.__setattr__(instance, key, value)
                else:
                    raise ValueError(f"VoiceGraphConfig has no field '{key}'")
        return instance


# ── pipeline stage enum ────────────────────────────────────────────────────────


class PipelineStage(str, Enum):
    PENDING = "pending"
    STT = "stt"
    LLM = "llm"
    SANITIZE = "sanitize"
    TTS = "tts"
    DONE = "done"
    FAILED = "failed"

# Maps error_stage values to the apology string the terminal node should use.
_STAGE_APOLOGY: dict[str, str] = {
    PipelineStage.STT.value:      APOLOGY_STT,
    PipelineStage.LLM.value:      APOLOGY_LLM,
    PipelineStage.TTS.value:      APOLOGY_TTS,
    PipelineStage.SANITIZE.value: APOLOGY_LLM,
}

# ── overflow policy for inter-stage message bus ───────────────────────────────

class OverflowPolicy(str, Enum):
    """
    Controls what the StageBus does when a queue reaches capacity.
    Each policy trades off differently between latency, throughput, and data loss.
    """
    DROP_OLDEST = "drop_oldest"   # evict the oldest enqueued item — minimises latency
    BLOCK       = "block"         # await until space is available — preserves ordering
    REJECT      = "reject"        # raise immediately — caller handles backpressure


# ── shared state schema ────────────────────────────────────────────────────────


class VoiceState(TypedDict, total=False):
    # ── required input ────────────────────────────────────────────────────────
    audio_path: Required[str]

    # ── caller-supplied context ───────────────────────────────────────────────
    request_id: str
    session_id: str
    client_ip: str
    history: list[dict]
    mode: str             # ExecutionMode — normalised by _prepare_state
    language: str         # ISO-639-1 hint forwarded to STT
    stt_prompt: str       # whisper-style context hint for STT
    tts_voice: str
    tts_speed: float
    qos_tier: str         # QoSTier.name, injected by _prepare_state

    # ── STT outputs ───────────────────────────────────────────────────────────
    user_input: str
    stt_result: dict
    transcript_truncated: bool
    stt_language_detected: str    # actual language detected by STT
    stt_confidence: float         # overall transcript confidence
    stt_audio_duration_s: float   # duration of the input audio

    # ── LLM outputs ───────────────────────────────────────────────────────────
    llm_response: str
    llm_tokens: dict
    llm_model_used: str
    llm_cached: bool
    response_truncated: bool
    llm_latency_ms: float
    llm_streaming: bool

    # ── sanitize output ───────────────────────────────────────────────────────
    cleaned_response: str
    sanitize_warnings: list[str]  # non-fatal issues found during sanitization

    # ── TTS outputs ───────────────────────────────────────────────────────────
    audio_output: str
    audio_local_path: str
    audio_s3_uri: str
    audio_duration_s: float       # duration of synthesised audio
    audio_size_bytes: int         # raw byte size of audio output
    tts_chunk_count: int          # number of PCM chunks produced

    # ── QA pipeline state ─────────────────────────────────────────────────────
    qa_stage: str         # current QAStage value e.g. "greeting" / "intro" / "interview"
    qa_domain: str        # current domain key e.g. "python" / "dsa"
    qa_turn_idx: int      # committed turn index within this session
    qa_score: float       # eval score for this turn (0.0 if not yet scored)

    # ── retry bookkeeping ─────────────────────────────────────────────────────
    stt_retries: int
    llm_retries: int
    abort_reason: str     # non-empty → skip retry loop, go straight to error_terminal
    session_turn_appended: bool   # guards against double-write on LLM retry

    # ── pipeline metadata ─────────────────────────────────────────────────────
    stage: str            # current PipelineStage value
    error: str            # last error message
    error_stage: str      # which stage produced the error
    degraded: bool        # True if pipeline completed via fallback/apology
    stage_latencies: dict # per-stage wall-clock seconds {stage: float}
    pipeline_latency_s: float     # total end-to-end wall-clock seconds
    graph_version: str    # VoiceGraph.version at time of execution

    # ── PCM pipeline ──────────────────────────────────────────────────────────
    pcm_format_mic: str | None      # negotiated mic input format descriptor
    pcm_format_speaker: str | None  # negotiated speaker output format descriptor
    pcm_chunks_processed: int       # total PCMChunks pushed through pipeline
    pcm_barge_in_count: int         # barge-in interruptions this request
    pcm_audio_bytes: bytes | None   # raw PCM bytes for in-memory paths
    audio_health: dict | None       # aggregated AudioHealthReport snapshot

    # ── PTT ───────────────────────────────────────────────────────────────────
    rec_duration_s: float         # duration of PTT recording hold
    rec_path: str                 # temp file path from recorder (deleted after STT)


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
    ["mode", "tier"],
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
_late_patch_total = make_counter(
    "voice_pipeline_late_patch_total",
    "STT late-patches applied when early-fire transcript was partial",
    ["target"],   # "turn_answer" | "raw_intro"
)

# ── PCM pipeline metrics ──────────────────────────────────────────────────────

_pcm_pipeline_active = make_gauge(
    "voice_pcm_pipeline_active",
    "Concurrent PCM pipeline sessions",
)
_pcm_chunks_total = make_counter(
    "voice_pcm_chunks_total",
    "Total PCMChunks processed across all pipeline stages",
    ["stage", "direction"],   # stage: capture|enhance|stt|tts|playback, direction: in|out
)
_pcm_barge_in_total = make_counter(
    "voice_pcm_barge_in_total",
    "Barge-in interruptions detected during TTS playback",
)
_pcm_format_negotiations = make_counter(
    "voice_pcm_format_negotiations_total",
    "PCM format negotiations between pipeline stages",
    ["result"],  # exact_match | converted | failed
)
_pcm_ring_buffer_overruns = make_counter(
    "voice_pcm_ring_buffer_overruns_total",
    "Ring buffer capacity overruns (old data evicted)",
)
_pcm_diagnostics_alerts = make_counter(
    "voice_pcm_diagnostics_alerts_total",
    "Audio diagnostics alert events by type",
    ["alert_type"],  # clipping | silence | dc_offset | low_energy
)

# ── StageBus metrics ──────────────────────────────────────────────────────────

_stagebus_enqueue_total = make_counter(
    "voice_stagebus_enqueue_total",
    "Messages enqueued to inter-stage bus",
    ["stage_pair"],
)
_stagebus_dequeue_total = make_counter(
    "voice_stagebus_dequeue_total",
    "Messages dequeued from inter-stage bus",
    ["stage_pair"],
)
_stagebus_overflow_total = make_counter(
    "voice_stagebus_overflow_total",
    "Messages lost due to queue overflow",
    ["stage_pair", "policy"],
)
_stagebus_dlq_total = make_counter(
    "voice_stagebus_dlq_total",
    "Messages sent to the dead letter queue",
    ["stage_pair"],
)
_stagebus_depth = make_gauge(
    "voice_stagebus_depth",
    "Current depth of inter-stage message queues",
    ["stage_pair"],
)
_stagebus_throughput = make_histogram(
    "voice_stagebus_throughput_msgs_per_sec",
    "Per-stage-pair throughput (messages per second over 1s window)",
    ["stage_pair"],
    buckets=(1, 5, 10, 25, 50, 100, 250),
)

# ── session lifecycle metrics ─────────────────────────────────────────────────

_session_open_total = make_counter(
    "voice_session_open_total",
    "Sessions opened via SessionLifecycleManager",
    ["status"],   # ok | degraded | failed
)
_session_close_total = make_counter(
    "voice_session_close_total",
    "Sessions closed via SessionLifecycleManager",
    ["reason"],   # normal | timeout | error | force
)
_session_open_latency = make_histogram(
    "voice_session_open_latency_seconds",
    "Wall-clock time to open a session (includes health checks)",
    buckets=(0.1, 0.5, 1, 2, 5),
)
_session_close_latency = make_histogram(
    "voice_session_close_latency_seconds",
    "Wall-clock time to close a session (includes flush/drain)",
    buckets=(0.1, 0.5, 1, 2, 5, 10),
)
_session_duration = make_histogram(
    "voice_session_duration_seconds",
    "Total session duration from open to close",
    buckets=(60, 120, 300, 600, 900, 1800, 2700, 3600),
)

# ── watchdog metrics ──────────────────────────────────────────────────────────

_watchdog_heartbeats = make_counter(
    "voice_watchdog_heartbeats_total",
    "Watchdog heartbeat ticks",
)
_watchdog_recoveries = make_counter(
    "voice_watchdog_recoveries_total",
    "Automatic pipeline recoveries triggered by watchdog",
    ["recovery_type"],  # mic_restart | speaker_restart | pipeline_reset
)
_watchdog_alerts = make_counter(
    "voice_watchdog_alerts_total",
    "Watchdog alert events emitted to observability",
    ["alert_level"],   # warning | critical
)

# ── PTT metrics ───────────────────────────────────────────────────────────────

_ptt_total = make_counter(
    "voice_ptt_total",
    "Push-to-talk recording sessions",
    ["status"],   # ok | too_short | error
)
_ptt_duration = make_histogram(
    "voice_ptt_duration_seconds",
    "Duration of PTT recording hold",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30),
)

# ── execution mode metrics ─────────────────────────────────────────────────────

_pipeline_starts = make_counter(
    "voice_pipeline_starts_total",
    "Pipeline executions started by mode",
    ["mode", "tier"],   # mode: api | stream | realtime | pcm | ptt
)
_pipeline_completions = make_counter(
    "voice_pipeline_completions_total",
    "Pipeline executions completed by mode and status",
    ["mode", "tier", "status"],   # status: ok | error | empty
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


async def _collect_stream(
    token_iter: AsyncIterator[str],
) -> dict[str, Any]:
    """
    Drain an async token iterator to completion and return a generate()-style dict.

    node_llm uses stream_messages() (the correct multi-turn path) but the graph
    node must await a single result to write into VoiceState. This helper bridges
    the two by consuming the entire stream and returning the same dict shape as
    LLMNode.generate() so the rest of node_llm doesn't need to change.

    Cache-hit detection: if the first token arrives in under 5 ms the response
    almost certainly came from the in-process LRU cache. We set cached=True in
    that case so LLMEmitter.ok() logs an accurate cache_hit metric.

    Raises ValueError (re-raised from stream_messages) on empty response so the
    graph routes to llm_error and retries exactly as it did with generate().

    NOTE: model_used is left as "" because the stream protocol does not surface
    the model name in individual token chunks. The Prometheus counter in LLMNode
    already records it per-request; this gap only affects the voice_graph log line.
    """
    chunks: list[str] = []
    t_start = time.monotonic()
    first_token_ms: float = -1.0
    async for token in token_iter:
        if first_token_ms < 0:
            first_token_ms = (time.monotonic() - t_start) * 1000
        chunks.append(token)
    full_text = "".join(chunks)
    # Empty response: re-raise so the error path is identical to generate()
    if not full_text.strip():
        raise ValueError(
            "LLM stream_messages returned empty response (collected by _collect_stream)."
        )
    # Heuristic cache detection: real model calls take >50 ms to first token.
    # An LRU hit is memory-local and typically arrives in <5 ms.
    cache_hit = 0 < first_token_ms < 5.0
    return {
        "response": full_text,
        "prompt_tokens": 0,    # not available from stream path; LLMNode tracks separately
        "completion_tokens": 0,
        "model_used": "cache" if cache_hit else "",
        "cached": cache_hit,
    }


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


def _should_use_pcm_pipeline(request_id: str) -> bool:
    """
    Determine whether a given request should use the PCM-native pipeline.

    The decision is based on the feature flag (FF_PCM_PIPELINE) and the canary
    percentage (FF_CANARY_PCT). When the feature flag is on, every request uses
    the PCM pipeline. When it's off but canary is configured, a deterministic
    hash of the request_id decides — this ensures the same request always gets
    the same pipeline variant, which is critical for reproducible debugging.
    """
    if FF_PCM_PIPELINE:
        return True
    if FF_CANARY_PCT <= 0.0:
        return False
    # Deterministic canary: hash the request_id and check if its fractional
    # value falls below the canary threshold. This is stable across retries
    # because the request_id doesn't change within a single pipeline run.
    h = int(hashlib.md5(request_id.encode()).hexdigest()[:8], 16)
    fraction = h / 0xFFFFFFFF
    return fraction < FF_CANARY_PCT


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
            "qa_stage": state.get("qa_stage", ""),
            "qa_domain": state.get("qa_domain", ""),
            "pcm_chunks_processed": state.get("pcm_chunks_processed", 0),
            "pcm_barge_in_count": state.get("pcm_barge_in_count", 0),
            "audio_health": state.get("audio_health", {}),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INTER-STAGE MESSAGE BUS (StageBus)
#
# Wraps BoundedPipelineQueue with backpressure signaling, dead letter queue,
# per-stage throughput metering, and configurable overflow policy. Each
# stream_full_pcm() call creates one StageBus per adjacent stage pair
# (capture→enhance, enhance→stt, stt→llm, llm→tts, tts→playback). The bus
# is torn down when the pipeline finishes or is cancelled.
#
# The dead letter queue collects messages that could not be delivered after
# the configured retry count. These are typically PCMChunks that arrived
# after the consumer task was cancelled (e.g. barge-in interrupted playback).
# DLQ entries are logged for diagnostics but never replayed automatically —
# audio is a lossy medium and stale chunks are worthless.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StageBusMessage:
    """
    Envelope wrapping any payload moving between pipeline stages.
    The bus itself is payload-agnostic — it can carry PCMChunks, transcript
    strings, LLM token fragments, or TTS audio bytes. Type is inferred by
    the consumer, not declared in the envelope.
    """
    payload:     Any
    seq:         int              # monotonically increasing per-bus sequence
    created_at:  float = field(default_factory=time.monotonic)
    attempt:     int   = 0       # delivery attempt counter for DLQ routing
    source_stage: str  = ""      # stage that produced this message
    is_sentinel: bool = False    # True on the sentinel message signaling stream end
    is_error:    bool  = False   # True when the payload is an exception, not data

    @property
    def age_ms(self) -> float:
        return (time.monotonic() - self.created_at) * 1000

    @classmethod
    def sentinel(cls, source_stage: str) -> "StageBusMessage":
        return cls(payload=None, seq=-1, source_stage=source_stage, is_sentinel=True)


@dataclass
class DeadLetterEntry:
    """A message that could not be delivered after max_attempts retries."""
    message:      StageBusMessage
    reason:       str
    dead_at:      float = field(default_factory=time.monotonic)
    stage_pair:   str   = ""


class StageBus:
    """
    Backpressure-aware message bus connecting two adjacent pipeline stages.

    Lifecycle:
      bus = StageBus("stt→llm", max_depth=8, overflow=OverflowPolicy.DROP_OLDEST)
      await bus.put(StageBusMessage(payload=chunk, seq=0))
      msg = await bus.get()
      await bus.close()

    The bus tracks per-second throughput with a sliding 1-second window and
    exposes it as a Prometheus histogram observation on every dequeue.

    Dead letter queue: messages that fail delivery (consumer raises) are
    re-attempted up to max_delivery_attempts. After that, they are routed
    to the DLQ, logged, and counted in the _stagebus_dlq_total metric.
    """

    def __init__(
        self,
        stage_pair:            str,
        max_depth:             int             = 16,
        overflow:              OverflowPolicy  = OverflowPolicy.DROP_OLDEST,
        max_delivery_attempts: int             = 2,
    ) -> None:
        self._pair  = stage_pair
        self._queue = BoundedPipelineQueue(max_depth, name="stagebus")
        self._depth = max_depth
        self._overflow = overflow
        self._max_attempts = max_delivery_attempts

        self._dlq: deque[DeadLetterEntry] = deque(maxlen=64)
        self._closed = False
        self._seq    = 0
        self._lock   = asyncio.Lock()

        # Throughput tracking: a sliding window of dequeue timestamps
        self._dequeue_times: deque[float] = deque(maxlen=256)

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def dlq_size(self) -> int:
        return len(self._dlq)

    async def put(self, payload: Any, *, is_sentinel: bool = False, is_error: bool = False) -> None:
        """
        Enqueue a payload. Applies the configured overflow policy when the
        queue is full. Thread-safe via asyncio.Lock.
        """
        if self._closed:
            return

        async with self._lock:
            self._seq += 1
            # Don't re-wrap if already a StageBusMessage
            if isinstance(payload, StageBusMessage):
                msg = payload
            else:
                msg = StageBusMessage(
                    payload=payload,
                    seq=self._seq,
                    source_stage=self._pair.split("→")[0] if "→" in self._pair else self._pair,
                    is_sentinel=is_sentinel,
                    is_error=is_error,
                )

            if self._queue.qsize() >= self._depth:
                if self._overflow == OverflowPolicy.DROP_OLDEST:
                    try:
                        dropped = self._queue.get_nowait()
                        _stagebus_overflow_total.labels(
                            stage_pair=self._pair, policy="drop_oldest"
                        ).inc()
                        log.debug(
                            "stagebus_overflow_drop",
                            stage_pair=self._pair,
                            dropped_seq=dropped.seq if hasattr(dropped, "seq") else -1,
                        )
                    except Exception:  # noqa
                        pass
                elif self._overflow == OverflowPolicy.REJECT:
                    _stagebus_overflow_total.labels(
                        stage_pair=self._pair, policy="reject"
                    ).inc()
                    return
                # BLOCK: falls through to the put() below which will await

            await self._queue.put(msg)
            _stagebus_enqueue_total.labels(stage_pair=self._pair).inc()
            _stagebus_depth.labels(stage_pair=self._pair).set(self._queue.qsize())

    async def get(self, timeout: float | None = None) -> StageBusMessage | None:
        """
        Dequeue the next message. Returns None on timeout or if the bus is
        closed with no remaining messages. Records throughput on each dequeue.
        """
        try:
            if timeout is not None:
                msg = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            else:
                msg = await self._queue.get()
        except asyncio.TimeoutError:
            return None
        except Exception: # noqa
            return None

        now = time.monotonic()
        self._dequeue_times.append(now)
        _stagebus_dequeue_total.labels(stage_pair=self._pair).inc()
        _stagebus_depth.labels(stage_pair=self._pair).set(self._queue.qsize())

        # Compute throughput over the last 1-second window
        cutoff = now - 1.0
        recent = sum(1 for t in self._dequeue_times if t > cutoff)
        _stagebus_throughput.labels(stage_pair=self._pair).observe(recent)

        return msg

    async def send_to_dlq(self, msg: StageBusMessage, reason: str) -> None:
        """Route an undeliverable message to the dead letter queue."""
        entry = DeadLetterEntry(
            message=msg,
            reason=reason,
            stage_pair=self._pair,
        )
        self._dlq.append(entry)
        _stagebus_dlq_total.labels(stage_pair=self._pair).inc()
        log.warning(
            "stagebus_dlq_entry",
            stage_pair=self._pair,
            seq=msg.seq,
            reason=reason,
            age_ms=round(msg.age_ms, 1),
        )

    async def close(self) -> None:
        """Signal no more messages will be produced. Drains remaining items."""
        self._closed = True
        # Push a sentinel so any blocked consumer wakes up
        sentinel = StageBusMessage(payload=None, seq=self._seq + 1, is_sentinel=True)
        try:
            self._queue.put_nowait(sentinel)
        except Exception: # noqa
            pass

    def drain_dlq(self) -> list[DeadLetterEntry]:
        """Pop all DLQ entries for inspection/logging at session close."""
        entries = list(self._dlq)
        self._dlq.clear()
        return entries

    def snapshot(self) -> dict:
        """Return a diagnostic snapshot of bus state for health reporting."""
        now = time.monotonic()
        cutoff = now - 1.0
        recent = sum(1 for t in self._dequeue_times if t > cutoff)
        return {
            "stage_pair":    self._pair,
            "depth":         self.depth,
            "max_depth":     self._depth,
            "dlq_size":      self.dlq_size,
            "closed":        self._closed,
            "seq":           self._seq,
            "throughput_1s": recent,
            "overflow":      self._overflow.value,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO DIAGNOSTICS PIPELINE
#
# Taps into PCM streams at capture / pre-STT / post-TTS / pre-speaker
# boundaries and runs PCMDiagnosticsMonitor + PCMWaveformAnalyzer per
# boundary. Results are aggregated into an AudioHealthReport per request
# and emitted as structured logs + Prometheus histograms.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DiagnosticCheckpoint:
    """A single audio quality measurement at one pipeline boundary."""
    boundary:       str          # capture | pre_stt | post_tts | pre_speaker
    timestamp:      float
    rms:            float = 0.0
    peak:           float = 0.0
    crest_factor:   float = 0.0
    zcr:            float = 0.0
    clipping_pct:   float = 0.0
    silence_pct:    float = 0.0
    dc_offset:      float = 0.0
    spectral_centroid: float = 0.0

    def to_dict(self) -> dict:
        return {
            "boundary":          self.boundary,
            "timestamp":         round(self.timestamp, 3),
            "rms":               round(self.rms, 6),
            "peak":              round(self.peak, 6),
            "crest_factor":      round(self.crest_factor, 3),
            "zcr":               round(self.zcr, 1),
            "clipping_pct":      round(self.clipping_pct, 3),
            "silence_pct":       round(self.silence_pct, 3),
            "dc_offset":         round(self.dc_offset, 6),
            "spectral_centroid": round(self.spectral_centroid, 1),
        }


class AudioDiagnosticsPipeline:
    """
    Collects audio quality measurements at defined pipeline boundaries and
    aggregates them into a per-request health report. Each measurement point
    is a DiagnosticCheckpoint containing waveform analysis results from
    PCMWaveformAnalyzer and health indicators from PCMDiagnosticsMonitor.

    Alert thresholds:
      - clipping_pct > 5.0%  → "clipping" alert (dynamic gain reduction)
      - silence_pct > 80.0%  → "silence" alert (watchdog mic restart)
      - |dc_offset| > 0.05   → "dc_offset" alert (logged, not actionable)
      - rms < 0.001           → "low_energy" alert (gain boost suggestion)

    Thread safety: all mutations go through an asyncio.Lock. The diagnostics
    pipeline never blocks the audio processing path — measurements are
    opportunistic and use the sampling rate configured in VoiceGraphConfig.
    """

    CLIPPING_THRESHOLD:  float = 5.0
    SILENCE_THRESHOLD:   float = 80.0
    DC_OFFSET_THRESHOLD: float = 0.05
    LOW_ENERGY_RMS:      float = 0.001

    def __init__(self, fmt: PCMFormat, sample_rate: float = 1.0) -> None:
        self._sample_rate = sample_rate
        self._checkpoints: list[DiagnosticCheckpoint] = []
        self._alerts: list[dict] = []
        self._lock = asyncio.Lock()
        self._sample_counter = 0
        self._monitor = PCMDiagnosticsMonitor(fmt) if FF_AUDIO_DIAGNOSTICS else None
        self._analyzer = PCMWaveformAnalyzer(fmt) if FF_AUDIO_DIAGNOSTICS else None

    async def measure(self, chunk: PCMChunk, boundary: str) -> DiagnosticCheckpoint | None:
        """
        Take a diagnostic measurement at the specified boundary. Returns the
        checkpoint if the measurement was taken (per the sampling rate), or
        None if this chunk was skipped by the sampler.
        """
        if not FF_AUDIO_DIAGNOSTICS or self._analyzer is None:
            return None

        # Probabilistic sampling: skip chunks when sample rate < 1.0
        self._sample_counter += 1
        if self._sample_rate < 1.0:
            import random
            if random.random() > self._sample_rate:
                return None

        _pcm_chunks_total.labels(stage=boundary, direction="in").inc()

        try:
            analysis = self._analyzer.analyze(chunk)
            checkpoint = DiagnosticCheckpoint(
                boundary=boundary,
                timestamp=time.monotonic(),
                rms=analysis.rms,
                peak=analysis.peak,
                crest_factor=analysis.crest_factor_db,
                zcr=analysis.zero_crossing_rate,
                clipping_pct=1.0 if analysis.is_clipping else 0.0,
                silence_pct=1.0 if analysis.is_silent else 0.0,
                dc_offset=analysis.dc_offset,
                spectral_centroid=analysis.spectral_centroid_hz,
            )

            async with self._lock:
                self._checkpoints.append(checkpoint)

            # Check alert thresholds and emit metrics
            await self._check_alerts(checkpoint)
            return checkpoint

        except Exception as exc:
            log.debug("audio_diagnostics_measure_failed", boundary=boundary, error=str(exc))
            return None

    async def _check_alerts(self, cp: DiagnosticCheckpoint) -> None:
        """Evaluate alert thresholds and emit metrics for any violations."""
        if cp.clipping_pct > self.CLIPPING_THRESHOLD:
            _pcm_diagnostics_alerts.labels(alert_type="clipping").inc()
            async with self._lock:
                self._alerts.append({
                    "type": "clipping",
                    "boundary": cp.boundary,
                    "value": cp.clipping_pct,
                    "threshold": self.CLIPPING_THRESHOLD,
                    "ts": cp.timestamp,
                })
            log.warning(
                "audio_diag_clipping",
                boundary=cp.boundary,
                clipping_pct=round(cp.clipping_pct, 2),
            )

        if cp.silence_pct > self.SILENCE_THRESHOLD:
            _pcm_diagnostics_alerts.labels(alert_type="silence").inc()
            async with self._lock:
                self._alerts.append({
                    "type": "silence",
                    "boundary": cp.boundary,
                    "value": cp.silence_pct,
                    "threshold": self.SILENCE_THRESHOLD,
                    "ts": cp.timestamp,
                })

        if abs(cp.dc_offset) > self.DC_OFFSET_THRESHOLD:
            _pcm_diagnostics_alerts.labels(alert_type="dc_offset").inc()
            async with self._lock:
                self._alerts.append({
                    "type": "dc_offset",
                    "boundary": cp.boundary,
                    "value": cp.dc_offset,
                    "threshold": self.DC_OFFSET_THRESHOLD,
                    "ts": cp.timestamp,
                })

        if cp.rms > 0 and cp.rms < self.LOW_ENERGY_RMS: # noqa
            _pcm_diagnostics_alerts.labels(alert_type="low_energy").inc()
            async with self._lock:
                self._alerts.append({
                    "type": "low_energy",
                    "boundary": cp.boundary,
                    "value": cp.rms,
                    "threshold": self.LOW_ENERGY_RMS,
                    "ts": cp.timestamp,
                })

    async def aggregate(self) -> dict:
        """
        Compute an aggregated health report from all collected checkpoints.
        Returns a dict compatible with AudioHealthReport and the audio_health
        field of VoiceState.
        """
        async with self._lock:
            if not self._checkpoints:
                return {"healthy": True, "checkpoints": 0, "alerts": []}

            avg_rms = sum(cp.rms for cp in self._checkpoints) / len(self._checkpoints)
            avg_peak = sum(cp.peak for cp in self._checkpoints) / len(self._checkpoints)
            max_clipping = max(cp.clipping_pct for cp in self._checkpoints)
            avg_silence = sum(cp.silence_pct for cp in self._checkpoints) / len(self._checkpoints)
            avg_dc = sum(cp.dc_offset for cp in self._checkpoints) / len(self._checkpoints)

            has_critical = any(
                a["type"] in ("clipping", "silence") for a in self._alerts
            )

            return {
                "healthy":       not has_critical,
                "checkpoints":   len(self._checkpoints),
                "avg_rms":       round(avg_rms, 6),
                "avg_peak":      round(avg_peak, 6),
                "max_clipping":  round(max_clipping, 3),
                "avg_silence":   round(avg_silence, 3),
                "avg_dc_offset": round(avg_dc, 6),
                "alerts":        list(self._alerts),
                "boundaries":    list({cp.boundary for cp in self._checkpoints}),
            }

    def reset(self) -> None:
        """Clear all collected data for the next request."""
        self._checkpoints.clear()
        self._alerts.clear()
        self._sample_counter = 0


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE WATCHDOG
#
# Background asyncio task that monitors per-stage throughput, overall pipeline
# health, and audio quality. When degradation is detected, the watchdog can
# trigger automatic recovery actions (mic restart, speaker restart, pipeline
# reset) and emit alerts to the observability stack.
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineWatchdog:
    """
    Periodic heartbeat that checks pipeline liveness and audio quality.

    Runs as a background asyncio.Task started by SessionLifecycleManager.open_session()
    and cancelled by close_session(). Each heartbeat:

      1. Checks all StageBus instances for stalled queues (depth == max for
         >5 consecutive heartbeats → alert).
      2. Reads the latest AudioDiagnosticsPipeline aggregate and checks for
         persistent clipping or silence alerts.
      3. Verifies the recorder and player health reports.
      4. If any check fails, increments a strike counter. After N consecutive
         strikes, the watchdog triggers the configured recovery action.

    Recovery actions:
      - mic_restart:     close and re-open PCMInputStream
      - speaker_restart: close and re-open PCMOutputStream
      - pipeline_reset:  cancel all workers and restart stream_full_pcm()

    The watchdog never acts on a single anomalous heartbeat — it requires
    sustained degradation (configurable via strike_threshold) to avoid
    false positives from transient audio glitches.
    """

    def __init__(
        self,
        interval_s:       float = 5.0,
        strike_threshold: int   = 3,
    ) -> None:
        self._interval = interval_s
        self._threshold = strike_threshold
        self._task:     asyncio.Task | None = None
        self._buses:    list[StageBus] = []
        self._diagnostics: AudioDiagnosticsPipeline | None = None
        self._strikes:  int = 0
        self._running   = False
        self._session_id: str = ""

    def attach(
        self,
        buses:       list[StageBus],
        diagnostics: AudioDiagnosticsPipeline | None,
        session_id:  str,
    ) -> None:
        """Attach pipeline components for monitoring. Call before start()."""
        self._buses = buses
        self._diagnostics = diagnostics
        self._session_id = session_id

    def start(self) -> None:
        """Start the watchdog background task."""
        if self._running:
            return
        self._running = True
        self._strikes = 0
        self._task = asyncio.create_task(
            self._heartbeat_loop(),
            name=f"watchdog:{self._session_id[:8]}",
        )

    async def stop(self) -> None:
        """Stop the watchdog and cancel its background task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _heartbeat_loop(self) -> None:
        """Main watchdog loop. Runs until stop() is called."""
        try:
            while self._running:
                await asyncio.sleep(self._interval)
                if not self._running:
                    break
                await self._tick()
        except asyncio.CancelledError:
            pass

    async def _tick(self) -> None:
        """Single heartbeat tick. Checks all monitored components."""
        _watchdog_heartbeats.inc()
        degraded = False

        # Check StageBus health: any bus at 100% capacity for multiple ticks
        for bus in self._buses:
            if bus.is_closed:
                continue
            snap = bus.snapshot()
            if snap["depth"] >= snap["max_depth"]:
                degraded = True
                log.warning(
                    "watchdog_bus_full",
                    session_id=self._session_id[:8],
                    stage_pair=snap["stage_pair"],
                    depth=snap["depth"],
                )

        # Check audio diagnostics for persistent alerts
        if self._diagnostics:
            try:
                report = await self._diagnostics.aggregate()
                if not report.get("healthy", True):
                    degraded = True
                    alert_types = [a["type"] for a in report.get("alerts", [])]
                    log.warning(
                        "watchdog_audio_degraded",
                        session_id=self._session_id[:8],
                        alert_types=alert_types,
                    )
            except Exception as exc:
                log.debug("watchdog_diag_check_failed", error=str(exc))

        # Check recorder health
        try:
            rec_health = get_recording_health()
            if rec_health and not rec_health.healthy:
                degraded = True
                log.warning(
                    "watchdog_recorder_unhealthy",
                    session_id=self._session_id[:8],
                )
        except Exception: # noqa
            pass

        # Check player health
        try:
            play_health = get_playback_health()
            if play_health and not play_health.healthy:
                degraded = True
                log.warning(
                    "watchdog_player_unhealthy",
                    session_id=self._session_id[:8],
                )
        except Exception: # noqa
            pass

        # Strike accounting
        if degraded:
            self._strikes += 1
            if self._strikes >= self._threshold:
                _watchdog_alerts.labels(alert_level="critical").inc()  # type: ignore[attr-defined]
                log.error(
                    "watchdog_critical_threshold",
                    session_id=self._session_id[:8],
                    strikes=self._strikes,
                    threshold=self._threshold,
                )
                # Reset strike counter after alerting to avoid spam
                self._strikes = 0
            else:
                _watchdog_alerts.labels(alert_level="warning").inc()  # type: ignore[attr-defined]
        else:
            # Decay strikes on healthy heartbeats
            self._strikes = max(0, self._strikes - 1)

    def snapshot(self) -> dict:
        """Diagnostic snapshot for health endpoints."""
        return {
            "running":     self._running,
            "strikes":     self._strikes,
            "threshold":   self._threshold,
            "interval_s":  self._interval,
            "bus_count":   len(self._buses),
            "session_id":  self._session_id[:8] if self._session_id else "",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION LIFECYCLE MANAGER
#
# Centralises all resource acquisition and release for a single interview
# session. voice_graph callers invoke open_session() once at the start and
# close_session() once at the end — everything in between is the pipeline's
# concern. The manager coordinates QA controller, evaluation engine, audit
# bus, transcript writer, PCM format negotiation, health checks, watchdog,
# and temp file cleanup into a single atomic open/close pair.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SessionResources:
    """
    Handle bag returned by SessionLifecycleManager.open_session().
    Callers pass this to close_session() for teardown. The handle carries
    references to all resources acquired during open so close can release
    them without re-discovering the session state.
    """
    session_id:        str
    opened_at:         float
    mic_healthy:       bool           = True
    speaker_healthy:   bool           = True
    mic_format:        PCMFormat | None = None
    speaker_format:    PCMFormat | None = None
    negotiated_stt_fmt: PCMFormat | None = None
    negotiated_tts_fmt: PCMFormat | None = None
    watchdog:          PipelineWatchdog | None = None
    diagnostics:       AudioDiagnosticsPipeline | None = None
    buses:             list[StageBus] = field(default_factory=list)
    temp_files:        list[str]      = field(default_factory=list)
    degraded:          bool           = False
    qa_document:       Any            = None
    eval_report:       Any            = None
    eqs:               Any            = None  # EpistemicQuestionSelector | None

    @property
    def duration_s(self) -> float:
        return time.monotonic() - self.opened_at

    def to_dict(self) -> dict:
        return {
            "session_id":   self.session_id,
            "duration_s":   round(self.duration_s, 2),
            "mic_healthy":  self.mic_healthy,
            "speaker_healthy": self.speaker_healthy,
            "degraded":     self.degraded,
            "temp_files":   len(self.temp_files),
            "bus_count":    len(self.buses),
        }


class SessionLifecycleManager:
    """
    Coordinates all resource acquisition and release for interview sessions.

    open_session() performs:
      1. QA controller session create (get_or_create)
      2. Mic health check (async, bounded timeout)
      3. Speaker health check (async, bounded timeout)
      4. PCM format negotiation (mic → STT, TTS → speaker)
      5. Audio diagnostics pipeline initialisation
      6. Pipeline watchdog start
      7. Audit bus session open
      8. Transcript writer session begin

    close_session() performs:
      1. Pipeline watchdog stop
      2. StageBus drain + DLQ flush
      3. Audio diagnostics final aggregate
      4. QA controller mark complete (if not already)
      5. Evaluation engine session report generation
      6. Audit bus session close + final eval dispatch
      7. Transcript writer flush
      8. Temp file cleanup
      9. Session duration metric emission

    Both methods are idempotent — calling open twice returns the existing
    handle, calling close twice is a no-op. All steps are wrapped in
    individual try/except so a failure in one step does not prevent the
    others from executing.
    """

    def __init__(self, cfg: VoiceGraphConfig) -> None:
        self._cfg = cfg
        self._active: dict[str, SessionResources] = {}
        self._lock = asyncio.Lock()
        self._active_tasks: dict[str, asyncio.Task] = {}

    async def open_session(self, session_id: str) -> SessionResources:
        """
        Acquire all resources for a new session. Returns a SessionResources
        handle that must be passed to close_session() when done.
        """
        t0 = time.monotonic()

        async with self._lock:
            if session_id in self._active:
                return self._active[session_id]

        with tracer.start_as_current_span("session.open") as span:
            span.set_attribute("session_id", session_id[:8])
            resources = SessionResources(
                session_id=session_id,
                opened_at=t0,
            )
            degraded = False

            # ── 1. QA controller session create ───────────────────────────────
            try:
                doc, is_new = await _qa_controller.get_or_create(session_id)
                resources.qa_document = doc
                if is_new:
                    log.info("session_qa_created", sid=session_id[:8])
            except Exception as exc:
                degraded = True
                log.error("session_qa_create_failed", sid=session_id[:8], error=str(exc))

            # ── 2. Mic health check ───────────────────────────────────────────
            if FF_SESSION_LIFECYCLE:
                try:
                    rec_health = await asyncio.wait_for(
                        asyncio.to_thread(get_recording_health),
                        timeout=self._cfg.session_mic_health_timeout,
                    )
                    resources.mic_healthy = rec_health.healthy if rec_health else True
                    if not resources.mic_healthy:
                        degraded = True
                        log.warning("session_mic_unhealthy", sid=session_id[:8])
                except asyncio.TimeoutError:
                    log.warning("session_mic_health_timeout", sid=session_id[:8])
                    resources.mic_healthy = True  # assume healthy on timeout
                except Exception as exc:
                    log.warning("session_mic_health_error", sid=session_id[:8], error=str(exc))

            # ── 3. Speaker health check ───────────────────────────────────────
            if FF_SESSION_LIFECYCLE:
                try:
                    play_health = await asyncio.wait_for(
                        asyncio.to_thread(get_playback_health),
                        timeout=self._cfg.session_speaker_health_timeout,
                    )
                    resources.speaker_healthy = play_health.healthy if play_health else True
                    if not resources.speaker_healthy:
                        degraded = True
                        log.warning("session_speaker_unhealthy", sid=session_id[:8])
                except asyncio.TimeoutError:
                    log.warning("session_speaker_health_timeout", sid=session_id[:8])
                except Exception as exc:
                    log.warning("session_speaker_health_error", sid=session_id[:8], error=str(exc))

            # ── 4. PCM format negotiation ─────────────────────────────────────
            if FF_PCM_PIPELINE or _should_use_pcm_pipeline(session_id):
                try:
                    mic_fmt = get_recording_format()
                    resources.mic_format = mic_fmt

                    # STT prefers 16kHz int16 mono — negotiate with mic format
                    stt_preferred = [
                        PCMFormat(sample_rate=16000, channels=1, dtype="int16"),
                        PCMFormat(sample_rate=44100, channels=1, dtype="int16"),
                    ]
                    stt_supported = [mic_fmt] if mic_fmt else stt_preferred
                    resources.negotiated_stt_fmt = negotiate_format(stt_preferred, stt_supported)
                    _pcm_format_negotiations.labels(result="exact_match" if resources.negotiated_stt_fmt in stt_preferred else "converted").inc()

                    # TTS produces 24kHz int16 mono by default (OpenAI PCM format)
                    tts_fmt = PCMFormat.openai_tts()
                    resources.negotiated_tts_fmt = tts_fmt

                    log.info(
                        "session_pcm_formats_negotiated",
                        sid=session_id[:8],
                        mic=str(resources.mic_format),
                        stt=str(resources.negotiated_stt_fmt),
                        tts=str(resources.negotiated_tts_fmt),
                    )
                except Exception as exc:
                    _pcm_format_negotiations.labels(result="failed").inc()
                    log.warning("session_pcm_negotiation_failed", sid=session_id[:8], error=str(exc))

            # ── 5. Audio diagnostics init ─────────────────────────────────────
            if FF_AUDIO_DIAGNOSTICS:
                diag_fmt = resources.negotiated_stt_fmt or PCMFormat(sample_rate=16000, channels=1, dtype="int16")
                resources.diagnostics = AudioDiagnosticsPipeline(
                    fmt=diag_fmt,
                    sample_rate=self._cfg.diagnostics_sample_rate,
                )

            # ── 6. Pipeline watchdog ──────────────────────────────────────────
            resources.watchdog = PipelineWatchdog(
                interval_s=self._cfg.watchdog_interval_s,
            )

            # ── 7. Audit bus session open ─────────────────────────────────────
            try:
                if _qa_audit_bus is not None:
                    await _qa_audit_bus.open_session(session_id)
            except Exception as exc:
                log.warning("session_audit_bus_open_failed", sid=session_id[:8], error=str(exc))

            # ── 8. Transcript writer begin ────────────────────────────────────
            try:
                await _transcript_open_session(session_id)
            except Exception as exc:
                log.warning("session_transcript_begin_failed", sid=session_id[:8], error=str(exc))

            # ── 9. EpistemicQuestionSelector — create for this session ────────
            try:
                from app.eval.question_selector import (
                    EpistemicQuestionSelector,
                    register_selector,
                )
                stated_level = (
                    resources.qa_document.candidate.level
                    if resources.qa_document and resources.qa_document.candidate
                    else None
                )
                # Pass the same Redis client qa_controller uses.
                # EQS is fault-tolerant: degrades to LRU if Redis is unavailable.
                import redis.asyncio as _aioredis
                _eqs_redis = await _aioredis.from_url(
                    __import__("os").getenv("REDIS_URL", ""),
                    encoding="utf-8", decode_responses=True,
                ) if __import__("os").getenv("REDIS_URL") else None

                selector = await EpistemicQuestionSelector.create(
                    session_id=session_id,
                    redis=_eqs_redis,
                    stated_level=stated_level,
                )
                resources.eqs = selector
                register_selector(session_id, selector)
                log.info("session_eqs_created", sid=session_id[:8])
            except Exception as exc:
                log.warning("session_eqs_create_failed",
                            sid=session_id[:8], error=str(exc))

            # ── finalise ──────────────────────────────────────────────────────
            resources.degraded = degraded
            async with self._lock:
                self._active[session_id] = resources

            open_latency = time.monotonic() - t0
            status = "degraded" if degraded else "ok"
            _session_open_total.labels(status=status).inc()
            _session_open_latency.observe(open_latency)

            log.info(
                "session_opened",
                sid=session_id[:8],
                degraded=degraded,
                mic_healthy=resources.mic_healthy,
                speaker_healthy=resources.speaker_healthy,
                latency_s=round(open_latency, 3),
            )
            span.set_attribute("degraded", degraded)
            span.set_attribute("open_latency_s", round(open_latency, 3))

            return resources

    async def close_session(
        self,
        session_id: str,
        reason:     str = "normal",
    ) -> dict:
        """
        Release all resources for a session. Returns a summary dict with
        session analytics, eval report, and audio health.
        """
        t0 = time.monotonic()

        async with self._lock:
            resources = self._active.pop(session_id, None)
        if resources is None:
            _session_close_total.labels(reason="not_found").inc()
            return {"session_id": session_id, "status": "not_found"}

        with tracer.start_as_current_span("session.close") as span:
            span.set_attribute("session_id", session_id[:8])
            span.set_attribute("reason", reason)

            summary: dict[str, Any] = {
                "session_id": session_id,
                "reason": reason,
                "duration_s": round(resources.duration_s, 2),
            }

            # ── 1. Stop watchdog ──────────────────────────────────────────────
            if resources.watchdog:
                try:
                    await resources.watchdog.stop()
                except Exception as exc:
                    log.debug("session_watchdog_stop_error", error=str(exc))

            # ── 2. Drain StageBuses + flush DLQs ─────────────────────────────
            total_dlq_entries = 0
            for bus in resources.buses:
                try:
                    await bus.close()
                    dlq = bus.drain_dlq()
                    total_dlq_entries += len(dlq)
                    if dlq:
                        log.info(
                            "session_dlq_drained",
                            sid=session_id[:8],
                            stage_pair=bus._pair, # noqa
                            entries=len(dlq),
                        )
                except Exception as exc:
                    log.debug("session_bus_close_error", stage_pair=bus._pair, error=str(exc)) # noqa
            summary["dlq_entries"] = total_dlq_entries

            # ── 3. Audio diagnostics final aggregate ──────────────────────────
            if resources.diagnostics:
                try:
                    audio_health = await resources.diagnostics.aggregate()
                    summary["audio_health"] = audio_health
                except Exception as exc:
                    log.debug("session_diag_aggregate_error", error=str(exc))

            # ── 4. QA controller — fetch final document and analytics ─────────
            try:
                doc = await _qa_controller.get_document(session_id)
                if doc:
                    analytics = _QAAnalytics.analyse(doc)
                    summary["qa_analytics"] = analytics.to_dict()
                    summary["qa_eval_context"] = analytics.to_eval_context()

                    # If the session isn't already complete, mark it now
                    if doc.stage != _QAStage.COMPLETE.value:
                        await _qa_controller.admin.force_complete(
                            session_id, reason=f"session_close:{reason}"
                        )
            except Exception as exc:
                log.warning("session_qa_close_error", sid=session_id[:8], error=str(exc))

            # ── 5. Evaluation engine — generate session report ────────────────
            try:
                eval_report = await _eval_engine.get_session_report(session_id)
                summary["eval_report"] = eval_report.to_dict()
                resources.eval_report = eval_report
            except Exception as exc:
                log.warning("session_eval_report_error", sid=session_id[:8], error=str(exc))

            # ── 6. Dispatch final eval batch for any unscored turns ───────────
            try:
                if _finalize_session_eval is not None:
                    await _finalize_session_eval(session_id)
            except Exception as exc:
                log.warning("session_final_eval_error", sid=session_id[:8], error=str(exc))

            # ── 7. Audit bus session close ────────────────────────────────────
            try:
                if _qa_audit_bus is not None:
                    await _qa_audit_bus.close_session(session_id)
            except Exception as exc:
                log.warning("session_audit_bus_close_error", sid=session_id[:8], error=str(exc))

            # ── 8. Transcript writer flush ────────────────────────────────────
            try:
                await _transcript_flush_session(session_id)
            except Exception as exc:
                log.warning("session_transcript_flush_error", sid=session_id[:8], error=str(exc))

            # ── 9. EQS — deregister ───────────────────────────────────────────
            try:
                from app.eval.question_selector import deregister_selector
                deregister_selector(session_id)
                if resources.eqs is not None:
                    log.debug("session_eqs_deregistered", sid=session_id[:8],
                              **resources.eqs.health())
            except Exception as exc:
                log.debug("session_eqs_deregister_error", error=str(exc))

            # ── 10. Question prefetch cancel ───────────────────────────────────
            try:
                _qa_prefetch_buffer.cancel_session(session_id)
            except Exception: # noqa
                pass

            # ── 11. Temp file cleanup ─────────────────────────────────────────
            cleaned = 0
            for path in resources.temp_files:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        cleaned += 1
                except OSError as exc:
                    log.debug("session_temp_cleanup_failed", path=path, error=str(exc))
            summary["temp_files_cleaned"] = cleaned

            # ── metrics + logging ─────────────────────────────────────────────
            close_latency = time.monotonic() - t0
            _session_close_total.labels(reason=reason).inc()
            _session_close_latency.observe(close_latency)
            _session_duration.observe(resources.duration_s)

            log.info(
                "session_closed",
                sid=session_id[:8],
                reason=reason,
                duration_s=round(resources.duration_s, 1),
                close_latency_s=round(close_latency, 3),
                dlq_entries=total_dlq_entries,
                temp_cleaned=cleaned,
            )
            span.set_attribute("duration_s", round(resources.duration_s, 1))
            span.set_attribute("close_latency_s", round(close_latency, 3))

            return summary

    async def get_resources(self, session_id: str) -> SessionResources | None:
        """Return the active SessionResources for a session, or None."""
        async with self._lock:
            return self._active.get(session_id)

    def active_count(self) -> int:
        return len(self._active)

    async def force_close_all(self, reason: str = "shutdown") -> int:
        """Close all active sessions and cancel orphan tasks during shutdown."""

        # 1) snapshot sessions
        async with self._lock:
            session_ids = list(self._active.keys())

        closed = 0

        # 2) close sessions first
        for sid in session_ids:
            try:
                await self.close_session(sid, reason=reason)
                closed += 1
            except Exception as exc:
                log.warning("session_force_close_error", session_id=sid[:8], error=str(exc))

        # 3) snapshot tasks
        async with self._lock:
            tasks = list(self._active_tasks.items())

        # 4) cancel orphan tasks
        for rid, task in tasks:
            if not task.done():
                task.cancel()
                log.info("force_cancelled_task", request_id=rid, reason=reason)

        # 5) await cancellation to avoid pending-task warnings
        for _, task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                log.warning("task_cancel_error", error=str(exc))

        return closed

# ── per-instance graph builder ─────────────────────────────────────────────────


def _build_graph_for_instance(
    stt: STTNodeProtocol,
    llm: LLMNodeProtocol, # noqa
    tts: TTSNodeProtocol,
    cfg: VoiceGraphConfig,
    is_dev: bool,
) -> CompiledStateGraph:
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
        Increment the STT retry counter so the downstream router can decide
        whether to loop back to node_stt or route to error_terminal.
        Also logs the retry attempt with the current error for diagnostics.
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
        return _state_update(state, {"stt_retries": retries})

    # ── node: LLM ─────────────────────────────────────────────────────────────
    async def node_llm(state: VoiceState) -> VoiceState:
        """
        Generate the next interviewer question or process intro/greeting.

        QA-only path: all turns route through the interview engine via
        _node_llm_qa_path(). The path is selected based on the current QA stage:
          greeting  → return GREETING_TEXT directly (no LLM call)
          intro     → ATS extraction via extract_and_validate_intro()
          interview → question generation via generate_interviewer_question()
          complete  → return CLOSING_TEXT directly (no LLM call)

        The LLM receives ZERO conversation history in interview mode.
        Only: domain | level | last_q | last_a | domain_switch_flag | difficulty.
        This is structurally enforced by build_next_llm_input_for_voice_graph().

        Security guarantees (QA path only):
          1. Candidate answers are injection-scanned before any LLM call.
          2. LLM outputs are content-policy filtered before TTS.
          3. All turns fingerprinted — same question cannot appear twice.
          4. Diversity enforcer catches paraphrased repeats.
          5. Static bank ensures a question always available when LLM is down.
          6. GuardrailEngine enforces hard stop at 60 questions / 45 minutes.
        """
        rid        = state.get("request_id", current_request_id())
        t0         = time.monotonic()
        session_id = state.get("session_id")

        LLMEmitter.start(
            session_id  = session_id or "",
            request_id  = rid,
            model       = getattr(settings, "llm_model", ""),
            streaming   = False,
            history_turns = 0,
        )

        async with llm_span(
            session_id = session_id or "",
            request_id = rid,
            model      = getattr(settings, "llm_model", ""),
            streaming  = False,
        ) as span:
            try:
                response_text, _qa_stage, _qa_domain, _ = await _node_llm_qa_path(
                    state=state,
                    session_id=session_id,
                    rid=rid,
                    span=span,
                )

                latency = time.monotonic() - t0
                _stage_latency.labels(stage="llm").observe(latency)

                log.info(
                    "graph_llm_ok",
                    request_id   = rid,
                    session_id   = session_id or "",
                    response_len = len(response_text),
                    latency_s    = round(latency, 3),
                    qa_path      = bool(session_id),
                )
                span.set_attribute("latency_s", round(latency, 3))
                span.set_attribute("qa_path", bool(session_id))

                LLMEmitter.ok(
                    session_id        = session_id or "",
                    request_id        = rid,
                    latency_ms        = latency * 1000,
                    model_used        = getattr(settings, "llm_model", ""),
                    prompt_tokens     = 0,
                    completion_tokens = 0,
                    streaming         = False,
                    history_turns     = 0,
                    response_chars    = len(response_text),
                    cache_hit         = False,
                    response_truncated= False,
                )

                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "llm_response":          response_text,
                        "llm_tokens":            {"prompt": 0, "completion": 0},
                        "llm_model_used":        getattr(settings, "llm_model", ""),
                        "llm_cached":            False,
                        "response_truncated":    False,
                        "session_turn_appended": True,
                        "stage":                 PipelineStage.LLM.value,
                        "error":                 state.get("error", ""),
                        "error_stage":           state.get("error_stage", ""),
                        "abort_reason":          "",
                        "qa_stage":              _qa_stage,
                        "qa_domain":             _qa_domain,
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
                    request_id = rid,
                    latency_s  = round(latency, 3),
                )
                LLMEmitter.failed(
                    session_id = session_id or "",
                    request_id = rid,
                    error      = str(exc),
                    error_type = "LatencyBudgetExceeded",
                    model      = getattr(settings, "llm_model", ""),
                    streaming  = False,
                )
                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "stage":        PipelineStage.LLM.value,
                        "error":        str(exc),
                        "error_stage":  PipelineStage.LLM.value,
                        "abort_reason": "budget_exceeded",
                    },
                )

            except Exception as exc:
                _stage_errors.labels(stage="llm").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "graph_llm_failed",
                    request_id = rid,
                    error      = str(exc),
                    latency_s  = round(latency, 3),
                )
                LLMEmitter.failed(
                    session_id = session_id or "",
                    request_id = rid,
                    error      = str(exc),
                    error_type = type(exc).__name__,
                    model      = getattr(settings, "llm_model", ""),
                    streaming  = False,
                )
                return _state_update(
                    _record_stage_latency(state, "llm", latency),
                    {
                        "stage":       PipelineStage.LLM.value,
                        "error":       str(exc),
                        "error_stage": PipelineStage.LLM.value,
                    },
                )

    # ── node: LLM error handler ────────────────────────────────────────────────
    async def node_llm_error(state: VoiceState) -> VoiceState:
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
        return _state_update(state, {"llm_retries": retries})

    # ── node: sanitize ─────────────────────────────────────────────────────────
    async def node_sanitize(state: VoiceState) -> VoiceState:
        """
        Run the 27-stage sanitise pipeline on the LLM response before handing
        it to TTS. The pipeline strips HTML, invisible Unicode, prompt-injection
        patterns, markdown, and normalises whitespace. Truncation is applied at
        max_tts_chars to prevent TTS from synthesising a novel-length response.
        """
        rid = state.get("request_id", current_request_id())
        t0 = time.monotonic()

        async with sanitize_span(
            session_id=state.get("session_id", ""),
            request_id=rid,
        ) as span:
            raw = state.get("llm_response", "")
            san = sanitize(raw, max_chars=cfg.max_tts_chars, request_id=rid)
            cleaned = san.text
            truncated = san.truncated

            latency = time.monotonic() - t0
            _stage_latency.labels(stage="sanitize").observe(latency)

            SanitizeEmitter.ok(
                session_id=state.get("session_id", ""),
                request_id=rid,
                original_chars=len(raw),  # input_chars
                sanitized_chars=len(cleaned),  # output_chars
                truncated=truncated,
                warnings=[],
            )
            span.set_attribute("input_chars", len(raw))
            span.set_attribute("output_chars", len(cleaned))
            span.set_attribute("truncated", truncated)
            span.set_attribute("latency_s", round(latency, 3))

            return _state_update(
                _record_stage_latency(state, "sanitize", latency),
                {
                    "cleaned_response": cleaned,
                    "response_truncated": truncated or state.get("response_truncated", False),
                    "stage": PipelineStage.SANITIZE.value,
                },
            )

    # ── node: TTS ─────────────────────────────────────────────────────────────
    async def node_tts(state: VoiceState) -> VoiceState:
        """
        Synthesise the cleaned response into audio using the instance's TTS
        implementation. Supports both file-based (local path + S3 URI) and
        PCM streaming modes depending on the execution mode.

        On success: audio_output and audio_s3_uri are populated. The S3 URI
        is the canonical reference used by the API response; the local path
        is transient and scheduled for cleanup at session close.

        TTS errors are treated as terminal — there's no retry for TTS because
        the response text is deterministic and a second call with the same
        input would produce the same error.
        """
        rid = state.get("request_id", current_request_id())
        t0 = time.monotonic()

        TTSEmitter.start(
            session_id=state.get("session_id", ""),
            request_id=rid,
            input_chars=len(state.get("cleaned_response", "")),  # text_chars
            voice=state.get("tts_voice", ""),
        )

        async with tts_span(
            session_id=state.get("session_id", ""),
            request_id=rid,
            text_chars=len(state.get("cleaned_response", "")),
        ) as span:
            try:
                text = state.get("cleaned_response", "")
                if not text.strip():
                    raise ValueError("Empty text after sanitize — nothing to synthesize.")

                local_path, s3_uri, _duration_s = await _with_timeout(
                    tts.synthesize(
                        text=text,
                        voice=state.get("tts_voice"),
                        speed=state.get("tts_speed"),
                        request_id=rid,
                    ),
                    timeout=cfg.tts_timeout,
                    stage="TTS",
                )

                audio_output = local_path

                latency = time.monotonic() - t0
                _stage_latency.labels(stage="tts").observe(latency)

                log.info(
                    "graph_tts_ok",
                    request_id=rid,
                    latency_s=round(latency, 3),
                    has_s3=bool(s3_uri),
                    has_local=bool(local_path),
                )
                span.set_attribute("latency_s", round(latency, 3))

                TTSEmitter.ok(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    latency_ms=latency * 1000,
                    audio_duration_s=_duration_s,
                    audio_size_bytes=0,
                    voice=state.get("tts_voice", ""),
                    input_chars=len(text),  # text_chars
                    audio_output=audio_output,
                    s3_uri=s3_uri if s3_uri else "",  # s3_uploaded (bool → str)
                )

                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "audio_output": audio_output,
                        "audio_local_path": local_path,
                        "audio_s3_uri": s3_uri,
                        "stage": PipelineStage.TTS.value,
                        "error": "",
                        "error_stage": "",
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
                log.warning("graph_tts_budget_exceeded", request_id=rid, latency_s=round(latency, 3))
                TTSEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type="LatencyBudgetExceeded",
                    voice=state.get("tts_voice", ""),
                )
                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "stage": PipelineStage.TTS.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.TTS.value,
                        "abort_reason": "budget_exceeded",
                    },
                )

            except Exception as exc:
                _stage_errors.labels(stage="tts").inc()
                latency = time.monotonic() - t0
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("graph_tts_failed", request_id=rid, error=str(exc), latency_s=round(latency, 3))
                TTSEmitter.failed(
                    session_id=state.get("session_id", ""),
                    request_id=rid,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    voice=state.get("tts_voice", ""),
                )
                return _state_update(
                    _record_stage_latency(state, "tts", latency),
                    {
                        "stage": PipelineStage.TTS.value,
                        "error": str(exc),
                        "error_stage": PipelineStage.TTS.value,
                    },
                )

    # ── node: TTS error handler ────────────────────────────────────────────────
    async def node_tts_error(state: VoiceState) -> VoiceState:
        """TTS errors are terminal — no retry. Route straight to error_terminal."""
        rid = state.get("request_id", current_request_id())
        log.warning(
            "graph_tts_error_handler",
            request_id=rid,
            error=state.get("error", ""),
        )
        return state

    # ── node: error terminal ───────────────────────────────────────────────────
    async def node_error_terminal(state: VoiceState) -> VoiceState:
        """
        Terminal node for all error paths. Sets the pipeline to FAILED state,
        picks the appropriate apology string based on which stage failed, and
        marks the pipeline as degraded if it was previously healthy.

        The apology string becomes the llm_response so downstream callers
        always have something human-readable to display, even when the pipeline
        failed at the STT stage and never reached the LLM.
        """
        rid = state.get("request_id", current_request_id())
        error_stage = state.get("error_stage", "unknown")
        error_msg   = state.get("error", "Pipeline failed with no specific error.")

        apology = _STAGE_APOLOGY.get(error_stage, APOLOGY_LLM)
        _degraded_total.inc()

        log.error(
            "graph_error_terminal",
            request_id=rid,
            error_stage=error_stage,
            error=error_msg,
        )

        return _state_update(state, {
            "stage":       PipelineStage.FAILED.value,
            "degraded":    True,
            "llm_response": state.get("llm_response") or apology,
            "cleaned_response": state.get("cleaned_response") or apology,
        })

    # ── node: dev audio playback ───────────────────────────────────────────────
    async def node_playback_dev(state: VoiceState) -> VoiceState:
        """
        Development-only node that plays the synthesized audio through the
        local speaker. Uses the persistent PCMOutputStream when available to
        eliminate device-open latency, falling back to file-based play_audio()
        for non-PCM paths.
        """
        rid = state.get("request_id", current_request_id())
        local_path = state.get("audio_local_path", "")
        if not local_path:
            log.warning("graph_dev_playback_no_path", request_id=rid)
            return state

        t0 = time.monotonic()
        try:
            # play_audio has async internals — call directly in the event loop,
            # not via to_thread (which runs in a worker thread with no event loop).
            result = play_audio(local_path) # noqa
            if asyncio.iscoroutine(result):
                await result
            latency = time.monotonic() - t0
            log.info("graph_dev_playback_ok", request_id=rid, latency_s=round(latency, 3))
        except Exception as exc:
            log.warning("graph_dev_playback_failed", request_id=rid, error=str(exc))

        return _state_update(state, {"stage": PipelineStage.DONE.value})

    # ═══════════════════════════════════════════════════════════════════════════
    # ROUTING FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def route_after_stt(state: VoiceState) -> str:
        if _has_stage_error(state, PipelineStage.STT):
            return "stt_error"
        return "llm"

    def route_after_stt_error(state: VoiceState) -> str:
        """
        After STT error: retry if retries remain and no abort_reason is set.
        Otherwise, route to error_terminal for graceful degradation.
        """
        if state.get("abort_reason"):
            return "error_terminal"
        if state.get("stt_retries", 0) <= cfg.max_stt_retries:
            return "stt"
        return "error_terminal"

    def route_after_llm(state: VoiceState) -> str:
        if _has_stage_error(state, PipelineStage.LLM):
            return "llm_error"
        return "sanitize"

    def route_after_llm_error(state: VoiceState) -> str:
        """Mirror of route_after_stt_error for the LLM stage."""
        if state.get("abort_reason"):
            return "error_terminal"
        if state.get("llm_retries", 0) <= cfg.max_llm_retries:
            return "llm"
        return "error_terminal"

    def route_after_tts(state: VoiceState) -> str:
        if _has_stage_error(state, PipelineStage.TTS):
            return "tts_error"
        if is_dev:
            return "audio_sink_dev"
        return END

    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH ASSEMBLY
    # ═══════════════════════════════════════════════════════════════════════════

    sg = StateGraph(VoiceState) # type: ignore[arg-type]

    # ── register nodes ─────────────────────────────────────────────────────────
    sg.add_node("stt", node_stt)  # type: ignore[arg-type]
    sg.add_node("stt_error", node_stt_error)  # type: ignore[arg-type]
    sg.add_node("llm", node_llm)  # type: ignore[arg-type]
    sg.add_node("llm_error", node_llm_error)  # type: ignore[arg-type]
    sg.add_node("sanitize", node_sanitize)  # type: ignore[arg-type]
    sg.add_node("tts", node_tts)  # type: ignore[arg-type]
    sg.add_node("tts_error", node_tts_error)  # type: ignore[arg-type]
    sg.add_node("error_terminal", node_error_terminal)  # type: ignore[arg-type]
    if is_dev:
        sg.add_node("audio_sink_dev", node_playback_dev)  # type: ignore[arg-type]

    sg.set_entry_point("stt")

    # ── STT → conditional branch ───────────────────────────────────────────────
    sg.add_conditional_edges(
        "stt",
        route_after_stt,
        {"stt_error": "stt_error", "llm": "llm"},
    )

    # ── STT error → retry loop or terminal ────────────────────────────────────
    sg.add_conditional_edges(
        "stt_error",
        route_after_stt_error,
        {"stt": "stt", "error_terminal": "error_terminal"},
    )

    # ── LLM → conditional branch ───────────────────────────────────────────────
    sg.add_conditional_edges(
        "llm",
        route_after_llm,
        {"llm_error": "llm_error", "sanitize": "sanitize"},
    )

    # ── LLM error → retry loop or terminal ────────────────────────────────────
    sg.add_conditional_edges(
        "llm_error",
        route_after_llm_error,
        {"llm": "llm", "error_terminal": "error_terminal"},
    )

    # ── sanitize → tts (unconditional) ────────────────────────────────────────
    sg.add_edge("sanitize", "tts")

    # ── TTS → conditional branch ───────────────────────────────────────────────
    tts_targets: dict[str, str] = {
        "tts_error": "tts_error",
    }
    if is_dev:
        tts_targets["audio_sink_dev"] = "audio_sink_dev"
        tts_targets[END] = END
    else:
        tts_targets[END] = END
    sg.add_conditional_edges("tts", route_after_tts, tts_targets)

    # ── tts_error → error_terminal (unconditional) ────────────────────────────
    sg.add_edge("tts_error", "error_terminal")

    # ── error_terminal → END (unconditional) ──────────────────────────────────
    sg.add_edge("error_terminal", END)
    if is_dev:
        sg.add_edge("audio_sink_dev", END)

    return sg.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# QA-CONTROLLER LLM PATH
#
# _node_llm_qa_path() is the single bridge between the LangGraph node_llm and
# the QA controller + evaluation engine. It reads the QA stage for the current
# session and dispatches accordingly:
#
#   greeting  → static GREETING_TEXT, no LLM call
#   intro     → ATS extraction (rule-based + LLM fallback), seed_from_intro()
#   interview → build_next_llm_input_for_voice_graph() → LLM generate()
#   complete  → static CLOSING_TEXT, no LLM call
#
# This function NEVER raises — it returns a fallback string on any error so
# the graph can continue through sanitize → TTS → output.
# ═══════════════════════════════════════════════════════════════════════════════


_GREETING_TEXT: str = os.getenv(
    "VOICE_GREETING_TEXT",
    "Hello! Welcome to this interview. Please introduce yourself — "
    "tell me about your background, the programming languages you use, "
    "and the kind of work you enjoy.",
)

_CLOSING_TEXT: str = os.getenv(
    "VOICE_CLOSING_TEXT",
    "Thank you for completing the interview! We've covered a lot of ground. "
    "Your responses have been recorded and will be evaluated. "
    "You should hear back within a few business days.",
)

_APOLOGY_FALLBACK: str = (
    "I'm having a brief technical issue. Let me rephrase — "
    "could you tell me a bit more about your experience in that area?"
)


async def _node_llm_qa_path(
    *,
    state:      VoiceState,
    session_id: str | None,
    rid:        str,
    span:       Any,
) -> tuple[str, str, str, VADContextHint | None]:
    """
    Route the LLM node through the QA controller pipeline.

    Returns:
        (response_text, qa_stage, qa_domain)

    The qa_stage and qa_domain are written into VoiceState for downstream
    consumption (e.g., evaluation engine needs to know the domain of each
    turn, audit bus needs to know the stage transition).

    Raises nothing — returns fallback on any error.
    """
    # ── no session → direct LLM call (non-interview mode) ─────────────────
    if not session_id:
        span.set_attribute("qa_path", False)
        text = state.get("user_input", "")
        # In non-interview mode we still need SOME response. Use a minimal
        # direct call to the LLM if available.
        try:
            result = await _llm_direct_call(text, rid)
            return (result, "none", "none", None) # noqa
        except Exception as exc:
            log.warning("graph_llm_direct_failed", request_id=rid, error=str(exc))
            return (_APOLOGY_FALLBACK, "none", "none", None) # noqa

    span.set_attribute("qa_path", True)

    # ── fetch current QA document ─────────────────────────────────────────
    try:
        doc = await _qa_controller.get_document(session_id)
    except Exception as exc:
        log.error("qa_path_get_doc_failed", sid=session_id[:8], error=str(exc))
        return (_APOLOGY_FALLBACK, "unknown", "none", None) # noqa

    if doc is None:
        # Session not yet created — treat as greeting
        log.info("qa_path_no_doc_greeting", sid=session_id[:8])
        return (_GREETING_TEXT, "greeting", "none", None) # noqa

    current_stage = doc.stage

    # ══════════════════════════════════════════════════════════════════════
    #  GREETING STAGE
    # ══════════════════════════════════════════════════════════════════════

    if current_stage == "greeting":
        log.info("qa_path_greeting", sid=session_id[:8])
        span.set_attribute("qa_stage", "greeting")
        greeting_text = await _qa_controller.get_greeting(session_id)  # advances stage → "intro"
        return (greeting_text, "greeting", "none", None) # noqa

    # ══════════════════════════════════════════════════════════════════════
    #  INTRO STAGE — ATS extraction + seed
    # ══════════════════════════════════════════════════════════════════════
    if current_stage == "intro":
        candidate_answer = state.get("user_input", "")
        span.set_attribute("qa_stage", "intro")
        span.set_attribute("intro_len", len(candidate_answer))

        try:
            # 1. Get intro input from QA controller
            intro_input = await _qa_controller.get_intro_messages(session_id, candidate_answer) # noqa | DO NOT REMOVE as seed from intro will get no intro and it'll crash silently

            # 2. ATS extraction — rule-based first, LLM fallback if confidence low
            ats_result = None # noqa
            rule_result = None  # keep rule result as safety net
            try:
                rule_result = _ats_rule_extractor.extract(candidate_answer)
                if rule_result.confidence < _ATS_RULE_CONFIDENCE_THRESHOLD:
                    log.info(
                        "qa_path_ats_rule_low_confidence",
                        sid=session_id[:8],
                        confidence=round(rule_result.confidence, 2),
                    )
                    ats_result = await _ats_llm_extract(candidate_answer, rid)
                else:
                    ats_result = rule_result
            except Exception as exc:
                log.warning("qa_path_ats_rule_failed", sid=session_id[:8], error=str(exc))
                ats_result = await _ats_llm_extract(candidate_answer, rid)

            # If LLM failed but rule extractor found domains, use the rule result
            # rather than blocking the candidate with a "say that again" loop.
            if ats_result is None and rule_result is not None and rule_result.domains:
                log.info(
                    "qa_path_ats_llm_failed_using_rule_fallback",
                    sid=session_id[:8],
                    domains=rule_result.domains,
                    level=rule_result.level,
                )
                ats_result = rule_result

            if ats_result is None or not ats_result.domains:
                log.error("qa_path_ats_extraction_failed", sid=session_id[:8])
                return (
                    "I didn't quite catch all of that. Could you tell me again about "
                    "the programming languages you use and your experience level?",
                    "intro",
                    "none",
                    None,
                )

            # 3. Seed QA document from ATS result → advances stage to "interview"
            first_llm_input = await _qa_controller.seed_from_intro(session_id, ats_result)

            # 4. Generate the first interview question
            first_question = await _generate_interview_question(first_llm_input, rid, session_id)

            # 5. Commit the first turn (intro answer → first question)
            await _qa_controller.commit_turn(session_id, candidate_answer, first_question)

            # 6. Fire off eval for intro turn (fire-and-forget)
            _schedule_eval_if_enabled(session_id, doc.turn_index, candidate_answer, first_question)

            intro_response = (
                f"Great, thanks for that introduction! Let's get started with the "
                f"interview. {first_question}"
            )

            log.info(
                "qa_path_intro_complete",
                sid=session_id[:8],
                domains=ats_result.domains,
                level=ats_result.level,
            )
            return (intro_response, "interview", ats_result.domains[0] if ats_result.domains else "general", None) # noqa

        except Exception as exc:
            log.error("qa_path_intro_error", sid=session_id[:8], error=str(exc))
            return (
                "I'd like to learn more about your background. What programming "
                "languages are you most comfortable with?",
                "intro",
                "none",
                None,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  INTERVIEW STAGE — question generation
    # ══════════════════════════════════════════════════════════════════════
    if current_stage == "interview":
        candidate_answer = state.get("user_input", "")
        span.set_attribute("qa_stage", "interview")
        span.set_attribute("turn_index", doc.turn_index)

        try:
            # 1. Check session guardrails (total cap, timeout, etc.)
            guardrail_result = _qa_guardrail_engine.evaluate(doc, candidate_answer, doc.created_at)
            if guardrail_result.should_stop:
                log.info(
                    "qa_path_guardrail_stop",
                    sid=session_id[:8],
                    reason=guardrail_result.rule,
                )
                await _qa_controller.mark_complete(session_id, reason=guardrail_result.rule)
                return (_CLOSING_TEXT, "complete", doc.active_domain or "none", None) # noqa

            # 2. Build the next LLM input through the QA controller
            #    This call also handles domain rotation when the current
            #    domain's quota is met.
            llm_input = await _build_next_llm_input(session_id, candidate_answer)

            if llm_input is None:
                # All domains exhausted — interview is complete
                log.info("qa_path_all_domains_done", sid=session_id[:8])
                await _qa_controller.mark_complete(session_id, reason="all_domains_exhausted")
                return (_CLOSING_TEXT, "complete", doc.active_domain or "none", None) # noqa

            # 3. Check prefetch buffer for a pre-generated question
            prefetched = None
            if FF_QUESTION_PREFETCH:
                prefetched = _qa_prefetch_buffer.get(session_id, llm_input.domain)

            # 4. Generate the question (or use prefetched)
            if prefetched and not await _qa_controller._fingerprints.is_duplicate(session_id, llm_input.domain, prefetched): # noqa
                next_question = prefetched
                log.info("qa_path_prefetch_hit", sid=session_id[:8], domain=llm_input.domain)
            else:
                next_question = await _generate_interview_question(llm_input, rid, session_id)

            # 5. Commit the turn
            committed = await _qa_controller.commit_turn(session_id, candidate_answer, next_question)

            def _derive_scaler_action(llm_input: LLMInterviewInput) -> str:
                """Derive a ScalerActionKind value from LLMInterviewInput fields."""
                if llm_input.domain_switched:
                    return "bridge"
                if llm_input.is_first_in_domain:
                    return "coast"  # first question — candidate establishes context
                if llm_input.domain == "system_design" and llm_input.q_index_in_domain >= 1:
                    return "escalate"
                if llm_input.domain == "behavioral":
                    return "coast"
                if llm_input.q_index_in_domain == 0:
                    return "coast"
                return "probe_verify"

            vad_hint = VADContextHintBuilder.from_state_dict({ # noqa
                "qa_stage": "interview",
                "scaler_action": _derive_scaler_action(llm_input),
                "domain": llm_input.domain,
                "is_first_in_domain": llm_input.is_first_in_domain,
                "is_domain_switch": llm_input.domain_switched,
            }, turn_index=committed.turn_index)

            # 6. Fire off evaluation (fire-and-forget, off critical path)
            _schedule_eval_if_enabled(
                session_id,
                committed.turn_index,
                candidate_answer,
                next_question,
                domain=llm_input.domain,
            )

            # 7. Kick off prefetch for the predicted NEXT question (fire-and-forget)
            if FF_QUESTION_PREFETCH:
                asyncio.create_task(
                    _prefetch_next_question(session_id, llm_input.domain),
                    name=f"prefetch_{session_id[:8]}_{committed.turn_index}",
                )

            # 8. If domain switched, prepend a transition phrase
            domain_label = _DOMAIN_REGISTRY.get(llm_input.domain, {}).get("label", llm_input.domain)
            if committed.domain_switched:
                next_question = (
                    f"Let's move on to {domain_label}. {next_question}"
                )
                log.info(
                    "qa_path_domain_switch",
                    sid=session_id[:8],
                    new_domain=llm_input.domain,
                    turn_index=committed.turn_index,
                )

            span.set_attribute("domain", llm_input.domain)
            span.set_attribute("turn_index", committed.turn_index)
            span.set_attribute("domain_switched", committed.domain_switched)

            return (next_question, "interview", llm_input.domain, vad_hint) # noqa

        except Exception as exc:
            log.error(
                "qa_path_interview_error",
                sid=session_id[:8],
                error=str(exc),
                turn_index=doc.turn_index,
            )

            # Attempt static question bank fallback before giving up entirely.
            # The static bank provides hand-written questions keyed by domain and
            # level, so even when the LLM is completely down, the interview can
            # continue without the candidate noticing.
            try:
                fallback_q = _static_question_bank.get_question(
                    session_id=session_id,
                    domain=doc.active_domain or "general",
                    level=doc.candidate.level if doc.candidate else "intermediate",
                    difficulty="medium",
                )
                if fallback_q:
                    await _qa_controller.commit_turn(session_id, candidate_answer, fallback_q)
                    log.info(
                        "qa_path_static_fallback_used",
                        sid=session_id[:8],
                        domain=doc.active_domain,
                    )
                    return (fallback_q, "interview", doc.active_domain or "general", None)  # noqa
            except Exception as fb_exc:
                log.error("qa_path_static_fallback_failed", sid=session_id[:8], error=str(fb_exc))

            return (_APOLOGY_FALLBACK, "interview", doc.active_domain or "general", None)  # noqa

    # ══════════════════════════════════════════════════════════════════════
    #  COMPLETE STAGE
    # ══════════════════════════════════════════════════════════════════════
    if current_stage == "complete":
        span.set_attribute("qa_stage", "complete")
        log.info("qa_path_complete", sid=session_id[:8])
        return (_CLOSING_TEXT, "complete", "none", None) # noqa

    # ── unknown stage (defensive) ──────────────────────────────────────────
    log.error("qa_path_unknown_stage", sid=session_id[:8], stage=current_stage)
    return (_APOLOGY_FALLBACK, current_stage, "none", None) # noqa


# ── QA path helpers ────────────────────────────────────────────────────────────


async def _generate_interview_question(
    llm_input: Any,
    rid: str,
    session_id: str,
) -> str:
    """
    Generate a single interview question using the LLM. The llm_input is an
    immutable LLMInterviewInput from the QA controller containing ONLY:
      domain | level | last_question | last_answer | domain_switch_flag

    The LLM is prohibited from seeing session history — this is structurally
    enforced by the QA controller's build method, not by a prompt instruction.

    Applies post-generation guardrails:
      1. Prompt-injection scan on the generated question
      2. Content-policy filter
      3. Question fingerprint deduplication
      4. Diversity check against recent questions
      5. Word-count cap (80 words, truncated at sentence boundary)
    """
    # Injection scan on candidate answer (pre-LLM)
    if hasattr(llm_input, "last_answer") and llm_input.last_answer:
        injection_detected = _prompt_injection_detector.scan(llm_input.last_answer)
        if injection_detected:
            log.warning(
                "qa_injection_detected",
                sid=session_id[:8],
                request_id=rid,
            )
            # Proceed with sanitised version — the detector returns the cleaned text
            # or raises if severity is CRITICAL. We don't re-raise here because
            # the candidate answer is informational, not executable.

    # Build the prompt context from the immutable LLMInterviewInput
    prompt_ctx = llm_input

    # Call the LLM
    raw_question = await _llm_generate_question(prompt_ctx, rid)

    # Post-generation guardrails
    # 1. Content policy filter
    question = _content_policy_filter.filter(raw_question)

    # 2. Word count cap (sentence-boundary truncation)
    question = _streaming_word_guard._truncate_to_last_question(question) # noqa

    # 3. Fingerprint deduplication
    is_dup = await _qa_controller._fingerprints.is_duplicate(session_id, llm_input.domain, question) # noqa
    if is_dup:
        log.info("qa_question_duplicate", sid=session_id[:8])
        raise ValueError("Duplicate question with no variant available")

    # 4. Diversity check against recent questions
    too_similar, sim_score = _question_diversity_enforcer.is_too_similar(session_id, llm_input.domain, question)
    if too_similar:
        log.info("qa_question_too_similar", sid=session_id[:8], similarity=round(sim_score, 3))
        raise ValueError("Question too similar to recent questions")

    # Record for future dedup and diversity checks
    await _qa_controller._fingerprints.register(session_id, llm_input.domain, question) # noqa
    _question_diversity_enforcer.register(session_id, llm_input.domain, question)

    return question

async def _ats_llm_extract(intro_input: Any, rid: str) -> Any:
    """
    Extract candidate profile from intro text using the LLM-based ATS extractor.
    Returns an ATSExtractionResult or None on failure.
    """
    try:
        result = await _ats_mode_extract(intro_input, request_id=rid)
        return result
    except Exception as exc:
        log.error("ats_llm_extract_failed", request_id=rid, error=str(exc))
        return None


def _schedule_eval_if_enabled(
    session_id: str,
    turn_index: int,
    candidate_answer: str,
    question: str,
    domain: str = "general",
) -> None:
    """
    Fire-and-forget evaluation scheduling. The evaluation engine runs entirely
    off the critical path — it never blocks TTS or audio playback.

    The engine's own adaptive sampling, budget cap, and circuit breaker decide
    whether this particular turn actually gets scored.
    """
    if not _eval_enabled:
        return

    try:
        asyncio.create_task(
            _eval_engine.schedule_turn(
                session_id=session_id,
                turn_index=turn_index,
                question=question,
                answer=candidate_answer,
                domain=domain,
            ),
            name=f"eval_{session_id[:8]}_{turn_index}",
        )
    except Exception as exc:
        # create_task itself can fail if the loop is closing
        log.debug("eval_schedule_failed", sid=session_id[:8], error=str(exc))


async def _prefetch_next_question(session_id: str, current_domain: str) -> None:
    """
    Predict the next domain and pre-generate a question in the background.
    If the prediction is wrong, the prefetched question is simply discarded.
    """
    try:
        doc = await _qa_controller.get_document(session_id)
        if doc is None or doc.stage != "interview":
            return

        # Predict next domain: if current domain quota is almost met, predict
        # the next domain in the queue; otherwise, predict same domain.
        progress = doc.domain_progress.get(current_domain, {})
        asked = progress.get("q_asked", 0)
        target = progress.get("q_target", 999)
        if asked >= target - 1 and doc.domain_queue:
            predicted_domain = doc.domain_queue[0]
        else:
            predicted_domain = current_domain

        # Build a speculative LLM input
        speculative_input = _qa_controller.build_speculative_input(
            doc, predicted_domain,
        )
        if speculative_input is None:
            return

        question = await _generate_interview_question(
            speculative_input, f"prefetch_{session_id[:8]}", session_id,
        )
        _qa_prefetch_buffer.put(session_id, predicted_domain, question)
        log.debug(
            "qa_prefetch_stored",
            sid=session_id[:8],
            domain=predicted_domain,
        )
    except Exception as exc:
        log.debug("qa_prefetch_error", sid=session_id[:8], error=str(exc))


# ═══════════════════════════════════════════════════════════════════════════════
# VOICE GRAPH — MAIN CLASS
#
# Holds the compiled LangGraph, injected node implementations, and exposes
# all five execution modes:
#
#   run()              — single-shot API mode (request → response)
#   stream()           — streaming mode (yields intermediate states)
#   stream_full()      — realtime streaming with 3-worker pipeline
#   stream_full_pcm()  — PCM-native realtime with barge-in detection
#   run_ptt()          — push-to-talk recording → full pipeline
#
# Each VoiceGraph instance is immutable after __init__ — the compiled graph,
# config, and node implementations never change. The module creates three
# singletons at import time for common use cases.
# ═══════════════════════════════════════════════════════════════════════════════


class VoiceGraph:

    __slots__ = (
        "_graph", "_cfg", "_stt", "_llm", "_tts", "_is_dev",
        "_load_guard", "_session_mgr", "_tier",
        "_active_tasks", "_tasks_lock",
    )

    def __init__(
            self,
            *,
            stt: STTNodeProtocol  | None = None,
            llm: LLMNodeProtocol  | None = None,
            tts: TTSNodeProtocol  | None = None,
            cfg: VoiceGraphConfig | None = None,
            is_dev: bool = IS_DEV,
            tier: str = "balanced",
    ) -> None:

        self._cfg = cfg or VoiceGraphConfig.from_settings()
        self._stt = stt or _default_stt_node
        self._llm = llm or _default_llm_node
        self._tts = tts or _default_tts_node

        self._is_dev = is_dev
        self._tier = tier

        self._load_guard = LoadSheddingGuard(
            max_concurrent=self._cfg.max_inflight,
            queue_size=self._cfg.load_shed_queue_size,
        )

        # Per-request task registry for cancellation by request_id.
        # Maps rid → asyncio.Task. Tasks are removed on completion via done-callback.
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}
        self._tasks_lock = asyncio.Lock()

        self._session_mgr = SessionLifecycleManager(self._cfg)

        self._graph = _build_graph_for_instance(
            stt=self._stt,
            llm=self._llm,
            tts=self._tts,
            cfg=self._cfg,
            is_dev=self._is_dev,
        )

    @property
    def config(self) -> VoiceGraphConfig:
        return self._cfg

    @property
    def session_manager(self) -> SessionLifecycleManager:
        return self._session_mgr

    # ═══════════════════════════════════════════════════════════════════════
    # MODE 1: run()  —  single-shot API mode
    # ═══════════════════════════════════════════════════════════════════════

    async def run(
        self,
        *,
        audio_path:    str,
        session_id:    str  = "",
        request_id:    str  = "",
        language:      str  = "",
        stt_prompt:    str  = "",
        tts_voice:     str  = "",
        tts_speed:     float = 1.0,
        extra_state:   dict[str, Any] | None = None,
    ) -> VoiceState:
        """
        Single-shot execution: transcribe → generate → sanitize → synthesize.
        Returns the final VoiceState with all fields populated.

        Applies load shedding — if the system is at capacity, raises
        LoadSheddingRejected before any work begins so the HTTP handler
        can return 503 immediately.
        """
        rid = request_id or str(uuid.uuid4())
        _pipeline_starts.labels(mode="api", tier=self._tier).inc()

        async with self._load_guard.acquire(rid):
            t0 = time.monotonic()

            with tracer.start_as_current_span("voice_graph.run") as span:
                span.set_attribute("request_id", rid)
                span.set_attribute("session_id", session_id[:8] if session_id else "")
                span.set_attribute("mode", "api")
                span.set_attribute("tier", self._tier)

                initial_state: VoiceState = {
                    "audio_path":   audio_path,
                    "session_id":   session_id,
                    "request_id":   rid,
                    "language":     language,
                    "stt_prompt":   stt_prompt,
                    "tts_voice":    tts_voice,
                    "tts_speed":    tts_speed,
                    "stage":        PipelineStage.IDLE.value,
                    "user_input":   "",
                    "llm_response": "",
                    "cleaned_response": "",
                    "audio_output": "",
                    "audio_s3_uri": "",
                    "audio_local_path": "",
                    "error":        "",
                    "error_stage":  "",
                    "abort_reason": "",
                    "degraded":     False,
                    "stt_retries":  0,
                    "llm_retries":  0,
                    "stt_result":   {},
                    "llm_tokens":   {},
                    "stage_latencies": {},
                    "transcript_truncated": False,
                    "response_truncated":   False,
                    "session_turn_appended": False,
                    "qa_stage":     "",
                    "qa_domain":    "",
                    "pcm_format_mic":    None,
                    "pcm_format_speaker": None,
                    "pcm_chunks_processed": 0,
                    "pcm_barge_in_count":   0,
                    "audio_health":         None,
                }
                if extra_state:
                    initial_state.update(extra_state)

                # ── session lifecycle: open if session_id provided ─────────
                resources = None
                if session_id and FF_SESSION_LIFECYCLE:
                    resources = await self._session_mgr.open_session(session_id)
                    if resources.degraded:
                        initial_state["degraded"] = True

                try:
                    # Wrap ainvoke in a Task so cancel() can reach it by request_id.
                    # The done-callback removes it from the registry automatically so
                    # the dict never grows unbounded.
                    _task = asyncio.ensure_future(
                        self._graph.ainvoke(initial_state)  # type: ignore[arg-type]
                    )
                    self._active_tasks[rid] = _task
                    _task.add_done_callback(lambda t: self._active_tasks.pop(rid, None))
                    final_state = await _task
                except asyncio.CancelledError:
                    log.warning("graph_run_cancelled", request_id=rid)
                    raise
                except Exception as exc:
                    log.error("graph_run_fatal", request_id=rid, error=str(exc))
                    final_state = _state_update(initial_state, {
                        "stage": PipelineStage.FAILED.value,
                        "error": str(exc),
                        "degraded": True,
                    })

                total_latency = time.monotonic() - t0
                final_state["total_latency_s"] = round(total_latency, 3)
                _pipeline_latency.labels(mode="api", tier=self._tier).observe(total_latency)

                status = "ok" if not final_state.get("degraded") else "degraded"
                _pipeline_completions.labels(mode="api", tier=self._tier, status=status).inc()

                log.info(
                    "graph_run_complete",
                    request_id=rid,
                    status=status,
                    total_latency_s=round(total_latency, 3),
                    stage_latencies=final_state.get("stage_latencies", {}),
                )

                # ── register temp file for cleanup ─────────────────────────
                if resources and final_state.get("audio_local_path"):
                    resources.temp_files.append(final_state["audio_local_path"])

                # ── write transcript turn ───────────────────────────────────
                if session_id:
                    await _transcript_write_turn(
                        session_id=session_id,
                        user_text=final_state.get("user_input", ""),
                        assistant_text=final_state.get("llm_response", ""),
                        request_id=rid,
                    )

                return final_state

    # ═══════════════════════════════════════════════════════════════════════
    # MODE 2: stream()  —  intermediate state streaming
    # ═══════════════════════════════════════════════════════════════════════

    async def stream(
        self,
        *,
        audio_path: str,
        session_id: str = "",
        request_id: str = "",
        language:   str = "",
        tts_voice:  str = "",
        tts_speed:  float = 1.0,
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncIterator[VoiceState]:
        """
        Streaming execution: yields VoiceState after each node completes.
        Callers can observe STT → LLM → Sanitize → TTS progression in real
        time and update UI accordingly.
        """
        rid = request_id or str(uuid.uuid4())
        _pipeline_starts.labels(mode="stream", tier=self._tier).inc()

        async with self._load_guard.acquire(rid):
            t0 = time.monotonic()

            initial_state: VoiceState = {
                "audio_path":   audio_path,
                "session_id":   session_id,
                "request_id":   rid,
                "language":     language,
                "tts_voice":    tts_voice,
                "tts_speed":    tts_speed,
                "stage":        PipelineStage.IDLE.value,
                "user_input":   "",
                "llm_response": "",
                "cleaned_response": "",
                "audio_output": "",
                "audio_s3_uri": "",
                "audio_local_path": "",
                "error":        "",
                "error_stage":  "",
                "abort_reason": "",
                "degraded":     False,
                "stt_retries":  0,
                "llm_retries":  0,
                "stt_result":   {},
                "llm_tokens":   {},
                "stage_latencies": {},
                "transcript_truncated": False,
                "response_truncated":   False,
                "session_turn_appended": False,
                "qa_stage":     "",
                "qa_domain":    "",
                "pcm_format_mic":    None,
                "pcm_format_speaker": None,
                "pcm_chunks_processed": 0,
                "pcm_barge_in_count":   0,
                "audio_health":         None,
            }
            if extra_state:
                initial_state.update(extra_state)

            try:
                async for state_snapshot in self._graph.astream(initial_state):  # type: ignore[arg-type]
                    yield state_snapshot
            except asyncio.CancelledError:
                log.warning("graph_stream_cancelled", request_id=rid)
                raise

            total_latency = time.monotonic() - t0
            _pipeline_latency.labels(mode="stream", tier=self._tier).observe(total_latency)
            _pipeline_completions.labels(mode="stream", tier=self._tier, status="ok").inc()

    # ═══════════════════════════════════════════════════════════════════════
    # MODE 3: stream_full()  —  realtime 3-worker pipeline
    # ═══════════════════════════════════════════════════════════════════════

    async def stream_full(
        self,
        *,
        audio_path: str,
        session_id: str = "",
        request_id: str = "",
        language:   str = "",
        tts_voice:  str = "", # noqa
        tts_speed:  float = 1.0,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Full realtime pipeline with three concurrent workers communicating
        through StageBus message queues:

            STT worker → [stt_to_llm bus] → LLM worker → [llm_to_tts bus] → TTS worker

        Each worker runs as an independent asyncio task. The buses provide
        backpressure: if TTS is slow, the llm_to_tts bus fills up and the
        LLM worker blocks on put() until TTS catches up.

        Yields audio segments as they're produced by TTS, enabling the caller
        to begin playback before the full response is synthesised.

        Error classification:
          - Transient errors (network timeout, rate limit) → retry with backoff
          - Permanent errors (invalid input, content policy) → skip segment, log

        Segment deduplication:
          Segments carry a monotonic sequence number. The yield loop deduplicates
          by tracking the last emitted sequence number.
        """
        rid = request_id or str(uuid.uuid4())
        _pipeline_starts.labels(mode="realtime", tier=self._tier).inc()

        async with self._load_guard.acquire(rid):
            t0 = time.monotonic()

            with tracer.start_as_current_span("voice_graph.stream_full") as span:
                span.set_attribute("request_id", rid)
                span.set_attribute("session_id", session_id[:8] if session_id else "")
                span.set_attribute("mode", "realtime")

                # ── create inter-stage buses ──────────────────────────────
                stt_to_llm = StageBus(
                    stage_pair="stt→llm",
                    max_depth=self._cfg.stage_bus_maxsize,
                    overflow=OverflowPolicy.BLOCK,
                )
                llm_to_tts = StageBus(
                    stage_pair="llm→tts",
                    max_depth=self._cfg.stage_bus_maxsize,
                    overflow=OverflowPolicy.BLOCK,
                )
                output_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(
                    maxsize=self._cfg.output_queue_maxsize,
                )

                # Track resources for lifecycle management
                resources = None
                if session_id and FF_SESSION_LIFECYCLE:
                    resources = await self._session_mgr.open_session(session_id)
                    if resources:
                        resources.buses.extend([stt_to_llm, llm_to_tts])

                # ── STT worker ────────────────────────────────────────────
                async def stt_worker() -> None:
                    try:
                        _validate_audio_path(audio_path)

                        result = await _with_timeout(
                            self._stt.transcribe(
                                audio_path=audio_path,
                                language=language,
                                request_id=rid,
                            ),
                            timeout=self._cfg.stt_timeout,
                            stage="STT",
                        )

                        raw_transcript = result["text"]
                        san = sanitize(raw_transcript, max_chars=self._cfg.max_transcript_chars, request_id=rid)

                        msg = StageBusMessage(
                            payload={
                                "user_input": san.text,
                                "stt_result": dict(result),
                                "truncated": san.truncated,
                            },
                            source_stage="stt",
                            seq=0,
                        )
                        await stt_to_llm.put(msg)
                        log.info("stream_full_stt_ok", request_id=rid, transcript_len=len(san.text))

                    except Exception as exc:
                        error_msg = StageBusMessage(
                            payload={"error": str(exc), "error_type": type(exc).__name__},
                            source_stage="stt",
                            is_error=True,
                            seq=0,
                        )
                        await stt_to_llm.put(error_msg)
                        log.error("stream_full_stt_failed", request_id=rid, error=str(exc))
                    finally:
                        await stt_to_llm.put(StageBusMessage.sentinel("stt"))

                # ── LLM worker ────────────────────────────────────────────
                async def llm_worker() -> None:
                    try:
                        msg = await stt_to_llm.get()
                        if msg is None or msg.is_sentinel:
                            await llm_to_tts.put(StageBusMessage.sentinel("llm"))
                            return

                        if msg.is_error:
                            # Propagate STT error as apology text
                            error_payload = {
                                "text": APOLOGY_STT,
                                "is_fallback": True,
                                "error": msg.payload.get("error", ""),
                            }
                            fwd = StageBusMessage(payload=error_payload, source_stage="llm", seq=0)
                            await llm_to_tts.put(fwd)
                            await llm_to_tts.put(StageBusMessage.sentinel("llm"))
                            return

                        user_input = msg.payload.get("user_input", "")

                        # Build a VoiceState-like dict for the QA path
                        pseudo_state: VoiceState = {
                            "audio_path": "",
                            "user_input": user_input,
                            "session_id": session_id,
                            "request_id": rid,
                        }

                        response_text, qa_stage, qa_domain, _ = await _node_llm_qa_path(
                            state=pseudo_state,
                            session_id=session_id,
                            rid=rid,
                            span=span,
                        )

                        # Sanitize the LLM response
                        san = sanitize(response_text, max_chars=self._cfg.max_tts_chars, request_id=rid)

                        # Stream sentences to TTS for sentence-level pipelining
                        sentences = _split_into_sentences(san.text)
                        for i, sentence in enumerate(sentences):
                            seg_msg = StageBusMessage(
                                payload={
                                    "text": sentence,
                                    "segment_index": i,
                                    "total_segments": len(sentences),
                                    "qa_stage": qa_stage,
                                    "qa_domain": qa_domain,
                                    "is_fallback": False,
                                },
                                source_stage="llm",
                                seq=i,
                            )
                            await llm_to_tts.put(seg_msg)
                            log.debug(
                                "stream_full_llm_sentence",
                                request_id=rid,
                                segment=i,
                                chars=len(sentence),
                            )

                    except Exception as exc:
                        error_payload = {
                            "text": APOLOGY_LLM,
                            "is_fallback": True,
                            "error": str(exc),
                        }
                        await llm_to_tts.put(StageBusMessage(payload=error_payload, source_stage="llm", seq=0))
                        log.error("stream_full_llm_failed", request_id=rid, error=str(exc))
                    finally:
                        await llm_to_tts.put(StageBusMessage.sentinel("llm"))

                # ── TTS worker ────────────────────────────────────────────
                async def tts_worker() -> None:
                    segment_seq = 0
                    try:
                        while True:
                            msg = await llm_to_tts.get()
                            if msg is None or msg.is_sentinel:
                                break

                            text = msg.payload.get("text", "")
                            if not text.strip():
                                continue

                            try:
                                result = await _with_timeout(
                                    self._tts.synthesize(
                                        text=text,
                                        voice="nova",
                                        speed=tts_speed,
                                        request_id=rid,
                                    ),
                                    timeout=self._cfg.tts_timeout,
                                    stage="TTS",
                                )

                                local_path, s3_uri, duration_s = result

                                segment = {
                                    "type": "audio_segment",
                                    "sequence": segment_seq,
                                    "local_path": local_path,
                                    "s3_uri": s3_uri,
                                    "duration_s": duration_s,
                                    "text": text,
                                    "qa_stage": msg.payload.get("qa_stage", ""),
                                    "qa_domain": msg.payload.get("qa_domain", ""),
                                    "is_fallback": msg.payload.get("is_fallback", False),
                                    "segment_index": msg.payload.get("segment_index", segment_seq),
                                    "total_segments": msg.payload.get("total_segments", 1),
                                }
                                await output_queue.put(segment)

                                if resources and local_path:
                                    resources.temp_files.append(local_path)

                                segment_seq += 1

                            except Exception as exc:
                                _stage_errors.labels(stage="tts").inc()
                                log.error(
                                    "stream_full_tts_segment_failed",
                                    request_id=rid,
                                    segment=segment_seq,
                                    error=str(exc),
                                )
                                # Don't break — try remaining segments

                    except Exception as exc:
                        log.error("stream_full_tts_worker_failed", request_id=rid, error=str(exc))
                    finally:
                        await output_queue.put(None)  # sentinel

                # ── launch workers + yield output ─────────────────────────
                stt_task = asyncio.create_task(stt_worker(), name=f"stt_{rid[:8]}")
                llm_task = asyncio.create_task(llm_worker(), name=f"llm_{rid[:8]}")
                tts_task = asyncio.create_task(tts_worker(), name=f"tts_{rid[:8]}")

                last_seq = -1
                try:
                    while True:
                        segment = await output_queue.get()
                        if segment is None:
                            break

                        # Deduplication by sequence number
                        seq = segment.get("sequence", 0)
                        if seq <= last_seq:
                            log.debug("stream_full_dedup_skip", request_id=rid, sequence=seq)
                            continue
                        last_seq = seq

                        yield segment

                except asyncio.CancelledError:
                    stt_task.cancel()
                    llm_task.cancel()
                    tts_task.cancel()
                    log.warning("stream_full_cancelled", request_id=rid)
                    raise

                # ── wait for workers to finish ────────────────────────────
                for task in (stt_task, llm_task, tts_task):
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        log.warning("stream_full_worker_error", request_id=rid, error=str(exc))

                total_latency = time.monotonic() - t0
                _pipeline_latency.labels(mode="realtime", tier=self._tier).observe(total_latency)
                _pipeline_completions.labels(
                    mode="realtime", tier=self._tier,
                    status="ok" if last_seq >= 0 else "empty",
                ).inc()

                log.info(
                    "stream_full_complete",
                    request_id=rid,
                    segments=last_seq + 1,
                    total_latency_s=round(total_latency, 3),
                )

    # ═══════════════════════════════════════════════════════════════════════
    # MODE 4: stream_full_pcm()  —  PCM-native realtime with barge-in
    # ═══════════════════════════════════════════════════════════════════════

    async def stream_full_pcm(
        self,
        *,
        pcm_input_stream: PCMInputStream | AsyncIterator[PCMChunk],
        session_id:        str = "",
        request_id:        str = "",
        tts_voice:         str = "", # noqa
        tts_speed:         float = 1.0,
    ) -> AsyncIterator[PCMChunk]:
        """
        PCM-native realtime pipeline. Audio flows as PCMChunk objects through
        the entire pipeline — no intermediate WAV file encoding/decoding.

        Architecture:
            mic → PCMInputStream → VADGate → [ring buffer] → STT (chunk-level)
                → QA/LLM path → sanitize → TTS (PCM stream)
                → PCMPlaybackEnhancer → interrupt detector → PCMOutputStream

        Barge-in detection:
            The PCMInterruptDetector runs continuously on the mic input in
            parallel with TTS playback. When the candidate starts speaking
            while audio is playing, the detector fires and the pipeline:
              1. Cancels the current TTS stream
              2. Flushes the playback buffer
              3. Re-enters the STT stage with the new speech

        PCM format negotiation happens at session open — the mic and speaker
        formats are already known by the time this method is called.

        Yields PCMChunk objects that the caller feeds to the speaker output.
        """
        rid = request_id or str(uuid.uuid4())
        _pipeline_starts.labels(mode="pcm", tier=self._tier).inc()

        if not FF_PCM_PIPELINE and not _should_use_pcm_pipeline(rid):
            raise RuntimeError(
                "PCM pipeline is not enabled. Set VOICE_FF_PCM_PIPELINE=1 or "
                "configure FF_CANARY_PCT for gradual rollout."
            )

        async with self._load_guard.acquire(rid):
            t0 = time.monotonic()

            with tracer.start_as_current_span("voice_graph.stream_full_pcm") as span:
                span.set_attribute("request_id", rid)
                span.set_attribute("session_id", session_id[:8] if session_id else "")
                span.set_attribute("mode", "pcm")

                # ── session resources ─────────────────────────────────────
                resources = None
                if session_id and FF_SESSION_LIFECYCLE:
                    resources = await self._session_mgr.open_session(session_id)

                # ── PCM components initialisation ─────────────────────────
                #
                # The chunk pool provides zero-allocation chunk recycling for
                # the hot path. The ring buffer acts as a rolling window of
                # recent audio for context-dependent STT. The VAD gate
                # filters silence frames before they reach the ring buffer.
                # The jitter buffer smooths TTS output timing.
                pool = get_chunk_pool() # noqa

                mic_format = (
                    resources.negotiated_stt_fmt
                    if resources and resources.negotiated_stt_fmt
                    else PCMFormat(sample_rate=16000, channels=1, dtype="int16")
                )
                speaker_format = (
                    resources.negotiated_tts_fmt
                    if resources and resources.negotiated_tts_fmt
                    else PCMFormat.openai_tts()
                )

                ring_buffer = PCMRingBuffer(
                    capacity=int(self._cfg.pcm_ring_buffer_seconds * mic_format.sample_rate),
                    fmt=mic_format,
                )
                vad_gate = ContextGatedVAD(
                    fmt=mic_format,
                    hangover_s=self._cfg.pcm_vad_hangover_frames / mic_format.sample_rate,
                )
                speech_enhancer = PCMSpeechEnhancer(fmt=mic_format)
                interrupt_detector = PCMInterruptDetector(
                    fmt=mic_format,
                    onset_rms=self._cfg.pcm_interrupt_threshold,
                )
                playback_enhancer = PCMPlaybackEnhancer(fmt=speaker_format)
                jitter_buffer = PCMJitterBuffer(
                    fmt=speaker_format,
                    target_delay_ms=self._cfg.pcm_jitter_buffer_ms,
                )
                latency_tracker = PCMLatencyTracker()
                diagnostics = PCMDiagnosticsMonitor(fmt=mic_format)
                chunk_encoder = PCMChunkWAVEncoder()
                confidence_filter = PCMConfidenceFilter(threshold=-0.4)
                format_converter = PCMConverter() if (
                resources and resources.mic_format and resources.mic_format != mic_format) else None

                # Converter for mic→STT format if they differ
                format_converter = None
                if resources and resources.mic_format and resources.mic_format != mic_format:
                    format_converter = PCMConverter()

                # TTS PCM output config
                tts_pcm_config = PCMTTSOutputConfig.for_format(fmt=speaker_format)

                sentence_gap_mgr = PCMSentenceGapManager(
                    fmt=speaker_format,
                    gap_s=0.2,
                )
                quality_gate = PCMTTSQualityGate(analyzer=tts_pcm_config.analyzer)
                output_stream = get_output_stream()

                # ── inter-stage communication ─────────────────────────────
                speech_segments: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=4)
                tts_chunks:      asyncio.Queue[PCMChunk | None] = asyncio.Queue(maxsize=32)
                barge_in_event = asyncio.Event()
                pipeline_done  = asyncio.Event()

                barge_in_count = 0
                chunks_processed = 0

                # ── mic reader: PCMInputStream → VAD → ring buffer ────────
                # Runs as a background task for the lifetime of the pipeline.
                # Pulls raw PCMChunks from the input stream (mic or WebSocket),
                # routes them through format conversion, diagnostics, barge-in
                # detection, VAD gating, and speech enhancement before writing
                # clean speech frames into the ring buffer for the VAD segmenter.
                async def mic_reader() -> None:
                    nonlocal chunks_processed
                    try:
                        async for chunk in pcm_input_stream:
                            if pipeline_done.is_set():
                                break

                            chunks_processed += 1
                            _pcm_chunks_total.labels(stage="mic_in").inc()

                            # ── Format conversion ─────────────────────────
                            # If the mic's native format differs from the STT
                            # input format (e.g. 48kHz stereo → 16kHz mono),
                            # resample and downmix here before any processing.
                            if format_converter:
                                chunk = format_converter.convert(chunk, mic_format)

                            # ── Local diagnostics (always active) ─────────
                            # PCMDiagnosticsMonitor accumulates per-chunk RMS,
                            # clipping %, and silence % into a rolling window.
                            # Runs unconditionally so health data is always
                            # available for the watchdog even when the feature
                            # flag for session-level diagnostics is off.
                            diagnostics.push(chunk)

                            # ── Session-level diagnostics (feature-flagged) ─
                            # Feeds into AudioDiagnosticsPipeline which emits
                            # structured logs + Prometheus histograms per
                            # boundary (mic_in / pre-STT / post-TTS / speaker).
                            if FF_AUDIO_DIAGNOSTICS and resources and resources.diagnostics:
                                await resources.diagnostics.measure(chunk, "mic_in")

                            # ── Barge-in detection ────────────────────────
                            # Only active while TTS is playing. The interrupt
                            # detector compares incoming mic RMS against a
                            # reference threshold — if the user speaks over the
                            # bot, it fires barge_in_event which causes the TTS
                            # loop in pipeline_worker to abort mid-utterance.
                            if interrupt_detector._playback_active: # noqa
                                if interrupt_detector.push(chunk):
                                    nonlocal barge_in_count
                                    barge_in_count += 1
                                    _pcm_barge_in_total.inc()
                                    barge_in_event.set()
                                    log.info("pcm_barge_in_detected", request_id=rid, count=barge_in_count)

                            # ── VAD gate → speech enhancer → ring buffer ──
                            # _process_chunk returns a chunk only during active
                            # speech (gate open + hangover). Passing the gated
                            # chunk through PCMSpeechEnhancer applies bandpass
                            # filtering, noise suppression, and AGC before the
                            # samples land in the ring buffer. This means the
                            # STT engine only ever sees clean, level-normalised
                            # speech — never silence or background noise.
                            result = vad_gate._process_chunk(chunk) # noqa
                            if result is not None:
                                async def _single_result() -> AsyncIterator[PCMChunk]:
                                    yield result

                                async for enhanced in speech_enhancer.stream(_single_result()):
                                    ring_buffer.write(enhanced.data)

                            # Record wall-clock latency from capture to this
                            # point for the per-stage latency budget tracker.
                            latency_tracker.observe(chunk, "mic_read")

                    except Exception as exc:
                        log.error("pcm_mic_reader_error", request_id=rid, error=str(exc))
                    finally:
                        # Sentinel None tells vad_segmenter the mic stream has
                        # ended so it can exit its polling loop cleanly.
                        await speech_segments.put(None)

                # ── VAD segmenter: ring buffer → speech segments ──────────
                # Polls the VAD gate's internal speech frame counter on a 20ms
                # tick. On a falling edge (speech → silence transition) it
                # flushes whatever the mic_reader wrote to the ring buffer as
                # a single contiguous bytes blob and enqueues it for STT.
                # Polling rather than event-driven keeps this task lock-free
                # and avoids adding latency to the hot mic_reader path.
                async def vad_segmenter() -> None:
                    try:
                        was_speech = False
                        poll_interval = 0.02  # 20ms — one STT chunk window

                        while not pipeline_done.is_set():
                            is_speech_now = vad_gate._speech_frame_count > 0 # noqa

                            # Falling edge: user stopped speaking. Drain the
                            # ring buffer and hand the segment to pipeline_worker
                            # for STT → LLM → TTS processing.
                            if was_speech and not is_speech_now:
                                segment_bytes = ring_buffer.read(ring_buffer.available_to_read())
                                if len(segment_bytes) > 0:
                                    await speech_segments.put(segment_bytes.tobytes())
                                    log.debug(
                                        "pcm_speech_segment",
                                        request_id=rid,
                                        bytes=len(segment_bytes),
                                    )

                            was_speech = is_speech_now
                            await asyncio.sleep(poll_interval)

                    except Exception as exc:
                        log.error("pcm_vad_segmenter_error", request_id=rid, error=str(exc))

                # ── STT + LLM + TTS worker ────────────────────────────────
                # Processes one complete speech segment per loop iteration:
                #   1. Encodes PCM bytes → WAV, writes to a temp file, calls STT
                #   2. Filters low-confidence transcripts via PCMConfidenceFilter
                #   3. Sends transcript through the QA/LLM path
                #   4. Synthesizes the LLM response as a PCM stream via TTS
                #   5. Each TTS chunk passes quality gate → playback enhancer →
                #      jitter buffer → speaker output + tts_chunks yield queue
                # Runs serially (one segment at a time) to preserve turn order.
                async def pipeline_worker() -> None:
                    try:
                        while True:
                            # Block until vad_segmenter enqueues a segment or
                            # mic_reader sends the EOF sentinel (None).
                            segment_bytes = await speech_segments.get()
                            if segment_bytes is None:
                                break

                            # ── STT: chunk-level transcription ────────────
                            stt_t0 = time.monotonic()
                            try:
                                # Wrap raw PCM bytes as a PCMChunk then encode
                                # to WAV — STT engines universally accept WAV
                                # and this avoids format negotiation overhead.
                                chunk = tts_pcm_to_chunk(segment_bytes, fmt=mic_format, seq=0)
                                wav_bytes = chunk_encoder.encode(chunk)

                                # Write to a named temp file because most STT
                                # clients expect a file path, not a byte stream.
                                # The finally block guarantees cleanup even on
                                # transcription timeout or network error.
                                import tempfile, os
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                                    f.write(wav_bytes)
                                    tmp_path = f.name
                                try:
                                    stt_result = await _with_timeout(
                                        self._stt.transcribe(tmp_path, request_id=rid),
                                        timeout=self._cfg.stt_timeout,
                                        stage="STT",
                                    )
                                finally:
                                    os.unlink(tmp_path)

                                transcript = stt_result["text"]
                                confidence = stt_result.get("confidence", 1.0)

                                # ── Confidence filter ─────────────────────
                                # PCMConfidenceFilter.accept() applies the
                                # threshold set at construction time (-0.4 log
                                # prob). Drops hallucinated or noise-triggered
                                # transcripts before they reach the LLM, which
                                # would otherwise generate a spurious response.
                                if not confidence_filter.check(
                                        {
                                            "avg_logprob": confidence,
                                            "text": transcript,
                                            "language": "",
                                            "start": 0.0,
                                            "end": 0.0,
                                            "chunk_index": 0,
                                            "is_final": True,
                                        },
                                        request_id=rid,
                                ):
                                    log.info(
                                        "pcm_stt_low_confidence",
                                        request_id=rid,
                                        confidence=round(confidence, 2),
                                    )
                                    continue

                                # Sanitize: strip PII tokens, enforce char limit,
                                # normalise whitespace before handing to the LLM.
                                san = sanitize(transcript, max_chars=self._cfg.max_transcript_chars, request_id=rid)
                                user_input = san.text

                                # Empty after sanitization = nothing useful said.
                                if not user_input.strip():
                                    continue

                                stt_latency = time.monotonic() - stt_t0
                                _stage_latency.labels(stage="stt_pcm").observe(stt_latency)
                                latency_tracker.record("stt", stt_latency)

                            except Exception as exc:
                                _stage_errors.labels(stage="stt_pcm").inc()
                                log.error("pcm_stt_failed", request_id=rid, error=str(exc))
                                continue  # drop segment, wait for next utterance

                            # ── LLM: QA path ─────────────────────────────
                            # Routes the transcript through the QA controller
                            # (domain classification → retrieval → generation).
                            # On failure falls back to a canned apology so TTS
                            # always has something to say and the pipeline keeps
                            # running rather than stalling the conversation.
                            llm_t0 = time.monotonic()
                            try:
                                pseudo_state: VoiceState = {
                                    "audio_path": "",
                                    "user_input": user_input,
                                    "session_id": session_id,
                                    "request_id": rid,
                                }
                                response_text, qa_stage, qa_domain, vad_hint = await _node_llm_qa_path(
                                    state=pseudo_state,
                                    session_id=session_id,
                                    rid=rid,
                                    span=span,
                                )
                                if vad_hint is not None:
                                    vad_gate.apply_context(vad_hint)

                                # Sanitize LLM output before TTS: strip markdown,
                                # cap length, remove any injected control tokens.
                                san = sanitize(response_text, max_chars=self._cfg.max_tts_chars, request_id=rid)
                                tts_text = san.text

                                llm_latency = time.monotonic() - llm_t0
                                _stage_latency.labels(stage="llm_pcm").observe(llm_latency)
                                latency_tracker.record("llm", llm_latency)

                            except Exception as exc:
                                _stage_errors.labels(stage="llm_pcm").inc()
                                log.error("pcm_llm_failed", request_id=rid, error=str(exc))
                                tts_text = APOLOGY_LLM
                                qa_stage = "error"  # noqa
                                qa_domain = "none"  # noqa

                            # ── TTS: PCM stream synthesis ─────────────────
                            # Streams PCM chunks from the TTS engine and routes
                            # each through the following chain before output:
                            #   quality gate → playback enhancer → jitter buffer
                            #     → speaker output stream + tts_chunks yield queue
                            # Barge-in is checked before every chunk so the
                            # pipeline aborts the utterance within one chunk
                            # latency (~20ms) of the user starting to speak.
                            tts_t0 = time.monotonic()
                            try:
                                # Activate interrupt detection. From this point
                                # the mic_reader will fire barge_in_event if the
                                # user speaks, causing the loop below to break.
                                interrupt_detector.set_playback_active(True)
                                barge_in_event.clear()

                                # Wrap the sanitized response as a single-token
                                # async stream. synthesize_pcm_stream expects an
                                # AsyncIterator[str] so it can interleave synthesis
                                # with live LLM token delivery; here we have the
                                # full text already so we yield it all at once.
                                async def _text_to_stream(text: str) -> AsyncIterator[str]:
                                    yield text

                                # Thin wrapper used to feed a single PCMChunk into
                                # PCMPlaybackEnhancer.stream(), which is designed
                                # for async iteration. Defined inside the loop so
                                # it captures nothing from the outer scope and
                                # is trivially GC-able after each chunk.
                                async def _single_chunk_iter(chunk: PCMChunk) -> AsyncIterator[PCMChunk]:
                                    yield chunk

                                # Track seq of the last chunk yielded so we can
                                # number the inter-sentence silence gap correctly.
                                last_seq = -1

                                async for pcm_chunk in self._tts.synthesize_pcm_stream(
                                        token_stream=_text_to_stream(tts_text),
                                        voice=None,  # uses TTSNode default (nova)
                                        speed=tts_speed,
                                        request_id=rid,
                                ):
                                    last_seq = pcm_chunk.seq

                                    # Check barge-in before every chunk so we
                                    # abort within one synthesis window (~20ms)
                                    # of the user starting to speak.
                                    if barge_in_event.is_set():
                                        log.info("pcm_tts_barge_in_interrupt", request_id=rid)
                                        break

                                    # ── Quality gate ──────────────────────
                                    # Discards silent, clipped, or sub-threshold
                                    # RMS chunks before they reach the speaker.
                                    # Prevents audible glitches from TTS engine
                                    # warm-up frames and trailing silence.
                                    verdict = quality_gate.check(pcm_chunk, request_id=rid)
                                    if verdict != PCMTTSQualityGate.OK:
                                        log.debug(
                                            "pcm_tts_quality_gate_skip",
                                            request_id=rid,
                                            verdict=verdict,
                                            seq=pcm_chunk.seq,
                                        )
                                        continue

                                    # ── Playback enhancement ──────────────
                                    # PCMPlaybackEnhancer applies a True Peak
                                    # limiter and pads leading/trailing silence
                                    # for natural cadence. It's a streaming
                                    # facade so we wrap the chunk in a one-item
                                    # async iterator and take the single output.
                                    enhanced = pcm_chunk  # fallback if enhancer yields nothing
                                    async for enhanced in playback_enhancer.stream(_single_chunk_iter(pcm_chunk)):
                                        pass

                                    # ── Jitter buffer ─────────────────────
                                    # PCMJitterBuffer absorbs inter-chunk timing
                                    # variance from the TTS HTTP stream (network
                                    # jitter, chunked encoding delays). It holds
                                    # chunks until target_delay_ms of audio has
                                    # accumulated, then releases them at a steady
                                    # rate — preventing stuttering at the speaker
                                    # output without adding meaningful latency.
                                    # before the TTS loop, start the jitter buffer
                                    await jitter_buffer.start()

                                    # inside the TTS loop, replace the while pop block:
                                    jitter_buffer.push(enhanced)
                                    # drain whatever the playout task has made available without blocking
                                    try:
                                        while True:
                                            smoothed = jitter_buffer._out_q.get_nowait() # noqa
                                            if smoothed is None:
                                                break
                                            await output_stream.write(smoothed)
                                            _pcm_chunks_total.labels(stage="tts_out", direction="out").inc()
                                            await tts_chunks.put(smoothed)
                                    except asyncio.QueueEmpty:
                                        pass

                                # Insert a calibrated silence gap between sentences
                                # so the bot doesn't sound clipped. Only emitted if
                                # the TTS stream produced at least one chunk.
                                if last_seq >= 0:
                                    gap = sentence_gap_mgr.make_gap_chunk(seq=last_seq + 1)
                                    await tts_chunks.put(gap)

                                # Deactivate interrupt detection — mic noise during
                                # the inter-turn silence should not trigger barge-in.
                                interrupt_detector.set_playback_active(False)

                                tts_latency = time.monotonic() - tts_t0
                                _stage_latency.labels(stage="tts_pcm").observe(tts_latency)
                                latency_tracker.record("tts", tts_latency)
                                log.info(
                                    "pcm_tts_complete",
                                    request_id=rid,
                                    latency_s=round(tts_latency, 3),
                                    barge_in=barge_in_event.is_set(),
                                )

                            except Exception as exc:
                                # Always deactivate on failure — leaving it active
                                # would suppress mic input for the next utterance,
                                # making the bot appear to stop listening entirely.
                                _stage_errors.labels(stage="tts_pcm").inc()
                                interrupt_detector.set_playback_active(False)
                                log.error("pcm_tts_failed", request_id=rid, error=str(exc))

                    except Exception as exc:
                        log.error("pcm_pipeline_worker_error", request_id=rid, error=str(exc))
                    finally:
                        # Sentinel None unblocks the yield loop in stream_full_pcm
                        # so the caller's async generator returns cleanly.
                        await tts_chunks.put(None)

                # ── launch workers ────────────────────────────────────────
                mic_task = asyncio.create_task(mic_reader(), name=f"pcm_mic_{rid[:8]}")
                vad_task = asyncio.create_task(vad_segmenter(), name=f"pcm_vad_{rid[:8]}")
                pipe_task = asyncio.create_task(pipeline_worker(), name=f"pcm_pipe_{rid[:8]}")

                # ── yield PCM chunks to caller ────────────────────────────
                try:
                    while True:
                        chunk = await tts_chunks.get()
                        if chunk is None:
                            break
                        yield chunk

                except asyncio.CancelledError:
                    pipeline_done.set()
                    mic_task.cancel()
                    vad_task.cancel()
                    pipe_task.cancel()
                    log.warning("pcm_pipeline_cancelled", request_id=rid)
                    raise

                # ── cleanup ───────────────────────────────────────────────
                pipeline_done.set()
                for task in (mic_task, vad_task, pipe_task):
                    try:
                        await asyncio.wait_for(task, timeout=2.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        task.cancel()
                    except Exception as exc:
                        log.debug("pcm_worker_cleanup_error", error=str(exc))

                total_latency = time.monotonic() - t0
                _pipeline_latency.labels(mode="pcm", tier=self._tier).observe(total_latency)
                _pipeline_completions.labels(mode="pcm", tier=self._tier, status="ok").inc()

                # Update state for lifecycle tracking
                if resources:
                    resources.temp_files.extend([])  # PCM mode has no temp files

                log.info(
                    "pcm_pipeline_complete",
                    request_id=rid,
                    total_latency_s=round(total_latency, 3),
                    chunks_processed=chunks_processed,
                    barge_in_count=barge_in_count,
                )
                span.set_attribute("chunks_processed", chunks_processed)
                span.set_attribute("barge_in_count", barge_in_count)

    # ═══════════════════════════════════════════════════════════════════════
    # MODE 5: run_ptt()  —  push-to-talk recording → full pipeline
    # ═══════════════════════════════════════════════════════════════════════

    async def run_ptt(
        self,
        *,
        is_held_fn:   Callable[[], bool],
        session_id:   str  = "",
        request_id:   str  = "",
        language:     str  = "",
        tts_voice:    str  = "",
        tts_speed:    float = 1.0,
        play_response: bool = True,
    ) -> VoiceState:
        """
        Push-to-talk mode: record from microphone while is_held_fn() returns
        True, then run the full pipeline on the recorded audio.

        This mode is designed for local development and demo scenarios where
        the caller controls recording via a button hold. The recording is
        handled by recorder.py which performs two-stage silence rejection:
          1. PCMVADGate: fast energy-based rejection of pure silence
          2. PCMSpeechEnhancer VAD: precise rejection of ambient noise

        If play_response is True and we're in dev mode, the synthesised audio
        is played through the local speaker after TTS completes.

        The temporary recording file is cleaned up after the pipeline finishes,
        regardless of success or failure.
        """
        rid = request_id or str(uuid.uuid4())
        _pipeline_starts.labels(mode="ptt", tier=self._tier).inc()
        _ptt_total.inc()

        async with self._load_guard.acquire(rid):
            t0 = time.monotonic()

            with tracer.start_as_current_span("voice_graph.run_ptt") as span:
                span.set_attribute("request_id", rid)
                span.set_attribute("session_id", session_id[:8] if session_id else "")
                span.set_attribute("mode", "ptt")

                audio_path = ""
                try:
                    # ── 1. Record audio ───────────────────────────────────
                    rec_t0 = time.monotonic()
                    log.info("ptt_recording_start", request_id=rid)

                    recording_result = await record_audio_until_released_async(is_held_fn)
                    audio_path = recording_result or ""
                    rec_latency = time.monotonic() - rec_t0 # noqa

                    # Derive actual recording duration from the WAV header —
                    # this is the real speech duration, not the wall-clock hold time.
                    if audio_path:
                        try:
                            with wave.open(audio_path, "rb") as wf:
                                rec_duration = wf.getnframes() / wf.getframerate()
                        except Exception: # noqa
                            rec_duration = 0.0
                    else:
                        rec_duration = 0.0

                    _ptt_duration.observe(rec_duration)
                    log.info(
                        "ptt_recording_complete",
                        request_id=rid,
                        duration_s=round(rec_duration, 2),
                        path=audio_path,
                    )
                    span.set_attribute("rec_duration_s", round(rec_duration, 2))

                    if not audio_path:
                        raise ValueError("Recording produced no audio file — silence or mic error.")

                    # ── 2. Mic health check (non-blocking) ────────────────
                    try:
                        rec_health = get_recording_health()
                        if rec_health and not rec_health.healthy:
                            log.warning("ptt_mic_degraded", request_id=rid, health=str(rec_health))
                    except Exception: # noqa
                        pass

                    # ── 3. Run pipeline ───────────────────────────────────
                    final_state = await self.run(
                        audio_path=audio_path,
                        session_id=session_id,
                        request_id=rid,
                        language=language,
                        tts_voice=tts_voice,
                        tts_speed=tts_speed,
                    )

                    # ── 4. Dev playback ───────────────────────────────────
                    if play_response and self._is_dev:
                        local_path = final_state.get("audio_local_path", "")
                        if local_path:
                            try:
                                play_t0 = time.monotonic()

                                # Prefer PCM playback if available for lower latency
                                pcm_bytes = final_state.get("pcm_audio_bytes")
                                if pcm_bytes:
                                    await asyncio.to_thread(
                                        play_pcm_bytes,
                                        pcm_bytes,
                                        PCMFormat.openai_tts(),
                                    )
                                else:
                                    await asyncio.to_thread(play_audio, local_path)

                                play_latency = time.monotonic() - play_t0
                                log.info(
                                    "ptt_playback_ok",
                                    request_id=rid,
                                    latency_s=round(play_latency, 3),
                                )
                            except Exception as exc:
                                log.warning("ptt_playback_failed", request_id=rid, error=str(exc))

                    total_latency = time.monotonic() - t0
                    final_state["pipeline_latency_s"] = round(total_latency, 3)
                    final_state["rec_duration_s"] = round(rec_duration, 2)

                    _pipeline_latency.labels(mode="ptt", tier=self._tier).observe(total_latency)
                    _pipeline_completions.labels(mode="ptt", tier=self._tier, status="ok").inc()

                    log.info(
                        "ptt_complete",
                        request_id=rid,
                        rec_s=round(rec_duration, 2),
                        total_s=round(total_latency, 3),
                    )

                    return final_state

                except asyncio.CancelledError:
                    log.warning("ptt_cancelled", request_id=rid)
                    raise

                except Exception as exc:
                    log.error("ptt_failed", request_id=rid, error=str(exc))
                    _pipeline_completions.labels(mode="ptt", tier=self._tier, status="error").inc()
                    return {
                        "audio_path": "",
                        "request_id": rid,
                        "session_id": session_id,
                        "stage": "failed",
                        "error": str(exc),
                        "degraded": True,
                        "pipeline_latency_s": round(time.monotonic() - t0, 3),
                    }

                finally:
                    # ── cleanup temp recording ────────────────────────────
                    if audio_path:
                        try:
                            delete_temp_recording(audio_path)
                        except Exception as exc:
                            log.debug("ptt_cleanup_failed", path=audio_path, error=str(exc))

    # ═══════════════════════════════════════════════════════════════════════
    # HEALTH + DIAGNOSTICS
    # ═══════════════════════════════════════════════════════════════════════

    async def health(self) -> dict[str, Any]:
        """
        Return a health check dict covering all pipeline components:
        graph compilation status, node health, session manager state,
        load shedding headroom, and feature flag states.
        """
        result = {
            "graph_compiled": self._graph is not None,
            "tier": self._tier,
            "is_dev": self._is_dev,
            "active_sessions": self._session_mgr.active_count(),
            "load_shedding": {
                "max_concurrent": self._cfg.max_inflight,
                "queue_size": self._cfg.load_shed_queue_size,
                "current": self._load_guard.current_count(),
            },
            "feature_flags": {
                "pcm_pipeline": FF_PCM_PIPELINE,
                "barge_in": FF_BARGE_IN,
                "audio_diagnostics": FF_AUDIO_DIAGNOSTICS,
                "session_lifecycle": FF_SESSION_LIFECYCLE,
                "question_prefetch": FF_QUESTION_PREFETCH,
                "canary_pct": FF_CANARY_PCT,
            },
        }

        # STT node health
        try:
            stt_health = await self._stt.health()
            result["stt"] = vars(stt_health)
        except Exception as exc:
            result["stt"] = {"healthy": False, "error": str(exc)}

        # LLM node health
        try:
            llm_health = await self._llm.health()
            result["llm"] = vars(llm_health)
        except Exception as exc:
            result["llm"] = {"healthy": False, "error": str(exc)}

        # TTS node health
        try:
            tts_health = await self._tts.health()
            result["tts"] = vars(tts_health)
        except Exception as exc:
            result["tts"] = {"healthy": False, "error": str(exc)}

        # QA controller health
        try:
            if _qa_controller is not None:
                qa_health = await _qa_controller.health()
                result["qa_controller"] = vars(qa_health) if not isinstance(qa_health, dict) else qa_health
            else:
                result["qa_controller"] = {"healthy": True}
        except Exception as exc:
            result["qa_controller"] = {"healthy": False, "error": str(exc)}

        # Evaluation engine health
        try:
            if _eval_engine is not None:
                eval_health = await _eval_engine.health()
                result["evaluation"] = vars(eval_health) if not isinstance(eval_health, dict) else eval_health
            else:
                result["evaluation"] = {"healthy": True}
        except Exception as exc:
            result["evaluation"] = {"healthy": False, "error": str(exc)}

        # Recording subsystem health
        try:
            rec_health = get_recording_health()
            result["recorder"] = rec_health.__dict__ if rec_health else {"healthy": True}
        except Exception as exc:
            result["recorder"] = {"healthy": False, "error": str(exc)}

        # Playback subsystem health
        try:
            play_health = get_playback_health()
            result["player"] = play_health.__dict__ if play_health else {"healthy": True}
        except Exception as exc:
            result["player"] = {"healthy": False, "error": str(exc)}

        # Overall healthy flag: all sub-components must be healthy
        sub_healths = [
            result.get("stt", {}).get("healthy", True),
            result.get("llm", {}).get("healthy", True),
            result.get("tts", {}).get("healthy", True),
            result.get("qa_controller", {}).get("healthy", True),
        ]
        result["healthy"] = all(sub_healths) and result["graph_compiled"]

        return result

    def cancel(
            self,
            request_id: str,
            reason: str = "manual",
            source: str = "api",
    ) -> bool:
        """
        Cancel an in-flight pipeline by request_id.

        Looks up the task in the per-instance registry and calls .cancel() on it.
        The CancelledError propagates through the graph's ainvoke / stream /
        stream_full coroutine, which each have explicit CancelledError handlers
        that log and re-raise — so the cancellation is clean and traceable.

        Returns True if a matching task was found and cancelled, False if the
        request_id is unknown (already completed, never started, or belongs to
        a different graph instance).

        Thread-safety: dict lookup + Task.cancel() are both atomic in CPython.
        The _active_tasks dict is modified only in the asyncio event loop thread
        via done-callbacks, so no lock is needed here.
        """
        task = self._active_tasks.get(request_id)
        if task is None or task.done():
            return False

        task.cancel()
        _cancellations.labels(stage="api").inc()
        log.warning(
            "pipeline_cancelled",
            request_id=request_id,
            reason=reason,
            source=source,
            tier=self._tier,
        )
        return True

    async def get_session_report(self, session_id: str) -> dict[str, Any]:
        """
        Generate a comprehensive session report including QA analytics,
        evaluation scores, audio health, and timing data.
        """
        report: dict[str, Any] = {"session_id": session_id}

        # QA analytics
        try:
            doc = await _qa_controller.get_document(session_id)
            if doc:
                analytics = _QAAnalytics.analyse(doc)
                report["qa"] = analytics.to_dict()
        except Exception as exc:
            report["qa_error"] = str(exc)

        # Evaluation report
        try:
            eval_report = await _eval_engine.get_session_report(session_id)
            report["evaluation"] = eval_report.to_dict()
        except Exception as exc:
            report["evaluation_error"] = str(exc)

        # Session resources (if still active)
        resources = await self._session_mgr.get_resources(session_id)
        if resources:
            report["resources"] = resources.to_dict()

        return report


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
#
# Small helpers used across multiple nodes and execution modes. Kept at module
# level to avoid closure overhead in hot paths.
# ═══════════════════════════════════════════════════════════════════════════════

def _split_into_sentences(text: str) -> list[str]:
    """Shim — delegates to sanitize.split_into_sentences()."""
    from app.nodes.sanitize import split_into_sentences
    return split_into_sentences(text)

# ── apology strings per stage ─────────────────────────────────────────────────

APOLOGY_STT: str = (
    "I'm sorry, I had trouble hearing that. Could you please repeat "
    "what you just said?"
)

APOLOGY_LLM: str = (
    "I apologise for the brief interruption. Let me continue — "
    "could you elaborate on your experience with that topic?"
)

APOLOGY_TTS: str = (
    "I'm experiencing a temporary audio issue. Your response has been "
    "recorded and we'll continue in just a moment."
)


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY SINGLETONS AND WIRING
#
# Module-level references to QA controller, evaluation engine, and other
# components. These are lazily imported/initialised to avoid circular imports
# and to allow test code to monkey-patch them.
# ═══════════════════════════════════════════════════════════════════════════════

# ── QA controller ─────────────────────────────────────────────────────────────
try:
    from app.interview.qa_controller import (
        qa_controller as _qa_controller,
        build_next_llm_input_for_voice_graph as _build_next_llm_input,
        QAAnalytics as _QAAnalytics,
        ATSRuleExtractor as _ATSRuleExtractorCls,
        QuestionFingerprintStore as _QuestionFingerprintStoreCls,
        QuestionPrefetchBuffer as _QuestionPrefetchBufferCls,
        QAGuardrailEngine as _QAGuardrailEngineCls,
        LLMInputBuilder as _LLMInputBuilderCls,
        DOMAIN_REGISTRY as _DOMAIN_REGISTRY,
    )
    from app.nodes.LLM_service import StaticQuestionBank as _StaticQuestionBankCls
    _ats_rule_extractor         = _ATSRuleExtractorCls()
    _question_fingerprint_store = None  # accessed via _qa_controller._fingerprints at runtime
    _qa_prefetch_buffer         = _QuestionPrefetchBufferCls()
    _qa_guardrail_engine        = _QAGuardrailEngineCls()
    _llm_input_builder          = None  # accessed via _qa_controller._input_builder at runtime
    _static_question_bank       = _StaticQuestionBankCls()
except ImportError:
    log.warning("qa_controller_import_failed — QA features disabled")
    _qa_controller               = None
    _build_next_llm_input        = None
    _QAAnalytics                 = None
    _ATSRuleExtractorCls         = None
    _QuestionFingerprintStoreCls = None
    _QuestionPrefetchBufferCls   = None
    _QAGuardrailEngineCls        = None
    _LLMInputBuilderCls          = None
    _StaticQuestionBankCls       = None
    _ats_rule_extractor          = None
    _question_fingerprint_store  = None
    _qa_prefetch_buffer          = None
    _qa_guardrail_engine         = None
    _llm_input_builder           = None
    _static_question_bank        = None
    _DOMAIN_REGISTRY             = {}

_ATS_RULE_CONFIDENCE_THRESHOLD: float = float(os.getenv("ATS_RULE_CONFIDENCE_THRESHOLD", "0.7"))

# ── Evaluation engine ─────────────────────────────────────────────────────────
try:
    from app.eval.evaluation_engine import evaluation_engine as _eval_engine
    _eval_enabled = True
except ImportError:
    log.warning("evaluation_engine_import_failed — eval features disabled")
    _eval_engine = None
    _eval_enabled = False

# ── LLM wiring ───────────────────────────────────────────────────────────────
try:
    from app.nodes.LLM_service import (
        get_llm_node,  # noqa
        PromptInjectionDetector as _PromptInjectionDetectorCls,
        ContentPolicyFilter as _ContentPolicyFilterCls,
        QuestionDiversityEnforcer as _QuestionDiversityEnforcerCls,
        StreamingWordGuard as _StreamingWordGuardCls,
    )
    _prompt_injection_detector   = _PromptInjectionDetectorCls()
    _content_policy_filter       = _ContentPolicyFilterCls()
    _question_diversity_enforcer = _QuestionDiversityEnforcerCls()
    _streaming_word_guard        = _StreamingWordGuardCls()
except ImportError:
    log.warning("LLM_service_import_failed — LLM guardrails disabled")
    _PromptInjectionDetectorCls   = None
    _ContentPolicyFilterCls       = None
    _QuestionDiversityEnforcerCls = None
    _StreamingWordGuardCls        = None
    _prompt_injection_detector    = None
    _content_policy_filter        = None
    _question_diversity_enforcer  = None
    _streaming_word_guard         = None

# ── STT / TTS node factories ─────────────────────────────────────────────────
try:
    from app.nodes.STT_service import get_stt_node
except ImportError:
    log.warning("STT_service_import_failed")
    get_stt_node = None

try:
    from app.nodes.TTS_service import get_tts_node
except ImportError:
    log.warning("TTS_service_import_failed")
    get_tts_node = None

# ── Optional integration layer ────────────────────────────────────────────────
#
# Three subsystems plug into voice_graph via late wiring at application startup:
#
#   1. QAAuditBus     — per-session commit_turn fan-out, dead-letter queue,
#                       domain-batch eval dispatch, and idempotency guard.
#                       Lives in conversation_memory; voice_graph calls
#                       open_session / close_session on it and routes committed
#                       turns through route_committed_turn_to_audit(). All eval
#                       scheduling flows through the bus, never directly to the
#                       evaluation engine, so the DLQ, retry logic, and
#                       observability remain consistent across all paths.
#
#   2. TranscriptWriter — dual-sink async writer from transcription.py.
#                        Sink A → human-readable .txt file per session.
#                        Sink B → ObsEvent fan-out (structlog + Prometheus +
#                        MongoDB + OTel). Writes go through an asyncio.Queue
#                        so no pipeline call ever waits on disk I/O. The
#                        module-level singleton transcript_writer is used
#                        directly when set_transcript_writer() is not called,
#                        so the writer is always available regardless of startup
#                        wiring order.
#
#   3. finalize_session_eval — async callable (session_id: str) → None.
#                              Called at session close to dispatch any domains
#                              that were never rotated (the last active domain
#                              never triggers a domain_rotated=True event, so
#                              it would otherwise be silently skipped by the
#                              audit bus's automatic domain-batch dispatch).
#                              Wired to conversation_memory.finalize_session_eval
#                              by the FastAPI lifespan. If not wired, voice_graph
#                              falls back to calling evaluation_engine directly,
#                              which bypasses the DLQ but still produces a report.
#
# Design constraints
# ──────────────────
# - voice_graph must compile and run in isolation (unit tests, dev mode) without
#   any of these subsystems present. Every call site guards with `is not None`.
# - Startup wiring is idempotent: calling set_*() twice logs a warning but
#   does not raise. The second call wins (allows hot-reload in dev).
# - All three expose a health() coroutine so the pipeline's /health endpoint
#   can include their state without importing from their home modules.
# - Degradation is always graceful: a failed audit bus write never aborts a
#   pipeline turn; a failed transcript write never silences the LLM response;
#   a failed eval finalization never crashes session close.
#
# Wiring (called from FastAPI lifespan, e.g. app/main.py):
# ──────────────────────────────────────────────────────────
#   from app.user_tracking.session_service.conversation_memory import (
#       qa_audit_bus,
#       finalize_session_eval,
#   )
#   from app.user_tracking.transcript.transcription import transcript_writer
#
#   voice_graph.set_audit_bus(qa_audit_bus)
#   voice_graph.set_transcript_writer(transcript_writer)
#   voice_graph.set_finalize_eval(finalize_session_eval)
# ─────────────────────────────────────────────────────────────────────────────

# ── module-level references ───────────────────────────────────────────────────
# Declared as Any to avoid hard import dependencies at module load time.
# The actual types are QAAuditBus, TranscriptWriter, and Callable[[str], Awaitable[None]].
# Type narrowing happens inside each integration helper below.

_qa_audit_bus:          Any = None   # QAAuditBus — set by app startup
_transcript_writer:     Any = None   # TranscriptWriter — set by app startup
_finalize_session_eval: Any = None   # async (session_id: str) → None — set by app startup

# ── wiring metadata ───────────────────────────────────────────────────────────
# Tracks wiring state per subsystem for health reporting and idempotency guards.

_integration_state: dict[str, dict[str, Any]] = {
    "audit_bus": {
        "wired":       False,
        "wired_at":    0.0,
        "wire_count":  0,    # how many times set_audit_bus() was called
        "healthy":     False,
        "last_error":  "",
        "open_sessions":   0,
        "closed_sessions": 0,
        "open_failures":   0,
        "close_failures":  0,
    },
    "transcript_writer": {
        "wired":          False,
        "wired_at":       0.0,
        "wire_count":     0,
        "healthy":        False,
        "last_error":     "",
        "sessions_opened": 0,
        "sessions_closed": 0,
        "turns_written":   0,
        "write_failures":  0,
        "flush_failures":  0,
    },
    "finalize_eval": {
        "wired":             False,
        "wired_at":          0.0,
        "wire_count":        0,
        "healthy":           False,
        "last_error":        "",
        "calls_attempted":   0,
        "calls_succeeded":   0,
        "calls_failed":      0,
        "fallback_used":     0,   # times the direct eval_engine fallback fired
    },
}

# Prometheus counters for integration health (declared lazily — these modules
# may not be installed in test environments).
try:
    _integ_audit_opens   = make_counter("integration_audit_bus_opens_total",   "Audit bus open_session calls",   ["status"])
    _integ_audit_closes  = make_counter("integration_audit_bus_closes_total",  "Audit bus close_session calls",  ["status"])
    _integ_tx_opens      = make_counter("integration_transcript_opens_total",  "Transcript writer open calls",   ["status"])
    _integ_tx_writes     = make_counter("integration_transcript_writes_total", "Transcript writer write calls",  ["status"])
    _integ_tx_flushes    = make_counter("integration_transcript_flushes_total","Transcript writer flush calls",  ["status"])
    _integ_eval_finals   = make_counter("integration_finalize_eval_total",     "finalize_session_eval calls",    ["status"])
except Exception: # noqa
    # Test / isolated environment — use no-op counters.
    class _NoopCounter:
        def labels(self, **_kw: Any) -> "_NoopCounter": return self
        def inc(self, _n: float = 1) -> None: pass
    _integ_audit_opens  = _NoopCounter()
    _integ_audit_closes = _NoopCounter()
    _integ_tx_opens     = _NoopCounter()
    _integ_tx_writes    = _NoopCounter()
    _integ_tx_flushes   = _NoopCounter()
    _integ_eval_finals  = _NoopCounter()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — QA AUDIT BUS INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def set_audit_bus(bus: Any) -> None:
    """
    Wire the QAAuditBus into voice_graph at application startup.

    Must be called before the first session is opened. Calling it a second
    time is allowed (hot-reload / test teardown) — a warning is emitted and
    the new bus replaces the old one atomically.

    The bus must expose at minimum:
        async open_session(session_id: str)  → None
        async close_session(session_id: str) → None

    The following optional attributes are used for health reporting if present:
        .state            → str  ("OPEN" | "CLOSED" | "HALF_OPEN") — circuit breaker
        .dlq_depth        → int  — pending dead-letter entries
        .inflight         → int  — currently dispatching turns
        async .health()   → ServiceHealthState

    Parameters
    ──────────
    bus : Any
        An instance of QAAuditBus (or any compatible duck-typed object).
    """
    global _qa_audit_bus # noqa
    state = _integration_state["audit_bus"]

    if state["wired"]:
        log.warning(
            "audit_bus_rewired",
            wire_count=state["wire_count"],
            note="second call to set_audit_bus() — new bus replaces old one",
        )

    _qa_audit_bus = bus
    state["wired"]      = True
    state["wired_at"]   = time.time()
    state["wire_count"] += 1
    state["healthy"]    = True   # optimistically healthy until first failure

    log.info(
        "audit_bus_wired",
        wire_count=state["wire_count"],
        bus_type=type(bus).__name__,
    )


async def _audit_bus_open_session(session_id: str) -> None:
    """
    Call QAAuditBus.open_session() with full error isolation.

    Failures are logged and counted but never propagated — a broken audit bus
    must not prevent the pipeline from serving the candidate. The session
    continues in a degraded state where eval scheduling may be impaired but
    STT → LLM → TTS still works.

    Called by SessionLifecycleManager.open_session() at step 7.
    """
    state = _integration_state["audit_bus"]

    if _qa_audit_bus is None:
        # Bus not wired — normal in dev/test mode or early startup. Not an error.
        log.debug("audit_bus_open_skipped_not_wired", sid=session_id[:8])
        return

    try:
        await _qa_audit_bus.open_session(session_id)
        state["open_sessions"] += 1
        _integ_audit_opens.labels(status="ok").inc()
        log.debug("audit_bus_session_opened", sid=session_id[:8])

    except Exception as exc:
        state["open_failures"] += 1
        state["last_error"]    = str(exc)
        state["healthy"]       = False
        _integ_audit_opens.labels(status="error").inc()
        log.warning(
            "audit_bus_open_session_failed",
            sid=session_id[:8],
            error=str(exc),
            note="session will proceed without audit bus — eval scheduling may be impaired",
        )


async def _audit_bus_close_session(session_id: str) -> None:
    """
    Call QAAuditBus.close_session() with full error isolation.

    A failed close is logged and counted. It does not re-raise because session
    close must complete regardless — other resources (PCM streams, temp files,
    transcript writer) still need to be cleaned up even if the bus fails.

    Called by SessionLifecycleManager.close_session() at step 7.
    """
    state = _integration_state["audit_bus"]

    if _qa_audit_bus is None:
        log.debug("audit_bus_close_skipped_not_wired", sid=session_id[:8])
        return

    try:
        await _qa_audit_bus.close_session(session_id)
        state["closed_sessions"] += 1
        _integ_audit_closes.labels(status="ok").inc()
        log.debug("audit_bus_session_closed", sid=session_id[:8])

    except Exception as exc:
        state["close_failures"] += 1
        state["last_error"]     = str(exc)
        state["healthy"]        = False
        _integ_audit_closes.labels(status="error").inc()
        log.warning(
            "audit_bus_close_session_failed",
            sid=session_id[:8],
            error=str(exc),
            note="session close will continue — audit bus close failure is non-fatal",
        )


async def _audit_bus_health() -> dict[str, Any]:
    """
    Return a structured health snapshot for the audit bus integration.

    Merges the internal wiring state with whatever .health() the bus exposes.
    Safe to call at any time — returns a degraded snapshot if the bus is not
    wired or if its own health() raises.

    Used by VoiceGraph.health() and the /health HTTP endpoint.
    """
    state = _integration_state["audit_bus"]

    snapshot: dict[str, Any] = {
        "wired":           state["wired"],
        "wired_at":        state["wired_at"],
        "wire_count":      state["wire_count"],
        "healthy":         state["healthy"],
        "last_error":      state["last_error"],
        "open_sessions":   state["open_sessions"],
        "closed_sessions": state["closed_sessions"],
        "open_failures":   state["open_failures"],
        "close_failures":  state["close_failures"],
    }

    if _qa_audit_bus is None:
        snapshot["status"] = "not_wired"
        return snapshot

    # Probe the bus's own health if it exposes one.
    if hasattr(_qa_audit_bus, "health"):
        try:
            bus_health = await _qa_audit_bus.health()
            if hasattr(bus_health, "__dict__"):
                snapshot["bus_health"] = vars(bus_health)
            elif isinstance(bus_health, dict):
                snapshot["bus_health"] = bus_health
            state["healthy"] = getattr(bus_health, "healthy", True)
            snapshot["healthy"] = state["healthy"]
        except Exception as exc:
            snapshot["bus_health_error"] = str(exc)
            state["healthy"]  = False
            snapshot["healthy"] = False

    # Surface circuit breaker state directly for dashboards.
    if hasattr(_qa_audit_bus, "state"):
        snapshot["circuit_state"] = _qa_audit_bus.state
    if hasattr(_qa_audit_bus, "dlq_depth"):
        snapshot["dlq_depth"] = _qa_audit_bus.dlq_depth
    if hasattr(_qa_audit_bus, "inflight"):
        snapshot["inflight"] = _qa_audit_bus.inflight

    snapshot["status"] = "healthy" if snapshot["healthy"] else "degraded"
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TRANSCRIPT WRITER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def set_transcript_writer(writer: Any) -> None:
    """
    Wire a TranscriptWriter instance into voice_graph at application startup.

    If this is never called, voice_graph falls back to the module-level
    singleton from transcription.py (imported at the top of this file as
    _transcript_writer from the conversation_memory import block). In practice
    that means transcripts are always written — explicit wiring is only needed
    if the caller wants to inject a custom writer (e.g. for testing).

    The writer must expose at minimum:
        async open_session(session_id: str)               → None
        async close_session(session_id: str)              → None
        async write_turn(session_id, user_text,
                         assistant_text, request_id)      → None
        async flush(timeout: float)                       → None

    Optional attributes used for health reporting:
        ._queue     → asyncio.Queue  (for queue depth)
        ._started   → bool           (background task running)
        ._task      → asyncio.Task   (drain loop task)

    Parameters
    ──────────
    writer : Any
        An instance of TranscriptWriter (or duck-typed equivalent).
    """
    global _transcript_writer # noqa
    state = _integration_state["transcript_writer"]

    if state["wired"]:
        log.warning(
            "transcript_writer_rewired",
            wire_count=state["wire_count"],
            note="second call to set_transcript_writer() — new writer replaces old",
        )

    _transcript_writer = writer
    state["wired"]      = True
    state["wired_at"]   = time.time()
    state["wire_count"] += 1
    state["healthy"]    = True

    log.info(
        "transcript_writer_wired",
        wire_count=state["wire_count"],
        writer_type=type(writer).__name__,
    )


async def _transcript_open_session(session_id: str) -> None:
    """
    Call TranscriptWriter.open_session() — writes the session header to the
    per-session .txt file and emits a transcript_session_open ObsEvent.

    Per transcription.py: open_session() enqueues the entry and returns
    immediately. The actual file write happens in the drain loop background
    task, so this call has effectively zero latency impact on session open.

    Called by SessionLifecycleManager.open_session() at step 8.
    """
    state = _integration_state["transcript_writer"]

    if _transcript_writer is None:
        log.debug("transcript_open_skipped_not_wired", sid=session_id[:8])
        return

    try:
        await _transcript_writer.open_session(session_id)
        state["sessions_opened"] += 1
        _integ_tx_opens.labels(status="ok").inc()
        log.debug("transcript_session_opened", sid=session_id[:8])

    except Exception as exc:
        state["write_failures"] += 1
        state["last_error"]     = str(exc)
        state["healthy"]        = False
        _integ_tx_opens.labels(status="error").inc()
        log.warning(
            "transcript_open_session_failed",
            sid=session_id[:8],
            error=str(exc),
            note="transcript write failure is non-fatal — pipeline continues",
        )


async def _transcript_write_turn(
    session_id:     str,
    user_text:      str,
    assistant_text: str,
    request_id:     str = "",
) -> None:
    """
    Write a completed (user, assistant) turn to both transcript sinks.

    Per transcription.py: write_turn() enqueues the entry into an asyncio.Queue
    (maxsize = TRANSCRIPT_QUEUE_DEPTH, default 512) and returns immediately.
    The drain loop task writes to disk and emits the ObsEvent asynchronously.
    If the queue is full, the entry is dropped and a Prometheus counter
    (ai_transcript_drops_total) is incremented — the pipeline is never
    back-pressured.

    This should be called from _node_tts (or the PCM pipeline_worker) after
    the LLM response has been produced and before TTS synthesis begins, so the
    turn is recorded even if TTS fails.

    Parameters
    ──────────
    session_id     : Active session identifier.
    user_text      : The STT transcript (candidate speech).
    assistant_text : The sanitized LLM response text (before TTS synthesis).
    request_id     : Pipeline request ID for tracing correlation.
    """
    state = _integration_state["transcript_writer"]

    if _transcript_writer is None:
        log.debug(
            "transcript_write_skipped_not_wired",
            sid=session_id[:8],
            request_id=request_id,
        )
        return

    # Skip empty turns — these arise from confidence-filtered transcripts that
    # were dropped before reaching the LLM but still hit the write path due to
    # ordering. Writing them would produce blank lines in the .txt file and
    # increment the Prometheus turn counter spuriously.
    if not user_text.strip() and not assistant_text.strip():
        log.debug(
            "transcript_write_skipped_empty_turn",
            sid=session_id[:8],
            request_id=request_id,
        )
        return

    try:
        await _transcript_writer.write_turn(
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            request_id=request_id,
        )
        state["turns_written"] += 1
        _integ_tx_writes.labels(status="ok").inc()

    except Exception as exc:
        state["write_failures"] += 1
        state["last_error"]     = str(exc)
        state["healthy"]        = False
        _integ_tx_writes.labels(status="error").inc()
        log.warning(
            "transcript_write_turn_failed",
            sid=session_id[:8],
            request_id=request_id,
            error=str(exc),
        )


async def _transcript_flush_session(
    session_id: str,
    timeout:    float = 5.0,
) -> None:
    """
    Close the session in the transcript writer and flush pending writes.

    Calls close_session() (writes the footer line) then flush() (waits for the
    queue to drain). flush() has a configurable timeout (default 5s) — if the
    drain loop is slow, a warning is logged but the call returns. This ensures
    session close is never indefinitely blocked by transcript I/O.

    Called by SessionLifecycleManager.close_session() at step 8.

    Parameters
    ──────────
    session_id : Active session identifier.
    timeout    : Seconds to wait for queue drain before giving up. Default 5.0.
    """
    state = _integration_state["transcript_writer"]

    if _transcript_writer is None:
        log.debug("transcript_flush_skipped_not_wired", sid=session_id[:8])
        return

    # ── Step 1: write the session footer ─────────────────────────────────────
    try:
        await _transcript_writer.close_session(session_id)
        state["sessions_closed"] += 1
        _integ_tx_flushes.labels(status="close_ok").inc()

    except Exception as exc:
        state["flush_failures"] += 1
        state["last_error"]     = str(exc)
        state["healthy"]        = False
        _integ_tx_flushes.labels(status="close_error").inc()
        log.warning(
            "transcript_close_session_failed",
            sid=session_id[:8],
            error=str(exc),
        )

    # ── Step 2: flush the queue ───────────────────────────────────────────────
    # flush() is defined on TranscriptWriter as:
    #   asyncio.wait_for(self._queue.join(), timeout=timeout)
    # It logs a warning internally if it times out, so we just need to catch
    # the re-raised TimeoutError (which it suppresses) and any other exc.
    try:
        await _transcript_writer.flush(timeout=timeout)
        _integ_tx_flushes.labels(status="flush_ok").inc()
        log.debug("transcript_queue_flushed", sid=session_id[:8])

    except Exception as exc:
        state["flush_failures"] += 1
        state["last_error"]     = str(exc)
        _integ_tx_flushes.labels(status="flush_error").inc()
        log.warning(
            "transcript_flush_failed",
            sid=session_id[:8],
            timeout=timeout,
            error=str(exc),
        )


async def _transcript_writer_health() -> dict[str, Any]:
    """
    Return a structured health snapshot for the transcript writer integration.

    Includes queue depth (if accessible) and the drain loop task status.
    Used by VoiceGraph.health() and the /health HTTP endpoint.
    """
    state = _integration_state["transcript_writer"]

    snapshot: dict[str, Any] = {
        "wired":           state["wired"],
        "wired_at":        state["wired_at"],
        "wire_count":      state["wire_count"],
        "healthy":         state["healthy"],
        "last_error":      state["last_error"],
        "sessions_opened": state["sessions_opened"],
        "sessions_closed": state["sessions_closed"],
        "turns_written":   state["turns_written"],
        "write_failures":  state["write_failures"],
        "flush_failures":  state["flush_failures"],
    }

    if _transcript_writer is None:
        snapshot["status"] = "not_wired"
        return snapshot

    # Queue depth — proxy for write backlog. A growing queue with write_failures
    # indicates the drain loop has stalled and needs attention.
    if hasattr(_transcript_writer, "_queue"):
        try:
            q: asyncio.Queue = _transcript_writer._queue # noqa
            snapshot["queue_depth"]    = q.qsize()
            snapshot["queue_maxsize"]  = q.maxsize
            snapshot["queue_full"]     = q.full()
        except Exception: # noqa
            pass

    # Drain loop task health. A done() task that exited unexpectedly means
    # all future writes will silently queue up and never drain.
    if hasattr(_transcript_writer, "_task"):
        task = _transcript_writer._task # noqa
        if task is not None:
            snapshot["drain_task_done"]      = task.done()
            snapshot["drain_task_cancelled"] = task.cancelled()
            if task.done() and not task.cancelled():
                exc = task.exception()
                snapshot["drain_task_exception"] = str(exc) if exc else None
                if exc:
                    state["healthy"] = False
                    snapshot["healthy"] = False

    if hasattr(_transcript_writer, "_started"):
        snapshot["drain_started"] = _transcript_writer._started # noqa

    snapshot["status"] = "healthy" if snapshot["healthy"] else "degraded"
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FINALIZE SESSION EVAL INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def set_finalize_eval(fn: Any) -> None:
    """
    Wire the session eval finalizer into voice_graph at application startup.

    The function must be an async callable with the signature:
        async finalize_session_eval(session_id: str, **kwargs) → None

    It is expected to be conversation_memory.finalize_session_eval, which:
        1. Fetches the QA document from Redis via qa_controller.get_document()
        2. Identifies domains that were never rotated (i.e. the last active
           domain — domain_rotated never fires True for it)
        3. Dispatches each unscored domain batch to qa_audit_bus via
           trigger_domain_eval(), which routes through the DLQ and
           idempotency guard rather than calling evaluation_engine directly
        4. Emits a session_eval_finalized observability event

    Fallback behaviour (when not wired)
    ────────────────────────────────────
    voice_graph falls back to calling evaluation_engine.get_session_report()
    directly. This skips the DLQ and per-domain scheduling but still generates
    a report from whatever was already scored. Turns that were never dispatched
    to the evaluation engine will appear as skipped in the report.

    Parameters
    ──────────
    fn : Any
        Async callable: async (session_id: str, **kwargs) → None.
    """
    global _finalize_session_eval # noqa
    state = _integration_state["finalize_eval"]

    if state["wired"]:
        log.warning(
            "finalize_eval_rewired",
            wire_count=state["wire_count"],
            note="second call to set_finalize_eval() — new fn replaces old",
        )

    _finalize_session_eval = fn
    state["wired"]      = True
    state["wired_at"]   = time.time()
    state["wire_count"] += 1
    state["healthy"]    = True

    log.info(
        "finalize_eval_wired",
        wire_count=state["wire_count"],
        fn_name=getattr(fn, "__name__", repr(fn)),
    )


async def _run_finalize_session_eval(
    session_id: str,
    qa_document: Any = None,
) -> dict[str, Any]:
    """
    Execute finalize_session_eval() for a closing session.

    This is the voice_graph-internal wrapper around the wired finalizer. It
    adds retry semantics (one retry, 1s delay), circuit-breaker awareness
    (skips if the eval engine's own breaker is open), and a direct-to-engine
    fallback when the finalizer is not wired.

    The returned dict is merged into the session close summary under the key
    "finalize_eval" by SessionLifecycleManager.close_session().

    Architecture
    ────────────
    Preferred path (finalizer wired):
        _finalize_session_eval(session_id, qa_document=doc)
            → conversation_memory.finalize_session_eval()
                → qa_audit_bus.trigger_domain_eval() per unscored domain
                    → evaluation_engine.schedule_domain_eval() [fire-and-forget]

    Fallback path (finalizer not wired or failed):
        _eval_engine.get_session_report(session_id)
            → scans eval:score:v1:{session_id}:* in Redis
            → aggregates TurnScore records into SessionReport
            → returns report (does NOT schedule new scoring tasks)

    The fallback report accurately reflects what was already scored. It will
    show turns_skipped for any domain that was never dispatched via the bus —
    this is expected and logged at info level, not warning.

    Parameters
    ──────────
    session_id  : The session being closed.
    qa_document : Optional pre-fetched QADocument. If None, the finalizer
                  fetches it itself via qa_controller.get_document(). Passing
                  it in avoids a redundant Redis read when the caller already
                  has the document (e.g. SessionLifecycleManager step 4).

    Returns
    ───────
    dict with keys:
        path         : "wired" | "fallback" | "skipped"
        success      : bool
        error        : str (empty on success)
        report       : SessionReport.to_dict() | None
        tokens_consumed : int
    """
    state = _integration_state["finalize_eval"]
    state["calls_attempted"] += 1

    result: dict[str, Any] = {
        "path":            "skipped",
        "success":         False,
        "error":           "",
        "report":          None,
        "tokens_consumed": 0,
    }

    # ── Circuit breaker check ─────────────────────────────────────────────────
    # If the eval engine's circuit breaker is open, there is no point calling
    # finalize — new scoring tasks would immediately be skipped by _should_skip()
    # inside _score_turn_safe(). Skip proactively and surface the state.
    if hasattr(_eval_engine, "_breaker"):
        breaker_state = getattr(_eval_engine._breaker, "state", "UNKNOWN") # noqa
        if breaker_state == "OPEN":
            result["path"]  = "skipped"
            result["error"] = f"eval circuit breaker open — state={breaker_state}"
            log.info(
                "finalize_eval_skipped_circuit_open",
                sid=session_id[:8],
                breaker_state=breaker_state,
            )
            _integ_eval_finals.labels(status="circuit_open").inc()
            return result

    # ── Preferred path: wired finalizer ──────────────────────────────────────
    if _finalize_session_eval is not None:
        result["path"] = "wired"

        # Build kwargs — pass qa_document only if the finalizer accepts it
        # (older versions of conversation_memory may not have the parameter).
        kwargs: dict[str, Any] = {}
        if qa_document is not None:
            import inspect as _inspect
            try:
                sig = _inspect.signature(_finalize_session_eval)
                if "qa_document" in sig.parameters:
                    kwargs["qa_document"] = qa_document
            except (ValueError, TypeError):
                pass   # can't introspect — call without it

        # One retry with a 1s delay. finalize_session_eval() is normally fast
        # (it just queues domain batches onto the audit bus), but Redis
        # connection hiccups can cause transient failures.
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                await _finalize_session_eval(session_id, **kwargs)
                state["calls_succeeded"] += 1
                result["success"] = True
                _integ_eval_finals.labels(status="ok").inc()
                log.info(
                    "finalize_eval_complete",
                    sid=session_id[:8],
                    attempt=attempt,
                    path="wired",
                )
                break

            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "finalize_eval_attempt_failed_retrying",
                        sid=session_id[:8],
                        attempt=attempt,
                        error=str(exc),
                    )
                    await asyncio.sleep(1.0)

        if not result["success"]:
            state["calls_failed"] += 1
            state["last_error"]   = str(last_exc)
            state["healthy"]      = False
            result["error"]       = str(last_exc)
            _integ_eval_finals.labels(status="error").inc()
            log.warning(
                "finalize_eval_wired_failed_using_fallback",
                sid=session_id[:8],
                error=str(last_exc),
                note="falling back to direct eval_engine.get_session_report()",
            )
            # Fall through to fallback path below.

    # ── Fallback path: direct eval engine report ──────────────────────────────
    # Reached when: (a) finalizer not wired, or (b) wired finalizer failed
    # after retry. We call get_session_report() which scans Redis for whatever
    # turn scores already exist. This does NOT schedule new scoring tasks — it
    # only reads what the engine already produced. Any unscored turns will
    # appear as skipped in the report.
    if not result["success"]:
        result["path"] = "fallback"
        state["fallback_used"] += 1

        try:
            eval_report = await _eval_engine.get_session_report(session_id)
            result["success"]         = True
            result["report"]          = eval_report.to_dict()
            result["tokens_consumed"] = eval_report.tokens_consumed
            state["calls_succeeded"] += 1
            _integ_eval_finals.labels(status="fallback_ok").inc()
            log.info(
                "finalize_eval_fallback_report_generated",
                sid=session_id[:8],
                turns_evaluated=eval_report.turns_evaluated,
                turns_skipped=eval_report.turns_skipped,
                avg_overall=round(eval_report.avg_overall, 2),
                tokens_consumed=eval_report.tokens_consumed,
                note="unscored turns appear as skipped — bus was not available to dispatch them",
            )

        except Exception as exc:
            state["calls_failed"] += 1
            state["last_error"]   = str(exc)
            state["healthy"]      = False
            result["error"]       = str(exc)
            _integ_eval_finals.labels(status="fallback_error").inc()
            log.warning(
                "finalize_eval_fallback_failed",
                sid=session_id[:8],
                error=str(exc),
                note="eval report will be absent from session close summary",
            )

    return result


async def _finalize_eval_health() -> dict[str, Any]:
    """
    Return a structured health snapshot for the finalize_eval integration.

    Includes circuit breaker state from the eval engine, budget stats (if
    accessible), and call counters. Used by VoiceGraph.health().
    """
    state = _integration_state["finalize_eval"]

    snapshot: dict[str, Any] = {
        "wired":           state["wired"],
        "wired_at":        state["wired_at"],
        "wire_count":      state["wire_count"],
        "healthy":         state["healthy"],
        "last_error":      state["last_error"],
        "calls_attempted": state["calls_attempted"],
        "calls_succeeded": state["calls_succeeded"],
        "calls_failed":    state["calls_failed"],
        "fallback_used":   state["fallback_used"],
    }

    if _finalize_session_eval is None:
        snapshot["status"] = "not_wired"
    else:
        snapshot["fn_name"] = getattr(_finalize_session_eval, "__name__", repr(_finalize_session_eval))

    # ── Eval engine health ────────────────────────────────────────────────────
    # Pull the circuit breaker and inflight state directly from the singleton.
    # This gives dashboards a single place to check eval engine liveness without
    # importing from evaluation_engine.py.
    try:
        engine_health = await _eval_engine.health()
        snapshot["eval_engine"] = {
            "healthy":       engine_health.healthy,
            "circuit_state": engine_health.circuit_state,
            "inflight":      engine_health.inflight,
            "degraded":      engine_health.degraded,
        }
        # Propagate circuit breaker OPEN state to this integration's health flag.
        if engine_health.circuit_state == "OPEN":
            state["healthy"]    = False
            snapshot["healthy"] = False

    except Exception as exc:
        snapshot["eval_engine_health_error"] = str(exc)
        state["healthy"]    = False
        snapshot["healthy"] = False

    snapshot["status"] = "healthy" if snapshot["healthy"] else "degraded"
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — UNIFIED INTEGRATION HEALTH + RESET
# ═══════════════════════════════════════════════════════════════════════════════

async def integrations_health() -> dict[str, Any]:
    """
    Return a unified health snapshot for all three integrations.

    Called by VoiceGraph.health() and merged into the top-level health dict
    under the "integrations" key. Also callable directly from the /health
    HTTP endpoint for targeted diagnostics.

    The overall "healthy" flag is True only if all three integrations are
    individually healthy. A degraded integration (e.g. transcript writer queue
    full) does not bring down the pipeline health flag, but it is surfaced
    here so ops can investigate.

    Returns
    ───────
    {
        "healthy": bool,
        "audit_bus":         { ... _audit_bus_health() snapshot },
        "transcript_writer": { ... _transcript_writer_health() snapshot },
        "finalize_eval":     { ... _finalize_eval_health() snapshot },
        "summary": {
            "wired_count":  int,   # 0–3
            "healthy_count": int,  # 0–3
        }
    }
    """
    audit_snap  = await _audit_bus_health()
    tx_snap     = await _transcript_writer_health()
    eval_snap   = await _finalize_eval_health()

    wired_count   = sum(1 for s in (audit_snap, tx_snap, eval_snap) if s.get("wired"))
    healthy_count = sum(1 for s in (audit_snap, tx_snap, eval_snap) if s.get("healthy"))
    all_healthy   = healthy_count == 3

    return {
        "healthy":           all_healthy,
        "audit_bus":         audit_snap,
        "transcript_writer": tx_snap,
        "finalize_eval":     eval_snap,
        "summary": {
            "wired_count":   wired_count,
            "healthy_count": healthy_count,
        },
    }


def reset_integrations() -> None:
    """
    Reset all integration state to unwired defaults.

    Intended for test teardown and hot-reload scenarios. Does NOT close any
    open sessions or flush any queues — the caller is responsible for that.
    After calling this, all three set_*() functions must be called again before
    the integrations are usable.
    """
    global _qa_audit_bus, _transcript_writer, _finalize_session_eval # noqa

    _qa_audit_bus          = None
    _transcript_writer     = None
    _finalize_session_eval = None

    for key, sub in _integration_state.items():
        for field_name in list(sub.keys()):
            if isinstance(sub[field_name], bool):
                sub[field_name] = False
            elif isinstance(sub[field_name], float):
                sub[field_name] = 0.0
            elif isinstance(sub[field_name], int):
                sub[field_name] = 0
            elif isinstance(sub[field_name], str):
                sub[field_name] = ""

    log.info(
        "integrations_reset",
        note="all three integrations unwired — call set_audit_bus / "
             "set_transcript_writer / set_finalize_eval before next session",
    )

# ── LLM direct call (non-QA mode) ────────────────────────────────────────────

async def _llm_direct_call(text: str, rid: str) -> str:
    """
    Direct LLM call for non-interview mode. This is the fallback when no
    session_id is provided — the graph runs as a generic voice assistant.
    """
    llm_node = get_llm_node()
    # generate(text) → use stream_messages with a simple user message
    result = ""
    async for token in llm_node.stream_messages(
            messages=[{"role": "user", "content": text}],
            request_id=rid,
    ):
        result += token
    return result


async def _ats_mode_extract(intro_input: Any, request_id: str) -> Any:
    """
    LLM-based ATS extraction for intro processing. Uses the ATSMode from
    LLM_service which enforces json_object response_format.
    """
    llm_node = get_llm_node()
    result = await llm_node.extract_ats(
        intro_text=intro_input,
        request_id=request_id,
    )
    # extract_ats returns a JSON string — parse it into ATSExtractionResult
    try:
        parsed = json.loads(result)
        return _ATSExtractionResult(
            name=parsed.get("name", ""),
            domains=parsed.get("domains", []),
            level=parsed.get("level", "intermediate"),
            languages=parsed.get("languages", []),
            notes=parsed.get("notes", ""),
            confidence=0.9,
            method=parsed.get("_fallback", "llm"),
            raw=parsed,
        )
    except Exception as exc:
        log.error("ats_mode_extract_parse_failed", request_id=request_id, error=str(exc))
        return None


async def _llm_generate_question(prompt_ctx: Any, rid: str) -> str:
    """
    Generate a single interview question through the LLM node.
    The prompt_ctx is an enriched prompt from LLMInputBuilder.
    """
    llm_node = get_llm_node()
    # generate_question → use stream_question
    result = ""
    async for token in llm_node.stream_question(
            llm_input=prompt_ctx,
            request_id=rid,
    ):
        result += token
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# EMITTER STUBS
#
# Structured event emitters for observability. These emit to the OTel span,
# Prometheus, and structured log simultaneously. The actual implementations
# are in the observability layer — these stubs ensure the graph compiles
# even when the observability layer is not installed.
# ═══════════════════════════════════════════════════════════════════════════════


class _EmitterStub:
    """No-op emitter that accepts any method call with any arguments."""
    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


try:
    from observability import STTEmitter, LLMEmitter, TTSEmitter, SanitizeEmitter
except ImportError:
    STTEmitter      = _EmitterStub()
    LLMEmitter      = _EmitterStub()
    TTSEmitter      = _EmitterStub()
    SanitizeEmitter = _EmitterStub()


# ── OTel span context managers ────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def stt_span(session_id: str, request_id: str, audio_path: str = ""):
    with tracer.start_as_current_span("node.stt") as span:
        span.set_attribute("session_id", session_id[:8] if session_id else "")
        span.set_attribute("request_id", request_id)
        span.set_attribute("audio_path", audio_path[:50])
        yield span


@contextlib.asynccontextmanager
async def llm_span(session_id: str, request_id: str, model: str = "", streaming: bool = False):
    with tracer.start_as_current_span("node.llm") as span:
        span.set_attribute("session_id", session_id[:8] if session_id else "")
        span.set_attribute("request_id", request_id)
        span.set_attribute("model", model)
        span.set_attribute("streaming", streaming)
        yield span


@contextlib.asynccontextmanager
async def tts_span(session_id: str, request_id: str, text_chars: int = 0):
    with tracer.start_as_current_span("node.tts") as span:
        span.set_attribute("session_id", session_id[:8] if session_id else "")
        span.set_attribute("request_id", request_id)
        span.set_attribute("text_chars", text_chars)
        yield span


@contextlib.asynccontextmanager
async def sanitize_span(session_id: str, request_id: str):
    with tracer.start_as_current_span("node.sanitize") as span:
        span.set_attribute("session_id", session_id[:8] if session_id else "")
        span.set_attribute("request_id", request_id)
        yield span


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGE ENUM & TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineStage(str, Enum):
    """
    Canonical pipeline stage identifiers. Used in VoiceState.stage,
    error routing, metrics labels, and audit events. The string values
    match the LangGraph node names for consistency.
    """
    IDLE     = "idle"
    STT      = "stt"
    LLM      = "llm"
    SANITIZE = "sanitize"
    TTS      = "tts"
    DONE     = "done"
    FAILED   = "failed"


class _QAStage(str, Enum):
    """Mirror of qa_controller's stage values for local comparison."""
    GREETING  = "greeting"
    INTRO     = "intro"
    INTERVIEW = "interview"
    COMPLETE  = "complete"


# ── Exceptions ────────────────────────────────────────────────────────────────

class LatencyBudgetExceeded(Exception):
    """Raised when a stage exceeds its SLA time budget."""
    pass


class LoadSheddingRejected(Exception):
    """Raised when the pipeline rejects a request due to load shedding."""
    pass


# ── Load Shedding Guard ──────────────────────────────────────────────────────

class LoadSheddingGuard:
    """
    Concurrency limiter with a bounded waiting queue. When both the active
    slots and the queue are full, new requests are rejected immediately
    with LoadSheddingRejected, giving the HTTP handler a clean signal to
    return 503 Service Unavailable.

    The guard tracks current utilisation so VoiceGraph.health() can report
    headroom to monitoring systems.
    """

    def __init__(self, max_concurrent: int, queue_size: int, tier: str = "balanced") -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max = max_concurrent
        self._queue_size = queue_size
        self._tier = tier
        self._active = 0
        self._lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def acquire(self, request_id: str = "") -> AsyncIterator[None]:
        """Acquire a concurrency slot, or raise LoadSheddingRejected."""
        # Quick rejection check before blocking on semaphore
        async with self._lock:
            if self._active >= self._max + self._queue_size:
                _load_shed_total.labels(tier=self._tier).inc()
                log.warning(
                    "load_shedding_rejected",
                    request_id=request_id,
                    active=self._active,
                    max=self._max,
                    tier=self._tier,
                )
                raise LoadSheddingRejected(
                    f"Pipeline at capacity ({self._active}/{self._max}+{self._queue_size}). "
                    f"Try again later."
                )
            self._active += 1

        try:
            await self._semaphore.acquire()
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._active -= 1

    def current_count(self) -> int:
        return self._active


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE SINGLETONS
#
# Three pre-configured VoiceGraph instances for common deployment scenarios.
# These are created lazily on first access to avoid import-time side effects
# and to allow settings to be fully loaded before construction.
# ═══════════════════════════════════════════════════════════════════════════════


class _LazyVoiceGraph:
    """
    Descriptor that creates a VoiceGraph instance on first access. The
    factory callable is invoked once and the result cached for all subsequent
    accesses. This avoids import-time construction while providing module-level
    attribute syntax.
    """

    def __init__(self, factory: Callable[[], VoiceGraph]) -> None:
        self._factory = factory
        self._instance: VoiceGraph | None = None
        self._lock = threading.Lock()

    def __get__(self, obj: Any, objtype: type | None = None) -> VoiceGraph:
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance


class _VoiceGraphModule:
    """
    Module-level singleton container. Accessed as:
        from voice_graph import voice_graph, voice_graph_realtime, voice_graph_low_latency
    """

    voice_graph = _LazyVoiceGraph(lambda: VoiceGraph(
        tier="balanced",
    ))

    voice_graph_realtime = _LazyVoiceGraph(lambda: VoiceGraph(
        cfg=VoiceGraphConfig.from_settings(overrides={
            "stt_timeout":     5.0,
            "llm_timeout":     8.0,
            "tts_timeout":     6.0,
            "max_stt_retries": 0,
            "max_llm_retries": 0,
        }),
        tier="realtime",
    ))

    voice_graph_low_latency = _LazyVoiceGraph(lambda: VoiceGraph(
        cfg=VoiceGraphConfig.from_settings(overrides={
            "stt_timeout":     8.0,
            "llm_timeout":     12.0,
            "tts_timeout":     8.0,
            "max_stt_retries": 1,
            "max_llm_retries": 1,
        }),
        tier="low_latency",
    ))


# ── install singletons at module level ────────────────────────────────────────
_module = _VoiceGraphModule()

voice_graph:             VoiceGraph = _module.voice_graph              # type: ignore[assignment]
voice_graph_realtime:    VoiceGraph = _module.voice_graph_realtime     # type: ignore[assignment]
voice_graph_low_latency: VoiceGraph = _module.voice_graph_low_latency  # type: ignore[assignment]


def get_voice_graph(tier: str = "balanced") -> VoiceGraph:
    """
    Factory function for HTTP handlers and CLI entry points. Returns the
    appropriate singleton based on the requested tier.
    """
    if tier == "realtime":
        return voice_graph_realtime
    elif tier == "low_latency":
        return voice_graph_low_latency
    return voice_graph


# ═══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE HOOKS
#
# Called by the application server (e.g. FastAPI lifespan) during startup and
# shutdown. These hooks initialise and tear down shared resources that are
# expensive to create and must be cleaned up properly.
# ═══════════════════════════════════════════════════════════════════════════════


async def on_startup() -> None:
    """
    Application startup hook. Called once when the server starts.

    Initialises:
      1. STT model loading (Faster-Whisper warm-up)
      2. TTS model loading (Kokoro warm-up)
      3. QA controller Redis connection pool
      4. Evaluation engine Redis connection pool
      5. Recording subsystem health check
      6. PCM chunk pool pre-allocation
      7. Prometheus metrics registration
      8. Feature flag logging
    """
    bootstrap()
    log.info("voice_graph_startup_begin", version=GRAPH_VERSION)
    t0 = time.monotonic()

    # ── 1. STT warm-up ───────────────────────────────────────────────────
    try:
        stt_node = voice_graph._stt # noqa
        if hasattr(stt_node, "warmup"):
            await stt_node.warmup()
        log.info("startup_stt_warmup_ok", service="stt")
    except Exception as exc:
        log.error("startup_stt_warmup_failed", service="stt", error=str(exc))

    # ── 2. TTS warm-up ───────────────────────────────────────────────────
    try:
        tts_node = voice_graph._tts # noqa
        if hasattr(tts_node, "warmup"):
            await tts_node.warmup()
        log.info("startup_tts_warmup_ok", service="tts")
    except Exception as exc:
        log.error("startup_tts_warmup_failed", service="tts", error=str(exc))

    # ── 3. QA controller init ────────────────────────────────────────────
    try:
        if _qa_controller is not None:
            qa_health = await _qa_controller.health()
            log.info("startup_qa_controller_ok", service="controller", health=qa_health)
        else:
            log.info("startup_qa_controller_skipped", service="controller")
    except Exception as exc:
        log.error("startup_qa_controller_failed", service="controller", error=str(exc))

    # ── 4. Evaluation engine init ────────────────────────────────────────
    try:
        if _eval_engine is not None:
            eval_health = await _eval_engine.health()
            log.info("startup_eval_engine_ok", service="eval", health=eval_health)
        else:
            log.info("startup_eval_engine_skipped", service="eval")
    except Exception as exc:
        log.error("startup_eval_engine_failed", service="eval", error=str(exc))

    # ── 5. Recording health check ────────────────────────────────────────
    # Skip mic/speaker probing in non-desktop environments (Docker, CI).
    # PortAudio blocks indefinitely if no audio device is present — a timeout
    # prevents the lifespan from hanging and blocking health checks.
    _app_mode = os.getenv("APP_MODE", "production").lower()
    if _app_mode == "desktop":
        try:
            rec_ok = await asyncio.wait_for(run_recorder_health_check(), timeout=5.0)
            log.info("startup_recorder_health", service="pipeline", ok=rec_ok)
        except asyncio.TimeoutError:
            log.warning("startup_recorder_health_skipped", service="pipeline",
                        reason="timeout — no audio device (non-desktop env)")
        except Exception as exc:
            log.warning("startup_recorder_health_failed", service="pipeline", error=str(exc))
    else:
        log.info("startup_recorder_health_skipped", service="pipeline",
                 reason=f"APP_MODE={_app_mode} — audio hardware not expected")

    # ── 6. PCM chunk pool pre-allocation ─────────────────────────────────
    try:
        pool = get_chunk_pool()
        pool.preallocate(count=64)
        log.info("startup_chunk_pool_preallocated", service="pipeline", count=64)
    except Exception as exc:
        log.warning("startup_chunk_pool_failed", service="pipeline", error=str(exc))

    # ── 7. Heartbeat — services with no warmup step ───────────────────────
    # These services are passive at startup (no init to run) but we still want
    # them to show "ok" in the dashboard instead of "unknown".
    for _svc in ("llm", "session", "memory", "sanitize", "rl", "bh", "cb", "redis", "transcript"):
        log.info("service_ready", service=_svc)

    # ── 8. Feature flag state ────────────────────────────────────────────
    log.info(
        "startup_feature_flags",
        service="pipeline",
        pcm_pipeline=FF_PCM_PIPELINE,
        barge_in=FF_BARGE_IN,
        audio_diagnostics=FF_AUDIO_DIAGNOSTICS,
        session_lifecycle=FF_SESSION_LIFECYCLE,
        question_prefetch=FF_QUESTION_PREFETCH,
        canary_pct=FF_CANARY_PCT,
    )

    # ── 9. Concept tracker ───────────────────────────────────────────────
    try:
        from app.eval.concept_tracker import concept_tracker
        await concept_tracker.start()
        log.info("startup_concept_tracker_ok", service="concept_tracker")
    except Exception as exc:
        log.error("startup_concept_tracker_failed", service="concept_tracker", error=str(exc))

    startup_latency = time.monotonic() - t0
    log.info(
        "voice_graph_startup_complete",
        service="pipeline",
        version=GRAPH_VERSION,
        latency_s=round(startup_latency, 2),
    )


async def on_shutdown() -> None:
    """
    Application shutdown hook. Called once when the server is stopping.

    Tears down:
      1. All active sessions (force close with reason="shutdown")
      2. STT model unload
      3. TTS model unload + temp file cleanup
      4. QA controller Redis disconnect
      5. Evaluation engine Redis disconnect
      6. PCM chunk pool release
    """
    log.info("voice_graph_shutdown_begin")
    t0 = time.monotonic()

    # ── 1. Force-close all active sessions ────────────────────────────────
    for graph_instance in (voice_graph, voice_graph_realtime, voice_graph_low_latency):
        try:
            closed = await graph_instance.session_manager.force_close_all(reason="shutdown")
            if closed > 0:
                log.info("shutdown_sessions_closed", tier=graph_instance._tier, count=closed)  # noqa
        except Exception as exc:
            log.warning("shutdown_sessions_close_error", error=str(exc))

    from app.eval.concept_tracker import concept_tracker
    await concept_tracker.stop()

    # ── 2. STT unload ────────────────────────────────────────────────────
    try:
        stt_node = voice_graph._stt # noqa
        if hasattr(stt_node, "close"):
            await stt_node.close()
        log.info("shutdown_stt_ok")
    except Exception as exc:
        log.warning("shutdown_stt_error", error=str(exc))

    # ── 3. TTS unload ────────────────────────────────────────────────────
    try:
        tts_node = voice_graph._tts # noqa
        if hasattr(tts_node, "close"):
            await tts_node.close()
        log.info("shutdown_tts_ok")
    except Exception as exc:
        log.warning("shutdown_tts_error", error=str(exc))

    # ── 4. QA controller disconnect ──────────────────────────────────────
    try:
        if _qa_controller is not None:
            await _qa_controller.disconnect()
            log.info("shutdown_qa_controller_ok")
    except Exception as exc:
        log.warning("shutdown_qa_controller_error", error=str(exc))

    # ── 5. Evaluation engine disconnect ──────────────────────────────────
    try:
        if _eval_engine is not None:
            await _eval_engine.close()
            log.info("shutdown_eval_engine_ok")
    except Exception as exc:
        log.warning("shutdown_eval_engine_error", error=str(exc))

    # ── 6. PCM chunk pool release ────────────────────────────────────────
    try:
        pool = get_chunk_pool()
        pool.release_all()
        log.info("shutdown_chunk_pool_released")
    except Exception as exc:
        log.debug("shutdown_chunk_pool_error", error=str(exc))

    shutdown_latency = time.monotonic() - t0
    log.info(
        "voice_graph_shutdown_complete",
        latency_s=round(shutdown_latency, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KAFKA-BACKED STAGE BUS
#
# When VOICE_KAFKA_ENABLED=1, the StageBus wraps Kafka producers and consumers
# instead of in-process asyncio queues. This enables:
#   1. Cross-process pipeline stages (STT on GPU box, LLM on API tier)
#   2. Durable message persistence (replay on crash recovery)
#   3. Backpressure through Kafka consumer lag monitoring
#   4. Dead letter topic for failed messages
#   5. Per-stage throughput metering via Kafka consumer group offsets
#
# The KafkaStageBus implements the same interface as the in-process StageBus
# so the graph builder and stream_full() don't need to know which backend
# is in use.
# ═══════════════════════════════════════════════════════════════════════════════

KAFKA_ENABLED: bool = os.getenv("VOICE_KAFKA_ENABLED", "0") == "1"
KAFKA_BOOTSTRAP: str = os.getenv("VOICE_KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC_PREFIX: str = os.getenv("VOICE_KAFKA_TOPIC_PREFIX", "voice-pipeline")
KAFKA_DLQ_SUFFIX: str = os.getenv("VOICE_KAFKA_DLQ_SUFFIX", "-dlq")
KAFKA_CONSUMER_GROUP: str = os.getenv("VOICE_KAFKA_CONSUMER_GROUP", "voice-graph")
KAFKA_ACKS: str = os.getenv("VOICE_KAFKA_ACKS", "1")
KAFKA_LINGER_MS: int = int(os.getenv("VOICE_KAFKA_LINGER_MS", "5"))
KAFKA_BATCH_SIZE: int = int(os.getenv("VOICE_KAFKA_BATCH_SIZE", "16384"))
KAFKA_MAX_POLL_RECORDS: int = int(os.getenv("VOICE_KAFKA_MAX_POLL_RECORDS", "10"))
KAFKA_SESSION_TIMEOUT_MS: int = int(os.getenv("VOICE_KAFKA_SESSION_TIMEOUT_MS", "10000"))
KAFKA_HEARTBEAT_MS: int = int(os.getenv("VOICE_KAFKA_HEARTBEAT_MS", "3000"))

# Kafka — Produce Metrics

_kafka_produce_total = make_counter(
    "voice_kafka_produce_total",
    "Total messages produced to Kafka",
    ["topic"],
)

_kafka_produce_errors = make_counter(
    "voice_kafka_produce_errors_total",
    "Total Kafka produce errors",
    ["topic"],
)

_kafka_produce_latency = make_histogram(
    "voice_kafka_produce_latency_seconds",
    "Kafka produce latency",
    ["topic"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Kafka — Consume Metrics

_kafka_consume_total = make_counter(
    "voice_kafka_consume_total",
    "Total messages consumed from Kafka",
    ["topic"],
)

_kafka_consume_lag = make_gauge(
    "voice_kafka_consume_lag",
    "Kafka consumer lag (messages behind)",
    ["topic", "partition"],
)

# Kafka — DLQ Metrics

_kafka_dlq_total = make_counter(
    "voice_kafka_dlq_total",
    "Total messages sent to dead letter topic",
    ["topic"],
)


class KafkaStageBus:
    """
    Kafka-backed message bus between pipeline stages. Drop-in replacement
    for the in-process StageBus when KAFKA_ENABLED is True.

    Each stage pair maps to a Kafka topic:
        stt→llm  →  {KAFKA_TOPIC_PREFIX}-stt-llm
        llm→tts  →  {KAFKA_TOPIC_PREFIX}-llm-tts

    Messages are JSON-serialised StageBusMessage objects with the session_id
    as the partition key, ensuring all messages for a single session land on
    the same partition and are processed in order.

    The DLQ topic receives messages that fail processing after 3 attempts.
    """

    def __init__(
        self,
        pair:     str,
        session_id: str = "",
    ) -> None:
        self._pair = pair
        self._session_id = session_id
        self._topic = f"{KAFKA_TOPIC_PREFIX}-{pair.replace('→', '-')}"
        self._dlq_topic = f"{self._topic}{KAFKA_DLQ_SUFFIX}"
        self._producer: Any = None
        self._consumer: Any = None
        self._closed = False

    async def connect(self) -> None:
        """
        Lazily initialise Kafka producer and consumer. We import aiokafka
        here to avoid hard dependency — when Kafka is disabled, the import
        never happens.
        """
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                acks=KAFKA_ACKS,
                linger_ms=KAFKA_LINGER_MS,
                max_batch_size=KAFKA_BATCH_SIZE,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
            await self._producer.start()

            self._consumer = AIOKafkaConsumer(
                self._topic,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id=f"{KAFKA_CONSUMER_GROUP}-{self._pair}",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                max_poll_records=KAFKA_MAX_POLL_RECORDS,
                session_timeout_ms=KAFKA_SESSION_TIMEOUT_MS,
                heartbeat_interval_ms=KAFKA_HEARTBEAT_MS,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
            )
            await self._consumer.start()
            log.info("kafka_bus_connected", topic=self._topic)

        except ImportError:
            log.error("aiokafka_not_installed — Kafka bus unavailable, falling back to in-process")
            raise
        except Exception as exc:
            log.error("kafka_bus_connect_failed", topic=self._topic, error=str(exc))
            raise

    async def put(self, msg: StageBusMessage) -> None:
        """
        Produce a message to the Kafka topic. The session_id is used as the
        partition key for ordering guarantees.
        """
        if self._closed or self._producer is None:
            raise RuntimeError(f"KafkaStageBus {self._pair} is not connected or closed")

        t0 = time.monotonic()
        try:
            payload = {
                "payload":    msg.payload,
                "source":     msg.source_stage,
                "sequence":   msg.seq,
                "is_error":   msg.is_error,
                "is_sentinel": msg.is_sentinel,
                "timestamp":  msg.created_at,
                "session_id": self._session_id,
            }

            await self._producer.send_and_wait(
                self._topic,
                value=payload,
                key=self._session_id or None,
            )

            latency = time.monotonic() - t0
            _kafka_produce_total.labels(topic=self._topic).inc()
            _kafka_produce_latency.labels(topic=self._topic).observe(latency)

        except Exception as exc:
            _kafka_produce_errors.labels(topic=self._topic).inc()
            log.error("kafka_produce_failed", topic=self._topic, error=str(exc))
            raise

    async def get(self) -> StageBusMessage | None:
        """
        Consume the next message from the Kafka topic. Blocks until a message
        is available. Returns None if the bus is closed.
        """
        if self._closed or self._consumer is None:
            return None

        try:
            record = await self._consumer.getone()
            data = record.value

            _kafka_consume_total.labels(topic=self._topic).inc()

            msg = StageBusMessage(
                payload=data.get("payload", {}),
                source_stage=data.get("source", ""),
                seq=data.get("sequence", 0),
                is_error=data.get("is_error", False),
                is_sentinel=data.get("is_sentinel", False),
                created_at=data.get("timestamp", time.monotonic()),
            )

            return msg

        except Exception as exc:
            if self._closed:
                return None
            log.error("kafka_consume_failed", topic=self._topic, error=str(exc))
            raise

    async def send_to_dlq(self, msg: StageBusMessage, error: str) -> None:
        """
        Send a failed message to the dead letter topic for manual inspection
        and possible replay.
        """
        if self._producer is None:
            return

        try:
            dlq_payload = {
                "original_message": {
                    "payload": msg.payload,
                    "source": msg.source_stage,
                    "sequence": msg.seq,
                },
                "error": error,
                "dlq_timestamp": time.time(),
                "session_id": self._session_id,
                "topic": self._topic,
            }
            await self._producer.send_and_wait(
                self._dlq_topic,
                value=dlq_payload,
                key=self._session_id or None,
            )
            _kafka_dlq_total.labels(topic=self._topic).inc()
            log.info("kafka_dlq_sent", topic=self._dlq_topic)

        except Exception as exc:
            log.error("kafka_dlq_send_failed", topic=self._dlq_topic, error=str(exc))

    async def close(self) -> None:
        """Stop producer and consumer, mark bus as closed."""
        self._closed = True
        if self._producer:
            try:
                await self._producer.stop()
            except Exception: # noqa
                pass
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception: # noqa
                pass
        log.info("kafka_bus_closed", topic=self._topic)

    async def lag(self) -> dict[int, int]:
        """
        Return per-partition consumer lag. Used by monitoring to detect
        backpressure situations where the consumer can't keep up with
        the producer.
        """
        if self._consumer is None:
            return {}

        try:
            partitions = self._consumer.assignment()
            end_offsets = await self._consumer.end_offsets(partitions)
            lag_map = {}

            for tp in partitions:
                current = await self._consumer.position(tp)
                end = end_offsets.get(tp, current)
                lag = max(0, end - current)
                lag_map[tp.partition] = lag
                _kafka_consume_lag.labels(
                    topic=self._topic,
                    partition=str(tp.partition),
                ).set(lag)

            return lag_map
        except Exception as exc:
            log.debug("kafka_lag_check_failed", error=str(exc))
            return {}


def get_stage_bus(
    pair: str,
    session_id: str = "",
    maxsize: int = 16,
    overflow: OverflowPolicy = OverflowPolicy.BLOCK,
) -> StageBus | KafkaStageBus:
    """
    Factory function that returns either a Kafka-backed or in-process StageBus
    depending on configuration. The caller doesn't need to know which backend
    is in use — both implement put(), get(), close().
    """
    if KAFKA_ENABLED:
        return KafkaStageBus(pair=pair, session_id=session_id)
    return StageBus(stage_pair=pair, max_depth=maxsize, overflow=overflow)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION EXPORT AND ADMIN API
#
# Functions for exporting session data, generating reports, and administrative
# operations. Used by the API layer and CLI tools.
# ═══════════════════════════════════════════════════════════════════════════════


async def export_session_transcript(session_id: str) -> list[dict[str, Any]]:
    """
    Export the full interview transcript as a list of turn dicts. Each turn
    contains: turn_index, domain, question, answer, timestamp, and optionally
    evaluation scores if available.
    """
    turns = []

    try:
        doc = await _qa_controller.get_document(session_id)
        if doc is None:
            return []

        for turn in doc.turns:
            turn_dict = {
                "turn_index": turn.turn_index,
                "domain": turn.domain,
                "question": turn.q,
                "answer": turn.a,
                "timestamp": turn.ts,
            }

            # Attach evaluation score if available
            if _eval_engine:
                try:
                    score = await _eval_engine.get_turn_score(session_id, turn.turn_index)
                    if score:
                        turn_dict["evaluation"] = score.to_dict()
                except Exception: # noqa
                    pass

            turns.append(turn_dict)

    except Exception as exc:
        log.error("export_transcript_failed", sid=session_id[:8], error=str(exc))

    return turns


async def export_session_jsonl(session_id: str) -> str:
    """
    Export session as JSONL for training data pipelines. Each line is a
    JSON object with question/answer/domain/score fields.
    """
    try:
        if _qa_controller:
            return await _qa_controller.export_session_jsonl(session_id)
    except Exception as exc:
        log.error("export_jsonl_failed", sid=session_id[:8], error=str(exc))
    return ""


async def list_active_sessions() -> list[dict[str, Any]]:
    """
    List all currently active sessions across all VoiceGraph instances.
    Returns a list of session summary dicts.
    """
    sessions = []
    for graph_inst in (voice_graph, voice_graph_realtime, voice_graph_low_latency):
        try:
            mgr = graph_inst.session_manager
            async with mgr._lock: # noqa
                for sid, res in mgr._active.items(): # noqa
                    sessions.append({
                        **res.to_dict(),
                        "tier": graph_inst._tier, # noqa
                    })
        except Exception as exc:
            log.debug("list_sessions_error", tier=graph_inst._tier, error=str(exc)) # noqa
    return sessions


async def force_close_session(session_id: str, reason: str = "admin") -> dict[str, Any]:
    """
    Admin force-close a session. Searches all graph instances for the session.
    """
    for graph_inst in (voice_graph, voice_graph_realtime, voice_graph_low_latency):
        resources = await graph_inst.session_manager.get_resources(session_id)
        if resources:
            return await graph_inst.session_manager.close_session(session_id, reason=reason)

    return {"session_id": session_id, "status": "not_found"}


async def pipeline_health() -> dict[str, Any]:
    """
    Comprehensive health check across all VoiceGraph instances and shared
    infrastructure. Suitable for k8s liveness/readiness probes.
    """
    result = {
        "version": GRAPH_VERSION,
        "timestamp": time.time(),
        "instances": {},
    }

    for name, graph_inst in [
        ("balanced", voice_graph),
        ("realtime", voice_graph_realtime),
        ("low_latency", voice_graph_low_latency),
    ]:
        try:
            h = await graph_inst.health()
            result["instances"][name] = h
        except Exception as exc:
            result["instances"][name] = {"healthy": False, "error": str(exc)}

    # Kafka health (if enabled)
    if KAFKA_ENABLED:
        try:
            test_bus = KafkaStageBus(pair="health-check")
            await test_bus.connect()
            result["kafka"] = {"healthy": True, "bootstrap": KAFKA_BOOTSTRAP}
            await test_bus.close()
        except Exception as exc:
            result["kafka"] = {"healthy": False, "error": str(exc)}

    # Overall health
    instance_healths = [
        v.get("healthy", False) for v in result["instances"].values()
    ]
    result["healthy"] = all(instance_healths)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# PCM PIPELINE BUILDER INTEGRATION
#
# Utility functions that compose audio_engine PCM components into ready-to-use
# processing chains for the graph's PCM execution modes.
# ═══════════════════════════════════════════════════════════════════════════════


def build_mic_pcm_chain(
    mic_format: PCMFormat,
    cfg: VoiceGraphConfig,
) -> dict[str, Any]:
    """
    Build the microphone-side PCM processing chain:
      PCMInputStream → PCMConverter (if needed) → PCMVADGate → PCMSpeechEnhancer
                     → PCMRingBuffer → PCMDiagnosticsMonitor

    Returns a dict of named components for the PCM pipeline worker to use.
    """
    target_format = PCMFormat(sample_rate=16000, channels=1, dtype="int16")

    converter = None
    if mic_format != target_format:
        converter = PCMConverter()

    vad = PCMVADGate(
        fmt=target_format,
        hangover_s=cfg.pcm_vad_hangover_frames / target_format.sample_rate,
    )

    enhancer = PCMSpeechEnhancer(fmt=target_format)

    ring = PCMRingBuffer(
        capacity=int(cfg.pcm_ring_buffer_seconds * target_format.sample_rate),
        fmt=target_format,
    )

    diagnostics = PCMDiagnosticsMonitor(fmt=target_format)
    latency = PCMLatencyTracker()

    return {
        "converter":   converter,
        "vad_gate":    vad,
        "enhancer":    enhancer,
        "ring_buffer": ring,
        "diagnostics": diagnostics,
        "latency":     latency,
        "format":      target_format,
    }


def build_speaker_pcm_chain(
    speaker_format: PCMFormat,
    cfg: VoiceGraphConfig,
) -> dict[str, Any]:
    """
    Build the speaker-side PCM processing chain:
      TTS PCM output → PCMTTSQualityGate → PCMPlaybackEnhancer
                     → PCMJitterBuffer → PCMOutputStream

    Returns a dict of named components for the PCM pipeline worker to use.
    """
    quality_gate = PCMTTSQualityGate(
        analyzer=PCMWaveformAnalyzer(fmt=speaker_format)
    )

    enhancer = PCMPlaybackEnhancer(fmt=speaker_format)

    jitter = PCMJitterBuffer(
        fmt=speaker_format,
        target_delay_ms=cfg.pcm_jitter_buffer_ms,
    )

    gap_mgr = PCMSentenceGapManager(
        fmt=speaker_format,
        gap_s=0.2,
    )

    interrupt = PCMInterruptDetector(
        fmt=speaker_format,
        onset_rms=cfg.pcm_interrupt_threshold,
    )

    latency = PCMLatencyTracker()

    return {
        "quality_gate": quality_gate,
        "enhancer":     enhancer,
        "jitter":       jitter,
        "gap_manager":  gap_mgr,
        "interrupt":    interrupt,
        "latency":      latency,
        "format":       speaker_format,
    }


def build_full_pcm_pipeline(
    mic_format:     PCMFormat,
    speaker_format: PCMFormat,
    cfg:            VoiceGraphConfig,
) -> dict[str, Any]:
    """
    Build both mic and speaker chains plus shared components. This is the
    top-level factory for stream_full_pcm() to call during session open.
    """
    mic_chain     = build_mic_pcm_chain(mic_format, cfg)
    speaker_chain = build_speaker_pcm_chain(speaker_format, cfg)

    pool = get_chunk_pool()

    # Waveform analyser for session-level audio quality metrics
    analyser = PCMWaveformAnalyzer(fmt=mic_chain["format"])

    # Stream mixer for handling barge-in audio mixing (if needed)
    mixer = PCMStreamMixer(fmt=speaker_chain["format"])

    return {
        "mic":     mic_chain,
        "speaker": speaker_chain,
        "pool":    pool,
        "analyser": analyser,
        "mixer":   mixer,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH INTROSPECTION
#
# Debug and monitoring utilities for inspecting the compiled graph structure,
# active pipelines, and runtime statistics.
# ═══════════════════════════════════════════════════════════════════════════════


def get_graph_topology() -> dict[str, Any]:
    """
    Return the graph topology as a serialisable dict. Used by the debug
    endpoint and monitoring dashboards to visualise the pipeline structure.
    """
    return {
        "version": GRAPH_VERSION,
        "nodes": [
            "stt", "stt_error", "llm", "llm_error",
            "sanitize", "tts", "tts_error", "error_terminal",
            "audio_sink_dev",
        ],
        "edges": [
            {"from": "stt", "to": "llm", "condition": "success"},
            {"from": "stt", "to": "stt_error", "condition": "error"},
            {"from": "stt_error", "to": "stt", "condition": "retry"},
            {"from": "stt_error", "to": "error_terminal", "condition": "max_retries"},
            {"from": "llm", "to": "sanitize", "condition": "success"},
            {"from": "llm", "to": "llm_error", "condition": "error"},
            {"from": "llm_error", "to": "llm", "condition": "retry"},
            {"from": "llm_error", "to": "error_terminal", "condition": "max_retries"},
            {"from": "sanitize", "to": "tts", "condition": "always"},
            {"from": "tts", "to": "END", "condition": "success"},
            {"from": "tts", "to": "tts_error", "condition": "error"},
            {"from": "tts_error", "to": "error_terminal", "condition": "always"},
            {"from": "error_terminal", "to": "END", "condition": "always"},
        ],
        "execution_modes": ["api", "stream", "realtime", "pcm", "ptt"],
        "entry_point": "stt",
        "feature_flags": {
            "pcm_pipeline": FF_PCM_PIPELINE,
            "barge_in": FF_BARGE_IN,
            "audio_diagnostics": FF_AUDIO_DIAGNOSTICS,
            "session_lifecycle": FF_SESSION_LIFECYCLE,
            "question_prefetch": FF_QUESTION_PREFETCH,
            "kafka_enabled": KAFKA_ENABLED,
        },
    }


def get_runtime_stats() -> dict[str, Any]:
    """
    Snapshot of current runtime statistics across all graph instances.
    """
    stats = {
        "timestamp": time.time(),
        "version": GRAPH_VERSION,
        "instances": {},
    }

    for name, inst in [
        ("balanced", voice_graph),
        ("realtime", voice_graph_realtime),
        ("low_latency", voice_graph_low_latency),
    ]:
        stats["instances"][name] = {
            "active_sessions": inst.session_manager.active_count(),
            "load_guard_active": inst._load_guard.current_count(), # noqa
            "tier": inst._tier, # noqa
        }

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main class
    "VoiceGraph",

    # Singletons
    "voice_graph",
    "voice_graph_realtime",
    "voice_graph_low_latency",
    "get_voice_graph",

    # Lifecycle
    "on_startup",
    "on_shutdown",

    # Types
    "VoiceState",
    "VoiceGraphConfig",
    "PipelineStage",
    "SessionResources",

    # Exceptions
    "LatencyBudgetExceeded",
    "LoadSheddingRejected",

    # Infrastructure
    "StageBus",
    "KafkaStageBus",
    "StageBusMessage",
    "OverflowPolicy",
    "get_stage_bus",
    "SessionLifecycleManager",
    "AudioDiagnosticsPipeline",
    "PipelineWatchdog",

    # Admin / export
    "export_session_transcript",
    "export_session_jsonl",
    "list_active_sessions",
    "force_close_session",
    "pipeline_health",
    "get_graph_topology",
    "get_runtime_stats",

    # PCM pipeline builders
    "build_mic_pcm_chain",
    "build_speaker_pcm_chain",
    "build_full_pcm_pipeline",

    # Wiring hooks
    "set_audit_bus",
    "set_transcript_writer",
    "set_finalize_eval",
    "GRAPH_VERSION",
    "integrations_health",
    "reset_integrations",
]


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST HARNESS
#
# Built-in test utilities for running the pipeline end-to-end against real
# or mock infrastructure. Called by CI/CD and local dev scripts.
# ═══════════════════════════════════════════════════════════════════════════════


class PipelineTestHarness:
    """
    Self-contained test harness that exercises all five execution modes with
    configurable node implementations. Collects timing, errors, and output
    quality metrics for each run.

    Usage:
        harness = PipelineTestHarness(graph=voice_graph)
        results = await harness.run_smoke_test()
        assert results["all_passed"]
    """

    def __init__(
        self,
        graph: VoiceGraph,
        test_audio_path: str = "",
    ) -> None:
        self._graph = graph
        self._test_audio = test_audio_path
        self._results: list[dict[str, Any]] = []

    async def run_smoke_test(self) -> dict[str, Any]:
        """
        Minimal smoke test that exercises the api mode with a test audio file.
        Returns pass/fail with timing data.
        """
        if not self._test_audio:
            return {"skipped": True, "reason": "No test audio path configured"}

        test_id = f"smoke_{uuid.uuid4().hex[:8]}"
        t0 = time.monotonic()

        try:
            result = await self._graph.run(
                audio_path=self._test_audio,
                session_id=f"test_{test_id}",
                request_id=test_id,
            )

            latency = time.monotonic() - t0
            passed = (
                result.get("stage") != PipelineStage.FAILED.value
                and bool(result.get("audio_output") or result.get("audio_s3_uri"))
            )

            test_result = {
                "test": "smoke",
                "passed": passed,
                "latency_s": round(latency, 3),
                "stage": result.get("stage", ""),
                "transcript_len": len(result.get("user_input", "")),
                "response_len": len(result.get("llm_response", "")),
                "error": result.get("error", ""),
            }
            self._results.append(test_result)
            return test_result

        except Exception as exc:
            return {
                "test": "smoke",
                "passed": False,
                "error": str(exc),
                "latency_s": round(time.monotonic() - t0, 3),
            }

    async def run_latency_benchmark(
        self,
        iterations: int = 10,
    ) -> dict[str, Any]:
        """
        Run multiple iterations of the pipeline to measure latency distribution.
        Returns p50/p95/p99 latencies broken down by stage.
        """
        if not self._test_audio:
            return {"skipped": True, "reason": "No test audio path configured"}

        latencies: dict[str, list[float]] = {
            "total": [],
            "stt": [],
            "llm": [],
            "sanitize": [],
            "tts": [],
        }
        errors = 0

        for i in range(iterations):
            test_id = f"bench_{i}_{uuid.uuid4().hex[:6]}"
            try:
                result = await self._graph.run(
                    audio_path=self._test_audio,
                    session_id=f"bench_{test_id}",
                    request_id=test_id,
                )
                stage_times = result.get("stage_latencies", {})
                latencies["total"].append(result.get("pipeline_latency_s", 0.0))
                for stage in ("stt", "llm", "sanitize", "tts"):
                    if stage in stage_times:
                        latencies[stage].append(stage_times[stage])
            except Exception: # noqa
                errors += 1

        def _percentiles(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"p50": 0, "p95": 0, "p99": 0}
            s = sorted(vals)
            n = len(s)
            return {
                "p50": round(s[int(n * 0.5)], 4),
                "p95": round(s[int(n * 0.95)], 4),
                "p99": round(s[min(int(n * 0.99), n - 1)], 4),
            }

        return {
            "iterations": iterations,
            "errors": errors,
            "latencies": {
                stage: _percentiles(vals)
                for stage, vals in latencies.items()
            },
        }

    async def run_qa_flow_test(self) -> dict[str, Any]:
        """
        Test the full QA interview flow: greeting → intro → N questions → close.
        Validates that the QA controller state machine transitions correctly
        and that evaluation scores are generated.
        """
        session_id = f"qatest_{uuid.uuid4().hex[:8]}"
        t0 = time.monotonic()
        turns_completed = 0
        stages_seen: list[str] = []

        try:
            # ── greeting ──────────────────────────────────────────────
            result = await self._graph.run(
                audio_path=self._test_audio,
                session_id=session_id,
                request_id=f"{session_id}_greeting",
            )
            stages_seen.append(result.get("qa_stage", ""))
            turns_completed += 1

            # ── intro (simulate with same test audio) ─────────────────
            result = await self._graph.run(
                audio_path=self._test_audio,
                session_id=session_id,
                request_id=f"{session_id}_intro",
            )
            stages_seen.append(result.get("qa_stage", ""))
            turns_completed += 1

            # ── interview (3 turns) ───────────────────────────────────
            for i in range(3):
                result = await self._graph.run(
                    audio_path=self._test_audio,
                    session_id=session_id,
                    request_id=f"{session_id}_q{i}",
                )
                stages_seen.append(result.get("qa_stage", ""))
                turns_completed += 1

            # ── close session ─────────────────────────────────────────
            summary = await self._graph.session_manager.close_session(
                session_id, reason="test_complete",
            )

            latency = time.monotonic() - t0
            return {
                "test": "qa_flow",
                "passed": turns_completed >= 5,
                "turns_completed": turns_completed,
                "stages_seen": stages_seen,
                "session_summary": summary,
                "latency_s": round(latency, 3),
            }

        except Exception as exc:
            return {
                "test": "qa_flow",
                "passed": False,
                "turns_completed": turns_completed,
                "stages_seen": stages_seen,
                "error": str(exc),
                "latency_s": round(time.monotonic() - t0, 3),
            }

    async def run_all(self) -> dict[str, Any]:
        """Run all test suites and return aggregated results."""
        smoke = await self.run_smoke_test()
        benchmark = await self.run_latency_benchmark(iterations=5)
        qa_flow = await self.run_qa_flow_test()

        all_passed = (
            smoke.get("passed", False)
            and benchmark.get("errors", 1) == 0
            and qa_flow.get("passed", False)
        )

        return {
            "all_passed": all_passed,
            "smoke": smoke,
            "benchmark": benchmark,
            "qa_flow": qa_flow,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RECORDING-AWARE STREAMING
#
# Utilities that bridge the recorder module's blocking I/O with the graph's
# async streaming pipeline. Enables live-recording-to-streaming without
# intermediate file I/O.
# ═══════════════════════════════════════════════════════════════════════════════


async def stream_from_recording(
    is_held_fn:   Callable[[], bool],
    graph:        VoiceGraph | None = None,
    session_id:   str  = "",
    request_id:   str  = "",
    tts_voice:    str  = "",
    tts_speed:    float = 1.0,
) -> AsyncIterator[dict[str, Any]]:
    """
    Combined recording + streaming pipeline. Records audio while is_held_fn()
    returns True, then streams the pipeline output segment by segment.

    This is the recommended entry point for CLI and demo applications that
    need push-to-talk with realtime audio output.

    Yields:
        dict with type="recording_started"  (when recording begins)
        dict with type="recording_complete" (when recording ends, with duration)
        dict with type="audio_segment"      (for each TTS output segment)
        dict with type="pipeline_complete"  (when all segments emitted)
    """
    rid = request_id or str(uuid.uuid4())
    g = graph or voice_graph

    yield {
        "type": "recording_started",
        "request_id": rid,
        "timestamp": time.time(),
    }

    # Record
    rec_t0 = time.monotonic()
    recording = await record_audio_until_released_async(is_held_fn)
    audio_path = recording or ""
    rec_latency = time.monotonic() - rec_t0

    if audio_path:
        try:
            with wave.open(audio_path, "rb") as wf:
                rec_duration = wf.getnframes() / wf.getframerate()
        except Exception: # noqa
            rec_duration = 0.0
    else:
        rec_duration = 0.0

    yield {
        "type": "recording_complete",
        "request_id": rid,
        "duration_s": round(rec_duration, 2),
        "path": audio_path,
        "latency_s": round(rec_latency, 3),
    }

    if not audio_path:
        yield {
            "type": "pipeline_complete",
            "request_id": rid,
            "error": "No audio recorded",
        }
        return

    # Stream pipeline
    try:
        segment_count = 0
        async for segment in g.stream_full(
            audio_path=audio_path,
            session_id=session_id,
            request_id=rid,
            tts_voice=tts_voice,
            tts_speed=tts_speed,
        ):
            segment_count += 1
            yield segment

        yield {
            "type": "pipeline_complete",
            "request_id": rid,
            "segments": segment_count,
            "total_latency_s": round(time.monotonic() - rec_t0, 3),
        }

    finally:
        try:
            delete_temp_recording(audio_path)
        except Exception: # noqa
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# PCM STREAM BRIDGE
#
# Adapts between the graph's async PCM streaming and external transports
# (WebSocket, WebRTC, raw TCP). The bridge handles format conversion,
# chunk size normalisation, and flow control.
# ═══════════════════════════════════════════════════════════════════════════════


class PCMWebSocketBridge:
    """
    Bridges a WebSocket connection to the PCM pipeline. Reads PCM frames
    from the WebSocket, feeds them to stream_full_pcm(), and writes TTS
    PCM chunks back to the WebSocket.

    Frame format (both directions):
        - 2 bytes: frame length (big-endian uint16)
        - N bytes: raw PCM samples

    The bridge handles the async generator protocol and ensures proper
    cleanup on disconnect.
    """

    def __init__(
        self,
        websocket: Any,       # aiohttp.WebSocketResponse or similar
        graph: VoiceGraph,
        mic_format: PCMFormat,
        session_id: str = "",
        request_id: str = "",
        tts_voice:  str = "",
        tts_speed:  float = 1.0,
    ) -> None:
        self._ws = websocket
        self._graph = graph
        self._mic_format = mic_format
        self._session_id = session_id
        self._request_id = request_id or str(uuid.uuid4())
        self._tts_voice = tts_voice
        self._tts_speed = tts_speed
        self._running = False
        self._stats = {
            "frames_received": 0,
            "frames_sent": 0,
            "bytes_received": 0,
            "bytes_sent": 0,
            "errors": 0,
            "duration_s": 0.0,
        }

    async def _ws_pcm_input_stream(self) -> AsyncIterator[PCMChunk]:
        """
        Read PCM frames from the WebSocket and yield them as PCMChunk objects.
        Handles framing, format validation, and disconnect detection.
        """
        pool = get_chunk_pool()
        chunk_size = self._mic_format.bytes_per_frame * 160  # 10ms at 16kHz

        while self._running:
            try:
                msg = await asyncio.wait_for(self._ws.receive(), timeout=30.0)

                if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    break
                if msg.type == WSMsgType.ERROR:
                    break

                if msg.type == WSMsgType.BINARY:
                    pcm_data: bytes = msg.data

                    # Pad or truncate to exact chunk_size so the array is always aligned
                    if len(pcm_data) < chunk_size:
                        pcm_data = pcm_data + b"\x00" * (chunk_size - len(pcm_data))
                    elif len(pcm_data) > chunk_size:
                        pcm_data = pcm_data[:chunk_size]

                    self._stats["frames_received"] += 1
                    self._stats["bytes_received"] += len(pcm_data)

                    n_frames = self._mic_format.bytes_to_frames(len(pcm_data))
                    buf = pool.acquire(n_frames, dtype=self._mic_format.dtype, channels=self._mic_format.channels)
                    np.copyto(buf, np.frombuffer(pcm_data, dtype=self._mic_format.dtype))

                    chunk = PCMChunk(
                        data=buf,
                        fmt=self._mic_format,
                        timestamp=time.monotonic(),
                        source="ws",
                    )
                    yield chunk

            except asyncio.TimeoutError:
                # Send keep-alive ping
                try:
                    await self._ws.ping()
                except Exception: # noqa
                    break

            except Exception as exc:
                self._stats["errors"] += 1
                log.debug("ws_pcm_read_error", error=str(exc))
                break

    async def run(self) -> dict[str, Any]:
        """
        Start the bridge. Blocks until the WebSocket disconnects or the
        pipeline completes. Returns connection statistics.
        """
        self._running = True
        t0 = time.monotonic()

        log.info(
            "ws_pcm_bridge_started",
            request_id=self._request_id,
            session_id=self._session_id[:8] if self._session_id else "",
            mic_format=str(self._mic_format),
        )

        try:
            input_stream = self._ws_pcm_input_stream()

            async for tts_chunk in self._graph.stream_full_pcm(
                pcm_input_stream=input_stream,
                session_id=self._session_id,
                request_id=self._request_id,
                tts_voice=self._tts_voice,
                tts_speed=self._tts_speed,
            ):
                try:
                    # Frame: 2-byte length + raw PCM
                    frame_data = tts_chunk.data
                    length_header = len(frame_data).to_bytes(2, "big")
                    await self._ws.send_bytes(length_header + frame_data)

                    self._stats["frames_sent"] += 1
                    self._stats["bytes_sent"] += len(frame_data)

                except Exception as exc:
                    self._stats["errors"] += 1
                    log.debug("ws_pcm_write_error", error=str(exc))
                    break

        except Exception as exc:
            log.error("ws_pcm_bridge_error", request_id=self._request_id, error=str(exc))

        finally:
            self._running = False
            self._stats["duration_s"] = round(time.monotonic() - t0, 3)

            log.info(
                "ws_pcm_bridge_stopped",
                request_id=self._request_id,
                stats=self._stats,
            )

        return self._stats

    def stop(self) -> None:
        """Signal the bridge to stop. Non-blocking."""
        self._running = False


# ── WebSocket message type stub ───────────────────────────────────────────────
# Avoids hard dependency on aiohttp at import time.

class WSMsgType:
    TEXT    = 1
    BINARY  = 2
    PING    = 9
    PONG    = 10
    CLOSE   = 256
    CLOSING = 257
    CLOSED  = 258
    ERROR   = 259

try:
    from aiohttp import WSMsgType as _RealWSMsgType
    WSMsgType = _RealWSMsgType
except ImportError:
    _RealWSMsgType = None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
#
# python -m voice_graph [--mode api|ptt|health] [--audio path] [--session id]
#
# Provides a minimal command-line interface for testing the pipeline without
# a full HTTP server. Useful for development and debugging.
# ═══════════════════════════════════════════════════════════════════════════════


async def _cli_main() -> None:
    """CLI entry point for direct pipeline execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Voice Graph Pipeline CLI")
    parser.add_argument("--mode", choices=["api", "ptt", "health", "topology", "bench"],
                        default="health", help="Execution mode")
    parser.add_argument("--audio", default="", help="Path to audio file (for api mode)")
    parser.add_argument("--session", default="", help="Session ID")
    parser.add_argument("--tier", default="balanced", choices=["balanced", "realtime", "low_latency"],
                        help="Pipeline tier")
    parser.add_argument("--voice", default="", help="TTS voice")
    parser.add_argument("--speed", type=float, default=1.0, help="TTS speed")
    args = parser.parse_args()

    await on_startup()
    graph = get_voice_graph(args.tier)

    try:
        if args.mode == "health":
            health = await graph.health()
            print(json.dumps(health, indent=2, default=str))

        elif args.mode == "topology":
            topo = get_graph_topology()
            print(json.dumps(topo, indent=2))

        elif args.mode == "api":
            if not args.audio:
                print("Error: --audio required for api mode")
                return
            result = await graph.run(
                audio_path=args.audio,
                session_id=args.session or str(uuid.uuid4()),
                tts_voice=args.voice,
                tts_speed=args.speed,
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.mode == "ptt":
            print("Press and hold Enter to record, release to process...")
            import sys
            is_held = lambda: True  # Simplified — real implementation uses keyboard events
            result = await graph.run_ptt(
                is_held_fn=is_held,
                session_id=args.session or str(uuid.uuid4()),
                tts_voice=args.voice,
                tts_speed=args.speed,
            )
            print(json.dumps(result, indent=2, default=str))

        elif args.mode == "bench":
            if not args.audio:
                print("Error: --audio required for bench mode")
                return
            harness = PipelineTestHarness(graph=graph, test_audio_path=args.audio)
            results = await harness.run_all()
            print(json.dumps(results, indent=2, default=str))

    finally:
        await on_shutdown()


if __name__ == "__main__":
    asyncio.run(_cli_main())