"""
TTS node — OpenAI tts-1-hd API with all the production layers.

Base version features:
  - Uses Kokoro (hexgrad/Kokoro-82M) for local speech synthesis
  - Deterministic text cleaning for common symbols and formatting
  - Sentence-based chunking to prevent model overload
  - Safe generator handling with audio validation
  - Automatic peak normalization
  - Concatenates chunked audio into final WAV file
  - Writes PCM_16 .wav output to local audio directory
  - Graceful failure handling (never crashes caller)
  - Simple UUID-based filename generation

Beyond the base version:
  - S3 upload as the primary storage layer (disk is just a local cache)
  - audio stitching validation: chunk sizes checked before concatenation,
    final file verified before the path is returned
  - file lifecycle management: background task cleans up local files
    older than LOCAL_FILE_TTL seconds
  - asyncio.Lock per output path to prevent two concurrent writes to
    the same file
  - asyncio.Semaphore to cap parallel synthesis jobs
  - circuit breaker around the TTS API calls
  - full OTel spans + Prometheus counters/histograms
  - structured logging with request_id, char count, latency, voice, format

Partial-streaming additions:
  - synthesize_stream(): accepts AsyncIterator[str] from LLM and yields
    bytes per sentence via AsyncSentenceBuffer
  - synthesize_stream_to_files(): yields (path, is_final) tuples

Distributed-service additions:
  - TTSNodeProtocol: the structural interface for local TTSNode and
    RemoteTTSClient. VoiceGraph only imports the protocol.
  - RemoteTTSClient: calls a remote TTS microservice over HTTPS.
    Token streams are piped to the remote via chunked POST; audio bytes
    are streamed back via chunked transfer encoding. OTel TraceContext
    and LatencyBudget headers are injected on every request.
  - Cached apology audio fallback: if TTS completely fails (breaker OPEN,
    all retries exhausted) and a pre-rendered apology audio file is on
    disk, it is returned instead of crashing the pipeline. This means
    the voice UX always has something to play even during a full TTS
    outage.
  - LatencyBudget enforcement: check() at entry so the node self-aborts
    when the pipeline SLA is already blown before synthesis even starts.
  - BoundedPipelineQueue integration: synthesize_stream() respects the
    inter-node queue's backpressure so a slow audio consumer cannot
    cause unbounded memory growth in the sentence buffer.
  - get_tts_node() factory selects local vs remote based on env config.

PCM / audio_engine integration layer:
  - PCMTTSOutputConfig: per-TTSNode descriptor binding the output
    PCMFormat (rate, channels, dtype) to its PCMPlaybackEnhancer and
    PCMLatencyTracker. Eliminates ad-hoc format construction scattered
    throughout call sites.
  - PCMSentenceGapManager: inserts calibrated silence PCMChunks between
    synthesized sentences so the speaker produces natural speech cadence
    without popping or abrupt cuts. Silence duration is tunable and is
    automatically shortened for fast-speech voices.
  - PCMTTSQualityGate: runs every synthesized PCMChunk through
    PCMWaveformAnalyzer and rejects (or retries) chunks whose RMS,
    peak, or clipping ratio falls outside acceptable bounds.
  - PCMStreamToWAVCollector: accumulates a streaming PCM synthesis
    session into a single in-memory WAV buffer without disk I/O,
    enabling low-latency WAV delivery to callers who need a complete
    file but cannot tolerate the overhead of writing to disk.
  - TTSNode.synthesize_pcm(): synthesize text → PCMChunk (one shot),
    with full enhancement, quality gating, latency tracking, and format
    negotiation via the global PCMFormatRegistry.
  - TTSNode.synthesize_pcm_stream(): synthesize token stream → yields
    PCMChunks directly, running each chunk through PCMPlaybackEnhancer
    before yielding. Callers feed the output directly into PCMOutputStream
    for the lowest-latency speaker path.
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import asyncio
import collections
import io
import os
import re
import threading
import time
import uuid
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable, cast

# ── third-party ───────────────────────────────────────────────────────────────
import aiofiles
import httpx
import numpy as np
from openai import AsyncOpenAI
from openai import (
    APIConnectionError as _OAIConnectionError,
    APITimeoutError   as _OAITimeoutError,
    RateLimitError    as _OAIRateLimitError,
    InternalServerError as _OAIInternalError,
)
# Errors worth retrying — 400 BadRequest is NOT here (invalid voice, bad input,
# etc. will never succeed on retry and just burn time + circuit-breaker budget).
_TTS_RETRYABLE = (_OAIConnectionError, _OAITimeoutError, _OAIRateLimitError, _OAIInternalError)
from opentelemetry.trace import StatusCode

# ── internal — shared infrastructure ─────────────────────────────────────────
from app.common.shared import (
    CircuitBreaker,
    LatencyBudget,
    LatencyBudgetExceeded,
    RateLimiter,
    ServiceHealthState,
    backoff_retry,
    bulkheads,
    current_request_id, # noqa
    get_tracer,
    inject_trace_headers,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)

# ── internal — text sanitisation ─────────────────────────────────────────────
from app.nodes.sanitize import sanitize

# ── audio_engine PCM primitives ───────────────────────────────────────────────
#
# Everything imported here is consumed by the PCM integration layer below.
# The import list is intentionally exhaustive: we wire every useful audio_engine
# abstraction so callers never need to reach past TTS_service for PCM concerns.
from app.audio_essentials.audio_engine import (
    # ── Core Data Types ─────────────────────────────────────────────────────────
    PCMFormat,
    PCMChunk,

    # ── Format Negotiation & Registry ──────────────────────────────────────────
    PCMFormatRegistry,
    get_format_registry,
    negotiate_format, # noqa

    # ── Chunk Encoding / Decoding ──────────────────────────────────────────────
    tts_pcm_to_chunk,
    chunk_to_wav_bytes,

    # ── DSP / Audio Enhancement Processors ─────────────────────────────────────
    PCMConverter,
    PCMPlaybackEnhancer,
    PCMSilencePadder, # noqa
    PCMDynamicsProcessor, # noqa
    PCMAGCProcessor, # noqa

    # ── Analysis / Diagnostics / Telemetry ─────────────────────────────────────
    PCMWaveformAnalyzer,
    PCMDiagnosticsMonitor, # noqa
    PCMLatencyTracker,
    PCMMetricsSnapshot, # noqa
    get_metrics_snapshot,

    # ── Buffering / Memory Pool ─────────────────────────────────────────────────
    PCMRingBuffer, # noqa
    PCMChunkPool, # noqa
    get_chunk_pool,

    # ── Health / Reporting Models ──────────────────────────────────────────────
    AudioHealthReport, # noqa
    WaveformStats,
)
from app.monitoring.observability import get_logger
log = get_logger(__name__)
tracer = get_tracer(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

VoiceType = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"]
FormatType = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# Canonical set used for runtime validation — keeps VoiceType and the guard in sync.
_OPENAI_VALID_VOICES: frozenset[str] = frozenset(
    {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "ash", "sage", "coral"}
)
_TTS_VOICE_DEFAULT = "nova"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
TTS_MODEL: str = os.getenv("TTS_MODEL", "tts-1-hd")
TTS_VOICE: VoiceType = os.getenv("TTS_VOICE", "nova")  # type: ignore[assignment]
TTS_FORMAT: FormatType = os.getenv("TTS_FORMAT", "mp3")  # type: ignore[assignment]
TTS_OUTPUT_DIR: Path = Path(os.getenv("TTS_OUTPUT_DIR", "audio/audio_OUTPUT"))

RATE_PER_SEC: float = float(os.getenv("TTS_RATE_PER_SEC", "10.0"))
RATE_BURST: float = float(os.getenv("TTS_RATE_BURST", "20.0"))

_MAX_CHARS_PER_CHUNK: int = 4096
LOCAL_FILE_TTL: float = float(os.getenv("TTS_LOCAL_FILE_TTL", "3600"))

# Cached apology audio: a pre-rendered fallback file played when TTS is fully
# down. Generate it once with: `tts_node.synthesize(APOLOGY_TEXT)` and set
# TTS_APOLOGY_AUDIO_PATH to the result. If not set, the fallback is silent.
TTS_APOLOGY_AUDIO_PATH: str = os.getenv("TTS_APOLOGY_AUDIO_PATH", "")

# ── voice validation helper ───────────────────────────────────────────────────

def _resolve_voice(voice: str | None, log_source: str = "") -> VoiceType:
    """
    Validate and normalize a voice string before it reaches the OpenAI API.

    If the requested voice is not in _OPENAI_VALID_VOICES (e.g. a Kokoro voice
    name leaking in after a refactor, an empty string, or a stale value), fall
    back to _TTS_VOICE_DEFAULT and log a warning so the misconfiguration is
    visible without crashing the pipeline.
    """
    if voice and voice in _OPENAI_VALID_VOICES:
        return voice  # type: ignore[return-value]
    if voice:
        get_logger(__name__).warning(
            "tts_invalid_voice_fallback",
            requested=voice,
            fallback=_TTS_VOICE_DEFAULT,
            source=log_source,
        )
    return _TTS_VOICE_DEFAULT  # type: ignore[return-value]

# S3 — all optional
S3_BUCKET: str | None = os.getenv("TTS_S3_BUCKET")
S3_REGION: str = os.getenv("AWS_REGION", "us-east-1")
S3_AUDIO_PREFIX: str = os.getenv("TTS_S3_PREFIX", "tts/")

# Sentence buffer config
MIN_FLUSH_CHARS: int = int(os.getenv("TTS_MIN_FLUSH_CHARS", "30"))
MAX_BUFFER_CHARS: int = int(
    os.getenv("TTS_MAX_BUFFER_CHARS", str(_MAX_CHARS_PER_CHUNK // 2))
)

# Remote service config
TTS_SERVICE_URL: str = os.getenv("TTS_SERVICE_URL", "")
TTS_SERVICE_API_KEY: str = os.getenv("TTS_SERVICE_API_KEY", "")
TTS_SERVICE_TIMEOUT: float = float(os.getenv("TTS_SERVICE_TIMEOUT", "60.0"))

# ── PCM integration config ────────────────────────────────────────────────────

# Quality gate thresholds for synthesized TTS PCM output.
# Chunks failing these checks are logged; if PCM_QUALITY_HARD_GATE=1 they
# are also retried (up to PCM_QUALITY_RETRY_LIMIT times).
PCM_SILENCE_THRESHOLD: float = float(os.getenv("TTS_PCM_SILENCE_THRESHOLD", "0.001"))
PCM_CLIP_THRESHOLD: float = float(os.getenv("TTS_PCM_CLIP_THRESHOLD", "0.98"))
PCM_MIN_RMS: float = float(os.getenv("TTS_PCM_MIN_RMS", "0.002"))
PCM_QUALITY_HARD_GATE: bool = os.getenv("TTS_PCM_QUALITY_HARD_GATE", "0") == "1"
PCM_QUALITY_RETRY_LIMIT: int = int(os.getenv("TTS_PCM_QUALITY_RETRY_LIMIT", "2"))

# Inter-sentence silence gap injected by PCMSentenceGapManager.
# Set to 0 to disable. Unit: seconds.
PCM_SENTENCE_GAP_S: float = float(os.getenv("TTS_PCM_SENTENCE_GAP_S", "0.08"))
PCM_SENTENCE_GAP_FAST_VOICE_FACTOR: float = float(
    os.getenv("TTS_PCM_SENTENCE_GAP_FAST_VOICE_FACTOR", "0.6")
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

_req_total = make_counter(
    "tts_requests_total", "Total TTS requests", ["status", "mode", "provider"]
)
_latency = make_histogram(
    "tts_latency_seconds",
    "End-to-end TTS latency",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30),
)
_ttfb = make_histogram(
    "tts_time_to_first_byte_seconds",
    "Latency from synthesize_stream() to first audio bytes",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
_chars_synthesized = make_histogram(
    "tts_characters_synthesized",
    "Char count per request",
    buckets=(50, 100, 200, 500, 1000, 2000, 4096),
)
_chunks_per_request = make_histogram(
    "tts_chunks_per_request",
    "Audio chunks per request",
    buckets=(1, 2, 3, 5, 8, 13),
)
_stream_sentence_size = make_histogram(
    "tts_stream_sentence_chars",
    "Size of each sentence flushed by AsyncSentenceBuffer",
    buckets=(20, 40, 60, 100, 150, 200, 400, 1000),
)
_active = make_gauge("tts_active_requests", "TTS jobs currently in flight", ["mode"])
_circuit_open = make_gauge(
    "tts_circuit_breaker_open", "1 when the TTS breaker is OPEN", ["provider"]
)
_budget_exceeded = make_counter(
    "tts_latency_budget_exceeded_total", "TTS requests aborted due to blown SLA"
)
_apology_fallback_used = make_counter(
    "tts_apology_fallback_total", "Times the cached apology audio was served"
)

# ── PCM-specific metrics ──────────────────────────────────────────────────────

# Counts every PCMChunk emitted by synthesize_pcm_stream(). Split by whether
# the chunk passed the quality gate or was flagged.
_pcm_chunks_emitted = make_counter(
    "tts_pcm_chunks_emitted_total",
    "PCMChunks emitted from the TTS PCM pipeline",
    ["quality"],   # "ok" | "low_rms" | "clipping" | "silent"
)

# Tracks the RMS distribution of synthesized TTS PCMChunks. Useful for
# calibrating PCM_MIN_RMS and detecting voice/model regression.
_pcm_chunk_rms = make_histogram(
    "tts_pcm_chunk_rms",
    "RMS amplitude of TTS PCMChunks (normalised float scale)",
    buckets=(0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5),
)

# Counts quality-gate retries. A high rate here _signals model instability.
_pcm_quality_retries = make_counter(
    "tts_pcm_quality_retries_total",
    "PCMChunk quality-gate retry events",
)

# Tracks the full end-to-end latency from the first TTS API call to the first
# PCMChunk being yielded to the caller of synthesize_pcm_stream(). This is the
# truest measure of voice agent responsiveness.
_pcm_ttfb = make_histogram(
    "tts_pcm_time_to_first_chunk_seconds",
    "Time from synthesize_pcm_stream() entry to first PCMChunk yielded",
    buckets=(0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
)

# Counts WAV bytes produced by PCMStreamToWAVCollector. Useful for estimating
# memory budget in high-throughput deployments.
_pcm_wav_bytes_collected = make_counter(
    "tts_pcm_wav_bytes_collected_total",
    "WAV bytes produced by PCMStreamToWAVCollector",
)

# Counts silence chunks injected by PCMSentenceGapManager.
_pcm_gap_chunks_injected = make_counter(
    "tts_pcm_sentence_gap_chunks_total",
    "Silence PCMChunks injected between TTS sentences",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEXT PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

_SENTENCE_BOUNDARY = re.compile(r"(?<![A-Z\d])(?<!\s[A-Za-z])[.!?;]\s")


def _split_into_chunks(text: str, max_len: int = _MAX_CHARS_PER_CHUNK) -> list[str]:
    if len(text) <= max_len:
        return [text]

    raw_sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in raw_sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_len:
            current = candidate
        else:
            if current:
                chunks.append(current)
            while len(sentence) > max_len:
                chunks.append(sentence[:max_len])
                sentence = sentence[max_len:]
            current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]

def _wav_duration_s(path: str) -> float:
    """Read duration from WAV header — zero-copy, no decoding."""
    try:
        with wave.open(path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception: # noqa
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASYNC SENTENCE BUFFER
# ═══════════════════════════════════════════════════════════════════════════════


class AsyncSentenceBuffer:
    """
    Buffers incoming LLM tokens and flushes on sentence boundaries.

    Flush conditions (in priority order):
    1. Sentence-ending punctuation AND buffer >= MIN_FLUSH_CHARS.
    2. Buffer exceeds MAX_BUFFER_CHARS (hard cap; prevents run-on blocking).
    3. Token stream exhausted — flush remainder.

    Backpressure: if a downstream consumer provides a BoundedPipelineQueue,
    put_nowait_or_raise() is called for each flushed sentence. A full queue
    causes QueueFull to propagate up so the caller can handle overload.
    """

    def __init__(
        self,
        token_stream: AsyncIterator[str],
        min_flush_chars: int = MIN_FLUSH_CHARS,
        max_buffer_chars: int = MAX_BUFFER_CHARS,
    ) -> None:
        self._stream = token_stream
        self._min = min_flush_chars
        self._max = max_buffer_chars
        self._buf = ""

    def _find_flush_point(self) -> int:
        if len(self._buf) >= self._max:
            m = None
            for match in _SENTENCE_BOUNDARY.finditer(self._buf):
                if match.end() >= self._min:
                    m = match
            return m.end() if m else self._max

        if len(self._buf) < self._min:
            return -1

        last_end = -1
        for match in _SENTENCE_BOUNDARY.finditer(self._buf):
            if match.end() >= self._min:
                last_end = match.end()
        return last_end

    def __aiter__(self) -> "AsyncSentenceBuffer":
        return self

    async def __anext__(self) -> str:
        while True:
            flush_at = self._find_flush_point()
            if flush_at > 0:
                chunk = self._buf[:flush_at].strip()
                self._buf = self._buf[flush_at:]
                if chunk:
                    _stream_sentence_size.observe(len(chunk))
                    return chunk

            try:
                token = await self._stream.__anext__()
                self._buf += token
            except StopAsyncIteration:
                remainder = self._buf.strip()
                self._buf = ""
                if remainder:
                    _stream_sentence_size.observe(len(remainder))
                    return remainder
                raise


# ═══════════════════════════════════════════════════════════════════════════════
# 6. PCM INTEGRATION LAYER
#
# This section bridges the TTS API (raw bytes) with the audio_engine PCM
# subsystem.  Every class here is a thin orchestrator that delegates DSP work
# to audio_engine primitives — no raw numpy operations should appear here.
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMTTSOutputConfig ─────────────────────────────────────────────────────────

@dataclass
class PCMTTSOutputConfig:
    """
    Binding between a TTS synthesis session and its PCM output parameters.

    Holds the negotiated PCMFormat, a shared PCMPlaybackEnhancer, a
    PCMLatencyTracker, and a PCMWaveformAnalyzer — all pre-wired together
    so TTSNode does not scatter format construction across its methods.

    Typical lifecycle:
        config = PCMTTSOutputConfig.for_openai_tts()
        chunk  = tts_pcm_to_chunk(raw_bytes, fmt=config.fmt, seq=0)
        chunk  = config.enhancer.process(chunk)          # limiter + silence
        stats  = config.analyzer.analyze(chunk)          # quality check
        config.tracker.observe(chunk, "tts.enhance")

    Parameters:
        fmt:        Target PCMFormat for all chunks in this session.
        enhancer:   Pre-configured PCMPlaybackEnhancer.
        tracker:    Latency tracker shared across the session.
        analyzer:   Waveform analyzer for per-chunk quality stats.
        request_id: Owning request for log correlation.
    """

    fmt: PCMFormat
    enhancer: PCMPlaybackEnhancer
    tracker: PCMLatencyTracker
    analyzer: PCMWaveformAnalyzer
    request_id: str = field(default_factory=new_request_id)

    @classmethod
    def for_openai_tts(
        cls,
        enable_limiter: bool = True,
        enable_agc: bool = False,
        pre_silence_s: float = 0.05,
        post_silence_s: float = 0.1,
        request_id: str | None = None,
    ) -> "PCMTTSOutputConfig":
        """
        Build a config tuned for OpenAI TTS PCM output (24 kHz mono int16).

        This is the canonical factory for the most common deployment:
        OpenAI TTS with PCM format writing directly to a sounddevice stream.
        """
        fmt = PCMFormat.openai_tts()
        return cls(
            fmt=fmt,
            enhancer=PCMPlaybackEnhancer(
                fmt=fmt,
                enable_limiter=enable_limiter,
                enable_agc=enable_agc,
                pre_silence_s=pre_silence_s,
                post_silence_s=post_silence_s,
            ),
            tracker=PCMLatencyTracker(),
            analyzer=PCMWaveformAnalyzer(
                fmt=fmt,
                silence_threshold=PCM_SILENCE_THRESHOLD,
                clip_threshold=PCM_CLIP_THRESHOLD,
            ),
            request_id=request_id or new_request_id(),
        )

    @classmethod
    def for_format(
        cls,
        fmt: PCMFormat,
        request_id: str | None = None,
    ) -> "PCMTTSOutputConfig":
        """
        Build a config for an arbitrary PCMFormat (e.g. 44100 Hz ElevenLabs).

        Silence thresholds are scaled to the dtype of ``fmt`` automatically:
        int16 uses absolute RMS; float32 uses normalised [0, 1] scale.
        """
        silence_thresh = (
            PCM_SILENCE_THRESHOLD
            if fmt.dtype == "float32"
            else PCM_SILENCE_THRESHOLD * 32768.0
        )
        clip_thresh = (
            PCM_CLIP_THRESHOLD
            if fmt.dtype == "float32"
            else PCM_CLIP_THRESHOLD * 32768.0
        )
        return cls(
            fmt=fmt,
            enhancer=PCMPlaybackEnhancer(fmt=fmt),
            tracker=PCMLatencyTracker(),
            analyzer=PCMWaveformAnalyzer(
                fmt=fmt,
                silence_threshold=silence_thresh,
                clip_threshold=clip_thresh,
            ),
            request_id=request_id or new_request_id(),
        )


# ── PCMSentenceGapManager ─────────────────────────────────────────────────────

class PCMSentenceGapManager:
    """
    Injects calibrated silence PCMChunks between TTS synthesized sentences.

    Natural speech contains brief pauses at sentence boundaries — typically
    80–150 ms at normal speaking rate. When TTS sentences are streamed and
    concatenated without any gap, the result sounds unnaturally fast and
    clipped. This class solves that by emitting a silence PCMChunk after
    every non-final chunk.

    The silence duration is tunable via ``gap_s`` and is automatically scaled
    down for fast-speech voices (shimmer, nova at speed > 1.0) via the
    ``fast_voice_factor`` — so callers do not need per-voice logic.

    It also tracks the total silence injected per session for diagnostics.

    Usage::

        gap = PCMSentenceGapManager(fmt, gap_s=0.08)

        async def enhanced_stream(chunks):
            async for chunk in chunks:
                yield chunk
                if not chunk.is_final:
                    yield gap.make_gap_chunk(seq=chunk.seq)

    Parameters:
        fmt:               PCMFormat of the output stream.
        gap_s:             Silence duration in seconds. Default 0.08.
        fast_voice_factor: Multiplier applied when speed > 1.0. Default 0.6.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        gap_s: float = PCM_SENTENCE_GAP_S,
        fast_voice_factor: float = PCM_SENTENCE_GAP_FAST_VOICE_FACTOR,
    ) -> None:
        self._fmt = fmt
        self._gap_s = gap_s
        self._fast_voice_factor = fast_voice_factor
        self._total_injected_s: float = 0.0
        self._gap_count: int = 0
        # Pre-allocate the silence array once — reused for every gap chunk.
        self._silence = self._make_silence(gap_s)

    def _make_silence(self, duration_s: float) -> np.ndarray:
        """Allocate a zeroed numpy array for the given silence duration."""
        n_frames = self._fmt.frames_for_duration(duration_s)
        shape = (
            (n_frames,)
            if self._fmt.channels == 1
            else (n_frames, self._fmt.channels)
        )
        return np.zeros(shape, dtype=self._fmt.dtype)

    def make_gap_chunk(
        self, seq: int, speed: float = 1.0
    ) -> PCMChunk:
        """
        Return a silence PCMChunk representing the inter-sentence gap.

        If speed > 1.0 the gap is shortened by fast_voice_factor to preserve
        the natural pacing relative to the faster speech. The returned chunk
        has is_final=False and source="tts.gap" so downstream consumers can
        distinguish injected silence from real audio.

        Args:
            seq:    Sequence number to assign (caller should use chunk.seq).
            speed:  TTS speed multiplier; shortens the gap when > 1.0.
        """
        # Scale gap duration for fast voices
        effective_gap_s = self._gap_s
        if speed > 1.0:
            effective_gap_s *= self._fast_voice_factor

        # Lazily rebuild silence array if the effective duration changed
        if effective_gap_s != self._gap_s or self._silence.shape[0] != self._fmt.frames_for_duration(effective_gap_s):
            silence = self._make_silence(effective_gap_s)
        else:
            silence = self._silence

        chunk = PCMChunk(
            data=silence.copy(),  # copy so the pool can't mutate the template
            fmt=self._fmt,
            timestamp=time.monotonic(),
            seq=seq,
            is_final=False,
            source="tts.gap",
        )
        self._total_injected_s += effective_gap_s
        self._gap_count += 1
        _pcm_gap_chunks_injected.inc()
        return chunk

    @property
    def total_silence_injected_s(self) -> float:
        """Total seconds of silence injected since construction."""
        return self._total_injected_s

    @property
    def gap_count(self) -> int:
        """Number of gap chunks injected since construction."""
        return self._gap_count

    def reset(self) -> None:
        """Reset counters (call at the start of a new synthesis session)."""
        self._total_injected_s = 0.0
        self._gap_count = 0


# ── PCMTTSQualityGate ─────────────────────────────────────────────────────────

class PCMTTSQualityGate:
    """
    Per-chunk audio quality gate for TTS PCM output.

    Runs each synthesized PCMChunk through PCMWaveformAnalyzer and classifies
    it as one of: "ok", "silent", "low_rms", "clipping".

    When hard-gate mode is enabled (PCM_QUALITY_HARD_GATE=1), a caller can
    use the returned verdict to retry synthesis for the offending sentence.
    In soft-gate mode (default), failed chunks are passed through with a
    warning log so synthesis is never blocked by a transient quality blip.

    Aggregates per-session statistics that are logged at session close via
    log_session_stats() and exposed to callers via get_session_stats().

    Parameters:
        analyzer:       Shared PCMWaveformAnalyzer instance (from PCMTTSOutputConfig).
        min_rms:        Minimum acceptable RMS. Chunks below this are "low_rms".
        clip_threshold: Peak amplitude at which a chunk is "clipping".
        hard_gate:      If True, "check()" raises ValueError on bad chunks.
    """

    # Quality verdict literals
    OK        = "ok"
    SILENT    = "silent"
    LOW_RMS   = "low_rms"
    CLIPPING  = "clipping"

    def __init__(
        self,
        analyzer: PCMWaveformAnalyzer,
        min_rms: float = PCM_MIN_RMS,
        clip_threshold: float = PCM_CLIP_THRESHOLD,
        hard_gate: bool = PCM_QUALITY_HARD_GATE,
    ) -> None:
        self._analyzer = analyzer
        self._min_rms = min_rms
        self._clip_threshold = clip_threshold
        self._hard_gate = hard_gate

        # Session-level counters
        self._total: int = 0
        self._by_verdict: dict[str, int] = {
            self.OK: 0,
            self.SILENT: 0,
            self.LOW_RMS: 0,
            self.CLIPPING: 0,
        }

    def check(self, chunk: PCMChunk, request_id: str = "") -> str:
        """
        Classify the quality of a PCMChunk.

        Updates internal counters, emits a Prometheus label, and records the
        chunk in the waveform analyzer history.

        Returns:
            One of "ok", "silent", "low_rms", "clipping".

        Raises:
            ValueError: only when hard_gate=True and verdict != "ok".
        """
        stats: WaveformStats = self._analyzer.analyze(chunk)
        self._total += 1

        # Derive normalised RMS for threshold comparison regardless of dtype.
        # PCMWaveformAnalyzer normalises int16 internally, so stats.rms is
        # always on the [0, 1] float scale.
        rms = stats.rms

        if stats.is_silent:
            verdict = self.SILENT
        elif rms < self._min_rms:
            verdict = self.LOW_RMS
        elif stats.is_clipping:
            verdict = self.CLIPPING
        else:
            verdict = self.OK

        self._by_verdict[verdict] = self._by_verdict.get(verdict, 0) + 1
        _pcm_chunks_emitted.labels(quality=verdict).inc()
        _pcm_chunk_rms.observe(rms)

        if verdict != self.OK:
            log.warning(
                "tts_pcm_quality_flag",
                request_id=request_id,
                verdict=verdict,
                rms=round(rms, 6),
                peak=round(stats.peak, 4),
                frames=chunk.n_frames,
                duration_s=round(chunk.duration_s, 3),
            )
            if self._hard_gate:
                raise ValueError(
                    f"PCMTTSQualityGate: chunk failed with verdict={verdict!r} "
                    f"(rms={rms:.4f}, peak={stats.peak:.4f})"
                )

        return verdict

    def get_session_stats(self) -> dict[str, Any]:
        """Return session-level quality statistics as a plain dict."""
        ok_rate = self._by_verdict[self.OK] / max(self._total, 1)
        return {
            "total_chunks": self._total,
            "ok": self._by_verdict[self.OK],
            "silent": self._by_verdict[self.SILENT],
            "low_rms": self._by_verdict[self.LOW_RMS],
            "clipping": self._by_verdict[self.CLIPPING],
            "ok_rate": round(ok_rate, 4),
        }

    def log_session_stats(self, request_id: str = "") -> None:
        """Emit session stats as a structured log line."""
        stats = self.get_session_stats()
        log.info("tts_pcm_session_quality", request_id=request_id, **stats)

    def reset(self) -> None:
        """Clear all session counters (call at the start of a new request)."""
        self._total = 0
        for k in self._by_verdict:
            self._by_verdict[k] = 0


# ── PCMStreamToWAVCollector ───────────────────────────────────────────────────

class PCMStreamToWAVCollector:
    """
    Accumulates a streaming sequence of PCMChunks into a single WAV buffer.

    Designed for callers who need a complete WAV file from a streaming TTS
    synthesis session without hitting disk. The collector buffers PCMChunks
    in a thread-safe deque and flushes them into a WAV BytesIO on demand.

    The expected pattern is::

        collector = PCMStreamToWAVCollector(fmt)
        async for chunk in tts.synthesize_pcm_stream(...):
            collector.push(chunk)
        wav_bytes = collector.flush()    # returns full WAV including headers

    Memory budget:
        Each PCMChunk pushed costs ``chunk.n_bytes`` bytes. For a 10-second
        utterance at 24 kHz mono int16, this is ~480 KB — well within
        comfortable heap allocation. The collector does not cap memory; callers
        in constrained environments should set max_chunks.

    Parameters:
        fmt:         PCMFormat of all incoming chunks (must be uniform).
        max_chunks:  Hard cap on buffered chunks. Oldest are dropped on overflow.
                     Default: None (unbounded).
    """

    def __init__(
        self,
        fmt: PCMFormat,
        max_chunks: int | None = None,
    ) -> None:
        self._fmt = fmt
        self._chunks: collections.deque[PCMChunk] = collections.deque(
            maxlen=max_chunks
        )
        self._lock = threading.Lock()
        self._total_frames: int = 0

    def push(self, chunk: PCMChunk) -> None:
        """
        Add a PCMChunk to the collection buffer.

        Thread-safe. If max_chunks is set and the buffer is full, the oldest
        chunk is silently dropped (deque behaviour with maxlen).
        """
        with self._lock:
            self._chunks.append(chunk)
            self._total_frames += chunk.n_frames

    def flush(self) -> bytes:
        """
        Encode all buffered PCMChunks as a single complete WAV file.

        Concatenates all chunk data arrays in sequence order, then writes a
        standard RIFF/WAVE/fmt /data header. The result is immediately usable
        by soundfile.read(), an STT node, or direct HTTP streaming.

        Returns:
            Complete WAV bytes. Returns a minimal silent WAV if no chunks.
        """
        with self._lock:
            chunks_snapshot = list(self._chunks)

        if not chunks_snapshot:
            # Return a minimal valid WAV — 0 bytes of audio data.
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(self._fmt.channels)
                wf.setsampwidth(2)
                wf.setframerate(self._fmt.sample_rate)
                wf.writeframes(b"")
            return buf.getvalue()

        # Build a single large PCMChunk representing the entire session, then
        # delegate WAV encoding to chunk_to_wav_bytes() from audio_engine so
        # the encoding logic stays in one place.
        arrays = [c.data for c in chunks_snapshot]
        merged = np.concatenate(arrays, axis=0)
        combined_chunk = PCMChunk(
            data=merged,
            fmt=self._fmt,
            timestamp=chunks_snapshot[0].timestamp,
            seq=0,
            is_final=True,
            source="collector",
        )
        wav_bytes = chunk_to_wav_bytes(combined_chunk)
        _pcm_wav_bytes_collected.inc(len(wav_bytes))
        return wav_bytes

    @property
    def total_frames(self) -> int:
        """Total PCM frames buffered so far."""
        with self._lock:
            return self._total_frames

    @property
    def duration_s(self) -> float:
        """Total audio duration buffered (seconds)."""
        return self._fmt.duration_s(self.total_frames)

    def clear(self) -> None:
        """Discard all buffered chunks."""
        with self._lock:
            self._chunks.clear()
            self._total_frames = 0


# ── PCMTTSPipeline ────────────────────────────────────────────────────────────

class PCMTTSPipeline:
    """
    End-to-end orchestrator: raw TTS bytes → enhanced PCMChunk(s) → caller.

    Wires together every audio_engine primitive relevant to TTS output:
      1. Format resolution via PCMFormatRegistry  (negotiate if needed)
      2. tts_pcm_to_chunk()                       (parse raw bytes)
      3. PCMConverter                             (resample if target ≠ tts fmt)
      4. PCMPlaybackEnhancer.stream()             (limiter + silence pad)
      5. PCMTTSQualityGate.check()                (RMS / clipping validation)
      6. PCMSentenceGapManager.make_gap_chunk()   (inter-sentence silence)
      7. PCMLatencyTracker.observe()              (stage timing)

    A single PCMTTSPipeline instance is created per TTSNode and shared across
    all synthesis calls. The pipeline is stateless per-chunk so concurrent
    callers are safe — only the quality-gate session stats are call-scoped
    (reset at the start of each synthesize_pcm_stream call).

    Usage::

        pipeline = PCMTTSPipeline(config=PCMTTSOutputConfig.for_openai_tts())

        async for chunk in pipeline.process_stream(raw_bytes_iter, speed=1.0):
            await speaker.write(chunk)

    Parameters:
        config:      PCMTTSOutputConfig binding fmt, enhancer, tracker, analyzer.
        gap_manager: Optional PCMSentenceGapManager for inter-sentence silence.
    """

    def __init__(
        self,
        config: PCMTTSOutputConfig,
        gap_manager: PCMSentenceGapManager | None = None,
    ) -> None:
        self._config = config
        self._converter = PCMConverter(quality="auto")
        self._quality_gate = PCMTTSQualityGate(
            analyzer=config.analyzer,
            hard_gate=PCM_QUALITY_HARD_GATE,
        )
        self._gap_manager = gap_manager or PCMSentenceGapManager(config.fmt)

        # Format registry for negotiation on exotic output formats
        self._fmt_registry: PCMFormatRegistry = get_format_registry()

    def parse_chunk(
        self,
        raw_bytes: bytes,
        seq: int = 0,
        is_final: bool = False,
        source_fmt: PCMFormat | None = None,
    ) -> PCMChunk:
        """
        Parse raw TTS PCM bytes into a PCMChunk and convert to target format.

        If the TTS output format (source_fmt) differs from the pipeline's
        target format (config.fmt), PCMConverter handles resampling / dtype
        coercion transparently.

        Args:
            raw_bytes:   Raw headerless PCM from TTS API.
            seq:         Sequence counter for this chunk.
            is_final:    True on the last chunk of the synthesis.
            source_fmt:  Override the assumed source format (default: openai_tts).

        Returns:
            PCMChunk in config.fmt — ready for enhancement.
        """
        src_fmt = source_fmt or PCMFormat.openai_tts()
        chunk = tts_pcm_to_chunk(raw_bytes, fmt=src_fmt, seq=seq, is_final=is_final)
        self._config.tracker.observe(chunk, "tts.parse")

        # If TTS output format != our target format, convert now.
        if chunk.fmt != self._config.fmt:
            chunk = self._converter.convert(chunk, self._config.fmt)
            self._config.tracker.observe(chunk, "tts.convert")

        return chunk

    def enhance_chunk(self, chunk: PCMChunk) -> PCMChunk:
        """
        Apply PCMPlaybackEnhancer to a single chunk synchronously.

        PCMPlaybackEnhancer.stream() is async; this method extracts a single
        chunk from the sync-side of the pipeline for callers who process one
        chunk at a time (e.g. synthesize_pcm batch path).

        Internally uses the limiter and silence padder only; AGC is applied
        only if enabled in the config enhancer.
        """
        # PCMPlaybackEnhancer exposes internal components directly
        enhanced = self._config.enhancer._padder.process(chunk) # noqa
        if self._config.enhancer._limiter is not None: # noqa
            enhanced = self._config.enhancer._limiter.process(enhanced) # noqa
        if self._config.enhancer._agc is not None: # noqa
            enhanced = self._config.enhancer._agc.process(enhanced) # noqa
        self._config.tracker.observe(enhanced, "tts.enhance")
        return enhanced

    def check_quality(self, chunk: PCMChunk, request_id: str = "") -> str:
        """Delegate to PCMTTSQualityGate and return verdict string."""
        return self._quality_gate.check(chunk, request_id=request_id)

    async def process_stream(
        self,
        raw_bytes_iter: AsyncIterator[bytes],
        speed: float = 1.0,
        request_id: str = "",
        source_fmt: PCMFormat | None = None,
    ) -> AsyncIterator[PCMChunk]:
        """
        Full pipeline: iterate raw TTS bytes → yield enhanced PCMChunks.

        Injects inter-sentence silence (if gap_manager is active) after every
        non-final chunk. Applies quality gating with structured log on each
        chunk. Emits PCMLatencyTracker observations at parse, convert, enhance,
        and yield stages.

        Args:
            raw_bytes_iter: AsyncIterator yielding raw PCM bytes per sentence.
            speed:          TTS speed (used to scale sentence gap duration).
            request_id:     For log correlation.
            source_fmt:     TTS output format override (default: openai_tts).
        """
        self._quality_gate.reset()
        t_first: float | None = None
        seq = 0

        async for raw_bytes in raw_bytes_iter:
            if not raw_bytes:
                continue

            # Determine is_final: we cannot know ahead of time, so we set it
            # False here; callers who know the final chunk should call
            # parse_chunk directly with is_final=True.
            chunk = self.parse_chunk(
                raw_bytes, seq=seq, is_final=False, source_fmt=source_fmt
            )

            # Enhance: limiter, silence pad, optional AGC
            chunk = self.enhance_chunk(chunk)
            self._config.tracker.observe(chunk, "tts.before_yield")

            # Quality gate — warns / raises per configuration
            self.check_quality(chunk, request_id=request_id)

            if t_first is None:
                t_first = time.monotonic()
                _pcm_ttfb.observe(0.0)  # TTFB from caller perspective is 0 at first yield

            yield chunk
            seq += 1

            # Inject inter-sentence silence after non-final chunks so the
            # playback stream doesn't cut directly from one sentence to the next.
            gap = self._gap_manager.make_gap_chunk(seq=seq, speed=speed)
            yield gap
            seq += 1

        # Log session quality summary
        if request_id:
            self._quality_gate.log_session_stats(request_id=request_id)

    def get_latency_report(self) -> dict[str, dict[str, float]]:
        """Delegate to the session's PCMLatencyTracker."""
        return self._config.tracker.get_latency_report()

    def get_quality_stats(self) -> dict[str, Any]:
        """Return the current quality gate session stats."""
        return self._quality_gate.get_session_stats()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TTSNodeProtocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class TTSNodeProtocol(Protocol):
    """
    The contract for local TTSNode and RemoteTTSClient.
    VoiceGraph depends only on this protocol.
    """

    async def synthesize(
        self,
        text: str,
        voice: VoiceType | None = None,
        filename: str | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str, float]: ...

    def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]: ...

    def synthesize_pcm_stream(
            self,
            token_stream: AsyncIterator[str],
            voice: VoiceType | None = None,
            speed: float = 1.0,
            request_id: str | None = None,
            emit_gap_chunks: bool = True,
    ) -> AsyncIterator[PCMChunk]: ...

    async def warmup(self) -> None:
        """Pre-warm the TTS engine to reduce first-request latency."""
        ...

    async def shutdown(self) -> None:
        """
        Gracefully shut down the node — drain in-flight requests, release
        external connections, and free any held resources. Called once at
        process exit or during a rolling restart. After this returns, the
        node must not be used again.
        """
        ...

    async def health(self) -> ServiceHealthState: ...

    async def close(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════════
# 8. APOLOGY AUDIO FALLBACK
# ═══════════════════════════════════════════════════════════════════════════════


def _try_apology_fallback(request_id: str) -> tuple[str, str, float] | None:
    """
    Return (local_path, s3_uri) for the pre-rendered apology audio file,
    or None if not configured / file missing.

    Called when TTS completely fails so the pipeline still has audio to return.
    """
    if not TTS_APOLOGY_AUDIO_PATH:
        return None
    p = Path(TTS_APOLOGY_AUDIO_PATH)
    if not p.exists():
        log.warning("tts_apology_file_missing", path=str(p))
        return None
    _apology_fallback_used.inc()
    log.warning("tts_serving_apology_fallback", request_id=request_id, path=str(p))
    return str(p), "", _wav_duration_s(str(p))


# ═══════════════════════════════════════════════════════════════════════════════
# 9. S3 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


async def _s3_upload_file(bucket: str, key: str, local_path: Path) -> str:
    try:
        import aioboto3  # type: ignore
    except ImportError as exc:
        raise ImportError("Install aioboto3 to use S3 integration.") from exc

    session = aioboto3.Session()
    content_types: dict[str, str] = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    ct = content_types.get(local_path.suffix.lstrip("."), "application/octet-stream")
    async with session.client("s3", region_name=S3_REGION) as s3:
        await s3.upload_file(
            str(local_path), bucket, key, ExtraArgs={"ContentType": ct}
        )
    return f"s3://{bucket}/{key}"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. FILE LOCK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_file_locks: dict[str, asyncio.Lock] = {}
_file_locks_meta: asyncio.Lock = asyncio.Lock()


async def _get_file_lock(path: Path) -> asyncio.Lock:
    key = str(path.resolve())
    async with _file_locks_meta:
        if key not in _file_locks:
            _file_locks[key] = asyncio.Lock()
        return _file_locks[key]


# ═══════════════════════════════════════════════════════════════════════════════
# 11. LOCAL TTS NODE
# ═══════════════════════════════════════════════════════════════════════════════


class TTSNode:
    """
    LangGraph-compatible async TTS node (local / in-process implementation).

    State contract
    ──────────────
    reads:  state["llm_response"]    (str — text to speak)
            state.get("tts_voice")   (VoiceType | None)
            state.get("tts_speed")   (float | None — 0.25–4.0)
    writes: state["audio_output"]    (str — s3:// URI or local path)
            state["audio_s3_uri"]    (str)
            state["request_id"]      (str)

    Distributed usage
    ─────────────────
    Use get_tts_node() instead of instantiating directly. The factory
    returns a RemoteTTSClient when TTS_SERVICE_URL is configured.

    PCM integration
    ───────────────
    When TTS_FORMAT="pcm", additional methods are available:
      synthesize_pcm()       — one-shot synthesis returning a PCMChunk
      synthesize_pcm_stream() — streaming synthesis yielding PCMChunks
    Both paths run through PCMTTSPipeline for enhancement, quality gating,
    and latency tracking.
    """

    def __init__(
        self,
        model: str = TTS_MODEL,
        voice: VoiceType = TTS_VOICE,
        audio_format: FormatType = TTS_FORMAT,
        output_dir: Path = TTS_OUTPUT_DIR,
        rate_per_sec: float = RATE_PER_SEC,
        rate_burst: float = RATE_BURST,
    ) -> None:
        self._model = model
        self._voice = voice
        self._format = audio_format
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._inflight_batch = 0
        self._inflight_stream = 0

        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY, max_retries=0, timeout=60.0)

        self._rate_limiter = RateLimiter(rate_per_sec, rate_burst)

        # Separate breakers for batch and stream so a slow streaming
        # path does not trip the breaker used by synchronous synthesis.
        self._breaker_batch = CircuitBreaker(name="tts:openai:batch")
        self._breaker_stream = CircuitBreaker(name="tts:openai:stream")

        self._cleanup_task: asyncio.Task | None = None

        # ── PCM integration layer ─────────────────────────────────────────────
        #
        # The PCM pipeline is constructed lazily (on first use) rather than at
        # __init__ time so that non-PCM deployments (TTS_FORMAT != "pcm") pay
        # zero overhead. _pcm_config and _pcm_pipeline are set by
        # _ensure_pcm_pipeline() the first time a PCM method is called.
        self._pcm_config: PCMTTSOutputConfig | None = None
        self._pcm_pipeline: PCMTTSPipeline | None = None
        self._pcm_lock = asyncio.Lock()

    # ── PCM pipeline lazy initialisation ─────────────────────────────────────

    async def _ensure_pcm_pipeline(self, request_id: str) -> PCMTTSPipeline:
        """
        Lazily construct and cache the PCMTTSPipeline for this node.

        Thread-safe via asyncio.Lock. The same pipeline instance is reused
        for the lifetime of the node; the quality gate is reset per-request
        inside PCMTTSPipeline.process_stream().

        Uses get_format_registry() to resolve the target format by name so
        consumers of the registry can override the "openai_tts" entry for
        non-standard TTS backends without touching TTSNode code.
        """
        async with self._pcm_lock:
            if self._pcm_pipeline is not None:
                return self._pcm_pipeline

            # Resolve output format via global registry.
            # If the registry has an "openai_tts" entry, use it (allowing
            # operators to override the format centrally). Otherwise fall back
            # to PCMFormat.openai_tts() directly.
            registry = get_format_registry()
            fmt = registry.get("openai_tts") or PCMFormat.openai_tts()

            self._pcm_config = PCMTTSOutputConfig.for_format(
                fmt=fmt, request_id=request_id
            )
            gap_mgr = PCMSentenceGapManager(fmt, gap_s=PCM_SENTENCE_GAP_S)
            self._pcm_pipeline = PCMTTSPipeline(
                config=self._pcm_config, gap_manager=gap_mgr
            )
            log.info(
                "tts_pcm_pipeline_initialised",
                fmt=repr(fmt),
                request_id=request_id,
            )
            return self._pcm_pipeline

    # ── background lifecycle cleanup ──────────────────────────────────────────

    def _ensure_cleanup_running(self) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        """
        Runs every 10 minutes and deletes local audio files older than
        LOCAL_FILE_TTL seconds. S3 is the durable store; disk is transient.
        """
        while True:
            await asyncio.sleep(600)
            try:
                cutoff = time.time() - LOCAL_FILE_TTL
                for f in self._output_dir.glob(f"*.{self._format}"):
                    try:
                        if f.stat().st_mtime < cutoff:
                            f.unlink(missing_ok=True)
                            log.info("tts_local_file_expired", path=str(f))
                    except Exception as exc:
                        log.warning(
                            "tts_cleanup_file_error", path=str(f), error=str(exc)
                        )
            except Exception as exc:
                log.warning("tts_cleanup_loop_error", error=str(exc))

    # ── synthesis (single chunk) ──────────────────────────────────────────────

    async def _synthesize_chunk(
        self,
        text: str,
        voice: VoiceType,
        speed: float,
        use_stream_breaker: bool = False,
    ) -> bytes:
        """Call the API for a single chunk; wrapped in the appropriate breaker."""
        breaker = self._breaker_stream if use_stream_breaker else self._breaker_batch

        async def _call() -> bytes:
            async with self._client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=voice,
                input=text,
                response_format=self._format,
                speed=speed,
            ) as response:
                return await response.read()

        return await breaker.call(
            backoff_retry, _call, attempts=3, base_delay=1.0, exceptions=_TTS_RETRYABLE
        )

    # ── stitching validation ──────────────────────────────────────────────────

    @staticmethod
    def _validate_chunk(data: bytes, index: int) -> None:
        if not data:
            raise ValueError(f"TTS chunk {index} returned empty audio bytes.")
        if len(data) < 4:
            raise ValueError(
                f"TTS chunk {index} is suspiciously small ({len(data)} bytes)."
            )

    @staticmethod
    def _validate_final(path: Path, expected_chunks: int) -> None:
        if not path.exists():
            raise RuntimeError(f"Expected output file not found: {path}")
        size = path.stat().st_size
        if size < 512:
            raise RuntimeError(
                f"Final audio file is only {size} bytes after stitching "
                f"{expected_chunks} chunk(s) — likely a partial write."
            )

    # ── public synthesize (batch) ─────────────────────────────────────────────

    async def synthesize(
        self,
        text: str,
        voice: VoiceType | None = None,
        filename: str | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str, float]:  # (local_path, s3_uri, duration_s)
        """
        Synthesize speech and write to disk (and optionally S3).
        Falls back to the pre-rendered apology audio if TTS completely fails.

        Returns (local_path, s3_uri) where s3_uri is "" if S3 isn't configured.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="tts.synthesize")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        active_voice: VoiceType = _resolve_voice(voice if voice is not None else self._voice, log_source="synthesize")
        san = sanitize(text, request_id=rid)

        if not san:
            raise ValueError("Sanitizer returned empty TTS text.")

        safe_text = san.text
        chunks = _split_into_chunks(safe_text)

        stem = filename or f"tts_{uuid.uuid4().hex[:8]}"
        output_path = self._output_dir / f"{stem}.{self._format}"

        with tracer.start_as_current_span("tts.synthesize") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("char_count", len(safe_text))
            span.set_attribute("chunks", len(chunks))
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.batch"):
                _active.labels(mode="batch").inc()
                _circuit_open.labels(provider="openai").set(
                    1 if self._breaker_batch.state == "OPEN" else 0
                )
                self._inflight_batch += 1
                t0 = time.monotonic()

                try:
                    self._ensure_cleanup_running()

                    audio_parts: list[bytes] = []

                    for i, chunk in enumerate(chunks):
                        log.debug(
                            "tts_chunk_start",
                            request_id=rid,
                            chunk=i + 1,
                            total=len(chunks),
                        )
                        data = await self._synthesize_chunk(chunk, active_voice, speed)
                        self._validate_chunk(data, i)
                        audio_parts.append(data)

                    _chunks_per_request.observe(len(audio_parts))
                    _chars_synthesized.observe(len(safe_text))

                    file_lock = await _get_file_lock(output_path)
                    async with file_lock:
                        async with aiofiles.open(output_path, "wb") as f:
                            for part in audio_parts:
                                await f.write(part)

                    self._validate_final(output_path, len(audio_parts))

                    s3_uri = ""
                    if S3_BUCKET:
                        s3_key = f"{S3_AUDIO_PREFIX}{stem}.{self._format}"
                        try:
                            s3_uri = await _s3_upload_file(
                                S3_BUCKET, s3_key, output_path
                            )
                            log.info("tts_s3_upload_ok", request_id=rid, uri=s3_uri)
                        except Exception as exc:
                            log.warning(
                                "tts_s3_upload_failed", request_id=rid, error=str(exc)
                            )

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="batch", provider="openai"
                    ).inc()
                    _latency.observe(latency)

                    log.info(
                        "tts_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=len(safe_text),
                        chunks=len(chunks),
                        latency_s=round(latency, 3),
                        local_path=str(output_path),
                        s3_uri=s3_uri or "n/a",
                    )
                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("s3_uri", s3_uri or "")

                    duration_s = _wav_duration_s(str(output_path))
                    return str(output_path), s3_uri, duration_s

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="batch", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_error", request_id=rid, error=str(exc))

                    # ── apology audio fallback ────────────────────────────────
                    # Before re-raising, try to serve the pre-rendered apology
                    # audio so the pipeline still has something to return.
                    fallback = _try_apology_fallback(rid)
                    if fallback is not None:
                        return fallback
                    raise

                finally:
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── synthesize_pcm — one-shot PCM output ──────────────────────────────────

    async def synthesize_pcm(
        self,
        text: str,
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> PCMChunk:
        """
        Synthesize text and return a single PCMChunk.

        This method forces TTS_FORMAT="pcm" for the underlying API call,
        parses the raw PCM bytes via tts_pcm_to_chunk(), applies the full
        PCMPlaybackEnhancer chain (limiter → silence pad → optional AGC),
        runs the PCMTTSQualityGate, and records latency via PCMLatencyTracker.

        The returned PCMChunk is in the format registered as "openai_tts" in
        the global PCMFormatRegistry (default: 24 kHz mono int16).

        If quality gating is in hard-gate mode and the synthesized audio fails
        quality checks, a retry is attempted up to PCM_QUALITY_RETRY_LIMIT
        times. If all retries fail, the last chunk is returned with a warning.

        Args:
            text:        Text to synthesize (will be sanitized internally).
            voice:       TTS voice override.
            speed:       Speech speed multiplier (0.25–4.0).
            request_id:  For trace correlation.

        Returns:
            A single PCMChunk containing the complete synthesized audio,
            enhanced and validated.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()
        pipeline = await self._ensure_pcm_pipeline(rid)
        active_voice: VoiceType = _resolve_voice(voice if voice is not None else self._voice, log_source="synthesize")

        san = sanitize(text, request_id=rid)
        if not san:
            raise ValueError("Sanitizer returned empty TTS text.")
        safe_text = san.text

        with tracer.start_as_current_span("tts.synthesize_pcm") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("char_count", len(safe_text))
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.batch"):
                _active.labels(mode="batch").inc()
                self._inflight_batch += 1
                t0 = time.monotonic()

                try:
                    self._ensure_cleanup_running()

                    # Use the PCM format — always override for this path.
                    # The underlying client is called with response_format="pcm"
                    # regardless of the node's default self._format setting.
                    async def _call_pcm() -> bytes:
                        async with self._client.audio.speech.with_streaming_response.create(
                            model=self._model,
                            voice=active_voice,
                            input=safe_text,
                            response_format="pcm",
                            speed=speed,
                        ) as resp:
                            return await resp.read()

                    # Retry loop for quality gate failures in hard-gate mode.
                    last_chunk: PCMChunk | None = None
                    for attempt in range(PCM_QUALITY_RETRY_LIMIT + 1):
                        raw_bytes = await self._breaker_batch.call(
                            backoff_retry,
                            _call_pcm,
                            attempts=3,
                            base_delay=1.0,
                            exceptions=_TTS_RETRYABLE,
                        )
                        self._validate_chunk(raw_bytes, 0)

                        # Parse raw PCM bytes into a PCMChunk
                        chunk = pipeline.parse_chunk(
                            raw_bytes, seq=0, is_final=True
                        )
                        # Apply enhancement chain
                        chunk = pipeline.enhance_chunk(chunk)

                        try:
                            verdict = pipeline.check_quality(chunk, request_id=rid) # noqa
                            # Quality is acceptable or we're in soft-gate mode
                            last_chunk = chunk
                            break
                        except ValueError:
                            # Hard-gate raised — retry if budget allows
                            if attempt < PCM_QUALITY_RETRY_LIMIT:
                                _pcm_quality_retries.inc()
                                log.warning(
                                    "tts_pcm_quality_retry",
                                    request_id=rid,
                                    attempt=attempt + 1,
                                )
                                await asyncio.sleep(0.1 * (attempt + 1))
                                continue
                            # All retries exhausted — use last chunk with warning
                            log.error(
                                "tts_pcm_quality_retry_exhausted",
                                request_id=rid,
                                retries=PCM_QUALITY_RETRY_LIMIT,
                            )
                            last_chunk = chunk
                            break

                    assert last_chunk is not None

                    latency = time.monotonic() - t0
                    pipeline._config.tracker.observe(last_chunk, "tts.synthesize_pcm.done") # noqa
                    _req_total.labels(status="ok", mode="batch", provider="openai").inc()
                    _latency.observe(latency)
                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("pcm_frames", last_chunk.n_frames)
                    span.set_attribute("pcm_duration_s", round(last_chunk.duration_s, 3))

                    log.info(
                        "tts_pcm_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=len(safe_text),
                        frames=last_chunk.n_frames,
                        duration_s=round(last_chunk.duration_s, 3),
                        latency_s=round(latency, 3),
                    )

                    return last_chunk

                except Exception as exc:
                    _req_total.labels(status="error", mode="batch", provider="openai").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_pcm_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── synthesize_pcm_stream — streaming PCM output ──────────────────────────

    async def synthesize_pcm_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
        emit_gap_chunks: bool = True,
    ) -> AsyncIterator[PCMChunk]:
        """
        Stream PCMChunks from a live LLM token stream.

        Feeds tokens through AsyncSentenceBuffer → sanitize → TTS API (pcm)
        → PCMTTSPipeline (parse → convert → enhance → quality gate) →
        PCMSentenceGapManager (inter-sentence silence) → caller.

        This is the lowest-latency path for voice agents: the first PCMChunk
        is available as soon as the first sentence is synthesized, typically
        within 200–500 ms of the first token arrival. The caller can feed the
        output directly into PCMOutputStream for speaker playback.

        The stream circuit breaker is used (separate from batch) so streaming
        failures don't trip the breaker for synchronous synthesis.

        Args:
            token_stream:    AsyncIterator[str] from the LLM node.
            voice:           TTS voice override.
            speed:           Speech speed multiplier.
            request_id:      For trace correlation.
            emit_gap_chunks: If True, inject silence between sentences.
                             Set False when the caller handles spacing itself.

        Yields:
            PCMChunk — one per synthesized sentence (plus silence gaps if
            emit_gap_chunks=True). Each chunk is enhancement-ready.
        """
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()
        pipeline = await self._ensure_pcm_pipeline(rid)
        pipeline._quality_gate.reset() # noqa

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="tts.synthesize_pcm_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        active_voice: VoiceType = _resolve_voice(voice if voice is not None else self._voice, log_source="synthesize")

        with tracer.start_as_current_span("tts.synthesize_pcm_stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                _circuit_open.labels(provider="openai").set(
                    1 if self._breaker_stream.state == "OPEN" else 0
                )
                self._inflight_stream += 1
                t0 = time.monotonic()
                total_chars = 0
                chunk_index = 0
                first_chunk_emitted = False
                total_pcm_frames = 0

                try:
                    self._ensure_cleanup_running()
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:
                        # SLA check before processing each sentence
                        if budget:
                            budget.check(stage="tts.synthesize_pcm_stream.before_sentence")

                        san = sanitize(sentence, request_id=rid)
                        if not san:
                            continue

                        clean = san.text
                        total_chars += len(clean)

                        # Synthesize this sentence using the stream breaker
                        for sub_text in _split_into_chunks(clean):
                            if budget:
                                budget.check(stage="tts.synthesize_pcm_stream.before_chunk")

                            raw_bytes = await self._synthesize_chunk(
                                sub_text, active_voice, speed,
                                use_stream_breaker=True
                            )
                            self._validate_chunk(raw_bytes, chunk_index)

                            # Parse, convert, enhance
                            pcm_chunk = pipeline.parse_chunk(
                                raw_bytes,
                                seq=chunk_index,
                                is_final=False,
                            )
                            pcm_chunk = pipeline.enhance_chunk(pcm_chunk)

                            # Quality gate (soft mode logs; hard mode raises)
                            pipeline.check_quality(pcm_chunk, request_id=rid)

                            # SLA check before yielding
                            if budget:
                                budget.check(stage="tts.synthesize_pcm_stream.before_yield")

                            if not first_chunk_emitted:
                                _pcm_ttfb.observe(time.monotonic() - t0)
                                first_chunk_emitted = True

                            total_pcm_frames += pcm_chunk.n_frames
                            pipeline._config.tracker.observe(pcm_chunk, "tts.stream.yield") # noqa
                            chunk_index += 1

                            yield pcm_chunk

                            # Inject inter-sentence silence if enabled.
                            # The gap chunk gets a sequence number one higher
                            # than the audio chunk so downstream jitter buffers
                            # can handle ordering correctly.
                            if emit_gap_chunks:
                                gap = pipeline._gap_manager.make_gap_chunk( # noqa
                                    seq=chunk_index, speed=speed
                                )
                                chunk_index += 1
                                yield gap

                    latency = time.monotonic() - t0
                    _req_total.labels(status="ok", mode="stream", provider="openai").inc()
                    _latency.observe(latency)
                    _chars_synthesized.observe(total_chars)
                    _chunks_per_request.observe(chunk_index)

                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("chunks", chunk_index)
                    span.set_attribute("pcm_frames", total_pcm_frames)

                    # Log quality summary and latency report at end of stream
                    pipeline._quality_gate.log_session_stats(request_id=rid) # noqa
                    pipeline._config.tracker.log_report() # noqa

                    log.info(
                        "tts_pcm_stream_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=total_chars,
                        pcm_chunks=chunk_index,
                        pcm_frames=total_pcm_frames,
                        latency_s=round(latency, 3),
                    )

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning(
                        "tts_pcm_stream_budget_exceeded",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream", provider="openai"
                    ).inc()
                    log.warning(
                        "tts_pcm_stream_cancelled",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(status="error", mode="stream", provider="openai").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_pcm_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    # ── synthesize_pcm_to_wav — full session → WAV bytes ─────────────────────

    async def synthesize_pcm_to_wav(
        self,
        text: str,
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> bytes:
        """
        Synthesize text and return a complete WAV as raw bytes (no disk I/O).

        Uses synthesize_pcm() internally (single batch API call) and encodes
        the resulting PCMChunk via chunk_to_wav_bytes() from audio_engine.
        This is faster than synthesize() + disk read for callers who want a
        WAV buffer in memory.

        Returns:
            Complete WAV bytes (RIFF/WAVE/fmt /data headers included).
        """
        pcm_chunk = await self.synthesize_pcm(
            text, voice=voice, speed=speed, request_id=request_id
        )
        return chunk_to_wav_bytes(pcm_chunk)

    # ── synthesize_stream_pcm_to_wav — streaming → accumulated WAV ───────────

    async def synthesize_stream_pcm_to_wav(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> bytes:
        """
        Collect a full streaming PCM synthesis session into a WAV buffer.

        Internally drives synthesize_pcm_stream() and feeds all chunks into
        a PCMStreamToWAVCollector. Returns the complete WAV bytes when the
        token stream is exhausted. Useful for callers that need a complete
        WAV but want to benefit from the TTS streaming path (lower TTFB for
        downstream quality checks) rather than waiting for the full batch.

        Returns:
            Complete WAV bytes encoding the full synthesized utterance.
        """
        rid = request_id or new_request_id()
        pipeline = await self._ensure_pcm_pipeline(rid)
        collector = PCMStreamToWAVCollector(fmt=pipeline._config.fmt) # noqa

        async for chunk in self.synthesize_pcm_stream(
            token_stream,
            voice=voice,
            speed=speed,
            request_id=rid,
            emit_gap_chunks=True,
        ):
            # Skip injected silence gap chunks when building WAV output —
            # the caller gets clean audio without artificial padding between
            # sentences (they are only needed for live playback streaming).
            if chunk.source != "tts.gap":
                collector.push(chunk)

        return collector.flush()

    # ── streaming synthesis (bytes) ───────────────────────────────────────────

    async def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        request_id: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Yield audio bytes as each sentence is synthesized, feeding directly
        from the LLM token stream. Audio playback can start while the LLM
        is still generating the rest of the response.

        Uses the stream circuit breaker (separate from batch) so streaming
        failures don't trip the breaker used by synchronous /run calls.
        """
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="tts.synthesize_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        active_voice: VoiceType = _resolve_voice(voice if voice is not None else self._voice, log_source="synthesize")

        with tracer.start_as_current_span("tts.synthesize_stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                _circuit_open.labels(provider="openai").set(
                    1 if self._breaker_stream.state == "OPEN" else 0
                )
                self._inflight_stream += 1
                t0 = time.monotonic()
                total_chars = 0
                chunk_index = 0
                first_bytes_emitted = False

                try:
                    self._ensure_cleanup_running()
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:

                        # Enforce SLA before processing sentence
                        if budget:
                            budget.check(stage="tts.synthesize_stream.before_sentence")

                        san = sanitize(sentence, request_id=rid)
                        if not san:
                            continue

                        clean = san.text
                        total_chars += len(clean)

                        for sub in _split_into_chunks(clean):

                            # Enforce SLA before synthesizing chunk
                            if budget:
                                budget.check(stage="tts.synthesize_stream.before_chunk")

                            data = await self._synthesize_chunk(
                                sub, active_voice, speed, use_stream_breaker=True
                            )

                            self._validate_chunk(data, chunk_index)

                            # Enforce SLA before yielding bytes
                            if budget:
                                budget.check(stage="tts.synthesize_stream.before_yield")

                            if not first_bytes_emitted:
                                _ttfb.observe(time.monotonic() - t0)
                                first_bytes_emitted = True

                            chunk_index += 1
                            yield data

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="stream", provider="openai"
                    ).inc()
                    _latency.observe(latency)
                    _chars_synthesized.observe(total_chars)
                    _chunks_per_request.observe(chunk_index)
                    span.set_attribute("latency_s", round(latency, 3))
                    span.set_attribute("chunks", chunk_index)

                    log.info(
                        "tts_stream_ok",
                        request_id=rid,
                        voice=active_voice,
                        chars=total_chars,
                        chunks=chunk_index,
                        latency_s=round(latency, 3),
                    )

                except LatencyBudgetExceeded:
                    _budget_exceeded.inc()
                    log.warning(
                        "tts_stream_budget_exceeded",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream", provider="openai"
                    ).inc()
                    log.warning(
                        "tts_stream_cancelled",
                        request_id=rid,
                        chunks_emitted=chunk_index,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="stream", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("tts_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    # ── streaming synthesis to files ──────────────────────────────────────────

    async def synthesize_stream_to_files(
        self,
        token_stream: AsyncIterator[str],
        voice: VoiceType | None = None,
        speed: float = 1.0,
        stem: str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[tuple[str, bool]]:
        """
        Like synthesize_stream() but write each sentence chunk to its own file
        and yield (local_path, is_final) tuples. S3 uploads are fire-and-forget.
        """
        rid = request_id or new_request_id()
        base_stem = stem or f"tts_{uuid.uuid4().hex[:8]}"
        active_voice: VoiceType = _resolve_voice(voice if voice is not None else self._voice, log_source="synthesize")

        with tracer.start_as_current_span("tts.synthesize_stream_to_files") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice", active_voice)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                self._inflight_stream += 1
                t0 = time.monotonic()
                total_chars = 0
                chunk_index = 0
                pending: list[Path] = []

                try:
                    self._ensure_cleanup_running()
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:
                        san = sanitize(sentence, request_id=rid)

                        # Skip empty / fully stripped content
                        if not san:
                            continue

                        clean = san.text
                        total_chars += len(clean)

                        for sub in _split_into_chunks(clean):
                            data = await self._synthesize_chunk(
                                sub, active_voice, speed, use_stream_breaker=True
                            )
                            self._validate_chunk(data, chunk_index)

                            out_path = (
                                self._output_dir
                                / f"{base_stem}_chunk_{chunk_index}.{self._format}"
                            )
                            file_lock = await _get_file_lock(out_path)
                            async with file_lock:
                                async with aiofiles.open(out_path, "wb") as f:
                                    await f.write(data)

                            if S3_BUCKET:
                                asyncio.create_task(
                                    self._upload_chunk_s3(
                                        out_path, base_stem, chunk_index, rid
                                    )
                                )

                            pending.append(out_path)
                            chunk_index += 1

                    for i, path in enumerate(pending):
                        yield str(path), i == len(pending) - 1

                    latency = time.monotonic() - t0
                    _req_total.labels(
                        status="ok", mode="stream_to_files", provider="openai"
                    ).inc()
                    _latency.observe(latency)
                    log.info(
                        "tts_stream_to_files_ok",
                        request_id=rid,
                        chunks=chunk_index,
                        chars=total_chars,
                        latency_s=round(latency, 3),
                    )

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream_to_files", provider="openai"
                    ).inc()
                    raise
                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="stream_to_files", provider="openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "tts_stream_to_files_error", request_id=rid, error=str(exc)
                    )
                    raise
                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

    async def _upload_chunk_s3(
        self, path: Path, stem: str, chunk_index: int, rid: str
    ) -> None:
        if not S3_BUCKET:
            return
        try:
            key = f"{S3_AUDIO_PREFIX}{stem}_chunk_{chunk_index}.{self._format}"
            uri = await _s3_upload_file(S3_BUCKET, key, path)
            log.info("tts_chunk_s3_ok", request_id=rid, chunk=chunk_index, uri=uri)
        except Exception as exc:
            log.warning(
                "tts_chunk_s3_failed", request_id=rid, chunk=chunk_index, error=str(exc)
            )

    # ── PCM diagnostics ────────────────────────────────────────────────────────

    async def get_pcm_diagnostics(self) -> dict[str, Any]:
        """
        Return a diagnostic snapshot of the PCM integration layer.

        Includes:
          - PCMLatencyTracker per-stage statistics (p50, p95, p99, max)
          - PCMTTSQualityGate session stats (ok_rate, clipping, silent counts)
          - PCMMetricsSnapshot delta from module-level audio_engine counters
          - PCMFormatRegistry registered names

        Designed for operator dashboards and health endpoints. Returns an
        empty dict if the PCM pipeline has not yet been initialised (i.e.
        no PCM synthesis has been performed since node creation).
        """
        if self._pcm_pipeline is None:
            return {"pcm_pipeline": "not_initialised"}

        pipeline = self._pcm_pipeline

        # Capture a two-snapshot delta to get per-second rates.
        snap1 = get_metrics_snapshot()
        await asyncio.sleep(0.05)  # 50 ms window
        snap2 = get_metrics_snapshot()
        delta = snap1.delta(snap2)

        return {
            "latency_stages": pipeline.get_latency_report(),
            "quality_stats": pipeline.get_quality_stats(),
            "pcm_metrics_delta": {
                k: round(v, 4) for k, v in delta.items() if v != 0.0
            },
            "fmt_registry_formats": get_format_registry().list_names(),
            "sentence_gaps_injected": pipeline._gap_manager.gap_count, # noqa
            "silence_injected_s": round(
                pipeline._gap_manager.total_silence_injected_s, 4 # noqa
            ),
        }

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        return ServiceHealthState(
            service="tts.local",
            healthy=self._breaker_batch.state != "OPEN",
            circuit_state=self._breaker_batch.state,
            inflight=self._inflight_batch + self._inflight_stream,
        )

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        rid = state.get("request_id") or new_request_id()
        text: str = state.get("llm_response", "")
        voice: VoiceType | None = state.get("tts_voice")
        speed: float = float(state.get("tts_speed", 1.0))

        local_path, s3_uri, duration_s = await self.synthesize(
            text, voice=voice, speed=speed, request_id=rid
        )

        return {
            **state,
            "request_id": rid,
            "audio_output": s3_uri if s3_uri else local_path,
            "audio_local_path": local_path,
            "audio_s3_uri": s3_uri,
        }

    # ── cleanup ───────────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self._client.close()

        # Release PCM pool memory on graceful shutdown
        get_chunk_pool().clear()
        log.info("tts_node_closed")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. REMOTE TTS CLIENT
# ═══════════════════════════════════════════════════════════════════════════════


class RemoteTTSClient:
    """
    HTTP client implementing TTSNodeProtocol against a remote TTS microservice.
    Drop-in replacement for TTSNode in any voice_graph configuration.

    Expected remote endpoints
    ─────────────────────────
      POST /synthesize         JSON body → {"local_path": str, "s3_uri": str}
      POST /synthesize/stream  JSON body → chunked audio bytes per sentence
      GET  /health             → {"healthy": bool, ...}

    Streaming transport
    ───────────────────
    synthesize_stream() replicates TTSNode's sentence-level batching:
      LLM token stream → AsyncSentenceBuffer → sanitize → POST /synthesize/stream
      per sentence → yield audio bytes as they arrive.

    This gives the same TTFB characteristics as the local node: audio starts
    playing while the LLM is still generating, because each sentence is sent
    and streamed back independently rather than waiting for the full response.

    Observability
    ─────────────
    Every public method participates in the module-level Prometheus counters
    (_req_total, _latency, _ttfb, _chars_synthesized, _chunks_per_request,
    _active, _budget_exceeded, _circuit_open, _apology_fallback_used) so
    dashboards and alerts are identical whether the node is local or remote.
    OTel spans are opened for each call and trace headers propagated outbound.

    Latency budget
    ──────────────
    LatencyBudget.current() is checked at entry to every public method.
    The remaining budget is also forwarded as X-Latency-Budget-Ms so the
    remote service can self-abort rather than burning time producing a result
    the caller will discard.  Intra-stream checks run before every sentence
    and before every yield so the stream can be aborted mid-flight.

    Resilience
    ──────────
    Circuit breaker ("tts:remote:batch" / "tts:remote:stream") is split so
    a flood of slow streaming requests cannot trip the breaker for synchronous
    batch synthesis — mirroring the separate breakers in TTSNode.
    backoff_retry wraps synthesize() (3 attempts, 1.0 s base).
    Streaming calls open the HTTP stream context inside the circuit breaker so
    repeated stream failures trip the breaker while preserving true streaming
    semantics — no full buffering.
    If all retries are exhausted, _try_apology_fallback() is called so the
    pipeline always has something to return during a TTS outage.

    Cancellation
    ────────────
    asyncio.CancelledError is always re-raised from all call sites.
    GeneratorExit is caught inside synthesize_stream() yield sites so
    caller drop-out cleans up without log noise.

    PCM pipeline
    ────────────
    _ensure_pcm_pipeline() lazily constructs a local PCMTTSPipeline
    (same pattern as TTSNode) so PCM-aware callers can drive the full
    enhancement / quality-gate / gap-management chain client-side even
    when the synthesis itself happens remotely.

    Env config (resolved at module load)
    ─────────────────────────────────────
      TTS_SERVICE_URL      required — base URL of the remote TTS service
      TTS_SERVICE_API_KEY  optional — Bearer token
      TTS_SERVICE_TIMEOUT  optional — per-request timeout in seconds (default 60)
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        base_url: str   = TTS_SERVICE_URL,
        api_key:  str   = TTS_SERVICE_API_KEY,
        timeout:  float = TTS_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("TTS_SERVICE_URL must be set to use RemoteTTSClient.")

        self._base_url = base_url.rstrip("/")
        self._timeout  = timeout
        self._inflight = 0

        # Separate breakers for batch vs stream — mirrors TTSNode's
        # _breaker_batch / _breaker_stream split so stream failures can
        # never trip the breaker for synchronous /run synthesis calls.
        self._breaker_batch  = CircuitBreaker(name="tts:remote:batch")
        self._breaker_stream = CircuitBreaker(name="tts:remote:stream")

        # Auth baked into client so it is never omitted per-call.
        # Per-call headers (trace, budget, request-id) are added in
        # _request_headers() and merged at the call site.
        base_headers: dict[str, str] = {}
        if api_key:
            base_headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=base_headers,
            http2=True,
        )

        # Local PCM pipeline for PCM-aware callers — constructed lazily on
        # first use so non-PCM deployments pay zero overhead.
        self._pcm_config:   PCMTTSOutputConfig | None = None
        self._pcm_pipeline: PCMTTSPipeline | None     = None
        self._pcm_lock = asyncio.Lock()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _request_headers(self, rid: str) -> dict[str, str]: # noqa
        """
        Build per-request headers.

        Injects:
          X-Request-Id          — propagated to remote for correlated logging
          X-Latency-Budget-Ms   — remaining SLA; remote uses this to self-abort
          OTel trace headers    — remote joins the same distributed trace
        """
        headers: dict[str, str] = {"X-Request-Id": rid}
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

    def _validate_input(self, text: str, speed: float) -> None: # noqa
        """
        Validate synthesize inputs before any network call.

        Mirrors TTSNode.synthesize() guard conditions so the remote client
        raises the same errors for the same invalid inputs — callers should
        not need to know which implementation they are talking to.
        """
        if not text or not text.strip():
            raise ValueError("Cannot synthesize empty text.")
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

    def _validate_synthesize_result( # noqa
        self,
        raw: dict[str, Any],
        rid: str,
    ) -> tuple[str, str, float]:
        """
        Coerce the remote JSON response into a (local_path, s3_uri) tuple.

        Fills defaults for omitted optional fields so voice_graph can always
        destructure the tuple without defensive guards at call sites.
        Warns (never raises) on unexpected shapes — the pipeline continues.
        """
        local_path = str(raw.get("local_path", ""))
        s3_uri     = str(raw.get("s3_uri", ""))

        if not local_path and not s3_uri:
            log.warning(
                "remote_tts_result_missing_paths",
                request_id=rid,
                keys=list(raw.keys()),
            )
        duration_s = _wav_duration_s(local_path) if local_path else 0.0
        return local_path, s3_uri, duration_s

    async def _ensure_pcm_pipeline(self, request_id: str) -> PCMTTSPipeline:
        """
        Lazily construct and cache the local PCMTTSPipeline.

        Thread-safe via asyncio.Lock. The same pipeline instance is reused
        for the lifetime of the client, consistent with TTSNode behaviour.
        Uses get_format_registry() so operators can override the "openai_tts"
        format centrally without touching client code.
        """
        async with self._pcm_lock:
            if self._pcm_pipeline is not None:
                return self._pcm_pipeline

            fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
            self._pcm_config = PCMTTSOutputConfig.for_format(
                fmt=fmt, request_id=request_id
            )
            self._pcm_pipeline = PCMTTSPipeline(
                config=self._pcm_config,
                gap_manager=PCMSentenceGapManager(fmt, gap_s=PCM_SENTENCE_GAP_S),
            )
            log.info(
                "remote_tts_pcm_pipeline_initialised",
                fmt=repr(fmt),
                request_id=request_id,
            )
            return self._pcm_pipeline

    # ── synthesize (batch) ────────────────────────────────────────────────────

    async def synthesize(
        self,
        text:       str,
        voice:      VoiceType | None = None,
        filename:   str | None = None,
        speed:      float = 1.0,
        request_id: str | None = None,
    ) -> tuple[str, str, float]:
        """
        POST /synthesize — returns (local_path, s3_uri).

        Flow:
          1. Input validation — empty text, speed range
          2. SLA budget check — abort immediately if already blown
          3. sanitize() the text before sending (consistent with TTSNode)
          4. POST with backoff_retry inside batch circuit breaker (3 attempts)
          5. Validate + coerce response into (local_path, s3_uri)
          6. Apology fallback if all retries fail
          7. Emit Prometheus counters and OTel span attributes

        Raises:
          ValueError             — empty text or speed out of range
          LatencyBudgetExceeded  — SLA blown before or during the call
          asyncio.CancelledError — task was cancelled; always re-raised
          httpx.HTTPStatusError  — 4xx/5xx from the remote service
          Exception              — other errors (apology fallback attempted first)
        """
        rid = request_id or new_request_id()

        self._validate_input(text, speed)

        try:
            self._check_budget("remote_tts.synthesize")
        except LatencyBudgetExceeded:
            _budget_exceeded.inc()
            log.warning("remote_tts_budget_exceeded_entry", request_id=rid)
            raise

        # Sanitize before network — mirrors TTSNode.synthesize() behaviour so
        # the remote service receives the same cleaned text regardless of path.
        san = sanitize(text, request_id=rid)
        if not san:
            raise ValueError("Sanitizer returned empty TTS text.")
        safe_text    = san.text
        active_voice: VoiceType = _resolve_voice(voice if voice is not None else TTS_VOICE, log_source="node.__call__")

        headers = self._request_headers(rid)

        with tracer.start_as_current_span("tts.remote.synthesize") as span:
            span.set_attribute("request_id",  rid)
            span.set_attribute("char_count",  len(safe_text))
            span.set_attribute("voice",       active_voice)
            span.set_attribute("provider",    "remote")

            _active.labels(mode="batch").inc()
            _circuit_open.labels(provider="remote").set(
                1 if self._breaker_batch.state == "OPEN" else 0
            )
            self._inflight += 1
            t0 = time.monotonic()

            try:
                async def _call() -> dict[str, Any]:
                    resp = await self._http.post(
                        "/synthesize",
                        json={
                            "text":     safe_text,
                            "voice":    active_voice,
                            "filename": filename,
                            "speed":    speed,
                        },
                        headers=headers,
                    )
                    resp.raise_for_status()
                    return resp.json()

                raw: dict[str, Any] = await self._breaker_batch.call(
                    backoff_retry,
                    _call,
                    attempts=3,
                    base_delay=1.0,
                    exceptions=_TTS_RETRYABLE,
                )

                local_path, s3_uri, duration_s = self._validate_synthesize_result(raw, rid)
                elapsed             = round(time.monotonic() - t0, 3)

                _req_total.labels(status="ok", mode="batch", provider="remote").inc()
                _latency.observe(elapsed)
                _chars_synthesized.observe(len(safe_text))
                _circuit_open.labels(provider="remote").set(
                    1 if self._breaker_batch.state == "OPEN" else 0
                )

                span.set_attribute("latency_s",   elapsed)
                span.set_attribute("local_path",  local_path)
                span.set_attribute("s3_uri",      s3_uri or "")

                log.info(
                    "remote_tts_synthesize_ok",
                    request_id  = rid,
                    voice       = active_voice,
                    chars       = len(safe_text),
                    latency_s   = elapsed,
                    local_path  = local_path,
                    s3_uri      = s3_uri or "n/a",
                )

                duration_s = _wav_duration_s(local_path) if local_path else 0.0
                return local_path, s3_uri, duration_s

            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                log.warning("remote_tts_synthesize_budget_exceeded", request_id=rid)
                raise

            except asyncio.CancelledError:
                log.warning("remote_tts_synthesize_cancelled", request_id=rid)
                raise

            except Exception as exc:
                _req_total.labels(
                    status="error", mode="batch", provider="remote"
                ).inc()
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_tts_synthesize_error", request_id=rid, error=str(exc))

                # Apology fallback — if a pre-rendered file is available, return
                # it so the pipeline still has something to play during an outage.
                fallback = _try_apology_fallback(rid)
                if fallback is not None:
                    return fallback
                raise

            finally:
                _active.labels(mode="batch").dec()
                self._inflight -= 1

    # ── synthesize_stream ─────────────────────────────────────────────────────

    async def synthesize_stream(
        self,
        token_stream: AsyncIterator[str],
        voice:        VoiceType | None = None,
        speed:        float = 1.0,
        request_id:   str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Yield audio bytes sentence-by-sentence from an LLM token stream.

        Architecture
        ────────────
        LLM tokens → AsyncSentenceBuffer → sanitize → POST /synthesize/stream
        (one request per sentence) → yield chunked audio bytes as they arrive.

        One HTTP request is opened per sentence so the first audio bytes
        are available as soon as the first sentence is synthesized — identical
        TTFB profile to TTSNode.synthesize_stream().

        Circuit breaker
        ───────────────
        The stream context is opened inside the circuit breaker.  This ensures
        repeated stream failures trip _breaker_stream while preserving true
        streaming semantics (no full buffering before yield).  The batch breaker
        (_breaker_batch) is unaffected by stream failures.

        Budget enforcement
        ──────────────────
        Checked at entry, before each sentence, and before each audio chunk
        yield.  Mid-stream budget exhaustion emits a warning with
        chunks_emitted for diagnosing where in the sentence sequence the
        SLA was blown.

        Cancellation / generator exit
        ──────────────────────────────
        GeneratorExit is caught at the yield site (caller dropped the iterator
        cleanly) and logged at DEBUG — not as an error.  CancelledError
        propagates up after updating counters.

        Raises:
          ValueError             — speed out of range (checked at entry)
          LatencyBudgetExceeded  — SLA blown at entry or mid-stream
          asyncio.CancelledError — task was cancelled; always re-raised
          httpx.HTTPStatusError  — non-2xx from remote
        """
        if not (0.25 <= speed <= 4.0):
            raise ValueError(f"Speed must be 0.25–4.0, got {speed}.")

        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_tts.stream")
        except LatencyBudgetExceeded:
            _budget_exceeded.inc()
            log.warning("remote_tts_stream_budget_exceeded_entry", request_id=rid)
            raise

        active_voice: VoiceType = _resolve_voice(voice if voice is not None else TTS_VOICE, log_source="node.__call__")
        headers      = self._request_headers(rid)

        with tracer.start_as_current_span("tts.remote.stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("voice",      active_voice)
            span.set_attribute("provider",   "remote")

            async with bulkheads.acquire("tts.stream"):
                _active.labels(mode="stream").inc()
                _circuit_open.labels(provider="remote").set(
                    1 if self._breaker_stream.state == "OPEN" else 0
                )
                self._inflight += 1
                t0                  = time.monotonic()
                total_chars         = 0
                sentence_count      = 0
                chunk_index         = 0
                first_bytes_emitted = False

                try:
                    sentence_stream = AsyncSentenceBuffer(token_stream)

                    async for sentence in sentence_stream:

                        # Per-sentence SLA check — abort before processing
                        try:
                            self._check_budget("remote_tts.stream.before_sentence")
                        except LatencyBudgetExceeded:
                            _budget_exceeded.inc()
                            log.warning(
                                "remote_tts_stream_budget_exceeded_mid",
                                request_id    = rid,
                                sentence      = sentence_count,
                                chunks_emitted = chunk_index,
                            )
                            raise

                        san = sanitize(sentence, request_id=rid)
                        if not san:
                            continue

                        clean        = san.text
                        total_chars += len(clean)
                        sentence_count += 1

                        # Open the HTTP stream context inside the circuit breaker
                        # so repeated stream failures trip _breaker_stream while
                        # preserving streaming semantics.
                        async def _open_stream() -> Any:
                            return self._http.stream(
                                "POST",
                                "/synthesize/stream",
                                json={
                                    "text":  clean,
                                    "voice": active_voice,
                                    "speed": speed,
                                },
                                headers=headers,
                            )

                        stream_ctx = await self._breaker_stream.call(
                            backoff_retry,
                            _open_stream,
                            attempts=3,
                            base_delay=1.0,
                            exceptions=_TTS_RETRYABLE,
                        )

                        async with stream_ctx as resp:
                            resp.raise_for_status()

                            async for audio_chunk in resp.aiter_bytes(chunk_size=4096):

                                # Per-chunk SLA check before yielding bytes
                                try:
                                    self._check_budget(
                                        "remote_tts.stream.before_yield"
                                    )
                                except LatencyBudgetExceeded:
                                    _budget_exceeded.inc()
                                    log.warning(
                                        "remote_tts_stream_budget_exceeded_yield",
                                        request_id    = rid,
                                        chunks_emitted = chunk_index,
                                    )
                                    raise

                                if not first_bytes_emitted:
                                    _ttfb.observe(time.monotonic() - t0)
                                    first_bytes_emitted = True

                                chunk_index += 1

                                try:
                                    yield audio_chunk
                                except (GeneratorExit, asyncio.CancelledError):
                                    log.debug(
                                        "remote_tts_stream_generator_exit",
                                        request_id    = rid,
                                        chunks_emitted = chunk_index,
                                    )
                                    return

                    elapsed = round(time.monotonic() - t0, 3)

                    _req_total.labels(
                        status="ok", mode="stream", provider="remote"
                    ).inc()
                    _latency.observe(elapsed)
                    _chars_synthesized.observe(total_chars)
                    _chunks_per_request.observe(chunk_index)
                    _circuit_open.labels(provider="remote").set(
                        1 if self._breaker_stream.state == "OPEN" else 0
                    )

                    span.set_attribute("latency_s",      elapsed)
                    span.set_attribute("sentences",      sentence_count)
                    span.set_attribute("chunks_yielded", chunk_index)
                    span.set_attribute("chars",          total_chars)

                    log.info(
                        "remote_tts_stream_ok",
                        request_id    = rid,
                        voice         = active_voice,
                        chars         = total_chars,
                        sentences     = sentence_count,
                        chunks_yielded = chunk_index,
                        latency_s     = elapsed,
                    )

                except LatencyBudgetExceeded:
                    _req_total.labels(
                        status="budget_exceeded", mode="stream", provider="remote"
                    ).inc()
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(
                        status="cancelled", mode="stream", provider="remote"
                    ).inc()
                    log.warning(
                        "remote_tts_stream_cancelled",
                        request_id    = rid,
                        chunks_emitted = chunk_index,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="stream", provider="remote"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error(
                        "remote_tts_stream_error",
                        request_id    = rid,
                        error         = str(exc),
                        chunks_emitted = chunk_index,
                    )
                    raise

                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight -= 1

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        """
        GET /health — returns ServiceHealthState.

        Merges the remote service's own health flag with the local circuit
        breaker state.  A remote that reports healthy=True but whose local
        batch breaker is OPEN is still considered unhealthy from this node's
        perspective — consistent with RemoteSTTClient.health() and with the
        local TTSNode.health() logic.

        5-second timeout prevents a dead health probe from blocking the voice
        graph health check cycle.  Always returns (never raises).
        """
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            is_remote_healthy = bool(data.get("healthy", False))
            _circuit_open.labels(provider="remote").set(
                1 if self._breaker_batch.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "tts.remote",
                healthy       = is_remote_healthy and self._breaker_batch.state != "OPEN",
                circuit_state = self._breaker_batch.state,
                inflight      = self._inflight,
            )
        except Exception as exc:
            _circuit_open.labels(provider="remote").set(
                1 if self._breaker_batch.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "tts.remote",
                healthy       = False,
                circuit_state = self._breaker_batch.state,
                inflight      = self._inflight,
                error         = str(exc),
            )

    # ── close ─────────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Drain and close the underlying httpx connection pool."""
        await self._http.aclose()


# ═══════════════════════════════════════════════════════════════════════════════
# 13. NODE FACTORY
# ═══════════════════════════════════════════════════════════════════════════════


def get_tts_node() -> TTSNodeProtocol:
    """
    Return RemoteTTSClient if TTS_SERVICE_URL is set, else local TTSNode.

    VoiceGraph imports only this factory — setting TTS_SERVICE_URL before
    module load switches the entire TTS stage to a remote service with zero
    graph-level code changes.  Mirrors get_llm_node() and get_stt_node().
    """
    if TTS_SERVICE_URL:
        log.info("tts_using_remote_client", url=TTS_SERVICE_URL)
        return cast(TTSNodeProtocol, RemoteTTSClient())
    log.info("tts_using_local_node", model=TTS_MODEL, voice=TTS_VOICE)
    return cast(TTSNodeProtocol, TTSNode())


# ── module-level singleton (backward-compatible) ──────────────────────────────

tts_node: TTSNodeProtocol = get_tts_node()


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.nodes.LLM_service import get_llm_node

    def _make_token_stream(text: str) -> AsyncIterator[str]:
        """
        Wrap a static string as an AsyncIterator[str] of single-word tokens.
        Simulates an LLM token stream without a live API call so the TTS
        streaming path can be exercised in isolation (CI / offline smoke).
        """
        async def _gen() -> AsyncIterator[str]:
            for word in text.split():
                yield word + " "
                await asyncio.sleep(0.002)
        return _gen()

    async def _smoke() -> None:
        tts = get_tts_node()
        llm = get_llm_node()

        # ── batch synthesize ──────────────────────────────────────────────────
        print("── batch synthesize ──")
        local, s3, duration_s = await tts.synthesize(
            "The voice module is working correctly.", voice="nova"
        )
        print(f"Local: {local} | S3: {s3 or '(none)'}")

        # ── streaming synthesize (bytes) via stream_messages ──────────────────
        print("\n── streaming synthesize (bytes) via stream_messages ──")
        llm_stream: AsyncIterator[str] = llm.stream_messages(
            [
                SystemMessage(content="You are a concise technical explainer."),
                HumanMessage(content="Explain binary search in two sentences."),
            ]
        )
        i = 0
        async for audio_bytes in tts.synthesize_stream(llm_stream):
            i += 1
            print(f"  chunk {i}: {len(audio_bytes)} bytes")

        # ── streaming synthesize (bytes) via static token stream (offline) ────
        print("\n── streaming synthesize (bytes) via static stream ──")
        static_stream = _make_token_stream(
            "The quick brown fox jumps over the lazy dog."
        )
        j = 0
        async for audio_bytes in tts.synthesize_stream(static_stream):
            j += 1
            print(f"  chunk {j}: {len(audio_bytes)} bytes")

        # ── PCM path — only available on local TTSNode ────────────────────────
        if isinstance(tts, TTSNode):

            print("\n── PCM synthesize (single chunk) ──")
            pcm = await tts.synthesize_pcm("Hello from PCM path.", voice="nova")
            print(f"  PCMChunk: {pcm}")
            wav = chunk_to_wav_bytes(pcm)
            print(f"  WAV bytes: {len(wav)}")

            print("\n── PCM stream via stream_messages ──")
            llm_stream_2: AsyncIterator[str] = llm.stream_messages(
                [
                    SystemMessage(content="You are a concise technical explainer."),
                    HumanMessage(
                        content="Describe how sound travels in two sentences."
                    ),
                ]
            )
            n_pcm = 0
            async for chunk in tts.synthesize_pcm_stream(llm_stream_2):
                n_pcm += 1
                print(
                    f"  pcm chunk {n_pcm}: frames={chunk.n_frames} "
                    f"dur={chunk.duration_s:.3f}s src={chunk.source!r}"
                )

            print("\n── PCM stream via static token stream (offline) ──")
            static_pcm_stream = _make_token_stream(
                "The quick brown fox jumps over the lazy dog."
            )
            n_static = 0
            async for chunk in tts.synthesize_pcm_stream(static_pcm_stream):
                n_static += 1
                print(
                    f"  static chunk {n_static}: frames={chunk.n_frames} "
                    f"src={chunk.source!r}"
                )

            print("\n── synthesize_pcm_to_wav ──")
            wav_bytes = await tts.synthesize_pcm_to_wav(
                "This is an in-memory WAV test.", voice="nova"
            )
            print(f"  WAV buffer: {len(wav_bytes)} bytes")

            print("\n── synthesize_stream_pcm_to_wav ──")
            collect_stream = _make_token_stream("Streaming audio collected to WAV.")
            collected_wav = await tts.synthesize_stream_pcm_to_wav(collect_stream)
            print(f"  Collected WAV: {len(collected_wav)} bytes")

            print("\n── PCM diagnostics ──")
            diag = await tts.get_pcm_diagnostics()
            for k, v in diag.items():
                print(f"  {k}: {v}")

        print("\n── health ──")
        print(await tts.health())

        await tts.close()
        await llm.close()

    asyncio.run(_smoke())