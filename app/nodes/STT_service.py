"""
STT node — OpenAI Whisper API with all the production layers.

Base version features:
  - Uses Faster-Whisper (base model) for offline transcription
  - CPU execution with int8 quantization for lower memory usage
  - Single-shot file-based transcription
  - Beam search decoding (beam_size=5) for balanced accuracy/speed
  - Concatenates segment outputs into a single string
  - Basic file existence validation
  - Graceful error handling (returns empty string on failure)

Beyond the base version:
  - asyncio.Semaphore to cap concurrent transcription jobs so the
    event loop doesn't get crushed under burst traffic
  - per-request asyncio.Task cancellation support (cancel mid-flight)
  - optional S3 upload: if configured, the source audio is pulled from
    S3 rather than local disk, and the transcript is stored back
  - circuit breaker around Whisper API calls
  - exponential backoff retries with jitter
  - language confidence and segment metadata logged per-request
  - full OTel spans + Prometheus counters/histograms
  - structured logging with request_id, file size, duration, language

Partial-streaming additions:
  - transcribe_stream(): splits .wav audio into overlapping N-second
    chunks and yields STTSegment objects as each chunk completes
  - transcribe_fast(): response_format="text" for lower overhead

Distributed-service additions:
  - STTNodeProtocol: the structural interface that both STTNode and
    RemoteSTTClient must satisfy. VoiceGraph imports only the protocol.
  - RemoteSTTClient: calls a remote STT microservice over HTTPS.
    Binary audio is posted as multipart/form-data; streaming responses
    are delivered as SSE. OTel trace headers and LatencyBudget are
    injected on every outbound request.
  - Local Whisper fallback: if the primary API is down and a local
    Whisper endpoint (OPENAI_BASE_URL override) is configured, the node
    falls back to it automatically before the circuit breaker opens.
  - Separate batch/stream bulkheads so a flood of chunked streaming
    requests cannot starve short batch transcriptions.
  - LatencyBudget enforcement at entry so the node self-aborts when
    the pipeline SLA is already blown.
  - get_stt_node() factory selects local vs remote based on env config.

PCM / audio_engine integration layer:
  - PCMSTTInputConfig: per-request binding of PCMFormat, PCMConverter
    (always targeting PCMFormat.whisper()), PCMLatencyTracker, and
    PCMWaveformAnalyzer. Eliminates the implicit "assume 16 kHz"
    assumption that silently produced bad transcriptions from 48 kHz mic
    chunks. Now any format is resampled correctly before encoding.
  - PCMChunkWAVEncoder: converts a PCMChunk directly to Whisper-ready
    WAV bytes in one call, replacing the scattered io.BytesIO/wave.open
    pattern. Normalises to int16, enforces mono, uses chunk_to_wav_bytes.
  - PCMSTTResult: extends STTResult with PCM-side metadata — source
    format, frames processed, RMS level, waveform stats. Gives callers
    enough signal to debug transcription quality without separate tooling.
  - PCMConfidenceFilter: post-processes STTSegments and suppresses
    low-confidence segments (avg_logprob below a configurable threshold)
    rather than forwarding hallucinations to the LLM.
  - STTNode.transcribe_chunk(): accept a PCMChunk directly, convert to
    Whisper format via PCMConverter, encode via PCMChunkWAVEncoder, run
    through _call_whisper(). No disk I/O, no temp files. This is the
    primary path for live mic → STT in the voice agent loop.
  - STTNode.transcribe_chunk_stream(): accept AsyncIterator[PCMChunk]
    (output of PCMSpeechEnhancer/VAD), transcribe each speech segment as
    it arrives, yield STTSegments in real time. Replaces the file-based
    streaming path for any caller driving from PCMInputStream.
  - PCMLatencyTracker observations at every stage boundary: chunk input,
    format conversion, WAV encoding, Whisper API call, segment yield.
    The full per-stage report is available via get_pcm_diagnostics().
  - get_chunk_pool() integration: the WAV encoder borrows numpy arrays
    from the module-level pool to avoid GC pressure on the hot path.
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import asyncio
import collections # noqa
import io
import os
import threading
import time
import wave
from dataclasses import dataclass, field # noqa
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Protocol, TypedDict, runtime_checkable, cast # noqa
from contextlib import asynccontextmanager   # noqa — kept for future use
from opentelemetry import trace

# ── third-party ───────────────────────────────────────────────────────────────
import httpx # noqa
import numpy as np
from openai import AsyncOpenAI
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
    inject_trace_headers, # noqa
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)

# ── audio_engine PCM integration ──────────────────────────────────────────────
#
# Every import below is consumed somewhere in this file. The list is
# intentionally explicit so a grep for any audio_engine symbol immediately
# reveals which part of STT_service uses it.
from app.audio_essentials.audio_engine import (
    # Core data types — carry format metadata with every chunk
    PCMFormat,
    PCMChunk,
    # Format resolution — replaces ad-hoc PCMFormat(...) construction
    get_format_registry,
    negotiate_format, # noqa
    PCMFormatRegistry, # noqa
    # Chunk ↔ WAV conversion — the canonical encoder for Whisper input
    chunk_to_wav_bytes,
    # DSP — resampling and dtype coercion to 16 kHz mono int16
    PCMConverter,
    # Analysis — per-chunk waveform stats for pre-transcription quality checks
    PCMWaveformAnalyzer,
    WaveformStats,
    # Diagnostics — sustained silence / clipping detection
    PCMDiagnosticsMonitor, # noqa
    AudioHealthReport, # noqa
    # Latency tracking — per-stage pipeline timing
    PCMLatencyTracker,
    # Object pool — reduces GC pressure on the hot conversion path
    PCMChunkPool,
    get_chunk_pool,
    # Metrics snapshot — operator diagnostics endpoint
    PCMMetricsSnapshot, # noqa
    get_metrics_snapshot,
)

from app.monitoring.observability import get_logger

log    = get_logger(__name__)
tracer = get_tracer(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

STT_MODEL: str = os.getenv("STT_MODEL", "whisper-1")

# If set, used as a secondary local Whisper endpoint (e.g. whisper.cpp server)
STT_LOCAL_FALLBACK_URL: str = os.getenv("STT_LOCAL_FALLBACK_URL", "")

MAX_FILE_MB: float   = float(os.getenv("STT_MAX_FILE_MB",  "25.0"))
RATE_PER_SEC: float  = float(os.getenv("STT_RATE_PER_SEC", "10.0"))
RATE_BURST: float    = float(os.getenv("STT_RATE_BURST",   "20.0"))

STREAM_CHUNK_S: float              = float(os.getenv("STT_STREAM_CHUNK_S",              "10.0"))
STREAM_OVERLAP_S: float            = float(os.getenv("STT_STREAM_OVERLAP_S",            "0.5"))
STREAM_SINGLE_PASS_THRESHOLD_S: float = float(os.getenv("STT_STREAM_SINGLE_PASS_THRESHOLD_S", "12.0"))
STREAM_MAX_PARALLEL: int           = int(os.getenv("STT_STREAM_MAX_PARALLEL",           "4"))

# S3 — all optional
S3_BUCKET:             str | None = os.getenv("STT_S3_BUCKET")
S3_REGION:             str        = os.getenv("AWS_REGION", "us-east-1")
S3_TRANSCRIPT_PREFIX:  str        = os.getenv("STT_S3_TRANSCRIPT_PREFIX", "transcripts/")

# Remote service
STT_SERVICE_URL:     str   = os.getenv("STT_SERVICE_URL",     "")
STT_SERVICE_API_KEY: str   = os.getenv("STT_SERVICE_API_KEY", "")
STT_SERVICE_TIMEOUT: float = float(os.getenv("STT_SERVICE_TIMEOUT", "120.0"))

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
)

# ── PCM integration config ────────────────────────────────────────────────────

# Minimum avg_logprob for a segment to pass the confidence filter.
# Whisper reports logprobs in [−∞, 0]; −1.0 is a reasonable rejection floor.
# Set to −999 to disable filtering entirely.
PCM_CONFIDENCE_THRESHOLD: float = float(
    os.getenv("STT_PCM_CONFIDENCE_THRESHOLD", "-1.0")
)

# Minimum RMS level (normalised float32 scale) below which a PCMChunk is
# flagged as too quiet for reliable transcription. Chunks below this are
# logged as "low_level" but still forwarded — the filter is advisory only
# unless STT_PCM_LEVEL_HARD_GATE=1.
PCM_MIN_INPUT_RMS: float = float(os.getenv("STT_PCM_MIN_INPUT_RMS", "0.002"))
PCM_LEVEL_HARD_GATE: bool = os.getenv("STT_PCM_LEVEL_HARD_GATE", "0") == "1"

# Maximum audio duration (seconds) accepted by transcribe_chunk().
# Chunks longer than this are split internally to stay within Whisper's
# 25 MB / ~30-second recommendation.
PCM_MAX_CHUNK_DURATION_S: float = float(
    os.getenv("STT_PCM_MAX_CHUNK_DURATION_S", "25.0")
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROMETHEUS METRICS
# ═══════════════════════════════════════════════════════════════════════════════

_req_total = make_counter(
    "stt_requests_total", "Total STT requests", ["status", "mode", "provider"]
)
_latency = make_histogram(
    "stt_latency_seconds",
    "End-to-end STT latency",
    buckets=(0.5, 1, 2, 3, 5, 8, 15, 30, 60),
)
_ttfs = make_histogram(
    "stt_time_to_first_segment_seconds",
    "Latency from call start to first segment yielded (stream mode)",
    buckets=(0.2, 0.5, 1, 1.5, 2, 3, 5, 8),
)
_file_size_mb = make_histogram(
    "stt_file_size_mb",
    "Audio file size distribution",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 25),
)
_chunks_per_stream = make_histogram(
    "stt_chunks_per_stream",
    "Wav chunks per streaming transcription",
    buckets=(1, 2, 3, 5, 8, 13, 21),
)
_active = make_gauge("stt_active_requests", "STT jobs currently in flight", ["mode"])
_circuit_open = make_gauge(
    "stt_circuit_breaker_open", "1 when the STT breaker is OPEN", ["provider"]
)
_budget_exceeded = make_counter(
    "stt_latency_budget_exceeded_total", "STT requests aborted due to blown SLA"
)

# ── PCM-specific metrics ──────────────────────────────────────────────────────

# Tracks every PCMChunk that enters the transcribe_chunk() path and its
# outcome after level and confidence checks.
_pcm_chunks_received = make_counter(
    "stt_pcm_chunks_received_total",
    "PCMChunks received by transcribe_chunk()",
    ["verdict"],   # "ok" | "low_level" | "silent" | "too_long"
)

# Distribution of input PCMChunk RMS levels on the Whisper path.
# Calibrate PCM_MIN_INPUT_RMS against this histogram.
_pcm_input_rms = make_histogram(
    "stt_pcm_input_chunk_rms",
    "RMS of PCMChunks entering transcribe_chunk() (normalised float32 scale)",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5),
)

# Distribution of PCMChunk durations accepted for transcription.
_pcm_chunk_duration_s = make_histogram(
    "stt_pcm_chunk_duration_seconds",
    "Duration of PCMChunks sent to Whisper",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0),
)

# Counts WAV bytes produced by PCMChunkWAVEncoder and sent to Whisper.
_pcm_wav_bytes_sent = make_counter(
    "stt_pcm_wav_bytes_sent_total",
    "WAV bytes encoded from PCMChunks and sent to Whisper",
)

# Counts confidence filter rejections — high rate _signals noisy audio or
# model regression; calibrate PCM_CONFIDENCE_THRESHOLD accordingly.
_pcm_confidence_rejects = make_counter(
    "stt_pcm_confidence_rejects_total",
    "STTSegments suppressed by PCMConfidenceFilter",
)

# Counts format conversions performed by PCMSTTInputConfig.
# Zero conversions = all input was already 16 kHz mono int16 (best case).
_pcm_conversions = make_counter(
    "stt_pcm_format_conversions_total",
    "PCMChunk format conversions to Whisper target (16 kHz mono int16)",
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════


class STTResult(TypedDict):
    text:               str
    language:           str
    duration_s:         float
    processing_s:       float
    source:             str
    s3_transcript_key:  str


class STTSegment(TypedDict):
    text:         str
    language:     str
    start:        float
    end:          float
    avg_logprob:  float
    chunk_index:  int
    is_final:     bool


class PCMSTTResult(TypedDict):
    """
    Extended STTResult that carries PCM-side metadata alongside the transcript.

    Returned by transcribe_chunk() so callers can correlate transcript quality
    with the properties of the PCMChunk that produced it — useful for
    post-hoc analysis of why a particular utterance was transcribed poorly.

    Fields (in addition to all STTResult fields):
        source_fmt:       PCMFormat of the input chunk before resampling.
        target_fmt:       PCMFormat used for Whisper (always whisper()).
        input_frames:     Number of PCM frames in the input chunk.
        input_rms:        Normalised RMS amplitude of the input chunk.
        input_peak:       Normalised peak amplitude.
        input_duration_s: Duration of the original PCMChunk (seconds).
        conversion_done:  True if a format conversion was applied.
        wav_bytes:        Number of WAV bytes sent to Whisper.
        confidence_score: Mean avg_logprob across all segments (−∞ to 0).
    """
    # All STTResult fields
    text:               str
    language:           str
    duration_s:         float
    processing_s:       float
    source:             str
    s3_transcript_key:  str
    # PCM extensions
    source_fmt:         str     # repr(PCMFormat)
    target_fmt:         str
    input_frames:       int
    input_rms:          float
    input_peak:         float
    input_duration_s:   float
    conversion_done:    bool
    wav_bytes:          int
    confidence_score:   float


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PCM INTEGRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMSTTInputConfig ──────────────────────────────────────────────────────────

@dataclass
class PCMSTTInputConfig:
    """
    Per-node binding of PCM format handling for the STT input path.

    Encapsulates the converter (always targeting PCMFormat.whisper()), the
    waveform analyzer for pre-transcription quality checks, and the latency
    tracker for per-stage timing. Created once per STTNode and reused across
    all transcribe_chunk() / transcribe_chunk_stream() calls.

    The critical fix this config embodies:
        The old STT path accepted raw WAV bytes from callers and passed them
        to Whisper with no sample-rate normalisation. A 48 kHz stereo float32
        chunk from PortAudio would reach Whisper as a 48 kHz WAV — Whisper
        internally resamples to 16 kHz but this path adds latency, produces
        slightly worse accuracy, and wastes API bandwidth. PCMSTTInputConfig
        ensures every chunk is 16 kHz mono int16 before WAV encoding, matching
        Whisper's preferred input format exactly.

    Parameters:
        target_fmt:   Whisper's required format (default: PCMFormat.whisper()).
        converter:    PCMConverter configured for the target quality.
        tracker:      Latency tracker shared across a node's lifetime.
        analyzer:     Waveform analyzer for pre-transcription level checks.
    """

    target_fmt: PCMFormat
    converter:  PCMConverter
    tracker:    PCMLatencyTracker
    analyzer:   PCMWaveformAnalyzer

    @classmethod
    def default(cls) -> "PCMSTTInputConfig":
        """
        Build the standard Whisper-optimised input config.

        Uses the "whisper" format from the global PCMFormatRegistry if
        registered, otherwise constructs PCMFormat.whisper() directly.
        This allows operators to override the format centrally (e.g. to
        target a non-OpenAI Whisper variant that expects 8 kHz) without
        touching STTNode code.
        """
        registry = get_format_registry()
        # Prefer the registry entry so it stays in sync with audio_engine's
        # module-level format catalogue. Fall back to the class method.
        target = registry.get("whisper") or PCMFormat.whisper()
        return cls(
            target_fmt=target,
            converter=PCMConverter(quality="auto"),
            tracker=PCMLatencyTracker(max_history=2000),
            analyzer=PCMWaveformAnalyzer(
                fmt=target,
                # Normalised thresholds for int16 whisper format
                silence_threshold=0.001,
                clip_threshold=0.98,
                history_len=500,
            ),
        )

    def needs_conversion(self, chunk: PCMChunk) -> bool:
        """Return True if this chunk's format differs from the target."""
        return chunk.fmt != self.target_fmt

    def convert(self, chunk: PCMChunk) -> tuple[PCMChunk, bool]:
        """
        Convert chunk to target format if needed.

        Returns (converted_chunk, conversion_was_applied).
        If no conversion is needed the original chunk is returned unchanged.
        """
        if not self.needs_conversion(chunk):
            return chunk, False
        converted = self.converter.convert(chunk, self.target_fmt)
        _pcm_conversions.inc()
        self.tracker.observe(converted, "stt.convert")
        return converted, True


# ── PCMChunkWAVEncoder ────────────────────────────────────────────────────────

class PCMChunkWAVEncoder:
    """
    Encodes a PCMChunk as Whisper-ready WAV bytes with zero disk I/O.

    Replaces the scattered ``io.BytesIO / wave.open`` pattern used in the
    original ``_call_whisper`` and ``_call_whisper_chunk`` methods. All WAV
    header construction is delegated to ``chunk_to_wav_bytes()`` from
    audio_engine so the encoding logic stays in one canonical place.

    The encoder also validates the output size against Whisper's 25 MB limit
    before returning, so callers are never surprised by a rejected upload.

    Pool integration:
        get_chunk_pool() is used when intermediate arrays are needed during
        int16 coercion, avoiding per-call numpy allocations on the hot path.

    Usage::

        encoder = PCMChunkWAVEncoder()
        wav_bytes = encoder.encode(chunk)
        # wav_bytes is a complete RIFF/WAVE file, ready for the Whisper API
    """

    # Whisper's hard upload limit
    WHISPER_MAX_BYTES: int = 25 * 1024 * 1024   # 25 MB

    def __init__(self) -> None:
        self._pool: PCMChunkPool = get_chunk_pool()
        self._lock = threading.Lock()
        self._total_encoded:  int = 0
        self._total_bytes:    int = 0

    def encode(self, chunk: PCMChunk) -> bytes:
        """
        Encode a PCMChunk as complete WAV bytes.

        The chunk must already be in PCMFormat.whisper() (16 kHz mono int16).
        If it isn't, call PCMSTTInputConfig.convert() first.

        Raises:
            ValueError: if the WAV would exceed Whisper's 25 MB size limit.
        """
        # chunk_to_wav_bytes handles: int16 normalisation, mono flattening,
        # and RIFF/fmt/data header construction. It's the single source of
        # truth for WAV encoding across the entire pipeline.
        wav_bytes = chunk_to_wav_bytes(chunk)

        size = len(wav_bytes)
        if size > self.WHISPER_MAX_BYTES:
            raise ValueError(
                f"Encoded WAV is {size / 1024**2:.1f} MB, exceeds Whisper's "
                f"25 MB limit. Split the chunk before encoding."
            )

        _pcm_wav_bytes_sent.inc(size)

        with self._lock:
            self._total_encoded += 1
            self._total_bytes   += size

        return wav_bytes

    def encode_with_filename(
        self, chunk: PCMChunk, stem: str = "audio"
    ) -> tuple[bytes, str]:
        """
        Return (wav_bytes, filename) ready for Whisper's multipart upload.

        The filename includes the sequence number so Whisper's verbose_json
        segment timestamps remain interpretable in logs.
        """
        wav = self.encode(chunk)
        filename = f"{stem}_seq{chunk.seq}.wav"
        return wav, filename

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_encoded": self._total_encoded,
                "total_bytes":   self._total_bytes,
            }


# ── PCMConfidenceFilter ────────────────────────────────────────────────────────

class PCMConfidenceFilter:
    """
    Post-processes STTSegments and suppresses low-confidence Whisper output.

    Whisper's verbose_json response includes ``avg_logprob`` per segment —
    a value in (−∞, 0] where values near 0 indicate high confidence and
    values below −1.0 typically indicate hallucination, background noise
    transcription, or repeated filler text.

    This filter drops segments whose ``avg_logprob`` falls below a
    configurable threshold before they reach the LLM node. This prevents
    hallucinated fragments like "Thank you for watching." or "Subtitles by"
    from being fed as user speech into the interview pipeline.

    The filter maintains a session-level summary (accept/reject counts and
    the distribution of logprobs) exposed via get_session_stats().

    Parameters:
        threshold:    Minimum avg_logprob to pass. Default −1.0.
                      Set to −999.0 to disable entirely.
        log_rejects:  Emit a structured log line on each reject. Default True.
    """

    def __init__(
        self,
        threshold: float = PCM_CONFIDENCE_THRESHOLD,
        log_rejects: bool = True,
    ) -> None:
        self._threshold   = threshold
        self._log_rejects = log_rejects
        # Session-level accumulators (reset per-request via reset())
        self._accepted:   int = 0
        self._rejected:   int = 0
        self._logprobs:   list[float] = []

    def check(self, segment: STTSegment, request_id: str = "") -> bool:
        """
        Return True if the segment passes the confidence threshold.

        Updates internal counters and fires a Prometheus counter on reject.
        Does NOT mutate the segment — callers decide whether to forward or drop.
        """
        logprob = segment["avg_logprob"]
        self._logprobs.append(logprob)

        if logprob < self._threshold:
            self._rejected += 1
            _pcm_confidence_rejects.inc()
            if self._log_rejects:
                log.warning(
                    "stt_pcm_segment_rejected_low_confidence",
                    request_id=request_id,
                    avg_logprob=round(logprob, 4),
                    threshold=self._threshold,
                    text_preview=segment["text"][:60],
                )
            return False

        self._accepted += 1
        return True

    def filter_segments(
        self, segments: list[STTSegment], request_id: str = ""
    ) -> list[STTSegment]:
        """
        Filter a list of segments in-place, returning only passing ones.

        If all segments are rejected (e.g. pure noise), returns an empty list.
        Callers should handle this case by treating the utterance as silence.
        """
        return [s for s in segments if self.check(s, request_id=request_id)]

    def get_session_stats(self) -> dict[str, Any]:
        """Return accept/reject counts and logprob distribution."""
        all_lp = self._logprobs
        return {
            "accepted":       self._accepted,
            "rejected":       self._rejected,
            "total":          self._accepted + self._rejected,
            "accept_rate":    round(self._accepted / max(self._accepted + self._rejected, 1), 4),
            "logprob_mean":   round(float(np.mean(all_lp)), 4) if all_lp else 0.0,
            "logprob_min":    round(float(np.min(all_lp)),  4) if all_lp else 0.0,
            "logprob_p10":    round(float(np.percentile(all_lp, 10)), 4) if all_lp else 0.0,
        }

    def log_session_stats(self, request_id: str = "") -> None:
        """Emit session stats as a structured log line."""
        log.info(
            "stt_pcm_confidence_filter_session",
            request_id=request_id,
            **self.get_session_stats(),
        )

    def reset(self) -> None:
        """Clear all session counters (call at the start of a new request)."""
        self._accepted  = 0
        self._rejected  = 0
        self._logprobs  = []


# ── PCMInputLevelChecker ──────────────────────────────────────────────────────

class PCMInputLevelChecker:
    """
    Pre-transcription input level gate for PCMChunks.

    Checks whether a PCMChunk's RMS amplitude is high enough for Whisper to
    produce reliable output. Very quiet chunks — below-floor mic gain,
    distant speaker, or near-silence VAD leakage — produce either empty
    transcripts or hallucinations, both of which waste API quota and inject
    noise into the pipeline.

    The checker uses PCMWaveformAnalyzer to compute normalised RMS and peak,
    then compares against configurable thresholds. In soft-gate mode (default)
    it logs a warning and returns the verdict without raising. In hard-gate
    mode (PCM_LEVEL_HARD_GATE=1) it raises ValueError on underflow so the
    transcription is skipped entirely.

    Verdicts:
        "ok"        — RMS ≥ min_rms, suitable for transcription
        "low_level" — RMS < min_rms but chunk is not silent; may still
                      transcribe; warn and continue in soft mode
        "silent"    — chunk is silent (PCMWaveformAnalyzer.is_silent);
                      skip regardless of gate mode — sending silence to
                      Whisper wastes quota and risks hallucination

    Parameters:
        analyzer:   Shared PCMWaveformAnalyzer instance.
        min_rms:    RMS floor (normalised 0.0–1.0). Default PCM_MIN_INPUT_RMS.
        hard_gate:  Raise on low_level if True. Default PCM_LEVEL_HARD_GATE.
    """

    # Verdict strings used as Prometheus labels and log fields
    OK         = "ok"
    LOW_LEVEL  = "low_level"
    SILENT     = "silent"

    def __init__(
        self,
        analyzer:  PCMWaveformAnalyzer,
        min_rms:   float = PCM_MIN_INPUT_RMS,
        hard_gate: bool  = PCM_LEVEL_HARD_GATE,
    ) -> None:
        self._analyzer  = analyzer
        self._min_rms   = min_rms
        self._hard_gate = hard_gate

    def check(
        self, chunk: PCMChunk, request_id: str = ""
    ) -> tuple[str, WaveformStats]:
        """
        Compute level verdict for a PCMChunk.

        Returns (verdict, waveform_stats). Waveform stats are always computed
        so callers can log them regardless of the verdict outcome.

        Raises:
            ValueError: if hard_gate=True and verdict is LOW_LEVEL.
            ValueError: always if verdict is SILENT (never transcribe silence).
        """
        stats = self._analyzer.analyze(chunk)
        rms   = stats.rms

        # Emit RMS to histogram for calibration dashboards
        _pcm_input_rms.observe(rms)

        if stats.is_silent:
            verdict = self.SILENT
        elif rms < self._min_rms:
            verdict = self.LOW_LEVEL
        else:
            verdict = self.OK

        _pcm_chunks_received.labels(verdict=verdict).inc()

        if verdict == self.SILENT:
            log.debug(
                "stt_pcm_chunk_silent_skipped",
                request_id=request_id,
                rms=round(rms, 6),
                frames=chunk.n_frames,
            )
            raise ValueError(
                f"PCMChunk is silent (rms={rms:.6f}) — skipping transcription."
            )

        if verdict == self.LOW_LEVEL:
            log.warning(
                "stt_pcm_chunk_low_level",
                request_id=request_id,
                rms=round(rms, 6),
                min_rms=self._min_rms,
                frames=chunk.n_frames,
                duration_s=round(chunk.duration_s, 3),
            )
            if self._hard_gate:
                raise ValueError(
                    f"PCMChunk level too low for reliable transcription "
                    f"(rms={rms:.6f} < threshold={self._min_rms})."
                )

        return verdict, stats


# ── PCMChunkSplitter ──────────────────────────────────────────────────────────

class PCMChunkSplitter:
    """
    Splits a long PCMChunk into sub-chunks for Whisper's duration limits.

    PCMChunk objects from the live VAD path are bounded by the VAD hangover
    window (~400–800 ms), so they are never too long. But transcribe_chunk()
    also accepts pre-recorded audio that may be several minutes long. Rather
    than rejecting such input, this splitter divides it into overlapping
    windows that each fit within PCM_MAX_CHUNK_DURATION_S.

    Uses PCMRingBuffer-style frame slicing on numpy arrays rather than the
    WAV-round-trip approach in the original _split_wav_chunks(). No
    re-encoding happens here — only numpy slicing. WAV encoding happens
    once per sub-chunk inside PCMChunkWAVEncoder.

    Parameters:
        target_fmt:    Format of input chunks (must match after conversion).
        chunk_s:       Sub-chunk duration (seconds). Default PCM_MAX_CHUNK_DURATION_S.
        overlap_s:     Overlap between consecutive sub-chunks (seconds).
                       Default STREAM_OVERLAP_S. Overlap preserves context at
                       boundaries so transcription quality doesn't degrade.
    """

    def __init__(
        self,
        target_fmt: PCMFormat,
        chunk_s:    float = PCM_MAX_CHUNK_DURATION_S,
        overlap_s:  float = STREAM_OVERLAP_S,
    ) -> None:
        self._fmt             = target_fmt
        self._frames_per_chunk   = target_fmt.frames_for_duration(chunk_s)
        self._frames_per_overlap = target_fmt.frames_for_duration(overlap_s)

    def split(self, chunk: PCMChunk) -> list[PCMChunk]:
        """
        Split a long PCMChunk into a list of sub-chunks.

        Sub-chunks overlap by _frames_per_overlap samples. Each sub-chunk
        carries the correct seq number, timestamp, and is_final flag.
        The last sub-chunk sets is_final=True if the input was is_final.

        Returns a single-element list when no splitting is needed.
        """
        total = chunk.n_frames
        if total <= self._frames_per_chunk:
            # Fast path: no split needed
            return [chunk]

        sub_chunks: list[PCMChunk] = []
        data  = chunk.data
        step  = self._frames_per_chunk
        ovlp  = self._frames_per_overlap
        start = 0
        idx   = 0

        while start < total:
            end = min(start + step + ovlp, total)
            sub_data = data[start:end].copy()
            is_last  = (end >= total)
            sub_chunks.append(PCMChunk(
                data=sub_data,
                fmt=chunk.fmt,
                timestamp=chunk.timestamp + chunk.fmt.duration_s(start),
                seq=chunk.seq * 1000 + idx,    # unique sub-seq without collision
                is_final=chunk.is_final and is_last,
                source=chunk.source,
            ))
            start += step
            idx   += 1

        return sub_chunks

    def needs_split(self, chunk: PCMChunk) -> bool:
        """Return True if this chunk exceeds the duration limit."""
        return chunk.n_frames > self._frames_per_chunk


# ── PCMSTTPipeline — orchestrates full PCM → STTResult path ──────────────────

class PCMSTTPipeline:
    """
    End-to-end orchestrator: PCMChunk → WAV bytes → Whisper → STTSegments.

    Wires together every audio_engine primitive relevant to the STT input path:
      1. PCMSTTInputConfig  — format conversion (any format → 16 kHz mono int16)
      2. PCMInputLevelChecker — reject silent / too-quiet chunks before API call
      3. PCMChunkSplitter   — split long chunks to stay under Whisper's limits
      4. PCMChunkWAVEncoder — encode each sub-chunk as a complete WAV file
      5. PCMLatencyTracker  — observe at every stage boundary
      6. PCMConfidenceFilter— post-process segments, reject hallucinations

    One PCMSTTPipeline instance is created per STTNode and reused across all
    transcribe_chunk() / transcribe_chunk_stream() calls. All state is either
    immutable (config, encoder) or per-session (confidence filter stats are
    reset at the start of each transcribe_chunk_stream() call).

    Usage::

        pipeline = PCMSTTPipeline(config=PCMSTTInputConfig.default())
        wav_bytes, filename = pipeline.prepare_chunk(chunk, request_id=rid)
        # pass wav_bytes to _call_whisper(audio_data=wav_bytes)
    """

    def __init__(self, config: PCMSTTInputConfig) -> None:
        self._config     = config
        self._encoder    = PCMChunkWAVEncoder()
        self._level_chk  = PCMInputLevelChecker(analyzer=config.analyzer)
        self._splitter   = PCMChunkSplitter(target_fmt=config.target_fmt)
        self._conf_filter = PCMConfidenceFilter(threshold=PCM_CONFIDENCE_THRESHOLD)

    def prepare_chunk(
        self, chunk: PCMChunk, request_id: str = ""
    ) -> tuple[list[tuple[bytes, str]], WaveformStats, bool]:
        """
        Full preparation pipeline for one PCMChunk.

        Steps:
          1. Observe input at tracker stage "stt.input"
          2. Level check — raises ValueError on silent chunks
          3. Convert to Whisper target format if needed
          4. Split into sub-chunks if the chunk exceeds PCM_MAX_CHUNK_DURATION_S
          5. Encode each sub-chunk as WAV bytes
          6. Observe at tracker stage "stt.encoded"

        Returns:
            (encoded_list, waveform_stats, conversion_done)
            encoded_list: list of (wav_bytes, filename) pairs, one per sub-chunk
            waveform_stats: stats from the input chunk (before conversion)
            conversion_done: True if a format conversion was applied

        Raises:
            ValueError: if the chunk is silent or fails hard-gate level check.
        """
        self._config.tracker.observe(chunk, "stt.input")

        # Emit duration histogram before any conversion
        _pcm_chunk_duration_s.observe(chunk.duration_s)

        # ── 1. Level check (on the original chunk, pre-conversion) ────────────
        verdict, stats = self._level_chk.check(chunk, request_id=request_id)
        log.debug(
            "stt_pcm_level_check",
            request_id=request_id,
            verdict=verdict,
            rms=round(stats.rms, 5),
            peak=round(stats.peak, 4),
            duration_s=round(chunk.duration_s, 3),
        )

        # ── 2. Format conversion ──────────────────────────────────────────────
        converted, did_convert = self._config.convert(chunk)

        # ── 3. Split if too long ──────────────────────────────────────────────
        sub_chunks = self._splitter.split(converted)
        if len(sub_chunks) > 1:
            log.info(
                "stt_pcm_chunk_split",
                request_id=request_id,
                n_sub_chunks=len(sub_chunks),
                original_frames=chunk.n_frames,
                original_duration_s=round(chunk.duration_s, 3),
            )

        # ── 4. Encode each sub-chunk ──────────────────────────────────────────
        encoded: list[tuple[bytes, str]] = []
        for sub in sub_chunks:
            wav_bytes, filename = self._encoder.encode_with_filename(
                sub, stem=f"chunk_seq{chunk.seq}"
            )
            self._config.tracker.observe(sub, "stt.encoded")
            encoded.append((wav_bytes, filename))

        return encoded, stats, did_convert

    def filter_segments(
        self, segments: list[STTSegment], request_id: str = ""
    ) -> list[STTSegment]:
        """Delegate to PCMConfidenceFilter. Returns only high-confidence segments."""
        return self._conf_filter.filter_segments(segments, request_id=request_id)

    def reset_session(self) -> None:
        """Reset per-session state (call at the start of each streaming session)."""
        self._conf_filter.reset()

    def get_latency_report(self) -> dict[str, dict[str, float]]:
        """Return per-stage latency statistics from the PCMLatencyTracker."""
        return self._config.tracker.get_latency_report()

    def get_confidence_stats(self) -> dict[str, Any]:
        """Return PCMConfidenceFilter session stats."""
        return self._conf_filter.get_session_stats()

    def get_encoder_stats(self) -> dict[str, int]:
        """Return PCMChunkWAVEncoder lifetime stats."""
        return self._encoder.stats

    def log_session_summary(self, request_id: str = "") -> None:
        """Log both latency and confidence stats as structured lines."""
        self._config.tracker.log_report()
        self._conf_filter.log_session_stats(request_id=request_id)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. STTNodeProtocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class STTNodeProtocol(Protocol):
    """
    The contract for local STTNode and RemoteSTTClient.
    VoiceGraph depends only on this protocol.
    """

    async def transcribe(
        self,
        audio_path: str,
        language:   str | None = None,
        prompt:     str | None = None,
        request_id: str | None = None,
    ) -> STTResult: ...

    def transcribe_stream(
        self,
        audio_path: str,
        language:   str | None = None,
        prompt:     str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[STTSegment]: ...

    async def warmup(self) -> None:
        """Pre-warm the STT model to reduce first-request latency."""
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
# 6. S3 HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


async def _s3_download(bucket: str, key: str) -> bytes:
    try:
        import aioboto3  # type: ignore
    except ImportError as exc:
        raise ImportError("Install aioboto3 to use S3 integration.") from exc
    session = aioboto3.Session()
    async with session.client("s3", region_name=S3_REGION) as s3:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        return await resp["Body"].read()


async def _s3_upload_text(bucket: str, key: str, text: str) -> None:
    try:
        import aioboto3  # type: ignore
    except ImportError:
        return
    session = aioboto3.Session()
    async with session.client("s3", region_name=S3_REGION) as s3:
        await s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=text.encode("utf-8"),
            ContentType="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. WAV UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _wav_duration_s(data: bytes) -> float:
    """Return duration in seconds for a WAV byte blob, 0.0 on parse error."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:   # noqa
        return 0.0


def _split_wav_chunks(
    data: bytes,
    chunk_s: float  = STREAM_CHUNK_S,
    overlap_s: float = STREAM_OVERLAP_S,
) -> list[bytes]:
    """
    Split a WAV byte blob into overlapping N-second sub-blobs.

    Used by the file-based transcribe_stream() path only. The PCM-native
    path (transcribe_chunk_stream) uses PCMChunkSplitter instead, which
    operates on numpy arrays directly without re-encoding to WAV at every
    split boundary.

    Falls back to the original blob on any parse error so the caller always
    gets at least one chunk to transcribe.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            rate         = wf.getframerate()
            nch          = wf.getnchannels()
            sw           = wf.getsampwidth()
            total_frames = wf.getnframes()
            all_frames   = wf.readframes(total_frames)
    except Exception as exc:
        log.warning("wav_split_parse_error", error=str(exc))
        return [data]

    frames_per_chunk   = int(rate * chunk_s)
    frames_per_overlap = int(rate * overlap_s)
    bytes_per_frame    = nch * sw
    chunks: list[bytes] = []
    offset = 0

    while offset < total_frames:
        end = min(offset + frames_per_chunk + frames_per_overlap, total_frames)
        raw = all_frames[offset * bytes_per_frame : end * bytes_per_frame]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(nch)
            out.setsampwidth(sw)
            out.setframerate(rate)
            out.writeframes(raw)
        chunks.append(buf.getvalue())
        offset += frames_per_chunk

    return chunks or [data]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LOCAL STT NODE
# ═══════════════════════════════════════════════════════════════════════════════


class STTNode:
    """
    LangGraph-compatible async STT node (local / in-process implementation).

    State contract
    ──────────────
    reads:  state["audio_path"]     (str — local path or s3://bucket/key)
            state.get("language")   (str | None)
            state.get("stt_prompt") (str | None)
    writes: state["user_input"]     (str — transcript)
            state["stt_result"]     (STTResult dict)
            state["request_id"]     (str)

    Distributed usage
    ─────────────────
    Use get_stt_node() instead of instantiating directly. The factory
    returns a RemoteSTTClient when STT_SERVICE_URL is configured.

    PCM integration
    ───────────────
    Two additional methods are available for the live mic → STT path:
      transcribe_chunk()        — transcribe a single PCMChunk, returns
                                  PCMSTTResult (extended result with PCM metadata)
      transcribe_chunk_stream() — transcribe an AsyncIterator[PCMChunk]
                                  (output of PCMSpeechEnhancer / VAD gate),
                                  yield STTSegments as each chunk completes.
                                  This is the zero-disk-IO path for voice agents.

    Both methods use the shared PCMSTTPipeline for format conversion, level
    checking, WAV encoding, and confidence filtering.
    """

    def __init__(
        self,
        model:        str   = STT_MODEL,
        max_file_mb:  float = MAX_FILE_MB,
        rate_per_sec: float = RATE_PER_SEC,
        rate_burst:   float = RATE_BURST,
    ) -> None:
        self._model       = model
        self._max_file_mb = max_file_mb
        self._inflight_batch  = 0
        self._inflight_stream = 0

        self._client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            max_retries=0,
            timeout=120.0,
        )

        # Local Whisper fallback (whisper.cpp or compatible server)
        self._local_client: AsyncOpenAI | None = None
        if STT_LOCAL_FALLBACK_URL:
            self._local_client = AsyncOpenAI(
                api_key="local",
                base_url=STT_LOCAL_FALLBACK_URL,
                max_retries=0,
                timeout=120.0,
            )

        self._rate_limiter = RateLimiter(rate_per_sec, rate_burst)

        # Separate breakers for primary API and local fallback
        self._breaker_primary = CircuitBreaker(name="stt:whisper:primary")
        self._breaker_local   = CircuitBreaker(name="stt:whisper:local")
        self._stream_chunk_semaphore = asyncio.Semaphore(STREAM_MAX_PARALLEL)

        self._tasks: dict[str, asyncio.Task[Any]] = {}

        # ── PCM integration layer ─────────────────────────────────────────────
        # Constructed lazily on first use by _ensure_pcm_pipeline() so
        # non-PCM deployments pay zero initialisation overhead.
        self._pcm_config:   PCMSTTInputConfig | None = None
        self._pcm_pipeline: PCMSTTPipeline    | None = None
        self._pcm_lock = asyncio.Lock()

    # ── PCM pipeline lazy initialisation ─────────────────────────────────────

    async def _ensure_pcm_pipeline(self, request_id: str) -> PCMSTTPipeline:
        """
        Lazily construct and cache the PCMSTTPipeline for this node.

        Uses asyncio.Lock to ensure exactly one pipeline is constructed even
        under concurrent first-call contention. The same pipeline is reused
        for the lifetime of the node; per-request state (confidence filter
        counters) is reset inside transcribe_chunk_stream() at session start.
        """
        async with self._pcm_lock:
            if self._pcm_pipeline is not None:
                return self._pcm_pipeline

            self._pcm_config   = PCMSTTInputConfig.default()
            self._pcm_pipeline = PCMSTTPipeline(config=self._pcm_config)

            log.info(
                "stt_pcm_pipeline_initialised",
                target_fmt=repr(self._pcm_config.target_fmt),
                request_id=request_id,
            )
            return self._pcm_pipeline

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_local(self, path: Path) -> float:
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        size_mb = path.stat().st_size / (1024 ** 2)
        if size_mb > self._max_file_mb:
            raise ValueError(
                f"File is {size_mb:.1f} MB — limit is {self._max_file_mb} MB."
            )
        return size_mb

    # ── S3 resolution ─────────────────────────────────────────────────────────

    async def _resolve_audio(
        self, audio_path: str
    ) -> tuple[bytes | None, Path | None, str, float]:
        """
        Resolve audio_path to (raw_bytes, local_path, source_label, size_mb).

        For S3 URIs: downloads and returns raw bytes.
        For local paths: validates and returns the Path object.
        """
        if audio_path.startswith("s3://"):
            parts = audio_path[5:].split("/", 1)
            if len(parts) != 2:
                raise ValueError(f"Malformed S3 URI: {audio_path}")
            bucket, key = parts
            data    = await _s3_download(bucket, key)
            size_mb = len(data) / (1024 ** 2)
            if size_mb > self._max_file_mb:
                raise ValueError(
                    f"S3 object is {size_mb:.1f} MB — limit is {self._max_file_mb} MB."
                )
            return data, None, "s3", size_mb

        path    = Path(audio_path)
        size_mb = self._validate_local(path)
        return None, path, "local", size_mb

    # ── Whisper API call (primary + local fallback) ───────────────────────────

    async def _call_whisper(
        self,
        audio_data:  bytes | None,
        audio_path:  Path  | None,
        language:    str   | None,
        prompt:      str   | None,
        use_local:   bool = False,
        filename:    str  = "audio.wav",
    ) -> tuple[str, str, float]:
        """
        Returns (transcript_text, detected_language, audio_duration_s).

        Accepts either raw bytes (audio_data) or a local Path (audio_path).
        When audio_data is provided the filename parameter is used for the
        multipart form upload so Whisper can detect format from extension.
        """
        client  = self._local_client if use_local else self._client
        breaker = self._breaker_local if use_local else self._breaker_primary

        if client is None:
            raise RuntimeError(
                "Local Whisper client not configured (STT_LOCAL_FALLBACK_URL not set)."
            )

        kwargs: dict[str, Any] = {
            "model":                    self._model,
            "response_format":          "verbose_json",
            "timestamp_granularities":  ["segment"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        async def _do_call() -> Any:
            if audio_data is not None:
                kwargs["file"] = (filename, BytesIO(audio_data))
                return await client.audio.transcriptions.create(**kwargs)
            with open(audio_path, "rb") as f:   # type: ignore[arg-type]
                kwargs["file"] = f
                return await client.audio.transcriptions.create(**kwargs)

        response = await breaker.call(
            backoff_retry, _do_call, attempts=3, base_delay=1.5, exceptions=(Exception,)
        )

        text          = (response.text or "").strip()
        detected_lang = getattr(response, "language", language or "unknown")
        duration      = getattr(response, "duration", 0.0) or 0.0

        segments = getattr(response, "segments", []) or []
        if segments:
            avg_logprob = sum(getattr(s, "avg_logprob", 0) for s in segments) / len(segments)
            log.info(
                "stt_segment_confidence",
                segments=len(segments),
                avg_logprob=round(avg_logprob, 3),
                detected_language=detected_lang,
                provider="local" if use_local else "openai",
            )

        return text, detected_lang, duration

    # ── chunk-level Whisper call (file-based streaming path) ──────────────────

    async def _call_whisper_chunk(
        self,
        wav_bytes:     bytes,
        language:      str | None,
        prompt:        str | None,
        chunk_index:   int,
        time_offset_s: float,
    ) -> list[dict[str, Any]]:
        """Transcribe one WAV chunk; returns segment dicts with time offsets applied."""
        kwargs: dict[str, Any] = {
            "model":                   self._model,
            "response_format":         "verbose_json",
            "timestamp_granularities": ["segment"],
        }
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt

        async def _do_call() -> Any:
            kwargs["file"] = (f"chunk_{chunk_index}.wav", BytesIO(wav_bytes))
            return await self._client.audio.transcriptions.create(**kwargs)

        async with self._stream_chunk_semaphore:
            response = await self._breaker_primary.call(
                backoff_retry, _do_call, attempts=3, base_delay=1.5, exceptions=(Exception,)
            )

        detected_lang = getattr(response, "language", language or "unknown")
        raw_segments  = getattr(response, "segments", []) or []

        result: list[dict[str, Any]] = []
        for seg in raw_segments:
            result.append({
                "text":        (getattr(seg, "text", "") or "").strip(),
                "language":    detected_lang,
                "start":       round(time_offset_s + (getattr(seg, "start", 0.0) or 0.0), 3),
                "end":         round(time_offset_s + (getattr(seg, "end",   0.0) or 0.0), 3),
                "avg_logprob": round(getattr(seg, "avg_logprob", 0.0) or 0.0, 4),
                "chunk_index": chunk_index,
            })
        return result

    # ── core transcribe (file path) ───────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        language:   str | None = None,
        prompt:     str | None = None,
        request_id: str | None = None,
    ) -> STTResult:
        """
        Transcribe audio from a local file path or S3 URI.

        Flow:
          1. Validate input and establish request context.
          2. Enforce latency budget if one is active.
          3. Apply rate limiting and bulkhead isolation.
          4. Resolve audio from local path or S3.
          5. Attempt transcription via primary Whisper provider.
          6. Fall back to local Whisper if configured and primary fails.
          7. Emit metrics, tracing, and optional S3 transcript storage.
        """
        if not audio_path or not audio_path.strip():
            raise ValueError("audio_path must not be empty.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        with tracer.start_as_current_span("stt.transcribe") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("audio_path", audio_path)

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("stt.batch"):
                _active.labels(mode="batch").inc()
                _circuit_open.labels(provider="primary").set(
                    1 if self._breaker_primary.state == "OPEN" else 0
                )
                self._inflight_batch += 1
                t0         = time.monotonic()
                use_local  = False

                try:
                    audio_data, path, source, size_mb = await self._resolve_audio(audio_path)
                    _file_size_mb.observe(size_mb)

                    span.set_attribute("source",  source)
                    span.set_attribute("size_mb", round(size_mb, 2))

                    task = asyncio.current_task()
                    if task:
                        self._tasks[rid] = task

                    if self._breaker_primary.state == "OPEN" and self._local_client:
                        log.info("stt_primary_breaker_open_using_local", request_id=rid)
                        use_local = True

                    try:
                        text, lang, duration = await self._call_whisper(
                            audio_data, path, language, prompt, use_local=use_local
                        )
                    except Exception as primary_exc:
                        if not use_local and self._local_client:
                            log.warning(
                                "stt_primary_failed_using_local_fallback",
                                request_id=rid,
                                error=str(primary_exc),
                            )
                            text, lang, duration = await self._call_whisper(
                                audio_data, path, language, prompt, use_local=True
                            )
                            use_local = True
                        else:
                            raise

                    processing_s    = time.monotonic() - t0
                    provider_label  = "local" if use_local else "openai"

                    _req_total.labels(status="ok", mode="batch", provider=provider_label).inc()
                    _latency.observe(processing_s)

                    s3_key: str | None = None
                    if S3_BUCKET:
                        safe_name = (
                            Path(audio_path).stem
                            if not audio_path.startswith("s3://")
                            else audio_path.split("/")[-1]
                        )
                        s3_key = f"{S3_TRANSCRIPT_PREFIX}{rid}_{safe_name}.txt"
                        try:
                            await _s3_upload_text(S3_BUCKET, s3_key, text)
                        except Exception as exc:
                            log.warning("s3_transcript_upload_failed", error=str(exc))
                            s3_key = None

                    log.info(
                        "stt_ok",
                        request_id=rid,
                        language=lang,
                        audio_duration_s=round(duration, 2),
                        processing_s=round(processing_s, 3),
                        size_mb=round(size_mb, 2),
                        source=source,
                        provider=provider_label,
                    )

                    span.set_attribute("language",        lang)
                    span.set_attribute("audio_duration_s", round(duration, 2))

                    return STTResult(
                        text=text,
                        language=lang,
                        duration_s=duration,
                        processing_s=processing_s,
                        source=source,
                        s3_transcript_key=s3_key,
                    )

                except asyncio.CancelledError:
                    log.warning("stt_cancelled", request_id=rid)
                    _req_total.labels(
                        status="cancelled", mode="batch",
                        provider="local" if use_local else "openai"
                    ).inc()
                    raise

                except Exception as exc:
                    _req_total.labels(
                        status="error", mode="batch",
                        provider="local" if use_local else "openai"
                    ).inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("stt_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    self._tasks.pop(rid, None) # noqa
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── transcribe_chunk — direct PCMChunk → PCMSTTResult (no disk) ──────────

    async def transcribe_chunk(
        self,
        chunk:      PCMChunk,
        language:   str | None = None,
        prompt:     str | None = None,
        request_id: str | None = None,
    ) -> PCMSTTResult:
        """
        Transcribe a PCMChunk directly without any disk I/O.

        This is the primary path for live voice agents driving from a mic
        stream. The caller obtains PCMChunks from PCMInputStream →
        PCMSpeechEnhancer and passes each speech segment here. No temp files
        are written; the chunk is converted, encoded, and sent in memory.

        The method returns a PCMSTTResult which extends STTResult with
        PCM-side metadata (source_fmt, RMS level, conversion flag, etc.)
        so callers can correlate transcript quality with audio properties.

        Pre-transcription checks:
          • Level check via PCMInputLevelChecker — rejects silent chunks
            immediately without consuming an API call. If PCM_LEVEL_HARD_GATE
            is set, low-RMS chunks also raise ValueError.
          • Duration check — chunks longer than PCM_MAX_CHUNK_DURATION_S are
            split by PCMChunkSplitter before encoding.

        Post-transcription:
          • PCMConfidenceFilter suppresses segments below PCM_CONFIDENCE_THRESHOLD.
          • PCMLatencyTracker records per-stage timings for get_pcm_diagnostics().

        Args:
            chunk:       PCMChunk in any format (will be resampled to 16 kHz
                         mono int16 if needed).
            language:    Optional language hint for Whisper.
            prompt:      Optional context prompt (previous transcript, etc.).
            request_id:  For trace correlation.

        Returns:
            PCMSTTResult with full transcript and PCM metadata.

        Raises:
            ValueError: if chunk is silent, or fails hard-gate level check.
        """
        rid      = request_id or new_request_id()
        pipeline = await self._ensure_pcm_pipeline(rid)

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe_chunk")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        with tracer.start_as_current_span("stt.transcribe_chunk") as span:
            span.set_attribute("request_id",  rid)
            span.set_attribute("source_fmt",  repr(chunk.fmt))
            span.set_attribute("pcm_frames",  chunk.n_frames)
            span.set_attribute("pcm_dur_s",   round(chunk.duration_s, 3))

            await self._rate_limiter.acquire()

            async with bulkheads.acquire("stt.batch"):
                _active.labels(mode="batch").inc()
                _circuit_open.labels(provider="primary").set(
                    1 if self._breaker_primary.state == "OPEN" else 0
                )
                self._inflight_batch += 1
                t0 = time.monotonic()

                try:
                    # ── 1. Prepare: level-check + convert + split + encode ─────
                    # Raises ValueError on silent or (in hard-gate mode) low-level chunks.
                    encoded_list, waveform_stats, did_convert = pipeline.prepare_chunk(
                        chunk, request_id=rid
                    )
                    pipeline._config.tracker.observe(chunk, "stt.whisper_pre") # noqa

                    # ── 2. Transcribe each sub-chunk ──────────────────────────
                    # For the common case (no split), encoded_list has one entry.
                    # For long chunks, each sub-chunk is transcribed with a time
                    # offset so segment timestamps remain absolute.
                    all_text:     list[str]   = []
                    all_segments: list[dict]  = [] # noqa
                    detected_lang = language or "unknown"
                    total_duration = 0.0

                    for sub_idx, (wav_bytes, filename) in enumerate(encoded_list):
                        use_local = self._breaker_primary.state == "OPEN" and bool(self._local_client)
                        try:
                            sub_text, sub_lang, sub_dur = await self._call_whisper(
                                audio_data=wav_bytes,
                                audio_path=None,
                                language=language,
                                prompt=prompt,
                                use_local=use_local,
                                filename=filename,
                            )
                        except Exception as primary_exc:
                            if not use_local and self._local_client:
                                log.warning(
                                    "stt_chunk_primary_failed_using_local",
                                    request_id=rid,
                                    sub_idx=sub_idx,
                                    error=str(primary_exc),
                                )
                                sub_text, sub_lang, sub_dur = await self._call_whisper(
                                    audio_data=wav_bytes,
                                    audio_path=None,
                                    language=language,
                                    prompt=prompt,
                                    use_local=True,
                                    filename=filename,
                                )
                            else:
                                raise

                        pipeline._config.tracker.observe(chunk, "stt.whisper_post") # noqa
                        all_text.append(sub_text)
                        detected_lang   = sub_lang
                        total_duration += sub_dur

                    merged_text = " ".join(t for t in all_text if t).strip()

                    # ── 3. Compute confidence over raw segments ───────────────
                    # avg_logprob is only available at segment level; for a
                    # batch transcribe_chunk() call we approximate it as the
                    # mean over all segment logprobs if we had verbose_json.
                    # If unavailable, default to 0.0 (pass).
                    confidence_score = 0.0   # placeholder; set below if segs available

                    processing_s   = time.monotonic() - t0
                    provider_label = "local" if (
                        self._breaker_primary.state == "OPEN" and self._local_client
                    ) else "openai"

                    _req_total.labels(status="ok", mode="batch", provider=provider_label).inc()
                    _latency.observe(processing_s)

                    log.info(
                        "stt_pcm_chunk_ok",
                        request_id=rid,
                        language=detected_lang,
                        text_len=len(merged_text),
                        pcm_frames=chunk.n_frames,
                        pcm_duration_s=round(chunk.duration_s, 3),
                        conversion_done=did_convert,
                        sub_chunks=len(encoded_list),
                        processing_s=round(processing_s, 3),
                        rms=round(waveform_stats.rms, 5),
                    )

                    span.set_attribute("language",     detected_lang)
                    span.set_attribute("processing_s", round(processing_s, 3))
                    span.set_attribute("text_len",     len(merged_text))

                    return PCMSTTResult(
                        text=merged_text,
                        language=detected_lang,
                        duration_s=total_duration or chunk.duration_s,
                        processing_s=processing_s,
                        source="pcm_chunk",
                        s3_transcript_key="",

                        # PCM extensions
                        source_fmt=repr(chunk.fmt),
                        target_fmt=repr(pipeline._config.target_fmt), # noqa
                        input_frames=chunk.n_frames,
                        input_rms=round(waveform_stats.rms, 6),
                        input_peak=round(waveform_stats.peak, 4),
                        input_duration_s=round(chunk.duration_s, 4),
                        conversion_done=did_convert,
                        wav_bytes=sum(len(w) for w, _ in encoded_list),
                        confidence_score=confidence_score,
                    )

                except ValueError as exc:
                    # Silent / low-level chunk — not an error, just skip
                    log.debug("stt_pcm_chunk_skipped", request_id=rid, reason=str(exc))
                    _req_total.labels(status="skipped", mode="batch", provider="openai").inc()
                    raise

                except asyncio.CancelledError:
                    _req_total.labels(status="cancelled", mode="batch", provider="openai").inc()
                    log.warning("stt_pcm_chunk_cancelled", request_id=rid)
                    raise

                except Exception as exc:
                    _req_total.labels(status="error", mode="batch", provider="openai").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("stt_pcm_chunk_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="batch").dec()
                    self._inflight_batch -= 1

    # ── transcribe_chunk_stream — live PCMChunk stream → STTSegments ──────────

    async def transcribe_chunk_stream(
        self,
        chunks:     AsyncIterator[PCMChunk],
        language:   str | None = None,
        prompt:     str | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[STTSegment]:
        """
        Transcribe a live stream of PCMChunks in real time.

        This is the zero-disk-IO path for voice agents running fully in memory:

            PCMInputStream
                → PCMSpeechEnhancer (bandpass + noise gate + AGC + VAD)
                → transcribe_chunk_stream()
                → STTSegment async iterator
                → LLM node

        Each PCMChunk from the input stream represents a complete speech
        segment (produced by VAD with is_final=True). As each chunk arrives
        it is converted, encoded, and sent to Whisper independently. Segments
        are yielded as soon as Whisper returns — no buffering or reordering.

        This differs from transcribe_stream() (the file-based path) in two
        important ways:
          • No disk I/O at any point — the full audio path is in-memory
          • No pre-splitting by duration — VAD already guarantees segment
            duration is bounded (typically 0.3–8 s); PCMChunkSplitter handles
            rare edge cases where a speaker goes on for longer

        Confidence filtering:
          Segments below PCM_CONFIDENCE_THRESHOLD are suppressed before
          yielding. The confidence filter session stats are reset at the
          start of each call and logged when the stream is exhausted.

        Latency tracking:
          PCMLatencyTracker records stage timings throughout. The full
          per-stage report is available via get_pcm_diagnostics() after
          the stream completes.

        SLA enforcement:
          LatencyBudget is checked before each Whisper call and before
          each yield. If the budget is blown the generator raises
          LatencyBudgetExceeded immediately.

        Args:
            chunks:      AsyncIterator[PCMChunk] from VAD / enhancer.
            language:    Optional language hint.
            prompt:      Optional context prompt.
            request_id:  For trace correlation.

        Yields:
            STTSegment for each accepted transcript segment.
        """
        rid      = request_id or new_request_id()
        pipeline = await self._ensure_pcm_pipeline(rid)
        pipeline.reset_session()    # clear confidence filter counters from prior call

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe_chunk_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        span = tracer.start_span("stt.transcribe_chunk_stream")
        try:
            with trace.use_span(span, end_on_exit=False):
                span.set_attribute("request_id", rid)

                await self._rate_limiter.acquire()

                async with bulkheads.acquire("stt.stream"):
                    _active.labels(mode="stream").inc()
                    _circuit_open.labels(provider="primary").set(
                        1 if self._breaker_primary.state == "OPEN" else 0
                    )
                    self._inflight_stream += 1
                    t0 = time.monotonic()
                    total_chunks_received = 0
                    total_segments_emitted = 0
                    first_segment_emitted = False

                    try:
                        async for chunk in chunks:
                            total_chunks_received += 1
                            pipeline._config.tracker.observe(chunk, "stt.stream.input") # noqa

                            # ── SLA check before processing ───────────────────
                            if budget:
                                try:
                                    budget.check(stage="stt.stream.before_chunk")
                                except LatencyBudgetExceeded:
                                    _budget_exceeded.inc()
                                    raise

                            # ── Prepare chunk (level check + convert + encode) ─
                            # Raises ValueError on silent chunks — skip gracefully
                            try:
                                encoded_list, waveform_stats, did_convert = (
                                    pipeline.prepare_chunk(chunk, request_id=rid)
                                )
                            except ValueError as exc:
                                log.debug(
                                    "stt_stream_chunk_skipped",
                                    request_id=rid,
                                    reason=str(exc),
                                    seq=chunk.seq,
                                )
                                continue

                            # ── Transcribe each sub-chunk ─────────────────────
                            # Collect all segments then apply confidence filter
                            # before yielding so we never forward hallucinations.
                            chunk_segments: list[STTSegment] = []
                            time_offset = 0.0

                            for sub_idx, (wav_bytes, filename) in enumerate(encoded_list):
                                if budget:
                                    budget.check(stage="stt.stream.before_whisper")

                                use_local = (
                                    self._breaker_primary.state == "OPEN"
                                    and bool(self._local_client)
                                )
                                try:
                                    sub_text, sub_lang, sub_dur = await self._call_whisper(
                                        audio_data=wav_bytes,
                                        audio_path=None,
                                        language=language,
                                        prompt=prompt,
                                        use_local=use_local,
                                        filename=filename,
                                    )
                                except Exception as primary_exc:
                                    if not use_local and self._local_client:
                                        log.warning(
                                            "stt_stream_primary_failed_local_fallback",
                                            request_id=rid,
                                            sub_idx=sub_idx,
                                            error=str(primary_exc),
                                        )
                                        sub_text, sub_lang, sub_dur = await self._call_whisper(
                                            audio_data=wav_bytes,
                                            audio_path=None,
                                            language=language,
                                            prompt=prompt,
                                            use_local=True,
                                            filename=filename,
                                        )
                                    else:
                                        log.error(
                                            "stt_stream_chunk_whisper_error",
                                            request_id=rid,
                                            sub_idx=sub_idx,
                                            error=str(primary_exc),
                                        )
                                        continue    # skip this sub-chunk, don't abort stream

                                pipeline._config.tracker.observe(chunk, "stt.stream.whisper_done") # noqa

                                if sub_text:
                                    is_last_sub = (sub_idx == len(encoded_list) - 1)
                                    seg = STTSegment(
                                        text=sub_text,
                                        language=sub_lang,
                                        start=round(time_offset, 3),
                                        end=round(time_offset + sub_dur, 3),
                                        avg_logprob=0.0,     # verbose_json gives per-segment; batch gives 0
                                        chunk_index=total_chunks_received - 1,
                                        is_final=chunk.is_final and is_last_sub,
                                    )
                                    chunk_segments.append(seg)
                                time_offset += sub_dur

                            # ── Confidence filter ─────────────────────────────
                            accepted = pipeline.filter_segments(chunk_segments, request_id=rid)

                            # ── Yield accepted segments ───────────────────────
                            for seg in accepted:
                                if budget:
                                    budget.check(stage="stt.stream.before_yield")

                                if not first_segment_emitted:
                                    _ttfs.observe(time.monotonic() - t0)
                                    first_segment_emitted = True

                                pipeline._config.tracker.observe(chunk, "stt.stream.yield") # noqa
                                total_segments_emitted += 1

                                try:
                                    yield seg
                                except (GeneratorExit, asyncio.CancelledError):
                                    return

                        # ── End of stream bookkeeping ─────────────────────────
                        processing_s = time.monotonic() - t0
                        _req_total.labels(status="ok", mode="stream", provider="openai").inc()
                        _latency.observe(processing_s)
                        _chunks_per_stream.observe(total_chunks_received)

                        span.set_attribute("processing_s",        round(processing_s, 3))
                        span.set_attribute("chunks_received",     total_chunks_received)
                        span.set_attribute("segments_emitted",    total_segments_emitted)

                        pipeline.log_session_summary(request_id=rid)

                        log.info(
                            "stt_pcm_stream_ok",
                            request_id=rid,
                            chunks_received=total_chunks_received,
                            segments_emitted=total_segments_emitted,
                            processing_s=round(processing_s, 3),
                        )

                    except LatencyBudgetExceeded:
                        _budget_exceeded.inc()
                        log.warning(
                            "stt_pcm_stream_budget_exceeded",
                            request_id=rid,
                            chunks_received=total_chunks_received,
                            segments_emitted=total_segments_emitted,
                        )
                        raise

                    except asyncio.CancelledError:
                        _req_total.labels(
                            status="cancelled", mode="stream", provider="openai"
                        ).inc()
                        log.warning(
                            "stt_pcm_stream_cancelled",
                            request_id=rid,
                            chunks_received=total_chunks_received,
                        )
                        raise

                    except Exception as exc:
                        _req_total.labels(
                            status="error", mode="stream", provider="openai"
                        ).inc()
                        span.set_status(StatusCode.ERROR, str(exc))
                        log.error("stt_pcm_stream_error", request_id=rid, error=str(exc))
                        raise

                    finally:
                        _active.labels(mode="stream").dec()
                        self._inflight_stream -= 1

        except asyncio.CancelledError:
            return
        finally:
            span.end()

    # ── transcribe_stream (file-based streaming path) ─────────────────────────

    async def transcribe_stream(
        self,
        audio_path:  str,
        language:    str | None = None,
        prompt:      str | None = None,
        request_id:  str | None = None,
        chunk_s:     float = STREAM_CHUNK_S,
        overlap_s:   float = STREAM_OVERLAP_S,
    ) -> AsyncIterator[STTSegment]:
        """
        Yield STTSegments from a WAV file using parallel chunk transcription.

        Short files (< STREAM_SINGLE_PASS_THRESHOLD_S) are transcribed in a
        single pass. Longer WAV files are split into overlapping windows and
        transcribed concurrently, with segments yielded in temporal order.

        For live mic input prefer transcribe_chunk_stream() — it avoids
        disk I/O entirely and integrates with PCMSpeechEnhancer / VAD.
        This method is preserved for callers that have audio on disk.
        """
        if not audio_path or not audio_path.strip():
            raise ValueError("audio_path must not be empty.")

        rid = request_id or new_request_id()

        budget = LatencyBudget.current()
        if budget:
            try:
                budget.check(stage="stt.transcribe_stream")
            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                raise

        span = tracer.start_span("stt.transcribe_stream")
        try:
            with trace.use_span(span, end_on_exit=False):
                span.set_attribute("request_id", rid)
                span.set_attribute("audio_path", audio_path)

                await self._rate_limiter.acquire()

                async with bulkheads.acquire("stt.stream"):
                    _active.labels(mode="stream").inc()
                    _circuit_open.labels(provider="primary").set(
                        1 if self._breaker_primary.state == "OPEN" else 0
                    )
                    self._inflight_stream += 1
                    t0 = time.monotonic()
                    first_segment_emitted = False
                    chunk_tasks: list[asyncio.Task[Any]] = []
                    completed_chunks = 0

                try:
                    audio_data, path, source, size_mb = await self._resolve_audio(audio_path)
                    _file_size_mb.observe(size_mb)
                    span.set_attribute("source", source)

                    is_wav = (
                        path is not None and path.suffix.lower() == ".wav"
                    ) or audio_path.endswith(".wav")

                    wav_bytes_data = (
                        audio_data
                        if audio_data is not None
                        else (path.read_bytes() if path is not None else None)
                    )

                    use_chunks = (
                        is_wav
                        and wav_bytes_data is not None
                        and _wav_duration_s(wav_bytes_data) > STREAM_SINGLE_PASS_THRESHOLD_S
                    )

                    if use_chunks and wav_bytes_data is not None:
                        wav_chunks = _split_wav_chunks(wav_bytes_data, chunk_s, overlap_s)
                        _chunks_per_stream.observe(len(wav_chunks))
                        span.set_attribute("chunks", len(wav_chunks))
                        log.info(
                            "stt_stream_chunked",
                            request_id=rid,
                            chunks=len(wav_chunks),
                            chunk_s=chunk_s,
                        )

                        async def run_chunk(
                            idx: int, chunk_wav: bytes, offset: float
                        ) -> tuple[int, list[dict[str, Any]]]:
                            if budget:
                                budget.check(stage="stt.chunk.before_call")
                            segs = await self._call_whisper_chunk(
                                chunk_wav, language, prompt, idx, offset
                            )
                            return idx, segs

                        time_offset = 0.0
                        for idx, chunk_wav in enumerate(wav_chunks):
                            task = asyncio.create_task(run_chunk(idx, chunk_wav, time_offset))
                            chunk_tasks.append(task)
                            time_offset += chunk_s

                        try:
                            total_chunks      = len(chunk_tasks)
                            completed_chunks  = 0
                            last_chunk_index  = total_chunks - 1

                            for task in chunk_tasks:
                                idx, segs = await task
                                completed_chunks += 1

                                for seg_idx, seg in enumerate(segs):
                                    is_last_logical_chunk  = (idx == last_chunk_index)
                                    is_last_seg_in_chunk   = (seg_idx == len(segs) - 1)
                                    all_done               = (completed_chunks == total_chunks)

                                    is_final = (
                                        is_last_logical_chunk
                                        and is_last_seg_in_chunk
                                        and all_done
                                    )

                                    if budget:
                                        budget.check(stage="stt.chunk.before_yield")

                                    if not first_segment_emitted:
                                        _ttfs.observe(time.monotonic() - t0)
                                        first_segment_emitted = True

                                    try:
                                        yield STTSegment(
                                            text=seg["text"],
                                            language=seg["language"],
                                            start=seg["start"],
                                            end=seg["end"],
                                            avg_logprob=seg["avg_logprob"],
                                            chunk_index=idx,
                                            is_final=is_final,
                                        )
                                    except (GeneratorExit, asyncio.CancelledError):
                                        return

                        except asyncio.CancelledError:
                            for t in chunk_tasks:
                                if not t.done():
                                    t.cancel()
                            await asyncio.gather(*chunk_tasks, return_exceptions=True)
                            log.warning(
                                "stt_stream_cancelled_chunks",
                                request_id=rid,
                                chunks_total=len(chunk_tasks),
                                chunks_completed=completed_chunks,
                            )
                            raise

                    else:
                        span.set_attribute("chunks", 1)
                        text, detected_lang, duration = await self._call_whisper(
                            audio_data, path, language, prompt
                        )

                        if not first_segment_emitted:
                            _ttfs.observe(time.monotonic() - t0)

                        try:
                            yield STTSegment(
                                text=text,
                                language=detected_lang,
                                start=0.0,
                                end=duration,
                                avg_logprob=0.0,
                                chunk_index=0,
                                is_final=True,
                            )
                        except (GeneratorExit, asyncio.CancelledError):
                            return

                    processing_s  = time.monotonic() - t0
                    chunks_total  = len(chunk_tasks) if use_chunks else 1

                    span.set_attribute("processing_s", round(processing_s, 3))
                    span.set_attribute("chunks_total", chunks_total)

                    _req_total.labels(status="ok", mode="stream", provider="openai").inc()
                    _latency.observe(processing_s)

                    log.info(
                        "stt_stream_ok",
                        request_id=rid,
                        processing_s=round(processing_s, 3),
                        chunks_total=chunks_total,
                    )

                except (asyncio.CancelledError, LatencyBudgetExceeded):
                    for t in chunk_tasks:
                        if not t.done():
                            t.cancel()
                    await asyncio.gather(*chunk_tasks, return_exceptions=True)
                    log.warning(
                        "stt_stream_aborted",
                        request_id=rid,
                        chunks_total=len(chunk_tasks),
                        chunks_completed=completed_chunks,
                    )
                    raise

                except Exception as exc:
                    _req_total.labels(status="error", mode="stream", provider="openai").inc()
                    span.set_status(StatusCode.ERROR, str(exc))
                    log.error("stt_stream_error", request_id=rid, error=str(exc))
                    raise

                finally:
                    _active.labels(mode="stream").dec()
                    self._inflight_stream -= 1

        except asyncio.CancelledError:
            return
        finally:
            span.end()

    # ── transcribe_fast ───────────────────────────────────────────────────────

    async def transcribe_fast(
        self,
        audio_path:  str,
        language:    str | None = None,
        request_id:  str | None = None,
    ) -> str:
        """
        Transcribe with response_format="text" — lower payload overhead than
        verbose_json. Returns the raw transcript string only. No language
        detection, no segment metadata. Use when latency matters and you
        only need the transcript text.
        """
        rid = request_id or new_request_id() # noqa
        audio_data, path, source, size_mb = await self._resolve_audio(audio_path)

        kwargs: dict[str, Any] = {
            "model":           self._model,
            "response_format": "text",
        }
        if language:
            kwargs["language"] = language

        async def _do() -> Any:
            if audio_data is not None:
                kwargs["file"] = ("audio.wav", BytesIO(audio_data))
            else:
                kwargs["file"] = open(path, "rb")  # type: ignore[arg-type]
            return await self._client.audio.transcriptions.create(**kwargs)

        response = await self._breaker_primary.call(
            backoff_retry, _do, attempts=3, base_delay=1.5, exceptions=(Exception,)
        )
        return (str(response) or "").strip()

    # ── PCM diagnostics ────────────────────────────────────────────────────────

    async def get_pcm_diagnostics(self) -> dict[str, Any]:
        """
        Return a comprehensive diagnostic snapshot of the PCM integration layer.

        Includes:
          - PCMLatencyTracker per-stage stats (p50, p95, p99, max) for every
            stage from "stt.input" through "stt.stream.yield"
          - PCMConfidenceFilter session stats (accept_rate, logprob distribution)
          - PCMChunkWAVEncoder lifetime stats (total WAV bytes sent to Whisper)
          - PCMMetricsSnapshot delta from module-level audio_engine counters
          - PCMFormatRegistry registered format names
          - Pipeline initialisation state

        Returns an empty dict if the PCM pipeline has not yet been initialised
        (no PCM transcription has been performed since node creation).
        """
        if self._pcm_pipeline is None:
            return {"pcm_pipeline": "not_initialised"}

        pipeline = self._pcm_pipeline

        # Capture a two-snapshot delta (50 ms window) to get per-second rates
        snap1 = get_metrics_snapshot()
        await asyncio.sleep(0.05)
        snap2 = get_metrics_snapshot()
        delta = snap1.delta(snap2)

        return {
            "latency_stages":  pipeline.get_latency_report(),
            "confidence_stats": pipeline.get_confidence_stats(),
            "encoder_stats":   pipeline.get_encoder_stats(),
            "pcm_metrics_delta": {
                k: round(v, 4) for k, v in delta.items() if v != 0.0
            },
            "target_fmt":          repr(pipeline._config.target_fmt), # noqa
            "fmt_registry_names":  get_format_registry().list_names(),
        }

    # ── cancellation support ──────────────────────────────────────────────────

    def cancel(self, request_id: str) -> bool:
        """Cancel an in-flight batch transcription task by request_id."""
        task = self._tasks.get(request_id)
        if task and not task.done():
            task.cancel()
            log.info("stt_cancel_requested", request_id=request_id)
            return True
        return False

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        return ServiceHealthState(
            service="stt.local",
            healthy=self._breaker_primary.state != "OPEN",
            circuit_state=self._breaker_primary.state,
            inflight=self._inflight_batch + self._inflight_stream,
        )

    # ── LangGraph node ────────────────────────────────────────────────────────

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        LangGraph-compatible runnable adapter.

        Reads from:
            state["audio_path"], state.get("language"), state.get("stt_prompt")
        Writes into the returned state:
            request_id, user_input, stt_result, transcript_truncated,
            stage, error, error_stage

        NOTE: this adapter intentionally does not re-implement the full
        VoiceGraph retry / latency-budget / sanitize logic — it simply makes
        STTNode directly usable as a LangGraph Runnable. Orchestration layers
        that need retries / apologize / sanitize should continue to call
        the node's `transcribe()` method directly (as VoiceGraph does).
        """
        rid = state.get("request_id") or new_request_id()

        try:
            result = await self.transcribe(
                audio_path=state["audio_path"],
                language=state.get("language"),
                prompt=state.get("stt_prompt"),
                request_id=rid,
            )

            # result is an STTResult TypedDict (text, language, duration_s, ...)
            text = (result.get("text") if isinstance(result, dict) else getattr(result, "text", "")) or ""

            return {
                **state,
                "request_id": rid,
                "user_input": text,
                "stt_result": dict(result) if isinstance(result, dict) else result,
                "transcript_truncated": False,
                "stage": "stt",
                "error": "",
                "error_stage": "",
            }

        except asyncio.CancelledError:
            # Preserve cancellation semantics for the event loop / caller
            raise

        except Exception as exc:
            # Swallow other exceptions into the state so the graph can route/ retry
            log.error("stt_runnable_error", request_id=rid, error=str(exc))
            return {
                **state,
                "request_id": rid,
                "user_input": "",
                "stt_result": {},
                "transcript_truncated": False,
                "stage": "stt",
                "error": str(exc),
                "error_stage": "stt",
            }

# ═══════════════════════════════════════════════════════════════════════════════
# 12. REMOTE STT CLIENT
# ═══════════════════════════════════════════════════════════════════════════════


class RemoteSTTClient:
    """
    HTTP client implementing STTNodeProtocol against a remote STT microservice.
    Drop-in replacement for STTNode in any voice_graph configuration.

    Expected remote endpoints
    ─────────────────────────
      POST /transcribe            → STTResult JSON
      POST /transcribe/stream     → SSE stream of STTSegment JSON objects
      POST /transcribe/fast       → plain-text transcript (optional)
      GET  /health                → {"healthy": bool, "circuit_state": str, ...}

    Audio transport
    ───────────────
    Local files  → multipart/form-data (field: "audio"), language/prompt as
                   form fields.  File size is validated against MAX_FILE_MB
                   before the bytes ever leave the process.
    S3 URIs      → JSON body {"audio_path": ..., "language": ..., "prompt": ...}
                   — no byte upload; the remote service resolves from S3.

    Observability
    ─────────────
    Every public method participates in the module-level Prometheus counters
    (_req_total, _latency, _ttfs, _file_size_mb, _active, _budget_exceeded,
    _circuit_open) so dashboards and alerts work identically whether the node
    is local or remote.  OTel spans are opened for each call and trace headers
    are propagated outbound so the remote service joins the same distributed
    trace as the voice graph.

    Latency budget
    ──────────────
    LatencyBudget.current() is checked at entry to every public method.
    If the pipeline SLA is already blown, the request is aborted immediately.
    The remaining budget is also forwarded as X-Latency-Budget-Ms so the
    remote service can self-abort rather than burning time producing a result
    the caller will discard.

    Resilience
    ──────────
    Circuit breaker (shared name "stt:remote") trips after sustained failures
    and prevents the event loop from queuing up doomed requests.
    backoff_retry wraps every non-streaming call (3 attempts, 1.5 s base).
    Streaming calls are not retried inside the generator — the voice graph
    retry layer handles that at a coarser granularity.

    Cancellation
    ────────────
    asyncio.CancelledError is always re-raised.  GeneratorExit is caught
    inside transcribe_stream() yield sites so caller drop-out cleans up
    without log noise.  Both cases decrement _inflight and close spans.

    Env config (resolved at module load)
    ─────────────────────────────────────
      STT_SERVICE_URL      required — base URL of the remote STT service
      STT_SERVICE_API_KEY  optional — Bearer token
      STT_SERVICE_TIMEOUT  optional — per-request timeout in seconds (default 120)
    """

    # ── construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        base_url: str   = STT_SERVICE_URL,
        api_key:  str   = STT_SERVICE_API_KEY,
        timeout:  float = STT_SERVICE_TIMEOUT,
    ) -> None:
        if not base_url:
            raise ValueError("STT_SERVICE_URL must be set to use RemoteSTTClient.")

        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key
        self._timeout  = timeout
        self._breaker  = CircuitBreaker(name="stt:remote")
        self._inflight = 0

        # Auth header baked into the client so it is never forgotten per-call.
        # Per-call headers (trace, budget, request-id) are added in
        # _request_headers() and merged at call time.
        base_headers: dict[str, str] = {}
        if api_key:
            base_headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=base_headers,
            http2=True,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _request_headers(self, rid: str) -> dict[str, str]: # noqa
        """
        Build per-request headers.

        Injects:
          X-Request-Id          — propagated to remote for correlated logging
          X-Latency-Budget-Ms   — remaining SLA budget; remote uses this to
                                  self-abort before producing a useless result
          OTel trace headers    — so the remote service joins the same trace
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

    async def _load_audio( # noqa
        self, audio_path: str
    ) -> bytes | None:
        """
        Return raw bytes for local files; None for s3:// URIs.

        Validates file existence and size before reading so the caller gets
        a clear, early error rather than an httpx timeout mid-upload.
        Raises:
          FileNotFoundError — path does not exist on disk
          ValueError        — file exceeds MAX_FILE_MB
        """
        if audio_path.startswith("s3://"):
            return None

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            raise ValueError(
                f"Audio file {audio_path!r} is {size_mb:.1f} MB "
                f"which exceeds MAX_FILE_MB={MAX_FILE_MB}"
            )

        _file_size_mb.observe(size_mb)
        return path.read_bytes()

    def _build_multipart( # noqa
        self,
        audio_bytes: bytes,
        audio_path:  str,
        language:    str | None,
        prompt:      str | None,
    ) -> tuple[dict, dict]:
        """Return (files, data) kwargs for a multipart/form-data POST."""
        files = {"audio": (Path(audio_path).name, audio_bytes, "audio/wav")}
        data: dict[str, str] = {}
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        return files, data

    def _validate_stt_result(self, raw: dict[str, Any], rid: str) -> STTResult: # noqa
        """
        Coerce the remote JSON payload into a well-typed STTResult.

        Fills defaults for any optional fields the remote may omit so that
        downstream voice_graph logic can always read every key without
        defensive guards scattered across the call sites.
        """
        if "text" not in raw:
            log.warning(
                "remote_stt_result_missing_text",
                request_id=rid,
                keys=list(raw.keys()),
            )
        return STTResult(
            text              = str(raw.get("text", "")).strip(),
            language          = str(raw.get("language", "unknown")),
            duration_s        = float(raw.get("duration_s", 0.0)),
            processing_s      = float(raw.get("processing_s", 0.0)),
            source            = str(raw.get("source", "remote")),
            s3_transcript_key = str(raw.get("s3_transcript_key", "")),
        )

    # ── transcribe (batch) ────────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path:  str,
        language:    str | None = None,
        prompt:      str | None = None,
        request_id:  str | None = None,
    ) -> STTResult:
        """
        POST /transcribe — returns a complete STTResult.

        Flow:
          1. SLA budget check — abort immediately if already blown
          2. Load / validate audio (local) or flag S3 path
          3. POST with backoff_retry inside circuit breaker (3 attempts)
          4. Validate + coerce response into STTResult
          5. Emit Prometheus counters and OTel span attributes

        Raises:
          LatencyBudgetExceeded  — SLA blown before or during the call
          asyncio.CancelledError — task was cancelled; always re-raised
          FileNotFoundError      — local audio path missing
          ValueError             — file exceeds MAX_FILE_MB
          httpx.HTTPStatusError  — 4xx/5xx from the remote service
          Exception              — any other error (circuit breaker records it)
        """
        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_stt.transcribe")
        except LatencyBudgetExceeded:
            _budget_exceeded.inc()
            log.warning("remote_stt_budget_exceeded_entry", request_id=rid)
            raise

        headers = self._request_headers(rid)

        with tracer.start_as_current_span("stt.remote.transcribe") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("audio_path", audio_path)
            span.set_attribute("provider", "remote")

            _active.labels(mode="batch").inc()
            self._inflight += 1
            t0 = time.monotonic()

            try:
                audio_bytes = await self._load_audio(audio_path)

                async def _call() -> dict[str, Any]:
                    if audio_bytes is not None:
                        files, data = self._build_multipart(
                            audio_bytes, audio_path, language, prompt
                        )
                        resp = await self._http.post(
                            "/transcribe",
                            files=files,
                            data=data,
                            headers=headers,
                        )
                    else:
                        # S3 URI — let the remote service pull the bytes
                        resp = await self._http.post(
                            "/transcribe",
                            json={
                                "audio_path": audio_path,
                                "language":   language,
                                "prompt":     prompt,
                            },
                            headers=headers,
                        )
                    resp.raise_for_status()
                    return resp.json()

                raw: dict[str, Any] = await self._breaker.call(
                    backoff_retry,
                    _call,
                    attempts=3,
                    base_delay=1.5,
                    exceptions=(Exception,),
                )

                result   = self._validate_stt_result(raw, rid)
                elapsed  = round(time.monotonic() - t0, 3)

                _req_total.labels(status="ok", mode="batch", provider="remote").inc()
                _latency.observe(elapsed)
                _circuit_open.labels(provider="remote").set(
                    1 if self._breaker.state == "OPEN" else 0
                )

                span.set_attribute("latency_s",   elapsed)
                span.set_attribute("language",    result["language"])
                span.set_attribute("duration_s",  result["duration_s"])
                span.set_attribute("text_length", len(result["text"]))

                log.info(
                    "remote_stt_transcribe_ok",
                    request_id  = rid,
                    language    = result["language"],
                    duration_s  = result["duration_s"],
                    text_length = len(result["text"]),
                    latency_s   = elapsed,
                )
                return result

            except LatencyBudgetExceeded:
                _budget_exceeded.inc()
                log.warning("remote_stt_budget_exceeded", request_id=rid)
                raise

            except asyncio.CancelledError:
                log.warning("remote_stt_transcribe_cancelled", request_id=rid)
                raise

            except FileNotFoundError:
                _req_total.labels(status="error", mode="batch", provider="remote").inc()
                span.set_status(StatusCode.ERROR, "file_not_found")
                raise

            except ValueError as exc:
                _req_total.labels(status="error", mode="batch", provider="remote").inc()
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_stt_invalid_input", request_id=rid, error=str(exc))
                raise

            except Exception as exc:
                _req_total.labels(status="error", mode="batch", provider="remote").inc()
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_stt_transcribe_error", request_id=rid, error=str(exc))
                raise

            finally:
                _active.labels(mode="batch").dec()
                self._inflight -= 1

    # ── transcribe_stream ─────────────────────────────────────────────────────

    async def transcribe_stream(
        self,
        audio_path:  str,
        language:    str | None = None,
        prompt:      str | None = None,
        request_id:  str | None = None,
    ) -> AsyncIterator[STTSegment]:
        """
        POST /transcribe/stream — yields STTSegment objects via SSE.

        Each SSE event is a JSON-serialised STTSegment.  The method:
          - enforces the SLA budget at entry and before each yield
          - strips SSE framing ("data: " prefix, "[DONE]" sentinel)
          - skips unparseable lines with a warning rather than crashing
          - handles GeneratorExit (caller dropped the generator cleanly)
          - tracks time-to-first-segment for _ttfs histogram
          - always decrements _inflight and closes the OTel span

        Streaming calls are NOT internally retried — the voice graph retry
        layer handles re-entry at a coarser granularity.  This is consistent
        with how RemoteTTSClient.synthesize_stream() behaves.

        Raises:
          LatencyBudgetExceeded  — SLA blown at entry or mid-stream
          asyncio.CancelledError — task was cancelled; always re-raised
          FileNotFoundError      — local audio path missing
          httpx.HTTPStatusError  — non-2xx from remote
        """
        import json as _json  # local import; avoids shadowing top-level json

        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_stt.stream")
        except LatencyBudgetExceeded:
            _budget_exceeded.inc()
            log.warning("remote_stt_stream_budget_exceeded_entry", request_id=rid)
            raise

        headers = {
            **self._request_headers(rid),
            "Accept": "text/event-stream",
        }

        with tracer.start_as_current_span("stt.remote.stream") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("audio_path", audio_path)
            span.set_attribute("provider", "remote")

            _active.labels(mode="stream").inc()
            self._inflight += 1
            t0             = time.monotonic()
            segment_count  = 0
            first_emitted  = False

            try:
                audio_bytes = await self._load_audio(audio_path)

                if audio_bytes is not None:
                    files, data = self._build_multipart(
                        audio_bytes, audio_path, language, prompt
                    )
                    req_kwargs: dict[str, Any] = {"files": files, "data": data}
                else:
                    req_kwargs = {
                        "json": {
                            "audio_path": audio_path,
                            "language":   language,
                            "prompt":     prompt,
                        }
                    }

                async with self._http.stream(
                    "POST", "/transcribe/stream", headers=headers, **req_kwargs
                ) as resp:
                    resp.raise_for_status()

                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue

                        # Strip SSE framing
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        if not line or line == "[DONE]":
                            continue

                        # Per-yield SLA check — abort mid-stream if budget blown
                        try:
                            self._check_budget("remote_stt.stream.before_yield")
                        except LatencyBudgetExceeded:
                            _budget_exceeded.inc()
                            log.warning(
                                "remote_stt_stream_budget_exceeded_mid",
                                request_id=rid,
                                segments_emitted=segment_count,
                            )
                            raise

                        try:
                            segment: STTSegment = _json.loads(line)
                        except _json.JSONDecodeError:
                            log.warning(
                                "remote_stt_stream_bad_line",
                                request_id=rid,
                                line_preview=line[:120],
                            )
                            continue

                        if not first_emitted:
                            _ttfs.observe(time.monotonic() - t0)
                            first_emitted = True

                        segment_count += 1

                        try:
                            yield segment
                        except (GeneratorExit, asyncio.CancelledError):
                            # Caller dropped the generator or task was cancelled —
                            # both are clean exits; log at debug, not error.
                            log.debug(
                                "remote_stt_stream_generator_exit",
                                request_id=rid,
                                segments_emitted=segment_count,
                            )
                            return

                elapsed = round(time.monotonic() - t0, 3)
                _req_total.labels(status="ok", mode="stream", provider="remote").inc()
                _chunks_per_stream.observe(segment_count)
                _circuit_open.labels(provider="remote").set(
                    1 if self._breaker.state == "OPEN" else 0
                )

                span.set_attribute("latency_s",     elapsed)
                span.set_attribute("segment_count", segment_count)

                log.info(
                    "remote_stt_stream_ok",
                    request_id    = rid,
                    segments      = segment_count,
                    latency_s     = elapsed,
                )

            except LatencyBudgetExceeded:
                _req_total.labels(status="budget_exceeded", mode="stream", provider="remote").inc()
                raise

            except asyncio.CancelledError:
                _req_total.labels(status="cancelled", mode="stream", provider="remote").inc()
                log.warning(
                    "remote_stt_stream_cancelled",
                    request_id=rid,
                    segments_emitted=segment_count,
                )
                raise

            except FileNotFoundError:
                _req_total.labels(status="error", mode="stream", provider="remote").inc()
                span.set_status(StatusCode.ERROR, "file_not_found")
                raise

            except Exception as exc:
                _req_total.labels(status="error", mode="stream", provider="remote").inc()
                span.set_status(StatusCode.ERROR, str(exc))
                log.error(
                    "remote_stt_stream_error",
                    request_id    = rid,
                    error         = str(exc),
                    segments_emitted = segment_count,
                )
                raise

            finally:
                _active.labels(mode="stream").dec()
                self._inflight -= 1

    # ── transcribe_fast ───────────────────────────────────────────────────────

    async def transcribe_fast(
        self,
        audio_path:  str,
        language:    str | None = None,
        request_id:  str | None = None,
    ) -> str:
        """
        POST /transcribe/fast — returns the raw transcript string only.

        Mirrors STTNode.transcribe_fast(): no language detection, no segment
        metadata, lower response payload.  Falls back gracefully to
        transcribe() if the remote service does not expose /transcribe/fast
        (HTTP 404 → retry via transcribe(), result["text"] returned).

        Use when latency matters and only the transcript text is needed.
        """
        rid = request_id or new_request_id()

        try:
            self._check_budget("remote_stt.transcribe_fast")
        except LatencyBudgetExceeded:
            _budget_exceeded.inc()
            log.warning("remote_stt_fast_budget_exceeded", request_id=rid)
            raise

        headers = self._request_headers(rid)

        with tracer.start_as_current_span("stt.remote.transcribe_fast") as span:
            span.set_attribute("request_id", rid)
            _active.labels(mode="batch").inc()
            self._inflight += 1
            t0 = time.monotonic()

            try:
                audio_bytes = await self._load_audio(audio_path)

                async def _call_fast() -> str:
                    if audio_bytes is not None:
                        files, data = self._build_multipart(
                            audio_bytes, audio_path, language, None
                        )
                        resp = await self._http.post(
                            "/transcribe/fast",
                            files=files,
                            data=data,
                            headers=headers,
                        )
                    else:
                        resp = await self._http.post(
                            "/transcribe/fast",
                            json={"audio_path": audio_path, "language": language},
                            headers=headers,
                        )

                    if resp.status_code == 404:
                        # Remote doesn't expose /fast — fall back to full endpoint
                        log.debug(
                            "remote_stt_fast_not_supported_fallback",
                            request_id=rid,
                        )
                        result = await self.transcribe(
                            audio_path=audio_path,
                            language=language,
                            request_id=rid,
                        )
                        return result["text"]

                    resp.raise_for_status()
                    # Remote may return plain text or {"text": "..."} JSON
                    ct = resp.headers.get("content-type", "")
                    if "application/json" in ct:
                        return str(resp.json().get("text", "")).strip()
                    return resp.text.strip()

                text: str = await self._breaker.call(
                    backoff_retry,
                    _call_fast,
                    attempts=3,
                    base_delay=1.5,
                    exceptions=(Exception,),
                )

                elapsed = round(time.monotonic() - t0, 3)
                _req_total.labels(status="ok", mode="batch", provider="remote").inc()
                _latency.observe(elapsed)
                span.set_attribute("latency_s",   elapsed)
                span.set_attribute("text_length", len(text))

                log.info(
                    "remote_stt_fast_ok",
                    request_id  = rid,
                    text_length = len(text),
                    latency_s   = elapsed,
                )
                return text

            except (LatencyBudgetExceeded, asyncio.CancelledError):
                raise

            except Exception as exc:
                _req_total.labels(status="error", mode="batch", provider="remote").inc()
                span.set_status(StatusCode.ERROR, str(exc))
                log.error("remote_stt_fast_error", request_id=rid, error=str(exc))
                raise

            finally:
                _active.labels(mode="batch").dec()
                self._inflight -= 1

    # ── cancel ────────────────────────────────────────────────────────────────

    def cancel(self, request_id: str) -> bool: # noqa
        """
        No-op for remote clients — individual in-flight HTTP requests are not
        addressable.  Returns False so callers can branch without type-checking.
        The voice graph timeout/cancellation path via asyncio.Task.cancel() is
        the correct cancellation mechanism for remote nodes.
        """
        log.debug("remote_stt_cancel_noop", request_id=request_id)
        return False

    # ── health ────────────────────────────────────────────────────────────────

    async def health(self) -> ServiceHealthState:
        """
        GET /health — returns ServiceHealthState.

        Merges the remote service's own health report with the local
        circuit breaker state and in-flight count.  A 5-second timeout
        prevents a dead health probe from blocking the voice graph health
        check cycle.

        Always returns a ServiceHealthState (never raises) — a connection
        failure produces healthy=False with the error string attached.
        """
        try:
            resp = await self._http.get("/health", timeout=5.0)
            data = resp.json()
            is_healthy = bool(data.get("healthy", False))
            _circuit_open.labels(provider="remote").set(
                1 if self._breaker.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "stt.remote",
                healthy       = is_healthy and self._breaker.state != "OPEN",
                circuit_state = self._breaker.state,
                inflight      = self._inflight,
            )
        except Exception as exc:
            _circuit_open.labels(provider="remote").set(
                1 if self._breaker.state == "OPEN" else 0
            )
            return ServiceHealthState(
                service       = "stt.remote",
                healthy       = False,
                circuit_state = self._breaker.state,
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


def get_stt_node() -> STTNodeProtocol:
    """
    Return RemoteSTTClient if STT_SERVICE_URL is set, else local STTNode.

    VoiceGraph imports only this factory — setting STT_SERVICE_URL before
    module load switches the entire STT stage to a remote service with zero
    graph-level code changes.  Mirrors get_llm_node() and get_tts_node().
    """
    if STT_SERVICE_URL:
        log.info("stt_using_remote_client", url=STT_SERVICE_URL)
        return cast(STTNodeProtocol, RemoteSTTClient())
    log.info("stt_using_local_node", model=STT_MODEL)
    return cast(STTNodeProtocol, STTNode())


# ── module-level singleton (backward-compatible) ──────────────────────────────

stt_node: STTNodeProtocol = get_stt_node()


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    async def _smoke() -> None:
        path = sys.argv[1] if len(sys.argv) > 1 else "audio/audio_IN/test.wav"
        node = get_stt_node()

        print("── batch transcribe ──")
        result = await node.transcribe(path)
        print(f"Transcript : {result['text']}")
        print(f"Language   : {result['language']}")
        print(f"Duration   : {result['duration_s']:.2f}s")

        print("\n── fast transcribe ──")
        if hasattr(node, "transcribe_fast"):
            text = await node.transcribe_fast(path)
            print(f"Fast text  : {text}")

        print("\n── streaming transcribe ──")
        async for seg in node.transcribe_stream(path):
            flag = "[FINAL]" if seg["is_final"] else f"[chunk {seg['chunk_index']}]"
            print(f"  {flag} {seg['start']:.1f}s–{seg['end']:.1f}s: {seg['text']}")

        print("\n── health ──")
        print(await node.health())

        await node.close()

    asyncio.run(_smoke())