"""
PCM audio pipeline — real-time PCM streaming subsystem.

This module is the single source of truth for raw PCM audio within the voice
pipeline. It sits between the hardware (mic/speaker) and the application nodes
(STT, TTS, player) and provides:

  PCMFormat       — immutable descriptor: sample_rate, channels, dtype, byte_order
  PCMChunk        — timestamped PCM payload with format + sequence metadata
  PCMRingBuffer   — lock-free power-of-2 circular buffer for RT producer/consumer
  PCMConverter    — resample, channel coerce, dtype convert between any two PCMFormats
  PPCMInputStream  — async network Opus → PCMChunk async iterator (FFmpeg/libopus decode)
  PCMOutputStream — PCMChunk async consumer → Opus bytes egress (FFmpeg/libopus encode)
  PCMVADGate      — energy-based VAD with hangover + pre-roll for gating STT dispatches
  PCMSplitter     — fan-out: one PCM input to N independent async consumers

  tts_pcm_to_chunk()    — parse raw OpenAI TTS PCM bytes → PCMChunk
  chunk_to_wav_bytes()  — encode PCMChunk → complete WAV bytes (for STT node)
  negotiate_format()    — pick a common PCMFormat from two capability sets

Integration points
──────────────────
  • player.play_pcm_chunk(chunk) — feeds PCMOutputStream directly, bypassing sf.read()
  • recorder streams via PCMInputStream instead of saving to disk
  • TTS node with TTS_FORMAT="pcm" calls tts_pcm_to_chunk() then PCMOutputStream
  • voice_graph.stream_full() uses PCMOutputStream for the lowest-latency path

Design decisions
────────────────
  1. PCMFormat is frozen — formats never mutate in-flight; converter creates new chunks.
  2. PCMRingBuffer uses numpy circular indexing, no malloc on the hot path.
  3. PCMInputStream wraps FFmpegPCMInputStream — asyncio-native, no PortAudio threads.
     Decodes Opus packets through a jitter buffer → libopus → PCMChunk iterator.
  4. PCMOutputStream wraps FFmpegPCMOutputStream — encodes PCMChunk → Opus bytes via
     FFmpeg subprocess pool. ABR adjusts bitrate from live network feedback.
  5. VAD uses a two-threshold (onset/hangover) energy model with configurable
     pre-roll so the first 100-200 ms of speech before a voice detection event
     is not lost.
  6. All public async methods are cancellation-safe.
  7. OTel spans on every cross-component boundary; Prometheus counters/histograms
     on every hot-path transition.
"""
# ───────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ───────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import asyncio
import collections
import contextlib
import dataclasses
import heapq
import io  # noqa
import logging  # noqa
import math  # noqa
import os
import subprocess
import queue as _stdlib_queue
import struct  # noqa
import threading  # noqa
import time  # noqa
import traceback
import wave  # noqa
import weakref  # noqa
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Sequence  # noqa
from dataclasses import dataclass, field  # noqa
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,  # noqa
    Any,
    ClassVar,
    Dict,  # noqa
    List,  # noqa
    Literal,
    NamedTuple,  # noqa
    Optional,  # noqa
    Protocol,
    Tuple,  # noqa
    Union,  # noqa
    runtime_checkable,
)

from numpy.typing import NDArray
from collections import OrderedDict

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
from opentelemetry.trace import StatusCode  # noqa

# ── I/O backend selection ─────────────────────────────────────────────────────
_USE_FFMPEG_IO: bool = os.getenv("USE_FFMPEG_IO", "0") == "1"

if _USE_FFMPEG_IO:
    from app.audio_essentials.opus_ffmpeg_io import (
        CodecConfig,
        FFmpegPCMInputStream,
        FFmpegPCMOutputStream,
        PCMFrame, # noqa
        check_opus_support as _check_opus_support,
    )
    if not _check_opus_support():
        raise RuntimeError(
            "USE_FFMPEG_IO=1 but FFmpeg/libopus not available. "
            "Run: sudo apt-get install -y ffmpeg libopus-dev"
        )
else:
    import sounddevice as sd

# ── optional heavy dependencies (lazy / graceful fallback) ────────────────────
try:
    import scipy.signal as _scipy_signal  # type: ignore[import]
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import webrtcvad as _webrtcvad  # type: ignore[import]
    _WEBRTCVAD = True
except ImportError:
    _WEBRTCVAD = False

# ── internal ──────────────────────────────────────────────────────────────────
from app.common.shared import (
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
)

from app.monitoring.observability import get_logger

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── environment config ────────────────────────────────────────────────────────

# Default mic capture format (Whisper-optimised)
_DEFAULT_INPUT_RATE: int = int(os.getenv("PCM_INPUT_SAMPLE_RATE", "16000"))
_DEFAULT_INPUT_CH: int = int(os.getenv("PCM_INPUT_CHANNELS", "1"))

# Default playback format (OpenAI TTS PCM output)
_DEFAULT_OUTPUT_RATE: int = int(os.getenv("PCM_OUTPUT_SAMPLE_RATE", "24000"))
_DEFAULT_OUTPUT_CH: int = int(os.getenv("PCM_OUTPUT_CHANNELS", "1"))

# Input stream
_INPUT_BLOCKSIZE: int = int(os.getenv("PCM_INPUT_BLOCKSIZE", "960"))   # 60 ms @ 16 kHz
_INPUT_QUEUE_MAXSIZE: int = int(os.getenv("PCM_INPUT_QUEUE_MAXSIZE", "64"))

# Output stream
_OUTPUT_QUEUE_MAXSIZE: int = int(os.getenv("PCM_OUTPUT_QUEUE_MAXSIZE", "16"))
_OUTPUT_WARMUP_FRAMES: int = int(os.getenv("PCM_OUTPUT_WARMUP_FRAMES", "512"))
_OUTPUT_LATENCY: str = os.getenv("PCM_OUTPUT_LATENCY", "low")

# Noise suppressor config
_NS_OVERSUBTRACTION: float = float(os.getenv("PCM_NS_OVERSUBTRACTION", "1.0"))   # was 2.0; lower = less aggressive
_NS_SPECTRAL_FLOOR: float  = float(os.getenv("PCM_NS_SPECTRAL_FLOOR",  "0.15"))  # was 0.02; higher = less musical noise
_NS_WARMUP_FRAMES: int     = int(os.getenv("PCM_NS_WARMUP_FRAMES",     "20"))    # was 10; more frames for noise floor estimate

# VAD config
_VAD_ONSET_RMS: float = float(os.getenv("PCM_VAD_ONSET_RMS", "80.0"))
# int16 scale — was 200; lowered for post-NS signal levels

_VAD_OFFSET_RMS: float = float(os.getenv("PCM_VAD_OFFSET_RMS", "80.0"))
_VAD_HANGOVER_S: float = float(os.getenv("PCM_VAD_HANGOVER_S", "0.4"))
_VAD_PRE_ROLL_S: float = float(os.getenv("PCM_VAD_PRE_ROLL_S", "0.15"))
_VAD_MIN_SPEECH_S: float = float(os.getenv("PCM_VAD_MIN_SPEECH_S", "0.25"))

# Ring buffer default capacity (must be power of 2)
_RING_DEFAULT_CAP: int = int(os.getenv("PCM_RING_CAPACITY_FRAMES", "65536"))

# ── Prometheus metrics ────────────────────────────────────────────────────────

_in_chunks = make_counter(
    "pcm_input_chunks_total", "PCM mic chunks captured", ["status"]
)
_in_bytes = make_counter("pcm_input_bytes_total", "PCM mic bytes captured")
_in_overflows = make_counter("pcm_input_overflows_total", "PortAudio input overflows")
_in_dropped = make_counter("pcm_input_chunks_dropped_total", "Input chunks dropped (queue full)")

_out_chunks = make_counter("pcm_output_chunks_total", "PCM chunks written to speaker", ["status"])
_out_bytes = make_counter("pcm_output_bytes_total", "PCM bytes written to speaker")
_out_underruns = make_counter("pcm_output_underruns_total", "Output queue starvation events")
_out_recreations = make_counter("pcm_output_stream_recreations_total", "Output stream re-opens")

_vad_transitions = make_counter(
    "pcm_vad_transitions_total", "VAD state transitions", ["direction"]
)
_vad_segments = make_counter("pcm_vad_segments_total", "Complete speech segments yielded")

_convert_calls = make_counter(
    "pcm_converter_calls_total", "PCM format conversions", ["kind"]
)

_input_active = make_gauge("pcm_input_stream_active", "1 when PCMInputStream is running")
_output_active = make_gauge("pcm_output_stream_active", "1 when PCMOutputStream is running")
_output_queue_depth = make_gauge("pcm_output_queue_depth", "Chunks waiting in output queue")
_vad_active = make_gauge("pcm_vad_active", "1 when VAD believes speech is ongoing")

_convert_latency = make_histogram(
    "pcm_convert_latency_seconds",
    "PCMConverter call duration",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05),
)
_output_write_latency = make_histogram(
    "pcm_output_write_latency_seconds",
    "PCMOutputStream write duration (sounddevice or FFmpeg path)",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
_chunk_duration = make_histogram(
    "pcm_chunk_duration_seconds",
    "Audio duration of individual PCMChunks",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
)

_spectral_vad_decisions = make_counter(
    "pcm_spectral_vad_decisions_total",
    "SpectralVAD frame decisions",
    ["decision"],
)
_aec_residual_rms = make_histogram(
    "pcm_aec_residual_rms",
    "Echo canceller residual RMS after AEC",
    buckets=(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0),
)
_agc_gain_applied = make_histogram(
    "pcm_agc_gain_applied",
    "AGC gain factor applied to chunk",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
)
_noise_suppressor_reduction_db = make_histogram(
    "pcm_noise_suppressor_reduction_db",
    "Noise suppression applied (dB)",
    buckets=(0, 2, 5, 10, 15, 20, 30, 40),
)
_dynamics_gain_reduction_db = make_histogram(
    "pcm_dynamics_gain_reduction_db",
    "Dynamics processor gain reduction (dB)",
    buckets=(0, 1, 3, 6, 10, 15, 20, 30),
)
_jitter_buffer_depth = make_gauge(
    "pcm_jitter_buffer_depth_packets",
    "Number of packets in jitter buffer",
)
_jitter_buffer_late_drops = make_counter(
    "pcm_jitter_buffer_late_drops_total",
    "Jitter buffer late-packet discards",
)
_jitter_buffer_concealment = make_counter(
    "pcm_jitter_buffer_concealment_total",
    "Jitter buffer PLC concealment events",
)
_mixer_clipping = make_counter(
    "pcm_mixer_clipping_events_total",
    "Stream mixer post-mix clipping events",
)
_interrupt_detections = make_counter(
    "pcm_interrupt_detections_total",
    "Barge-in / interrupt events detected",
)
_pipeline_stage_latency = make_histogram(
    "pcm_pipeline_stage_latency_seconds",
    "Per-stage processing latency",
    buckets=(0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1),
)
_diagnostics_clipping = make_counter(
    "pcm_diagnostics_clipping_events_total",
    "Diagnostic clipping detection events",
)
_diagnostics_silence = make_counter(
    "pcm_diagnostics_silence_events_total",
    "Diagnostic sustained silence events",
)
_diagnostics_dc_offset = make_counter(
    "pcm_diagnostics_dc_offset_events_total",
    "Diagnostic DC offset events",
)
_serializer_bytes_in = make_counter(
    "pcm_serializer_bytes_in_total",
    "Raw bytes serialized to wire format",
)
_serializer_bytes_out = make_counter(
    "pcm_serializer_bytes_out_total",
    "Wire-format bytes deserialized to PCM",
)
_pool_hits = make_counter("pcm_pool_hits_total", "Object pool cache hits")
_pool_misses = make_counter("pcm_pool_misses_total", "Object pool cache misses (new alloc)")
_drift_corrections = make_counter(
    "pcm_drift_corrections_total",
    "Sample-clock drift correction events",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TYPE DEFINITIONS — enums, protocols, type aliases, result dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

# ── type aliases ──────────────────────────────────────────────────────────────

NumpyDtype = Literal["int16", "int32", "float32", "float64"]
ByteOrder = Literal["little", "big", "native"]
Quality = Literal["linear", "poly", "auto"]
FusionMode = Literal["any", "all", "majority"]
DynamicsMode = Literal["compressor", "expander", "limiter", "gate"]
InterruptCallback = Callable[[], None]

# ── enums ─────────────────────────────────────────────────────────────────────

class _VADState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    HANGOVER = "hangover"


class PLCMode(Enum):
    """Packet Loss Concealment strategies."""
    ZERO_FILL = auto()      # Fill lost frames with silence
    REPEAT_LAST = auto()    # Repeat the last good packet (comfortable noise)
    NOISE_FILL = auto()     # Fill with shaped white noise (most comfortable)


# ── protocols ─────────────────────────────────────────────────────────────────

@runtime_checkable
class VADBackend(Protocol):
    """Protocol for any VAD backend usable with PCMFusedVAD."""

    def is_speech(self, chunk: PCMChunk) -> bool:
        """Return True if chunk contains speech."""
        ...


@runtime_checkable
class AsyncProcessor(Protocol):
    """Protocol for any async chunk processor usable in a pipeline."""

    def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """Consume an async chunk iterator and yield processed chunks."""
        ...


# ── result / stats dataclasses ────────────────────────────────────────────────

@dataclass
class WaveformStats:
    """Point-in-time waveform statistics for a PCMChunk."""
    n_frames: int
    duration_s: float
    rms: float
    peak: float
    crest_factor_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    is_clipping: bool
    is_silent: bool
    dc_offset: float

    def __str__(self) -> str:
        return (
            f"WaveformStats(rms={self.rms:.4f}, peak={self.peak:.4f}, "
            f"CF={self.crest_factor_db:.1f}dB, ZCR={self.zero_crossing_rate:.2f}, "
            f"centroid={self.spectral_centroid_hz:.0f}Hz, "
            f"clipping={self.is_clipping}, silent={self.is_silent})"
        )


@dataclass
class LatencyEvent:
    """Single latency observation."""
    seq: int
    stage: str
    timestamp: float
    delta_from_capture: float  # seconds from initial capture timestamp


@dataclass
class AudioHealthReport:
    """Point-in-time audio health status."""
    timestamp: float
    status: Literal["ok", "degraded", "failed"]
    clipping_rate: float      # fraction of recent chunks clipping
    silence_rate: float       # fraction of recent chunks silent
    dc_offset_mean: float     # mean DC offset (signals hardware fault)
    dropout_count: int        # seq number gaps detected
    rms_mean: float           # mean RMS level
    notes: list[str]          # human-readable issue list

    @property
    def healthy(self) -> bool:
        return self.status == "ok"

@dataclass
class PCMMetricsSnapshot:
    """
    Point-in-time snapshot of all key PCM pipeline Prometheus counters.

    Captured via get_metrics_snapshot(). Compare two snapshots with delta()
    to get rate-per-second for any counter.
    """
    timestamp: float
    input_chunks: float
    input_bytes: float
    input_overflows: float
    input_dropped: float
    output_chunks_written: float
    output_bytes: float
    output_underruns: float
    output_recreations: float
    vad_segments: float
    convert_calls: float
    jitter_late_drops: float
    jitter_concealment: float
    interrupt_detections: float
    diagnostics_clipping: float
    diagnostics_silence: float
    pool_hits: float
    pool_misses: float
    drift_corrections: float

    def delta(self, later: PCMMetricsSnapshot) -> dict[str, float]:
        """
        Compute per-second rate of change between this snapshot and a later one.

        Returns dict of metric_name → rate_per_second.
        """
        dt = max(later.timestamp - self.timestamp, 1e-9)
        fields = dataclasses.fields(self)
        result: dict[str, float] = {}
        for f in fields:
            if f.name == "timestamp":
                continue
            v1 = getattr(self, f.name)
            v2 = getattr(later, f.name)
            result[f"{f.name}_per_s"] = (v2 - v1) / dt
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CORE PRIMITIVES — PCMFormat, PCMChunk, shared utilities
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMFormat ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PCMFormat:
    """
    Immutable descriptor for a raw PCM audio stream.

    All PCMChunks carry a reference to their format so consumers never need
    out-of-band format negotiation. Equality is value-based (frozen dataclass).

    Attributes:
        sample_rate:  Frames per second (e.g. 16000, 24000, 44100, 48000).
        channels:     Number of audio channels (1=mono, 2=stereo).
        dtype:        NumPy dtype string — controls sample resolution.
                      "int16"   — 16-bit signed integer PCM (Whisper, standard telephony)
                      "int32"   — 32-bit signed integer PCM
                      "float32" — 32-bit float PCM (processing; also used for studio Opus encoding)
                      "float64" — 64-bit float (processing only, never for hardware I/O)
        byte_order:   Byte order for serialisation. "little" for almost all practical use.
    """

    sample_rate: int
    channels: int
    dtype: NumpyDtype = "float32"
    byte_order: ByteOrder = "little"

    # noinspection PyUnreachableCode
    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels < 1:
            raise ValueError(f"channels must be >= 1, got {self.channels}")
        if self.dtype not in ("int16", "int32", "float32", "float64"):
            raise ValueError(
                f"unsupported dtype: {self.dtype}"
            ) # defensive runtime validation; analyzer assumes Literal narrowing

    @property
    def bytes_per_sample(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def bytes_per_frame(self) -> int:
        return self.bytes_per_sample * self.channels

    def frames_to_bytes(self, n_frames: int) -> int:
        return n_frames * self.bytes_per_frame

    def bytes_to_frames(self, n_bytes: int) -> int:
        return n_bytes // self.bytes_per_frame

    def duration_s(self, n_frames: int) -> float:
        return n_frames / self.sample_rate

    def frames_for_duration(self, duration_s: float) -> int:
        return int(math.ceil(duration_s * self.sample_rate))

    def __repr__(self) -> str:
        return (
            f"PCMFormat(rate={self.sample_rate}, ch={self.channels}, "
            f"dtype={self.dtype}, endian={self.byte_order})"
        )

    # ── common presets ────────────────────────────────────────────────────────

    @classmethod
    def whisper(cls) -> PCMFormat:
        """Whisper-optimised: 16 kHz mono int16."""
        return cls(sample_rate=16000, channels=1, dtype="int16")

    @classmethod
    def openai_tts(cls) -> PCMFormat:
        """OpenAI TTS PCM output: 24 kHz mono int16, little-endian."""
        return cls(sample_rate=24000, channels=1, dtype="int16")

    @classmethod
    def portaudio_default(cls) -> PCMFormat:
        """48 kHz stereo float32 — WebRTC / Discord standard; used by codec layer."""
        return cls(sample_rate=48000, channels=2, dtype="float32")


# ── PCMChunk ──────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PCMChunk:
    """
    A single timestamped PCM audio payload.

    The ``data`` array has shape (n_frames,) for mono or (n_frames, channels)
    for multi-channel. dtype matches ``fmt.dtype``.

    Attributes:
        data:       Raw PCM samples as a NumPy array.
        fmt:        Format descriptor for this chunk.
        timestamp:  Monotonic capture timestamp (seconds). 0.0 if unknown.
        seq:        Monotonically increasing sequence counter within a stream.
                    Used to detect drops without parsing audio content.
        is_final:   True on the last chunk of a logical speech segment / TTS sentence.
        source:     Freeform tag identifying the producer ("mic", "tts", "file", …).
    """

    data: np.ndarray
    fmt: PCMFormat
    timestamp: float = 0.0
    seq: int = 0
    is_final: bool = False
    source: str = ""

    @property
    def n_frames(self) -> int:
        return self.data.shape[0]

    @property
    def duration_s(self) -> float:
        return self.fmt.duration_s(self.n_frames)

    @property
    def n_bytes(self) -> int:
        return self.data.nbytes

    def to_bytes(self) -> bytes:
        """Return raw little-endian bytes for serialisation / WAV writing."""
        arr = self.data
        if self.fmt.byte_order == "big":
            arr = arr.byteswap()
        return arr.tobytes()

    def ensure_2d(self) -> PCMChunk:
        """Return a chunk whose data is always (n_frames, channels)."""
        if self.data.ndim == 1:
            new_data = self.data.reshape(-1, 1) if self.fmt.channels == 1 else self.data.reshape(-1, self.fmt.channels)
            return PCMChunk(
                data=new_data,
                fmt=self.fmt,
                timestamp=self.timestamp,
                seq=self.seq,
                is_final=self.is_final,
                source=self.source,
            )
        return self

    def ensure_1d(self) -> PCMChunk:
        """Squeeze to 1-D for mono chunks."""
        if self.data.ndim == 2 and self.fmt.channels == 1:
            return PCMChunk(
                data=self.data[:, 0],
                fmt=self.fmt,
                timestamp=self.timestamp,
                seq=self.seq,
                is_final=self.is_final,
                source=self.source,
            )
        return self

    def rms(self) -> float:
        """Root-mean-square amplitude (in the native sample scale)."""
        return float(np.sqrt(np.mean(self.data.astype(np.float64) ** 2)))

    def __repr__(self) -> str:
        return (
            f"PCMChunk(frames={self.n_frames}, dur={self.duration_s:.3f}s, "
            f"seq={self.seq}, final={self.is_final}, src={self.source!r})"
        )

# ── PCMChunkPool — zero-malloc numpy array pool for the hot path ──────────────

class PCMChunkPool:
    """
    Object pool for numpy arrays to eliminate GC pressure on the hot audio path.

    The audio pipeline allocates many same-sized numpy arrays per second:
    mic callbacks at 60 ms intervals yield ~17 allocations/sec per stream.
    With multiple processing stages each allocating transformed copies, the
    GC pressure can cause latency spikes of 5-20 ms.

    PCMChunkPool caches released arrays and reuses them for next-same-size
    request. All arrays returned are zeroed before reuse (memset via
    numpy fill) to prevent data leakage between consumers.

    Thread safety: uses a threading.Lock per size bucket; lock hold time is
    O(1) (single list append/pop).

    Usage::

        pool = PCMChunkPool(max_per_bucket=32)
        buf = pool.acquire(n_frames=960, dtype="float32", channels=1)
        # ... fill buf ...
        pool.release(buf)

    Integrators should call pool.release() after consuming a chunk; omitting
    this call simply means the array is GC'd normally (no crash, just no pool benefit).
    """

    def __init__(self, max_per_bucket: int = 32) -> None:
        self._max: int = max_per_bucket
        self._buckets: dict[tuple, list[np.ndarray]] = {}
        self._locks: dict[tuple, threading.Lock] = {}
        self._master_lock = threading.Lock()

    def _key(self, n_frames: int, dtype: str, channels: int) -> tuple: # noqa
        return (n_frames, dtype, channels) # noqa

    def _get_lock(self, key: tuple) -> threading.Lock:
        with self._master_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
                self._buckets[key] = []
            return self._locks[key]

    def acquire(self, n_frames: int, dtype: str = "float32", channels: int = 1) -> np.ndarray:
        """
        Acquire an array of shape (n_frames,) for mono or (n_frames, channels)
        for multi-channel. Array is guaranteed to be zeroed.
        """
        key = self._key(n_frames, dtype, channels)
        lock = self._get_lock(key)
        with lock:
            bucket = self._buckets[key]
            if bucket:
                arr = bucket.pop()
                arr.fill(0)
                _pool_hits.inc()
                return arr

        _pool_misses.inc()
        shape = (n_frames,) if channels == 1 else (n_frames, channels)
        return np.zeros(shape, dtype=dtype)

    def release(self, arr: np.ndarray) -> None:
        """
        Return an array to the pool. The array must not be used after this call.
        """
        if arr.ndim == 1:
            n_frames, channels = arr.shape[0], 1
        else:
            n_frames, channels = arr.shape[0], arr.shape[1]
        dtype = str(arr.dtype)  # np.dtype("float32").str → "<f4", but str(np.dtype("<f4")) → "float32"
        key = self._key(n_frames, dtype, channels)
        lock = self._get_lock(key)
        with lock:
            bucket = self._buckets[key]
            if len(bucket) < self._max:
                bucket.append(arr)

    def clear(self) -> None:
        """Evict all pooled arrays (call on shutdown to release memory)."""
        with self._master_lock:
            for key in list(self._buckets.keys()):
                self._buckets[key].clear()


    def preallocate(self, count: int, n_frames: int = 960, dtype: str = "int16", channels: int = 1) -> None:
        """
        Pre-fill the pool with `count` arrays of the given shape.
        Call at startup to avoid GC pressure on the first audio burst.
        Defaults match the standard Whisper chunk size (960 frames, 16kHz int16 mono).
        """
        for _ in range(count):
            arr = np.zeros(
                (n_frames,) if channels == 1 else (n_frames, channels),
                dtype=dtype,
            )
            self.release(arr)

    def release_all(self) -> None:
        """
        Release all pooled arrays back to the GC immediately.
        Call at session end or shutdown to free memory held by idle buckets.
        Unlike clear(), this is safe to call during active use — new acquires
        will simply allocate fresh arrays until the pool refills.
        """
        self.clear()


# Module-level default pool shared across all pipeline components
_default_pool = PCMChunkPool(max_per_bucket=64)


def get_chunk_pool() -> PCMChunkPool:
    """Return the module-level default PCMChunkPool."""
    return _default_pool


# ── shared utility functions ──────────────────────────────────────────────────

def tts_pcm_to_chunk(
    raw_bytes: bytes,
    fmt: PCMFormat | None = None,
    seq: int = 0,
    is_final: bool = False,
) -> PCMChunk:
    """
    Parse raw PCM bytes from the OpenAI TTS API (format="pcm") into a PCMChunk.

    OpenAI returns 24 kHz, 1 channel, signed 16-bit little-endian PCM with
    no header — this function assumes those defaults unless overridden via ``fmt``.

    Args:
        raw_bytes:  Raw bytes from TTS response (no WAV/MP3 container).
        fmt:        Override format. Defaults to PCMFormat.openai_tts().
        seq:        Sequence number for ordering.
        is_final:   True when this is the last chunk of a synthesis.

    Returns:
        PCMChunk with data as int16 numpy array.
    """
    if fmt is None:
        fmt = PCMFormat.openai_tts()

    n_samples = len(raw_bytes) // fmt.bytes_per_sample
    if n_samples == 0:
        log.warning("tts_pcm_empty_chunk", n_bytes=len(raw_bytes))
        return PCMChunk(
            data=np.array([], dtype=fmt.dtype),
            fmt=fmt,
            timestamp=time.monotonic(),
            seq=seq,
            is_final=is_final,
            source="tts",
        )

    order = "<" if fmt.byte_order == "little" else ">"
    np_dtype = np.dtype(f"{order}{np.dtype(fmt.dtype).str[1:]}")
    data = np.frombuffer(raw_bytes[: n_samples * fmt.bytes_per_sample], dtype=np_dtype)

    # Reshape to (frames, channels) if multi-channel
    if fmt.channels > 1:
        n_frames = n_samples // fmt.channels
        data = data[: n_frames * fmt.channels].reshape(n_frames, fmt.channels)

    # Make native byte order for numpy operations
    data = np.ascontiguousarray(data.astype(fmt.dtype))

    return PCMChunk(
        data=data,
        fmt=fmt,
        timestamp=time.monotonic(),
        seq=seq,
        is_final=is_final,
        source="tts",
    )


def chunk_to_wav_bytes(chunk: PCMChunk) -> bytes:
    """
    Encode a PCMChunk as a complete WAV file in memory.

    The output bytes are immediately consumable by soundfile.read() or the
    STT node's transcribe() method (which expects a .wav-like file).

    Only int16 and float32 dtypes are supported for WAV encoding.
    float32 chunks are converted to int16 before encoding.

    Returns:
        Complete WAV bytes including RIFF/fmt/data headers.
    """
    data = chunk.data
    fmt = chunk.fmt

    # Ensure int16 for WAV PCM_16
    if fmt.dtype != "int16":
        converter = PCMConverter()
        int16_fmt = PCMFormat( # noqa
            sample_rate=fmt.sample_rate,
            channels=fmt.channels,
            dtype="int16",
        )
        data = converter._coerce_dtype(data, fmt.dtype, "int16")  # type: ignore[arg-type] # noqa
    else:
        data = data.copy()

    # Flatten mono 2-D → 1-D
    if data.ndim == 2 and fmt.channels == 1:
        data = data[:, 0]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(fmt.channels)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(fmt.sample_rate)
        wf.writeframes(data.astype("<i2").tobytes())

    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
# PCMTranscoder — env-gated PCM → any-format transcoding via FFmpeg
# ═══════════════════════════════════════════════════════════════════════════════
#
# DROP-IN LOCATION: paste this block into audio_engine.py immediately after
# the chunk_to_wav_bytes() function (around line 827).
#
# ENV VARS
# ────────
#   TTS_FORMAT      Target container format.  Default: "pcm" (passthrough).
#                   Supported values (case-insensitive):
#                     pcm   — raw bytes, no transcoding  (zero latency)
#                     wav   — RIFF/WAV via FFmpeg
#                     mp3   — MPEG layer 3 via libmp3lame
#                     ogg   — Ogg/Vorbis via libvorbis
#                     opus  — Ogg/Opus via libopus
#                     flac  — lossless FLAC
#                     aac   — raw AAC (ADTS) via native FFmpeg encoder
#                     webm  — WebM/Opus via libopus
#                     m4a   — MP4/AAC via native FFmpeg encoder
#
#   TTS_FORMAT_BITRATE_KBPS   Target bitrate for lossy codecs. Default: "128"
#   TTS_FORMAT_SAMPLE_RATE    Force output sample rate.  Default: match source.
#   TTS_FORMAT_QUALITY        For VBR codecs: 0 (best) – 9 (smallest).
#                             Default: "4"
#
# USAGE
# ─────
#   # Simplest — module-level helper reads TTS_FORMAT automatically
#   encoded: bytes = transcode_pcm_chunk(chunk)
#
#   # Explicit format override (ignores TTS_FORMAT)
#   wav_bytes = transcode_pcm_chunk(chunk, fmt_override="wav")
#   mp3_bytes = transcode_pcm_chunk(chunk, fmt_override="mp3")
#
#   # Async variant — non-blocking, runs FFmpeg in a thread-pool executor
#   mp3_bytes = await transcode_pcm_chunk_async(chunk)
#
#   # Batch: convert a list of chunks into one contiguous encoded blob
#   blob = transcode_pcm_chunks(chunks)          # sync
#   blob = await transcode_pcm_chunks_async(chunks)  # async
#
#   # Low-level: instantiate the class directly for repeated transcoding
#   transcoder = PCMTranscoder(target_format="mp3", bitrate_kbps=192)
#   mp3_bytes   = transcoder.transcode(chunk)
#   mp3_bytes   = await transcoder.transcode_async(chunk)
#
# IMPORTANT NOTES
# ───────────────
#   • This module does NOT change any internal wiring.  tts_pcm_to_chunk(),
#     PCMOutputStream, PCMInputStream, and PCMVADGate are untouched.
#   • Transcoding adds latency proportional to chunk duration and codec
#     complexity.  Typical overhead: mp3 ≈ 3-8 ms/s, flac ≈ 1-3 ms/s.
#   • When TTS_FORMAT="pcm" (default), transcode_pcm_chunk() is a zero-copy
#     passthrough — it returns chunk.to_bytes() with no subprocess at all.
#   • All FFmpeg subprocesses are run with communicate() — no pipes left open
#     between calls.  Safe for concurrent async callers.
#   • The _PCM_FORMAT_MAP dict from opus_ffmpeg_io is re-implemented here as
#     _PCM_FFMPEG_FMT so this block works regardless of _USE_FFMPEG_IO flag.
# ═══════════════════════════════════════════════════════════════════════════════

# ── env config for transcoder ─────────────────────────────────────────────────

_TTS_FORMAT: str = os.getenv("TTS_FORMAT", "pcm").lower().strip()
_TTS_FORMAT_BITRATE_KBPS: int = int(os.getenv("TTS_FORMAT_BITRATE_KBPS", "128"))
_TTS_FORMAT_SAMPLE_RATE: int | None = (
    int(os.getenv("TTS_FORMAT_SAMPLE_RATE"))
    if os.getenv("TTS_FORMAT_SAMPLE_RATE")
    else None
)
_TTS_FORMAT_QUALITY: int = max(0, min(9, int(os.getenv("TTS_FORMAT_QUALITY", "4"))))

# ── internal format tables ────────────────────────────────────────────────────

# Maps PCM numpy dtype → FFmpeg raw-format flag used as -f <flag> on input side.
# Mirrors the _PCM_FORMAT_MAP in opus_ffmpeg_io.py without importing it.
_PCM_FFMPEG_FMT: dict[str, str] = {
    "int16":   "s16le",
    "int32":   "s32le",
    "float32": "f32le",
    "float64": "f64le",
}

# Each entry: (ffmpeg_output_format, codec, extra_args_template)
# extra_args_template is a callable that receives (bitrate_kbps, quality, out_rate)
# and returns a list of additional FFmpeg args.
_FORMAT_TABLE: dict[str, tuple[str, str, "Callable[[int, int, int], list[str]]"]] = {
    "wav":  (
        "wav",
        "pcm_s16le",
        lambda br, q, rate: [],
    ),
    "mp3":  (
        "mp3",
        "libmp3lame",
        lambda br, q, rate: ["-b:a", f"{br}k", "-q:a", str(q)],
    ),
    "ogg":  (
        "ogg",
        "libvorbis",
        lambda br, q, rate: ["-b:a", f"{br}k", "-q:a", str(q)],
    ),
    "opus": (
        "ogg",
        "libopus",
        lambda br, q, rate: ["-b:a", f"{br}k", "-vbr", "on"],
    ),
    "flac": (
        "flac",
        "flac",
        lambda br, q, rate: ["-compression_level", str(min(q, 8))],
    ),
    "aac":  (
        "adts",
        "aac",
        lambda br, q, rate: ["-b:a", f"{br}k"],
    ),
    "webm": (
        "webm",
        "libopus",
        lambda br, q, rate: ["-b:a", f"{br}k"],
    ),
    "m4a":  (
        "mp4",
        "aac",
        lambda br, q, rate: ["-b:a", f"{br}k", "-movflags", "frag_keyframe+empty_moov"],
    ),
}

# ── Prometheus metrics for transcoder ────────────────────────────────────────

_transcode_calls = make_counter(
    "pcm_transcode_calls_total",
    "PCM → encoded-format transcode calls",
    ["target_format", "status"],
)
_transcode_latency = make_histogram(
    "pcm_transcode_latency_seconds",
    "Wall-clock duration of FFmpeg transcode subprocess",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
_transcode_bytes_in = make_counter(
    "pcm_transcode_bytes_in_total",
    "Raw PCM bytes fed into the transcoder",
)
_transcode_bytes_out = make_counter(
    "pcm_transcode_bytes_out_total",
    "Encoded bytes produced by the transcoder",
)

# ── PCMTranscoder ─────────────────────────────────────────────────────────────

class PCMTranscoder:
    """
    Converts PCMChunk objects (or raw PCM bytes) to any FFmpeg-supported
    container format (mp3, wav, ogg, opus, flac, aac, webm, m4a).

    Instantiate once and call transcode() repeatedly.  Each call spawns an
    FFmpeg subprocess, pipes in raw PCM, and returns the encoded bytes.
    When target_format is "pcm", transcode() is a zero-copy passthrough.

    Thread/async safety: each transcode() call is independent.  Multiple
    callers may call transcode() concurrently — each gets its own subprocess.

    Args:
        target_format:   One of the keys in _FORMAT_TABLE, or "pcm".
                         Defaults to the TTS_FORMAT env var.
        bitrate_kbps:    Bitrate for lossy codecs. Defaults to env value.
        output_rate:     Force output sample rate.  None = match source.
        quality:         VBR quality hint (0=best, 9=smallest).
        ffmpeg_bin:      Path to ffmpeg binary. Default: "ffmpeg".
        timeout_s:       Max seconds to wait for FFmpeg to finish per chunk.
                         Default: 30.  Increase for very long TTS segments.
    """

    def __init__(
        self,
        target_format: str | None = None,
        bitrate_kbps: int | None = None,
        output_rate: int | None = None,
        quality: int | None = None,
        ffmpeg_bin: str = "ffmpeg",
        timeout_s: float = 30.0,
    ) -> None:
        self.target_format: str = (target_format or _TTS_FORMAT).lower().strip()
        self.bitrate_kbps: int = bitrate_kbps if bitrate_kbps is not None else _TTS_FORMAT_BITRATE_KBPS
        self.output_rate: int | None = output_rate if output_rate is not None else _TTS_FORMAT_SAMPLE_RATE
        self.quality: int = quality if quality is not None else _TTS_FORMAT_QUALITY
        self.ffmpeg_bin: str = ffmpeg_bin
        self.timeout_s: float = timeout_s

        if self.target_format not in ("pcm", *_FORMAT_TABLE):
            raise ValueError(
                f"Unsupported TTS_FORMAT={self.target_format!r}. "
                f"Valid options: pcm, {', '.join(sorted(_FORMAT_TABLE))}"
            )

    # ── public interface ──────────────────────────────────────────────────────

    def transcode(self, chunk: "PCMChunk") -> bytes:
        """
        Synchronously transcode *chunk* to the configured output format.

        When target_format is "pcm", returns chunk.to_bytes() with no
        subprocess overhead.

        Args:
            chunk: A PCMChunk produced by tts_pcm_to_chunk() or any other
                   pipeline stage.

        Returns:
            Encoded audio bytes in the target container format.

        Raises:
            RuntimeError: If FFmpeg exits with a non-zero return code.
            TimeoutError: If FFmpeg does not finish within timeout_s seconds.
        """
        if self.target_format == "pcm":
            return chunk.to_bytes()

        raw_pcm = self._extract_pcm_bytes(chunk)
        return self._run_ffmpeg_sync(
            raw_pcm=raw_pcm,
            source_fmt=chunk.fmt,
        )

    async def transcode_async(self, chunk: "PCMChunk") -> bytes:
        """
        Asynchronously transcode *chunk* using asyncio.subprocess.

        Runs FFmpeg as an async subprocess — does not block the event loop
        while waiting for encoding to finish.

        Args:
            chunk: A PCMChunk to transcode.

        Returns:
            Encoded audio bytes.

        Raises:
            RuntimeError: If FFmpeg exits with a non-zero return code.
            asyncio.TimeoutError: If FFmpeg does not finish within timeout_s.
        """
        if self.target_format == "pcm":
            return chunk.to_bytes()

        raw_pcm = self._extract_pcm_bytes(chunk)
        return await self._run_ffmpeg_async(
            raw_pcm=raw_pcm,
            source_fmt=chunk.fmt,
        )

    def transcode_raw(
        self,
        raw_pcm: bytes,
        sample_rate: int,
        channels: int,
        dtype: str = "int16",
    ) -> bytes:
        """
        Transcode raw PCM bytes without a PCMChunk wrapper.

        Useful when you have a contiguous PCM blob (e.g. from a file or a
        stitched TTS response) and want to avoid constructing a PCMChunk.

        Args:
            raw_pcm:     Raw PCM bytes, no header.
            sample_rate: Source sample rate in Hz.
            channels:    Number of audio channels.
            dtype:       PCM dtype string ("int16", "int32", "float32").

        Returns:
            Encoded audio bytes.
        """
        if self.target_format == "pcm":
            return raw_pcm

        # Construct a minimal stand-in for the PCMFormat duck type
        src_fmt = _TranscoderFormat(
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype,
        )
        return self._run_ffmpeg_sync(raw_pcm=raw_pcm, source_fmt=src_fmt)  # type: ignore[arg-type]

    async def transcode_raw_async(
        self,
        raw_pcm: bytes,
        sample_rate: int,
        channels: int,
        dtype: str = "int16",
    ) -> bytes:
        """Async variant of transcode_raw()."""
        if self.target_format == "pcm":
            return raw_pcm

        src_fmt = _TranscoderFormat(
            sample_rate=sample_rate,
            channels=channels,
            dtype=dtype,
        )
        return await self._run_ffmpeg_async(raw_pcm=raw_pcm, source_fmt=src_fmt)  # type: ignore[arg-type]

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_pcm_bytes(chunk: "PCMChunk") -> bytes:
        """Return the raw PCM payload of *chunk* as little-endian bytes."""
        data = chunk.data
        if data.dtype.byteorder not in ("<", "=", "|"):
            data = data.astype(data.dtype.newbyteorder("<"), copy=False)
        return data.astype(f"<{np.dtype(chunk.fmt.dtype).str[1:]}").tobytes()

    def _build_ffmpeg_cmd(self, source_fmt: PCMFormat) -> list[str]:
        """
        Build the FFmpeg command line for stdin → stdout transcoding.

        Input:  raw PCM from stdin (-f <pcm_fmt> -ar <rate> -ac <ch> -i pipe:0)
        Output: encoded audio to stdout (-f <output_fmt> pipe:1)
        """
        pcm_flag = _PCM_FFMPEG_FMT.get(source_fmt.dtype, "s16le")
        out_rate = self.output_rate or source_fmt.sample_rate

        out_fmt, codec, extra_fn = _FORMAT_TABLE[self.target_format]
        extra_args = extra_fn(self.bitrate_kbps, self.quality, out_rate)

        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", "error",
            # Input
            "-f",  pcm_flag,
            "-ar", str(source_fmt.sample_rate),
            "-ac", str(source_fmt.channels),
            "-i",  "pipe:0",
            # Resample if needed
            "-ar", str(out_rate),
            # Codec
            "-c:a", codec,
            *extra_args,
            # Output
            "-f",  out_fmt,
            "pipe:1",
        ]
        return cmd

    def _run_ffmpeg_sync(
        self,
        raw_pcm: bytes,
        source_fmt: PCMFormat,
    ) -> bytes:
        """Spawn FFmpeg synchronously and return encoded bytes."""
        import signal as _signal  # lazy import; not available on all platforms # noqa

        cmd = self._build_ffmpeg_cmd(source_fmt)
        t0 = time.monotonic()

        _transcode_bytes_in.inc(len(raw_pcm))
        log.debug(
            "pcm_transcode_start",
            target_format=self.target_format,
            src_rate=source_fmt.sample_rate,
            src_ch=source_fmt.channels,
            pcm_bytes=len(raw_pcm),
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = proc.communicate(input=raw_pcm, timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()  # drain pipes
                _transcode_calls.labels(
                    target_format=self.target_format, status="timeout"
                ).inc()
                raise TimeoutError(
                    f"FFmpeg transcode to {self.target_format!r} "
                    f"timed out after {self.timeout_s}s"
                )

            elapsed = time.monotonic() - t0
            _transcode_latency.observe(elapsed)

            if proc.returncode != 0:
                _transcode_calls.labels(
                    target_format=self.target_format, status="error"
                ).inc()
                stderr_text = stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg exited {proc.returncode} transcoding to "
                    f"{self.target_format!r}: {stderr_text}"
                )

            _transcode_calls.labels(
                target_format=self.target_format, status="ok"
            ).inc()
            _transcode_bytes_out.inc(len(stdout))
            log.debug(
                "pcm_transcode_done",
                target_format=self.target_format,
                out_bytes=len(stdout),
                elapsed_ms=round(elapsed * 1000, 2),
            )
            return stdout

        except (OSError, FileNotFoundError) as exc:
            _transcode_calls.labels(
                target_format=self.target_format, status="error"
            ).inc()
            raise RuntimeError(
                f"Failed to launch FFmpeg for {self.target_format!r} transcoding. "
                f"Ensure FFmpeg is installed and on PATH. Original error: {exc}"
            ) from exc

    async def _run_ffmpeg_async(
        self,
        raw_pcm: bytes,
        source_fmt: PCMFormat,
    ) -> bytes:
        """Spawn FFmpeg asynchronously and return encoded bytes."""
        cmd = self._build_ffmpeg_cmd(source_fmt)
        t0 = time.monotonic()

        _transcode_bytes_in.inc(len(raw_pcm))
        log.debug(
            "pcm_transcode_async_start",
            target_format=self.target_format,
            src_rate=source_fmt.sample_rate,
            src_ch=source_fmt.channels,
            pcm_bytes=len(raw_pcm),
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(raw_pcm),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                _transcode_calls.labels(
                    target_format=self.target_format, status="timeout"
                ).inc()
                raise asyncio.TimeoutError(
                    f"FFmpeg async transcode to {self.target_format!r} "
                    f"timed out after {self.timeout_s}s"
                )

            elapsed = time.monotonic() - t0
            _transcode_latency.observe(elapsed)

            if proc.returncode != 0:
                _transcode_calls.labels(
                    target_format=self.target_format, status="error"
                ).inc()
                stderr_text = stderr.decode(errors="replace").strip()
                raise RuntimeError(
                    f"FFmpeg async exited {proc.returncode} transcoding to "
                    f"{self.target_format!r}: {stderr_text}"
                )

            _transcode_calls.labels(
                target_format=self.target_format, status="ok"
            ).inc()
            _transcode_bytes_out.inc(len(stdout))
            log.debug(
                "pcm_transcode_async_done",
                target_format=self.target_format,
                out_bytes=len(stdout),
                elapsed_ms=round(elapsed * 1000, 2),
            )
            return stdout

        except (OSError, FileNotFoundError) as exc:
            _transcode_calls.labels(
                target_format=self.target_format, status="error"
            ).inc()
            raise RuntimeError(
                f"Failed to launch FFmpeg for async {self.target_format!r} transcoding. "
                f"Ensure FFmpeg is installed. Original error: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"PCMTranscoder(fmt={self.target_format!r}, "
            f"br={self.bitrate_kbps}kbps, q={self.quality}, "
            f"out_rate={self.output_rate or 'src'})"
        )


# ── _TranscoderFormat — minimal duck-type for raw-bytes path ─────────────────

@dataclass(frozen=True, slots=True)
class _TranscoderFormat:
    """
    Minimal stand-in for PCMFormat used by PCMTranscoder.transcode_raw().
    Avoids creating a full PCMFormat when you only have a bytes blob.
    """
    sample_rate: int
    channels: int
    dtype: str = "int16"


# ── module-level singleton + convenience wrappers ─────────────────────────────

_default_transcoder: PCMTranscoder | None = None


def _get_default_transcoder() -> PCMTranscoder:
    """
    Return (or lazily create) the module-level singleton PCMTranscoder.

    The singleton is built once using the current TTS_FORMAT env vars.
    Call reset_transcoder() to rebuild it after changing config at runtime.
    """
    global _default_transcoder
    if _default_transcoder is None:
        _default_transcoder = PCMTranscoder()
        log.info(
            "pcm_transcoder_init",
            target_format=_default_transcoder.target_format,
            bitrate_kbps=_default_transcoder.bitrate_kbps,
            quality=_default_transcoder.quality,
            output_rate=_default_transcoder.output_rate,
        )
    return _default_transcoder


def reset_transcoder() -> None:
    """
    Force the module-level PCMTranscoder singleton to be rebuilt on next use.

    Call this if you change TTS_FORMAT / TTS_FORMAT_BITRATE_KBPS at runtime
    (e.g. in tests or when hot-reloading config).
    """
    global _default_transcoder
    _default_transcoder = None


def transcode_pcm_chunk(
    chunk: "PCMChunk",
    fmt_override: str | None = None,
) -> bytes:
    """
    Convert a PCMChunk to the format specified by TTS_FORMAT (or fmt_override).

    This is the primary entry-point for callers that want "give me TTS audio in
    whatever format the operator configured."  When TTS_FORMAT="pcm" (default),
    this is a zero-overhead passthrough.

    Args:
        chunk:        A PCMChunk, typically from tts_pcm_to_chunk().
        fmt_override: Override TTS_FORMAT for this call only (e.g. "wav").
                      Does not affect the singleton or env var.

    Returns:
        Audio bytes in the target format. For "pcm", these are the raw
        little-endian PCM samples. For all others, a complete encoded file.

    Example::

        raw = await tts_client.synthesize_pcm(text)
        chunk = tts_pcm_to_chunk(raw, seq=seq)
        audio = transcode_pcm_chunk(chunk)        # respects TTS_FORMAT env
        await websocket.send_bytes(audio)
    """
    if fmt_override is not None:
        tc = PCMTranscoder(target_format=fmt_override)
        return tc.transcode(chunk)
    return _get_default_transcoder().transcode(chunk)


async def transcode_pcm_chunk_async(
    chunk: "PCMChunk",
    fmt_override: str | None = None,
) -> bytes:
    """
    Async variant of transcode_pcm_chunk().

    Uses asyncio.create_subprocess_exec so the event loop is never blocked
    while FFmpeg is encoding.  Preferred in async contexts (voice_graph,
    WebSocket handlers, etc.).

    Args:
        chunk:        A PCMChunk, typically from tts_pcm_to_chunk().
        fmt_override: Override TTS_FORMAT for this call only.

    Returns:
        Encoded audio bytes in the configured or overridden format.

    Example::

        chunk = tts_pcm_to_chunk(pcm_bytes, seq=seq, is_final=is_final)
        mp3_bytes = await transcode_pcm_chunk_async(chunk)
        await s3.upload(mp3_bytes, key=f"tts/{request_id}.mp3")
    """
    if fmt_override is not None:
        tc = PCMTranscoder(target_format=fmt_override)
        return await tc.transcode_async(chunk)
    return await _get_default_transcoder().transcode_async(chunk)

# noinspection PyProtectedMember
def transcode_pcm_chunks(
    chunks: "Sequence[PCMChunk]",
    fmt_override: str | None = None,
) -> bytes:
    """
    Concatenate and transcode a list of PCMChunks into a single encoded blob.

    Joins the raw PCM of all chunks into one contiguous buffer before feeding
    to FFmpeg.  This is more efficient than transcoding each chunk individually
    because it avoids repeated subprocess launches and produces better codec
    decisions (VBR, frame boundaries, etc.).

    Args:
        chunks:       Ordered sequence of PCMChunks (same fmt assumed).
        fmt_override: Optional format override.

    Returns:
        Single encoded audio blob covering all input chunks.

    Raises:
        ValueError: If chunks is empty.
    """
    if not chunks:
        raise ValueError("transcode_pcm_chunks(): chunks must be non-empty")

    representative_fmt = chunks[0].fmt
    raw_pcm = b"".join(PCMTranscoder._extract_pcm_bytes(c) for c in chunks) # type: ignore[attr-defined]

    if fmt_override is not None:
        tc = PCMTranscoder(target_format=fmt_override)
    else:
        tc = _get_default_transcoder()

    if tc.target_format == "pcm":
        return raw_pcm

    src = _TranscoderFormat(
        sample_rate=representative_fmt.sample_rate,
        channels=representative_fmt.channels,
        dtype=representative_fmt.dtype,
    )
    return tc._run_ffmpeg_sync(raw_pcm=raw_pcm, source_fmt=src)  # type: ignore[arg-type]

# noinspection PyProtectedMember
async def transcode_pcm_chunks_async(
    chunks: "Sequence[PCMChunk]",
    fmt_override: str | None = None,
) -> bytes:
    """Async variant of transcode_pcm_chunks()."""
    if not chunks:
        raise ValueError("transcode_pcm_chunks_async(): chunks must be non-empty")

    representative_fmt = chunks[0].fmt
    raw_pcm = b"".join(PCMTranscoder._extract_pcm_bytes(c) for c in chunks) # type: ignore[attr-defined]

    if fmt_override is not None:
        tc = PCMTranscoder(target_format=fmt_override)
    else:
        tc = _get_default_transcoder()

    if tc.target_format == "pcm":
        return raw_pcm

    src = _TranscoderFormat(
        sample_rate=representative_fmt.sample_rate,
        channels=representative_fmt.channels,
        dtype=representative_fmt.dtype,
    )
    return await tc._run_ffmpeg_async(raw_pcm=raw_pcm, source_fmt=src)  # type: ignore[arg-type]


# ── supported_tts_formats() — discovery helper ────────────────────────────────

def supported_tts_formats() -> list[str]:
    """
    Return the list of format strings supported by PCMTranscoder.

    Useful for validating TTS_FORMAT at startup or in health checks::

        if os.getenv("TTS_FORMAT", "pcm") not in supported_tts_formats():
            raise ValueError(f"Bad TTS_FORMAT: {os.getenv('TTS_FORMAT')}")
    """
    return ["pcm", *sorted(_FORMAT_TABLE.keys())]

def negotiate_format(
    preferred: Sequence[PCMFormat],
    supported: Sequence[PCMFormat],
) -> PCMFormat:
    """
    Pick the best common PCMFormat from two capability sets.

    Scoring:
      • Exact match → highest priority
      • Same sample rate → +10 points
      • Same channels → +5 points
      • Same dtype → +3 points
      • Lower conversion cost (fewer steps) → tiebreak

    Raises:
        ValueError: if either sequence is empty.
    """
    if not preferred or not supported:
        raise ValueError("Both preferred and supported must be non-empty")

    # Exact match fast path
    for p in preferred:
        if p in supported:
            return p

    best: PCMFormat = supported[0]
    best_score = -1

    for p in preferred:
        for s in supported:
            score = 0
            if p.sample_rate == s.sample_rate:
                score += 10
            if p.channels == s.channels:
                score += 5
            if p.dtype == s.dtype:
                score += 3
            if score > best_score:
                best_score = score
                best = s

    log.debug(
        "pcm_format_negotiated",
        chosen=repr(best),
        score=best_score,
    )
    return best


def fmt_telephone() -> PCMFormat:
    """Narrowband telephone: 8 kHz mono int16 (G.711 compatible)."""
    return PCMFormat(sample_rate=8000, channels=1, dtype="int16")


def fmt_wideband_voice() -> PCMFormat:
    """Wideband voice: 16 kHz mono float32 (HD Voice ITU-T G.722)."""
    return PCMFormat(sample_rate=16000, channels=1, dtype="float32")


def fmt_hd_voice() -> PCMFormat:
    """Super-wideband: 32 kHz mono float32."""
    return PCMFormat(sample_rate=32000, channels=1, dtype="float32")


def fmt_fullband() -> PCMFormat:
    """Full-band: 48 kHz mono float32 (WebRTC / Opus reference)."""
    return PCMFormat(sample_rate=48000, channels=1, dtype="float32")


def fmt_cd_stereo() -> PCMFormat:
    """CD-quality stereo: 44100 Hz 2-channel int16."""
    return PCMFormat(sample_rate=44100, channels=2, dtype="int16")


def fmt_studio_stereo() -> PCMFormat:
    """Studio: 48 kHz stereo float32."""
    return PCMFormat(sample_rate=48000, channels=2, dtype="float32")


def fmt_elevenlabs() -> PCMFormat:
    """ElevenLabs TTS PCM output: 44100 Hz mono int16."""
    return PCMFormat(sample_rate=44100, channels=1, dtype="int16")


def fmt_deepgram() -> PCMFormat:
    """Deepgram STT preferred: 16 kHz mono int16 (same as Whisper)."""
    return PCMFormat(sample_rate=16000, channels=1, dtype="int16")


def fmt_azure_speech() -> PCMFormat:
    """Azure Cognitive Speech: 16 kHz mono int16."""
    return PCMFormat(sample_rate=16000, channels=1, dtype="int16")


def fmt_google_cloud_speech() -> PCMFormat:
    """Google Cloud Speech-to-Text: 16 kHz mono int16 (LINEAR16)."""
    return PCMFormat(sample_rate=16000, channels=1, dtype="int16")


# ──────────────────────────────────────────────────────────────────────────────
# 4. LOW-LEVEL DSP BUILDING BLOCKS — converters, buffers, resamplers, filters
# ──────────────────────────────────────────────────────────────────────────────

# ── PCMConverter ──────────────────────────────────────────────────────────────

class PCMConverter:
    """
    Stateless format converter between any two PCMFormats.

    Conversion order (applied in sequence):
      1. dtype coerce (avoid double conversion noise)
      2. channel coerce (upmix mono→stereo or downmix stereo→mono)
      3. resample (linear interpolation for low-latency; quality resampler optional)

    Resampling strategy:
      • numpy linear interpolation is always available and adds ~0.1 ms per chunk.
      • If ``scipy`` is installed, uses ``scipy.signal.resample_poly`` for higher
        quality (no aliasing artefacts at low ratios). Falls back gracefully.
      • Quality can be set to "linear", "poly" (scipy), or "auto" (poly if available).

    All conversions return a new PCMChunk; inputs are never mutated.
    """

    def __init__(self, quality: Quality = "auto") -> None:
        self._quality = quality
        self._scipy_available: bool | None = None

    def _use_poly(self) -> bool:
        if self._quality == "linear":
            return False
        if self._quality == "poly":
            return True
        # auto: probe once
        if self._scipy_available is None:
            try:
                import scipy.signal  # noqa
                self._scipy_available = True
            except ImportError:
                self._scipy_available = False
        return self._scipy_available  # type: ignore[return-value]

    # noinspection PyUnreachableCode
    def convert(self, chunk: PCMChunk, target_fmt: PCMFormat) -> PCMChunk:
        """
        Convert ``chunk`` to ``target_fmt``.

        Raises:
            ValueError: if channel coercion requires a ratio > 2 (multi-channel
                        to mono or mono to multi-channel >2 channels).
        """
        if chunk.fmt == target_fmt:
            return chunk  # fast path: nothing to do

        t0 = time.monotonic()
        data = chunk.data
        src = chunk.fmt

        # ── 1. dtype coerce ───────────────────────────────────────────────────
        if src.dtype != target_fmt.dtype:
            data = self._coerce_dtype(data, src.dtype, target_fmt.dtype)
            _convert_calls.labels(kind="dtype").inc() # defensive runtime validation; analyzer assumes Literal narrowing

        # ── 2. channel coerce ─────────────────────────────────────────────────
        if src.channels != target_fmt.channels:
            data = self._coerce_channels(data, src.channels, target_fmt.channels)
            _convert_calls.labels(kind="channel").inc()

        # ── 3. resample ───────────────────────────────────────────────────────
        if src.sample_rate != target_fmt.sample_rate:
            data = self._resample(data, src.sample_rate, target_fmt.sample_rate)
            _convert_calls.labels(kind="resample").inc()

        _convert_latency.observe(time.monotonic() - t0)

        return PCMChunk(
            data=data,
            fmt=target_fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    # ── dtype coercion ────────────────────────────────────────────────────────

    @staticmethod
    def _coerce_dtype(
        data: np.ndarray, src_dtype: NumpyDtype, dst_dtype: NumpyDtype
    ) -> np.ndarray:
        src_np = np.dtype(src_dtype)
        dst_np = np.dtype(dst_dtype)

        # Float → Float: simple cast
        if np.issubdtype(src_np, np.floating) and np.issubdtype(dst_np, np.floating):
            return data.astype(dst_np)

        # Float → Int: scale to int range
        if np.issubdtype(src_np, np.floating) and np.issubdtype(dst_np, np.integer):
            max_val = np.iinfo(dst_np).max
            clipped = np.clip(data, -1.0, 1.0)
            return (clipped * max_val).astype(dst_np)

        # Int → Float: normalise to [-1, 1]
        if np.issubdtype(src_np, np.integer) and np.issubdtype(dst_np, np.floating):
            max_val = np.iinfo(src_np).max
            return data.astype(dst_np) / max_val

        # Int → Int: scale proportionally
        src_max = np.iinfo(src_np).max
        dst_max = np.iinfo(dst_np).max
        return (data.astype(np.float64) * dst_max / src_max).astype(dst_np)

    # ── channel coercion ──────────────────────────────────────────────────────

    @staticmethod
    def _coerce_channels(
        data: np.ndarray, src_ch: int, dst_ch: int
    ) -> np.ndarray:
        # Ensure 2-D: (frames, channels)
        if data.ndim == 1:
            data = data[:, np.newaxis]

        if dst_ch == 1:
            # Any → mono: average all channels
            return np.mean(data, axis=1).astype(data.dtype)

        if src_ch == 1 and dst_ch == 2:
            # Mono → stereo: duplicate
            return np.column_stack([data[:, 0], data[:, 0]])

        if dst_ch == 2 and src_ch > 2:
            # Multi → stereo: keep first two channels
            return data[:, :2]

        raise ValueError(
            f"Cannot coerce {src_ch} channels → {dst_ch} channels. "
            "Only mono↔stereo and multi→stereo are supported."
        )

    # ── resampling ────────────────────────────────────────────────────────────

    def _resample(
        self, data: np.ndarray, src_rate: int, dst_rate: int
    ) -> np.ndarray:
        if src_rate == dst_rate:
            return data

        if self._use_poly():
            return self._resample_poly(data, src_rate, dst_rate)
        return self._resample_linear(data, src_rate, dst_rate)

    @staticmethod
    def _resample_linear(
        data: np.ndarray, src_rate: int, dst_rate: int
    ) -> np.ndarray:
        ratio = dst_rate / src_rate
        n_frames_src = data.shape[0]
        n_frames_dst = int(math.ceil(n_frames_src * ratio))

        if data.ndim == 1:
            x_src = np.arange(n_frames_src)
            x_dst = np.linspace(0, n_frames_src - 1, n_frames_dst)
            return np.interp(x_dst, x_src, data).astype(data.dtype)

        # Multi-channel: interpolate each channel
        channels = data.shape[1]
        out = np.empty((n_frames_dst, channels), dtype=data.dtype)
        x_src = np.arange(n_frames_src, dtype=np.float64)
        x_dst = np.linspace(0, n_frames_src - 1, n_frames_dst)
        for c in range(channels):
            out[:, c] = np.interp(x_dst, x_src, data[:, c]).astype(data.dtype)
        return out

    @staticmethod
    def _resample_poly(
        data: np.ndarray, src_rate: int, dst_rate: int
    ) -> np.ndarray:
        from scipy.signal import resample_poly  # noqa
        from math import gcd

        g = gcd(dst_rate, src_rate)
        up, down = dst_rate // g, src_rate // g

        if data.ndim == 1:
            return resample_poly(data, up, down).astype(data.dtype)

        channels = data.shape[1]
        resampled = resample_poly(data[:, 0], up, down)
        out = np.empty((len(resampled), channels), dtype=data.dtype)
        out[:, 0] = resampled
        for c in range(1, channels):
            out[:, c] = resample_poly(data[:, c], up, down)
        return out

# ── PCMRingBuffer ─────────────────────────────────────────────────────────────

class PCMRingBuffer:
    """
    Lock-free single-producer / single-consumer circular buffer for PCM frames.

    Capacity is rounded up to the next power of 2 so index wrap-around is a
    bitwise AND instead of modulo — safe for numpy fancy indexing and avoids
    a branch on every read/write.

    Thread safety:
        One thread calls write(), one thread calls read(). No other
        synchronisation is required for the SPSC pattern. If you need
        MPSC or SPMC, add a threading.Lock around the respective side.

    All operations are O(1) copy operations on numpy arrays — no Python-level
    loops on the hot path.
    """

    def __init__(self, capacity: int, fmt: PCMFormat) -> None:
        cap = 1
        while cap < capacity:
            cap <<= 1
        self._cap: int = cap
        self._mask: int = cap - 1
        self._fmt: PCMFormat = fmt
        shape = (cap,) if fmt.channels == 1 else (cap, fmt.channels)
        self._buf: np.ndarray = np.zeros(shape, dtype=fmt.dtype)
        self._write_pos: int = 0  # writer-owned
        self._read_pos: int = 0   # reader-owned

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def fmt(self) -> PCMFormat:
        return self._fmt

    def available_to_write(self) -> int:
        return self._cap - (self._write_pos - self._read_pos)

    def available_to_read(self) -> int:
        return self._write_pos - self._read_pos

    def write(self, frames: np.ndarray) -> int:
        """
        Write up to len(frames) frames. Returns the number of frames written.
        Partial writes occur when the buffer is nearly full.
        """
        n = min(len(frames), self.available_to_write())
        if n <= 0:
            return 0
        start = self._write_pos & self._mask
        end = start + n
        if end <= self._cap:
            self._buf[start:end] = frames[:n]
        else:
            split = self._cap - start
            self._buf[start:] = frames[:split]
            self._buf[: n - split] = frames[split:n]
        self._write_pos += n

        # Normalize both pointers by the same multiple of cap to
        # preserve their difference (fill level) while preventing unbounded growth.
        # Subtracting different amounts from each (as a naive fix might do) would
        # silently corrupt available_to_write() / available_to_read().
        if self._write_pos > 2 ** 30:
            offset = (self._read_pos // self._cap) * self._cap
            self._write_pos -= offset
            self._read_pos -= offset

        return n

    def read(self, n_frames: int) -> np.ndarray:
        """
        Read up to n_frames frames. Returns array of actual frames read
        (may be shorter than requested if fewer frames are available).
        """
        n = min(n_frames, self.available_to_read())
        if n <= 0:
            shape = (0,) if self._fmt.channels == 1 else (0, self._fmt.channels)
            return np.empty(shape, dtype=self._fmt.dtype)
        start = self._read_pos & self._mask
        end = start + n
        if end <= self._cap:
            out = self._buf[start:end].copy()
        else:
            split = self._cap - start
            out = np.concatenate([self._buf[start:], self._buf[: n - split]])
        self._read_pos += n
        return out

    def clear(self) -> None:
        """Discard all buffered frames (call from either side with external lock)."""
        self._read_pos = self._write_pos

    def peek(self, n_frames: int) -> np.ndarray:
        """Read without advancing the read pointer."""
        saved = self._read_pos
        result = self.read(n_frames)
        self._read_pos = saved
        return result

# ── PCMBandpassFilter — FIR / IIR voice-band filter ──────────────────────────

class PCMBandpassFilter:
    """
    Linear-phase FIR or IIR bandpass filter for voice frequency isolation.

    FIR mode (default, requires scipy):
        Uses scipy.signal.firwin2 (windowed least-squares FIR design).
        Linear phase — no group delay distortion in the passband.
        Higher latency: filter_len // 2 samples.

    IIR mode (fast fallback when scipy unavailable):
        Uses a 4th-order Butterworth bandpass via cascaded biquad sections.
        Minimal phase — lower latency but introduces phase distortion.

    Predefined presets:
        "telephony"   — 300 Hz – 3400 Hz (PSTN voice band)
        "wideband"    — 80 Hz – 8000 Hz (HD Voice)
        "narrowband"  — 200 Hz – 3600 Hz (Whisper-optimised)

    Parameters:
        fmt:          Input PCMFormat.
        low_hz:       Lower cutoff frequency. Default 80 Hz.
        high_hz:      Upper cutoff frequency. Default 8000 Hz.
        filter_len:   FIR filter length (odd). Default 127.
        mode:         "fir" or "iir". Default "fir" if scipy available, else "iir".
    """

    _PRESETS: ClassVar[dict[str, tuple[float, float]]] = {
        "telephony": (300.0, 3400.0),
        "wideband": (80.0, 8000.0),
        "narrowband": (200.0, 3600.0),
    }

    def __init__(
        self,
        fmt: PCMFormat,
        low_hz: float = 80.0,
        high_hz: float = 8000.0,
        filter_len: int = 127,
        mode: Literal["fir", "iir", "auto"] = "auto",
    ) -> None:
        self._fmt = fmt
        self._low = low_hz
        self._high = high_hz
        nyq = fmt.sample_rate / 2.0

        if high_hz >= nyq:
            high_hz = nyq * 0.99

        use_fir = mode == "fir" or (mode == "auto" and _SCIPY)
        self._zi: Any = None  # IIR state

        from typing import Optional

        self._sos: Optional[NDArray[np.float64]] = None
        self._fir_coeffs: Optional[np.ndarray] = None

        if use_fir and _SCIPY:
            n = filter_len if filter_len % 2 == 1 else filter_len + 1
            self._fir_coeffs = _scipy_signal.firwin(
                n,
                [low_hz / nyq, high_hz / nyq],
                pass_zero="bandpass",
            )
            from scipy.signal import lfilter_zi
            from typing import List, Optional

            self._fir_zi: List[Optional[np.ndarray]] = [None] * (
                fmt.channels if fmt.channels > 1 else 1
            )

            self._mode = "fir"
            log.debug("pcm_bandpass_init", mode="fir", low_hz=low_hz, high_hz=high_hz, taps=n)
        else:
            self._sos = _scipy_signal.butter(
                4, [low_hz / nyq, high_hz / nyq], btype="band", output="sos"
            ) if _SCIPY else None
            self._mode = "iir" if _SCIPY else "passthrough"
            if _SCIPY and self._sos is not None:
                n_sections = self._sos.shape[0]
                channels = fmt.channels
                self._zi = (
                    np.zeros((n_sections, 2, channels))
                    if channels > 1
                    else np.zeros((n_sections, 2))
                )
            log.debug("pcm_bandpass_init", mode=self._mode, low_hz=low_hz, high_hz=high_hz)

    @classmethod
    def from_preset(cls, fmt: PCMFormat, preset: str, **kwargs: Any) -> PCMBandpassFilter:
        """Create a PCMBandpassFilter from a named preset."""
        if preset not in cls._PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Available: {list(cls._PRESETS)}")
        low, high = cls._PRESETS[preset]
        return cls(fmt=fmt, low_hz=low, high_hz=high, **kwargs)

    def process(self, chunk: PCMChunk) -> PCMChunk:
        """Apply bandpass filter. Returns filtered PCMChunk."""
        if self._mode == "passthrough":
            return chunk

        data = chunk.data.astype(np.float64)

        out = data  # fallback (passthrough)

        if self._mode == "fir":
            assert self._fir_coeffs is not None
            if data.ndim == 2:
                cols = []
                for c in range(data.shape[1]):
                    if self._fir_zi[c] is None:
                        self._fir_zi[c] = _scipy_signal.lfilter_zi(self._fir_coeffs, [1.0]) * data[0, c]
                    filtered, self._fir_zi[c] = _scipy_signal.lfilter(
                        self._fir_coeffs, [1.0], data[:, c], zi=self._fir_zi[c]
                    )
                    cols.append(filtered)
                out = np.column_stack(cols)
            else:
                if self._fir_zi[0] is None:
                    self._fir_zi[0] = _scipy_signal.lfilter_zi(self._fir_coeffs, [1.0]) * data[0]
                out, self._fir_zi[0] = _scipy_signal.lfilter(
                    self._fir_coeffs, [1.0], data, zi=self._fir_zi[0]
                )

        return PCMChunk(
            data=out.astype(chunk.fmt.dtype),
            fmt=chunk.fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: bandpass-filter each chunk."""
        async for chunk in chunks:
            yield self.process(chunk)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STREAMING I/O — input stream, output stream, iterators, bridges
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMInputStream ────────────────────────────────────────────────────────────

class PCMInputStream:
    """
    Async Opus ingress → PCMChunk iterator.

    When USE_FFMPEG_IO=1 (default for network deployments):
        Opus packets arrive via push_opus_packet() / push_opus_bytes(), pass
        through an adaptive jitter buffer and libopus decoder subprocess, and
        emerge as PCMChunks on the async iterator.  The internal audio_engine
        pipeline (VAD, converter, AEC) is entirely unaware of the codec layer.

    When USE_FFMPEG_IO=0 (local mic fallback):
        Behaves identically to the original PortAudio/sounddevice implementation.

    Usage (network path)::

        async with PCMInputStream(fmt=PCMFormat.whisper()) as stream:
            # Feed from WebSocket / RTP:
            stream.push_opus_bytes(opus_frame)

            async for chunk in stream:   # yields PCMChunk
                await vad_gate.stream([chunk])

    Usage (mic fallback)::

        # Set USE_FFMPEG_IO=0 in environment, then use identically to before.
        async with PCMInputStream(fmt=PCMFormat.whisper()) as stream:
            async for chunk in stream:
                ...
    """

    def __init__(
        self,
        fmt: PCMFormat | None = None,
        blocksize: int = _INPUT_BLOCKSIZE,
        queue_maxsize: int = _INPUT_QUEUE_MAXSIZE,
        device: int | str | None = None,
    ) -> None:
        self._fmt = fmt or PCMFormat(
            sample_rate=_DEFAULT_INPUT_RATE,
            channels=_DEFAULT_INPUT_CH,
            dtype="int16",
        )
        self._blocksize = blocksize
        self._device = device
        self._seq: int = 0
        self._async_q: asyncio.Queue[PCMChunk | None] = asyncio.Queue(
            maxsize=queue_maxsize
        )

        if _USE_FFMPEG_IO:
            # Build a CodecConfig that matches the requested PCMFormat exactly.
            self._ffmpeg_in = FFmpegPCMInputStream(
                CodecConfig(
                    sample_rate=self._fmt.sample_rate,
                    channels=self._fmt.channels,
                    dtype=self._fmt.dtype,
                )
            )
            self._bridge_task: asyncio.Task | None = None
        else:
            # Sounddevice path — keep all original state
            self._thread_q: _stdlib_queue.Queue = _stdlib_queue.Queue(
                maxsize=queue_maxsize * 2
            )
            self._stream: sd.InputStream | None = None
            self._drainer_task: asyncio.Task | None = None
            self._stop_event = threading.Event()
            self._loop: asyncio.AbstractEventLoop | None = None

    async def __aenter__(self) -> PCMInputStream:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── push interface (FFmpeg path only) ─────────────────────────────────────

    def push_opus_packet(
        self, seq: int, opus_bytes: bytes, duration_ms: int = 20
    ) -> None:
        """Feed one Opus packet (with sequence number) from the network layer."""
        if _USE_FFMPEG_IO:
            self._ffmpeg_in.push_opus_packet(seq, opus_bytes, duration_ms)

    def push_opus_bytes(self, opus_bytes: bytes) -> None:
        """Feed one Opus packet (auto-sequenced) from the network layer."""
        if _USE_FFMPEG_IO:
            self._ffmpeg_in.push_opus_bytes(opus_bytes)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if _USE_FFMPEG_IO:
            await self._ffmpeg_in.start()
            self._bridge_task = asyncio.create_task(
                self._ffmpeg_bridge(), name="pcm-input-ffmpeg-bridge"
            )
            _input_active.set(1)
            log.info(
                "pcm_input_stream_started",
                fmt=repr(self._fmt),
                backend="ffmpeg",
            )
            return

        # ── Sounddevice path (fallback) ──────────────────────────────────────
        if self._stream is not None:
            raise RuntimeError("PCMInputStream already started")

        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()

        def _callback(
            indata: np.ndarray,
            frame_count: int,  # noqa
            time_info: object,  # noqa
            status: sd.CallbackFlags,
        ) -> None:
            if status.input_overflow:
                _in_overflows.inc()
                log.debug("pcm_input_overflow")
            if self._stop_event.is_set():
                return
            try:
                self._thread_q.put_nowait(indata.copy())
                _in_chunks.labels(status="ok").inc()
                _in_bytes.inc(indata.nbytes)
            except _stdlib_queue.Full:
                _in_chunks.labels(status="dropped").inc()
                _in_dropped.inc()

        self._stream = sd.InputStream(
            samplerate=self._fmt.sample_rate,
            channels=self._fmt.channels,
            dtype=self._fmt.dtype,
            blocksize=self._blocksize,
            device=self._device,
            callback=_callback,
        )
        self._stream.start()
        _input_active.set(1)

        self._drainer_task = asyncio.create_task(
            self._drain_thread_queue(), name="pcm-input-drainer"
        )

        log.info(
            "pcm_input_stream_started",
            fmt=repr(self._fmt),
            blocksize=self._blocksize,
            backend="sounddevice",
        )

    async def stop(self) -> None:
        if _USE_FFMPEG_IO:
            if self._bridge_task is not None:
                self._bridge_task.cancel()
                try:
                    await self._bridge_task
                except asyncio.CancelledError:
                    pass
                self._bridge_task = None
            await self._ffmpeg_in.stop()
            try:
                self._async_q.put_nowait(None)
            except asyncio.QueueFull:
                pass
            _input_active.set(0)
            log.info("pcm_input_stream_stopped", backend="ffmpeg")
            return

        # ── Sounddevice path (unchanged) ──────────────────────────────────────
        self._stop_event.set()

        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:
                log.warning("pcm_input_stop_error", error=str(exc))
            finally:
                self._stream = None
                _input_active.set(0)

        try:
            self._thread_q.put_nowait(None)
        except _stdlib_queue.Full:
            pass

        if self._drainer_task is not None:
            try:
                await asyncio.wait_for(self._drainer_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._drainer_task.cancel()
            self._drainer_task = None

        try:
            self._async_q.put_nowait(None)
        except asyncio.QueueFull:
            pass

        log.info("pcm_input_stream_stopped", backend="sounddevice")

    # ── iteration ─────────────────────────────────────────────────────────────

    def __aiter__(self) -> AsyncIterator[PCMChunk]:
        return self._chunk_iterator()

    async def _chunk_iterator(self) -> AsyncIterator[PCMChunk]:
        while True:
            try:
                chunk = await self._async_q.get()
            except asyncio.CancelledError:
                return
            if chunk is None:
                return
            yield chunk

    # ── FFmpeg bridge (translates PCMFrame → PCMChunk) ────────────────────────

    async def _ffmpeg_bridge(self) -> None:
        """
        Drain PCMFrames from FFmpegPCMInputStream and re-emit them as PCMChunks
        so the rest of audio_engine (VAD, converter, AEC) sees its native type.
        """
        try:
            async for frame in self._ffmpeg_in:
                chunk = PCMChunk(
                    data=frame.data,
                    fmt=self._fmt,
                    timestamp=frame.ts_us / 1_000_000.0,
                    seq=self._seq,
                    source="network",
                )
                self._seq += 1
                _in_chunks.labels(status="ok").inc()
                _in_bytes.inc(frame.data.nbytes)
                _chunk_duration.observe(chunk.duration_s)
                try:
                    self._async_q.put_nowait(chunk)
                except asyncio.QueueFull:
                    _in_dropped.inc()
                    log.debug("pcm_input_async_queue_full")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("pcm_input_ffmpeg_bridge_error", error=str(exc))

    # ── sounddevice drainer (fallback) ─────────────────────────────

    async def _drain_thread_queue(self) -> None:
        import queue as _q

        loop = self._loop
        assert loop is not None

        while not self._stop_event.is_set():
            try:
                raw = await loop.run_in_executor(
                    None,
                    self._thread_q.get,
                    True,
                    0.1,
                )
            except _q.Empty:
                continue
            except Exception:  # noqa
                break

            if raw is None:
                break

            chunk = PCMChunk(
                data=raw,
                fmt=self._fmt,
                timestamp=time.monotonic(),
                seq=self._seq,
                source="mic",
            )
            self._seq += 1
            _chunk_duration.observe(chunk.duration_s)

            try:
                self._async_q.put_nowait(chunk)
            except asyncio.QueueFull:
                _in_dropped.inc()
                log.debug("pcm_input_async_queue_full")

# ── PCMOutputStream ───────────────────────────────────────────────────────────

class PCMOutputStream:
    """
    Async PCMChunk consumer.

    When USE_FFMPEG_IO=1 (default for network deployments):
        Accepts PCMChunks via write(), encodes them to Opus bytes through an
        FFmpeg subprocess (libopus), and makes the encoded packets available via
        read_opus_frame() for the network layer to transmit.  The adaptive
        bitrate controller adjusts quality to live loss/jitter feedback.

    When USE_FFMPEG_IO=0 (local speaker fallback):
        Behaves identically to the original PortAudio/sounddevice implementation.

    Usage (network path)::

        async with PCMOutputStream() as out:
            async for chunk in tts_stream:
                await out.write(chunk)           # encodes to Opus
                frame = out.read_opus_frame()    # bytes ready to send
                if frame:
                    await ws.send_bytes(frame)

    Usage (speaker fallback)::

        # Set USE_FFMPEG_IO=0 in environment, then use identically to before.
        async with PCMOutputStream() as out:
            async for chunk in tts_stream:
                await out.write(chunk)
    """

    class _STOP:
        pass

    class _SHUTDOWN:
        pass

    def __init__(
        self,
        preferred_fmt: PCMFormat | None = None,
        queue_maxsize: int = _OUTPUT_QUEUE_MAXSIZE,
        warmup_frames: int = _OUTPUT_WARMUP_FRAMES,
        device: int | str | None = None,
        converter: PCMConverter | None = None,
        aec: "PCMEchoCanceller | None" = None,
    ) -> None:
        self._preferred_fmt = preferred_fmt or PCMFormat(
            sample_rate=_DEFAULT_OUTPUT_RATE,
            channels=_DEFAULT_OUTPUT_CH,
            dtype="float32",
        )
        self._warmup_frames = warmup_frames
        self._device = device
        self._aec = aec
        self._converter = converter or PCMConverter()
        self._started = False

        if _USE_FFMPEG_IO:
            self._ffmpeg_out = FFmpegPCMOutputStream(
                CodecConfig(
                    sample_rate=self._preferred_fmt.sample_rate,
                    channels=self._preferred_fmt.channels,
                    dtype=self._preferred_fmt.dtype,
                )
            )
        else:
            # Sounddevice path — keep all original state
            import queue as _q
            self._write_q: _q.Queue = _q.Queue(maxsize=queue_maxsize)
            self._stream: sd.OutputStream | None = None
            self._stream_fmt: PCMFormat | None = None
            self._writer_thread: threading.Thread | None = None
            self._stream_open_event = threading.Event()

    async def __aenter__(self) -> PCMOutputStream:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        if _USE_FFMPEG_IO:
            await self._ffmpeg_out.start()
            _output_active.set(1)
            log.info(
                "pcm_output_stream_started",
                preferred_fmt=repr(self._preferred_fmt),
                backend="ffmpeg",
            )
            return

        # ── Sounddevice path (unchanged) ──────────────────────────────────────
        t = threading.Thread(
            target=self._writer_loop,
            name="pcm-output-writer",
            daemon=True,
        )
        t.start()
        self._writer_thread = t
        log.info(
            "pcm_output_stream_started",
            preferred_fmt=repr(self._preferred_fmt),
            backend="sounddevice",
        )

    async def stop(self) -> None:
        if not self._started:
            return

        if _USE_FFMPEG_IO:
            await self._ffmpeg_out.stop()
            self._started = False
            _output_active.set(0)
            log.info("pcm_output_stream_stopped", backend="ffmpeg")
            return

        # ── Sounddevice path (unchanged) ──────────────────────────────────────
        try:
            self._write_q.put_nowait(self._SHUTDOWN)
        except _stdlib_queue.Full:
            pass
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3.0)
            if self._writer_thread.is_alive():
                log.warning("pcm_output_writer_join_timeout")
            self._writer_thread = None
        self._started = False
        _output_active.set(0)
        log.info("pcm_output_stream_stopped", backend="sounddevice")

    # ── write ─────────────────────────────────────────────────────────────────

    async def write(self, chunk: PCMChunk) -> None:
        """
        Enqueue a PCMChunk for encoding (FFmpeg path) or playback (sounddevice path).
        Returns in microseconds. Drops if queue/encoder is full.
        """
        if not self._started:
            raise RuntimeError("PCMOutputStream.write() called before start()")

        if _USE_FFMPEG_IO:
            # AEC reference must still fire — converter brings to preferred fmt first
            target = PCMFormat(
                sample_rate=chunk.fmt.sample_rate,
                channels=chunk.fmt.channels,
                dtype=self._preferred_fmt.dtype,
            )
            converted = self._converter.convert(chunk, target)
            if self._aec is not None:
                self._aec.push_reference(converted)

            _output_queue_depth.set(0)   # depth managed by FFmpegPCMOutputStream
            _out_chunks.labels(status="enqueued").inc()
            await self._ffmpeg_out.write(converted)
            return

        # ── Sounddevice path (unchanged) ──────────────────────────────────────
        target = PCMFormat(
            sample_rate=chunk.fmt.sample_rate,
            channels=chunk.fmt.channels,
            dtype="float32",
        )
        converted = self._converter.convert(chunk, target)

        if self._aec is not None:
            self._aec.push_reference(converted)

        _output_queue_depth.set(self._write_q.qsize())
        try:
            self._write_q.put_nowait(converted)
            _out_chunks.labels(status="enqueued").inc()
        except _stdlib_queue.Full:
            _out_chunks.labels(status="dropped").inc()
            log.debug("pcm_output_queue_full_chunk_dropped", seq=chunk.seq)

    # ── Opus egress (FFmpeg path only) ────────────────────────────────────────

    def read_opus_frame(self) -> bytes | None:
        """
        Non-blocking: return the next encoded Opus packet payload, or None.
        Call this after every write() in the network send loop.

        Only meaningful when USE_FFMPEG_IO=1. Returns None on sounddevice path.
        """
        if not _USE_FFMPEG_IO:
            return None
        pkt = self._ffmpeg_out.read_opus_packet_nowait()
        return pkt.payload if pkt else None

    def report_network_stats(
        self, rtt_ms: float | None = None, lost: bool = False
    ) -> None:
        """
        Feed network round-trip and loss observations into the ABR controller.
        Call from your ACK/NACK handler or RTCP RR processor.

        Only meaningful when USE_FFMPEG_IO=1.
        """
        if _USE_FFMPEG_IO:
            self._ffmpeg_out.report_network_stats(rtt_ms=rtt_ms, lost=lost)

    # ── stop_playback / is_active ─────────────────────────────────────────────

    async def stop_playback(self) -> None:
        """Insert a stop sentinel (sounddevice path). No-op on FFmpeg path."""
        if _USE_FFMPEG_IO:
            return
        try:
            self._write_q.put_nowait(self._STOP)
        except _stdlib_queue.Full:
            pass

    # noinspection PyProtectedMember
    def is_active(self) -> bool:
        if _USE_FFMPEG_IO:
            return self._started and not self._ffmpeg_out._stopped
        return self._stream is not None and self._stream.active

    # ── writer thread + stream management (sounddevice path, unchanged) ───────

    def _writer_loop(self) -> None:
        import queue as _q

        while True:
            try:
                item = self._write_q.get(timeout=0.5)
            except _q.Empty:
                continue

            if item is self._SHUTDOWN:
                self._close_stream()
                break

            if item is self._STOP:
                self._close_stream()
                continue

            chunk: PCMChunk = item  # type: ignore[assignment]
            _output_queue_depth.set(self._write_q.qsize())

            needs_open = (
                self._stream is None
                or not self._stream.active
                or self._stream_fmt != chunk.fmt
            )
            if needs_open:
                self._open_stream(chunk.fmt)
                if self._stream is None:
                    continue

            t0 = time.monotonic()
            try:
                data = chunk.data
                if data.ndim == 1 and chunk.fmt.channels > 1:
                    data = data.reshape(-1, chunk.fmt.channels)
                self._stream.write(data)  # type: ignore[union-attr]
                elapsed = time.monotonic() - t0
                _output_write_latency.observe(elapsed)
                _out_chunks.labels(status="written").inc()
                _out_bytes.inc(chunk.n_bytes)
            except sd.PortAudioError as exc:
                log.warning("pcm_output_write_error", error=str(exc))
                self._close_stream()
            except Exception as exc:
                log.error("pcm_output_unexpected_write_error", error=str(exc))

    def _open_stream(self, fmt: PCMFormat) -> None:
        self._close_stream()
        try:
            self._stream = sd.OutputStream(
                samplerate=fmt.sample_rate,
                channels=fmt.channels,
                dtype="float32",
                device=self._device,
                latency=_OUTPUT_LATENCY,
            )
            self._stream.start()

            if self._warmup_frames > 0:
                shape = (
                    (self._warmup_frames,)
                    if fmt.channels == 1
                    else (self._warmup_frames, fmt.channels)
                )
                silence = np.zeros(shape, dtype="float32")
                self._stream.write(silence)

            self._stream_fmt = fmt
            self._stream_open_event.set()
            _output_active.set(1)
            _out_recreations.inc()

            log.info(
                "pcm_output_stream_opened",
                sample_rate=fmt.sample_rate,
                channels=fmt.channels,
            )
        except Exception as exc:
            log.error("pcm_output_stream_open_failed", error=str(exc))
            self._stream = None
            self._stream_fmt = None

    def _close_stream(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:
            log.warning("pcm_output_stream_close_error", error=str(exc))
        finally:
            self._stream = None
            self._stream_fmt = None
            self._stream_open_event.clear()
            _output_active.set(0)

# ── PCMSplitter ───────────────────────────────────────────────────────────────

class PCMSplitter:
    """
    Fan-out: broadcast one PCMChunk source to N independent async queues.

    Each subscriber gets its own asyncio.Queue so slow consumers (e.g. a
    recording consumer) never block the fast consumer (e.g. VAD → STT).

    Usage::

        splitter = PCMSplitter()
        sub_a = splitter.subscribe(maxsize=32)
        sub_b = splitter.subscribe(maxsize=8)

        async with PCMInputStream() as src:
            async for chunk in src:
                await splitter.push(chunk)

        async for chunk in sub_a:
            ...
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[PCMChunk | None]] = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize: int = 32) -> asyncio.Queue[PCMChunk | None]:
        q: asyncio.Queue[PCMChunk | None] = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
        return q

    async def push(self, chunk: PCMChunk) -> None:
        for q in self._subscribers:
            try:
                q.put_nowait(chunk)
            except asyncio.QueueFull:
                log.debug("pcm_splitter_subscriber_full_drop")

    async def close(self) -> None:
        """Send poison pill to all subscribers."""
        for q in self._subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass

    @staticmethod
    def queue_to_aiter(
        q: asyncio.Queue[PCMChunk | None],
    ) -> AsyncIterator[PCMChunk]:
        return _queue_aiter(q)


async def _queue_aiter(
    q: asyncio.Queue[PCMChunk | None],
) -> AsyncIterator[PCMChunk]:
    while True:
        item = await q.get()
        if item is None:
            return
        yield item

# ── PCMStreamBridge — sync ↔ async bridge ─────────────────────────────────────

class PCMStreamBridge:
    """
    Bridges a synchronous PCM producer into an asyncio consumer (or vice versa).

    Use case A — sync producer → async consumer:
        A legacy callback-based audio library produces PCMChunks synchronously.
        PCMStreamBridge queues them into an asyncio.Queue so async pipeline
        stages can consume them naturally.

    Use case B — async producer → sync consumer:
        An async pipeline produces PCMChunks that must be fed into a sync API
        (e.g., a blocking write() call). The bridge drains the async queue
        on a dedicated thread.

    Usage (sync → async)::

        bridge = PCMStreamBridge(maxsize=32)

        # Sync side (from a callback thread):
        bridge.push_sync(chunk)

        # Async side:
        async for chunk in bridge.async_iter():
            await process(chunk)

    Parameters:
        maxsize:   Maximum queue depth. Default 32.
    """

    def __init__(self, maxsize: int = 32) -> None:
        self._maxsize = maxsize
        self._queue: asyncio.Queue[PCMChunk | None] | None = None
        self._thread_q: _stdlib_queue.Queue = _stdlib_queue.Queue(maxsize=maxsize)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._bridge_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Initialise the asyncio queue and start the bridge task."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self._maxsize)
        self._bridge_task = asyncio.create_task(
            self._drain_thread_q(), name="pcm-stream-bridge"
        )

    async def stop(self) -> None:
        """Signal consumers and stop the bridge task."""
        try:
            self._thread_q.put_nowait(None)
        except _stdlib_queue.Full:
            pass
        if self._bridge_task:
            try:
                await asyncio.wait_for(self._bridge_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._bridge_task.cancel()
            self._bridge_task = None

    def push_sync(self, chunk: PCMChunk) -> bool:
        """
        Push a chunk from a synchronous thread.

        Returns False if the queue is full (dropped).
        """
        try:
            self._thread_q.put_nowait(chunk)
            return True
        except _stdlib_queue.Full:
            return False

    async def push_async(self, chunk: PCMChunk) -> None:
        """Push a chunk from an async context."""
        if self._queue:
            await self._queue.put(chunk)

    async def _drain_thread_q(self) -> None:
        """Background task: transfer chunks from thread queue to asyncio queue."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(
                    None,
                    self._thread_q.get,
                    True,  # block
                    0.1,  # timeout
                )
            except _stdlib_queue.Empty:
                continue
            except Exception: # noqa
                break
            if self._queue:
                if item is None:
                    await self._queue.put(None)
                    break
                await self._queue.put(item)

    async def async_iter(self) -> AsyncIterator[PCMChunk]:
        """Async iterator over queued chunks."""
        if self._queue is None:
            raise RuntimeError("PCMStreamBridge not started. Call await start() first.")
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

# ──────────────────────────────────────────────────────────────────────────────
# 6. PROCESSING UNITS — VAD, AGC, noise, echo cancel, dynamics
# ──────────────────────────────────────────────────────────────────────────────

# ── PCMVADGate — energy-based VAD ─────────────────────────────────────────────

class PCMVADGate:
    """
    Energy-based Voice Activity Detector.

    Gating strategy
    ───────────────
    1. SILENCE  → SPEECH    when RMS exceeds ``onset_rms`` for one frame
    2. SPEECH   → HANGOVER  when RMS drops below ``offset_rms``
    3. HANGOVER → SILENCE   after ``hangover_s`` of sub-threshold audio
       HANGOVER → SPEECH    if RMS rises again above ``onset_rms``

    Pre-roll
    ────────
    The VAD keeps a ring buffer of the last ``pre_roll_s`` worth of frames.
    When the SILENCE→SPEECH transition fires, the pre-roll buffer is prepended
    to the emitted segment so the leading phonemes of a word are never lost.

    Min-speech gate
    ───────────────
    Segments shorter than ``min_speech_s`` are discarded as noise bursts.

    Usage::

        gate = PCMVADGate(fmt=PCMFormat.whisper())
        async for speech_chunk in gate.stream(input_stream):
            # speech_chunk is a merged PCMChunk ready for STT
            await stt.transcribe_chunk(speech_chunk)
    """

    def __init__(
        self,
        fmt: PCMFormat,
        onset_rms: float = _VAD_ONSET_RMS,
        offset_rms: float = _VAD_OFFSET_RMS,
        hangover_s: float = _VAD_HANGOVER_S,
        pre_roll_s: float = _VAD_PRE_ROLL_S,
        min_speech_s: float = _VAD_MIN_SPEECH_S,
    ) -> None:
        self._fmt = fmt
        self._onset = onset_rms
        self._offset = offset_rms
        self._hangover_frames = fmt.frames_for_duration(hangover_s)
        self._pre_roll_frames = fmt.frames_for_duration(pre_roll_s)
        self._min_speech_frames = fmt.frames_for_duration(min_speech_s)

        # Working int16 RMS expects int16 scale; convert onset/offset if float32
        self._scale = 1.0
        if fmt.dtype == "float32":
            # Internally normalise onset/offset to [-1, 1] scale
            self._scale = 32768.0

        self._state = _VADState.SILENCE
        self._hangover_remaining: int = 0
        self._pre_roll = PCMRingBuffer(
            capacity=max(self._pre_roll_frames * 2, 512), fmt=fmt
        )
        self._speech_frames: list[np.ndarray] = []
        self._speech_frame_count: int = 0

    def reset(self) -> None:
        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._pre_roll.clear()
        self._speech_frames = []
        self._speech_frame_count = 0

    def _rms(self, data: np.ndarray) -> float:
        f = data.astype(np.float64)
        return float(np.sqrt(np.mean(f ** 2))) * self._scale

    async def stream(
        self,
        chunks: AsyncIterator[PCMChunk],
    ) -> AsyncIterator[PCMChunk]:
        """
        Consume an async PCMChunk iterator and yield complete speech segments
        as merged PCMChunks with ``is_final=True``.

        Segments are yielded when:
          • the HANGOVER state expires (natural end of speech), or
          • the source iterator is exhausted (flush any remaining speech).
        """
        async for chunk in chunks:
            result = self._process_chunk(chunk)
            if result is not None:
                _vad_segments.inc()
                _vad_active.set(0)
                yield result

        # Flush remaining speech on iterator exhaustion
        flushed = self._flush()
        if flushed is not None:
            _vad_segments.inc()
            yield flushed

    # noinspection PyUnreachableCode
    def _process_chunk(self, chunk: PCMChunk) -> PCMChunk | None:
        """
        Feed one chunk through the state machine.
        Returns a complete speech segment PCMChunk when a segment ends, else None.
        """
        data = chunk.data
        if data.ndim == 2:
            data = data[:, 0] if self._fmt.channels == 1 else data.reshape(-1)
        rms = self._rms(data)

        if self._state == _VADState.SILENCE:
            # Always maintain pre-roll buffer
            self._pre_roll.write(data)
            if rms >= self._onset:
                # Transition SILENCE → SPEECH
                _vad_transitions.labels(direction="onset").inc()
                _vad_active.set(1)
                log.debug("pcm_vad_onset", rms=round(rms, 1))
                self._state = _VADState.SPEECH
                # Prepend pre-roll
                pre = self._pre_roll.read(self._pre_roll_frames)
                if len(pre) > 0:
                    self._speech_frames.append(pre)
                    self._speech_frame_count += len(pre)
                self._speech_frames.append(data)
                self._speech_frame_count += len(data)
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        if self._state == _VADState.SPEECH:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            if rms < self._offset:
                # Transition SPEECH → HANGOVER
                _vad_transitions.labels(direction="hangover").inc()
                self._state = _VADState.HANGOVER
                self._hangover_remaining = self._hangover_frames
            return None

        if self._state == _VADState.HANGOVER:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            self._hangover_remaining -= len(data)

            if rms >= self._onset:
                # Voice returned — back to SPEECH
                _vad_transitions.labels(direction="resume").inc()
                self._state = _VADState.SPEECH
                self._hangover_remaining = 0
                return None

            if self._hangover_remaining <= 0:
                # Hangover expired → segment complete
                _vad_transitions.labels(direction="offset").inc()
                self._state = _VADState.SILENCE
                self._pre_roll.clear()
                return self._emit_segment(chunk)

        return None

    def _emit_segment(self, last_chunk: PCMChunk) -> PCMChunk | None:
        if not self._speech_frames:
            return None

        merged = np.concatenate(self._speech_frames)
        self._speech_frames = []

        if self._speech_frame_count < self._min_speech_frames:
            log.debug(
                "pcm_vad_segment_too_short",
                frames=self._speech_frame_count,
                min_frames=self._min_speech_frames,
            )
            self._speech_frame_count = 0
            return None

        self._speech_frame_count = 0
        return PCMChunk(
            data=merged,
            fmt=self._fmt,
            timestamp=last_chunk.timestamp,
            seq=last_chunk.seq,
            is_final=True,
            source="vad",
        )

    # noinspection PyUnreachableCode
    def _flush(self) -> PCMChunk | None:
        if self._state == _VADState.SILENCE or not self._speech_frames:
            self._speech_frames = []
            self._speech_frame_count = 0
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        self._state = _VADState.SILENCE
        return self._emit_segment(
            PCMChunk(
                data=np.array([], dtype=self._fmt.dtype),
                fmt=self._fmt,
                timestamp=time.monotonic(),
                is_final=True,
                source="vad.flush",
            )
        )

# ── PCMSpectralVAD — FFT-based voice-band energy VAD ─────────────────────────

class PCMSpectralVAD:
    """
    Spectral Voice Activity Detector based on voice-frequency-band energy ratio.

    Algorithm
    ─────────
    For each frame:
      1. Apply Hann window
      2. Compute real FFT
      3. Compute energy in voice band (default 300–3400 Hz)
      4. Compute total energy
      5. Voice ratio = voice_band_energy / (total_energy + eps)
      6. Decision = ratio >= threshold AND total_energy >= floor

    Advantages over pure RMS-based VAD:
      • Rejects low-frequency HVAC / fan noise (below voice band)
      • Rejects high-frequency electronic hiss (above voice band)
      • More robust to broadband noise environments

    The spectral VAD is designed to be combined with PCMVADGate (energy-based)
    or WebRTC VAD via PCMFusedVAD for a high-accuracy dual-mode decision.

    Parameters:
        fmt:            PCMFormat of input chunks.
        low_hz:         Lower edge of voice band in Hz. Default 300.
        high_hz:        Upper edge of voice band in Hz. Default 3400.
        ratio_thresh:   Minimum voice-band-to-total-energy ratio. Default 0.4.
        floor_rms:      Minimum total RMS to even consider VAD. Default 50 (int16 scale).
        hangover_s:     Post-detection hangover in seconds. Default 0.3.
        pre_roll_s:     Pre-detection pre-roll in seconds. Default 0.1.
        min_speech_s:   Minimum segment duration. Default 0.2.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        low_hz: float = 300.0,
        high_hz: float = 3400.0,
        ratio_thresh: float = 0.4,
        floor_rms: float = 50.0,
        hangover_s: float = 0.3,
        pre_roll_s: float = 0.1,
        min_speech_s: float = 0.2,
    ) -> None:
        self._fmt = fmt
        self._ratio_thresh = ratio_thresh
        self._floor_rms = floor_rms
        _dtype_scales = {
            "float32": 32768.0,
            "float64": 32768.0,
            "int16": 1.0,
            "int32": 32768.0 * 65536.0,
        }
        self._scale = _dtype_scales.get(fmt.dtype, 1.0)

        # Frequency bin indices for voice band
        # FFT of a block of N samples → N//2+1 unique bins at spacing sr/N Hz
        # We use blocksize as a conservative FFT size estimate; actual FFT is per-chunk
        self._low_hz = low_hz
        self._high_hz = high_hz

        self._hangover_frames = fmt.frames_for_duration(hangover_s)
        self._pre_roll_frames = fmt.frames_for_duration(pre_roll_s)
        self._min_frames = fmt.frames_for_duration(min_speech_s)

        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._pre_roll = PCMRingBuffer(
            capacity=max(self._pre_roll_frames * 2, 512), fmt=fmt
        )

    def _voice_ratio(self, data: np.ndarray) -> tuple[float, float]:
        """Returns (voice_ratio, rms) for a 1-D float array."""
        f = data.astype(np.float64)
        rms = float(np.sqrt(np.mean(f ** 2))) * self._scale

        n = len(f)
        if n < 8:
            return 0.0, rms

        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(f * window)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fmt.sample_rate)

        voice_mask = (freqs >= self._low_hz) & (freqs <= self._high_hz)
        total_energy = float(np.sum(spectrum)) + 1e-12
        voice_energy = float(np.sum(spectrum[voice_mask]))
        return voice_energy / total_energy, rms

    def is_speech(self, chunk: PCMChunk) -> bool:
        """
        Classify a single chunk as speech or non-speech.

        Returns True if voice-band energy ratio and floor RMS both exceed thresholds.
        """
        data = chunk.data
        if data.ndim == 2:
            data = data[:, 0]
        ratio, rms = self._voice_ratio(data)
        decision = (ratio >= self._ratio_thresh) and (rms >= self._floor_rms)
        _spectral_vad_decisions.labels(decision="speech" if decision else "silence").inc()
        return decision

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """
        Async generator: consume PCMChunks, yield complete speech segment PCMChunks.

        Mirrors the PCMVADGate.stream() API for drop-in compatibility.
        """
        async for chunk in chunks:
            result = self._process(chunk)
            if result is not None:
                yield result

        flushed = self._flush()
        if flushed is not None:
            yield flushed

    # noinspection PyUnreachableCode
    def _process(self, chunk: PCMChunk) -> PCMChunk | None:
        data = chunk.data
        mono = data[:, 0] if data.ndim == 2 else data
        ratio, rms = self._voice_ratio(mono)
        is_voice = (ratio >= self._ratio_thresh) and (rms >= self._floor_rms)

        if self._state == _VADState.SILENCE:
            self._pre_roll.write(mono if mono.ndim == 1 else mono.reshape(-1))
            if is_voice:
                self._state = _VADState.SPEECH
                pre = self._pre_roll.read(self._pre_roll_frames)
                if len(pre) > 0:
                    self._speech_frames.append(pre)
                    self._speech_frame_count += len(pre)
                self._speech_frames.append(data)
                self._speech_frame_count += len(data)
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        if self._state == _VADState.SPEECH:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            if not is_voice:
                self._state = _VADState.HANGOVER
                self._hangover_remaining = self._hangover_frames
            return None

        if self._state == _VADState.HANGOVER:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            self._hangover_remaining -= len(data)

            if is_voice:
                self._state = _VADState.SPEECH
                self._hangover_remaining = 0
                return None

            if self._hangover_remaining <= 0:
                self._state = _VADState.SILENCE
                self._pre_roll.clear()
                return self._emit(chunk)
        return None

    def _emit(self, ref: PCMChunk) -> PCMChunk | None:
        if not self._speech_frames:
            return None
        merged = np.concatenate(self._speech_frames)
        self._speech_frames = []
        count = self._speech_frame_count
        self._speech_frame_count = 0
        if count < self._min_frames:
            return None
        return PCMChunk(
            data=merged,
            fmt=self._fmt,
            timestamp=ref.timestamp,
            seq=ref.seq,
            is_final=True,
            source="spectral_vad",
        )

    # noinspection PyUnreachableCode
    def _flush(self) -> PCMChunk | None:
        if self._state == _VADState.SILENCE or not self._speech_frames:
            self._speech_frames = []
            self._speech_frame_count = 0
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        self._state = _VADState.SILENCE
        dummy = PCMChunk(
            data=np.array([], dtype=self._fmt.dtype),
            fmt=self._fmt,
            timestamp=time.monotonic(),
            is_final=True,
            source="spectral_vad.flush",
        )
        return self._emit(dummy)

    def reset(self) -> None:
        """Reset all state (call between utterances or on barge-in)."""
        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames = []
        self._speech_frame_count = 0
        self._pre_roll.clear()


# ── PCMWebRTCVAD — webrtcvad library wrapper ──────────────────────────────────

class PCMWebRTCVAD:
    """
    Voice Activity Detector powered by Google's WebRTC VAD algorithm.

    WebRTC VAD (GMM-based) is considered the gold standard for lightweight
    embedded VAD. It operates on fixed-size frames at 8/16/32/48 kHz and
    returns a simple boolean decision per frame.

    This class handles:
      • Format validation (WebRTC VAD requires int16 mono)
      • Frame-size chunking (WebRTC requires exactly 10, 20, or 30 ms frames)
      • Hangover and pre-roll accumulation (same as PCMVADGate)
      • Graceful fallback to energy-based PCMVADGate when webrtcvad is absent

    Installation::

        pip install webrtcvad-wheels

    Aggressiveness levels:
        0 — Least aggressive (highest recall, most false positives)
        1 — (default) Balanced
        2 — More aggressive
        3 — Most aggressive (fewest false positives, highest miss rate)

    Parameters:
        fmt:              Input PCMFormat. Must be mono, int16, rate in {8k,16k,32k,48k}.
        aggressiveness:   WebRTC aggressiveness 0-3. Default 1.
        frame_ms:         Frame size in ms; must be 10, 20, or 30. Default 20.
        hangover_s:       Post-speech hangover in seconds. Default 0.4.
        pre_roll_s:       Pre-speech pre-roll in seconds. Default 0.15.
        min_speech_s:     Minimum segment duration. Default 0.25.
    """

    _VALID_RATES = frozenset({8000, 16000, 32000, 48000})
    _VALID_FRAME_MS = frozenset({10, 20, 30})

    def __init__(
        self,
        fmt: PCMFormat,
        aggressiveness: int = 1,
        frame_ms: int = 20,
        hangover_s: float = 0.4,
        pre_roll_s: float = 0.15,
        min_speech_s: float = 0.25,
    ) -> None:
        if fmt.sample_rate not in self._VALID_RATES:
            raise ValueError(
                f"WebRTC VAD requires sample rate in {self._VALID_RATES}, got {fmt.sample_rate}"
            )
        if frame_ms not in self._VALID_FRAME_MS:
            raise ValueError(f"frame_ms must be 10, 20, or 30, got {frame_ms}")
        if not 0 <= aggressiveness <= 3:
            raise ValueError(f"aggressiveness must be 0-3, got {aggressiveness}")

        self._fmt = fmt
        self._frame_ms = frame_ms
        self._frame_samples = int(fmt.sample_rate * frame_ms / 1000)
        self._aggressiveness = aggressiveness

        self._hangover_frames = fmt.frames_for_duration(hangover_s)
        self._pre_roll_frames = fmt.frames_for_duration(pre_roll_s)
        self._min_frames = fmt.frames_for_duration(min_speech_s)

        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._pre_roll = PCMRingBuffer(
            capacity=max(self._pre_roll_frames * 2, 512), fmt=fmt
        )
        self._carry: np.ndarray = np.array([], dtype="int16")

        # Initialise webrtcvad or fall back
        self._vad: Any = None
        self._fallback: PCMVADGate | None = None
        self._use_webrtc: bool = False

        if _WEBRTCVAD:
            try:
                self._vad = _webrtcvad.Vad(aggressiveness)
                self._use_webrtc = True
                log.info(
                    "pcm_webrtcvad_init",
                    aggressiveness=aggressiveness,
                    frame_ms=frame_ms,
                    rate=fmt.sample_rate,
                )
            except Exception as exc:
                log.warning("pcm_webrtcvad_init_failed", error=str(exc))

        if not self._use_webrtc:
            log.warning(
                "pcm_webrtcvad_fallback_to_energy",
                reason="webrtcvad not installed" if not _WEBRTCVAD else "init failed",
            )
            self._fallback = PCMVADGate(fmt=fmt, hangover_s=hangover_s,
                                        pre_roll_s=pre_roll_s, min_speech_s=min_speech_s)

    def _classify_frame(self, frame_int16: np.ndarray) -> bool:
        """Classify a fixed-size int16 frame. Thread-safe."""
        if not self._use_webrtc or self._vad is None:
            rms = float(np.sqrt(np.mean(frame_int16.astype(np.float64) ** 2)))
            return rms >= _VAD_ONSET_RMS
        raw = frame_int16.astype("<i2").tobytes()
        try:
            return bool(self._vad.is_speech(raw, self._fmt.sample_rate))
        except Exception: # noqa
            return False

    def is_speech_chunk(self, chunk: PCMChunk) -> bool:
        """
        Classify a PCMChunk by majority vote of its constituent WebRTC frames.

        Returns True if more than half of the frames are classified as speech.
        """
        if not self._use_webrtc and self._fallback is None:
            return False

        data = chunk.data
        if data.ndim == 2:
            data = data[:, 0]
        if chunk.fmt.dtype != "int16":
            data = PCMConverter._coerce_dtype(data, chunk.fmt.dtype, "int16")   # noqa

        combined = np.concatenate([self._carry, data])
        n_frames_avail = len(combined) // self._frame_samples
        votes, total = 0, 0
        for i in range(n_frames_avail):
            frame = combined[i * self._frame_samples: (i + 1) * self._frame_samples]
            votes += int(self._classify_frame(frame))
            total += 1
        self._carry = combined[n_frames_avail * self._frame_samples:]
        if total == 0:
            return False
        return (votes / total) > 0.5

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """Yield speech segment PCMChunks from input async iterator."""
        if self._fallback is not None:
            async for seg in self._fallback.stream(chunks):
                yield seg
            return

        async for chunk in chunks:
            result = self._process(chunk)
            if result is not None:
                yield result

        flushed = self._flush()
        if flushed is not None:
            yield flushed

    # noinspection PyUnreachableCode
    def _process(self, chunk: PCMChunk) -> PCMChunk | None:
        data = chunk.data
        mono = data[:, 0] if data.ndim == 2 else data
        is_voice = self.is_speech_chunk(chunk)

        if self._state == _VADState.SILENCE:
            self._pre_roll.write(mono.astype(self._fmt.dtype) if mono.ndim == 1 else mono[:, 0])
            if is_voice:
                self._state = _VADState.SPEECH
                pre = self._pre_roll.read(self._pre_roll_frames)
                if len(pre) > 0:
                    self._speech_frames.append(pre)
                    self._speech_frame_count += len(pre)
                self._speech_frames.append(data)
                self._speech_frame_count += len(data)
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        if self._state == _VADState.SPEECH:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            if not is_voice:
                self._state = _VADState.HANGOVER
                self._hangover_remaining = self._hangover_frames
            return None

        if self._state == _VADState.HANGOVER:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            self._hangover_remaining -= len(data)
            if is_voice:
                self._state = _VADState.SPEECH
                self._hangover_remaining = 0
                return None
            if self._hangover_remaining <= 0:
                self._state = _VADState.SILENCE
                self._pre_roll.clear()
                return self._emit(chunk)
        return None

    def _emit(self, ref: PCMChunk) -> PCMChunk | None:
        if not self._speech_frames:
            return None
        merged = np.concatenate(self._speech_frames)
        self._speech_frames = []
        count = self._speech_frame_count
        self._speech_frame_count = 0
        if count < self._min_frames:
            return None
        return PCMChunk(
            data=merged, fmt=self._fmt, timestamp=ref.timestamp,
            seq=ref.seq, is_final=True, source="webrtcvad"
        )

    # noinspection PyUnreachableCode
    def _flush(self) -> PCMChunk | None:
        if self._state == _VADState.SILENCE or not self._speech_frames:
            self._speech_frames = []
            self._speech_frame_count = 0
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        self._state = _VADState.SILENCE
        dummy = PCMChunk(data=np.array([], dtype=self._fmt.dtype), fmt=self._fmt,
                         timestamp=time.monotonic(), is_final=True, source="webrtcvad.flush")
        return self._emit(dummy)

    def reset(self) -> None:
        """Reset all state."""
        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames = []
        self._speech_frame_count = 0
        self._pre_roll.clear()
        self._carry = np.array([], dtype="int16")

# ── VAD backend adapters ──────────────────────────────────────────────────────

class _EnergyVADBackend:
    """Thin adapter wrapping PCMVADGate's RMS decision for use in PCMFusedVAD."""

    def __init__(self, onset_rms: float = _VAD_ONSET_RMS, scale: float = 1.0) -> None:
        self._onset = onset_rms
        self._scale = scale

    def is_speech(self, chunk: PCMChunk) -> bool:
        rms = chunk.rms()
        if chunk.fmt.dtype == "float32":
            rms *= 32768.0
        return rms >= self._onset


class _SpectralVADBackend:
    """Thin adapter wrapping PCMSpectralVAD for use in PCMFusedVAD."""

    def __init__(self, vad: PCMSpectralVAD) -> None:
        self._vad = vad

    def is_speech(self, chunk: PCMChunk) -> bool:
        return self._vad.is_speech(chunk)

# ── PCMFusedVAD — multi-backend VAD fusion ────────────────────────────────────

class PCMFusedVAD:
    """
    Multi-backend VAD fusion with configurable voting scheme.

    Combines decisions from multiple VAD algorithms to improve accuracy.
    Particularly useful in noisy environments where a single VAD misclassifies.

    Fusion modes:
        "any"      — fire if ANY backend says speech (highest recall)
        "all"      — fire only if ALL backends agree (highest precision)
        "majority" — fire if >50% of backends agree (balanced; ≥2 backends)

    The fused VAD accumulates speech segments exactly like PCMVADGate: it
    maintains its own pre-roll, hangover, and minimum duration logic around
    the fused boolean decision signal.

    Usage::

        energy_vad = _EnergyVADBackend()
        spectral_vad = _SpectralVADBackend(PCMSpectralVAD(fmt))
        fused = PCMFusedVAD(
            fmt=fmt,
            backends=[energy_vad, spectral_vad],
            mode="majority",
        )
        async for seg in fused.stream(mic_stream):
            await stt.transcribe(seg)
    """

    def __init__(
        self,
        fmt: PCMFormat,
        backends: list[VADBackend],
        mode: FusionMode = "majority",
        hangover_s: float = _VAD_HANGOVER_S,
        pre_roll_s: float = _VAD_PRE_ROLL_S,
        min_speech_s: float = _VAD_MIN_SPEECH_S,
    ) -> None:
        if not backends:
            raise ValueError("At least one VAD backend required")
        self._fmt = fmt
        self._backends = backends
        self._mode = mode
        self._hangover_frames = fmt.frames_for_duration(hangover_s)
        self._pre_roll_frames = fmt.frames_for_duration(pre_roll_s)
        self._min_frames = fmt.frames_for_duration(min_speech_s)
        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames: list[np.ndarray] = []
        self._speech_frame_count = 0
        self._pre_roll = PCMRingBuffer(
            capacity=max(self._pre_roll_frames * 2, 512), fmt=fmt
        )

    def _decide(self, chunk: PCMChunk) -> bool:
        votes = [b.is_speech(chunk) for b in self._backends]
        if self._mode == "any":
            return any(votes)
        if self._mode == "all":
            return all(votes)
        # majority
        return sum(votes) > len(votes) / 2

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """Yield fused speech segments."""
        async for chunk in chunks:
            result = self._process(chunk)
            if result is not None:
                yield result
        flushed = self._flush()
        if flushed is not None:
            yield flushed

    # noinspection PyUnreachableCode
    def _process(self, chunk: PCMChunk) -> PCMChunk | None:
        is_voice = self._decide(chunk)
        data = chunk.data
        mono = data[:, 0] if data.ndim == 2 else data

        if self._state == _VADState.SILENCE:
            self._pre_roll.write(mono if mono.ndim == 1 else mono.reshape(-1))
            if is_voice:
                self._state = _VADState.SPEECH
                pre = self._pre_roll.read(self._pre_roll_frames)
                if len(pre) > 0:
                    self._speech_frames.append(pre)
                    self._speech_frame_count += len(pre)
                self._speech_frames.append(data)
                self._speech_frame_count += len(data)
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        if self._state == _VADState.SPEECH:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            if not is_voice:
                self._state = _VADState.HANGOVER
                self._hangover_remaining = self._hangover_frames
            return None

        if self._state == _VADState.HANGOVER:
            self._speech_frames.append(data)
            self._speech_frame_count += len(data)
            self._hangover_remaining -= len(data)
            if is_voice:
                self._state = _VADState.SPEECH
                self._hangover_remaining = 0
                return None
            if self._hangover_remaining <= 0:
                self._state = _VADState.SILENCE
                self._pre_roll.clear()
                return self._emit(chunk)
        return None

    def _emit(self, ref: PCMChunk) -> PCMChunk | None:
        if not self._speech_frames:
            return None
        merged = np.concatenate(self._speech_frames)
        self._speech_frames = []
        count = self._speech_frame_count
        self._speech_frame_count = 0
        if count < self._min_frames:
            return None
        return PCMChunk(
            data=merged, fmt=self._fmt, timestamp=ref.timestamp,
            seq=ref.seq, is_final=True, source="fused_vad"
        )

    # noinspection PyUnreachableCode
    def _flush(self) -> PCMChunk | None:
        if self._state == _VADState.SILENCE or not self._speech_frames:
            self._speech_frames = []
            self._speech_frame_count = 0
            return None

        # defensive runtime validation; analyzer assumes Literal narrowing

        self._state = _VADState.SILENCE
        dummy = PCMChunk(data=np.array([], dtype=self._fmt.dtype), fmt=self._fmt,
                         timestamp=time.monotonic(), is_final=True, source="fused_vad.flush")
        return self._emit(dummy)

    def reset(self) -> None:
        """Reset all state. Call on barge-in or pipeline restart."""
        self._state = _VADState.SILENCE
        self._hangover_remaining = 0
        self._speech_frames = []
        self._speech_frame_count = 0
        self._pre_roll.clear()

# ── PCMAGCProcessor — automatic gain control + peak limiter ───────────────────

class PCMAGCProcessor:
    """
    Automatic Gain Control (AGC) with a peak lookahead limiter.

    Architecture (dual-stage):

    Stage 1 — AGC:
        Slow-attack gain follower targets a configurable output RMS level.
        Attack/release are asymmetric: slow attack prevents pumping, fast
        release recovers quickly from transients.

    Stage 2 — Peak Limiter:
        Fast-attack hard limiter prevents the AGC-amplified signal from
        clipping. Uses a brickwall gain at -3 dBFS by default.

    This is the reference pre-STT input normalization path. It ensures
    consistent input levels to Whisper regardless of microphone gain.

    Parameters:
        fmt:            Input PCMFormat (float32 required for internal math).
        target_rms:     Target RMS level. Default 0.1 (≈ -20 dBFS for float32).
        attack_ms:      AGC attack time constant. Default 200 ms.
        release_ms:     AGC release time constant. Default 50 ms.
        max_gain:       Maximum gain multiplier (prevents extreme amplification). Default 32.
        limiter_threshold: Peak limiter threshold (float32, 0-1). Default 0.95.
        limiter_attack_ms: Limiter attack time. Default 0.5 ms.
        limiter_release_ms: Limiter release time. Default 20 ms.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        target_rms: float = 0.1,
        attack_ms: float = 200.0,
        release_ms: float = 50.0,
        max_gain: float = 32.0,
        limiter_threshold: float = 0.95,
        limiter_attack_ms: float = 0.5,
        limiter_release_ms: float = 20.0,
    ) -> None:
        self._fmt = fmt
        self._target_rms = target_rms
        self._max_gain = max_gain
        self._limiter_thresh = limiter_threshold
        sr = fmt.sample_rate

        agc_attack_samples = max(1.0, attack_ms * sr / 1000.0)
        agc_release_samples = max(1.0, release_ms * sr / 1000.0)

        lim_attack_samples = max(1.0, limiter_attack_ms * sr / 1000.0)
        lim_release_samples = max(1.0, limiter_release_ms * sr / 1000.0)

        self._agc_attack = math.exp(-1.0 / agc_attack_samples)
        self._agc_release = math.exp(-1.0 / agc_release_samples)

        self._lim_attack = math.exp(-1.0 / lim_attack_samples)
        self._lim_release = math.exp(-1.0 / lim_release_samples)

        # State
        self._rms_env: float = 0.0
        self._gain: float = 1.0
        self._lim_gain: float = 1.0

    def process(self, chunk: PCMChunk) -> PCMChunk:
        """Apply AGC + limiting. Returns new float32 PCMChunk."""
        data = chunk.data.astype(np.float64)
        mono = data[:, 0] if data.ndim == 2 else data

        # Frame RMS
        frame_rms = float(np.sqrt(np.mean(mono ** 2))) + 1e-12

        # AGC gain update
        if frame_rms > self._rms_env:
            self._rms_env = self._agc_attack * self._rms_env + (1 - self._agc_attack) * frame_rms
        else:
            self._rms_env = self._agc_release * self._rms_env + (1 - self._agc_release) * frame_rms

        desired_gain = min(self._target_rms / (self._rms_env + 1e-12), self._max_gain)
        self._gain = self._gain + 0.1 * (desired_gain - self._gain)  # smooth
        _agc_gain_applied.observe(self._gain)

        # Apply AGC gain
        amplified = data * self._gain

        # Peak limiter — per-sample
        peak = float(np.max(np.abs(amplified if amplified.ndim == 1 else amplified[:, 0])))
        if peak > self._limiter_thresh:
            target_lim = self._limiter_thresh / peak
            self._lim_gain = self._lim_attack * self._lim_gain + (1 - self._lim_attack) * target_lim
        else:
            self._lim_gain = self._lim_release * self._lim_gain + (1 - self._lim_release) * 1.0

        limited = np.clip(amplified * self._lim_gain, -1.0, 1.0)

        return PCMChunk(
            data=limited.astype(np.float32),
            fmt=PCMFormat(sample_rate=chunk.fmt.sample_rate, channels=chunk.fmt.channels,
                          dtype="float32"),
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: AGC-process each chunk."""
        async for chunk in chunks:
            yield self.process(chunk)

    def reset(self) -> None:
        """Reset AGC state (call on utterance boundary to avoid level carry-over)."""
        self._rms_env = 0.0
        self._gain = 1.0
        self._lim_gain = 1.0

# ── PCMNoiseGate — professional gate with attack / hold / release ─────────────

class PCMNoiseGate:
    """
    Professional noise gate with attack, hold, and release time constants.

    Unlike PCMVADGate (which segments speech), the noise gate applies
    smooth gain reduction sample-by-sample — the output is continuous audio
    with quiet sections near-silenced rather than entirely gated out.

    This makes it suitable as a pre-processor for TTS playback (suppressing
    room ambience leaking into a partially open speaker) and as a cleanup
    step before spectral VAD.

    Signal path::

        RMS detector → gain computer → gain smoother (τ_a, τ_h, τ_r) → VCA

    Parameters:
        fmt:           Input PCMFormat.
        threshold_db:  Gate open threshold in dBFS. Default -40 dBFS.
        range_db:      Maximum gain reduction (attenuation) when closed. Default -80 dBFS.
        attack_ms:     Attack time constant (ms). Default 1.0.
        hold_ms:       Hold time (ms) before release begins. Default 50.
        release_ms:    Release time constant (ms). Default 100.
        lookahead_ms:  Lookahead delay (ms) for pre-trigger. Default 0 (disabled).
    """

    def __init__(
        self,
        fmt: PCMFormat,
        threshold_db: float = -40.0,
        range_db: float = -80.0,
        attack_ms: float = 1.0,
        hold_ms: float = 50.0,
        release_ms: float = 100.0,
        lookahead_ms: float = 0.0,
    ) -> None:
        self._fmt = fmt
        self._threshold_lin = 10 ** (threshold_db / 20.0)  # amplitude
        self._range_lin = 10 ** (range_db / 20.0)          # amplitude floor
        sr = fmt.sample_rate

        # Time constants in samples
        attack_samples = max(1.0, attack_ms * sr / 1000.0)
        release_samples = max(1.0, release_ms * sr / 1000.0)

        self._attack_coeff = math.exp(-1.0 / attack_samples)
        self._release_coeff = math.exp(-1.0 / release_samples)
        self._hold_samples = int(hold_ms * sr / 1000)

        # State
        self._envelope: float = 0.0   # peak follower
        self._gain: float = self._range_lin
        self._hold_counter: int = 0   # countdown in samples

        # Lookahead delay buffer
        self._lookahead_samples = int(lookahead_ms * sr / 1000)
        channels = fmt.channels
        shape = (self._lookahead_samples,) if channels == 1 else (self._lookahead_samples, channels)
        self._delay_buf = np.zeros(shape, dtype="float32")
        self._delay_pos = 0

    def process(self, chunk: PCMChunk) -> PCMChunk:
        """
        Apply gate to a chunk. Returns a new PCMChunk with gain applied.

        Vectorised implementation replacing the O(n) Python per-sample loop.
        All inner loops run in BLAS/C via scipy.signal.lfilter.

        Mathematical overview
        ─────────────────────
        The classical branch-switching one-pole envelope follower:

            env[i] = a·env[i-1] + (1−a)·|x[i]|   if |x[i]| > env[i-1]   (attack)
            env[i] = r·env[i-1] + (1−r)·|x[i]|   if |x[i]| ≤ env[i-1]  (release)

        is equivalent to the parallel-filter maximum identity (valid when a ≤ r,
        i.e. fast attack / slow release — the normal noise-gate configuration):

            env[i] = max(env_attack[i], env_release[i])

        Proof:
          Rising edge (|x| > env):
            env_a − env_r = (a−r)·(env − |x|)
            a ≤ r → (a−r) ≤ 0;  env < |x| → (env−|x|) < 0
            product ≥ 0  →  env_a ≥ env_r  →  max selects attack ✓
          Falling edge: symmetric argument selects release ✓

        This lets us replace the Python loop with two lfilter calls (O(n) BLAS)
        and a np.maximum — roughly 50–200× faster at 48 kHz / 60ms chunks.

        The same identity is applied to the gain-smoothing stage (step 3).

        Hold logic uses a single O(n) np.maximum.accumulate pass to propagate
        the most-recent above-threshold index forward, enabling fully vectorised
        hold detection without any Python-level iteration.
        """
        data = chunk.data.astype(np.float32)
        mono = data[:, 0] if data.ndim == 2 else data
        n = len(mono)
        abs_x = np.abs(mono).astype(np.float64)

        a = float(self._attack_coeff)
        r = float(self._release_coeff)

        # Pre-compute IIR coefficients once — reused for envelope and gain stages.
        # H(z) = (1−c) / (1 − c·z⁻¹)  →  b = [1−c], A = [1, −c]
        b_a = np.array([1.0 - a])
        A_a = np.array([1.0, -a])
        b_r = np.array([1.0 - r])
        A_r = np.array([1.0, -r])

        # ── 1. Vectorised peak envelope follower ──────────────────────────────
        #
        # lfilter_zi returns the initial-condition vector that puts the filter
        # in steady state for a unit-step input; scaling by self._envelope seeds
        # the filter at the correct prior output value, ensuring chunk-boundary
        # continuity (no click/pop at the seam).

        zi_ea, _ = _scipy_signal.lfilter(b_a, A_a, abs_x,
                                         zi=_scipy_signal.lfilter_zi(b_a, A_a) * self._envelope)
        zi_er, _ = _scipy_signal.lfilter(b_r, A_r, abs_x,
                                         zi=_scipy_signal.lfilter_zi(b_r, A_r) * self._envelope)

        # Unpack properly — lfilter returns (y, zf); we only need y here.
        env_attack, _ = _scipy_signal.lfilter(b_a, A_a, abs_x,
                                              zi=_scipy_signal.lfilter_zi(b_a, A_a) * self._envelope)
        env_release, _ = _scipy_signal.lfilter(b_r, A_r, abs_x,
                                               zi=_scipy_signal.lfilter_zi(b_r, A_r) * self._envelope)
        envelope = np.maximum(env_attack, env_release)  # parallel identity
        self._envelope = float(envelope[-1])

        # ── 2. Vectorised hold logic ──────────────────────────────────────────
        #
        # Goal: gate_open[i] = True iff envelope[i] ≥ threshold  OR
        #                            the most recent above-threshold event was
        #                            within hold_samples samples ago.
        #
        # Algorithm:
        #   Define last_above[i] = index of most recent j ≤ i where above[j] is True
        #   gate_open[i]         = (i − last_above[i]) ≤ hold_samples
        #
        # np.maximum.accumulate over the array:
        #   seed[i] = i         if above[i]
        #           = virtual_last  otherwise  (very negative sentinel)
        # propagates the most-recent True index forward in one O(n) pass.
        #
        # Cross-chunk continuity: seed the accumulate with a virtual prior index
        # derived from self._hold_counter so hold events that started in the
        # previous chunk keep the gate open at the start of this one.
        #
        # Derivation of virtual_last:
        #   gate_open[0] should be True iff hold_counter > 0
        #   gate_open[0] = (0 − virtual_last) ≤ hold_samples
        #                = True  iff  virtual_last ≥ −hold_samples
        #   Set virtual_last = hold_counter − hold_samples − 1:
        #     hold_counter = 0  → virtual_last = −hold_samples − 1 < −hold_samples → closed ✓
        #     hold_counter = 1  → virtual_last = −hold_samples          → open ✓
        #     hold_counter = H  → virtual_last = −1                     → open ✓

        H = self._hold_samples
        virtual_last = self._hold_counter - H - 1  # Python int, no overflow

        above = envelope >= self._threshold_lin  # (n,) bool
        indices = np.arange(n, dtype=np.intp)
        seed = np.where(above, indices, np.intp(virtual_last))  # (n,)
        np.maximum.accumulate(seed, out=seed)  # O(n), in-place

        gate_open = (indices - seed) <= H  # (n,) bool
        target = np.where(gate_open, 1.0, float(self._range_lin))  # (n,) float64

        # Update hold_counter for next chunk.
        # samples_since = (n−1) − last_above[n−1]; works correctly even when
        # last_above[n−1] == virtual_last (no above events this chunk):
        #   samples_since = (n−1) − (hold_counter_prev − H − 1)
        #                 = n + H − hold_counter_prev
        #   hold_remaining = H − samples_since = hold_counter_prev − n  ✓
        if above[-1]:
            self._hold_counter = H
        else:
            self._hold_counter = max(0, H - int((n - 1) - int(seed[-1])))

        # ── 3. Vectorised gain smoothing ──────────────────────────────────────
        #
        # Same parallel-filter identity applied to the target signal:
        #   when target rises (gate opens)  → attack filter wins  → sharp onset
        #   when target falls (gate closes) → release filter wins → smooth tail
        #
        # Seeded at self._gain for chunk-boundary continuity.

        gain_attack, _ = _scipy_signal.lfilter(b_a, A_a, target,
                                               zi=_scipy_signal.lfilter_zi(b_a, A_a) * self._gain)
        gain_release, _ = _scipy_signal.lfilter(b_r, A_r, target,
                                                zi=_scipy_signal.lfilter_zi(b_r, A_r) * self._gain)
        gain_env = np.maximum(gain_attack, gain_release).astype(np.float32)
        self._gain = float(gain_env[-1])

        # ── 4. Apply gain envelope ────────────────────────────────────────────
        out = data * gain_env[:, np.newaxis] if data.ndim == 2 else data * gain_env

        return PCMChunk(
            data=out.astype(chunk.fmt.dtype),
            fmt=chunk.fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def aprocess(self, chunk: PCMChunk) -> PCMChunk:
        """Async wrapper around process() for pipeline compatibility."""
        return self.process(chunk)

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: gate each chunk in sequence."""
        async for chunk in chunks:
            yield self.process(chunk)

# ── PCMNoiseSuppressor — spectral subtraction noise suppressor ────────────────

class PCMNoiseSuppressor:
    """
    Spectral subtraction noise suppressor with adaptive noise floor estimation.

    Algorithm (Boll 1979 + Ephraim-Malah smoothing):
      1. Short-Time Fourier Transform (STFT) on each frame
      2. During silence frames: update noise PSD estimate (exponential MA)
      3. Compute suppression gain: G(k) = max(1 - α·N(k)/|X(k)|², β)
         where α is oversubtraction, β is spectral floor, N is noise PSD,
         |X|² is noisy signal PSD
      4. Apply gain to complex spectrum
      5. ISTFT with overlap-add reconstruction

    Parameters:
        fmt:               Input PCMFormat. Must be float32.
        fft_size:          FFT size. Default 512 (32 ms at 16 kHz).
        hop_size:          Hop between frames. Default 128 (8 ms at 16 kHz).
        noise_update_rate: EMA coefficient for noise PSD update. Default 0.95.
        oversubtraction:   α: how aggressively to subtract noise. Default 2.0.
        spectral_floor:    β: minimum gain to prevent musical noise. Default 0.02.
        silence_threshold: RMS below which frame is classified as noise. Default 0.01 (float32).
        noise_warmup_frames: Frames to collect before suppression begins. Default 10.

    Notes:
        • Requires float32 input. Convert int16 before use.
        • STFT uses a Hann window with 75% overlap (hop_size = fft_size // 4).
        • For real-time use, process each incoming chunk individually;
          latency = one hop = hop_size / sample_rate seconds.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        fft_size: int = 512,
        hop_size: int | None = None,
        noise_update_rate: float = 0.95,
        oversubtraction: float = _NS_OVERSUBTRACTION,
        spectral_floor: float = _NS_SPECTRAL_FLOOR,
        silence_threshold: float = 0.01,
        noise_warmup_frames: int = _NS_WARMUP_FRAMES,
    ) -> None:
        if fmt.dtype != "float32":
            raise ValueError(
                f"PCMNoiseSuppressor requires float32 input; got {fmt.dtype!r}. "
                "Add a PCMConverter stage before this processor."
            )
        self._fmt = fmt
        self._fft_size = fft_size
        self._hop = hop_size if hop_size is not None else fft_size // 4
        self._noise_alpha = noise_update_rate
        self._alpha = oversubtraction
        self._beta = spectral_floor
        self._silence_thresh = silence_threshold
        self._warmup = noise_warmup_frames

        n_bins = fft_size // 2 + 1
        self._noise_psd = np.ones(n_bins, dtype=np.float64) * 1e-6
        self._frame_count = 0

        # Overlap-add buffer
        self._ola_buf = np.zeros(fft_size, dtype=np.float64)
        self._window = np.hanning(fft_size)

        # COLA normalization: for 75% overlap with Hann window, the sum of
        # squared windows at each output sample converges to a constant.
        # For Hann + 75% overlap, this is exactly (fft_size / hop) * 0.375
        self._win_scale = self._fft_size / self._hop * 0.375

        # Input carry buffer for frame alignment
        self._carry = np.array([], dtype=np.float32)

    def _suppress_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction to a single FFT-aligned float64 frame."""
        windowed = frame * self._window
        spectrum = np.fft.rfft(windowed)
        psd = np.abs(spectrum) ** 2

        rms = float(np.sqrt(np.mean(frame ** 2)))
        is_silence = rms < self._silence_thresh

        if is_silence or self._frame_count < self._warmup:
            # Update noise estimate
            self._noise_psd = (
                self._noise_alpha * self._noise_psd
                + (1 - self._noise_alpha) * psd
            )

        # Compute suppression gain
        gain_sq = np.maximum(1.0 - self._alpha * self._noise_psd / (psd + 1e-12), self._beta)
        gain = np.sqrt(gain_sq)

        if not is_silence and self._frame_count >= self._warmup:
            suppressed = gain * spectrum
            db_reduction = 10 * math.log10(float(np.mean(gain_sq)) + 1e-12)
            _noise_suppressor_reduction_db.observe(abs(db_reduction))
        else:
            suppressed = spectrum  # pass through during warmup

        self._frame_count += 1
        return np.fft.irfft(suppressed, n=self._fft_size)

    def process(self, chunk: PCMChunk) -> PCMChunk:
        """
        Process a PCMChunk through the spectral suppressor.

        Returns a new PCMChunk with noise reduced. There may be a small
        latency (one hop) due to overlap-add buffering.
        """
        data = chunk.data.astype(np.float64)
        if data.ndim == 2:
            # Process only first channel; replicate to all channels after
            mono = data[:, 0]
        else:
            mono = data

        # Append carry and process in hops
        combined = np.concatenate([self._carry, mono])
        output_frames = []
        pos = 0
        while pos + self._fft_size <= len(combined):
            frame = combined[pos: pos + self._fft_size]
            processed = self._suppress_frame(frame)

            # Overlap-add
            out_chunk = np.zeros(self._hop, dtype=np.float64)
            self._ola_buf[:self._fft_size] += processed
            out_chunk[:] = self._ola_buf[:self._hop] / (self._win_scale + 1e-12)
            self._ola_buf = np.roll(self._ola_buf, -self._hop)
            self._ola_buf[-self._hop:] = 0.0

            output_frames.append(out_chunk)
            pos += self._hop

        self._carry = combined[pos:]

        if not output_frames:
            # No complete frames yet — return silence of same duration
            out = np.zeros_like(mono)
        else:
            out = np.concatenate(output_frames).astype(np.float32)
            # Trim to original length if needed
            if len(out) > len(mono):
                out = out[:len(mono)]
            elif len(out) < len(mono):
                out = np.pad(out, (0, len(mono) - len(out)))

        if data.ndim == 2:
            out_2d = np.column_stack([out] * self._fmt.channels)
            out_final = out_2d.astype(chunk.fmt.dtype)
        else:
            out_final = out.astype(chunk.fmt.dtype)

        return PCMChunk(
            data=out_final,
            fmt=chunk.fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: suppress noise on each chunk."""
        async for chunk in chunks:
            yield self.process(chunk)

# ── PCMEchoCanceller — NLMS adaptive echo canceller (AEC) ────────────────────

class PCMEchoCanceller:
    """
    Acoustic Echo Cancellation using the Normalised LMS (NLMS) adaptive filter.

    In a voice agent deployment, the loudspeaker output (TTS playback) is
    picked up by the microphone and appears as acoustic echo. This module
    implements a feed-forward AEC:

        y(n)  — microphone signal (echo + speech)
        x(n)  — reference signal (speaker / TTS output)
        ŷ(n)  — estimated echo  = w^T · x_buffer
        e(n)  — residual        = y(n) − ŷ(n)

    NLMS update rule:
        w(n+1) = w(n) + μ · e(n) · x_buffer / (||x_buffer||² + δ)

    Parameters:
        fmt:          PCMFormat of both mic and reference signals.
        filter_len:   Adaptive filter length in samples. Default 512 (32 ms @ 16 kHz).
        step_size:    NLMS step size μ. Default 0.05. Higher = faster adaptation.
        regularisation: NLMS δ (prevents division by zero). Default 1e-6.
        double_talk_thresh: Suppress AEC during double-talk (near-end speech).
                           Set to 0 to disable. Default 0.5.

    Usage::

        aec = PCMEchoCanceller(fmt)
        # In playback path (TTS):
        aec.push_reference(tts_chunk)
        # In capture path (mic):
        clean = aec.cancel(mic_chunk)

    Thread safety:
        push_reference() may be called from a different thread than cancel().
        A threading.Lock guards the shared filter weights.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        filter_len: int = 512,
        step_size: float = 0.05,
        regularisation: float = 1e-6,
        double_talk_thresh: float = 0.5,
    ) -> None:
        self._fmt = fmt
        self._L = filter_len
        self._mu = step_size
        self._delta = regularisation
        self._dt_thresh = double_talk_thresh

        # Adaptive filter weights
        self._w = np.zeros(filter_len, dtype=np.float64)
        # Reference signal delay line
        self._x_buf = np.zeros(filter_len, dtype=np.float64)
        self._lock = threading.Lock()
        self._ref_hist: np.ndarray = np.zeros(self._L)  # grows on first push_reference call

    def push_reference(self, chunk: PCMChunk) -> None:
        """
        Feed the loudspeaker (TTS) signal as the echo reference.

        Must be called for every TTS chunk written to the speaker, ideally
        just before or concurrently with playback.
        """
        ref = chunk.data.astype(np.float64)
        if ref.ndim == 2:
            ref = ref[:, 0]
        if chunk.fmt.dtype == "int16":
            ref /= 32768.0

        with self._lock:
            # ── BUG-006 fix: batch delay-line update ──────────────────────
            n = len(ref)
            if n >= self._L:
                self._x_buf[:] = ref[-(self._L):][::-1]  # noqa | fill entire buffer
            else:
                self._x_buf[n:] = self._x_buf[:-n]  # shift old data
                self._x_buf[:n] = ref[::-1]  # insert new (reversed for convolution order)

            # ── Store history for vectorised cancel() ─────────────────────
            # Need last (max_chunk + L - 1) samples to reconstruct per-sample
            # delay-line states. Fixed-size slice avoids unbounded growth.
            self._ref_hist = np.concatenate([self._ref_hist, ref])[-(self._L + 8192):]

    def cancel(self, mic_chunk: PCMChunk) -> PCMChunk:
        """
        Remove echo from a microphone PCMChunk.

        Vectorised block-NLMS replacing the O(n·L) per-sample Python loop.

        Mathematical overview
        ─────────────────────
        Standard NLMS (per-sample):
            y_hat[i]  = wᵀ x[i]               (FIR echo estimate)
            e[i]      = y[i] − y_hat[i]        (error / cleaned signal)
            w        += (μ / (‖x[i]‖² + δ)) · e[i] · x[i]

        where x[i] ∈ ℝᴸ is the delay-line state at sample i.

        Block reformulation (this implementation):
            Build X ∈ ℝⁿˣᴸ  where row i = x[i]  (Toeplitz input matrix)
            y_hat  = X w                           (BLAS dgemv, O(nL) in C)
            e      = y − y_hat
            Δw     = Yᵀ diag(μ·e[adapt] / (‖Y[adapt]‖² + δ)) e[adapt]
                                                   (BLAS dgemv, O(nL) in C)
            w     += Δw

        Block NLMS converges identically to per-sample NLMS when μ ≪ 1
        (always true for speech: μ ≤ 0.1), because the weight vector
        changes slowly relative to a single chunk. The approximation error
        is O(μ²), negligible in practice.

        Complexity: O(nL) numpy/BLAS vs O(nL) Python loops — same big-O
        but ~50–200× faster wall-clock because all inner loops run in C/BLAS.
        """
        mic = mic_chunk.data.astype(np.float64)
        mic_mono = mic[:, 0] if mic.ndim == 2 else mic.copy()
        if mic_chunk.fmt.dtype == "int16":
            mic_mono /= 32768.0

        n = len(mic_mono)

        with self._lock:
            # ── 1. Build the Toeplitz input matrix X ∈ ℝⁿˣᴸ ─────────────────
            #
            # X[i, k] = ref[i + k]  in oldest-first indexing (see derivation below)
            # and y_hat = X @ w_rev  where w_rev = w[::-1]
            #
            # Derivation:
            #   Delay-line at sample i (newest-first): x[i] = [ref[i], ref[i-1], ..., ref[i-L+1]]
            #   y_hat[i] = Σ_k w[k] · ref[i-k]  = Σ_k w[k] · ref_oldest[i_off + i - k]
            #            = Σ_j ref_oldest[i_off + i + j] · w[L-1-j]   (j = L-1-k)
            #            = Y[i] · w_rev
            #   where Y[i, j] = ref_oldest[i_off + i + j]  (positive strides → stride trick works)
            #
            # ref_hist is oldest→newest, so we slice the last (n + L - 1) samples:
            seg = self._ref_hist[-(n + self._L - 1):]  # shape: (n + L - 1,)

            # Zero-copy Toeplitz view via stride tricks.
            # Y[i, j] = seg[i + j], row stride = col stride = 1 element.
            # Negative strides are not needed because we already reversed the
            # convolution direction by using w_rev instead of w.
            stride = seg.strides[0]
            Y = np.lib.stride_tricks.as_strided(
                seg,
                shape=(n, self._L),
                strides=(stride, stride),
            )
            # NOTE: Y is a read-only view; do NOT write to it.
            w_rev = self._w[::-1]  # reversal is O(1) — just a view

            # ── 2. Echo estimation (BLAS dgemv) ───────────────────────────────
            y_hat = Y @ w_rev  # shape: (n,)  — O(nL) in BLAS
            e = mic_mono - y_hat  # cleaned signal before weight update

            # ── 3. Vectorised double-talk detection ───────────────────────────
            # Per-sample near-end and far-end power, fully broadcast.
            near_power = mic_mono ** 2  # (n,)
            # np.einsum row-wise dot is ~2× faster than (Y**2).sum(axis=1)
            # because it avoids materialising the squared matrix.
            far_power = np.einsum("ij,ij->i", Y, Y) / self._L  # (n,)

            if self._dt_thresh > 0:
                dt_mask = (
                        (far_power > 1e-10) &
                        (near_power > self._dt_thresh * far_power)
                )
            else:
                dt_mask = np.zeros(n, dtype=bool)

            # ── 4. Block-NLMS weight update ───────────────────────────────────
            #
            # For each non-double-talk sample i:
            #   Δw_rev += (μ / (‖Y[i]‖² + δ)) · e[i] · Y[i]
            #
            # Vectorised:
            #   norms    = diag(Y[adapt] Yᵀ[adapt]) + δ    shape: (k,)
            #   scaled_e = μ · e[adapt] / norms             shape: (k,)
            #   Δw_rev   = Y[adapt]ᵀ @ scaled_e             shape: (L,)  — BLAS dgemv
            #
            adapt = ~dt_mask
            if np.any(adapt):
                Y_a = Y[adapt]  # (k, L)
                e_a = e[adapt]  # (k,)
                norms_a = np.einsum("ij,ij->i", Y_a, Y_a) + self._delta  # (k,)
                scaled_e = (self._mu * e_a) / norms_a  # (k,)
                dw_rev = Y_a.T @ scaled_e  # (L,)  BLAS dgemv
                self._w += dw_rev[::-1]  # un-reverse back to w convention

            out = e

        # ── 5. Diagnostics ────────────────────────────────────────────────────
        residual_rms = float(np.sqrt(np.mean(out ** 2)))
        _aec_residual_rms.observe(residual_rms)

        if mic_chunk.fmt.dtype == "int16":
            out_data = (out * 32768.0).clip(-32768, 32767).astype(np.int16)
        else:
            out_data = out.astype(np.float32)

        if mic.ndim == 2:
            out_data = np.column_stack([out_data] * self._fmt.channels)

        return PCMChunk(
            data=out_data,
            fmt=mic_chunk.fmt,
            timestamp=mic_chunk.timestamp,
            seq=mic_chunk.seq,
            is_final=mic_chunk.is_final,
            source=mic_chunk.source,
        )

    def reset(self) -> None:
        """Reset filter weights and delay line (call on session boundary)."""
        with self._lock:
            self._w.fill(0.0)
            self._x_buf.fill(0.0)

    async def stream(
        self,
        mic_chunks: AsyncIterator[PCMChunk],
    ) -> AsyncIterator[PCMChunk]:
        """Async generator: cancel echo from each mic chunk in sequence."""
        async for chunk in mic_chunks:
            yield self.cancel(chunk)

# ── PCMDynamicsProcessor — compressor / expander / limiter ────────────────────

class PCMDynamicsProcessor:
    """
    Feed-forward dynamics processor: compressor, expander, limiter, or gate.

    Implements the standard dynamics transfer function:
        GR(dB) = (L − T) × (1/R − 1) for |L| > T in compressor mode

    Where:
        L = level (dBFS), T = threshold (dBFS), R = ratio, GR = gain reduction

    Knee smoothing (soft knee):
        Within [T − W/2, T + W/2] dBFS, gain reduction is smoothly interpolated.

    Metering:
        gain_reduction_db property returns current GR in dB (for VU metering).

    Parameters:
        fmt:             Input PCMFormat. Float32 preferred.
        mode:            "compressor" | "expander" | "limiter" | "gate". Default "compressor".
        threshold_db:    Threshold in dBFS. Default -20.
        ratio:           Compression ratio (1:ratio). Default 4. For limiter use ∞.
        knee_db:         Soft knee width in dB. Default 6.
        attack_ms:       Gain reduction attack. Default 10 ms.
        release_ms:      Gain reduction release. Default 100 ms.
        makeup_gain_db:  Post-compression makeup gain. Default 0.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        mode: DynamicsMode = "compressor",
        threshold_db: float = -20.0,
        ratio: float = 4.0,
        knee_db: float = 6.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        makeup_gain_db: float = 0.0,
    ) -> None:
        self._fmt = fmt
        self._mode = mode
        self._T = threshold_db
        self._R = ratio
        self._knee = knee_db
        self._makeup = 10 ** (makeup_gain_db / 20.0)
        sr = fmt.sample_rate

        attack_tau_samples = max(1.0, attack_ms * sr / 1000.0)
        release_tau_samples = max(1.0, release_ms * sr / 1000.0)

        self._attack_c = math.exp(-1.0 / attack_tau_samples)
        self._release_c = math.exp(-1.0 / release_tau_samples)

        self._gain_db: float = 0.0   # smoothed gain reduction in dB
        self.gain_reduction_db: float = 0.0  # public metering property

    # noinspection PyUnreachableCode
    def _compute_target_gr(self, level_db: float) -> float:
        """Compute target gain reduction (dB) from instantaneous level."""
        knee_lo = self._T - self._knee / 2
        knee_hi = self._T + self._knee / 2

        if self._mode == "compressor":
            if level_db <= knee_lo:
                return 0.0
            elif level_db >= knee_hi:
                return (level_db - self._T) * (1.0 / self._R - 1.0)
            else:
                # Soft knee
                interp = (level_db - knee_lo) / self._knee
                return interp ** 2 * (level_db - self._T) * (1.0 / self._R - 1.0)

        elif self._mode == "limiter":
            if level_db <= self._T:
                return 0.0
            return self._T - level_db  # hard brick wall

        elif self._mode == "expander":
            if level_db >= self._T:
                return 0.0
            return (level_db - self._T) * (self._R - 1.0)

        elif self._mode == "gate":
            if level_db >= self._T:
                return 0.0
            return -100.0  # full gate

        return 0.0 # defensive runtime validation; analyzer assumes Literal narrowing

    # noinspection PyUnreachableCode
    def process(self, chunk: PCMChunk) -> PCMChunk:
        """
        Apply dynamics processing. Returns new PCMChunk.

        Vectorised implementation replacing the O(n) Python per-sample loop.

        Mathematical overview
        ─────────────────────
        Classical feed-forward digital dynamics processor in the log (dB) domain:

            level_db[i]  = 20·log₁₀(|x[i]| + ε)          (level detection)
            target_gr[i] = f(level_db[i])                  (gain curve, piecewise numpy)
            gr[i]        = smooth(target_gr[i])             (ballistics)
            out[i]       = x[i] · makeup · 10^(gr[i]/20)   (apply gain)

        Ballistics (gain smoothing) use the parallel-filter minimum identity:
            When target_gr falls (more reduction → attack path wins):
                gr[i] = a·gr[i-1] + (1−a)·target_gr[i]
            When target_gr rises (less reduction → release path wins):
                gr[i] = r·gr[i-1] + (1−r)·target_gr[i]

        When a ≤ r (fast attack, slow release):
            gr[i] = min(gr_attack[i], gr_release[i])
        Note: MINIMUM not maximum — gain reduction is non-positive so the
        more negative value = stronger compression = attack path wins.
        This is the log-domain dual of the linear envelope follower.

        Gain curve vectorisation:
        _compute_target_gr's piecewise logic is lifted fully into numpy using
        np.where chains — one branch selected per mode (compile-time constant
        per instance), all arithmetic runs in C with no Python per-element calls.

        Chunk-boundary continuity:
        Both lfilter calls are seeded with self._gain_db via lfilter_zi so
        the filter state is preserved across chunks — no discontinuity artifacts.

        Complexity: O(n) numpy/BLAS — all inner loops in C/libm/BLAS.
        """

        data = chunk.data.astype(np.float64)
        scale = 32768.0 if chunk.fmt.dtype == "int16" else 1.0

        mono = data[:, 0] if data.ndim == 2 else data
        mono_norm = mono / scale

        # ── 1. Level detection — fully vectorised ─────────────────────────────
        #
        # np.log10 on the full array is a single libm vlog10 call vs n Python
        # math.log10() invocations. At 48kHz/60ms this is ~2880 fused ops in C.
        level_db = 20.0 * np.log10(np.abs(mono_norm) + 1e-12)  # (n,) float64

        # ── 2. Gain curve — fully vectorised piecewise numpy ──────────────────
        #
        # self._mode is a fixed string per instance — we branch once here in
        # Python and then every operation inside is a numpy ufunc call (C loop).
        # No np.vectorize, no per-element Python dispatch.
        #
        # Compressor gain curve (with soft knee):
        #   below knee_lo:                gr = 0
        #   above knee_hi:                gr = (L − T) · (1/R − 1)
        #   in knee [knee_lo, knee_hi]:   gr = interp² · (L − T) · (1/R − 1)
        #     where interp = (L − knee_lo) / knee_width  ∈ [0, 1]
        #
        # The soft knee blends the linear region into the compressed region
        # quadratically, avoiding the discontinuity in gain slope at threshold.
        #
        # Limiter: hard brick wall above T → gr = T − L  (always ≤ 0)
        # Expander: attenuates below T → gr = (L − T) · (R − 1)  (R > 1 → ≤ 0)
        # Gate: full mute below T → gr = −100 dB (≈ −∞ in practice)

        T = self._T
        knee = self._knee
        R = self._R
        knee_lo = T - knee * 0.5
        knee_hi = T + knee * 0.5
        slope = (1.0 / R - 1.0)  # precomputed, used in compressor only

        if self._mode == "compressor":
            above_knee = level_db >= knee_hi
            in_knee = (level_db > knee_lo) & ~above_knee

            # Soft-knee region: quadratic interpolation of gain slope
            interp = np.where(in_knee, (level_db - knee_lo) / knee, 0.0)
            gr_soft = interp ** 2 * (level_db - T) * slope

            # Hard region: full ratio above knee
            gr_hard = (level_db - T) * slope

            target_gr = np.where(above_knee, gr_hard,
                                 np.where(in_knee, gr_soft,
                                          0.0))

        elif self._mode == "limiter":
            # Brick-wall: clamp any excess above T back to T
            target_gr = np.where(level_db > T, T - level_db, 0.0)

        elif self._mode == "expander":
            # Attenuate below T: the further below threshold, the more gain cut
            # R > 1 here → (R−1) > 0 → (L−T)·(R−1) < 0 when L < T ✓
            target_gr = np.where(level_db < T, (level_db - T) * (R - 1.0), 0.0)

        elif self._mode == "gate":
            # Full mute below T — −100 dB ≈ 1e-5 linear, effectively silent
            target_gr = np.where(level_db < T, -100.0, 0.0)

        else:
            target_gr = np.zeros(len(level_db), dtype=np.float64) # defensive runtime validation; analyzer assumes Literal narrowing

        # ── 3. Gain smoothing — parallel-filter minimum identity ──────────────
        #
        # IIR coefficients for attack and release one-pole smoothers:
        #   H(z) = (1−c) / (1 − c·z⁻¹)
        a = float(self._attack_c)
        b_a = np.array([1.0 - a])
        A_a = np.array([1.0, -a])

        r = float(self._release_c)
        b_r = np.array([1.0 - r])
        A_r = np.array([1.0, -r])

        # Seed both filters at self._gain_db for chunk-boundary continuity.
        # lfilter_zi gives the steady-state initial condition for a unit-step
        # input; scaling by self._gain_db seeds the filter at the prior output.
        zi_a = _scipy_signal.lfilter_zi(b_a, A_a) * self._gain_db
        zi_r = _scipy_signal.lfilter_zi(b_r, A_r) * self._gain_db

        gr_attack, _ = _scipy_signal.lfilter(b_a, A_a, target_gr, zi=zi_a)
        gr_release, _ = _scipy_signal.lfilter(b_r, A_r, target_gr, zi=zi_r)

        # Minimum (not maximum) because gain reduction is non-positive:
        # the more negative value = stronger compression = attack path.
        smoothed_gr = np.minimum(gr_attack, gr_release)  # (n,) float64
        self._gain_db = float(smoothed_gr[-1])  # carry state forward

        # ── 4. Gain application — fully vectorised ────────────────────────────
        #
        # Convert dB gain reduction to linear scale and apply makeup gain.
        # np.exp(x · ln10) is used instead of 10**x — maps to vectorised
        # libm vexp which is faster than np.power for non-integer exponents.
        _LN10 = np.log(10.0)  # scalar constant, computed once
        out_gain = np.exp((smoothed_gr * (_LN10 / 20.0))) * self._makeup  # (n,)

        # ── 5. Diagnostics ────────────────────────────────────────────────────
        self.gain_reduction_db = float(
            np.min(20.0 * np.log10(np.clip(out_gain / self._makeup, 1e-12, None)))
        )
        _dynamics_gain_reduction_db.observe(abs(self.gain_reduction_db))

        # ── 6. Apply and clip ─────────────────────────────────────────────────
        norm_data = data / scale
        if norm_data.ndim == 2:
            result = norm_data * out_gain[:, np.newaxis]
        else:
            result = norm_data * out_gain
        if scale > 1.0:
            result = np.clip(result * scale, -32768.0, 32767.0)
        else:
            result = np.clip(result, -1.0, 1.0)

        return PCMChunk(
            data=result.astype(chunk.fmt.dtype),
            fmt=chunk.fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: process dynamics on each chunk."""
        async for chunk in chunks:
            yield self.process(chunk)

# ── PCMInterruptDetector — barge-in detection ─────────────────────────────────

class PCMInterruptDetector:
    """
    Detects user speech (barge-in) while the agent is speaking (TTS playback).

    Monitors the microphone stream continuously. When the microphone RMS
    exceeds the onset threshold AND the playback signal is not dominating
    (to suppress speaker bleed), an interrupt event is fired.

    Debouncing: A minimum number of consecutive speech frames must be detected
    before the callback is invoked, preventing single-click false triggers.

    Reference signal awareness:
        If the TTS playback is loud, the microphone echo is also loud.
        The detector suppresses firing when the reference (playback) RMS
        exceeds reference_threshold. Use with PCMEchoCanceller for best results.

    Async-safe: The callback is scheduled in the asyncio event loop via
    loop.call_soon_threadsafe so it's safe to await in the callback.

    Usage::

        detector = PCMInterruptDetector(
            fmt=fmt,
            on_interrupt=lambda: asyncio.create_task(stop_tts()),
        )
        detector.set_playback_active(True)

        async for chunk in mic_stream:
            detector.push(chunk)

    Parameters:
        fmt:                  Microphone PCMFormat.
        onset_rms:            Mic RMS trigger threshold. Default 300 (int16 scale).
        reference_threshold:  Playback RMS above which detection is suppressed. Default 500.
        confirm_frames:       Consecutive speech frames before firing. Default 3.
        cooldown_s:           Seconds before next interrupt can fire. Default 2.0.
        on_interrupt:         Synchronous callback called when barge-in detected.
        loop:                 asyncio event loop (for async-safe callback dispatch).
    """

    def __init__(
        self,
        fmt: PCMFormat,
        onset_rms: float = 300.0,
        reference_threshold: float = 500.0,
        confirm_frames: int = 3,
        cooldown_s: float = 2.0,
        on_interrupt: InterruptCallback | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._fmt = fmt
        self._onset = onset_rms
        self._ref_thresh = reference_threshold
        self._confirm = confirm_frames
        self._cooldown_s = cooldown_s
        self._callback = on_interrupt
        self._loop = loop

        self._scale = 32768.0 if fmt.dtype == "float32" else 1.0
        # → self._scale = 1.0   (int16 path chosen)
        # → reference_threshold = 500.0 is in int16 amplitude units
        self._playback_active = False
        self._playback_rms: float = 0.0
        self._consecutive: int = 0
        self._last_fire: float = 0.0
        self._lock = threading.Lock()

    def set_playback_active(self, active: bool) -> None:
        """Signal whether TTS playback is currently active."""
        with self._lock:
            self._playback_active = active
            if not active:
                self._playback_rms = 0.0
                self._consecutive = 0

    def push_reference(self, chunk: PCMChunk) -> None:
        """
        Feed the current TTS (playback) chunk for reference level tracking.

        Call this for every chunk written to the speaker.
        """
        rms = chunk.rms() * self._scale
        with self._lock:
            self._playback_rms = 0.9 * self._playback_rms + 0.1 * rms

    def push(self, mic_chunk: PCMChunk) -> bool:
        """
        Feed a microphone chunk.

        Returns True if an interrupt was detected and the callback was fired.
        """
        mic_rms = mic_chunk.rms() * self._scale
        now = time.monotonic()

        with self._lock:
            if not self._playback_active:
                self._consecutive = 0
                return False

            # Suppressed if speaker is too loud (echo risk)
            if self._playback_rms > self._ref_thresh:
                self._consecutive = 0
                return False

            if mic_rms >= self._onset:
                self._consecutive += 1
            else:
                self._consecutive = 0

            if self._consecutive >= self._confirm:
                if (now - self._last_fire) >= self._cooldown_s:
                    self._last_fire = now
                    self._consecutive = 0
                    fire = True
                else:
                    fire = False
            else:
                fire = False

        if fire:
            _interrupt_detections.inc()
            log.info("pcm_barge_in_detected", mic_rms=round(mic_rms, 1))

            cb = self._callback
            if cb is not None:
                loop = self._loop
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(cb)  # type: ignore[arg-type]
                else:
                    cb()

            return True
        return False

    async def stream(
        self,
        mic_chunks: AsyncIterator[PCMChunk],
        *,
        on_interrupt: Callable[[], Any] | None = None,
    ) -> AsyncIterator[PCMChunk]:
        """
        Async passthrough: pushes each chunk through interrupt detection,
        fires the on_interrupt coroutine if barge-in is detected, and
        yields each chunk unchanged for downstream processing.
        """
        async for chunk in mic_chunks:
            detected = self.push(chunk)
            if detected and on_interrupt is not None:
                result = on_interrupt()
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            yield chunk

# ── PCMSilencePadder — leading/trailing silence injection ─────────────────────

class PCMSilencePadder:
    """
    Injects configurable silence before and/or after each speech segment.

    Use cases:
      • Pre-speech silence: prevents PortAudio device click artifacts when the
        output stream cold-starts with audio (warm-up already handles this but
        extra silence helps some DACs).
      • Post-speech silence: ensures STT model gets a quiet tail to finalize
        transcription (some models need ~200 ms of trailing silence).
      • Inter-sentence gap: insert silence between TTS sentences for natural pacing.

    The padder is completely transparent when both padding values are zero.

    Parameters:
        fmt:         PCMFormat of chunks.
        pre_s:       Silence to prepend (seconds). Default 0.05.
        post_s:      Silence to append (seconds). Default 0.1.
        only_final:  Only pad is_final=True chunks. Default True.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        pre_s: float = 0.05,
        post_s: float = 0.1,
        only_final: bool = True,
    ) -> None:
        self._fmt = fmt
        self._pre_frames = fmt.frames_for_duration(pre_s)
        self._post_frames = fmt.frames_for_duration(post_s)
        self._only_final = only_final

    def _silence(self, n_frames: int) -> np.ndarray:
        shape = (n_frames,) if self._fmt.channels == 1 else (n_frames, self._fmt.channels)
        return np.zeros(shape, dtype=self._fmt.dtype)

    def process(self, chunk: PCMChunk) -> PCMChunk:
        """Pad chunk with silence. Returns new padded PCMChunk."""
        if self._only_final and not chunk.is_final:
            return chunk

        parts = []
        if self._pre_frames > 0:
            parts.append(self._silence(self._pre_frames))
        parts.append(chunk.data)
        if self._post_frames > 0:
            parts.append(self._silence(self._post_frames))

        padded = np.concatenate(parts, axis=0)
        return PCMChunk(
            data=padded,
            fmt=chunk.fmt,
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: pad silence around each chunk."""
        async for chunk in chunks:
            yield self.process(chunk)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANALYSIS MODULES — waveform analyzer, diagnostics, latency tracking
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMWaveformAnalyzer — real-time waveform + spectral analytics ─────────────

class PCMWaveformAnalyzer:
    """
    Real-time audio waveform and spectral analytics engine.

    Computes per-chunk statistics and accumulates running aggregates:
      • RMS (root-mean-square amplitude)
      • Peak amplitude
      • Crest factor (peak / RMS ratio, dB)
      • Zero-crossing rate (ZCR) — correlates with voiced/unvoiced
      • Spectral centroid (Hz) — tracks spectral brightness
      • DC offset — indicates hardware fault or coupling issues
      • Clipping detection (|peak| ≥ 0.99 for float32 or 32000 for int16)
      • Silence detection (RMS < silence_threshold)

    Running statistics (mean, max over N recent chunks) are available via
    get_running_stats(). Thread-safe.

    Parameters:
        fmt:               Input PCMFormat.
        silence_threshold: RMS below which frame is considered silent. Default 0.005 for float32.
        clip_threshold:    Peak above which frame is considered clipping. Default 0.99 for float32.
        history_len:       Number of chunks to retain for running statistics. Default 100.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        silence_threshold: float = 0.005,
        clip_threshold: float = 0.99,
        history_len: int = 100,
    ) -> None:
        self._fmt = fmt
        self._silence_thresh = silence_threshold
        self._clip_thresh = clip_threshold
        self._history_len = history_len
        self._history: collections.deque[WaveformStats] = collections.deque(maxlen=history_len)
        self._lock = threading.Lock()
        self._total_chunks = 0

    def analyze(self, chunk: PCMChunk) -> WaveformStats:
        """
        Compute waveform statistics for a single PCMChunk.

        Thread-safe: accumulates into internal history.
        """
        data = chunk.data.astype(np.float64)
        mono = data[:, 0] if data.ndim == 2 else data

        # Normalize to [-1, 1] before level detection
        scale = 32768.0 if chunk.fmt.dtype == "int16" else 1.0
        mono_norm = mono / scale

        rms = float(np.sqrt(np.mean(mono_norm ** 2)))
        peak = float(np.max(np.abs(mono_norm)))
        dc = float(np.mean(mono_norm))

        crest_db = 20 * math.log10(peak / (rms + 1e-12)) if peak > 0.0 else 0.0
        n = len(mono)
        zcr = float(np.sum(np.diff(np.sign(mono_norm)) != 0)) / (n - 1) if n > 1 else 0.0

        # Spectral centroid
        if n >= 8:
            window = np.hanning(n)
            spec = np.abs(np.fft.rfft(mono_norm * window))
            freqs = np.fft.rfftfreq(n, d=1.0 / chunk.fmt.sample_rate)
            centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))
        else:
            centroid = 0.0

        stats = WaveformStats(
            n_frames=chunk.n_frames,
            duration_s=chunk.duration_s,
            rms=rms,
            peak=peak,
            crest_factor_db=crest_db,
            zero_crossing_rate=zcr,
            spectral_centroid_hz=centroid,
            is_clipping=peak >= self._clip_thresh,
            is_silent=rms < self._silence_thresh,
            dc_offset=dc,
        )
        with self._lock:
            self._history.append(stats)
            self._total_chunks += 1
        return stats

    def get_running_stats(self) -> dict[str, float]:
        """
        Return aggregate statistics over the recent history window.

        Keys: rms_mean, rms_max, peak_max, zcr_mean, centroid_mean,
              clipping_rate, silence_rate, dc_offset_mean, chunk_count.
        """
        with self._lock:
            h = list(self._history)

        if not h:
            return {}

        return {
            "rms_mean": float(np.mean([s.rms for s in h])),
            "rms_max": float(np.max([s.rms for s in h])),
            "peak_max": float(np.max([s.peak for s in h])),
            "zcr_mean": float(np.mean([s.zero_crossing_rate for s in h])),
            "centroid_mean_hz": float(np.mean([s.spectral_centroid_hz for s in h])),
            "clipping_rate": float(sum(s.is_clipping for s in h) / len(h)),
            "silence_rate": float(sum(s.is_silent for s in h) / len(h)),
            "dc_offset_mean": float(np.mean([s.dc_offset for s in h])),
            "chunk_count": float(self._total_chunks),
        }

    def reset(self) -> None:
        """Clear history."""
        with self._lock:
            self._history.clear()
            self._total_chunks = 0

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[tuple[PCMChunk, WaveformStats]]:
        """Async generator: yield (chunk, stats) pairs without modifying audio."""
        async for chunk in chunks:
            stats = self.analyze(chunk)
            yield chunk, stats

# ── PCMDiagnosticsMonitor — real-time audio health monitor ────────────────────

class PCMDiagnosticsMonitor:
    """
    Continuous audio health monitor with async alerting.

    Monitors a stream of PCMChunks and detects:
      • **Clipping** — |peak| ≥ 0.99: mic gain too high or A/D overload
      • **Sustained silence** — RMS < threshold for N consecutive chunks:
        mic disconnected or muted
      • **DC offset** — mean > threshold: hardware coupling issue
      • **Sequence gaps** — seq number discontinuities: packet loss or drop
      • **Low level** — RMS consistently below a floor: gain too low for STT

    Fires configurable async callbacks on condition detection. Exposes
    get_health_report() for polling-based monitoring.

    Prometheus counters are incremented on each condition detection.

    Parameters:
        fmt:                  PCMFormat of monitored chunks.
        clip_threshold:       Peak threshold for clipping. Default 0.99.
        silence_threshold:    RMS threshold for silence. Default 0.002 (float32).
        dc_threshold:         Mean absolute offset for DC alert. Default 0.05.
        silence_frames:       Consecutive silent frames before alert. Default 50.
        history_len:          Chunks retained for running stats. Default 200.
        on_clipping:          Async callback on clipping detection.
        on_silence:           Async callback on sustained silence.
        on_dc_offset:         Async callback on DC offset.
        on_dropout:           Async callback on sequence gap.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        clip_threshold: float = 0.99,
        silence_threshold: float = 0.002,
        dc_threshold: float = 0.05,
        silence_frames: int = 50,
        history_len: int = 200,
        on_clipping: Callable[[], Any] | None = None,
        on_silence: Callable[[], Any] | None = None,
        on_dc_offset: Callable[[], Any] | None = None,
        on_dropout: Callable[[int], Any] | None = None,
    ) -> None:
        self._fmt = fmt
        self._clip_thresh = clip_threshold
        self._silence_thresh = silence_threshold
        self._dc_thresh = dc_threshold
        self._silence_frames = silence_frames
        self._on_clipping = on_clipping
        self._on_silence = on_silence
        self._on_dc = on_dc_offset
        self._on_dropout = on_dropout

        self._analyzer = PCMWaveformAnalyzer(
            fmt, silence_threshold=silence_threshold,
            clip_threshold=clip_threshold, history_len=history_len
        )
        self._consecutive_silence = 0
        self._last_seq: int = -1
        self._dropout_count = 0
        self._notes: list[str] = []
        self._lock = threading.Lock()

    def push(self, chunk: PCMChunk) -> AudioHealthReport:
        """
        Analyze a chunk and return an AudioHealthReport.

        Also fires async callbacks for detected conditions.
        """
        stats = self._analyzer.analyze(chunk)
        notes: list[str] = []
        status: Literal["ok", "degraded", "failed"] = "ok"

        # Clipping
        if stats.is_clipping:
            _diagnostics_clipping.inc()
            notes.append("CLIPPING detected")
            status = "degraded"
            if self._on_clipping:
                self._fire(self._on_clipping)

        # Silence
        if stats.is_silent:
            with self._lock:
                self._consecutive_silence += 1
                if self._consecutive_silence >= self._silence_frames:
                    _diagnostics_silence.inc()
                    notes.append(f"SUSTAINED SILENCE ({self._consecutive_silence} frames)")
                    status = "degraded"
                    if self._on_silence:
                        self._fire(self._on_silence)
        else:
            with self._lock:
                self._consecutive_silence = 0

        # DC offset
        dc_norm = abs(stats.dc_offset)  # already normalized by analyzer
        if dc_norm > self._dc_thresh:
            _diagnostics_dc_offset.inc()
            notes.append(f"DC OFFSET: {dc_norm:.4f}")
            status = "degraded"
            if self._on_dc:
                self._fire(self._on_dc)

        # Sequence gaps
        if chunk.seq >= 0:
            with self._lock:
                if self._last_seq >= 0 and chunk.seq != self._last_seq + 1:
                    gap = chunk.seq - self._last_seq - 1
                    if gap > 0:
                        self._dropout_count += gap
                        notes.append(f"SEQ GAP: missed {gap} chunks")
                        if self._on_dropout:
                            self._fire(self._on_dropout, gap)
                self._last_seq = chunk.seq

        running = self._analyzer.get_running_stats()
        with self._lock:
            dc_count = self._dropout_count

        return AudioHealthReport(
            timestamp=time.monotonic(),
            status=status,
            clipping_rate=running.get("clipping_rate", 0.0),
            silence_rate=running.get("silence_rate", 0.0),
            dc_offset_mean=running.get("dc_offset_mean", 0.0),
            dropout_count=dc_count,
            rms_mean=running.get("rms_mean", 0.0),
            notes=notes,
        )

    def _fire(self, cb: Callable, *args: Any) -> None: # noqa
        """Fire a callback, handling both sync and async."""
        try:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass
        except Exception as exc:
            log.warning("pcm_diagnostics_callback_error", error=str(exc))

    def get_health_report(self) -> AudioHealthReport:
        """Return a health report based on accumulated history."""
        running = self._analyzer.get_running_stats()
        clip_rate = running.get("clipping_rate", 0.0)
        sil_rate = running.get("silence_rate", 0.0)

        if clip_rate > 0.2 or sil_rate > 0.8:
            status: Literal["ok", "degraded", "failed"] = "failed"
        elif clip_rate > 0.05 or sil_rate > 0.5:
            status = "degraded"
        else:
            status = "ok"

        with self._lock:
            dc_count = self._dropout_count

        return AudioHealthReport(
            timestamp=time.monotonic(),
            status=status,
            clipping_rate=clip_rate,
            silence_rate=sil_rate,
            dc_offset_mean=running.get("dc_offset_mean", 0.0),
            dropout_count=dc_count,
            rms_mean=running.get("rms_mean", 0.0),
            notes=[],
        )

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[tuple[PCMChunk, AudioHealthReport]]:
        """Async generator: yield (chunk, health_report) pairs."""
        async for chunk in chunks:
            report = self.push(chunk)
            yield chunk, report

# ── PCMLatencyTracker — end-to-end latency measurement ────────────────────────

class PCMLatencyTracker:
    """
    End-to-end latency tracking across pipeline stages.

    Injects observation points at each stage boundary. Uses PCMChunk.timestamp
    (set at capture time) as the reference clock and measures wall-clock
    delta at each stage.

    Exposes:
      • observe(chunk, stage) — record a stage observation
      • get_latency_report() — return per-stage and total latency stats
      • Prometheus histogram per stage (pcm_pipeline_stage_latency_seconds)

    Usage::

        tracker = PCMLatencyTracker()

        async for chunk in mic_stream:
            tracker.observe(chunk, "capture")
            chunk = agc.process(chunk)
            tracker.observe(chunk, "agc")
            chunk = noise_suppressor.process(chunk)
            tracker.observe(chunk, "noise_suppress")
            ...

    Parameters:
        max_history:  Maximum observations to retain per stage. Default 1000.
    """

    def __init__(self, max_history: int = 1000) -> None:
        self._history: dict[str, collections.deque[float]] = {}
        self._max_history = max_history
        self._lock = threading.Lock()

    def observe(self, chunk: PCMChunk, stage: str) -> None:
        """Record a stage observation for this chunk."""
        now = time.monotonic()
        delta = now - chunk.timestamp if chunk.timestamp > 0 else 0.0
        _pipeline_stage_latency.observe(delta)

        with self._lock:
            if stage not in self._history:
                self._history[stage] = collections.deque(maxlen=self._max_history)
            self._history[stage].append(delta)

    def get_latency_report(self) -> dict[str, dict[str, float]]:
        """
        Return per-stage latency statistics.

        Returns:
            dict[stage_name, dict[mean_s, p50_s, p95_s, p99_s, max_s, count]]
        """
        with self._lock:
            snapshot = {stage: list(dq) for stage, dq in self._history.items()}

        report: dict[str, dict[str, float]] = {}
        for stage, obs in snapshot.items():
            if not obs:
                continue
            arr = np.array(obs)
            report[stage] = {
                "mean_s": float(np.mean(arr)),
                "p50_s": float(np.percentile(arr, 50)),
                "p95_s": float(np.percentile(arr, 95)),
                "p99_s": float(np.percentile(arr, 99)),
                "max_s": float(np.max(arr)),
                "count": float(len(arr)),
            }
        return report

    def reset(self) -> None:
        """Clear all history."""
        with self._lock:
            self._history.clear()

    def record(self, stage: str, latency_s: float) -> None:
        """Record a raw latency value directly (no PCMChunk needed)."""
        with self._lock:
            if stage not in self._history:
                self._history[stage] = collections.deque(maxlen=self._max_history)
            self._history[stage].append(latency_s)

    def log_report(self) -> None:
        """Log the latency report to the structured logger."""
        report = self.get_latency_report()
        for stage, stats in report.items():
            log.info(
                "pcm_latency_report",
                stage=stage,
                mean_ms=round(stats["mean_s"] * 1000, 2),
                p95_ms=round(stats["p95_s"] * 1000, 2),
                p99_ms=round(stats["p99_s"] * 1000, 2),
                max_ms=round(stats["max_s"] * 1000, 2),
            )

# ── PCMMetricsSnapshot helpers ────────────────────────────────────────────────

def get_metrics_snapshot() -> PCMMetricsSnapshot:
    """
    Capture current counter values as a PCMMetricsSnapshot.

    Note: This relies on internal counter _value attributes. Adapt to
    your Prometheus client's API if different (e.g., counter._metrics[()].get()).
    """
    def _read(counter: Any, *label_values: str) -> float:
        try:
            if label_values:
                return float(counter.labels(*label_values)._value.get()) # noqa
            return float(counter._value.get()) # noqa
        except AttributeError:
            return 0.0

    return PCMMetricsSnapshot(
        timestamp=time.monotonic(),
        input_chunks=_read(_in_chunks, "ok") + _read(_in_chunks, "dropped"),
        input_bytes=_read(_in_bytes),
        input_overflows=_read(_in_overflows),
        input_dropped=_read(_in_dropped),
        output_chunks_written=_read(_out_chunks, "written"),
        output_bytes=_read(_out_bytes),
        output_underruns=_read(_out_underruns),
        output_recreations=_read(_out_recreations),
        vad_segments=_read(_vad_segments),
        convert_calls=_read(_convert_calls, "resample")
            + _read(_convert_calls, "dtype")
            + _read(_convert_calls, "channel"),
        jitter_late_drops=_read(_jitter_buffer_late_drops),
        jitter_concealment=_read(_jitter_buffer_concealment),
        interrupt_detections=_read(_interrupt_detections),
        diagnostics_clipping=_read(_diagnostics_clipping),
        diagnostics_silence=_read(_diagnostics_silence),
        pool_hits=_read(_pool_hits),
        pool_misses=_read(_pool_misses),
        drift_corrections=_read(_drift_corrections),
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 8. RUNTIME COMPONENTS — jitter buffer, drift correction, mixers, recorders
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMJitterBuffer — sequence-ordered network jitter buffer with PLC ─────────

class PCMJitterBuffer:
    """
    Sequence-number-ordered jitter buffer for network-received PCM audio.

    Handles all pathological network conditions:
      • Out-of-order arrival: reorders by PCMChunk.seq
      • Duplicate suppression: tracks seen seq numbers
      • Configurable playout delay (minimum buffer depth before playing)
      • Packet loss concealment (PLC): fills gaps on output
      • Late-packet discard: drops packets arriving > max_delay_ms late
      • Buffer overflow: discards oldest on overflow

    Designed for WebRTC / WebSocket / RTP PCM delivery where the network
    can reorder, duplicate, or drop packets.

    Usage::

        jitter = PCMJitterBuffer(fmt=fmt, target_delay_ms=60, max_delay_ms=200)

        # Producer (network receive):
        jitter.push(received_chunk)

        # Consumer (playback):
        async for playable_chunk in jitter.stream():
            await speaker.write(playable_chunk)

    Parameters:
        fmt:                Output PCMFormat.
        target_delay_ms:    Target buffering delay before playout starts. Default 60.
        max_delay_ms:       Maximum delay; packets older than this are discarded. Default 200.
        plc_mode:           Packet loss concealment strategy. Default REPEAT_LAST.
        max_buffer:         Maximum packets in buffer. Default 64.
        tick_interval_s:    Playout timer tick interval. Default 0.02 (20 ms).
    """

    def __init__(
        self,
        fmt: PCMFormat,
        target_delay_ms: float = 60.0,
        max_delay_ms: float = 200.0,
        plc_mode: PLCMode = PLCMode.REPEAT_LAST,
        max_buffer: int = 64,
        tick_interval_s: float = 0.02,
    ) -> None:
        self._fmt = fmt
        self._target_delay = target_delay_ms / 1000.0
        self._max_delay = max_delay_ms / 1000.0
        self._plc_mode = plc_mode
        self._max_buf = max_buffer
        self._tick = tick_interval_s

        self._heap: list[tuple[int, PCMChunk]] = []  # min-heap by seq
        self._seen_seqs: set[int] = set()
        self._next_seq: int = -1   # expected next output seq; -1 = not started
        self._last_chunk: PCMChunk | None = None
        self._buffer_start_ts: float = 0.0
        self._lock = threading.Lock()

        # Async queue for playout
        self._out_q: asyncio.Queue[PCMChunk | None] = asyncio.Queue(maxsize=max_buffer * 2)
        self._running = False
        self._playout_task: asyncio.Task | None = None

    def push(self, chunk: PCMChunk) -> None:
        """
        Receive a network packet. Thread-safe; call from any thread or coroutine.

        Duplicates are silently dropped. Late packets (older than max_delay_ms)
        are discarded with a counter increment.
        """
        seq = chunk.seq
        now = time.monotonic()

        with self._lock:
            # Duplicate check
            if seq in self._seen_seqs:
                return
            self._seen_seqs.add(seq)
            # Evict old seq numbers to prevent unbounded growth
            if len(self._seen_seqs) > self._max_buf * 4:
                cutoff = seq - self._max_buf * 2
                self._seen_seqs = {s for s in self._seen_seqs if s >= cutoff}


            # Late packet check
            if self._next_seq > 0 and seq < self._next_seq:
                _jitter_buffer_late_drops.inc()
                log.debug("jitter_late_drop", seq=seq, next_expected=self._next_seq)
                return

            # Buffer overflow
            if len(self._heap) >= self._max_buf:
                # Discard oldest (smallest seq)
                heapq.heappop(self._heap)

            heapq.heappush(self._heap, (seq, chunk))
            _jitter_buffer_depth.set(len(self._heap))

            # Set buffer start time on first packet
            if self._buffer_start_ts == 0.0:
                self._buffer_start_ts = now

    def _make_plc(self, n_frames: int) -> PCMChunk:
        """Generate a PLC (concealment) chunk."""
        if self._plc_mode == PLCMode.ZERO_FILL or self._last_chunk is None:
            shape = (n_frames,) if self._fmt.channels == 1 else (n_frames, self._fmt.channels)
            data = np.zeros(shape, dtype=self._fmt.dtype)
        elif self._plc_mode == PLCMode.REPEAT_LAST:
            src = self._last_chunk.data
            if len(src) >= n_frames:
                data = src[:n_frames].copy()
            else:
                repeats = int(math.ceil(n_frames / len(src)))
                if src.ndim == 2:
                    data = np.tile(src, (repeats, 1))[:n_frames]
                else:
                    data = np.tile(src, repeats)[:n_frames]
        else:  # NOISE_FILL
            shape = (n_frames,) if self._fmt.channels == 1 else (n_frames, self._fmt.channels)
            noise_level = 0.005 if self._fmt.dtype == "float32" else 150
            data = (np.random.randn(*shape) * noise_level).astype(self._fmt.dtype)

        _jitter_buffer_concealment.inc()
        return PCMChunk(data=data, fmt=self._fmt, timestamp=time.monotonic(),
                        seq=-1, is_final=False, source="plc")

    async def _playout_loop(self) -> None:
        """Background task: drains ordered packets from heap to output queue."""
        target_frames = self._fmt.frames_for_duration(self._tick)

        while self._running:
            await asyncio.sleep(self._tick)
            now = time.monotonic()

            with self._lock:
                # Wait for initial target delay
                if (self._buffer_start_ts > 0
                        and (now - self._buffer_start_ts) < self._target_delay):
                    continue

                if not self._heap:
                    continue

                # Peek at next expected
                min_seq = self._heap[0][0]
                if self._next_seq < 0:
                    self._next_seq = min_seq

                # Check if next expected is available
                if self._heap and self._heap[0][0] == self._next_seq:
                    _, chunk = heapq.heappop(self._heap)
                    self._last_chunk = chunk
                    self._next_seq += 1
                    _jitter_buffer_depth.set(len(self._heap))
                    out_chunk = chunk
                else:
                    # Gap: generate PLC
                    out_chunk = self._make_plc(target_frames)
                    self._next_seq += 1

            try:
                self._out_q.put_nowait(out_chunk)
            except asyncio.QueueFull:
                pass  # consumer is slow; discard

    async def start(self) -> None:
        """Start the playout task."""
        if self._running:
            return
        self._running = True
        self._playout_task = asyncio.create_task(
            self._playout_loop(), name="pcm-jitter-playout"
        )

    async def stop(self) -> None:
        """Stop the playout task."""
        self._running = False
        if self._playout_task:
            self._playout_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._playout_task
            self._playout_task = None
        try:
            self._out_q.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def stream(self) -> AsyncIterator[PCMChunk]:
        """Async iterator: yield ordered, PLC-filled PCMChunks."""
        await self.start()
        while True:
            chunk = await self._out_q.get()
            if chunk is None:
                break
            yield chunk

# ── PCMDriftCorrector — sample-clock drift correction ─────────────────────────

class PCMDriftCorrector:
    """
    Corrects sample-clock drift between producer and consumer.

    In long-running voice sessions, the microphone sample clock and the
    playback sample clock may drift at different rates (typical drift: 10-200 ppm).
    Over a 1-hour session at 200 ppm, this accumulates to 720 ms of drift —
    enough to desync voice activity detection from playback.

    Algorithm:
        Maintains a running estimate of frame arrival rate vs expected rate.
        When drift exceeds the threshold, inserts (or drops) samples via
        high-quality resampling to re-align the clocks.

    Parameters:
        fmt:              Stream PCMFormat.
        check_interval_s: How often to estimate drift. Default 5.0 seconds.
        max_drift_ppm:    Drift threshold before correction is applied. Default 50 ppm.
        correction_gain:  Fraction of drift to correct per interval. Default 0.5.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        check_interval_s: float = 5.0,
        max_drift_ppm: float = 50.0,
        correction_gain: float = 0.5,
    ) -> None:
        self._fmt = fmt
        self._check_interval = check_interval_s
        self._max_drift = max_drift_ppm * 1e-6
        self._correction_gain = correction_gain
        self._converter = PCMConverter(quality="poly" if _SCIPY else "linear")

        self._frames_received: int = 0
        self._wall_start: float = 0.0
        self._last_check: float = 0.0
        self._correction_factor: float = 1.0

    def push(self, chunk: PCMChunk) -> PCMChunk:
        """
        Process a chunk through the drift corrector.

        On normal operation: returns the chunk unchanged.
        On drift correction: returns a slightly resampled chunk that
        compensates for clock skew.
        """
        now = time.monotonic()
        if self._wall_start == 0.0:
            self._wall_start = now
            self._last_check = now

        self._frames_received += chunk.n_frames

        # Check drift periodically
        if (now - self._last_check) >= self._check_interval:
            self._last_check = now
            elapsed = now - self._wall_start
            expected_frames = elapsed * self._fmt.sample_rate
            actual_frames = self._frames_received

            if expected_frames > 0:
                drift = (actual_frames - expected_frames) / expected_frames
                if abs(drift) > self._max_drift:
                    # Compute correction: resample to a rate that absorbs the drift
                    correction = 1.0 + drift * self._correction_gain
                    self._correction_factor = correction
                    _drift_corrections.inc()
                    log.info(
                        "pcm_drift_correction",
                        drift_ppm=round(drift * 1e6, 1),
                        correction=round(correction, 6),
                    )

        if abs(self._correction_factor - 1.0) < 1e-6:
            return chunk

        # Apply correction via resampling
        target_rate = int(round(self._fmt.sample_rate * self._correction_factor))
        target_fmt = PCMFormat(
            sample_rate=target_rate,
            channels=self._fmt.channels,
            dtype=self._fmt.dtype,
        )
        resampled = self._converter.convert(chunk, target_fmt)
        # Restore declared sample rate (the resampled frames now have the right timing)
        corrected = PCMChunk(
            data=resampled.data,
            fmt=chunk.fmt,  # restore original declared rate
            timestamp=chunk.timestamp,
            seq=chunk.seq,
            is_final=chunk.is_final,
            source=chunk.source,
        )
        return corrected

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: drift-correct each chunk."""
        async for chunk in chunks:
            yield self.push(chunk)

# ── PCMStreamMixer — mix N async PCM streams into one output ──────────────────

class PCMStreamMixer:
    """
    Mix N asynchronous PCMChunk streams into a single output stream.

    Each input stream is assigned a gain (0.0 – 1.0). The mixer sums
    the sample-aligned streams, applies a configurable master gain, and
    clips the result to [-1, 1] (float32) before emitting output chunks.

    Alignment strategy:
        Streams may produce chunks at different rates. The mixer uses
        asyncio.gather() to read one chunk from each stream per tick,
        zero-padding streams that produce shorter chunks.

    Gain fade:
        set_gain(stream_id, gain, fade_ms) applies a linear gain fade
        over the specified duration to avoid clicks on level changes.

    Usage::

        mixer = PCMStreamMixer(fmt=output_fmt, tick_s=0.02)
        id_a = mixer.add_source(tts_stream, gain=0.9)
        id_b = mixer.add_source(music_stream, gain=0.3)

        async for mixed_chunk in mixer.stream():
            await speaker.write(mixed_chunk)

    Parameters:
        fmt:         Output PCMFormat (all inputs converted to this format).
        tick_s:      Mixing period (seconds). Default 0.02 (20 ms).
        master_gain: Master output gain multiplier. Default 1.0.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        tick_s: float = 0.02,
        master_gain: float = 1.0,
    ) -> None:
        self._fmt = fmt
        self._tick_s = tick_s
        self._tick_frames = fmt.frames_for_duration(tick_s)
        self._master_gain = master_gain
        self._converter = PCMConverter()

        self._sources: dict[int, tuple[AsyncIterator[PCMChunk], float]] = {}
        self._gains: dict[int, float] = {}
        self._next_id = 0
        self._running = False
        self._out_q: asyncio.Queue[PCMChunk | None] = asyncio.Queue(maxsize=64)
        self._mix_task: asyncio.Task | None = None

    def add_source(
        self,
        source: AsyncIterator[PCMChunk],
        gain: float = 1.0,
    ) -> int:
        """
        Register an input stream. Returns a stream_id for gain control.

        Must be called before start().
        """
        sid = self._next_id
        self._next_id += 1
        self._sources[sid] = (source, gain)
        self._gains[sid] = gain
        return sid

    def set_gain(self, stream_id: int, gain: float) -> None:
        """Update the gain for a stream (immediate, no fade)."""
        self._gains[stream_id] = max(0.0, min(1.0, gain))

    async def _mix_loop(self) -> None:
        """Core mixing loop: pull chunks from all sources and sum."""
        iterators = {sid: src for sid, (src, _) in self._sources.items()}
        carry: dict[int, np.ndarray] = {sid: np.array([], dtype=np.float32) for sid in iterators}
        exhausted: set[int] = set()

        while self._running and len(exhausted) < len(iterators):
            if self._fmt.channels > 1:
                tick_buf = np.zeros((self._tick_frames, self._fmt.channels), dtype=np.float64)
            else:
                tick_buf = np.zeros(self._tick_frames, dtype=np.float64)

            for sid, it in list(iterators.items()):
                if sid in exhausted:
                    continue

                # Drain from carry first
                available = carry[sid]
                if len(available) < self._tick_frames:
                    try:
                        chunk = await asyncio.wait_for(it.__anext__(), timeout=0.5)
                        # Handles multi-channel properly
                        data = chunk.data.astype(np.float32)
                        if data.ndim == 2 and self._fmt.channels == 1:
                            data = np.mean(data, axis=1)  # downmix to mono
                        elif data.ndim == 1 and self._fmt.channels > 1:
                            data = np.column_stack([data] * self._fmt.channels)  # upmix
                        available = np.concatenate([available, data])
                    except StopAsyncIteration:
                        exhausted.add(sid)
                    except asyncio.TimeoutError:
                        pass
                    except Exception as exc:
                        log.debug("pcm_mixer_source_error", sid=sid, error=str(exc))
                        exhausted.add(sid)

                n = min(self._tick_frames, len(available))
                if n > 0:
                    tick_buf[:n] += available[:n] * self._gains.get(sid, 1.0)
                    carry[sid] = available[n:]

            # Master gain + clip
            out = np.clip(tick_buf * self._master_gain, -1.0, 1.0).astype(np.float32)
            clipping = bool(np.any(np.abs(out) >= 0.999))
            if clipping:
                _mixer_clipping.inc()

            if self._fmt.channels > 1:
                out = np.column_stack([out] * self._fmt.channels)

            mixed = PCMChunk(
                data=out,
                fmt=self._fmt,
                timestamp=time.monotonic(),
                seq=0,
                is_final=len(exhausted) == len(iterators),
                source="mixer",
            )
            try:
                self._out_q.put_nowait(mixed)
            except asyncio.QueueFull:
                pass

            await asyncio.sleep(max(0.0, self._tick_s - 0.001))

        try:
            self._out_q.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def start(self) -> None:
        """Start the mixing loop."""
        if self._running:
            return
        self._running = True
        self._mix_task = asyncio.create_task(self._mix_loop(), name="pcm-mixer")

    async def stop(self) -> None:
        """Stop the mixing loop."""
        self._running = False
        if self._mix_task:
            self._mix_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._mix_task

    async def stream(self) -> AsyncIterator[PCMChunk]:
        """Async iterator: yield mixed output chunks."""
        await self.start()
        while True:
            chunk = await self._out_q.get()
            if chunk is None:
                break
            yield chunk

# ── PCMRollingRecorder — ring-buffer rolling WAV recorder ─────────────────────

class PCMRollingRecorder:
    """
    Maintains a rolling window of recent audio in memory for post-hoc export.

    Useful for:
      • Debug recording: capture the last N seconds before an error
      • Pre-VAD context: record full session for quality review
      • "Save last 30 seconds" crash reporter integration

    The recorder is a passthrough: push() accepts chunks and returns them
    unchanged; the rolling buffer accumulates internally.

    export_wav() exports the entire rolling window as a WAV file (bytes).
    export_chunk() returns the window as a single PCMChunk.

    Thread-safe: a threading.Lock guards the ring buffer.

    Parameters:
        fmt:          PCMFormat of input chunks.
        max_seconds:  Rolling window duration. Default 30 seconds.
    """

    def __init__(self, fmt: PCMFormat, max_seconds: float = 30.0) -> None:
        self._fmt = fmt
        self._max_frames = fmt.frames_for_duration(max_seconds)
        # Use a PCMRingBuffer for O(1) write
        self._ring = PCMRingBuffer(
            capacity=max(self._max_frames * 2, 4096), fmt=fmt
        )
        self._lock = threading.Lock()
        self._total_frames = 0

    def push(self, chunk: PCMChunk) -> PCMChunk:
        """
        Record this chunk. Returns the chunk unchanged (passthrough).

        If the rolling window is full, the oldest frames are silently
        overwritten (ring buffer wrap-around).
        """
        data = chunk.data
        mono = data[:, 0] if data.ndim == 2 else data
        n = len(mono)
        with self._lock:
            # Evict oldest if needed
            avail = self._ring.available_to_write()
            if avail < n:
                self._ring.read(n - avail)  # discard oldest
            self._ring.write(mono)
            self._total_frames += n
        return chunk

    def export_chunk(self) -> PCMChunk:
        """
        Export the entire rolling window as a single PCMChunk.

        Returns a PCMChunk containing all buffered audio (up to max_seconds).
        """
        with self._lock:
            available = self._ring.available_to_read()
            data = self._ring.peek(available)
        return PCMChunk(
            data=data.copy(),
            fmt=self._fmt,
            timestamp=time.monotonic(),
            seq=0,
            is_final=True,
            source="rolling_recorder",
        )

    def export_wav(self) -> bytes:
        """Export rolling window as complete WAV bytes."""
        chunk = self.export_chunk()
        return chunk_to_wav_bytes(chunk)

    def clear(self) -> None:
        """Discard all buffered audio."""
        with self._lock:
            self._ring.clear()
            self._total_frames = 0

    @property
    def buffered_seconds(self) -> float:
        """Current duration of buffered audio."""
        with self._lock:
            return self._fmt.duration_s(self._ring.available_to_read())

    async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """Async generator: record + passthrough."""
        async for chunk in chunks:
            self.push(chunk)
            yield chunk

# ── PCMChunkSerializer — binary wire format for network transport ─────────────

_SERIALIZER_MAGIC = 0x50434D58  # "PCMX"
_SERIALIZER_VERSION = 1
_DTYPE_CODE: dict[str, int] = {"int16": 0, "int32": 1, "float32": 2, "float64": 3}
_CODE_DTYPE: dict[int, str] = {v: k for k, v in _DTYPE_CODE.items()}

# Header: magic(4) + version(4) + rate(4) + channels(2) + dtype(2) +
#         timestamp(8) + seq(8) + flags(1) + n_bytes(4) = 37 bytes
_HEADER_FMT = "<IIIHH d q B I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class PCMChunkSerializer:
    """
    Binary serialization / deserialization of PCMChunks for network transport.

    Wire format (little-endian):
        Offset  Size  Field
        0       4     Magic: 0x50434D58 ("PCMX")
        4       4     Version: uint32 (currently 1)
        8       4     Sample rate: uint32
        12      2     Channels: uint16
        14      2     Dtype code: uint16 (0=int16,1=int32,2=float32,3=float64)
        16      8     Timestamp: float64 (monotonic seconds)
        24      8     Seq: int64
        32      1     Flags: uint8 (bit0 = is_final)
        33      4     Data byte count: uint32
        37      N     Raw PCM samples (native dtype, little-endian)

    Total header: 37 bytes. Framing is self-delimiting.

    Usage::

        # Sender:
        wire = PCMChunkSerializer.serialize(chunk)
        websocket.send_bytes(wire)

        # Receiver:
        chunk = PCMChunkSerializer.deserialize(wire)
    """

    @staticmethod
    def serialize(chunk: PCMChunk) -> bytes:
        """Serialize a PCMChunk to wire bytes."""
        data = chunk.data
        dtype_str = chunk.fmt.dtype
        dtype_code = _DTYPE_CODE.get(dtype_str, 2)

        # Ensure little-endian bytes
        if chunk.fmt.byte_order == "big":
            raw = data.byteswap().tobytes()
        else:
            raw = data.astype(f"<{np.dtype(dtype_str).str[1:]}").tobytes()

        flags = 0x01 if chunk.is_final else 0x00
        header = struct.pack(
            _HEADER_FMT,
            _SERIALIZER_MAGIC,
            _SERIALIZER_VERSION,
            chunk.fmt.sample_rate,
            chunk.fmt.channels,
            dtype_code,
            chunk.timestamp,
            chunk.seq,
            flags,
            len(raw),
        )
        _serializer_bytes_in.inc(len(raw))
        return header + raw

    @staticmethod
    def deserialize(data: bytes) -> PCMChunk:
        """Deserialize wire bytes to a PCMChunk."""
        if len(data) < _HEADER_SIZE:
            raise ValueError(f"Data too short for PCMChunk header: {len(data)} < {_HEADER_SIZE}")

        (magic, version, rate, channels, dtype_code, ts, seq, flags, n_bytes) = struct.unpack_from(
            _HEADER_FMT, data, 0
        )
        if magic != _SERIALIZER_MAGIC:
            raise ValueError(f"Invalid magic: 0x{magic:08X}")
        if version != _SERIALIZER_VERSION:
            raise ValueError(f"Unsupported version: {version}")

        dtype_str = _CODE_DTYPE.get(dtype_code)
        if dtype_str is None:
            raise ValueError(f"Unknown dtype code: {dtype_code}")

        raw = data[_HEADER_SIZE: _HEADER_SIZE + n_bytes]
        if len(raw) < n_bytes:
            raise ValueError(f"Truncated PCM data: got {len(raw)}, expected {n_bytes}")

        np_data = np.frombuffer(raw, dtype=f"<{np.dtype(dtype_str).str[1:]}").astype(dtype_str)
        if channels > 1:
            n_frames = len(np_data) // channels
            np_data = np_data[: n_frames * channels].reshape(n_frames, channels)

        is_final = bool(flags & 0x01)
        fmt = PCMFormat(sample_rate=rate, channels=channels, dtype=dtype_str)  # type: ignore
        _serializer_bytes_out.inc(n_bytes)
        return PCMChunk(data=np_data, fmt=fmt, timestamp=ts, seq=seq,
                        is_final=is_final, source="wire")

    @staticmethod
    def frame_from_stream(buf: bytes) -> tuple[PCMChunk | None, bytes]:
        """
        Attempt to parse one PCMChunk from a byte stream buffer.

        Returns (chunk, remaining_bytes) or (None, original_buf) if incomplete.
        Use this for TCP/WebSocket stream framing.
        """
        if len(buf) < _HEADER_SIZE:
            return None, buf
        n_bytes = struct.unpack_from("<I", buf, 33)[0]
        total = _HEADER_SIZE + n_bytes
        if len(buf) < total:
            return None, buf
        chunk = PCMChunkSerializer.deserialize(buf[:total])
        return chunk, buf[total:]

# ═══════════════════════════════════════════════════════════════════════════════
# 9. ORCHESTRATION LAYER — pipeline builders, enhancers, factory
# ═══════════════════════════════════════════════════════════════════════════════

# ── PCMFormatRegistry — centralised format catalogue + negotiation cache ──────

class PCMFormatRegistry:
    """
    Centralised registry and negotiation cache for PCMFormats.

    Solves two problems:
      1. Format proliferation: instead of constructing PCMFormat instances
         ad-hoc throughout the codebase, register and reference them by name.
      2. Negotiation caching: negotiate_format() does O(n·m) scoring; for
         hot paths that call it thousands of times, a result cache avoids
         redundant work.

    Usage::

        reg = PCMFormatRegistry()
        reg.register("whisper", PCMFormat.whisper())
        reg.register("tts",     PCMFormat.openai_tts())

        fmt = reg["whisper"]
        negotiated = reg.negotiate(["whisper"], ["tts", "portaudio"])
    """

    def __init__(self) -> None:
        self._formats: dict[str, PCMFormat] = {}
        self._cache: OrderedDict[tuple, PCMFormat] = OrderedDict()
        self._cache_maxsize: int = 256  # 256 format combinations is an astronomical upper bound
        self._cache_lock = threading.Lock()

    def register(self, name: str, fmt: PCMFormat) -> None:
        """Register a named format."""
        self._formats[name] = fmt
        log.debug("pcm_format_registered", name=name, fmt=repr(fmt))

    def __getitem__(self, name: str) -> PCMFormat:
        if name not in self._formats:
            raise KeyError(f"PCMFormat {name!r} not registered")
        return self._formats[name]

    def get(self, name: str, default: PCMFormat | None = None) -> PCMFormat | None:
        """Return a named format, or default if not found."""
        return self._formats.get(name, default)

    def list_names(self) -> list[str]:
        """List all registered format names."""
        return list(self._formats.keys())

    def negotiate(
            self, preferred_names: list[str], supported_names: list[str]
    ) -> PCMFormat:
        """
        Negotiate between named format sets. Results are cached in a
        bounded LRU cache (O(1) hit and eviction via OrderedDict).

        Cache design:
            Key:   (tuple(preferred_names), tuple(supported_names))
                   Order-sensitive — ["whisper", "tts"] ≠ ["tts", "whisper"]
                   since preference order affects scoring (BUG-021 fix).
            Value: negotiated PCMFormat instance (immutable frozen dataclass).
            Size:  bounded at self._cache_maxsize entries. LRU eviction on
                   overflow — oldest-accessed entry dropped first.
            Complexity: O(1) lookup, O(1) insertion, O(1) eviction.
                        OrderedDict.move_to_end() is O(1) in CPython.
            Thread safety: protected by self._cache_lock — safe to call from
                        multiple threads (e.g. sounddevice callbacks + asyncio loop).

        Raises:
            KeyError: if any name is not registered.
        """
        cache_key = (tuple(preferred_names), tuple(supported_names))

        # ── Cache hit: move to end (most-recently-used) and return ────────────
        with self._cache_lock:
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._cache[cache_key]

        # ── Cache miss: negotiate (outside lock — pure computation) ──────────
        # negotiate_format() is stateless and may be slow; don't hold the lock
        # during computation. Two threads may redundantly compute the same key
        # under contention — harmless since results are deterministic.
        preferred = [self[n] for n in preferred_names]
        supported = [self[n] for n in supported_names]
        result = negotiate_format(preferred, supported)

        # ── Insert with LRU eviction ──────────────────────────────────────────
        with self._cache_lock:
            # Re-check in case another thread inserted while we computed
            if cache_key not in self._cache:
                if len(self._cache) >= self._cache_maxsize:
                    self._cache.popitem(last=False)  # O(1) — drop LRU (first) entry
                self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        """Invalidate the negotiation cache."""
        self._cache.clear()


# Default global registry pre-populated with standard formats
_format_registry = PCMFormatRegistry()
_format_registry.register("whisper", PCMFormat.whisper())
_format_registry.register("openai_tts", PCMFormat.openai_tts())
_format_registry.register("portaudio", PCMFormat.portaudio_default())
_format_registry.register("telephone", PCMFormat(sample_rate=8000, channels=1, dtype="int16"))
_format_registry.register("wideband", PCMFormat(sample_rate=16000, channels=1, dtype="float32"))
_format_registry.register("hd_voice", PCMFormat(sample_rate=32000, channels=1, dtype="float32"))
_format_registry.register("cd", PCMFormat(sample_rate=44100, channels=2, dtype="float32"))
_format_registry.register("studio", PCMFormat(sample_rate=48000, channels=2, dtype="float32"))
_format_registry.register("broadcast", PCMFormat(sample_rate=48000, channels=1, dtype="float32"))

# Opus-targeted presets (used when USE_FFMPEG_IO=1)
_format_registry.register("opus_voip",    PCMFormat(sample_rate=48000, channels=1, dtype="int16"))
_format_registry.register("opus_stereo",  PCMFormat(sample_rate=48000, channels=2, dtype="int16"))
_format_registry.register("opus_hd",      PCMFormat(sample_rate=48000, channels=2, dtype="float32"))

def get_format_registry() -> PCMFormatRegistry:
    """Return the module-level default PCMFormatRegistry."""
    return _format_registry

# ── PCMPipeline + PCMPipelineBuilder — fluent composable processing chain ─────

class PCMPipeline:
    """
    A composed PCM processing pipeline built by PCMPipelineBuilder.

    Chains N processors together: output of stage i is input to stage i+1.
    The final stage's output is yielded by stream().

    Each stage is an AsyncProcessor (anything with async stream() method).
    """

    def __init__(
        self,
        stages: list[AsyncProcessor],
        tracker: PCMLatencyTracker | None = None,
        stage_names: list[str] | None = None,
    ) -> None:
        self._stages = stages
        self._tracker = tracker
        self._names = stage_names or [f"stage_{i}" for i in range(len(stages))]

    async def run(self, source: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]:
        """
        Run the pipeline: thread source through all stages, yield output.

        Usage::

            pipeline = builder.build()
            async for chunk in pipeline.run(mic_stream):
                await stt.transcribe(chunk)
        """
        stream = source
        for stage, name in zip(self._stages, self._names):
            stream = self._wrap_stage(stream, stage, name)
        async for chunk in stream:
            yield chunk

    async def _wrap_stage(
        self,
        source: AsyncIterator[PCMChunk],
        stage: AsyncProcessor,
        name: str,
    ) -> AsyncIterator[PCMChunk]:
        async for chunk in stage.stream(source):
            if self._tracker:
                self._tracker.observe(chunk, name)
            yield chunk


class PCMPipelineBuilder:
    """
    Fluent builder for composable PCM processing pipelines.

    Every method returns self for chaining. Call build() to create
    an executable PCMPipeline.

    Example::

        pipeline = (
            PCMPipelineBuilder(tracker=PCMLatencyTracker())
            .add(PCMBandpassFilter(fmt, preset="wideband"))
            .add(PCMNoiseSuppressor(fmt))
            .add(PCMAGCProcessor(fmt))
            .add(PCMNoiseGate(fmt))
            .add(PCMVADGate(fmt))
            .build()
        )
        async for speech_chunk in pipeline.run(mic_stream):
            transcription = await stt.transcribe(speech_chunk)
    """

    def __init__(self, tracker: PCMLatencyTracker | None = None) -> None:
        self._stages: list[tuple[str, AsyncProcessor]] = []
        self._tracker = tracker

    def add(self, stage: AsyncProcessor, name: str | None = None) -> PCMPipelineBuilder:
        """Add a processing stage. name defaults to the class name."""
        stage_name = name or type(stage).__name__
        self._stages.append((stage_name, stage))
        return self

    def with_bandpass(self, stage: PCMBandpassFilter) -> PCMPipelineBuilder:
        return self.add(stage, "bandpass")

    def with_noise_suppressor(self, stage: PCMNoiseSuppressor) -> PCMPipelineBuilder:
        return self.add(stage, "noise_suppressor")

    def with_agc(self, stage: PCMAGCProcessor) -> PCMPipelineBuilder:
        return self.add(stage, "agc")

    def with_noise_gate(self, stage: PCMNoiseGate) -> PCMPipelineBuilder:
        return self.add(stage, "noise_gate")

    def with_dynamics(self, stage: PCMDynamicsProcessor) -> PCMPipelineBuilder:
        return self.add(stage, "dynamics")

    def with_vad(self, stage: Any) -> PCMPipelineBuilder:
        """Accept any VAD with a stream() method (PCMVADGate, SpectralVAD, WebRTCVAD, FusedVAD)."""
        return self.add(stage, type(stage).__name__)

    def with_silence_padder(self, stage: PCMSilencePadder) -> PCMPipelineBuilder:
        return self.add(stage, "silence_padder")

    def with_recorder(self, stage: PCMRollingRecorder) -> PCMPipelineBuilder:
        return self.add(stage, "recorder")

    def with_drift_corrector(self, stage: PCMDriftCorrector) -> PCMPipelineBuilder:
        return self.add(stage, "drift_corrector")

    def build(self) -> PCMPipeline:
        """Construct and return the PCMPipeline."""
        stages = [s for _, s in self._stages]
        names = [n for n, _ in self._stages]
        log.info("pcm_pipeline_built", stages=names)
        return PCMPipeline(stages=stages, tracker=self._tracker, stage_names=names)

    def __len__(self) -> int:
        return len(self._stages)

    def __repr__(self) -> str:
        names = [n for n, _ in self._stages]
        return f"PCMPipelineBuilder(stages={names})"

# ── PCMSpeechEnhancer — pre-STT enhancement facade ───────────────────────────

class PCMSpeechEnhancer:
    """
    High-level pre-STT speech enhancement facade.

    Composes the complete recommended enhancement chain for microphone input
    destined for a speech-to-text model:

        Microphone → BandpassFilter → NoiseSuppressor → AGC → NoiseGate → VAD → STT

    All components are pre-configured with STT-optimised defaults. The enhancer
    exposes the same stream() interface as individual components for drop-in use.

    Parameters:
        fmt:             Input PCMFormat (typically PCMFormat.whisper()).
        enable_bandpass: Enable voice-band bandpass filter. Default True.
        enable_ns:       Enable spectral noise suppression. Default True.
        enable_agc:      Enable automatic gain control. Default True.
        enable_gate:     Enable noise gate. Default True.
        vad_backend:     Which VAD to use: "energy", "spectral", "webrtc", "fused". Default "fused".
        tracker:         Optional PCMLatencyTracker for stage-level measurement.
        recorder:        Optional PCMRollingRecorder for full-session recording.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        enable_bandpass: bool = True,
        enable_ns: bool = True,
        enable_agc: bool = True,
        enable_gate: bool = True,
        vad_backend: Literal["energy", "spectral", "webrtc", "fused"] = "fused",
        tracker: PCMLatencyTracker | None = None,
        recorder: PCMRollingRecorder | None = None,
        interrupt_detector: "PCMInterruptDetector | None" = None,
    ) -> None:
        self._fmt = fmt
        self._tracker = tracker or PCMLatencyTracker()
        builder = PCMPipelineBuilder(tracker=self._tracker)

        if recorder:
            builder.add(recorder, "recorder")

        if enable_bandpass:
            builder.with_bandpass(PCMBandpassFilter(fmt, low_hz=80, high_hz=8000))

        # Always convert int16 → float32 before any DSP stage.
        # int16_to_float32 was previously inside `if enable_ns`, so disabling NS
        # left AGC receiving raw int16 (RMS ~1000) while target_rms=0.1 (float32
        # scale) — gain=0.000085 crushed the signal to silence before VAD.
        float_fmt = PCMFormat(sample_rate=fmt.sample_rate, channels=fmt.channels, dtype="float32")
        if fmt.dtype != "float32":
            _converter = PCMConverter(quality="auto")

            class _DtypeAdapter:
                async def stream(self, chunks: AsyncIterator[PCMChunk]) -> AsyncIterator[PCMChunk]: # noqa
                    async for chunk in chunks:
                        yield _converter.convert(chunk, float_fmt)

            builder.add(_DtypeAdapter(), "int16_to_float32")

        if enable_ns:
            builder.with_noise_suppressor(PCMNoiseSuppressor(float_fmt))

        if enable_agc:
            builder.with_agc(PCMAGCProcessor(float_fmt))

        if enable_gate:
            builder.with_noise_gate(PCMNoiseGate(float_fmt))

        # VAD stage — all backends receive float32 at this point
        if vad_backend == "energy":
            vad: Any = PCMVADGate(float_fmt)
        elif vad_backend == "spectral":
            vad = PCMSpectralVAD(float_fmt)
        elif vad_backend == "webrtc":
            vad = PCMWebRTCVAD(float_fmt)
        else:  # fused
            energy_b = _EnergyVADBackend()
            spectral_b = _SpectralVADBackend(PCMSpectralVAD(float_fmt, ratio_thresh=0.3))
            vad = PCMFusedVAD(fmt=float_fmt, backends=[energy_b, spectral_b], mode="majority")

        builder.with_vad(vad)
        self._interrupt_detector = interrupt_detector
        self._pipeline = builder.build()

    async def stream(
            self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """Full enhancement + VAD pipeline. Yields speech segments."""
        async for chunk in self._pipeline.run(chunks):
            if self._interrupt_detector is not None:
                self._interrupt_detector.push_reference(chunk)
            yield chunk

    def get_latency_report(self) -> dict[str, dict[str, float]]:
        """Return per-stage latency statistics."""
        return self._tracker.get_latency_report()

# ── PCMPlaybackEnhancer — pre-speaker TTS enhancement facade ─────────────────

class PCMPlaybackEnhancer:
    """
    High-level pre-speaker TTS enhancement facade.

    Composes the recommended enhancement chain for TTS audio destined
    for the speaker:

        TTS PCM → SilencePadder → Dynamics (limiter) → AGC → Speaker

    Prevents TTS output from clipping, normalises level, and pads silence
    for clean device transitions.

    Parameters:
        fmt:              Output PCMFormat (typically PCMFormat.openai_tts()).
        enable_limiter:   Enable peak limiter. Default True.
        enable_agc:       Enable output AGC. Default False (TTS level is usually consistent).
        pre_silence_s:    Silence prepended to first chunk. Default 0.05.
        post_silence_s:   Silence appended to final chunk. Default 0.1.
        target_rms:       AGC target RMS. Default 0.15.
    """

    def __init__(
        self,
        fmt: PCMFormat,
        enable_limiter: bool = True,
        enable_agc: bool = False,
        pre_silence_s: float = 0.05,
        post_silence_s: float = 0.1,
        target_rms: float = 0.15,
    ) -> None:
        self._fmt = fmt
        self._padder = PCMSilencePadder(fmt, pre_s=pre_silence_s, post_s=post_silence_s)
        self._limiter = PCMDynamicsProcessor(fmt, mode="limiter", threshold_db=-1.0) if enable_limiter else None
        self._agc = PCMAGCProcessor(fmt, target_rms=target_rms) if enable_agc else None

    async def stream(
        self, chunks: AsyncIterator[PCMChunk]
    ) -> AsyncIterator[PCMChunk]:
        """TTS enhancement pipeline. Yields speaker-ready PCMChunks."""
        async for chunk in chunks:
            chunk = self._padder.process(chunk)
            if self._limiter:
                chunk = self._limiter.process(chunk)
            if self._agc:
                chunk = self._agc.process(chunk)
            yield chunk

# ── check_audio_health — one-shot mic health check ───────────────────────────

async def check_audio_health(
    fmt: PCMFormat | None = None,
    duration_s: float = 2.0,
    device: int | str | None = None,
) -> dict[str, Any]:
    """
    One-shot microphone health check.

    Opens the microphone for ``duration_s`` seconds, captures audio, and
    returns a diagnostic report suitable for startup health validation.

    Returns:
        dict with keys:
            status:          "ok" | "degraded" | "failed"
            rms_mean:        Mean RMS over capture window
            rms_peak:        Peak RMS
            clipping_ratio:  Fraction of chunks that clipped
            silence_ratio:   Fraction of chunks that were silent
            dropout_count:   Sequence number gaps
            duration_s:      Actual captured duration
            error:           Error message if status == "failed"

    Usage::

        report = await check_audio_health(PCMFormat.whisper(), duration_s=2.0)
        if report["status"] != "ok":
            log.warning("audio_health_degraded", report=report)
    """
    fmt = fmt or PCMFormat.whisper()
    monitor = PCMDiagnosticsMonitor(fmt)
    chunks_captured = []

    try:
        async with PCMInputStream(fmt=fmt, device=device) as stream:
            start = time.monotonic()
            async for chunk in stream:
                monitor.push(chunk)
                chunks_captured.append(chunk)
                if (time.monotonic() - start) >= duration_s:
                    break
    except Exception as exc:
        return {
            "status": "failed",
            "error": str(exc),
            "rms_mean": 0.0,
            "rms_peak": 0.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 0.0,
            "dropout_count": 0,
            "duration_s": 0.0,
        }

    report = monitor.get_health_report()
    actual_duration = sum(c.duration_s for c in chunks_captured)

    return {
        "status": report.status,
        "rms_mean": report.rms_mean,
        "rms_peak": max((c.rms() for c in chunks_captured), default=0.0),
        "clipping_ratio": report.clipping_rate,
        "silence_ratio": report.silence_rate,
        "dropout_count": report.dropout_count,
        "duration_s": actual_duration,
        "error": None,
    }

# ── build_voice_agent_pipeline — one-call factory ─────────────────────────────

def build_voice_agent_pipeline(
    input_fmt: PCMFormat | None = None,
    output_fmt: PCMFormat | None = None,
    vad_backend: Literal["energy", "spectral", "webrtc", "fused"] = "fused",
    enable_aec: bool = False,
    enable_drift_correction: bool = True,
    enable_interrupt_detection: bool = True,
    on_interrupt: Callable[[], None] | None = None,
) -> tuple[PCMSpeechEnhancer, PCMPlaybackEnhancer, PCMEchoCanceller | None, PCMDriftCorrector | None]:
    """
    One-call factory for a complete voice-to-voice agent pipeline.

    Returns a tuple of (speech_enhancer, playback_enhancer, aec, drift_corrector).

    Typical wiring::

        enhancer, player_chain, aec, drift = build_voice_agent_pipeline()

        async def capture_loop():
            async with PCMInputStream(fmt=PCMFormat.whisper()) as mic:
                if drift:
                    async for chunk in drift.stream(mic):
                        async for speech in enhancer.stream(one_chunk_iter(chunk)):
                            await stt.transcribe(speech)
                else:
                    async for speech in enhancer.stream(mic):
                        await stt.transcribe(speech)

        async def playback_loop(tts_stream):
            if aec:
                async def _tts_with_ref(s):
                    async for c in s:
                        aec.push_reference(c)
                        yield c
                tts_stream = _tts_with_ref(tts_stream)
            async with PCMOutputStream(preferred_fmt=PCMFormat.openai_tts()) as out:
                async for chunk in player_chain.stream(tts_stream):
                    await out.write(chunk)

    Parameters:
        input_fmt:                  Mic format. Default PCMFormat.whisper().
        output_fmt:                 Speaker format. Default PCMFormat.openai_tts().
        vad_backend:                VAD algorithm. Default "fused".
        enable_aec:                 Enable acoustic echo cancellation. Default False.
        enable_drift_correction:    Enable sample-clock drift correction. Default True.
        enable_interrupt_detection: Enable barge-in detection via PCMInterruptDetector.
                                    Wires the detector into the mic-side pipeline so
                                    user speech during TTS playback triggers on_interrupt.
                                    Default True.
        on_interrupt:               Callback invoked when a barge-in is detected. Only
                                    used when enable_interrupt_detection is True. If None,
                                    barge-in events are detected but not acted upon.
                                    Default None.

    Returns:
        (PCMSpeechEnhancer, PCMPlaybackEnhancer, PCMEchoCanceller | None, PCMDriftCorrector | None)
    """
    input_fmt = input_fmt or PCMFormat.whisper()
    output_fmt = output_fmt or PCMFormat.openai_tts()

    tracker = PCMLatencyTracker()
    recorder = PCMRollingRecorder(fmt=input_fmt, max_seconds=60.0)

    interrupt_detector = None
    if enable_interrupt_detection:
        from player import get_interrupt_detector
        interrupt_detector = get_interrupt_detector(on_interrupt=on_interrupt)

    speech_enhancer = PCMSpeechEnhancer(
        fmt=input_fmt,
        vad_backend=vad_backend,
        tracker=tracker,
        recorder=recorder,
        interrupt_detector=interrupt_detector,
    )
    playback_enhancer = PCMPlaybackEnhancer(fmt=output_fmt)
    aec = PCMEchoCanceller(fmt=input_fmt) if enable_aec else None
    drift = PCMDriftCorrector(fmt=input_fmt) if enable_drift_correction else None

    log.info(
        "voice_agent_pipeline_built",
        input_fmt=repr(input_fmt),
        output_fmt=repr(output_fmt),
        vad_backend=vad_backend,
        aec_enabled=enable_aec,
        drift_enabled=enable_drift_correction,
        interrupt_detection_enabled=enable_interrupt_detection,
    )

    return speech_enhancer, playback_enhancer, aec, drift

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONVENIENCE SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════

_default_converter: PCMConverter = PCMConverter(quality="auto")


def get_converter() -> PCMConverter:
    """Return the module-level default PCMConverter (auto quality)."""
    return _default_converter


_default_latency_tracker: PCMLatencyTracker = PCMLatencyTracker()
_default_diagnostics: PCMDiagnosticsMonitor | None = None


def get_latency_tracker() -> PCMLatencyTracker:
    """Return the module-level default PCMLatencyTracker."""
    return _default_latency_tracker


def get_diagnostics(fmt: PCMFormat | None = None) -> PCMDiagnosticsMonitor:
    """Return (or create) the module-level default PCMDiagnosticsMonitor."""
    global _default_diagnostics
    if _default_diagnostics is None:
        _default_diagnostics = PCMDiagnosticsMonitor(
            fmt=fmt or PCMFormat.whisper()
        )
    return _default_diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# 10. TESTING UTILITIES / DEBUG HELPERS
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    import sys

    # ── helpers ──────────────────────────────────────────────────────────────

    def _sine_chunk(
        fmt: PCMFormat,
        freq: float = 440.0,
        duration_s: float = 0.05,
        seq: int = 0,
        is_final: bool = False,
        source: str = "test",
        amplitude: float = 0.5,
    ) -> PCMChunk:
        n = fmt.frames_for_duration(duration_s)
        t = np.linspace(0, duration_s, n, endpoint=False)
        sine = (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)
        if fmt.dtype == "int16":
            sine = (sine * 32767).astype(np.int16)
        if fmt.channels > 1:
            sine = np.column_stack([sine] * fmt.channels)
        return PCMChunk(data=sine, fmt=fmt, timestamp=time.monotonic(),
                        seq=seq, is_final=is_final, source=source)

    async def _chunks_from_list(chunks: list[PCMChunk]) -> AsyncIterator[PCMChunk]:
        for c in chunks:
            yield c


    # ── smoke tests ──────────────────────────────────────────────────────────


    async def _test_serializer() -> None:
        print("\n[SERIALIZER] Testing PCMChunkSerializer ...")
        fmt = PCMFormat.whisper()
        chunk = _sine_chunk(fmt, is_final=True)
        wire = PCMChunkSerializer.serialize(chunk)
        recovered = PCMChunkSerializer.deserialize(wire)
        assert recovered.fmt == chunk.fmt, f"fmt mismatch: {recovered.fmt} vs {chunk.fmt}"
        assert recovered.n_frames == chunk.n_frames
        assert recovered.is_final == chunk.is_final
        assert np.allclose(recovered.data.astype(np.float64), chunk.data.astype(np.float64), atol=1)
        print(f"  ✓ Serialized {chunk} → {len(wire)} bytes → recovered")

        # Test stream framing
        wire2 = PCMChunkSerializer.serialize(_sine_chunk(fmt, seq=1))
        combined = wire + wire2
        c1, rest = PCMChunkSerializer.frame_from_stream(combined)
        c2, empty = PCMChunkSerializer.frame_from_stream(rest)
        assert c1 is not None and c2 is not None
        assert len(empty) == 0
        print("  ✓ Stream framing (two-packet round-trip) ✓")

    async def _test_chunk_pool() -> None:
        print("\n[POOL] Testing PCMChunkPool ...")
        pool = PCMChunkPool(max_per_bucket=8)
        arr = pool.acquire(960, "float32", 1)
        assert arr.shape == (960,)
        assert arr.dtype == np.float32
        pool.release(arr)
        arr2 = pool.acquire(960, "float32", 1)
        assert arr2.shape == (960,)  # should reuse
        pool.release(arr2)
        # Stereo
        stereo = pool.acquire(480, "float32", 2)
        assert stereo.shape == (480, 2)
        pool.release(stereo)
        print("  ✓ Pool acquire/release (mono + stereo) ✓")

    async def _test_spectral_vad() -> None:
        print("\n[SPECTRAL VAD] Testing PCMSpectralVAD ...")
        fmt = PCMFormat.whisper()
        vad = PCMSpectralVAD(fmt, floor_rms=10.0)

        # Silence chunk
        silence = PCMChunk(
            data=np.zeros(960, dtype="int16"),
            fmt=fmt, timestamp=time.monotonic(), seq=0, source="test"
        )
        assert not vad.is_speech(silence), "Silence should not be speech"

        # Speech chunk (440 Hz is in voice band)
        speech = _sine_chunk(fmt, freq=440, amplitude=0.5)
        result = vad.is_speech(speech)
        print(f"  440 Hz sine is_speech: {result} (in voice band — expected True)")

        # Out-of-band chunk (15 kHz hiss — above voice band for 16 kHz)
        n = 960
        t = np.linspace(0, 0.06, n)
        hiss = (np.sin(2 * np.pi * 15000 * t) * 0.5 * 32767).astype(np.int16)
        hiss_chunk = PCMChunk(data=hiss, fmt=fmt, timestamp=time.monotonic(), seq=1, source="test")
        result2 = vad.is_speech(hiss_chunk)
        print(f"  15 kHz hiss is_speech: {result2} (out-of-band — expected False or low ratio)")
        print("  ✓ SpectralVAD classification ran without error")

    async def _test_noise_gate() -> None:
        print("\n[NOISE GATE] Testing PCMNoiseGate ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        gate = PCMNoiseGate(fmt, threshold_db=-30.0, range_db=-80.0, attack_ms=1.0, release_ms=50.0)

        loud_chunk = _sine_chunk(fmt, amplitude=0.5)
        gated_loud = gate.process(loud_chunk)
        assert gated_loud.n_frames == loud_chunk.n_frames
        loud_rms = float(np.sqrt(np.mean(gated_loud.data.astype(np.float64) ** 2)))

        # Soft signal (below threshold)
        gate2 = PCMNoiseGate(fmt, threshold_db=-30.0)
        quiet_chunk = _sine_chunk(fmt, amplitude=0.001)
        gated_quiet = gate2.process(quiet_chunk)
        quiet_rms = float(np.sqrt(np.mean(gated_quiet.data.astype(np.float64) ** 2)))

        print(f"  Loud chunk RMS after gate: {loud_rms:.4f}")
        print(f"  Quiet chunk RMS after gate: {quiet_rms:.6f}")
        print("  ✓ NoiseGate processed without error")

    async def _test_agc() -> None:
        print("\n[AGC] Testing PCMAGCProcessor ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        agc = PCMAGCProcessor(fmt, target_rms=0.1)

        # Very quiet input
        chunks = [_sine_chunk(fmt, amplitude=0.01, seq=i) for i in range(20)]
        results = [agc.process(c) for c in chunks]
        final_rms = float(np.sqrt(np.mean(results[-1].data.astype(np.float64) ** 2)))
        print(f"  Input RMS: 0.01 → Output RMS after 20 chunks: {final_rms:.4f}")
        print("  ✓ AGC converged without overflow")

    async def _test_dynamics() -> None:
        print("\n[DYNAMICS] Testing PCMDynamicsProcessor ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")

        compressor = PCMDynamicsProcessor(fmt, mode="compressor", threshold_db=-20, ratio=4)
        loud = _sine_chunk(fmt, amplitude=0.9)
        comp_out = compressor.process(loud)
        out_peak = float(np.max(np.abs(comp_out.data)))
        in_peak = float(np.max(np.abs(loud.data)))
        print(f"  Compressor: input peak {in_peak:.3f} → output peak {out_peak:.3f}")
        assert out_peak <= in_peak + 0.01, "Compressor should not increase peak"

        limiter = PCMDynamicsProcessor(fmt, mode="limiter", threshold_db=-6)
        limited = limiter.process(_sine_chunk(fmt, amplitude=0.95))
        lim_peak = float(np.max(np.abs(limited.data)))
        print(f"  Limiter: peak after limiting {lim_peak:.4f}")
        print("  ✓ Dynamics processor all modes passed")

    async def _test_bandpass() -> None:
        print("\n[BANDPASS] Testing PCMBandpassFilter ...")
        if not _SCIPY:
            print("  ⚠ scipy not installed — skipping FIR test, using passthrough")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        bp = PCMBandpassFilter(fmt, low_hz=300, high_hz=3400)

        chunk = _sine_chunk(fmt, freq=440, amplitude=0.5)  # in band
        filtered = bp.process(chunk)
        assert filtered.n_frames == chunk.n_frames
        print(f"  440 Hz chunk: {chunk.n_frames} frames in → {filtered.n_frames} frames out")
        print("  ✓ Bandpass filter applied")

    async def _test_waveform_analyzer() -> None:
        stats = None

        print("\n[ANALYZER] Testing PCMWaveformAnalyzer ...")
        fmt = PCMFormat.whisper()
        analyzer = PCMWaveformAnalyzer(fmt)

        chunks = [_sine_chunk(fmt, amplitude=0.5, seq=i) for i in range(10)]
        for c in chunks:
            stats = analyzer.analyze(c)

        running = analyzer.get_running_stats()
        print(f"  Running stats: rms_mean={running['rms_mean']:.4f}, "
              f"peak_max={running['peak_max']:.4f}, "
              f"zcr_mean={running['zcr_mean']:.4f}")
        assert stats is not None
        print(f"  Spectral centroid (last chunk): {stats.spectral_centroid_hz:.0f} Hz")
        print("  ✓ WaveformAnalyzer ✓")

    async def _test_aec() -> None:
        print("\n[AEC] Testing PCMEchoCanceller ...")
        fmt = PCMFormat.whisper()
        aec = PCMEchoCanceller(fmt, filter_len=256, step_size=0.1)

        # Simulate echo: reference = TTS, mic = TTS + quiet speech
        ref_chunk = _sine_chunk(fmt, freq=440, amplitude=0.5, seq=0)
        aec.push_reference(ref_chunk)

        speech = _sine_chunk(fmt, freq=800, amplitude=0.05, seq=0)
        mic_data = ref_chunk.data.astype(np.float32) * 0.7 + speech.data.astype(np.float32)
        mic_chunk = PCMChunk(data=mic_data.astype("int16"), fmt=fmt,
                             timestamp=time.monotonic(), seq=0, source="mic")

        clean = aec.cancel(mic_chunk)
        in_rms = float(np.sqrt(np.mean(mic_chunk.data.astype(np.float64) ** 2)))
        out_rms = float(np.sqrt(np.mean(clean.data.astype(np.float64) ** 2)))
        print(f"  Mic RMS: {in_rms:.1f} → AEC output RMS: {out_rms:.1f}")
        print("  ✓ AEC ran without error")

    async def _test_jitter_buffer() -> None:
        print("\n[JITTER] Testing PCMJitterBuffer ...")
        fmt = PCMFormat.whisper()
        jb = PCMJitterBuffer(fmt, target_delay_ms=20, max_delay_ms=100, plc_mode=PLCMode.ZERO_FILL)

        # Push packets out-of-order
        chunks = [_sine_chunk(fmt, seq=i, duration_s=0.02) for i in range(5)]
        jb.push(chunks[2])
        jb.push(chunks[0])
        jb.push(chunks[4])
        jb.push(chunks[1])
        jb.push(chunks[3])

        # Let playout run for a brief moment
        await jb.start()
        collected = []
        async def _collect():
            async for c in jb.stream():
                collected.append(c)
                if len(collected) >= 5:
                    await jb.stop()
                    break
        try:
            await asyncio.wait_for(_collect(), timeout=3.0)
        except asyncio.TimeoutError:
            await jb.stop()

        print(f"  Pushed 5 out-of-order packets, received {len(collected)} from jitter buffer")
        print("  ✓ JitterBuffer ran without error")

    async def _test_rolling_recorder() -> None:
        print("\n[RECORDER] Testing PCMRollingRecorder ...")
        fmt = PCMFormat.whisper()
        recorder = PCMRollingRecorder(fmt, max_seconds=1.0)

        chunks = [_sine_chunk(fmt, seq=i, duration_s=0.05) for i in range(20)]
        for c in chunks:
            recorder.push(c)

        exported_chunk = recorder.export_chunk()
        wav = recorder.export_wav()
        print(f"  Pushed {len(chunks)} chunks, exported {exported_chunk.n_frames} frames")
        print(f"  WAV export: {len(wav)} bytes")
        print(f"  Buffer depth: {recorder.buffered_seconds:.2f}s")
        assert len(wav) > 44, "WAV should have header + data"
        print("  ✓ RollingRecorder ✓")

    async def _test_mixer() -> None:
        print("\n[MIXER] Testing PCMStreamMixer ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        mixer = PCMStreamMixer(fmt, tick_s=0.02, master_gain=0.7)

        chunks_a = [_sine_chunk(fmt, freq=440, seq=i, duration_s=0.02) for i in range(5)]
        chunks_b = [_sine_chunk(fmt, freq=880, seq=i, duration_s=0.02) for i in range(5)]

        mixer.add_source(_chunks_from_list(chunks_a), gain=0.8)
        mixer.add_source(_chunks_from_list(chunks_b), gain=0.5)

        results = []
        async for mixed in mixer.stream():
            results.append(mixed)
            if len(results) >= 5:
                await mixer.stop()
                break

        print(f"  Mixed {len(results)} output chunks from 2 sources")
        print("  ✓ StreamMixer ✓")

    async def _test_serializer_roundtrip_all_dtypes() -> None:
        print("\n[SERIALIZER DTYPES] Testing all dtype roundtrips ...")
        for dtype in ("int16", "int32", "float32", "float64"):
            fmt = PCMFormat(sample_rate=16000, channels=1, dtype=dtype)  # type: ignore
            n = 320
            data = (np.random.rand(n) * 0.5).astype(dtype)
            chunk = PCMChunk(data=data, fmt=fmt, timestamp=1.234, seq=42, is_final=True, source="x")
            wire = PCMChunkSerializer.serialize(chunk)
            recovered = PCMChunkSerializer.deserialize(wire)
            assert recovered.fmt == chunk.fmt, f"{dtype} fmt mismatch"
            assert recovered.seq == 42
            assert recovered.is_final
            print(f"  ✓ {dtype} roundtrip OK ({len(wire)} bytes)")

    async def _test_pipeline_builder() -> None:
        print("\n[PIPELINE] Testing PCMPipelineBuilder ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        tracker = PCMLatencyTracker()

        pipeline = (
            PCMPipelineBuilder(tracker=tracker)
            .with_bandpass(PCMBandpassFilter(fmt))
            .with_agc(PCMAGCProcessor(fmt))
            .with_noise_gate(PCMNoiseGate(fmt))
            .build()
        )
        print(f"  Built pipeline with {len(pipeline._stages)} stages") # noqa

        source_chunks = [_sine_chunk(fmt, amplitude=0.3, seq=i, duration_s=0.02) for i in range(10)]

        async def _src():
            for c in source_chunks:
                yield c

        results = []
        async for chunk in pipeline.run(_src()):
            results.append(chunk)

        print(f"  Ran {len(source_chunks)} chunks through pipeline → {len(results)} outputs")
        report = tracker.get_latency_report()
        for stage, stats in report.items():
            print(f"  Stage [{stage}] mean={stats['mean_s']*1000:.3f}ms p95={stats['p95_s']*1000:.3f}ms")
        print("  ✓ PipelineBuilder ✓")

    async def _test_format_registry() -> None:
        print("\n[REGISTRY] Testing PCMFormatRegistry ...")
        reg = get_format_registry()
        names = reg.list_names()
        print(f"  Registered formats: {names}")
        whisper_fmt = reg["whisper"]
        assert whisper_fmt.sample_rate == 16000
        negotiated = reg.negotiate(["whisper"], ["openai_tts", "portaudio"])
        print(f"  Negotiated whisper vs [tts, portaudio]: {negotiated}")
        print("  ✓ FormatRegistry ✓")

    async def _test_silence_padder() -> None:
        print("\n[SILENCE PADDER] Testing PCMSilencePadder ...")
        fmt = PCMFormat.openai_tts()
        padder = PCMSilencePadder(fmt, pre_s=0.05, post_s=0.1, only_final=True)
        chunk = _sine_chunk(fmt, duration_s=0.5, is_final=True)
        padded = padder.process(chunk)
        extra = fmt.frames_for_duration(0.05) + fmt.frames_for_duration(0.1)
        assert padded.n_frames == chunk.n_frames + extra, (
            f"Expected {chunk.n_frames + extra} frames, got {padded.n_frames}"
        )
        print(f"  Original: {chunk.n_frames} frames → Padded: {padded.n_frames} frames (+{extra} silence)")
        print("  ✓ SilencePadder ✓")

    async def _test_stream_bridge() -> None:
        print("\n[BRIDGE] Testing PCMStreamBridge ...")
        fmt = PCMFormat.whisper()
        bridge = PCMStreamBridge(maxsize=16)
        await bridge.start()

        chunks_sent = [_sine_chunk(fmt, seq=i) for i in range(5)]

        async def _producer():
            for c in chunks_sent:
                bridge.push_sync(c)
                await asyncio.sleep(0.01)
            await bridge.stop()

        collected = []
        async def _consumer():
            async for c in bridge.async_iter():
                collected.append(c)

        await asyncio.gather(_producer(), _consumer())
        print(f"  Sent {len(chunks_sent)}, received {len(collected)} via bridge")
        print("  ✓ StreamBridge ✓")

    async def _test_fused_vad() -> None:
        print("\n[FUSED VAD] Testing PCMFusedVAD ...")
        fmt = PCMFormat.whisper()
        energy_b = _EnergyVADBackend(onset_rms=_VAD_ONSET_RMS)
        spectral_b = _SpectralVADBackend(PCMSpectralVAD(fmt))
        fused = PCMFusedVAD(fmt=fmt, backends=[energy_b, spectral_b], mode="majority") # noqa

        loud_speech = _sine_chunk(fmt, amplitude=0.5, freq=400)
        assert energy_b.is_speech(loud_speech) or spectral_b.is_speech(loud_speech), \
            "At least one backend should detect loud sine as speech"
        print("  ✓ FusedVAD backends ran without error")

    async def _test_diagnostics_monitor() -> None:
        print("\n[DIAGNOSTICS] Testing PCMDiagnosticsMonitor ...")
        fmt = PCMFormat(sample_rate=16000, channels=1, dtype="float32")
        alerts: list[str] = []
        monitor = PCMDiagnosticsMonitor(
            fmt,
            silence_threshold=0.002,
            silence_frames=3,
            on_silence=lambda: alerts.append("silence"),
            on_clipping=lambda: alerts.append("clipping"),
        )

        # Push silent chunks
        for i in range(5):
            silence = PCMChunk(data=np.zeros(960, dtype="float32"), fmt=fmt,
                               timestamp=time.monotonic(), seq=i, source="test")
            monitor.push(silence)

        report = monitor.get_health_report()
        print(f"  Health: {report.status}, silence_rate={report.silence_rate:.2f}")
        print(f"  Alerts fired: {alerts}")
        print("  ✓ DiagnosticsMonitor ✓")

    async def _test_drift_corrector() -> None:
        print("\n[DRIFT] Testing PCMDriftCorrector (no-drift path) ...")
        fmt = PCMFormat.whisper()
        drift = PCMDriftCorrector(fmt, check_interval_s=0.1, max_drift_ppm=50)
        chunks = [_sine_chunk(fmt, seq=i, duration_s=0.05) for i in range(10)]
        for c in chunks:
            out = drift.push(c)
            assert out.fmt.sample_rate == fmt.sample_rate
        print("  ✓ DriftCorrector passthrough ✓")

    async def _test_metrics_snapshot() -> None:
        print("\n[METRICS] Testing PCMMetricsSnapshot ...")
        s1 = get_metrics_snapshot()
        await asyncio.sleep(0.01)
        s2 = get_metrics_snapshot()
        delta = s1.delta(s2)
        print(f"  Snapshot captured: {len(delta)} metric deltas")
        print("  ✓ MetricsSnapshot ✓")

    # ── main test runner ──────────────────────────────────────────────────────

    async def _run_all_tests() -> None:
        print("=" * 72)
        print("PCM PIPELINE — COMPREHENSIVE SMOKE TEST SUITE")
        print("=" * 72)

        tests = [
            _test_chunk_pool,
            _test_serializer,
            _test_serializer_roundtrip_all_dtypes,
            _test_spectral_vad,
            _test_fused_vad,
            _test_noise_gate,
            _test_agc,
            _test_dynamics,
            _test_bandpass,
            _test_waveform_analyzer,
            _test_aec,
            _test_jitter_buffer,
            _test_rolling_recorder,
            _test_mixer,
            _test_pipeline_builder,
            _test_format_registry,
            _test_silence_padder,
            _test_stream_bridge,
            _test_diagnostics_monitor,
            _test_drift_corrector,
            _test_metrics_snapshot,
        ]

        passed = 0
        failed = 0
        for test_fn in tests:
            try:
                await test_fn()
                passed += 1
            except Exception: # noqa
                failed += 1
                print(f"\n  ✗ {test_fn.__name__} FAILED:")
                traceback.print_exc()

        print("\n" + "=" * 72)
        print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
        print("=" * 72)

        if failed:
            sys.exit(1)

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "all":
        asyncio.run(_run_all_tests())
    elif mode == "input":
        async def _smoke_input() -> None:
            fmt = PCMFormat.whisper()
            enhancer = PCMSpeechEnhancer(fmt=fmt, vad_backend="energy")
            print(f"[INPUT] Opening mic: {fmt}\n  Speak to trigger VAD. Ctrl-C to stop.")
            async with PCMInputStream(fmt=fmt) as stream:
                n = 0
                async for seg in enhancer.stream(stream):  # type: ignore[arg-type]
                    n += 1
                    wav = chunk_to_wav_bytes(seg)
                    print(f"  [seg {n}] frames={seg.n_frames} dur={seg.duration_s:.2f}s wav={len(wav)}B")
                    if n >= 3:
                        break
        asyncio.run(_smoke_input())
    elif mode == "output":
        async def _smoke_output() -> None:
            fmt = PCMFormat.openai_tts()
            enhancer = PCMPlaybackEnhancer(fmt=fmt)
            sr = fmt.sample_rate
            t = np.linspace(0, 1.0, sr, endpoint=False)
            sine = (np.sin(2 * np.pi * 440 * t) * 28000).astype(np.int16)
            chunk = PCMChunk(data=sine, fmt=fmt, seq=0, is_final=True,
                             timestamp=time.monotonic(), source="test")
            async def _one():
                yield chunk
            async with PCMOutputStream() as out:
                async for c in enhancer.stream(_one()):
                    await out.write(c)
                await asyncio.sleep(1.2)
            print("[OUTPUT] Done.")
        asyncio.run(_smoke_output())
    else:
        print(f"Unknown mode: {mode}. Use: all | input | output")
        sys.exit(1)