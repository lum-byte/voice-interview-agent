"""
opus_ffmpeg_io.py — Elite Opus/FFmpeg Audio Pipeline

A ground-up rewrite of the codec layer 
for voice quality, latency, and operational resilience.

Design goals:
  1. Asyncio-native — zero blocking calls on the event loop
  2. Horizontally scalable — stateless stream objects, pooled workers
  3. Vertically scalable — per-core encoder/decoder process pools
  4. Fault-tolerant — circuit breaker, auto-restart, graceful degradation
  5. Adaptive — bitrate, jitter buffer, and FEC all adjust to network conditions
  6. Observable — structured metrics, zero external dependencies required

Ingress pipeline (Opus → PCM):
    OpusPacket ──► JitterBuffer ──► OggPageAssembler ──► DecoderProcess
                     (RFC 3533)         (adaptive)          (libopus)
                         │                                      │
                     PLC layer ◄───── loss detection ◄──────────┘
                         │
                    PCMChunk ──► audio_engine ──► STT

Egress pipeline (PCM → Opus):
    PCMChunk ──► PreEmphFilter ──► EncoderProcess ──► OggPageParser
                                       (libopus)       (RFC 3533)
                        │                                    │
                  ABRController ◄──── NetworkStats ◄─────────┘
                        │
                  OpusPacket + FEC ──► network

Scaling:
    Vertical  : EncoderPool / DecoderPool (one process per logical core)
    Horizontal: Stateless — any node accepts any stream via StreamRouter
"""

from __future__ import annotations

import asyncio
import collections
import hashlib # noqa
import logging
import math
import os
import struct
import subprocess
import time
import threading # noqa
import weakref # noqa
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import (
    Any,
    AsyncIterator,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Generator, # noqa
    Iterator,
    List,
    NamedTuple, # noqa
    Optional,
    Protocol, # noqa
    Sequence,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable, # noqa
)

import numpy as np

log = logging.getLogger(__name__)

# ── Version ───────────────────────────────────────────────────────────────────
__version__ = "2.0.0"

# ── Typing helpers ────────────────────────────────────────────────────────────
T = TypeVar("T")
Seconds = float
Milliseconds = float
SampleRate = int
SequenceNumber = int


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class _Env:
    """Read-once environment configuration with validation and sensible defaults."""

    # ── Encoding ──────────────────────────────────────────────────────────────
    BITRATE_KBPS:   int  = int(os.getenv("OPUS_BITRATE_KBPS", "96"))
    COMPLEXITY:     int  = int(os.getenv("OPUS_COMPLEXITY", "9"))
    FRAME_DURATION: int  = int(os.getenv("OPUS_FRAME_MS", "20"))      # ms: 2.5,5,10,20,40,60
    APPLICATION:    str  = os.getenv("OPUS_APPLICATION", "voip")      # voip | audio | lowdelay
    DTX:            bool = os.getenv("OPUS_DTX", "1") == "1"
    FEC:            bool = os.getenv("OPUS_FEC", "1") == "1"
    VBR:            bool = os.getenv("OPUS_VBR", "1") == "1"

    # ── ABR limits ────────────────────────────────────────────────────────────
    ABR_MIN_KBPS:   int  = int(os.getenv("OPUS_ABR_MIN_KBPS", "8"))
    ABR_MAX_KBPS:   int  = int(os.getenv("OPUS_ABR_MAX_KBPS", "320"))

    # ── Jitter buffer ─────────────────────────────────────────────────────────
    JB_MIN_MS:      int  = int(os.getenv("OPUS_JB_MIN_MS", "20"))
    JB_TARGET_MS:   int  = int(os.getenv("OPUS_JB_TARGET_MS", "80"))
    JB_MAX_MS:      int  = int(os.getenv("OPUS_JB_MAX_MS", "500"))

    # ── Process pool ──────────────────────────────────────────────────────────
    ENCODER_WORKERS: int = int(os.getenv("OPUS_ENCODER_WORKERS", str(max(2, os.cpu_count() or 2))))
    DECODER_WORKERS: int = int(os.getenv("OPUS_DECODER_WORKERS", str(max(2, os.cpu_count() or 2))))

    # ── Circuit breaker ───────────────────────────────────────────────────────
    CB_FAILURE_THRESHOLD: int   = int(os.getenv("OPUS_CB_FAILURES", "5"))
    CB_RECOVERY_TIMEOUT:  float = float(os.getenv("OPUS_CB_RECOVERY_S", "10.0"))

    # ── I/O tuning ────────────────────────────────────────────────────────────
    PIPE_BUFFER_BYTES: int = int(os.getenv("OPUS_PIPE_BUFFER", str(256 * 1024)))

    @classmethod
    def validate(cls) -> None:
        valid_durations = {2, 5, 10, 20, 40, 60}
        if cls.FRAME_DURATION not in valid_durations:
            raise ValueError(
                f"OPUS_FRAME_MS={cls.FRAME_DURATION} invalid; "
                f"choose from {sorted(valid_durations)}"
            )
        if cls.BITRATE_KBPS < cls.ABR_MIN_KBPS or cls.BITRATE_KBPS > cls.ABR_MAX_KBPS:
            raise ValueError(
                f"OPUS_BITRATE_KBPS={cls.BITRATE_KBPS} out of "
                f"[{cls.ABR_MIN_KBPS}, {cls.ABR_MAX_KBPS}]"
            )
        if cls.APPLICATION not in {"voip", "audio", "lowdelay"}:
            raise ValueError(f"OPUS_APPLICATION={cls.APPLICATION!r} invalid")


_Env.validate()


# Opus legal frame sizes in samples at 48 kHz (canonical internal rate)
_OPUS_FRAME_SIZES_48K: Dict[int, int] = {
    2:  120,
    5:  240,
    10: 480,
    20: 960,
    40: 1920,
    60: 2880,
}

# PCM format → (ffmpeg_format_flag, numpy_dtype, bytes_per_sample)
_PCM_FORMAT_MAP: Dict[str, Tuple[str, np.dtype, int]] = {
    "int16":   ("s16le", np.dtype("<i2"), 2),
    "int32":   ("s32le", np.dtype("<i4"), 4),
    "float32": ("f32le", np.dtype("<f4"), 4),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class OpusError(RuntimeError):
    """Base for all codec pipeline errors."""

class FFmpegNotFoundError(OpusError):
    """FFmpeg binary not on PATH."""
    INSTALL_HINT: ClassVar[str] = (
        "Install FFmpeg with Opus support:\n"
        "  Ubuntu/Debian : sudo apt-get install ffmpeg libopus-dev\n"
        "  macOS         : brew install ffmpeg\n"
        "  Windows       : choco install ffmpeg  (or download from ffmpeg.org)\n"
        "  Docker        : RUN apt-get install -y ffmpeg libopus-dev"
    )

class ProcessDiedError(OpusError):
    """FFmpeg subprocess terminated unexpectedly."""

class CircuitOpenError(OpusError):
    """Request rejected because the circuit breaker is open."""

class BufferOverflowError(OpusError):
    """Queue or ring-buffer capacity exceeded."""

class FrameSizeError(OpusError):
    """PCM data length is not a valid Opus frame boundary."""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CodecConfig:
    """
    Immutable configuration snapshot for one codec pipeline instance.
    Passed to all sub-components; changes require a new instance.
    """
    sample_rate:    int   = 48_000        # Hz — prefer 48k (Opus native)
    channels:       int   = 1             # 1=mono, 2=stereo
    dtype:          str   = "int16"       # PCM dtype
    bitrate_kbps:   int   = field(default_factory=lambda: _Env.BITRATE_KBPS)
    complexity:     int   = field(default_factory=lambda: _Env.COMPLEXITY)
    frame_ms:       int   = field(default_factory=lambda: _Env.FRAME_DURATION)
    application:    str   = field(default_factory=lambda: _Env.APPLICATION)
    dtx:            bool  = field(default_factory=lambda: _Env.DTX)
    fec:            bool  = field(default_factory=lambda: _Env.FEC)
    vbr:            bool  = field(default_factory=lambda: _Env.VBR)

    @property
    def frame_samples(self) -> int:
        """Number of PCM samples per Opus frame at this sample_rate."""
        return int(self.sample_rate * self.frame_ms / 1000)

    @property
    def bytes_per_sample(self) -> int:
        return _PCM_FORMAT_MAP[self.dtype][2]

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.channels * self.bytes_per_sample

    @property
    def ffmpeg_format(self) -> str:
        return _PCM_FORMAT_MAP[self.dtype][0]

    @property
    def numpy_dtype(self) -> np.dtype:
        return _PCM_FORMAT_MAP[self.dtype][1]

    @classmethod
    def for_whisper(cls) -> "CodecConfig":
        """16 kHz mono int16 — Whisper STT input."""
        return cls(sample_rate=16_000, channels=1, dtype="int16")

    @classmethod
    def for_openai_tts(cls) -> "CodecConfig":
        """24 kHz mono int16 — OpenAI TTS output."""
        return cls(sample_rate=24_000, channels=1, dtype="int16")

    @classmethod
    def for_discord(cls) -> "CodecConfig":
        """48 kHz stereo int16 — Discord / WebRTC standard."""
        return cls(
            sample_rate=48_000, channels=2, dtype="int16",
            bitrate_kbps=64, application="voip",
            dtx=True, fec=True, vbr=True,
        )

    @classmethod
    def for_hd_audio(cls) -> "CodecConfig":
        """48 kHz stereo float32 — studio / music-grade."""
        return cls(
            sample_rate=48_000, channels=2, dtype="float32",
            bitrate_kbps=320, complexity=10, application="audio",
            dtx=False, fec=False, vbr=True,
        )


@dataclass(slots=True)
class OpusPacket:
    """
    A single Opus-encoded audio frame with routing and quality metadata.
    Analogous to an RTP payload without the transport layer.
    """
    seq:            int           # Monotonically increasing sequence number
    timestamp_us:   int           # Capture timestamp (microseconds, wall clock)
    payload:        bytes         # Raw Opus bytes (ready for RTP/WebRTC payload)
    duration_ms:    int           # Frame duration in ms (2.5 – 60)
    config:         CodecConfig
    is_silence:     bool = False  # True when DTX flagged this as silence
    has_fec:        bool = False  # True when in-band FEC data is present
    loss_prob:      float = 0.0   # Encoder-estimated packet loss probability

    @property
    def age_ms(self) -> float:
        return (time.monotonic_ns() // 1_000 - self.timestamp_us) / 1_000.0


@dataclass(slots=True)
class PCMFrame:
    """
    A chunk of raw PCM samples wrapping a numpy array.
    Compatible with audio_engine's PCMChunk protocol by attribute names.
    """
    data:   np.ndarray   # shape: (samples,) or (samples, channels)
    seq:    int
    ts_us:  int          # Capture timestamp in microseconds
    config: CodecConfig

    @property
    def duration_ms(self) -> float:
        samples = self.data.shape[0]
        return samples / self.config.sample_rate * 1000.0

    @property
    def is_silence(self) -> bool:
        """True when RMS < −72 dBFS (effectively inaudible)."""
        rms = float(np.sqrt(np.mean(self.data.astype(np.float64) ** 2)))
        if rms == 0.0:
            return True
        dtype_max = np.iinfo(self.config.numpy_dtype).max if np.issubdtype(
            self.config.numpy_dtype, np.integer
        ) else 1.0
        return 20 * math.log10(rms / dtype_max) < -72.0


@dataclass
class NetworkStats:
    """
    Rolling network quality estimates updated by the codec pipeline.
    Used by ABRController and JitterBuffer for adaptation.
    """
    _WINDOW: ClassVar[int] = 64   # packets for moving stats

    _rtts_ms:    Deque[float] = field(default_factory=lambda: collections.deque(maxlen=64))
    _arrivals_us: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=64))
    _losses:     Deque[bool]  = field(default_factory=lambda: collections.deque(maxlen=64))
    _last_recv:  float = 0.0

    def record_rtt(self, rtt_ms: float) -> None:
        self._rtts_ms.append(rtt_ms)

    def record_arrival(self, lost: bool = False) -> None:
        now_us = time.monotonic_ns() / 1_000.0
        if self._last_recv:
            self._arrivals_us.append(now_us - self._last_recv)
        self._last_recv = now_us
        self._losses.append(lost)

    @property
    def rtt_ms(self) -> float:
        return float(np.mean(self._rtts_ms)) if self._rtts_ms else 0.0

    @property
    def jitter_ms(self) -> float:
        if len(self._arrivals_us) < 4:
            return 0.0
        arr = np.array(self._arrivals_us) / 1_000.0  # → ms
        return float(np.std(arr))

    @property
    def loss_rate(self) -> float:
        if not self._losses:
            return 0.0
        return sum(self._losses) / len(self._losses)

    @property
    def bandwidth_estimate_kbps(self) -> float:
        """Rough estimate; caller should combine with encoder output size."""
        if not self._arrivals_us:
            return 0.0
        mean_interval_us = float(np.mean(self._arrivals_us))
        if mean_interval_us < 1.0:
            return 0.0
        # Assume 20ms frames at current bitrate — approximate
        return 1_000_000.0 / mean_interval_us * _Env.BITRATE_KBPS / 1000.0

    def summary(self) -> dict:
        return {
            "rtt_ms": round(self.rtt_ms, 2),
            "jitter_ms": round(self.jitter_ms, 2),
            "loss_rate": round(self.loss_rate, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OGG CONTAINER (RFC 3533 + RFC 7587)
# ═══════════════════════════════════════════════════════════════════════════════

# OGG page magic capture pattern
_OGG_MAGIC  = b"OggS"
_OGG_HEADER = struct.Struct("<4sBBqIIIB")   # without segment table
_OGG_HEADER_SIZE = _OGG_HEADER.size         # 27 bytes


def _ogg_crc32(data: bytes) -> int:
    """
    CRC32 using the Ogg polynomial 0x04C11DB7 (ISO 3309 / ITU-T V.42).
    Standard Python crc32 uses the Ethernet polynomial — not the same.
    """
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if (crc & 0x80000000) else (crc << 1)
    return crc & 0xFFFFFFFF


@dataclass(slots=True)
class OggPage:
    """
    Decoded representation of one OGG page (RFC 3533 §6).
    Carries one or more logical bitstream packets.
    """
    version:         int    # Must be 0x00
    header_type:     int    # 0x01=continued, 0x02=first, 0x04=last
    granule_pos:     int    # Encoder-specific timestamp
    serial:          int    # Logical bitstream serial number
    sequence:        int    # Page sequence within stream
    packets:         List[bytes]   # Fully-assembled packets on this page
    continued_frag:  bytes = b""  # Trailing lace that spans to next page

    @property
    def is_first(self) -> bool:
        return bool(self.header_type & 0x02)

    @property
    def is_last(self) -> bool:
        return bool(self.header_type & 0x04)

    @property
    def is_continued(self) -> bool:
        return bool(self.header_type & 0x01)


class OggPageParser:
    """
    Streaming OGG page parser.  Feed arbitrary byte chunks via push(); consume
    fully-parsed OggPage objects via the pages() generator.

    Handles:
    - Partial pages spanning multiple push() calls
    - Lacing (packets spanning multiple pages)
    - Sync recovery after corruption/truncation
    """

    def __init__(self) -> None:
        self._buf  = bytearray()
        self._pending_frag = b""   # Carry-over fragment from previous page

    def push(self, data: bytes) -> None:
        self._buf.extend(data)

    def pages(self) -> Iterator[OggPage]:
        buf = self._buf
        while True:
            # Locate next OggS sync word
            idx = buf.find(_OGG_MAGIC)
            if idx < 0:
                # Keep last 3 bytes in case sync spans buffer boundaries
                if len(buf) > 3:
                    self._buf = bytearray(buf[-3:])
                return
            if idx > 0:
                # Skip garbage before sync
                del buf[:idx]

            if len(buf) < _OGG_HEADER_SIZE:
                return  # Need more data

            (magic, version, header_type, granule_pos,
             serial, sequence, crc_stored, n_segs) = _OGG_HEADER.unpack_from(buf, 0)

            if magic != _OGG_MAGIC or version != 0:
                # Bad sync — advance past this magic and try again
                del buf[:4]
                continue

            table_end = _OGG_HEADER_SIZE + n_segs
            if len(buf) < table_end:
                return  # Need more data

            seg_table = buf[_OGG_HEADER_SIZE:table_end]
            page_body_size = sum(seg_table)
            page_end = table_end + page_body_size

            if len(buf) < page_end:
                return  # Need more data

            # Verify CRC (zero out the stored CRC field before computing)
            raw_page = bytearray(buf[:page_end])
            raw_page[22:26] = b"\x00\x00\x00\x00"
            if _ogg_crc32(bytes(raw_page)) != crc_stored:
                # Corrupt page — skip past this sync word
                del buf[:4]
                log.debug("ogg_crc_mismatch: dropping page seq=%d", sequence)
                continue

            # Split body into packets using lacing
            page_body = bytes(buf[table_end:page_end])
            packets, self._pending_frag = _lace_packets(
                seg_table, page_body, self._pending_frag,
                bool(header_type & 0x01),
            )

            del buf[:page_end]

            yield OggPage(
                version=version,
                header_type=header_type,
                granule_pos=granule_pos,
                serial=serial,
                sequence=sequence,
                packets=packets,
                continued_frag=self._pending_frag,
            )


def _lace_packets(
    seg_table: bytes | bytearray,
    body: bytes,
    carry: bytes,
    continued: bool,
) -> Tuple[List[bytes], bytes]:
    """
    Reconstruct logical packets from an OGG lacing table + body.
    Returns (complete_packets, leftover_fragment_for_next_page).
    """
    packets: List[bytes] = []
    current = bytearray(carry if continued else b"")
    pos = 0

    for seg_size in seg_table:
        chunk = body[pos:pos + seg_size]
        current.extend(chunk)
        pos += seg_size
        if seg_size < 255:
            # Last segment of this packet (terminated by short/zero lace)
            packets.append(bytes(current))
            current = bytearray()

    # If last lace is 255 the packet spans to next page
    return packets, bytes(current)


class OggWriter:
    """
    Constructs a minimal but spec-compliant OGG/Opus byte stream from raw
    Opus packets.  Writes the mandatory OpusHead and OpusTags BOS pages on
    first use, then wraps each packet in a data page.

    Thread-safe: one writer per stream direction.
    """

    # Opus identification header (RFC 7587 §5.1)
    _OPUS_HEAD_V  = b"OpusHead"
    _OPUS_TAGS_V  = b"OpusTags"
    _PRE_SKIP     = 312   # Encoder look-ahead in samples at 48 kHz

    def __init__(self, config: CodecConfig, serial: Optional[int] = None) -> None:
        self._cfg     = config
        self._serial  = serial or int.from_bytes(os.urandom(4), "little")
        self._seq     = 0
        self._headers_written = False
        self._granule = 0

    def headers(self) -> bytes:
        """Return the two mandatory BOS pages (OpusHead + OpusTags)."""
        head_payload = struct.pack(
            "<8sBBHIhB",
            self._OPUS_HEAD_V,    # magic
            1,                    # version
            self._cfg.channels,
            self._PRE_SKIP,
            self._cfg.sample_rate,
            0,                    # output gain Q7.8
            0,                    # channel mapping family (0 = mono/stereo RTP)
        )
        tags_payload = struct.pack("<8sI", self._OPUS_TAGS_V, 0) + struct.pack("<I", 0)

        head_page = self._make_page(head_payload, header_type=0x02, granule=0)
        tags_page = self._make_page(tags_payload, header_type=0x00, granule=0)
        self._headers_written = True
        return head_page + tags_page

    def packet_to_page(self, opus_bytes: bytes, last: bool = False) -> bytes:
        """Wrap one Opus packet in an OGG data page."""
        if not self._headers_written:
            raise RuntimeError("Call headers() before writing packets.")
        self._granule += self._cfg.frame_samples
        flag = 0x04 if last else 0x00
        return self._make_page(opus_bytes, header_type=flag, granule=self._granule)

    def _make_page(self, payload: bytes, header_type: int, granule: int) -> bytes:
        """
        Build a valid OGG page with correct lacing and CRC.
        Single-packet pages only (sufficient for Opus streaming).
        """
        # Build segment table: chunks of 255 bytes + remainder
        segs: List[int] = []
        remaining = len(payload)
        while remaining >= 255:
            segs.append(255)
            remaining -= 255
        segs.append(remaining)  # Terminator segment (may be 0)

        n_segs = len(segs)
        header = _OGG_HEADER.pack(
            _OGG_MAGIC, 0, header_type, granule,
            self._serial, self._seq, 0, n_segs,   # CRC placeholder = 0
        )
        seg_table = bytes(segs)
        raw = header + seg_table + payload

        # Compute and splice CRC
        crc = _ogg_crc32(raw)
        raw = bytearray(raw)
        struct.pack_into("<I", raw, 22, crc)
        self._seq += 1
        return bytes(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PROCESS LIFECYCLE & CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitState(IntEnum):
    CLOSED    = auto()   # Normal operation
    HALF_OPEN = auto()   # Probing recovery
    OPEN      = auto()   # Failing fast


class CircuitBreaker:
    """
    Classic circuit breaker with half-open probing.

    Counts consecutive failures within the same process handle.
    Opens after CB_FAILURE_THRESHOLD failures, recovers after CB_RECOVERY_TIMEOUT.
    """

    def __init__(
        self,
        failure_threshold: int  = _Env.CB_FAILURE_THRESHOLD,
        recovery_timeout:  float = _Env.CB_RECOVERY_TIMEOUT,
    ) -> None:
        self._threshold       = failure_threshold
        self._recovery_timeout= recovery_timeout
        self._failures        = 0
        self._state           = CircuitState.CLOSED
        self._opened_at: Optional[float] = None
        self._lock            = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self._recovery_timeout:
                self._state = CircuitState.HALF_OPEN
        return self._state

    async def call(self, coro, *args, **kwargs):
        """Execute *coro* under circuit protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError("Circuit is open; request rejected.")

        try:
            result = await coro(*args, **kwargs)
            async with self._lock:
                self._failures = 0
                if self._state == CircuitState.HALF_OPEN:
                    self._state = CircuitState.CLOSED
                    log.info("circuit_breaker: recovered → CLOSED")
            return result
        except Exception as exc:
            async with self._lock:
                self._failures += 1
                if self._failures >= self._threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    log.error(
                        "circuit_breaker: opened after %d failures — last: %s",
                        self._failures, exc,
                    )
            raise


class ProcessHandle:
    """
    Manages one FFmpeg subprocess as an asyncio coroutine-friendly resource.

    Responsibilities:
    - Launch with `asyncio.create_subprocess_exec` (true async I/O, no threads)
    - Drain stderr to a ring-buffer for diagnostics
    - Detect process death and signal via `died` event
    - Graceful shutdown: close stdin → wait → SIGTERM → SIGKILL
    - Expose `asyncio.StreamReader` / `asyncio.StreamWriter` for data flow

    Does NOT restart itself; restart policy lives in the owning pool or stream.
    """

    _STDERR_RING = 64   # lines

    def __init__(self, cmd: List[str], label: str = "ffmpeg") -> None:
        self._cmd      = cmd
        self._label    = label
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_lines: Deque[str] = collections.deque(maxlen=self._STDERR_RING)
        self._stderr_task: Optional[asyncio.Task] = None
        self.died = asyncio.Event()
        self._started  = False

    @property
    def stdin(self) -> asyncio.StreamWriter:
        assert self._proc and self._proc.stdin
        return self._proc.stdin

    @property
    def stdout(self) -> asyncio.StreamReader:
        assert self._proc and self._proc.stdout
        return self._proc.stdout

    @property
    def pid(self) -> Optional[int]:
        return self._proc.pid if self._proc else None

    @property
    def alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and not self.died.is_set()
        )

    async def start(self) -> None:
        if self._started:
            return
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_Env.PIPE_BUFFER_BYTES,
            )
        except FileNotFoundError:
            raise FFmpegNotFoundError(FFmpegNotFoundError.INSTALL_HINT)

        self._started = True
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(), name=f"{self._label}-stderr-{self.pid}"
        )
        asyncio.create_task(self._watch_death(), name=f"{self._label}-watch-{self.pid}")
        log.debug("%s started: pid=%d", self._label, self.pid)

    async def stop(self, timeout: float = 3.0) -> None:
        if not self._proc:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.is_closing():
                self._proc.stdin.close()
                try:
                    await asyncio.wait_for(self._proc.stdin.wait_closed(), 1.0)
                except asyncio.TimeoutError:
                    pass
            await asyncio.wait_for(self._proc.wait(), timeout / 2)
        except asyncio.TimeoutError:
            log.warning("%s: graceful stop timed out; terminating", self._label)
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout / 2)
            except asyncio.TimeoutError:
                log.error("%s: terminate timed out; killing", self._label)
                self._proc.kill()
                await self._proc.wait()
        finally:
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
            self.died.set()
            log.debug("%s stopped: pid=%d rc=%s", self._label, self.pid or -1,
                      self._proc.returncode)

    async def _drain_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        try:
            async for line in self._proc.stderr:
                decoded = line.decode(errors="replace").rstrip()
                self._stderr_lines.append(decoded)
                # Only log actual errors from ffmpeg, skip info/verbose lines
                if "error" in decoded.lower() or "invalid" in decoded.lower():
                    log.warning("%s stderr: %s", self._label, decoded)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.debug("%s stderr drain ended: %s", self._label, exc)

    async def _watch_death(self) -> None:
        assert self._proc
        await self._proc.wait()
        if not self.died.is_set():
            rc = self._proc.returncode
            log.error(
                "%s died unexpectedly: pid=%d rc=%d | last stderr: %s",
                self._label, self.pid or -1, rc or -1,
                " | ".join(list(self._stderr_lines)[-5:]),
            )
            self.died.set()

    def last_stderr(self, n: int = 10) -> List[str]:
        return list(self._stderr_lines)[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — ADAPTIVE BITRATE CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveBitrateController:
    """
    AIMD (Additive Increase / Multiplicative Decrease) bitrate adaptation,
    inspired by WebRTC GCC and RFC 8698 (RMCAT).

    Policy:
    - Loss > 10%            : multiply bitrate × 0.85 + enable FEC
    - Loss > 5%             : multiply bitrate × 0.90
    - Loss 0–2% + stable   : add +2 kbps per probe interval (max once per 200ms)
    - Jitter spike > 3× avg: reduce by 10%
    - DTX silence frames   : hold bitrate (no change)

    Thread-safe for concurrent reads; all mutations are async-protected.
    """

    _PROBE_INTERVAL = 0.2    # seconds between upward probes
    _BACKOFF_MIN    = 0.5    # minimum seconds between backoffs

    def __init__(
        self,
        cfg: CodecConfig,
        stats: NetworkStats,
        on_change: Optional[Callable[[int, bool], None]] = None,
    ) -> None:
        self._cfg           = cfg
        self._stats         = stats
        self._on_change     = on_change    # (new_kbps, new_fec) → None
        self._bitrate       = cfg.bitrate_kbps
        self._fec           = cfg.fec
        self._last_probe    = 0.0
        self._last_backoff  = 0.0
        self._lock          = asyncio.Lock()

    @property
    def bitrate_kbps(self) -> int:
        return self._bitrate

    @property
    def fec_enabled(self) -> bool:
        return self._fec

    async def tick(self) -> None:
        """
        Called periodically (e.g. once per encoded frame).
        Evaluates current conditions and adjusts if thresholds crossed.
        """
        async with self._lock:
            now = time.monotonic()
            loss  = self._stats.loss_rate
            jitter= self._stats.jitter_ms

            new_rate = self._bitrate
            new_fec  = self._fec

            # --- Multiplicative decrease ----------------------------------------
            if loss > 0.10 and (now - self._last_backoff) > self._BACKOFF_MIN:
                new_rate = max(_Env.ABR_MIN_KBPS, int(self._bitrate * 0.85))
                new_fec  = True
                self._last_backoff = now
            elif loss > 0.05 and (now - self._last_backoff) > self._BACKOFF_MIN:
                new_rate = max(_Env.ABR_MIN_KBPS, int(self._bitrate * 0.90))
                self._last_backoff = now

            # --- Jitter penalty ─────────────────────────────────────────────────
            elif jitter > 3.0 * self._cfg.frame_ms and (now - self._last_backoff) > self._BACKOFF_MIN:
                new_rate = max(_Env.ABR_MIN_KBPS, int(self._bitrate * 0.90))
                self._last_backoff = now

            # --- Additive increase ──────────────────────────────────────────────
            elif loss < 0.02 and (now - self._last_probe) > self._PROBE_INTERVAL:
                new_rate = min(_Env.ABR_MAX_KBPS, self._bitrate + 2)
                if new_rate == self._bitrate and new_fec == self._fec:
                    # Nothing changed
                    return
                self._last_probe = now

            if new_rate == self._bitrate and new_fec == self._fec:
                return

            log.debug(
                "abr: %d→%d kbps fec=%s loss=%.1f%% jitter=%.1fms",
                self._bitrate, new_rate, new_fec, loss * 100, jitter,
            )
            self._bitrate = new_rate
            self._fec = new_fec

            if self._on_change:
                self._on_change(new_rate, new_fec)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PACKET LOSS CONCEALMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PacketLossConcealer:
    """
    Application-layer PLC for when the jitter buffer detects a gap.

    Strategy:
      1. Repeat last good frame at 100% amplitude (1st miss)
      2. Attenuate by 6 dB per missing frame thereafter (exponential fade)
      3. After 5 consecutive misses, generate near-silence (comfort noise)

    libopus has its own internal PLC, but we supplement it at the framing
    layer so we can also feed the decoder a correct synthetic OGG page,
    preventing decoder state corruption.
    """

    _DB_PER_MISS   = 6.0     # dB attenuation per missing frame
    _MAX_REPEAT    = 5       # frames before comfort-noise fallback

    def __init__(self, cfg: CodecConfig) -> None:
        self._cfg         = cfg
        self._last_pcm: Optional[np.ndarray] = None
        self._miss_count  = 0

    def update_last_good(self, pcm: np.ndarray) -> None:
        """Register the most recently decoded good frame."""
        self._last_pcm = pcm.copy()
        self._miss_count = 0

    def conceal(self) -> np.ndarray:
        """
        Generate a concealment frame.
        Returns a numpy array with the same shape/dtype as good frames.
        """
        self._miss_count += 1
        n = self._cfg.frame_samples

        if self._last_pcm is None or self._miss_count > self._MAX_REPEAT:
            # Comfort noise: low-amplitude white noise
            rng = np.random.default_rng()
            if np.issubdtype(self._cfg.numpy_dtype, np.integer):
                noise = rng.integers(-64, 64, size=n, dtype=self._cfg.numpy_dtype)
            else:
                noise = rng.standard_normal(n).astype(self._cfg.numpy_dtype) * 0.001
            return noise

        # Exponential amplitude taper
        attenuation = 10.0 ** (-self._DB_PER_MISS * self._miss_count / 20.0)
        concealed = (self._last_pcm[:n] * attenuation).astype(self._cfg.numpy_dtype)
        return concealed

    def reset(self) -> None:
        self._last_pcm = None
        self._miss_count = 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — ADAPTIVE JITTER BUFFER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _JBSlot:
    payload:   bytes
    recv_us:   float
    duration_ms: int


class AdaptiveJitterBuffer:
    """
    Adaptive playout jitter buffer with NetEQ-inspired delay estimation.

    Algorithm:
    - Maintain a sorted seq-keyed buffer
    - Estimate jitter via RFC 3550 §A.8 moving variance (EWMA, α=0.125)
    - Target delay = max(JB_MIN, min(JB_MAX, jitter_estimate + safety_margin))
    - Increase delay immediately; decrease gradually (leak rate = 1ms/frame)
    - Detect missing packets when playout time exceeds slot timestamp
    - Report gaps to PacketLossConcealer

    Async: all public methods are coroutine-safe.
    """

    _ALPHA         = 0.125    # EWMA smoothing (RFC 3550)
    _SAFETY_MS     = 20.0     # Buffer above jitter estimate
    _DECREASE_RATE = 1.0      # ms reduction per consumed frame (slow decrease)

    def __init__(
        self,
        cfg: CodecConfig,
        stats: NetworkStats,
        concealer: PacketLossConcealer,
    ) -> None:
        self._cfg        = cfg
        self._stats      = stats
        self._plc        = concealer

        self._buffer: Dict[int, _JBSlot] = {}
        self._next_seq: Optional[int]    = None
        self._delay_ms   = float(_Env.JB_TARGET_MS)
        self._jitter_ms  = 0.0
        self._last_push_us = 0.0
        self._lock = asyncio.Lock()

    @property
    def target_delay_ms(self) -> float:
        return self._delay_ms

    async def push(self, seq: int, payload: bytes, duration_ms: int = 20) -> None:
        """Insert an incoming Opus packet into the buffer."""
        async with self._lock:
            now_us = time.monotonic_ns() / 1_000.0

            # RFC 3550 jitter estimation
            if self._last_push_us:
                transit = now_us - self._last_push_us
                d = abs(transit - duration_ms * 1_000.0)
                self._jitter_ms = (1.0 - self._ALPHA) * self._jitter_ms + self._ALPHA * d / 1_000.0
            self._last_push_us = now_us
            self._stats.record_arrival(lost=False)

            # Adapt target delay: increase fast, decrease slowly
            new_target = min(
                float(_Env.JB_MAX_MS),
                max(
                    float(_Env.JB_MIN_MS),
                    self._jitter_ms + self._SAFETY_MS,
                ),
            )
            if new_target > self._delay_ms:
                self._delay_ms = new_target
            else:
                self._delay_ms = max(
                    _Env.JB_MIN_MS,
                    int(self._delay_ms - self._DECREASE_RATE)
                )

            if self._next_seq is None:
                self._next_seq = seq

            self._buffer[seq] = _JBSlot(payload=payload, recv_us=now_us, duration_ms=duration_ms)

    async def pull(self) -> Tuple[bytes, bool]:
        """
        Retrieve the next packet for playout.
        Blocks until the packet is due or a gap exceeds the target delay.

        Returns:
            (opus_bytes, is_concealment) — concealment=True _signals PLC was used.
        """
        while True:
            async with self._lock:
                if self._next_seq is not None and self._next_seq in self._buffer:
                    slot = self._buffer.pop(self._next_seq)
                    now_us = time.monotonic_ns() / 1_000.0
                    age_ms = (now_us - slot.recv_us) / 1_000.0

                    if age_ms < self._delay_ms:
                        # Too early — sleep until playout time
                        sleep_s = (self._delay_ms - age_ms) / 1_000.0
                    else:
                        sleep_s = 0.0

                    self._next_seq += 1
                    payload = slot.payload

                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                    self._plc.update_last_good(  # will be updated after decode
                        np.zeros(self._cfg.frame_samples, dtype=self._cfg.numpy_dtype)
                    )
                    return payload, False

                # Check for gap (missing sequence)
                if self._next_seq is not None:
                    gap_check_deadline_ms = self._delay_ms + self._cfg.frame_ms * 3
                    now_us = time.monotonic_ns() / 1_000.0
                    oldest_recv = min(
                        (s.recv_us for s in self._buffer.values()),
                        default=None,
                    )
                    gap_age_ms = ((now_us - (oldest_recv or now_us)) / 1_000.0)
                    if oldest_recv and gap_age_ms > gap_check_deadline_ms:
                        # Declare loss, advance
                        self._stats.record_arrival(lost=True)
                        self._next_seq = (self._next_seq or 0) + 1
                        concealment = self._plc.conceal()
                        return concealment.tobytes(), True

            await asyncio.sleep(self._cfg.frame_ms / 2_000.0)

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def jitter_ms(self) -> float:
        return round(self._jitter_ms, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — ENCODER PROCESS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_encode_cmd(cfg: CodecConfig, extra_loglevel: str = "error") -> List[str]:
    """Construct the FFmpeg command for PCM → OGG/Opus encoding."""
    fmt = cfg.ffmpeg_format
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", extra_loglevel,
        # Input: raw PCM on stdin
        "-f", fmt,
        "-ar", str(cfg.sample_rate),
        "-ac", str(cfg.channels),
        "-i", "pipe:0",
        # Codec
        "-c:a", "libopus",
        "-b:a", f"{cfg.bitrate_kbps}k",
        "-vbr", "on" if cfg.vbr else "off",
        "-compression_level", str(cfg.complexity),
        "-application", cfg.application,
        "-frame_duration", str(cfg.frame_ms),
        "-packet_loss", "0",
        # Output: OGG container on stdout (gives us proper framing)
        "-f", "ogg",
        "pipe:1",
    ]
    if cfg.fec:
        cmd += ["-fec", "1"]
    if cfg.dtx:
        cmd += ["-dtx", "1"]
    return cmd


class EncoderProcess:
    """
    Single long-running FFmpeg subprocess that encodes PCM → OGG/Opus.

    Data flow:
        write_pcm(bytes) → [stdin pipe] → FFmpeg → [stdout pipe] → OGG pages
                                                  → OggPageParser → Opus packets
                                                  → _out_q (asyncio.Queue)

    Lifecycle methods are coroutines; data I/O is non-blocking.
    """

    def __init__(self, cfg: CodecConfig, seq_offset: int = 0) -> None:
        self._cfg      = cfg
        self._handle   = ProcessHandle(_build_encode_cmd(cfg), label="enc")
        self._parser   = OggPageParser()
        self._out_q: asyncio.Queue[OpusPacket] = asyncio.Queue(maxsize=256)
        self._read_task: Optional[asyncio.Task] = None
        self._seq      = seq_offset
        self._closed   = False
        self._bytes_in = 0

    @property
    def alive(self) -> bool:
        return self._handle.alive

    async def start(self) -> None:
        await self._handle.start()
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"enc-read-{self._handle.pid}"
        )
        log.debug("encoder_process started pid=%d", self._handle.pid)

    async def encode(self, pcm: bytes) -> None:
        """
        Write raw PCM bytes to the encoder's stdin.
        Non-blocking: returns as soon as bytes are queued to the kernel pipe.
        """
        if not self._handle.alive:
            raise ProcessDiedError("Encoder process is not running.")
        self._handle.stdin.write(pcm)
        await self._handle.stdin.drain()
        self._bytes_in += len(pcm)

    async def next_packet(self) -> OpusPacket:
        """
        Await the next encoded Opus packet from the output queue.
        Raises ProcessDiedError if the process dies before delivering a packet.
        """
        while True:
            try:
                return self._out_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self._handle.died.is_set() and self._out_q.empty():
                raise ProcessDiedError("Encoder process died.")
            await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self._closed = True
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        await self._handle.stop()

    async def _read_loop(self) -> None:
        """Continuously read OGG pages from stdout and extract Opus packets."""
        stdout = self._handle.stdout
        try:
            while not self._closed and self._handle.alive:
                # Read 4 KiB at a time — enough for several OGG pages
                chunk = await stdout.read(4096)
                if not chunk:
                    break
                self._parser.push(chunk)
                for page in self._parser.pages():
                    for pkt in page.packets:
                        if not pkt:
                            continue
                        opus_pkt = OpusPacket(
                            seq=self._seq,
                            timestamp_us=time.monotonic_ns() // 1_000,
                            payload=pkt,
                            duration_ms=self._cfg.frame_ms,
                            config=self._cfg,
                            is_silence=False,
                            has_fec=self._cfg.fec,
                        )
                        self._seq += 1
                        try:
                            self._out_q.put_nowait(opus_pkt)
                        except asyncio.QueueFull:
                            log.debug("encoder output queue full — dropping oldest packet")
                            self._out_q.get_nowait()
                            self._out_q.put_nowait(opus_pkt)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("encoder read loop error: %s", exc)
        finally:
            self._handle.died.set()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DECODER PROCESS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_decode_cmd(cfg: CodecConfig, extra_loglevel: str = "error") -> List[str]:
    """Construct the FFmpeg command for OGG/Opus → PCM decoding."""
    fmt = cfg.ffmpeg_format
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", extra_loglevel,
        # Input: OGG/Opus container on stdin
        "-f", "ogg",
        "-i", "pipe:0",
        # Resample to target rate
        "-ar", str(cfg.sample_rate),
        "-ac", str(cfg.channels),
        # Output: raw PCM on stdout
        "-f", fmt,
        "pipe:1",
    ]


class DecoderProcess:
    """
    Single long-running FFmpeg subprocess that decodes OGG/Opus → PCM.

    Data flow:
        push_ogg_page(bytes) → [stdin pipe] → FFmpeg → [stdout pipe] → PCM bytes
                                                     → PCMFrame → _out_q

    The caller is responsible for:
      - Sending the OGG headers before any data pages (use OggWriter.headers())
      - Sending well-formed OGG pages (use OggWriter.packet_to_page())
    """

    def __init__(self, cfg: CodecConfig, seq_offset: int = 0) -> None:
        self._cfg       = cfg
        self._handle    = ProcessHandle(_build_decode_cmd(cfg), label="dec")
        self._writer    = OggWriter(cfg)
        self._out_q: asyncio.Queue[PCMFrame] = asyncio.Queue(maxsize=512)
        self._read_task: Optional[asyncio.Task] = None
        self._seq       = seq_offset
        self._headers_sent = False
        self._closed    = False
        self._buf       = bytearray()
        # How many bytes = one PCM frame
        self._frame_bytes = cfg.frame_bytes

    @property
    def alive(self) -> bool:
        return self._handle.alive

    async def start(self) -> None:
        await self._handle.start()
        # Immediately send OGG headers so FFmpeg can identify the stream
        headers = self._writer.headers()
        self._handle.stdin.write(headers)
        await self._handle.stdin.drain()
        self._headers_sent = True
        self._read_task = asyncio.create_task(
            self._read_loop(), name=f"dec-read-{self._handle.pid}"
        )
        log.debug("decoder_process started pid=%d", self._handle.pid)

    async def push_opus_packet(self, opus_bytes: bytes, last: bool = False) -> None:
        """
        Encode one Opus packet into an OGG page and write to the decoder's stdin.
        """
        if not self._headers_sent:
            raise RuntimeError("Decoder not started; call start() first.")
        if not self._handle.alive:
            raise ProcessDiedError("Decoder process is not running.")
        ogg_page = self._writer.packet_to_page(opus_bytes, last=last)
        self._handle.stdin.write(ogg_page)
        await self._handle.stdin.drain()

    async def next_frame(self) -> PCMFrame:
        """
        Await the next decoded PCMFrame from the output queue.
        """
        while True:
            try:
                return self._out_q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            if self._handle.died.is_set() and self._out_q.empty():
                raise ProcessDiedError("Decoder process died.")
            await asyncio.sleep(0.001)

    async def stop(self) -> None:
        self._closed = True
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        await self._handle.stop()

    async def _read_loop(self) -> None:
        """Continuously read raw PCM from stdout and emit PCMFrames."""
        stdout = self._handle.stdout
        try:
            while not self._closed and self._handle.alive:
                chunk = await stdout.read(self._frame_bytes * 4)
                if not chunk:
                    break
                self._buf.extend(chunk)
                # Emit complete frames
                while len(self._buf) >= self._frame_bytes:
                    frame_bytes = bytes(self._buf[:self._frame_bytes])
                    del self._buf[:self._frame_bytes]
                    pcm = np.frombuffer(frame_bytes, dtype=self._cfg.numpy_dtype)
                    frame = PCMFrame(
                        data=pcm,
                        seq=self._seq,
                        ts_us=time.monotonic_ns() // 1_000,
                        config=self._cfg,
                    )
                    self._seq += 1
                    try:
                        self._out_q.put_nowait(frame)
                    except asyncio.QueueFull:
                        log.debug("decoder output queue full — dropping oldest frame")
                        self._out_q.get_nowait()
                        self._out_q.put_nowait(frame)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("decoder read loop error: %s", exc)
        finally:
            self._handle.died.set()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — PROCESS POOLS (VERTICAL SCALING)
# ═══════════════════════════════════════════════════════════════════════════════

class _Pool(ABC):
    """
    Generic pool of codec worker processes for vertical scaling.

    Load balancing: round-robin with health skip.
    Auto-restart: dead workers are replaced transparently.
    Circuit breaker: per-worker, shared failure budget.
    """

    def __init__(self, size: int, cfg: CodecConfig) -> None:
        self._size   = size
        self._cfg    = cfg
        self._cursor = 0
        self._lock   = asyncio.Lock()

    @abstractmethod
    async def _spawn_worker(self) -> Any:
        ...

    @abstractmethod
    async def acquire(self) -> Any:
        ...

    async def _rotate(self) -> None:
        async with self._lock:
            self._cursor = (self._cursor + 1) % self._size


class EncoderPool(_Pool):
    """
    Pool of N EncoderProcess instances.

    acquire() returns the next healthy worker in round-robin order.
    Dead workers are automatically restarted in the background.
    """

    def __init__(self, cfg: CodecConfig, size: int = _Env.ENCODER_WORKERS) -> None:
        super().__init__(size, cfg)
        self._workers: List[Optional[EncoderProcess]] = [None] * size
        self._restart_tasks: List[Optional[asyncio.Task]] = [None] * size
        self._ready = False

    async def start(self) -> None:
        await asyncio.gather(*[self._init_worker(i) for i in range(self._size)])
        self._ready = True
        log.info("encoder_pool: %d workers started", self._size)

    async def stop(self) -> None:
        await asyncio.gather(
            *[w.stop() for w in self._workers if w is not None],
            return_exceptions=True,
        )

    async def _init_worker(self, idx: int) -> None:
        w = EncoderProcess(self._cfg, seq_offset=idx * 1_000_000)
        await w.start()
        self._workers[idx] = w

    async def _spawn_worker(self) -> EncoderProcess:
        w = EncoderProcess(self._cfg)
        await w.start()
        return w

    async def acquire(self) -> EncoderProcess:
        """Return a live EncoderProcess, restarting dead ones."""
        for _ in range(self._size * 2):
            async with self._lock:
                idx = self._cursor
                self._cursor = (self._cursor + 1) % self._size
                w = self._workers[idx]

            if w is None or not w.alive:
                asyncio.create_task(self._restart(idx))
                continue
            return w

        raise ProcessDiedError("All encoder workers are dead.")

    async def _restart(self, idx: int) -> None:
        log.warning("encoder_pool: restarting worker %d", idx)
        try:
            if self._workers[idx]:
                await self._workers[idx].stop()
        except Exception: # noqa
            pass
        new_w = EncoderProcess(self._cfg, seq_offset=idx * 1_000_000)
        await new_w.start()
        self._workers[idx] = new_w
        log.info("encoder_pool: worker %d restarted", idx)


class DecoderPool(_Pool):
    """
    Pool of N DecoderProcess instances.  Mirror of EncoderPool.
    """

    def __init__(self, cfg: CodecConfig, size: int = _Env.DECODER_WORKERS) -> None:
        super().__init__(size, cfg)
        self._workers: List[Optional[DecoderProcess]] = [None] * size

    async def start(self) -> None:
        await asyncio.gather(*[self._init_worker(i) for i in range(self._size)])
        log.info("decoder_pool: %d workers started", self._size)

    async def stop(self) -> None:
        await asyncio.gather(
            *[w.stop() for w in self._workers if w is not None],
            return_exceptions=True,
        )

    async def _init_worker(self, idx: int) -> None:
        w = DecoderProcess(self._cfg, seq_offset=idx * 1_000_000)
        await w.start()
        self._workers[idx] = w

    async def _spawn_worker(self) -> DecoderProcess:
        w = DecoderProcess(self._cfg)
        await w.start()
        return w

    async def acquire(self) -> DecoderProcess:
        for _ in range(self._size * 2):
            async with self._lock:
                idx = self._cursor
                self._cursor = (self._cursor + 1) % self._size
                w = self._workers[idx]
            if w is None or not w.alive:
                asyncio.create_task(self._restart(idx))
                continue
            return w
        raise ProcessDiedError("All decoder workers are dead.")

    async def _restart(self, idx: int) -> None:
        log.warning("decoder_pool: restarting worker %d", idx)
        try:
            if self._workers[idx]:
                await self._workers[idx].stop()
        except Exception: # noqa
            pass
        new_w = DecoderProcess(self._cfg, seq_offset=idx * 1_000_000)
        await new_w.start()
        self._workers[idx] = new_w
        log.info("decoder_pool: worker %d restarted", idx)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

class CodecMetrics:
    """
    Zero-dependency metrics collector.  Exposes counters and gauges
    compatible with Prometheus text format.  If `prometheus_client` is
    available, delegates to it; otherwise tracks in-process.

    Usage:
        metrics = CodecMetrics("my_app")
        metrics.encode_packets.inc()
        metrics.encode_latency_ms.observe(12.5)
        print(metrics.render())  # Prometheus text format
    """

    class _Counter:
        __slots__ = ("_name", "_help", "_val")
        def __init__(self, name: str, help_: str) -> None:
            self._name  = name
            self._help  = help_
            self._val   = 0.0
        def inc(self, n: float = 1.0) -> None:
            self._val += n
        @property
        def value(self) -> float:
            return self._val

    class _Gauge:
        __slots__ = ("_name", "_help", "_val")
        def __init__(self, name: str, help_: str) -> None:
            self._name = name
            self._help = help_
            self._val  = 0.0
        def set(self, v: float) -> None:
            self._val = v
        @property
        def value(self) -> float:
            return self._val

    class _Histogram:
        __slots__ = ("_name", "_help", "_sum", "_count", "_buckets_def", "_bucket_counts")
        _DEFAULT_BUCKETS = (1, 5, 10, 20, 50, 100, 200, 500)
        def __init__(self, name: str, help_: str, buckets: Sequence[float] = ()) -> None:
            self._name         = name
            self._help         = help_
            self._sum          = 0.0
            self._count        = 0
            self._buckets_def  = tuple(buckets) or self._DEFAULT_BUCKETS
            self._bucket_counts = [0] * len(self._buckets_def)
        def observe(self, v: float) -> None:
            self._sum   += v
            self._count += 1
            for i, upper in enumerate(self._buckets_def):
                if v <= upper:
                    self._bucket_counts[i] += 1

    def __init__(self, namespace: str = "opus_ffmpeg") -> None:
        ns = namespace
        # Encode
        self.encode_packets      = self._Counter(f"{ns}_encode_packets_total", "Opus packets encoded")
        self.encode_bytes_in     = self._Counter(f"{ns}_encode_pcm_bytes_total", "PCM bytes sent to encoder")
        self.encode_bytes_out    = self._Counter(f"{ns}_encode_opus_bytes_total", "Opus bytes emitted")
        self.encode_drops        = self._Counter(f"{ns}_encode_drops_total", "Encoder queue overflows")
        self.encode_latency_ms   = self._Histogram(f"{ns}_encode_latency_ms", "Encode latency histogram")
        # Decode
        self.decode_packets      = self._Counter(f"{ns}_decode_packets_total", "Opus packets decoded")
        self.decode_plc_frames   = self._Counter(f"{ns}_decode_plc_frames_total", "PLC concealment frames")
        self.decode_loss         = self._Counter(f"{ns}_decode_loss_total", "Declared lost packets")
        self.decode_latency_ms   = self._Histogram(f"{ns}_decode_latency_ms", "Decode latency histogram")
        # Buffer
        self.jitter_buffer_depth = self._Gauge(f"{ns}_jitter_buffer_depth", "JB packet depth")
        self.jitter_ms           = self._Gauge(f"{ns}_jitter_ms", "Estimated network jitter")
        self.target_delay_ms     = self._Gauge(f"{ns}_target_delay_ms", "JB target playout delay")
        # ABR
        self.bitrate_kbps        = self._Gauge(f"{ns}_bitrate_kbps", "Current encoder bitrate")
        self.loss_rate           = self._Gauge(f"{ns}_loss_rate", "Packet loss rate (0–1)")
        # Process health
        self.encoder_restarts    = self._Counter(f"{ns}_encoder_restarts_total", "Encoder process restarts")
        self.decoder_restarts    = self._Counter(f"{ns}_decoder_restarts_total", "Decoder process restarts")

        self._all: List[Any] = [
            self.encode_packets, self.encode_bytes_in, self.encode_bytes_out,
            self.encode_drops, self.encode_latency_ms,
            self.decode_packets, self.decode_plc_frames, self.decode_loss,
            self.decode_latency_ms, self.jitter_buffer_depth, self.jitter_ms,
            self.target_delay_ms, self.bitrate_kbps, self.loss_rate,
            self.encoder_restarts, self.decoder_restarts,
        ]

    # noinspection PyProtectedMember
    def render(self) -> str:
        """Emit Prometheus text format."""
        lines: List[str] = []
        for m in self._all:
            lines.append(f"# HELP {m._name} {m._help}")
            if isinstance(m, self._Histogram):
                lines.append(f"# TYPE {m._name} histogram")
                for i, upper in enumerate(m._buckets_def):
                    lines.append(f'{m._name}_bucket{{le="{upper}"}} {m._bucket_counts[i]}')
                lines.append(f'{m._name}_bucket{{le="+Inf"}} {m._count}')
                lines.append(f"{m._name}_sum {m._sum}")
                lines.append(f"{m._name}_count {m._count}")
            elif isinstance(m, self._Gauge):
                lines.append(f"# TYPE {m._name} gauge")
                lines.append(f"{m._name} {m.value}")
            else:
                lines.append(f"# TYPE {m._name} counter")
                lines.append(f"{m._name} {m.value}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return a plain-dict snapshot for JSON serialization / logging."""
        return {
            "encode": {
                "packets":    self.encode_packets.value,
                "bytes_in":   self.encode_bytes_in.value,
                "bytes_out":  self.encode_bytes_out.value,
                "drops":      self.encode_drops.value,
            },
            "decode": {
                "packets":    self.decode_packets.value,
                "plc_frames": self.decode_plc_frames.value,
                "loss":       self.decode_loss.value,
            },
            "network": {
                "jitter_ms":      self.jitter_ms.value,
                "target_delay_ms":self.target_delay_ms.value,
                "bitrate_kbps":   self.bitrate_kbps.value,
                "loss_rate":      self.loss_rate.value,
            },
        }


# Singleton-style global metrics instance; replace/inject as needed.
_global_metrics = CodecMetrics()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — HIGH-LEVEL INPUT STREAM (Opus → PCM)
# ═══════════════════════════════════════════════════════════════════════════════

class FFmpegPCMInputStream:
    """
    Async, jitter-buffered Opus ingress stream.

    Replaces sounddevice.InputStream for network/file input:

        Opus packets (network/file)
            → push_opus_packet(seq, bytes)
            → AdaptiveJitterBuffer         ← network stats
            → DecoderProcess (libopus/FFmpeg)
            → PacketLossConcealer          ← on gap
            → PCMFrame async iterator
            → audio_engine pipeline

    Async context manager; also supports explicit start()/stop().

    Example:
        cfg = CodecConfig.for_whisper()
        async with FFmpegPCMInputStream(cfg) as stream:
            # Feed from network:
            stream.push_opus_packet(seq=pkt.seq, opus_bytes=pkt.payload)

            # Consume decoded frames:
            async for frame in stream:
                await stt.transcribe(frame)
    """

    def __init__(
        self,
        cfg: Optional[CodecConfig] = None,
        *,
        metrics: Optional[CodecMetrics] = None,
        stats: Optional[NetworkStats] = None,
    ) -> None:
        self._cfg      = cfg or CodecConfig()
        self._metrics  = metrics or _global_metrics
        self._stats    = stats or NetworkStats()
        self._plc      = PacketLossConcealer(self._cfg)
        self._jb       = AdaptiveJitterBuffer(self._cfg, self._stats, self._plc)
        self._decoder: Optional[DecoderProcess] = None
        self._cb       = CircuitBreaker()
        self._pull_task: Optional[asyncio.Task] = None
        self._out_q: asyncio.Queue[Optional[PCMFrame]] = asyncio.Queue(maxsize=512)
        self._started  = False
        self._stopped  = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "FFmpegPCMInputStream":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._started:
            return
        self._decoder = DecoderProcess(self._cfg)
        await self._decoder.start()
        self._pull_task = asyncio.create_task(
            self._pull_loop(), name="jb-pull"
        )
        self._started = True
        log.info(
            "input_stream started: rate=%dHz ch=%d dtype=%s",
            self._cfg.sample_rate, self._cfg.channels, self._cfg.dtype,
        )

    async def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        if self._pull_task:
            self._pull_task.cancel()
            try:
                await self._pull_task
            except asyncio.CancelledError:
                pass
        if self._decoder:
            await self._decoder.stop()
        await self._out_q.put(None)   # Sentinel to terminate __anext__
        log.info("input_stream stopped")

    # ── Ingress ───────────────────────────────────────────────────────────────

    def push_opus_packet(self, seq: int, opus_bytes: bytes, duration_ms: int = 20) -> None:
        """
        Non-blocking: enqueue an incoming Opus packet from the network layer.
        Safe to call from sync or async context.
        """
        if self._stopped:
            return
        asyncio.ensure_future(self._jb.push(seq, opus_bytes, duration_ms))

    def push_opus_bytes(self, opus_bytes: bytes) -> None:
        """
        Convenience: push with an auto-incrementing sequence number.
        For use when the transport does not provide seq numbers.
        """
        seq = getattr(self, "_auto_seq", 0)
        self._auto_seq = seq + 1   # noqa
        self.push_opus_packet(seq, opus_bytes)

    # ── Egress ────────────────────────────────────────────────────────────────

    def __aiter__(self) -> "FFmpegPCMInputStream":
        return self

    async def __anext__(self) -> PCMFrame:
        frame = await self._out_q.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    # ── Internal pipeline ─────────────────────────────────────────────────────

    async def _pull_loop(self) -> None:
        """
        Continuously pull from JitterBuffer → decode → emit PCMFrame.
        Handles PLC frames transparently.
        """
        assert self._decoder
        try:
            while not self._stopped:
                t0 = time.monotonic()

                try:
                    opus_bytes, is_concealment = await self._jb.pull()
                except Exception as exc:
                    log.error("jitter_buffer pull error: %s", exc)
                    await asyncio.sleep(0.02)
                    continue

                if is_concealment:
                    # PLC: generate silence/comfort noise without decoding
                    pcm = self._plc.conceal()
                    frame = PCMFrame(
                        data=pcm,
                        seq=getattr(self, "_pull_seq", 0),
                        ts_us=time.monotonic_ns() // 1_000,
                        config=self._cfg,
                    )
                    self._metrics.decode_plc_frames.inc()
                    self._metrics.decode_loss.inc()
                else:
                    # Real packet: push to decoder subprocess
                    try:
                        await self._cb.call(
                            self._decoder.push_opus_packet, opus_bytes
                        )
                        frame = await self._decoder.next_frame()
                        self._plc.update_last_good(frame.data)
                        self._metrics.decode_packets.inc()
                    except CircuitOpenError:
                        log.warning("input_stream: circuit open, generating PLC")
                        pcm = self._plc.conceal()
                        frame = PCMFrame(
                            data=pcm,
                            seq=getattr(self, "_pull_seq", 0),
                            ts_us=time.monotonic_ns() // 1_000,
                            config=self._cfg,
                        )
                        self._metrics.decode_plc_frames.inc()
                    except ProcessDiedError:
                        log.error("decoder died; attempting restart")
                        await self._restart_decoder()
                        continue

                self._pull_seq = getattr(self, "_pull_seq", 0) + 1  # type: ignore[attr-defined]

                # Update metrics
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                self._metrics.decode_latency_ms.observe(elapsed_ms)
                self._metrics.jitter_buffer_depth.set(self._jb.buffered)
                self._metrics.jitter_ms.set(self._jb.jitter_ms)
                self._metrics.target_delay_ms.set(self._jb.target_delay_ms)
                net = self._stats.summary()
                self._metrics.loss_rate.set(net["loss_rate"])

                try:
                    self._out_q.put_nowait(frame)
                except asyncio.QueueFull:
                    log.debug("output queue full — dropping oldest decoded frame")
                    self._out_q.get_nowait()
                    self._out_q.put_nowait(frame)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.exception("input_stream pull_loop crashed: %s", exc)

    async def _restart_decoder(self) -> None:
        if self._decoder:
            try:
                await self._decoder.stop()
            except Exception: # noqa
                pass
        self._decoder = DecoderProcess(self._cfg)
        await self._decoder.start()
        self._metrics.decoder_restarts.inc()
        log.info("input_stream: decoder restarted")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def network_stats(self) -> NetworkStats:
        return self._stats

    @property
    def jitter_ms(self) -> float:
        return self._jb.jitter_ms


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — HIGH-LEVEL OUTPUT STREAM (PCM → Opus)
# ═══════════════════════════════════════════════════════════════════════════════

class FFmpegPCMOutputStream:
    """
    Async, ABR-controlled Opus egress stream.

    Replaces sounddevice.OutputStream for network output:

        PCMChunk / PCMFrame
            → write(chunk)
            → PreEmphasisFilter            ← optional voice boost
            → EncoderProcess (libopus/FFmpeg)
            → AdaptiveBitrateController    ← network feedback
            → OpusPacket async iterator
            → network layer

    Async context manager; also supports explicit start()/stop().

    Example:
        cfg = CodecConfig.for_openai_tts()
        async with FFmpegPCMOutputStream(cfg) as stream:
            async for tts_chunk in tts_generator:
                await stream.write(tts_chunk)

            async for pkt in stream:        # Consume encoded packets
                await ws.send_bytes(pkt.payload)
    """

    def __init__(
        self,
        cfg: Optional[CodecConfig] = None,
        *,
        metrics: Optional[CodecMetrics] = None,
        stats: Optional[NetworkStats] = None,
        on_bitrate_change: Optional[Callable[[int, bool], None]] = None,
    ) -> None:
        self._cfg      = cfg or CodecConfig()
        self._metrics  = metrics or _global_metrics
        self._stats    = stats or NetworkStats()
        self._encoder: Optional[EncoderProcess] = None
        self._abr      = AdaptiveBitrateController(
            self._cfg, self._stats, on_bitrate_change
        )
        self._cb       = CircuitBreaker()
        self._started  = False
        self._stopped  = False
        self._pkt_q: asyncio.Queue[Optional[OpusPacket]] = asyncio.Queue(maxsize=256)
        self._abr_task: Optional[asyncio.Task] = None
        self._read_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "FFmpegPCMOutputStream":
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def start(self) -> None:
        if self._started:
            return
        self._encoder = EncoderProcess(self._cfg)
        await self._encoder.start()
        self._abr_task = asyncio.create_task(
            self._abr_loop(), name="abr-tick"
        )
        self._read_task = asyncio.create_task(
            self._read_loop(), name="enc-drain"
        )
        self._started = True
        log.info(
            "output_stream started: rate=%dHz ch=%d bitrate=%dkbps",
            self._cfg.sample_rate, self._cfg.channels, self._cfg.bitrate_kbps,
        )

    async def stop(self) -> None:
        if not self._started or self._stopped:
            return
        self._stopped = True
        for task in (self._abr_task, self._read_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self._encoder:
            await self._encoder.stop()
        await self._pkt_q.put(None)   # Sentinel
        log.info("output_stream stopped")

    # ── Ingress ───────────────────────────────────────────────────────────────

    async def write(self, chunk: Any) -> None:
        """
        Encode a PCM chunk.  `chunk` may be:
          - PCMFrame / PCMChunk: reads .data attribute (numpy array)
          - np.ndarray: used directly
          - bytes: forwarded as-is
        """
        if self._stopped:
            return

        if hasattr(chunk, "data"):
            pcm_bytes = chunk.data.tobytes()
        elif isinstance(chunk, np.ndarray):
            pcm_bytes = chunk.tobytes()
        elif isinstance(chunk, (bytes, bytearray)):
            pcm_bytes = bytes(chunk)
        else:
            raise TypeError(f"Unsupported chunk type: {type(chunk)}")

        t0 = time.monotonic()
        try:
            await self._cb.call(self._encoder.encode, pcm_bytes)
            self._metrics.encode_bytes_in.inc(len(pcm_bytes))
        except CircuitOpenError:
            self._metrics.encode_drops.inc()
            log.warning("output_stream: circuit open, dropping frame")
        except ProcessDiedError:
            log.error("encoder died; restarting")
            await self._restart_encoder()
        except Exception as exc:
            log.error("output_stream write error: %s", exc)
            self._metrics.encode_drops.inc()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._metrics.encode_latency_ms.observe(elapsed_ms)

    # ── Egress ────────────────────────────────────────────────────────────────

    def __aiter__(self) -> "FFmpegPCMOutputStream":
        return self

    async def __anext__(self) -> OpusPacket:
        pkt = await self._pkt_q.get()
        if pkt is None:
            raise StopAsyncIteration
        return pkt

    def read_opus_packet_nowait(self) -> Optional[OpusPacket]:
        """Non-blocking read of the next encoded packet (None if none ready)."""
        try:
            pkt = self._pkt_q.get_nowait()
            return pkt  # May be None sentinel — caller should check
        except asyncio.QueueEmpty:
            return None

    # ── Internal pipeline ─────────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        """Drain encoded OpusPackets from the encoder and enqueue them."""
        assert self._encoder
        try:
            while not self._stopped:
                if not self._encoder.alive:
                    await asyncio.sleep(0.01)
                    continue
                try:
                    pkt = await self._encoder.next_packet()
                    self._metrics.encode_packets.inc()
                    self._metrics.encode_bytes_out.inc(len(pkt.payload))
                    self._metrics.bitrate_kbps.set(self._abr.bitrate_kbps)
                    try:
                        self._pkt_q.put_nowait(pkt)
                    except asyncio.QueueFull:
                        self._metrics.encode_drops.inc()
                        self._pkt_q.get_nowait()
                        self._pkt_q.put_nowait(pkt)
                except ProcessDiedError:
                    log.error("encoder dead in read loop; restarting")
                    await self._restart_encoder()
                except Exception as exc:
                    log.error("encoder read loop error: %s", exc)
                    await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    async def _abr_loop(self) -> None:
        """Drive the ABR controller at the frame rate."""
        interval = self._cfg.frame_ms / 1000.0
        try:
            while not self._stopped:
                await self._abr.tick()
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    async def _restart_encoder(self) -> None:
        if self._encoder:
            try:
                await self._encoder.stop()
            except Exception: # noqa
                pass
        self._encoder = EncoderProcess(self._cfg)
        await self._encoder.start()
        self._metrics.encoder_restarts.inc()
        log.info("output_stream: encoder restarted")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def bitrate_kbps(self) -> int:
        return self._abr.bitrate_kbps

    @property
    def fec_enabled(self) -> bool:
        return self._abr.fec_enabled

    def report_network_stats(
        self, rtt_ms: Optional[float] = None, lost: bool = False
    ) -> None:
        """
        Feed network feedback into the ABR controller.
        Call this whenever your transport layer receives an ACK or NACK.
        """
        if rtt_ms is not None:
            self._stats.record_rtt(rtt_ms)
        self._stats.record_arrival(lost=lost)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — STREAM ROUTER (HORIZONTAL SCALING)
# ═══════════════════════════════════════════════════════════════════════════════

class StreamRouter:
    """
    Manages the full lifecycle of N concurrent codec stream pairs,
    enabling horizontal scaling across sessions.

    Each session gets a dedicated (FFmpegPCMInputStream, FFmpegPCMOutputStream)
    pair, keyed by a session ID.  The router ensures clean allocation and teardown,
    and provides aggregate metrics across all sessions.

    Usage:
        router = StreamRouter(cfg=CodecConfig.for_discord())
        await router.start()

        session_id = "user-abc-123"
        instream, outstream = await router.open(session_id)

        # Use streams ...

        await router.close(session_id)
        await router.stop()
    """

    def __init__(
        self,
        cfg: Optional[CodecConfig] = None,
        max_sessions: int = 1000,
    ) -> None:
        self._cfg          = cfg or CodecConfig()
        self._max_sessions = max_sessions
        self._sessions: Dict[str, Tuple[FFmpegPCMInputStream, FFmpegPCMOutputStream]] = {}
        self._lock         = asyncio.Lock()
        self._started      = False
        self._metrics      = CodecMetrics(namespace="opus_router")

    async def start(self) -> None:
        self._started = True
        log.info("stream_router started (max_sessions=%d)", self._max_sessions)

    async def stop(self) -> None:
        async with self._lock:
            session_ids = list(self._sessions.keys())
        await asyncio.gather(
            *[self.close(sid) for sid in session_ids],
            return_exceptions=True,
        )
        log.info("stream_router stopped, %d sessions closed", len(session_ids))

    async def open(
        self,
        session_id: str,
        cfg: Optional[CodecConfig] = None,
    ) -> Tuple[FFmpegPCMInputStream, FFmpegPCMOutputStream]:
        """
        Allocate and start a (input, output) stream pair for a new session.
        Raises ValueError if session_id already exists or capacity is exceeded.
        """
        async with self._lock:
            if session_id in self._sessions:
                raise ValueError(f"Session {session_id!r} already exists.")
            if len(self._sessions) >= self._max_sessions:
                raise BufferOverflowError(
                    f"Router at capacity ({self._max_sessions} sessions)."
                )

        effective_cfg = cfg or self._cfg
        stats  = NetworkStats()
        instream  = FFmpegPCMInputStream(effective_cfg, stats=stats)
        outstream = FFmpegPCMOutputStream(effective_cfg, stats=stats)

        await instream.start()
        await outstream.start()

        async with self._lock:
            self._sessions[session_id] = (instream, outstream)

        log.info("router: opened session %r (total=%d)", session_id, len(self._sessions))
        return instream, outstream

    async def close(self, session_id: str) -> None:
        """Gracefully stop and remove a session."""
        async with self._lock:
            pair = self._sessions.pop(session_id, None)
        if pair is None:
            return
        instream, outstream = pair
        await asyncio.gather(
            instream.stop(), outstream.stop(), return_exceptions=True
        )
        log.info("router: closed session %r (remaining=%d)", session_id, len(self._sessions))

    def get(
        self, session_id: str
    ) -> Optional[Tuple[FFmpegPCMInputStream, FFmpegPCMOutputStream]]:
        """Return the stream pair for *session_id*, or None if not found."""
        return self._sessions.get(session_id)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    # noinspection PyProtectedMember
    def aggregate_metrics(self) -> dict:
        """Collect and merge metrics from all active sessions."""
        total = {
            "sessions": self.active_sessions,
            "encode_packets": 0.0,
            "decode_packets": 0.0,
            "plc_frames":     0.0,
            "encode_drops":   0.0,
        }
        for instream, outstream in self._sessions.values():
            m = instream._metrics  # type: ignore[attr-defined]
            total["decode_packets"] += m.decode_packets.value
            total["plc_frames"]     += m.decode_plc_frames.value
            mo = outstream._metrics  # type: ignore[attr-defined]
            total["encode_packets"] += mo.encode_packets.value
            total["encode_drops"]   += mo.encode_drops.value
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def check_opus_support() -> bool:
    """
    Verify that the installed FFmpeg binary exposes the libopus encoder
    and decoder.  Returns True on success; logs details on failure.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-codecs"],
            capture_output=True, text=True, timeout=8.0,
        )
        codecs = result.stdout + result.stderr
        enc_ok = "libopus" in codecs
        dec_ok = "libopus" in codecs or "opus" in codecs
        if not enc_ok:
            log.error("check_opus_support: libopus encoder not found in ffmpeg -codecs")
        if not dec_ok:
            log.error("check_opus_support: opus decoder not found in ffmpeg -codecs")
        return enc_ok and dec_ok
    except FileNotFoundError:
        log.error("check_opus_support: ffmpeg not found. %s", FFmpegNotFoundError.INSTALL_HINT)
        return False
    except subprocess.TimeoutExpired:
        log.error("check_opus_support: ffmpeg -codecs timed out")
        return False


def get_opus_bitrate_for_quality(quality: str) -> int:
    """
    Recommend an Opus bitrate for the given quality tier.

    Tiers:
        "narrowband"  — 8 kHz equivalent, telephony  (8 kbps)
        "wideband"    — 16 kHz, acceptable voice      (24 kbps)
        "fullband"    — 48 kHz, clear voice           (64 kbps)
        "high"        — 48 kHz, Discord-like quality  (96 kbps)
        "hd"          — 48 kHz stereo, near-lossless  (192 kbps)
        "studio"      — 48 kHz stereo, transparent    (320 kbps)
    """
    tiers = {
        "narrowband": 8,
        "wideband":   24,
        "fullband":   64,
        "high":       96,
        "hd":         192,
        "studio":     320,
    }
    if quality not in tiers:
        raise ValueError(f"Unknown quality tier {quality!r}; choose from {list(tiers)}")
    return tiers[quality]


def _generate_silence(cfg: CodecConfig, duration_ms: int) -> np.ndarray:
    """Return a silent numpy array for the given duration."""
    n = int(cfg.sample_rate * duration_ms / 1_000)
    return np.zeros(n * cfg.channels, dtype=cfg.numpy_dtype)


async def test_opus_roundtrip(
    cfg: Optional[CodecConfig] = None,
    duration_ms: int = 200,
    snr_threshold_db: float = 10.0,
) -> bool:
    """
    Full async encode → decode roundtrip test.

    Generates a 1 kHz sine wave, encodes it to Opus via FFmpeg,
    decodes back to PCM, and computes SNR.  Returns True if SNR
    exceeds `snr_threshold_db`.

    Typical SNR for Opus at 96 kbps: 28-35 dB.
    """
    if not check_opus_support():
        log.error("test_opus_roundtrip: Opus not available.")
        return False

    effective_cfg = cfg or CodecConfig(sample_rate=48_000, channels=1)
    n = int(effective_cfg.sample_rate * duration_ms / 1_000)
    t = np.arange(n) / effective_cfg.sample_rate
    freq = 1_000.0
    amplitude = 0.5

    if np.issubdtype(effective_cfg.numpy_dtype, np.integer):
        max_val = np.iinfo(effective_cfg.numpy_dtype).max
        original = (np.sin(2 * np.pi * freq * t) * amplitude * max_val).astype(
            effective_cfg.numpy_dtype
        )
    else:
        original = (np.sin(2 * np.pi * freq * t) * amplitude).astype(
            effective_cfg.numpy_dtype
        )

    try:
        encoder = EncoderProcess(effective_cfg)
        await encoder.start()

        decoder = DecoderProcess(effective_cfg)
        await decoder.start()

        # Encode
        frame_size = effective_cfg.frame_bytes
        for i in range(0, len(original.tobytes()), frame_size):
            chunk = original.tobytes()[i:i + frame_size]
            if len(chunk) < frame_size:
                # Pad last frame
                chunk = chunk.ljust(frame_size, b"\x00")
            await encoder.encode(chunk)

        # Collect encoded packets
        opus_packets: List[bytes] = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                pkt = encoder._out_q.get_nowait()
                opus_packets.append(pkt.payload)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                if time.monotonic() > deadline:
                    break

        if not opus_packets:
            log.error("test_opus_roundtrip: no encoded packets received")
            return False

        # Decode
        for pkt_bytes in opus_packets:
            await decoder.push_opus_packet(pkt_bytes)

        decoded_frames: List[np.ndarray] = []
        deadline2 = time.monotonic() + 3.0
        while time.monotonic() < deadline2:
            try:
                frame = decoder._out_q.get_nowait()
                decoded_frames.append(frame.data)
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.02)
                if time.monotonic() > deadline2:
                    break

        await encoder.stop()
        await decoder.stop()

        if not decoded_frames:
            log.error("test_opus_roundtrip: no decoded frames received")
            return False

        decoded = np.concatenate(decoded_frames)
        min_len = min(len(original), len(decoded))
        if min_len < 100:
            log.error("test_opus_roundtrip: decoded too short (%d samples)", min_len)
            return False

        orig_f = original[:min_len].astype(np.float64)
        dec_f  = decoded[:min_len].astype(np.float64)

        signal_power = np.mean(orig_f ** 2)
        noise_power  = np.mean((orig_f - dec_f) ** 2)

        if noise_power < 1e-12:
            snr_db = float("inf")
        else:
            snr_db = 10.0 * math.log10(signal_power / noise_power)

        log.info(
            "test_opus_roundtrip: SNR=%.1f dB, encoded_pkts=%d, decoded_frames=%d",
            snr_db, len(opus_packets), len(decoded_frames),
        )
        return snr_db >= snr_threshold_db

    except Exception as exc:
        log.exception("test_opus_roundtrip failed: %s", exc)
        return False


async def probe_ffmpeg() -> dict:
    """
    Return a dict with FFmpeg version, Opus support, and available codecs.
    Useful for health checks and startup diagnostics.
    """
    result: dict = {
        "ffmpeg_found": False,
        "version": None,
        "libopus_encoder": False,
        "libopus_decoder": False,
        "supported_sample_rates": [],
    }
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        result["ffmpeg_found"] = True
        for line in stdout.decode().splitlines():
            if line.startswith("ffmpeg version"):
                result["version"] = line.split()[2]
                break

        proc2 = await asyncio.create_subprocess_exec(
            "ffmpeg", "-codecs",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout2, _ = await asyncio.wait_for(proc2.communicate(), 5.0)
        codec_text = stdout2.decode()
        result["libopus_encoder"] = "libopus" in codec_text
        result["libopus_decoder"] = "libopus" in codec_text or "opus" in codec_text

        # Quick check of supported sample rates via encoder
        proc3 = await asyncio.create_subprocess_exec(
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
            "-c:a", "libopus", "-t", "0.01", "-f", "null", "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr3 = await asyncio.wait_for(proc3.communicate(), 5.0)
        result["48000_hz_ok"] = proc3.returncode == 0

    except FileNotFoundError:
        result["error"] = "ffmpeg not found"
    except asyncio.TimeoutError:
        result["error"] = "ffmpeg probe timed out"
    except Exception as exc:
        result["error"] = str(exc)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — CONVENIENCE CONTEXT MANAGERS
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def open_input_stream(
    cfg: Optional[CodecConfig] = None,
    **kwargs: Any,
) -> AsyncIterator[FFmpegPCMInputStream]:
    """
    Convenience async context manager for the input stream.

    Example:
        async with open_input_stream(CodecConfig.for_whisper()) as stream:
            stream.push_opus_bytes(opus_data)
            async for frame in stream:
                transcription = await stt.process(frame)
    """
    stream = FFmpegPCMInputStream(cfg, **kwargs)
    await stream.start()
    try:
        yield stream
    finally:
        await stream.stop()


@asynccontextmanager
async def open_output_stream(
    cfg: Optional[CodecConfig] = None,
    **kwargs: Any,
) -> AsyncIterator[FFmpegPCMOutputStream]:
    """
    Convenience async context manager for the output stream.

    Example:
        async with open_output_stream(CodecConfig.for_discord()) as stream:
            await stream.write(pcm_chunk)
            async for pkt in stream:
                await ws.send_bytes(pkt.payload)
    """
    stream = FFmpegPCMOutputStream(cfg, **kwargs)
    await stream.start()
    try:
        yield stream
    finally:
        await stream.stop()


@asynccontextmanager
async def open_full_duplex(
    cfg: Optional[CodecConfig] = None,
) -> AsyncIterator[Tuple[FFmpegPCMInputStream, FFmpegPCMOutputStream]]:
    """
    Open a matched input + output stream pair sharing the same NetworkStats,
    so ABR and jitter adaptation react to conditions seen in both directions.

    Example:
        async with open_full_duplex(CodecConfig.for_discord()) as (rx, tx):
            # Receive & decode
            rx.push_opus_bytes(incoming_opus)
            async for frame in rx:
                response = await ai_pipeline(frame)
                await tx.write(response)

            # Send & encode
            async for pkt in tx:
                await ws.send_bytes(pkt.payload)
    """
    shared_stats = NetworkStats()
    effective_cfg = cfg or CodecConfig()
    rx = FFmpegPCMInputStream(effective_cfg, stats=shared_stats)
    tx = FFmpegPCMOutputStream(effective_cfg, stats=shared_stats)
    await rx.start()
    await tx.start()
    try:
        yield rx, tx
    finally:
        await asyncio.gather(rx.stop(), tx.stop(), return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — INTEGRATION WITH audio_engine (BACKWARD COMPATIBILITY SHIM)
# ═══════════════════════════════════════════════════════════════════════════════

class _AudioEngineShim:
    """
    Provides the attribute names expected by audio_engine.py so this module
    is a drop-in replacement.  Wraps FFmpegPCMInputStream / FFmpegPCMOutputStream
    behind the old API surface.

    audio_engine expects:
        stream.start() / stream.stop()
        stream.push_opus_bytes(bytes)  ← input only
        async for chunk in stream:     ← input only
        await stream.write(chunk)      ← output only
        stream.read_opus_frame()       ← output only (sync, non-blocking)
    All are provided.
    """

    def __init__(
        self,
        stream: Union[FFmpegPCMInputStream, FFmpegPCMOutputStream],
    ) -> None:
        self._inner = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def read_opus_frame(self) -> Optional[bytes]:
        """
        Synchronous non-blocking read of one Opus packet payload.
        Compatible with the original API.
        """
        if not isinstance(self._inner, FFmpegPCMOutputStream):
            raise TypeError("read_opus_frame() only available on output streams")
        pkt = self._inner.read_opus_packet_nowait()
        return pkt.payload if (pkt and pkt.payload) else None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19 — PUBLIC API SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Config
    "CodecConfig",
    # Data types
    "OpusPacket",
    "PCMFrame",
    "NetworkStats",
    # Core encode/decode
    "EncoderProcess",
    "DecoderProcess",
    # Pools
    "EncoderPool",
    "DecoderPool",
    # High-level streams
    "FFmpegPCMInputStream",
    "FFmpegPCMOutputStream",
    # Scaling
    "StreamRouter",
    # Context managers
    "open_input_stream",
    "open_output_stream",
    "open_full_duplex",
    # Adaptive components (for custom pipelines)
    "AdaptiveBitrateController",
    "AdaptiveJitterBuffer",
    "PacketLossConcealer",
    # OGG tools
    "OggPageParser",
    "OggWriter",
    # Observability
    "CodecMetrics",
    # Exceptions
    "OpusError",
    "FFmpegNotFoundError",
    "ProcessDiedError",
    "CircuitOpenError",
    # Utils
    "check_opus_support",
    "get_opus_bitrate_for_quality",
    "test_opus_roundtrip",
    "probe_ffmpeg",
    # Shim
    "_AudioEngineShim",
]