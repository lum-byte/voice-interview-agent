"""
Microphone recording — PTT-safe, audio_engine integrated.

Public API
──────────
  record_audio_until_released(is_held_fn) → str | None
      Record while is_held_fn() returns True.
      Returns the temp-file path on success, None on failure or silence.

  record_audio_until_released_async(is_held_fn) → str | None
      Async version for callers already inside an event loop.

  delete_temp_recording(path) → None
      Explicitly unlink a temp file after STT has consumed it.
      Also available as a context manager via temp_recording().

  get_recording_health()         → AudioHealthReport
      Point-in-time health report from the module-level diagnostics monitor.
      Reflects clipping, silence rate, DC offset, and dropout counts
      accumulated across all sessions in this process lifetime.

  get_recording_latency_report() → dict[str, dict[str, float]]
      Per-stage latency from the module-level PCMLatencyTracker.
      Stages: "capture_start", "capture", "frames_collected",
              "vad_pass", "enhancement", "wav_encode".

  get_recording_format()         → PCMFormat
      The active recording PCMFormat (derived from env vars at import time).

  run_startup_health_check()     → dict
      One-shot async mic health check via audio_engine's check_audio_health().


Audio-engine integration
────────────────────────
  PCMFormat            — Single format authority replacing raw SAMPLE_RATE /
                         CHANNELS / dtype ints. All sd.InputStream construction,
                         WAV encoding, and processor initialisation derive
                         parameters from this immutable descriptor.

  PCMInputStream       — Async mic capture replacing the sd.InputStream +
                         callback + threading.Event pattern. Owns the PortAudio
                         device, bridges PortAudio callbacks → asyncio.Queue,
                         and exposes a clean async-for iterator of PCMChunks.

  PCMChunk             — Typed, timestamped audio payload. All captured audio
                         lives as PCMChunks so every audio_engine processor
                         (VAD, enhancer, diagnostics) can consume them directly
                         without format negotiation or manual numpy casts.

  PCMVADGate           — Lightweight energy-based silence rejection gate. Runs
                         on raw int16 captured chunks as a fast-path check
                         before the heavy PCMSpeechEnhancer pipeline. If the
                         gate emits no speech segments, the recording is
                         rejected immediately — no DSP cost, no STT billing.

  PCMRingBuffer        — Rolling frame accumulation buffer. Raw sample data from
                         every incoming PCMChunk is written here during capture.
                         ring.available_to_read() gives the total accumulated
                         frame count in O(1) for duration validation without any
                         data copies. At session end, ring.read() produces a
                         contiguous merged numpy array in one O(n) call,
                         replacing np.concatenate() and its heap allocation.

  PCMConverter         — Format converter. Used in the enhancement-disabled
                         fallback path to normalise captured int16 chunks to
                         float32 before WAV encoding, ensuring a consistent
                         output dtype regardless of enhancement state. Uses the
                         module-level get_converter() singleton to avoid
                         per-session object construction.

  PCMSpeechEnhancer    — Full pre-STT enhancement pipeline:
                         bandpass → noise suppressor → AGC → noise gate → VAD.
                         Runs on collected chunks as a batch async generator
                         after PCMVADGate confirms speech is present. Yields
                         only confirmed speech segments ready for transcription.

  PCMDiagnosticsMonitor — Module-level singleton (get_diagnostics()) fed every
                          captured chunk in real-time. Tracks clipping rate,
                          sustained silence, DC offset, and sequence gaps.
                          Results surface via get_recording_health() and
                          Prometheus counters.

  check_audio_health() — One-shot async mic health check function from
                         audio_engine. Exposed via run_startup_health_check()
                         for pre-session device validation.

  PCMLatencyTracker    — Module-level singleton (get_latency_tracker()) with
                         observation points at every major stage boundary.
                         Feeds pcm_pipeline_stage_latency Prometheus histogram.
                         Results surface via get_recording_latency_report().

  get_chunk_pool()     — Module-level numpy array pool. Used to acquire the
                         merged output buffer sized to all enhanced frames
                         without a heap allocation. Released after WAV bytes
                         are written to disk.


Design decisions
────────────────
  Sync public API / async core
      record_audio_until_released() preserves the original synchronous contract
      for backward compatibility. Internally it delegates to the async core via
      asyncio.run(). When called from inside an already-running event loop,
      falls back to a dedicated ThreadPoolExecutor thread with its own loop to
      avoid the nested-loop RuntimeError. Async callers use the _async variant.

  Two-stage silence rejection
      Stage 1 — PCMVADGate (fast): energy-based pass on raw int16 chunks.
                 Rejects before any DSP runs.
      Stage 2 — PCMSpeechEnhancer VAD (precise): spectral + energy fusion VAD
                 inside the full enhancement pipeline. Rejects if clean speech
                 is absent even after noise suppression.

  PCMRingBuffer role
      The ring buffer accumulates raw sample frames during recording in O(1)
      per chunk. available_to_read() provides frame-count validation without
      any data access. ring.read() at session end produces one contiguous
      merged numpy array, replacing np.concatenate() on the collected list.

  Safety caps
      MAX_DURATION_S:  PCMInputStream is stopped if elapsed > cap regardless of
                       PTT state. Prevents runaway sessions from filling disk.
      MIN_DURATION_S:  Checked against ring.available_to_read() before any DSP.
      Temp file guard: delete_temp_recording() refuses paths outside TEMP_DIR.

  Lazy init
      Cleanup thread and PCMSpeechEnhancer initialise on first use; importing
      this module has zero side-effects in unit-test environments.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import os
import threading
import time
import uuid
from collections.abc import Generator, Callable
from pathlib import Path

import numpy as np
import sounddevice as sd

from app.common.shared import make_counter, make_histogram
from app.monitoring.observability import get_logger

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every symbol imported here has an explicit documented role above and below.
# Ordering mirrors audio_engine's own section layout:
#   Primitives → DSP blocks → Streaming I/O → Processing → Diagnostics →
#   Utilities → Singletons
# ═══════════════════════════════════════════════════════════════════════════════
from app.audio_essentials.audio_engine import (

    # ── Core primitives ───────────────────────────────────────────────────────
    PCMFormat,              # Immutable format descriptor; single source of truth
    PCMChunk,               # Typed timestamped audio payload

    # ── Low-level DSP building blocks ─────────────────────────────────────────
    PCMRingBuffer,          # O(1) rolling frame accumulation + contiguous read
    PCMConverter,           # int16→float32 in the enhancement-disabled path # noqa

    # ── Streaming I/O ─────────────────────────────────────────────────────────
    PCMInputStream,         # Async mic capture replacing sd.InputStream+callback

    # ── Processing units ──────────────────────────────────────────────────────
    PCMVADGate,             # Fast energy-based early silence rejection gate
    PCMSpeechEnhancer,      # Full pipeline: bandpass + NS + AGC + gate + fine VAD

    # ── Diagnostics & observability ───────────────────────────────────────────
    PCMDiagnosticsMonitor,  # Real-time mic health: clipping, silence, DC, gaps
    PCMLatencyTracker,      # Per-stage wall-clock latency tracking  # noqa
    AudioHealthReport,      # Structured health report dataclass

    # ── Utility functions ─────────────────────────────────────────────────────
    chunk_to_wav_bytes,     # PCMChunk → complete WAV bytes (replaces _write_wav)
    check_audio_health,     # One-shot async mic health check

    # ── Module-level singletons ───────────────────────────────────────────────
    get_chunk_pool,         # Zero-malloc numpy array pool for the merge step
    get_converter,          # Shared PCMConverter(quality="auto") singleton
    get_latency_tracker,    # Shared PCMLatencyTracker singleton
    get_diagnostics,        # Shared PCMDiagnosticsMonitor factory / singleton
)

# ── Logging ───────────────────────────────────────────────────────────────────

log = get_logger(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

_recordings_started = make_counter(
    "recorder_recordings_total", "Recording sessions started"
)
_recordings_saved = make_counter(
    "recorder_recordings_saved_total", "Recordings saved to disk"
)
_recordings_empty = make_counter(
    "recorder_recordings_empty_total", "Recordings rejected: no audio frames captured"
)
_recordings_too_short = make_counter(
    "recorder_recordings_too_short_total", "Recordings rejected: below min duration"
)
_recordings_silent = make_counter(
    "recorder_recordings_silent_total",
    "Recordings rejected: no speech detected by VAD",
)
_recordings_maxed = make_counter(
    "recorder_recordings_maxed_total", "Recordings truncated at max duration cap"
)
_recordings_errors = make_counter(
    "recorder_errors_total", "Unhandled errors during recording"
)

_recording_duration = make_histogram(
    "recorder_duration_seconds",
    "PTT hold duration for accepted recordings",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)
_recording_rms = make_histogram(
    "recorder_rms",
    "RMS level of accepted recordings (normalised float32 scale 0–1)",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.6, 1.0),
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ── Format authority (PCMFormat replaces raw SAMPLE_RATE / CHANNELS ints) ─────
#
# _RECORDING_FMT is the single source of truth for every audio parameter in this
# module. PCMInputStream construction, PCMVADGate sizing, PCMSpeechEnhancer
# initialisation, PCMRingBuffer capacity, and chunk_to_wav_bytes encoding all
# derive from this one frozen object. No raw SAMPLE_RATE / CHANNELS ints appear
# anywhere else in the file.
#
# Env vars are still honoured for backward compatibility; values are funnelled
# through PCMFormat rather than scattered as bare ints across functions.

SAMPLE_RATE: int = int(os.getenv("RECORDER_SAMPLE_RATE", "16000"))
CHANNELS: int   = int(os.getenv("RECORDER_CHANNELS", "1"))

# 16 kHz mono int16 — Whisper-optimised. PortAudio captures int16 natively;
# processors that need float32 handle the conversion via PCMConverter internally.
_RECORDING_FMT: PCMFormat = PCMFormat(
    sample_rate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
)

# ── Safety caps ───────────────────────────────────────────────────────────────

MAX_DURATION_S: float = float(os.getenv("RECORDER_MAX_DURATION_S", "60.0"))
MIN_DURATION_S: float = float(os.getenv("RECORDER_MIN_DURATION_S", "0.3"))

# ── Enhancement config ────────────────────────────────────────────────────────
#
# When enabled (default), PCMSpeechEnhancer runs the full DSP chain before WAV
# encoding. Disable for lowest latency or when the STT model is fine-tuned on
# noisy data and handles raw input directly.

_ENABLE_ENHANCEMENT: bool = os.getenv("RECORDER_ENABLE_ENHANCEMENT", "1") == "1"

# VAD backend inside PCMSpeechEnhancer.
#   "energy"   — fastest, pure numpy RMS (suitable for quiet office environments)
#   "spectral" — adds frequency-domain voice-band analysis (better with HVAC noise)
#   "webrtc"   — Google WebRTC VAD (requires webrtcvad-wheels, very robust)
#   "fused"    — majority-vote ensemble of energy + spectral (recommended default)
_VAD_BACKEND: str = os.getenv("RECORDER_VAD_BACKEND", "energy")

# ── Ring buffer capacity ──────────────────────────────────────────────────────
#
# Sized for MAX_DURATION_S + 1 s headroom. PCMRingBuffer rounds capacity up to
# the next power-of-2, so the actual allocation is slightly larger. This is
# computed once from _RECORDING_FMT so changing the env-driven sample rate is
# sufficient to resize the buffer.

_RING_CAPACITY_FRAMES: int = _RECORDING_FMT.frames_for_duration(MAX_DURATION_S + 1.0)

# ── Temp file management ──────────────────────────────────────────────────────

BASE_DIR: Path    = Path(os.getenv("RECORDER_BASE_DIR", "audio"))
TEMP_DIR: Path    = BASE_DIR / "temp"
TEMP_MAX_AGE_S: float = float(
    os.getenv("RECORDER_TEMP_MAX_AGE_S", str(60 * 60 * 24 * 3))
)  # 3 days

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Shared across all recording sessions in the process lifetime.
# Sharing lets the diagnostics monitor and latency tracker accumulate data
# across sessions for richer Prometheus / health reporting.

# PCMDiagnosticsMonitor: receives every captured chunk via push(). Tracks
# clipping rate, sustained silence, DC offset, and sequence-number gaps.
# Fired callbacks and Prometheus counters update in real-time during recording.
_module_diagnostics: PCMDiagnosticsMonitor = get_diagnostics(fmt=_RECORDING_FMT)

# PCMSpeechEnhancer: lazily initialised on first use.
# Protected by a lock so concurrent first-calls don't create two instances.
_speech_enhancer: PCMSpeechEnhancer | None = None
_speech_enhancer_lock = threading.Lock()


def _get_speech_enhancer() -> PCMSpeechEnhancer:
    """
    Lazily initialise and return the module-level PCMSpeechEnhancer singleton.

    Thread-safe via double-checked locking.

    The enhancer is stateful (VAD, AGC), but because each recording session
    calls enhancer.stream() independently, each call returns a fresh async
    generator with its own local state. Sessions do not share VAD or AGC
    state — the generator captures its own closures on creation.
    """
    global _speech_enhancer
    if _speech_enhancer is None:
        with _speech_enhancer_lock:
            if _speech_enhancer is None:
                _speech_enhancer = PCMSpeechEnhancer(
                    fmt=_RECORDING_FMT,
                    enable_bandpass=True,
                    enable_ns=True,
                    enable_agc=True,
                    enable_gate=True,
                    vad_backend=_VAD_BACKEND,   # type: ignore[arg-type]
                    tracker=get_latency_tracker(),
                )
                log.info(
                    "recorder_speech_enhancer_init",
                    fmt=repr(_RECORDING_FMT),
                    vad_backend=_VAD_BACKEND,
                )
    return _speech_enhancer


# ═══════════════════════════════════════════════════════════════════════════════
# BACKGROUND CLEANUP THREAD
# ═══════════════════════════════════════════════════════════════════════════════

_cleanup_started: bool        = False
_cleanup_lock: threading.Lock = threading.Lock()


def _ensure_cleanup_thread() -> None:
    """
    Start the temp-dir sweep thread on the first recording.

    Lazy init prevents module-import side-effects in unit tests that merely
    import the module without ever recording anything.
    """
    global _cleanup_started
    with _cleanup_lock:
        if _cleanup_started:
            return
        _cleanup_started = True

    thread = threading.Thread(
        target=_cleanup_loop, daemon=True, name="recorder-cleanup"
    )
    thread.start()
    log.info("recorder_cleanup_thread_started", temp_dir=str(TEMP_DIR))


def _cleanup_loop() -> None:
    """Sweep TEMP_DIR once per hour, removing files older than TEMP_MAX_AGE_S."""
    while True:
        _sweep_temp_dir()
        time.sleep(3600)


def _sweep_temp_dir() -> None:
    """Delete stale temp WAV files. Errors on individual files are logged and skipped."""
    if not TEMP_DIR.exists():
        return
    now = time.time()
    removed = 0
    for wav in TEMP_DIR.glob("rec_*.wav"):
        try:
            age = now - wav.stat().st_mtime
            if age > TEMP_MAX_AGE_S:
                wav.unlink()
                removed += 1
        except Exception as exc:
            log.warning("recorder_cleanup_error", file=str(wav), error=str(exc))
    if removed:
        log.info("recorder_cleanup_done", files_removed=removed)


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC RECORDING CORE
# ═══════════════════════════════════════════════════════════════════════════════


async def _record_pcm_async(is_held_fn: Callable[[], bool]) -> str | None:
    """
    Async core of the PTT recording pipeline.

    Called by the synchronous public API via asyncio.run() (or from a thread
    pool when a loop is already running). Async callers invoke this directly
    through record_audio_until_released_async().

    Pipeline stages
    ───────────────
    1.  PCMInputStream        → async mic capture; PTT status polled per-chunk
    2.  PCMRingBuffer         → O(1) raw frame accumulation + frame-count query
    3.  PCMDiagnosticsMonitor → per-chunk health: clipping / silence / DC offset
    4.  PCMLatencyTracker     → stage boundary observations throughout
    5.  Duration validation   → fast-path rejection before any DSP runs
    6.  PCMVADGate            → energy-based early rejection on raw int16 chunks
    7a. PCMSpeechEnhancer     → full pipeline (if RECORDER_ENABLE_ENHANCEMENT=1)
    7b. PCMConverter          → int16→float32 fallback (if enhancement disabled)
    8.  get_chunk_pool()      → acquire merged output buffer without heap alloc
    9.  chunk_to_wav_bytes()  → encode enhanced PCMChunk → WAV bytes
    10. filepath.write_bytes  → atomic single-call write to TEMP_DIR
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_cleanup_thread()

    fmt     = _RECORDING_FMT       # PCMFormat: single format authority
    tracker = get_latency_tracker() # shared latency tracker
    pool    = get_chunk_pool()      # shared numpy array pool

    t_start = time.monotonic()
    _recordings_started.inc()
    log.info("recorder_start", fmt=repr(fmt))

    # ── PCMRingBuffer: O(1) frame accumulation ─────────────────────────────────
    #
    # Raw sample data from every incoming PCMChunk is written here during
    # capture. Writing is O(1) per chunk — no list reallocs, no intermediate
    # concatenations. ring.available_to_read() gives the total frame count in
    # O(1) for duration validation without touching any sample data. ring.read()
    # at session end produces one contiguous merged numpy array.
    ring = PCMRingBuffer(capacity=_RING_CAPACITY_FRAMES, fmt=fmt)

    # Parallel list of full PCMChunks for the async processor pipeline.
    # PCMVADGate and PCMSpeechEnhancer consume PCMChunk iterators — they need
    # typed payloads with timestamp / seq metadata, not just raw numpy arrays.
    collected_chunks: list[PCMChunk] = []

    # ── Stage 1: PCMInputStream — async mic capture ────────────────────────────
    #
    # PCMInputStream owns the PortAudio device for the duration of the context.
    # It bridges PortAudio's high-priority callback thread → asyncio.Queue via
    # call_soon_threadsafe, then exposes a clean async-for iterator of PCMChunks.
    # This replaces the original sd.InputStream + callback + threading.Event
    # + frames.append(indata.copy()) pattern entirely.
    try:
        async with PCMInputStream(fmt=fmt) as stream:

            # Observe the moment the stream opens so the latency tracker can
            # measure stream-open → first-chunk arrival. A sentinel chunk
            # carries the start timestamp without containing any sample data.
            _sentinel = PCMChunk(
                data=np.array([], dtype=fmt.dtype),
                fmt=fmt,
                timestamp=t_start,
                seq=-1,
                source="recorder_sentinel",
            )
            tracker.observe(_sentinel, "capture_start")

            async for chunk in stream:

                # ── PTT gate ───────────────────────────────────────────────────
                # Polled once per chunk cadence (~60 ms at 16 kHz / 960 blocksize).
                # No explicit sd.sleep() needed — the chunk arrival IS the tick.
                if not is_held_fn():
                    break

                # ── Max duration cap ───────────────────────────────────────────
                elapsed = time.monotonic() - t_start
                if elapsed >= MAX_DURATION_S:
                    log.warning(
                        "recorder_max_duration_reached",
                        max_s=MAX_DURATION_S,
                        elapsed_s=round(elapsed, 2),
                    )
                    _recordings_maxed.inc()
                    break

                # ── PCMDiagnosticsMonitor: real-time health ────────────────────
                # push() is O(1). Fires clipping / silence / DC offset callbacks
                # and increments Prometheus counters in-line. Health status is
                # available via get_recording_health() at any time.
                _module_diagnostics.push(chunk)

                # ── PCMLatencyTracker: per-chunk observation ───────────────────
                # Measures wall-clock delta from chunk.timestamp (set at capture
                # inside PCMInputStream._drain_thread_queue) to now.
                tracker.observe(chunk, "capture")

                # ── PCMRingBuffer: write raw sample data ───────────────────────
                # Mono slice for multi-channel formats; single channel arrays
                # pass through unchanged. The ring buffer stores raw frames for
                # frame-count queries and the final merged-array extraction.
                raw_mono = chunk.data[:, 0] if chunk.data.ndim == 2 else chunk.data
                ring.write(raw_mono)

                # Keep the full PCMChunk for the processor pipeline.
                collected_chunks.append(chunk)

    except sd.PortAudioError as exc:
        log.error("recorder_device_error", error=str(exc))
        _recordings_errors.inc()
        return None
    except Exception as exc:
        log.error("recorder_unexpected_error", error=str(exc))
        _recordings_errors.inc()
        return None

    # ── Validate: frames captured ──────────────────────────────────────────────
    if not collected_chunks:
        log.warning("recorder_no_frames", duration_s=round(time.monotonic() - t_start, 3))
        _recordings_empty.inc()
        return None

    # ── Validate: minimum duration ─────────────────────────────────────────────
    #
    # ring.available_to_read() is O(1) — no data access — unlike len(frames) on
    # the original list (which also required np.concatenate to get total frames).
    total_frames       = ring.available_to_read()
    actual_duration_s  = total_frames / fmt.sample_rate

    if actual_duration_s < MIN_DURATION_S:
        log.warning(
            "recorder_too_short",
            actual_s=round(actual_duration_s, 3),
            min_s=MIN_DURATION_S,
        )
        _recordings_too_short.inc()
        return None

    tracker.observe(collected_chunks[-1], "frames_collected")

    # ── Stage 2: PCMVADGate — fast energy-based early rejection ───────────────
    #
    # Before running the expensive PCMSpeechEnhancer pipeline (bandpass filter,
    # noise suppressor, AGC are all O(n) DSP), run a lightweight energy-based
    # VAD pass on the raw int16 collected chunks.
    #
    # PCMVADGate.stream() consumes the chunk iterator and yields speech segments.
    # If it emits nothing after the full iterator is exhausted (including its
    # internal flush on StopAsyncIteration), the recording is pure silence and
    # we reject without touching the enhancement pipeline at all.
    #
    # This avoids STT billing on muted-mic / accidental PTT presses and avoids
    # the 5–20 ms DSP overhead on silent recordings.
    vad_gate = PCMVADGate(fmt=fmt)
    quick_speech_found = False

    async def _raw_source():
        """Finite async iterator yielding all collected raw int16 chunks."""
        for c in collected_chunks:
            yield c

    # PCMVADGate.stream() flushes any in-progress speech segment when its source
    # iterator is exhausted, so a single burst entirely within a hangover window
    # is still caught. One confirmed segment is enough to pass this gate.
    async for _ in vad_gate.stream(_raw_source()):
        quick_speech_found = True
        break   # No need to drain the full iterator — one segment proves speech

    if not quick_speech_found:
        log.warning(
            "recorder_silent_quick_vad",
            duration_s=round(actual_duration_s, 3),
        )
        _recordings_silent.inc()
        return None

    tracker.observe(collected_chunks[-1], "vad_pass")

    # ── Stage 3: Enhancement / conversion pipeline ─────────────────────────────

    enhanced_chunks: list[PCMChunk] = []

    async def _collected_source():
        """Finite async iterator yielding all collected PCMChunks."""
        for c in collected_chunks:
            yield c

    if _ENABLE_ENHANCEMENT:
        # ── Path A: PCMSpeechEnhancer — full DSP pipeline ─────────────────────
        #
        # Runs the complete pre-STT chain in sequence:
        #   PCMBandpassFilter  → isolate voice-band frequencies (80–8000 Hz)
        #   PCMNoiseSuppressor → spectral subtraction noise floor removal
        #   PCMAGCProcessor    → normalise input level for Whisper
        #   PCMNoiseGate       → smooth gate on room ambience in pauses
        #   VAD (fine)         → energy + spectral fusion; yields speech segments
        #
        # The fine VAD inside PCMSpeechEnhancer operates on cleaned-up audio,
        # so it may catch speech that the quick PCMVADGate missed under noise, or
        # trim down to cleaner speech windows. Only confirmed is_final=True speech
        # segments are appended; inter-speech silence is discarded.
        enhancer = _get_speech_enhancer()
        async for speech_chunk in enhancer.stream(_collected_source()):
            enhanced_chunks.append(speech_chunk)

    else:
        # ── Path B: PCMConverter — int16 → float32 normalisation ──────────────
        #
        # Enhancement disabled; bypass all DSP. Use the module-level
        # get_converter() singleton (PCMConverter(quality="auto")) to normalise
        # raw int16 chunks to float32.
        #
        # This keeps the output dtype consistent with the enhancement path so
        # downstream consumers (STT node, debug player) see uniform float32
        # PCMChunks regardless of which path ran.
        converter = get_converter()
        float_fmt = PCMFormat(
            sample_rate=fmt.sample_rate,
            channels=fmt.channels,
            dtype="float32",
        )
        for raw_chunk in collected_chunks:
            enhanced_chunks.append(converter.convert(raw_chunk, float_fmt))

    if not enhanced_chunks:
        # PCMSpeechEnhancer's fine VAD found no clean speech segments even
        # after noise suppression — recording is too noisy or truly silent.
        log.warning(
            "recorder_silent_enhancer_vad",
            duration_s=round(actual_duration_s, 3),
        )
        _recordings_silent.inc()
        return None

    tracker.observe(enhanced_chunks[-1], "enhancement")

    # ── Stage 4: Merge enhanced chunks via get_chunk_pool() ───────────────────
    #
    # Acquire a pre-zeroed numpy array from the module-level pool instead of
    # calling np.concatenate(), which allocates on the heap and then copies all
    # segments. For a 10-second recording at 16 kHz float32, this avoids a
    # ~640 KB heap allocation on every successful recording session.
    #
    # The acquired buffer is released back to the pool after WAV bytes are written
    # to disk so future sessions can reuse the allocation.
    out_fmt         = enhanced_chunks[0].fmt
    enhanced_frames = sum(c.n_frames for c in enhanced_chunks)

    merged_buf = pool.acquire(
        n_frames=enhanced_frames,
        dtype=out_fmt.dtype,
        channels=out_fmt.channels,
    )

    # Copy each enhanced segment's sample data into the contiguous merged buffer.
    write_offset = 0
    for ec in enhanced_chunks:
        n = ec.n_frames
        merged_buf[write_offset : write_offset + n] = ec.data
        write_offset += n

    merged_chunk = PCMChunk(
        data=merged_buf,
        fmt=out_fmt,
        timestamp=collected_chunks[0].timestamp,  # preserve original capture time
        seq=0,
        is_final=True,
        source="recorder",
    )

    # ── Stage 5: WAV encoding — chunk_to_wav_bytes replaces _write_wav ────────
    #
    # chunk_to_wav_bytes() handles dtype normalisation internally (float32 chunks
    # are converted to int16 before RIFF encoding), so the output is always a
    # standard 16-bit PCM WAV consumable by any STT node regardless of whether
    # the enhancement path ran.
    filename = f"rec_{uuid.uuid4().hex[:8]}.wav"
    filepath  = TEMP_DIR / filename

    try:
        wav_bytes = chunk_to_wav_bytes(merged_chunk)
        filepath.write_bytes(wav_bytes)   # single atomic write — no partial-file risk
        tracker.observe(merged_chunk, "wav_encode")
    except Exception as exc:
        log.error("recorder_write_error", path=str(filepath), error=str(exc))
        _recordings_errors.inc()
        pool.release(merged_buf)           # always return to pool on error
        return None

    # Release the merged buffer back to the pool. The WAV bytes are written to
    # disk and the merged_chunk is no longer needed.
    pool.release(merged_buf)

    # ── Metrics ───────────────────────────────────────────────────────────────
    _recordings_saved.inc()
    _recording_duration.observe(actual_duration_s)

    # .rms() normalises to [0, 1] for float32 chunks; used directly for the
    # histogram which is now keyed on float32 scale (not legacy int16 scale).
    rms_float = merged_chunk.rms()
    _recording_rms.observe(rms_float)

    log.info(
        "recorder_saved",
        path=str(filepath),
        duration_s=round(actual_duration_s, 3),
        rms=round(rms_float, 4),
        enhanced=_ENABLE_ENHANCEMENT,
        raw_chunks=len(collected_chunks),
        enhanced_segments=len(enhanced_chunks),
    )

    return str(filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════


def record_audio_until_released(is_held_fn: Callable[[], bool]) -> str | None:
    """
    Record microphone input for as long as ``is_held_fn()`` returns True.

    Blocks the calling thread until PTT is released (or max-duration is hit),
    then validates, optionally enhances via PCMSpeechEnhancer, and saves the
    recording to a temp WAV file.

    Internally delegates to _record_pcm_async() via asyncio.run(). When called
    from inside an already-running event loop (FastAPI, Jupyter, uvicorn, etc.),
    falls back to a dedicated ThreadPoolExecutor thread with its own event loop
    to avoid the nested-loop RuntimeError.

    Async callers should prefer record_audio_until_released_async() to avoid
    the thread-pool overhead.

    Args:
        is_held_fn: Zero-arg callable returning True while PTT is held.
                    Polled once per PCMInputStream chunk cadence (~60 ms at 16 kHz).

    Returns:
        Absolute path string of the saved WAV file, or None if:
          • No audio frames were captured (device error or zero-length press)
          • Recording shorter than MIN_DURATION_S
          • PCMVADGate detected no speech in raw audio
          • PCMSpeechEnhancer VAD detected no clean speech after enhancement
          • An unexpected error occurred
    """
    try:
        # Fast path: no running loop in this thread — asyncio.run() is clean.
        return asyncio.run(_record_pcm_async(is_held_fn))
    except RuntimeError as exc:
        # asyncio.run() raises RuntimeError("This event loop is already running")
        # when called from within an async framework. Spin up a private thread
        # with its own loop so we never nest within the caller's loop.
        if "running event loop" in str(exc):
            log.debug(
                "recorder_event_loop_fallback",
                reason="asyncio.run() called from within a running loop — "
                       "delegating to a dedicated thread",
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="recorder-loop"
            ) as executor:
                future = executor.submit(
                    lambda: asyncio.run(_record_pcm_async(is_held_fn))
                )
                return future.result()
        raise


async def record_audio_until_released_async(
    is_held_fn: Callable[[], bool],
) -> str | None:
    """
    Async version of record_audio_until_released for callers already inside
    an asyncio event loop.

    Identical behaviour to the sync version; no thread-pool overhead.

    Args:
        is_held_fn: Zero-arg callable returning True while PTT is held.

    Returns:
        Same as record_audio_until_released().
    """
    return await _record_pcm_async(is_held_fn)


def delete_temp_recording(path: str) -> None:
    """
    Delete a temp recording after it has been consumed by the STT node.

    Idempotent — safe to call on a path already deleted or never created.
    Logs a warning on unexpected errors but never raises.

    Args:
        path: The string path returned by record_audio_until_released().
              Paths outside TEMP_DIR are refused to prevent accidental
              deletion of non-temp files.
    """
    if not path:
        return
    try:
        p = Path(path).resolve()
        temp_resolved = TEMP_DIR.resolve()
        if not str(p).startswith(str(temp_resolved)):
            log.error(
                "recorder_delete_refused_outside_tempdir",
                path=path,
                temp_dir=str(temp_resolved),
            )
            return
        if p.exists():
            p.unlink()
            log.debug("recorder_temp_deleted", path=path)
    except Exception as exc:
        log.warning("recorder_delete_error", path=path, error=str(exc))


@contextlib.contextmanager
def temp_recording(
    is_held_fn: Callable[[], bool],
) -> Generator[str | None, None, None]:
    """
    Context manager wrapper around record_audio_until_released.

    Records while is_held_fn() is True, yields the temp path to the caller,
    then deletes the temp file on exit regardless of whether the caller raised.

    Usage::

        with temp_recording(lambda: keyboard.is_pressed("h")) as path:
            if path:
                transcript = await stt.transcribe(path)
    """
    path = record_audio_until_released(is_held_fn)
    try:
        yield path
    finally:
        if path:
            delete_temp_recording(path)


def get_recording_health() -> AudioHealthReport:
    """
    Return a point-in-time health report from the module-level
    PCMDiagnosticsMonitor.

    Reflects clipping rate, silence rate, DC offset, and sequence-gap dropout
    counts accumulated across ALL recording sessions since process start.
    Use at any time to check mic hardware status without requiring a dedicated
    recording session.

    Returns:
        AudioHealthReport(status, clipping_rate, silence_rate, dc_offset_mean,
                          dropout_count, rms_mean, notes).
        status is "ok" | "degraded" | "failed".
    """
    return _module_diagnostics.get_health_report()


def get_recording_latency_report() -> dict[str, dict[str, float]]:
    """
    Return per-stage latency statistics from the module-level PCMLatencyTracker.

    Observation stages:
        "capture_start"    — stream-open to first chunk arrival
        "capture"          — per-chunk arrival latency (one observation per chunk)
        "frames_collected" — last chunk to end of collection loop
        "vad_pass"         — frames_collected to PCMVADGate completion
        "enhancement"      — vad_pass to PCMSpeechEnhancer completion
        "wav_encode"       — enhancement to WAV file written to disk

    Returns:
        dict[stage_name, dict[mean_s, p50_s, p95_s, p99_s, max_s, count]]
    """
    return get_latency_tracker().get_latency_report()


def get_recording_format() -> PCMFormat:
    """
    Return the active recording PCMFormat.

    Derived from RECORDER_SAMPLE_RATE / RECORDER_CHANNELS env vars at module
    import time. Immutable for the process lifetime — all pipeline components
    are configured from this object.
    """
    return _RECORDING_FMT


async def run_startup_health_check(
    duration_s: float = 2.0,
    device: int | str | None = None,
) -> dict:
    """
    One-shot microphone health check via audio_engine's check_audio_health().

    Opens the mic for ``duration_s`` seconds, captures audio through
    PCMInputStream + PCMDiagnosticsMonitor, and returns a structured diagnostic
    report. Call this at application startup before the first PTT session to
    validate the recording device is present and functioning.

    Args:
        duration_s: Capture duration for health analysis. Default 2 s.
        device:     PortAudio device index or name. None = system default.

    Returns:
        dict with keys: status ("ok"|"degraded"|"failed"), rms_mean, rms_peak,
        clipping_ratio, silence_ratio, dropout_count, duration_s, error.

    Usage::

        report = await run_startup_health_check()
        if report["status"] != "ok":
            log.warning("recorder_mic_degraded", **report)
    """
    report = await check_audio_health(
        fmt=_RECORDING_FMT,
        duration_s=duration_s,
        device=device,
    )
    log.info(
        "recorder_startup_health_check",
        status=report.get("status"),
        rms_mean=round(report.get("rms_mean", 0.0), 4),
        clipping_ratio=round(report.get("clipping_ratio", 0.0), 4),
        silence_ratio=round(report.get("silence_ratio", 0.0), 4),
        duration_s=round(report.get("duration_s", 0.0), 3),
    )
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print(f"Recording format : {_RECORDING_FMT}")
    print(f"Enhancement      : {'enabled (' + _VAD_BACKEND + ' VAD)' if _ENABLE_ENHANCEMENT else 'disabled'}")
    print(f"Max duration     : {MAX_DURATION_S}s | Min: {MIN_DURATION_S}s")

    if len(sys.argv) > 1 and sys.argv[1] == "health":
        print("\nRunning startup health check (2 s)...")
        report = asyncio.run(run_startup_health_check(duration_s=2.0))
        for k, v in report.items():
            print(f"  {k}: {v}")
        sys.exit(0 if report.get("status") == "ok" else 1)

    # Standard 3-second PTT smoke test
    print(f"\nRecording for 3 seconds...")
    t0   = time.monotonic()
    path = record_audio_until_released(lambda: time.monotonic() - t0 < 3)

    if path:
        print(f"  Saved: {path}")

        lat = get_recording_latency_report()
        if lat:
            print("\nLatency report:")
            for stage, stats in lat.items():
                print(
                    f"  [{stage:20s}] "
                    f"mean={stats['mean_s'] * 1000:.2f}ms  "
                    f"p95={stats['p95_s'] * 1000:.2f}ms  "
                    f"count={int(stats['count'])}"
                )

        health = get_recording_health()
        print(
            f"\nMic health: {health.status} | "
            f"rms_mean={health.rms_mean:.4f} | "
            f"clip={health.clipping_rate:.2%} | "
            f"silence={health.silence_rate:.2%}"
        )

        delete_temp_recording(path)
        print("  Temp file deleted.")
    else:
        print("  No recording saved (silent, too short, or device error).")
        health = get_recording_health()
        print(f"  Mic health: {health.status} — {health.notes}")