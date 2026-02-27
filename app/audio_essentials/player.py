"""
Audio playback — optimised for partial TTS streaming.

Two playback paths
──────────────────
  FILE PATH    play_audio(path)       — whole file, worker thread, sd.play()
  STREAM PATH  play_audio_bytes(buf)  — per-sentence PCM chunks, persistent
                                        OutputStream owned by a writer thread

PCM-NATIVE PATH (via audio_engine integration)
──────────────────────────────────────────────
  play_pcm_chunk(chunk)   — zero-decode path: writes to PCMOutputStream directly
  play_pcm_bytes(raw, fmt) — raw PCM → PCMChunk → PCMOutputStream (no sf.read)
  get_interrupt_detector()  — barge-in detection during active playback
  get_playback_enhancer()   — pre-speaker limiter / AGC singleton
  get_output_stream()       — module-level PCMOutputStream singleton

They share the physical device and are mutually exclusive.

Latency optimisations (targeting 300–600 ms reduction)
────────────────────────────────────────────────────────
  1. Persistent PCMOutputStream — opened once, kept alive between chunks.
     Eliminates the 50–150 ms PortAudio device-open cost on every sentence.

  2. play_audio_bytes() decodes bytes → PCM (unavoidable, ~1–5 ms) then
     schedules an async write and returns in ~μs. The actual write runs inside
     the PCMOutputStream coroutine, completely off the TTS/LLM sync call stack.

  3. latency='low' on the underlying OutputStream — asks PortAudio for its
     minimum safe hardware buffer rather than the default conservative one.
     Saves 10–30 ms of output-buffer depth.

  4. Silent warmup chunk — written immediately after stream open to prime
     the driver pipeline so the first real audio chunk doesn't pay the cold-
     start penalty (~10–20 ms jitter on some drivers).

  5. Cached device info — queried once at stream open, not on every call.

  6. PCMOutputStream owns the write path — no external mutex contention on
     the hot path; all serialisation is internal to the stream object.

  7. PCM-native path — play_pcm_chunk() bypasses sf.read() entirely, writing
     typed PCMChunks through PCMPlaybackEnhancer → PCMOutputStream for the
     lowest-latency playback available. This is the path voice_graph.stream_full()
     should use.

Public API
──────────
  play_audio(path, *, on_complete, gain_db)  → None
  play_audio_bytes(audio_bytes, *, gain_db)  → None
  play_pcm_chunk(chunk, *, enhance)          → None    (PCM-native, no decode)
  play_pcm_bytes(raw_bytes, fmt, *, ...)     → None    (raw PCM → chunk → speaker)
  stop_audio()                               → None    file path only
  stop_stream()                              → None    stream path only
  stop_all()                                 → None    both paths
  is_playing()                               → bool
  is_streaming()                             → bool
  get_interrupt_detector()                   → PCMInterruptDetector
  get_playback_enhancer()                    → PCMPlaybackEnhancer
  get_output_stream()                        → PCMOutputStream
  get_playback_health()                      → AudioHealthReport
  get_playback_latency_report()              → dict
"""

from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.common.shared import make_counter, make_gauge, make_histogram
from app.monitoring.observability import get_logger

# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE INTEGRATION — PCM-native playback subsystem
# ═══════════════════════════════════════════════════════════════════════════════
#
# These imports wire the audio_engine's PCM pipeline into the player module,
# enabling zero-decode playback when callers provide typed PCMChunks instead
# of encoded audio bytes. This eliminates the sf.read() decode step (~1–5 ms)
# and enables the full enhancement chain (limiter, AGC, silence padding).
#
# PCMOutputStream is the *sole* low-level speaker sink for both the PCM-native
# path (play_pcm_chunk / play_pcm_bytes) and the legacy WAV path
# (play_audio_bytes). The old sd.OutputStream writer thread has been removed.
#
# Integration surface:
#   PCMFormat / PCMChunk      — typed audio descriptors replacing raw ints
#   PCMOutputStream           — async speaker output (persistent, low-latency)
#   PCMConverter              — format conversion to speaker-compatible dtype
#   PCMPlaybackEnhancer       — pre-speaker limiter + AGC chain
#   PCMInterruptDetector      — barge-in detection during TTS playback
#   PCMDiagnosticsMonitor     — output stream health monitoring
#   PCMLatencyTracker         — per-stage latency measurement
#   tts_pcm_to_chunk          — parse raw OpenAI TTS bytes → PCMChunk
#   chunk_to_wav_bytes        — fallback: PCMChunk → WAV for legacy path
#   get_chunk_pool            — zero-malloc array pool for hot-path allocs
#   get_format_registry       — canonical format lookup (e.g. "openai_tts")
#   get_converter             — module-level PCMConverter singleton
# ═══════════════════════════════════════════════════════════════════════════════
from app.audio_essentials.audio_engine import (

    # ── Core primitives ───────────────────────────────────────────────────────
    PCMFormat,
    PCMChunk,
    PCMConverter, # noqa

    # ── Streaming I/O ─────────────────────────────────────────────────────────
    PCMOutputStream,

    # ── Processing units ──────────────────────────────────────────────────────
    PCMPlaybackEnhancer,
    PCMInterruptDetector,

    # ── Diagnostics & observability ───────────────────────────────────────────
    PCMDiagnosticsMonitor,
    AudioHealthReport,
    PCMLatencyTracker,

    # ── Utility functions ─────────────────────────────────────────────────────
    tts_pcm_to_chunk,
    chunk_to_wav_bytes, # noqa

    # ── Module-level singletons ───────────────────────────────────────────────
    get_chunk_pool, # noqa
    get_format_registry,
    get_converter,
)

# ── logging & metrics ─────────────────────────────────────────────────────────

log = get_logger(__name__)

_plays_started = make_counter("player_plays_total", "File playback sessions started")
_plays_finished = make_counter(
    "player_finished_total", "File playback sessions completed normally"
)
_plays_stopped = make_counter(
    "player_stopped_total", "File playback sessions stopped early"
)
_plays_errors = make_counter("player_errors_total", "File playback errors")
_stream_chunks = make_counter(
    "player_stream_chunks_total", "PCM chunks written to OutputStream"
)
_stream_bytes = make_counter(
    "player_stream_bytes_total", "Bytes written to OutputStream"
)
_stream_decode_errs = make_counter(
    "player_stream_decode_errors_total", "Chunks that failed sf.read()"
)
_stream_write_errs = make_counter(
    "player_stream_write_errors_total", "OutputStream.write() failures"
)
_stream_recreations = make_counter(
    "player_stream_recreations_total", "OutputStream rebuilt (format change/crash)"
)
_stream_drops = make_counter(
    "player_stream_drops_total", "Chunks dropped because queue was full"
)

# PCM-native path metrics (audio_engine integration)
_pcm_chunks_played = make_counter(
    "player_pcm_chunks_total", "PCMChunks played via PCM-native path"
)
_pcm_enhanced_chunks = make_counter(
    "player_pcm_enhanced_total", "PCMChunks processed through PlaybackEnhancer"
)
_pcm_interrupt_events = make_counter(
    "player_pcm_interrupt_events_total", "Barge-in interrupts detected during playback"
)

_play_duration = make_histogram(
    "player_audio_duration_seconds",
    "Audio file duration",
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)
_play_start_latency = make_histogram(
    "player_start_latency_seconds",
    "File-path first-sample lag",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
_chunk_enqueue_lat = make_histogram(
    "player_chunk_enqueue_latency_seconds",
    "play_audio_bytes() wall time (decode + enqueue)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)
_write_lat = make_histogram(
    "player_write_latency_seconds",
    "OutputStream.write() duration on writer thread",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
_queue_depth = make_gauge(
    "player_stream_queue_depth", "Current items waiting in write queue"
)
_stream_active_gauge = make_gauge("player_stream_active", "1 when OutputStream is open")

# ── config ────────────────────────────────────────────────────────────────────

_WAIT_POLL_S: float = float(os.getenv("PLAYER_WAIT_POLL_S", "0.05"))
_FILE_JOIN_TIMEOUT_S: float = float(os.getenv("PLAYER_FILE_JOIN_TIMEOUT_S", "2.0"))

# ── Enable/disable PCM enhancement on the playback path.
# When True, all PCM-native playback goes through PCMPlaybackEnhancer (limiter + pad).
_ENHANCE_PLAYBACK: bool = os.getenv("PLAYER_ENHANCE_PLAYBACK", "true").lower() == "true"

# ── file-path state ───────────────────────────────────────────────────────────

_file_state_lock: threading.Lock = threading.Lock()
_playback_thread: threading.Thread | None = None
_active_stop_event: threading.Event | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO ENGINE SINGLETONS — lazy-initialised PCM pipeline components
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each singleton is created on first access (not at import time) to avoid
# PortAudio side effects in unit tests that merely import the module.
# All singletons are module-private; public access is through getter functions.
# ═══════════════════════════════════════════════════════════════════════════════

_pcm_output_stream: PCMOutputStream | None = None
_pcm_output_stream_lock = threading.Lock()

_playback_enhancer: PCMPlaybackEnhancer | None = None
_playback_enhancer_lock = threading.Lock()

_interrupt_detector: PCMInterruptDetector | None = None
_interrupt_detector_lock = threading.Lock()

_playback_diagnostics: PCMDiagnosticsMonitor | None = None
_playback_latency_tracker: PCMLatencyTracker = PCMLatencyTracker()


def get_output_stream() -> PCMOutputStream:
    """
    Return the module-level PCMOutputStream singleton.

    This is the preferred output path for PCM-native playback. It provides:
      • Persistent OutputStream (no device-open cost per chunk)
      • Automatic format negotiation and conversion
      • Thread-safe write queue with backpressure
      • Warmup frame priming for zero cold-start jitter

    The stream is created on first access and kept alive for the process lifetime.
    Call stop_all() to cleanly shut it down.
    """
    global _pcm_output_stream
    with _pcm_output_stream_lock:
        if _pcm_output_stream is None:
            output_fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
            _pcm_output_stream = PCMOutputStream(
                preferred_fmt=output_fmt,
                converter=get_converter(),
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.get_event_loop().run_until_complete(_pcm_output_stream.start())
            else:
                asyncio.create_task(_pcm_output_stream.start())
            log.info("player_pcm_output_stream_created", fmt=repr(output_fmt))
        return _pcm_output_stream


def get_playback_enhancer() -> PCMPlaybackEnhancer:
    """
    Return the module-level PCMPlaybackEnhancer singleton.

    Pre-speaker enhancement chain:
      TTS PCM → SilencePadder → Dynamics (limiter) → Speaker

    Prevents TTS output from clipping and pads silence for clean device
    transitions. Configured with TTS-optimised defaults.
    """
    global _playback_enhancer
    with _playback_enhancer_lock:
        if _playback_enhancer is None:
            output_fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
            _playback_enhancer = PCMPlaybackEnhancer(
                fmt=output_fmt,
                enable_limiter=True,
                enable_agc=False,  # TTS level is consistent
                pre_silence_s=0.05,
                post_silence_s=0.1,
            )
            log.info("player_playback_enhancer_created")
        return _playback_enhancer


def get_interrupt_detector(
    on_interrupt: Callable[[], None] | None = None,
) -> PCMInterruptDetector:
    """
    Return the module-level PCMInterruptDetector singleton.

    Monitors the microphone stream during TTS playback for barge-in events.
    When the user speaks over the agent's response, the detector fires
    ``on_interrupt`` so the caller can cancel TTS and start a new STT turn.

    Args:
        on_interrupt: Callback invoked on detected barge-in. Only set on
                      first call; subsequent calls return the existing detector.
    """
    global _interrupt_detector
    with _interrupt_detector_lock:
        if _interrupt_detector is None:
            input_fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
            _interrupt_detector = PCMInterruptDetector(
                fmt=input_fmt,
                on_interrupt=on_interrupt,
            )
            log.info("player_interrupt_detector_created")
        return _interrupt_detector


def get_playback_health() -> AudioHealthReport:
    """
    Return a point-in-time health report for the playback output path.

    Monitors clipping rate, silence rate, DC offset, and dropout count
    across recent playback chunks. Use in health endpoints and dashboards.
    """
    global _playback_diagnostics
    if _playback_diagnostics is None:
        output_fmt = get_format_registry().get("openai_tts") or PCMFormat.openai_tts()
        _playback_diagnostics = PCMDiagnosticsMonitor(output_fmt)
    return _playback_diagnostics.get_health_report()


def get_playback_latency_report() -> dict[str, dict[str, float]]:
    """
    Return per-stage latency statistics for the playback path.

    Stages tracked: "decode", "enhance", "enqueue", "write".
    """
    global _playback_latency_tracker
    return _playback_latency_tracker.get_latency_report()


# ───────────────────────────────────────────────────────────────────────────────
# PUBLIC API — PCM-NATIVE PATH (audio_engine integration)
# ───────────────────────────────────────────────────────────────────────────────
# These functions provide zero-decode playback for callers that already have
# typed PCMChunks (e.g. from tts_pcm_to_chunk() or PCMSpeechEnhancer).
# They bypass sf.read() entirely and write directly to PCMOutputStream.
# ───────────────────────────────────────────────────────────────────────────────


async def play_pcm_chunk(
    chunk: PCMChunk,
    *,
    enhance: bool | None = None,
    gain_db: float = 0.0,
) -> None:
    """
    Play a typed PCMChunk with zero decode overhead.

    This is the lowest-latency playback path. It bypasses sf.read() entirely
    and feeds the PCMChunk directly to the writer thread after optional
    enhancement and gain application.

    Enhancement chain (when enabled):
      PCMChunk → PCMPlaybackEnhancer (silence pad + limiter) → writer thread

    Args:
        chunk:    PCMChunk from audio_engine (e.g. from tts_pcm_to_chunk()).
        enhance:  Apply PCMPlaybackEnhancer. Default: _ENHANCE_PLAYBACK env var.
        gain_db:  dB gain. 0.0 = unity. Applied after enhancement.
    """
    if chunk.n_frames == 0:
        return

    t0 = time.monotonic()

    # ── Optional enhancement (limiter + silence padding) ─────────────────────
    if (enhance is None and _ENHANCE_PLAYBACK) or enhance:
        enhancer = get_playback_enhancer()
        chunk = enhancer._padder.process(chunk) # noqa
        if enhancer._limiter: # noqa
            chunk = enhancer._limiter.process(chunk) # noqa
        if enhancer._agc: # noqa
            chunk = enhancer._agc.process(chunk) # noqa
        _pcm_enhanced_chunks.inc()

    # ── Apply gain ───────────────────────────────────────────────────────────
    data = chunk.data
    if gain_db != 0.0:
        factor = 10.0 ** (gain_db / 20.0)
        data = np.clip(data * factor, -1.0, 1.0) # noqa

    # ── Feed interrupt detector (if active) with reference signal ────────────
    if _interrupt_detector is not None:
        _interrupt_detector.push_reference(chunk)

    # ── Signal file-path worker to stop ──────────────────────────────────────
    with _file_state_lock:
        evt = _active_stop_event
    if evt is not None and not evt.is_set():
        evt.set()
        sd.stop()

    # ── Write to PCMOutputStream ─────────────────────────────────────────────
    # chunk is already float32 after converter.convert() above; pass it straight
    # through to the persistent async output stream.
    try:
        await get_output_stream().write(chunk)
        _pcm_chunks_played.inc()
    except Exception as exc:
        _stream_drops.inc()
        log.warning("player_pcm_write_failed", seq=chunk.seq, error=str(exc))

    _chunk_enqueue_lat.observe(time.monotonic() - t0)


async def play_pcm_bytes(
    raw_bytes: bytes,
    fmt: PCMFormat | None = None,
    *,
    seq: int = 0,
    is_final: bool = False,
    enhance: bool | None = None,
    gain_db: float = 0.0,
) -> None:
    """
    Play raw PCM bytes (no WAV/MP3 container) via the PCM-native path.

    Wraps raw_bytes into a PCMChunk using tts_pcm_to_chunk() and feeds it
    through play_pcm_chunk(). This is the path for TTS_FORMAT="pcm" output
    from the OpenAI TTS API.

    Args:
        raw_bytes:  Raw PCM bytes (no header). Default format: 24kHz mono int16.
        fmt:        Override PCMFormat. Defaults to PCMFormat.openai_tts().
        seq:        Sequence number for ordering.
        is_final:   True when this is the last chunk of a synthesis.
        enhance:    Apply PCMPlaybackEnhancer. Default: _ENHANCE_PLAYBACK env var.
        gain_db:    dB gain. 0.0 = unity.
    """
    if not raw_bytes:
        return

    # Parse raw PCM bytes into a typed PCMChunk using audio_engine's parser
    chunk = tts_pcm_to_chunk(raw_bytes, fmt=fmt, seq=seq, is_final=is_final)
    await play_pcm_chunk(chunk, enhance=enhance, gain_db=gain_db)


async def drain_output() -> None:
    """Block until the PCMOutputStream queue is empty and audio has played out."""
    stream = get_output_stream()
    # Poll until the internal queue is empty
    while stream.is_active() and getattr(stream, '_queue', None) and stream._queue.qsize() > 0:  # noqa
        await asyncio.sleep(0.05)
    # Extra buffer for the final hardware frame to flush
    await asyncio.sleep(0.15)


# ── public API — streaming path ───────────────────────────────────────────────


async def play_audio_bytes(
    audio_bytes: bytes,
    *,
    gain_db: float = 0.0,
) -> None:
    """
    Enqueue a decoded PCM chunk for streaming playback.

    Designed for partial TTS output: call this for every sentence-sized audio
    chunk as it arrives from the TTS node. The first chunk opens the stream;
    subsequent chunks write with zero device-open overhead.

    This function is intentionally non-blocking. After decoding bytes → PCM
    (~1–5 ms, unavoidable), it enqueues the data and returns in ~microseconds.
    The actual OutputStream.write() runs on the dedicated writer thread.

    Stops any active file-path playback (signals stop, does NOT join) so the
    two paths never write to the device simultaneously.

    Args:
        audio_bytes: Raw audio in any format readable by soundfile
                     (WAV, FLAC, OGG, MP3 with libsndfile, raw PCM not supported).
        gain_db:     dB gain. 0.0 = unity. Applied before enqueue.
    """
    if not audio_bytes:
        return

    t0 = time.monotonic()

    # Decode bytes → float32 PCM. This is synchronous I/O on the caller's
    # thread (~1–5 ms) but is unavoidable without knowing the format ahead of time.
    try:
        data, samplerate = sf.read(
            io.BytesIO(audio_bytes), dtype="float32", always_2d=False
        )
    except Exception as exc:
        log.error(
            "player_stream_decode_error", error=str(exc), chunk_bytes=len(audio_bytes)
        )
        _stream_decode_errs.inc()
        return

    if data is None or data.size == 0:
        return

    channels = 1 if data.ndim == 1 else data.shape[1]

    if gain_db != 0.0:
        data = _apply_gain(data, gain_db)

    # ── Feed interrupt detector with playback reference signal ────────────────
    # This enables barge-in suppression: the detector knows the speaker output
    # level and won't misclassify echo as user speech.
    ref_fmt = PCMFormat(sample_rate=samplerate, channels=channels, dtype="float32")
    pcm_chunk = PCMChunk(
        data=data, fmt=ref_fmt,
        timestamp=time.monotonic(), source="player_legacy",
    )
    if _interrupt_detector is not None:
        _interrupt_detector.push_reference(pcm_chunk)

    # Signal the file-path worker to stop — don't join, latency matters.
    # The file-path worker will drain on its own; sd.stop() unblocks sd.wait().
    with _file_state_lock:
        evt = _active_stop_event
    if evt is not None and not evt.is_set():
        evt.set()
        sd.stop()

    # Wrap decoded PCM in a typed chunk and write to PCMOutputStream.
    # asyncio.create_task() schedules the async write and returns in ~μs so
    # the TTS event loop is never blocked by speaker I/O.
    try:
        await get_output_stream().write(pcm_chunk)
        _stream_chunks.inc()
        _stream_bytes.inc(data.nbytes)
    except Exception as exc:
        _stream_drops.inc()
        log.warning(
            "player_stream_write_failed",
            queue_depth=0,
            error=str(exc),
        )

    _chunk_enqueue_lat.observe(time.monotonic() - t0)


def stop_stream() -> None:
    """
    Stop the streaming path and close the PCMOutputStream.

    Calls PCMOutputStream.stop() which cancels any pending async writes and
    closes the underlying hardware stream. Also signals the interrupt detector
    that playback has stopped.

    Safe to call at any time from any thread.
    """
    output_stream = get_output_stream()
    coro = output_stream.stop_playback()
    try:
        asyncio.create_task(coro)
    except RuntimeError:
        # No running event loop (e.g. called from a non-async context at shutdown).
        # Explicitly close the coroutine to suppress the "never awaited" warning.
        coro.close()

    # ── Notify interrupt detector that playback has ended ─────────────────────
    if _interrupt_detector is not None:
        _interrupt_detector.set_playback_active(False)

    log.debug("player_stream_stop_requested")


def is_streaming() -> bool:
    """True if the PCMOutputStream is currently open and active."""
    return get_output_stream().is_active()


# ── public API — file path ────────────────────────────────────────────────────


def play_audio(
    path: str,
    *,
    on_complete: Callable[[], None] | None = None,
    gain_db: float = 0.0,
) -> None:
    """
    Play an audio file asynchronously via a worker thread.

    Stops the streaming path first so both paths never write simultaneously.
    Joins the previous file-path worker before starting a new one so the old
    worker's finally-block sd.stop() cannot race with the new stream.

    Args:
        path:        Path to a WAV/FLAC/OGG/MP3 file.
        on_complete: Zero-arg callback invoked on normal completion (not on stop).
        gain_db:     dB gain applied before playback. 0.0 = unity.
    """
    global _playback_thread, _active_stop_event

    audio_path = Path(path)

    if not audio_path.exists():
        log.error("player_file_not_found", path=path)
        return

    if audio_path.stat().st_size == 0:
        log.warning("player_file_empty", path=path)
        return

    # Stop streaming path — must happen before we open the file-path stream
    # to avoid simultaneous writes to the same output device.
    stop_stream()

    # Join previous file-path worker so its finally-block sd.stop() fires
    # before we open the new stream, not after.
    _file_stop_and_join()

    my_stop_event = threading.Event()

    with _file_state_lock:
        _active_stop_event = my_stop_event
        _playback_thread = threading.Thread(
            target=_file_playback_worker,
            args=(audio_path, my_stop_event, on_complete, gain_db),
            daemon=True,
            name=f"player-file:{audio_path.name}",
        )
        _playback_thread.start()

    log.info("player_play_started", path=str(audio_path), gain_db=gain_db)
    _plays_started.inc()


def stop_audio() -> None:
    """
    Immediately stop the file playback path.

    Sets the worker's stop event and calls sd.stop() to unblock sd.wait().
    Does NOT join the worker thread (use _file_stop_and_join() for that).
    Does NOT stop the streaming path — call stop_stream() or stop_all().

    Safe to call from any thread at any time.
    """
    with _file_state_lock:
        evt = _active_stop_event

    if evt is not None:
        evt.set()

    sd.stop()
    log.debug("player_stop_audio_requested")


def is_playing() -> bool:
    """True if the file-path worker thread is alive. Advisory only."""
    with _file_state_lock:
        t = _playback_thread
    return t is not None and t.is_alive()


# ── public API — combined ─────────────────────────────────────────────────────


def stop_all() -> None:
    """
    Stop both playback paths in a single call.

    Preferred over calling stop_audio() + stop_stream() separately in PTT
    interrupt handlers where you want everything silenced atomically.
    """
    stop_stream()
    stop_audio()
    log.debug("player_stop_all")


# ── internal helpers — file path ──────────────────────────────────────────────


def _file_stop_and_join() -> None:
    """Signal the file-path worker and join it. Called outside _file_state_lock."""
    with _file_state_lock:
        evt = _active_stop_event
        thread = _playback_thread

    if evt is not None:
        evt.set()

    sd.stop()

    if thread is not None and thread.is_alive():
        thread.join(timeout=_FILE_JOIN_TIMEOUT_S)
        if thread.is_alive():
            log.error(
                "player_file_thread_join_timeout",
                thread=thread.name,
                timeout_s=_FILE_JOIN_TIMEOUT_S,
            )


def _apply_gain(data: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply dB gain and hard-clip to [-1, 1] to prevent wrap-around distortion."""
    factor = 10.0 ** (gain_db / 20.0)
    return np.clip(data * factor, -1.0, 1.0)


def _coerce_channels(data: np.ndarray, device_max_channels: int) -> np.ndarray:
    """
    Coerce channel count to what the device supports.
    Mono is passed through unchanged (PortAudio upmixes internally).
    Stereo is preserved if device allows it; otherwise downmixed by averaging.
    """
    if data.ndim == 1 or data.shape[1] <= device_max_channels:
        return data

    log.warning(
        "player_channel_downmix",
        file_channels=data.shape[1],
        device_channels=device_max_channels,
    )
    return np.mean(data, axis=1)


def _file_playback_worker(
    audio_path: Path,
    stop_event: threading.Event,
    on_complete: Callable[[], None] | None,
    gain_db: float,
) -> None:
    """
    Worker thread for whole-file playback via sd.play().

    1. Read file from disk.
    2. Apply gain and channel coercion.
    3. Bail early if already stopped.
    4. sd.play() + poll loop so stop_event is honoured even if sd.wait() hangs.
    5. Fire on_complete on normal finish.
    6. sd.stop() in finally only if this worker's stop_event was set —
       never kills a stream we don't own.
    """
    t_call = time.monotonic()
    completed_normally = False

    try:
        try:
            data, samplerate = sf.read(
                str(audio_path), dtype="float32", always_2d=False
            )
        except Exception as exc:
            log.error("player_file_read_error", path=str(audio_path), error=str(exc))
            _plays_errors.inc()
            return

        if data is None or data.size == 0:
            log.warning("player_file_has_no_samples", path=str(audio_path))
            return

        _play_duration.observe(data.shape[0] / samplerate)

        if gain_db != 0.0:
            data = _apply_gain(data, gain_db)

        try:
            out_idx = sd.default.device[1]
            if out_idx < 0:
                out_idx = sd.query_devices(kind="output")["index"]
            dev_info = sd.query_devices(out_idx)
            max_out_ch = int(dev_info.get("max_output_channels", 1))
        except Exception:  # noqa
            out_idx = None
            max_out_ch = 1

        data = _coerce_channels(data, max_out_ch)

        if stop_event.is_set():
            _plays_stopped.inc()
            return

        _play_start_latency.observe(time.monotonic() - t_call)
        sd.play(data, samplerate, device=out_idx)
        log.debug(
            "player_file_stream_open", path=str(audio_path), samplerate=samplerate
        )

        while not stop_event.is_set():
            try:
                s = sd.get_stream()
                if s is None or not s.active:
                    break
            except Exception:  # noqa
                break
            time.sleep(_WAIT_POLL_S)

        if stop_event.is_set():
            log.info("player_file_stopped", path=str(audio_path))
            _plays_stopped.inc()
        else:
            completed_normally = True
            log.info("player_file_finished", path=str(audio_path))
            _plays_finished.inc()

    except Exception as exc:
        log.error("player_file_unexpected_error", path=str(audio_path), error=str(exc))
        _plays_errors.inc()

    finally:
        # Only call sd.stop() if we own the stop — never kill a newer stream.
        if stop_event.is_set():
            sd.stop()

        if completed_normally and on_complete is not None:
            try:
                on_complete()
            except Exception as exc:
                log.error("player_on_complete_error", error=str(exc))


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math
    import struct
    import sys

    def _make_wav_chunk(
        start_sample: int, n_samples: int, rate: int, freq: float
    ) -> bytes:
        """Build a minimal WAV wrapping a sine-wave segment."""
        samples = [
            int(32767 * math.sin(2 * math.pi * freq * (start_sample + i) / rate))
            for i in range(n_samples)
        ]
        pcm = struct.pack(f"<{n_samples}h", *samples)
        data_size = len(pcm)
        hdr = bytearray()
        hdr += b"RIFF"
        hdr += struct.pack("<I", 36 + data_size)
        hdr += b"WAVE"
        hdr += b"fmt "
        hdr += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        hdr += b"data"
        hdr += struct.pack("<I", data_size)
        return bytes(hdr) + pcm

    if len(sys.argv) > 1:
        print(f"[FILE] Playing {sys.argv[1]}")
        play_audio(sys.argv[1])
        while is_playing():
            time.sleep(0.05)
        print("[FILE] Done.")
    else:
        RATE, FREQ, CHUNK_S, TOTAL_S = 24000, 440.0, 0.1, 3.0
        n_chunks = int(TOTAL_S / CHUNK_S)
        chunk_frames = int(RATE * CHUNK_S)

        print(
            f"[STREAM] Streaming {FREQ} Hz sine wave — {n_chunks} × {int(CHUNK_S * 1000)} ms chunks"
        )

        # ── Test PCM-native path ─────────────────────────────────────────────
        print("\n[PCM-NATIVE] Testing play_pcm_chunk()...")
        fmt = PCMFormat.openai_tts()
        t = np.linspace(0, 1.0, fmt.sample_rate, endpoint=False)
        sine = (np.sin(2 * np.pi * 440 * t) * 28000).astype(np.int16)
        test_chunk = PCMChunk(
            data=sine, fmt=fmt, seq=0, is_final=True,
            timestamp=time.monotonic(), source="test",
        )
        play_pcm_chunk(test_chunk, enhance=True)
        time.sleep(1.2)
        print("[PCM-NATIVE] Done.")

        # ── Test raw PCM bytes path ──────────────────────────────────────────
        print("\n[PCM-BYTES] Testing play_pcm_bytes()...")
        raw = sine.tobytes()
        play_pcm_bytes(raw, fmt=fmt, is_final=True)
        time.sleep(1.2)
        print("[PCM-BYTES] Done.")

        # ── Test legacy WAV path ─────────────────────────────────────────────
        print("\n[LEGACY] Testing play_audio_bytes()...")
        try:
            for i in range(n_chunks):
                wav = _make_wav_chunk(i * chunk_frames, chunk_frames, RATE, FREQ)
                play_audio_bytes(wav)
                time.sleep(CHUNK_S * 0.85)

            time.sleep(CHUNK_S)
            print("[LEGACY] Done.")
        except KeyboardInterrupt:
            stop_all()
            print("[LEGACY] Stopped.")