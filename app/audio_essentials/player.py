"""
Audio playback — production-grade, optimised for partial TTS streaming.

Two playback paths
──────────────────
  FILE PATH    play_audio(path)       — whole file, worker thread, sd.play()
  STREAM PATH  play_audio_bytes(buf)  — per-sentence PCM chunks, persistent
                                        OutputStream owned by a writer thread

They share the physical device and are mutually exclusive.

Latency optimisations (targeting 300–600 ms reduction)
────────────────────────────────────────────────────────
  1. Persistent OutputStream — opened once, kept alive between chunks.
     Eliminates the 50–150 ms PortAudio device-open cost on every sentence.

  2. Dedicated writer thread + bounded queue — play_audio_bytes() decodes
     bytes → PCM (unavoidable, ~1–5 ms) then enqueues and returns in ~μs.
     The 20–50 ms write() call happens on the writer thread, completely off
     the TTS/LLM async event loop.

  3. latency='low' on OutputStream — asks PortAudio for its minimum safe
     hardware buffer rather than the default conservative one. Saves 10–30 ms
     of output-buffer depth.

  4. Silent warmup chunk — written immediately after stream open to prime
     the driver pipeline so the first real audio chunk doesn't pay the cold-
     start penalty (~10–20 ms jitter on some drivers).

  5. Cached device info — queried once at stream open, not on every call.

  6. No lock on the write path — the writer thread owns the OutputStream
     exclusively. Zero mutex contention on the hot path.

Public API
──────────
  play_audio(path, *, on_complete, gain_db)  → None
  play_audio_bytes(audio_bytes, *, gain_db)  → None
  stop_audio()                               → None  file path only
  stop_stream()                              → None  stream path only
  stop_all()                                 → None  both paths
  is_playing()                               → bool
  is_streaming()                             → bool
"""

from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.common.shared import get_logger, make_counter, make_gauge, make_histogram

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
_WRITER_JOIN_TIMEOUT_S: float = float(os.getenv("PLAYER_WRITER_JOIN_TIMEOUT_S", "1.0"))

# Maximum chunks buffered before new arrivals are dropped.
# At ~100ms per TTS sentence chunk, 8 = ~800ms of audio buffer depth.
_QUEUE_MAXSIZE: int = int(os.getenv("PLAYER_STREAM_QUEUE_MAXSIZE", "8"))

# How many silent frames to write after stream open to prime the driver pipeline.
# 512 frames @ 24kHz = 21ms — large enough to prime, small enough to be inaudible.
_WARMUP_FRAMES: int = int(os.getenv("PLAYER_STREAM_WARMUP_FRAMES", "512"))


# ── queue item types ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class _AudioChunk:
    data: np.ndarray  # float32 PCM
    samplerate: int
    channels: int


class _StopSentinel:
    """Tells the writer thread to close the stream and idle."""


class _ShutdownSentinel:
    """Tells the writer thread to exit permanently (process teardown)."""


_STOP = _StopSentinel()
_SHUTDOWN = _ShutdownSentinel()

# ── file-path state ───────────────────────────────────────────────────────────

_file_state_lock: threading.Lock = threading.Lock()
_playback_thread: threading.Thread | None = None
_active_stop_event: threading.Event | None = None

# ── stream-path state (writer-thread owned) ───────────────────────────────────

# The writer thread is the *sole* owner of _stream, _stream_sr, _stream_ch.
# No other thread touches these after _writer_thread is started.
_stream: sd.OutputStream | None = None
_stream_sr: int = 0
_stream_ch: int = 0

# Shared between threads (set/cleared by writer thread, read by is_streaming()).
_stream_open_event: threading.Event = threading.Event()

# The bounded write queue. Producer: play_audio_bytes(). Consumer: writer thread.
_write_queue: queue.Queue[_AudioChunk | _StopSentinel | _ShutdownSentinel] = (
    queue.Queue(maxsize=_QUEUE_MAXSIZE)
)


# ── writer thread ─────────────────────────────────────────────────────────────


def _writer_loop() -> None:
    """
    Daemon thread that owns the OutputStream for its entire lifetime.

    Owns all stream open/close/write operations — no other thread touches
    the OutputStream object, so zero mutex contention on the write path.

    Items pulled from _write_queue:
      _AudioChunk      — decode already done; open/recreate stream if needed, then write.
      _StopSentinel    — close stream, idle until next _AudioChunk arrives.
      _ShutdownSentinel — close stream, exit loop (process teardown only).
    """
    global _stream, _stream_sr, _stream_ch

    while True:
        try:
            item = _write_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        # ── shutdown ──────────────────────────────────────────────────────────
        if isinstance(item, _ShutdownSentinel):
            _writer_close_stream()
            break

        # ── stop ──────────────────────────────────────────────────────────────
        if isinstance(item, _StopSentinel):
            _writer_close_stream()
            continue

        # ── audio chunk ───────────────────────────────────────────────────────
        chunk: _AudioChunk = item  # type: ignore[assignment]
        _queue_depth.set(_write_queue.qsize())

        # Open or recreate stream when format changes or after a crash.
        needs_open = (
            _stream is None
            or not _stream.active
            or _stream_sr != chunk.samplerate
            or _stream_ch != chunk.channels
        )
        if needs_open:
            _writer_open_stream(chunk.samplerate, chunk.channels)
            if _stream is None:
                # Device open failed — skip chunk, will retry on next one.
                continue

        # Write PCM. This blocks until PortAudio has consumed the buffer —
        # that's fine here because this is the dedicated writer thread.
        t0 = time.monotonic()
        try:
            _stream.write(chunk.data)
        except Exception as exc:
            log.error("player_stream_write_error", error=str(exc))
            _stream_write_errs.inc()
            _writer_close_stream()
            continue

        write_s = time.monotonic() - t0
        _write_lat.observe(write_s)
        _stream_chunks.inc()
        _stream_bytes.inc(chunk.data.nbytes)

        log.debug(
            "player_stream_chunk_written",
            samples=chunk.data.shape[0],
            samplerate=chunk.samplerate,
            write_s=round(write_s, 4),
        )


def _writer_open_stream(samplerate: int, channels: int) -> None:
    """Open (or reopen) the OutputStream. Called only from the writer thread."""
    global _stream, _stream_sr, _stream_ch

    _writer_close_stream()

    try:
        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            latency="low",  # ask PortAudio for its minimum safe buffer depth
            blocksize=0,  # let PortAudio choose block size within latency budget
        )
        stream.start()

        # Prime the driver pipeline with silence so the first real chunk
        # doesn't pay the cold-start jitter penalty (~10–20ms on many drivers).
        silence = np.zeros(
            (_WARMUP_FRAMES, channels) if channels > 1 else (_WARMUP_FRAMES,),
            dtype=np.float32,
        )
        stream.write(silence)

        _stream = stream
        _stream_sr = samplerate
        _stream_ch = channels
        _stream_open_event.set()
        _stream_active_gauge.set(1)
        _stream_recreations.inc()

        log.info("player_stream_opened", samplerate=samplerate, channels=channels)

    except Exception as exc:
        log.error(
            "player_stream_open_failed",
            samplerate=samplerate,
            channels=channels,
            error=str(exc),
        )
        _stream = None
        _stream_sr = 0
        _stream_ch = 0
        _stream_open_event.clear()
        _stream_active_gauge.set(0)


def _writer_close_stream() -> None:
    """Close the OutputStream. Called only from the writer thread."""
    global _stream, _stream_sr, _stream_ch

    if _stream is None:
        return

    try:
        _stream.stop()
        _stream.close()
        log.info("player_stream_closed")
    except Exception as exc:
        log.warning("player_stream_close_error", error=str(exc))
    finally:
        _stream = None
        _stream_sr = 0
        _stream_ch = 0
        _stream_open_event.clear()
        _stream_active_gauge.set(0)


# Start the writer thread once at module load.
# daemon=True means it is killed automatically when the main thread exits.
_writer_thread = threading.Thread(
    target=_writer_loop,
    daemon=True,
    name="player-writer",
)
_writer_thread.start()


# ── public API — streaming path ───────────────────────────────────────────────


def play_audio_bytes(
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

    # Signal the file-path worker to stop — don't join, latency matters.
    # The file-path worker will drain on its own; sd.stop() unblocks sd.wait().
    with _file_state_lock:
        evt = _active_stop_event
    if evt is not None and not evt.is_set():
        evt.set()
        sd.stop()

    # Enqueue for the writer thread. Drop if full rather than blocking —
    # a full queue means the consumer (hardware) can't keep up, and blocking
    # the TTS async loop would cause a larger audio gap than a dropped chunk.
    chunk = _AudioChunk(data=data, samplerate=samplerate, channels=channels)
    try:
        _write_queue.put_nowait(chunk)
    except queue.Full:
        _stream_drops.inc()
        log.warning(
            "player_stream_queue_full_chunk_dropped",
            queue_depth=_write_queue.qsize(),
        )

    _chunk_enqueue_lat.observe(time.monotonic() - t0)


def stop_stream() -> None:
    """
    Stop the streaming path and close the OutputStream.

    Drains the write queue (drops pending chunks) then sends the stop sentinel.
    The writer thread closes the stream on the next loop iteration.

    Safe to call at any time from any thread.
    """
    # Drain pending chunks so the writer thread doesn't play stale audio
    # from a prior response after the interrupt.
    drained = 0
    while True:
        try:
            _write_queue.get_nowait()
            drained += 1
        except queue.Empty:
            break

    # Send the stop sentinel so the writer thread closes the stream even if
    # it was already idle (no pending chunks to drain past).
    try:
        _write_queue.put_nowait(_STOP)
    except queue.Full:
        # Queue is full despite our drain — extremely unlikely but harmless;
        # the writer thread will drain the remaining chunks and eventually stop.
        pass

    if drained:
        log.info("player_stream_stop_drained", chunks_dropped=drained)

    log.debug("player_stream_stop_sent")


def is_streaming() -> bool:
    """True if the OutputStream is currently open and active."""
    return _stream_open_event.is_set()


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
        print("         Press Ctrl-C to interrupt.\n")

        try:
            for i in range(n_chunks):
                wav = _make_wav_chunk(i * chunk_frames, chunk_frames, RATE, FREQ)
                play_audio_bytes(wav)
                # Sleep slightly less than chunk duration to keep the queue fed
                # without over-filling it.  In production the TTS node controls
                # this pacing naturally via its own synthesis speed.
                time.sleep(CHUNK_S * 0.85)

            time.sleep(CHUNK_S)  # let the last chunk drain
            print("[STREAM] Done.")
        except KeyboardInterrupt:
            stop_all()
            print("[STREAM] Stopped.")