"""
Desktop hold-to-talk controller.

This file is the desktop input layer — keyboard events, audio recording,
and pipeline dispatch only. All pipeline logic stays in voice_graph.py.

Flow
────
  Hold PTT_KEY  → interrupt in-flight pipeline + stop playback
                → record microphone until key released
                → dispatch recording into the shared async event loop
  EXIT_KEY      → cancel active pipeline, drain loop, exit cleanly

A single asyncio event loop runs on a background thread for the lifetime
of the process. Dispatches use run_coroutine_threadsafe() — one loop,
zero teardown overhead per PTT press, clean cancellation semantics.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import os
import signal
import sys
import threading
import time
import contextvars  # noqa

import keyboard

from app.common.settings import settings
from app.common.shared import (
    get_tracer,
    make_counter,
    make_histogram,
    new_request_id,
)

from app.monitoring.observability import get_logger
from app.startup.startup_display import show_boot_sequence

from app.monitoring.observability import (
    bootstrap as _obs_bootstrap,
    session_context,
    ControllerEmitter,
)

from app.audio_essentials.player import (
    play_pcm_bytes,
    drain_output,
    stop_all,
)

from app.audio_essentials.audio_engine import PCMFormat
from app.audio_essentials.recorder import record_audio_until_released
from app.orchestration.voice_graph import (
    voice_graph,
    on_startup as _voice_graph_startup,
    on_shutdown as _voice_graph_shutdown,
)

log = get_logger(__name__)
tracer = get_tracer(__name__)

from app.eval.evaluation_engine import evaluation_engine
import socket

from app.user_tracking.session_service.session_store import session_store
from app.user_tracking.transcript.transcription import transcript_writer

# ── IP resolver ───────────────────────────────────────────────────────────────


def _get_local_ip() -> str:
    """
    Resolve the machine's outbound IP address.
    Uses a UDP connect trick — no packet is actually sent,
    but the OS selects the correct source interface.
    Falls back to 127.0.0.1 if the machine is fully offline.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:  # noqa
        return "127.0.0.1"


# ── config ────────────────────────────────────────────────────────────────────
# Controller-specific timings come from the validated settings singleton where
# available so that fat-fingered env vars are caught at startup. Keys that are
# genuinely controller-only (not shared with the pipeline) fall back to
# getattr(..., default) so the code degrades gracefully if settings doesn't
# expose them yet — add them to settings.py to gain startup-time validation.

POLL_INTERVAL_S: float = getattr(settings, "controller_poll_interval_s", 0.03)
DEBOUNCE_S: float = getattr(settings, "controller_debounce_s", 0.35)
INTERRUPT_DRAIN_S: float = getattr(settings, "controller_interrupt_drain_s", 0.05)
LOOP_DRAIN_S: float = getattr(settings, "controller_loop_drain_s", 0.10)
LOOP_JOIN_TIMEOUT: float = getattr(settings, "controller_loop_join_timeout", 3.0)

PTT_KEY: str = getattr(settings, "controller_ptt_key", None) or os.getenv(
    "CONTROLLER_PTT_KEY", "h"
)
EXIT_KEY: str = getattr(settings, "controller_exit_key", None) or os.getenv(
    "CONTROLLER_EXIT_KEY", "esc"
)

# ── metrics ───────────────────────────────────────────────────────────────────

_sessions_started = make_counter("controller_sessions_total", "PTT sessions dispatched")
_sessions_interrupted = make_counter(
    "controller_interrupts_total", "Pipelines interrupted by user"
)
_empty_recordings = make_counter(
    "controller_empty_recordings_total", "PTT releases with no usable audio"
)
_pipeline_errors = make_counter(
    "controller_pipeline_errors_total", "Unhandled pipeline errors in controller"
)

_recording_duration = make_histogram(
    "controller_recording_duration_seconds",
    "Microphone hold duration per PTT press",
    buckets=(0.5, 1, 2, 5, 10, 30),
)


# ── pipeline state ─────────────────────────────────────────────────────────────


class _ActivePipeline:
    """
    Thread-safe slot for the currently executing pipeline's request ID.

    Using an explicit lock rather than relying on the GIL means the
    read-modify-write in take() is safe even if we ever move off CPython.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rid: str | None = None

    def set(self, rid: str | None) -> None:
        with self._lock:
            self._rid = rid

    def get(self) -> str | None:
        with self._lock:
            return self._rid

    def take(self) -> str | None:
        """Return and clear atomically — prevents double-cancels."""
        with self._lock:
            rid, self._rid = self._rid, None
            return rid


_active = _ActivePipeline()

# Set by _interrupt() before cancel() so _dispatch() drops any coroutine
# that was scheduled but hasn't been picked up by the event loop yet.
_interrupt_flag = threading.Event()

# ── session state ──────────────────────────────────────────────────────────────

_local_ip: str = _get_local_ip()
_session_id: str | None = None  # set once in listen_loop() before the poll starts

# ── async event loop ───────────────────────────────────────────────────────────
# One loop, one thread, for the lifetime of the process.
# Avoids creating and tearing down a new loop on every PTT press.

_loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
_loop_thread: threading.Thread  # assigned in listen_loop() before first use


def _run_loop() -> None:
    asyncio.set_event_loop(_loop)

    # Create one stable context for the entire loop lifetime so ContextVar
    # values (e.g. current_request_id) are properly scoped per-task.
    ctx = contextvars.copy_context()

    def _loop_forever():
        _loop.run_forever()

    ctx.run(_loop_forever)


# ── pipeline dispatch ──────────────────────────────────────────────────────────


def _dispatch(audio_path: str) -> None:
    """
    Schedule a pipeline run on the shared event loop from the keyboard thread.

    Assigns request_id before coroutine creation so the cancel path can
    find this pipeline by ID from the instant we're scheduled, not just
    once the coroutine starts executing.
    """
    if _interrupt_flag.is_set():
        # User pressed PTT again before the previous dispatch even started —
        # discard silently rather than running a pipeline they already cancelled.
        log.info("controller_dispatch_dropped", reason="interrupt_pending")
        return

    rid = new_request_id()
    _active.set(rid)

    async def _run() -> None:
        async with session_context(
            session_id=_session_id or "",
            request_id=rid,
        ):
            with tracer.start_as_current_span("controller.pipeline") as span:
                span.set_attribute("request_id", rid)
                try:
                    log.info(
                        "controller_pipeline_start",
                        request_id=rid,
                        audio_path=audio_path,
                    )

                    _pcm_fmt = PCMFormat.openai_tts()

                    async for segment in voice_graph.stream_full(
                            audio_path=audio_path,
                            session_id=_session_id or "",
                            request_id=rid,
                    ):
                        local_path: str = segment.get("local_path", "")
                        if not local_path:
                            continue
                        with open(local_path, "rb") as _f:
                            pcm_bytes = _f.read()
                        await play_pcm_bytes(pcm_bytes, _pcm_fmt)

                    await drain_output()  # wait for speaker buffer to flush
                    log.info("controller_pipeline_done", request_id=rid)

                except asyncio.CancelledError:
                    # Normal path — user pressed PTT or ESC mid-response.
                    log.info("controller_pipeline_cancelled", request_id=rid)

                except Exception as exc:
                    _pipeline_errors.inc()
                    ControllerEmitter.pipeline_error(
                        session_id=_session_id or "",
                        request_id=rid,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    log.error(
                        "controller_pipeline_error", request_id=rid, error=str(exc)
                    )

                finally:
                    # Guard against clearing a newer request's ID if a second
                    # PTT press arrived while this pipeline was still running.
                    if _active.get() == rid:
                        _active.set(None)

    asyncio.run_coroutine_threadsafe(_run(), _loop)


# ── interrupt ──────────────────────────────────────────────────────────────────


def _interrupt(*, source: str) -> None:
    """
    Cancel the active pipeline and stop audio playback in one shot.

    Sets the interrupt flag first so _dispatch() drops any concurrently
    scheduled-but-not-yet-started coroutine before it hits the event loop.
    """
    _interrupt_flag.set()

    rid = _active.take()
    if rid:
        voice_graph.cancel(rid, reason="interrupt", source=source)
        _sessions_interrupted.inc()
        ControllerEmitter.interrupt(session_id=_session_id or "", request_id=rid)
        log.info("controller_interrupt", request_id=rid, source=source)

    stop_all()


# ── shutdown ───────────────────────────────────────────────────────────────────


def _shutdown(*, source: str) -> None:
    """
    Cancel the active pipeline, drain the event loop, then stop it.
    Safe to call from the keyboard loop, signal handlers, or tests.

    Shutdown order
    ──────────────
      1. _interrupt()                  — cancel active task + stop audio
      2. transcript close_session()    — write footer line to .txt file
      3. session_store.end()           — release IP lock; must run while node
                                         pools are still open (before step 4)
      4. evaluation_engine.close()     — flush pending eval tasks
      5. voice_graph.on_shutdown()     — canonical teardown sequence:
                                           a. flush QA audit bus (15 s timeout)
                                           b. flush transcript writer queue
                                           c. stop LLM health probe
                                           d. shutdown all three graph singletons
                                              (voice_graph, realtime, low_latency)
                                           e. close LLM node connection pool
                                           f. close QA audit bus drain task
      6. loop.stop() + thread join     — stop the asyncio loop
    """
    log.info("controller_shutdown_start", source=source)
    ControllerEmitter.shutdown(reason=source)
    _interrupt(source=source)

    time.sleep(LOOP_DRAIN_S)

    # ── close transcript file for this session ─────────────────────────────────
    # Write the footer line *before* on_shutdown() flushes and drains the queue.
    # This ensures the close footer is the last line in the .txt file.
    if _session_id:
        try:
            asyncio.run_coroutine_threadsafe(
                transcript_writer.close_session(_session_id),
                _loop,
            ).result(timeout=3.0)
        except Exception as exc:
            log.warning("controller_transcript_close_failed", error=str(exc))

    # ── end session ────────────────────────────────────────────────────────────
    # Must run before voice_graph.on_shutdown() closes node pools so any
    # session-related in-flight tasks (e.g. session_store.append_turn) complete
    # while the pools are still available.
    if _session_id:
        try:
            asyncio.run_coroutine_threadsafe(
                session_store.end(_session_id, reason=f"shutdown:{source}"),
                _loop,
            ).result(timeout=5.0)
            log.info("controller_session_ended", session_id=_session_id)
        except Exception as exc:
            log.warning("controller_session_end_failed", error=str(exc))

    # ── close eval engine ──────────────────────────────────────────────────────
    try:
        asyncio.run_coroutine_threadsafe(
            evaluation_engine.close(),
            _loop,
        ).result(timeout=5.0)
    except Exception as exc:
        log.warning("controller_eval_close_failed", error=str(exc))

    # ── canonical voice graph teardown ─────────────────────────────────────────
    # on_shutdown() owns the full teardown sequence:
    #   - QA audit bus flush (15 s) — eval data not silently dropped
    #   - transcript writer queue drain
    #   - LLM health probe stop
    #   - all three VoiceGraph singletons (voice_graph, realtime, low_latency)
    #   - LLM node connection pool close
    #   - QA audit bus task close
    # Calling voice_graph.shutdown() directly would only tear down the one
    # singleton and skip the audit bus flush — use on_shutdown() instead.
    try:
        asyncio.run_coroutine_threadsafe(
            _voice_graph_shutdown(),
            _loop,
        ).result(timeout=25.0)
        log.info("controller_voice_graph_shutdown")
    except Exception as exc:
        log.warning("controller_voice_graph_shutdown_failed", error=str(exc))

    _loop.call_soon_threadsafe(_loop.stop)  # type: ignore[arg-type]
    _loop_thread.join(timeout=LOOP_JOIN_TIMEOUT)

    log.info("controller_shutdown_complete")


# ── signal handlers ────────────────────────────────────────────────────────────


def _install_signal_handlers() -> None:
    """Route SIGTERM and SIGINT through the same shutdown path as ESC."""

    def _handle(sig: int, _frame: object) -> None:
        signame = signal.Signals(sig).name
        log.warning("controller_signal_received", signal=signame)
        # Emit structured signal event so dashboards can distinguish user-ESC
        # exits from OS-initiated shutdowns (systemd SIGTERM, Ctrl-C SIGINT).
        ControllerEmitter.signal(signum=sig, signame=signame)
        _shutdown(source=f"signal:{signame}")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)


# ── health pre-flight ──────────────────────────────────────────────────────────


def _check_pipeline_health() -> bool:
    """
    Poll the voice graph's health endpoint and log the result.

    Called once after on_startup() completes and before the poll loop starts.
    Returns True if all three nodes report healthy, False otherwise.

    The controller never blocks on this — an unhealthy node at startup is a
    warning, not a hard stop, because circuit breakers may recover within the
    first few seconds of traffic. Callers should log a warning and continue.
    """
    try:
        health = asyncio.run_coroutine_threadsafe(
            voice_graph.health(),
            _loop,
        ).result(timeout=10.0)

        if health["healthy"]:
            log.info(
                "controller_health_ok",
                version=health.get("version"),
                inflight=health.get("inflight", 0),
            )
        else:
            # Log per-node details so the operator knows which node is unhealthy.
            unhealthy = [
                name
                for name, node_h in health.get("nodes", {}).items()
                if not node_h.get("healthy")
            ]
            log.warning(
                "controller_health_degraded",
                unhealthy_nodes=unhealthy,
                version=health.get("version"),
            )

        return bool(health["healthy"])

    except Exception as exc:
        log.warning("controller_health_check_failed", error=str(exc))
        return False


# ── input loop ─────────────────────────────────────────────────────────────────


def listen_loop() -> None:
    """
    Main desktop input loop — runs on the main thread, never blocks.

    Polling at POLL_INTERVAL_S (~33 Hz default) keeps CPU negligible while
    still feeling instantaneous to the user. All heavy work runs on the
    daemon thread or the shared async loop.

    Keybindings (override via env or settings)
    ──────────────────────────────────────────
      PTT_KEY  (default: H)    hold-to-talk
      EXIT_KEY (default: ESC)  clean exit
    """
    global _loop_thread, _session_id

    # Boot observability before the loop thread starts so that OTel providers,
    # Prometheus collectors, and the session-context ContextVar are all
    # initialised before any emit() or span call happens on the loop thread.
    _obs_bootstrap(
        start_prometheus=False,
        start_mongo=False,
        write_grafana=False,
        start_otel=True,
    )

    _loop_thread = threading.Thread(target=_run_loop, daemon=True, name="voice-loop")
    _loop_thread.start()

    _install_signal_handlers()

    # ── pipeline startup ───────────────────────────────────────────────────────
    # on_startup() warms the LLM node (pre-loads model weights / connection
    # pool), starts the LLM background health probe, pre-initialises the QA
    # controller Redis pool, and bootstraps the QA audit bus drain task.
    # Must run before the boot display and session registration so that the
    # pipeline is ready to handle the first PTT press immediately after the
    # user sees the "ready" prompt.
    try:
        asyncio.run_coroutine_threadsafe(
            _voice_graph_startup(),
            _loop,
        ).result(timeout=20.0)
        log.info("controller_voice_graph_startup_ok")
    except Exception as exc:
        log.warning("controller_voice_graph_startup_failed", error=str(exc))

    # ── boot display ───────────────────────────────────────────────────────────
    # Shown before session registration so the user sees something immediately.
    # Skipped in verbose mode — JSON logs serve the same purpose there.
    show_boot_sequence(
        [
            {"label": "STT", "model": getattr(settings, "stt_model", "whisper-1")},
            {
                "label": "LLM",
                "model": getattr(settings, "llm_model", "gpt-5-mini"),
            },
            {
                "label": "TTS",
                "model": f"{getattr(settings, 'tts_model', 'tts-1')} / {getattr(settings, 'tts_voice', 'nova')}",
            },
            {"label": "Session Store", "model": "redis + lru"},
        ],
        ptt_key=PTT_KEY,
        exit_key=EXIT_KEY,
    )

    # ── register session ───────────────────────────────────────────────────────
    try:
        future = asyncio.run_coroutine_threadsafe(
            session_store.register(_local_ip),
            _loop,
        )
        result = future.result(timeout=10.0)
        _session_id = result["session_id"]

        # open transcript file for this session
        asyncio.run_coroutine_threadsafe(
            transcript_writer.open_session(_session_id),
            _loop,
        ).result(timeout=3.0)

        log.info(
            "controller_session_registered",
            session_id=_session_id,
            ip=result["registered_from_ip"],
            ttl_s=result["ttl_s"],
        )
    except Exception as exc:
        log.warning("controller_session_register_failed", error=str(exc))

    # ── health pre-flight ──────────────────────────────────────────────────────
    # Check all three nodes after on_startup() has had a chance to warm them.
    # Degraded nodes are a warning, not a hard stop — circuit breakers recover.
    _check_pipeline_health()

    log.info("controller_ready", ptt_key=PTT_KEY, exit_key=EXIT_KEY)

    try:
        while True:
            if keyboard.is_pressed(EXIT_KEY):
                _shutdown(source="esc")
                break

            if keyboard.is_pressed(PTT_KEY):
                _interrupt(source="ptt")
                time.sleep(INTERRUPT_DRAIN_S)
                _interrupt_flag.clear()

                ControllerEmitter.ptt_press(
                    session_id=_session_id or "",
                    ptt_key=PTT_KEY,
                )
                log.info("controller_recording_start")

                t_start = time.monotonic()
                audio_path = record_audio_until_released(
                    lambda: keyboard.is_pressed(PTT_KEY)
                )
                duration = time.monotonic() - t_start
                _recording_duration.observe(duration)

                if audio_path:
                    _sessions_started.inc()
                    ControllerEmitter.ptt_release(
                        session_id=_session_id or "",
                        ptt_key=PTT_KEY,
                        recording_s=round(duration, 2),
                    )
                    log.info(
                        "controller_recording_done",
                        duration_s=round(duration, 2),
                        audio_path=audio_path,
                    )
                    _dispatch(audio_path)
                else:
                    _empty_recordings.inc()
                    ControllerEmitter.empty_recording(
                        session_id=_session_id or "",
                        recording_s=round(duration, 2),
                    )
                    log.warning(
                        "controller_recording_empty", duration_s=round(duration, 2)
                    )

                time.sleep(DEBOUNCE_S)

            time.sleep(POLL_INTERVAL_S)

    finally:
        # Only reached on an unexpected crash — clean shutdown already
        # handled session teardown inside _shutdown() before stopping the loop.
        if _session_id and _loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(
                    session_store.end(_session_id, reason="crash"),
                    _loop,
                ).result(timeout=3.0)
            except Exception:  # noqa
                pass


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    listen_loop()