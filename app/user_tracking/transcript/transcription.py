"""
transcription.py — Dual-sink transcript writer for the interview pipeline.

Every completed turn is written to two independent sinks:

  Sink A — Human-readable .txt transcript (per session)
  ──────────────────────────────────────────────────────
  One file per session, named by session_id, written to TRANSCRIPT_DIR.
  Format:

      [10:17:02] [d056d716]  You said something here.
      [10:17:10] [AI]        I responded with something here.

  Simple, grep-able, shareable with candidates for review.

  Sink B — Structured observability (via observability.emit())
  ─────────────────────────────────────────────────────────────
  All turn, session-open, and session-close events are emitted through
  observability.emit(), which fans out to:
    • structlog  → Rich console + JSON file (LOG_FILE env var)
    • Prometheus → ai_transcript_turns_total / ai_transcript_drops_total
    • MongoDB    → full structured document per event
    • OTel       → span annotation on the active span

  This replaces the former hand-rolled _write_json_line sink and makes
  transcript events first-class citizens in the observability stack,
  searchable alongside all other pipeline events in Datadog, Loki,
  CloudWatch, or any OTLP-compatible backend.

Both sinks are async and non-blocking. Writes go through an asyncio.Queue
so no pipeline call ever waits on disk I/O. A background task drains the
queue. If the queue is full (TRANSCRIPT_QUEUE_DEPTH), older entries are
dropped and a counter is incremented — the pipeline is never back-pressured.

Integration
───────────
  In voice_graph.py, after conversation_memory.commit():

      from app.user_tracking.transcript.transcription import transcript_writer

      await transcript_writer.write_turn(
          session_id=state.get("session_id"),
          user_text=state["transcript"],
          assistant_text=state["llm_response"],
          request_id=state["request_id"],
      )

  Or from controller.py after _dispatch() completes if you want it
  at the controller layer instead:

      await transcript_writer.write_turn(...)

  Call transcript_writer.open_session(session_id) on session start and
  transcript_writer.close_session(session_id) on session end to add
  header/footer lines to the .txt file.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.monitoring.observability import (
    EventKind,
    ObsEvent,
    TranscriptEmitter,
    emit,
    get_logger,
)

log = get_logger(__name__)

# ── config ────────────────────────────────────────────────────────────────────

TRANSCRIPT_DIR: str = os.getenv("TRANSCRIPT_DIR", "transcripts")
TRANSCRIPT_QUEUE_DEPTH: int = int(os.getenv("TRANSCRIPT_QUEUE_DEPTH", "512"))


# ── internal write entry ──────────────────────────────────────────────────────


@dataclass
class _TurnEntry:
    kind: Literal["turn", "open", "close"]
    session_id: str
    user_text: str = ""
    assistant_text: str = ""
    request_id: str = ""
    ts: float = 0.0

    def __post_init__(self):
        if not self.ts:
            self.ts = time.time()


# ── file-level helpers ────────────────────────────────────────────────────────

_file_lock = threading.Lock()


def _txt_path(session_id: str) -> Path:
    sid_short = session_id[:16] if len(session_id) > 16 else session_id
    return Path(TRANSCRIPT_DIR) / f"session_{sid_short}.txt"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S")


def _fmt_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _sid_label(session_id: str) -> str:
    """Short 8-char label for display in the .txt file."""
    return session_id[:8] if session_id else "unknown"


def _write_txt_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ── transcript writer ─────────────────────────────────────────────────────────


class TranscriptWriter:
    """
    Async dual-sink transcript writer.

    All public methods are fire-and-forget coroutines — they enqueue the
    write and return immediately. A single background task drains the queue
    and does the actual I/O so the pipeline never blocks on disk.

    The writer starts itself automatically on first use via _ensure_started().
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_TurnEntry] = asyncio.Queue(
            maxsize=TRANSCRIPT_QUEUE_DEPTH
        )
        self._task: asyncio.Task | None = None
        self._started: bool = False
        self._lock = asyncio.Lock()

    # ── public API ─────────────────────────────────────────────────────────────

    async def open_session(self, session_id: str) -> None:
        """
        Write a session-start header to the .txt transcript.
        Call once when a new session is registered.
        """
        await self._ensure_started()
        entry = _TurnEntry(kind="open", session_id=session_id, ts=time.time())
        await self._enqueue(entry)

    async def close_session(self, session_id: str) -> None:
        """
        Write a session-end footer to the .txt transcript.
        Call from controller._shutdown() or session_store.end().
        """
        await self._ensure_started()
        entry = _TurnEntry(kind="close", session_id=session_id, ts=time.time())
        await self._enqueue(entry)

    async def write_turn(
        self,
        *,
        session_id: str | None,
        user_text: str,
        assistant_text: str,
        request_id: str = "",
    ) -> None:
        """
        Enqueue a completed (user, assistant) turn for writing.

        Both sinks receive the full turn. Never blocks or raises.
        """
        if not session_id:
            return

        await self._ensure_started()
        entry = _TurnEntry(
            kind="turn",
            session_id=session_id,
            user_text=user_text,
            assistant_text=assistant_text,
            request_id=request_id,
            ts=time.time(),
        )
        await self._enqueue(entry)

    async def flush(self, timeout: float = 5.0) -> None:
        """
        Wait for the queue to drain. Call during graceful shutdown.
        """
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("transcript_flush_timeout", remaining=self._queue.qsize())

    # ── internals ──────────────────────────────────────────────────────────────

    async def _ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if not self._started:
                self._task = asyncio.ensure_future(self._drain_loop())
                self._started = True

    async def _enqueue(self, entry: _TurnEntry) -> None:
        try:
            self._queue.put_nowait(entry)
        except asyncio.QueueFull:
            # Delegates counter increment + structured warning log to observability.
            TranscriptEmitter.queue_drop(session_id=entry.session_id)

    async def _drain_loop(self) -> None:
        """Background task — drains the queue and dispatches to sinks."""
        while True:
            try:
                entry = await self._queue.get()
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._process, entry
                    )
                except Exception as exc:
                    log.error("transcript_drain_error", error=str(exc))
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("transcript_drain_fatal", error=str(exc))

    def _process(self, entry: _TurnEntry) -> None:
        """
        Synchronous I/O — runs in a thread-pool executor.
        Dispatches to both sinks based on entry kind.
        """
        if entry.kind == "open":
            self._handle_open(entry)
        elif entry.kind == "close":
            self._handle_close(entry)
        else:
            self._handle_turn(entry)

    def _handle_open(self, e: _TurnEntry) -> None:  # noqa
        # ── Sink A: .txt ───────────────────────────────────────────────────────
        path = _txt_path(e.session_id)
        label = _sid_label(e.session_id)
        header = (
            f"{'─' * 60}\n"
            f"Session  : {e.session_id}\n"
            f"Started  : {_fmt_date(e.ts)}\n"
            f"{'─' * 60}"
        )
        try:
            _write_txt_line(path, header)
        except Exception as exc:
            log.warning("transcript_open_txt_failed", session_id=label, error=str(exc))

        # ── Sink B: observability ──────────────────────────────────────────────
        emit(
            ObsEvent(
                kind="transcript_session_open",
                service="transcript",
                session_id=e.session_id,
                ts=e.ts,
            )
        )

    def _handle_close(self, e: _TurnEntry) -> None:  # noqa
        # ── Sink A: .txt ───────────────────────────────────────────────────────
        path = _txt_path(e.session_id)
        label = _sid_label(e.session_id)
        footer = f"{'─' * 60}\n" f"Session ended : {_fmt_date(e.ts)}\n" f"{'─' * 60}"
        try:
            _write_txt_line(path, footer)
        except Exception as exc:
            log.warning("transcript_close_txt_failed", session_id=label, error=str(exc))

        # ── Sink B: observability ──────────────────────────────────────────────
        emit(
            ObsEvent(
                kind="transcript_session_close",
                service="transcript",
                session_id=e.session_id,
                ts=e.ts,
            )
        )

    def _handle_turn(self, e: _TurnEntry) -> None:  # noqa
        """
        Write one turn to both sinks.

        .txt format:
            [HH:MM:SS] [sid_short]   <user text>
            [HH:MM:SS] [AI]          <assistant text>

        Observability: one TRANSCRIPT_TURN ObsEvent carrying user text in
        `transcript` and assistant text in `extra`, fanned out by emit() to
        structlog (JSON file), Prometheus, MongoDB, and OTel simultaneously.
        The Prometheus counter (ai_transcript_turns_total) is incremented
        automatically by emit() via _update_prometheus.
        """
        path = _txt_path(e.session_id)
        label = _sid_label(e.session_id)
        ts = _fmt_time(e.ts)

        # ── Sink A: .txt ───────────────────────────────────────────────────────
        try:
            user_line = f"[{ts}] [{label}]   {e.user_text}"
            ai_line = f"[{ts}] [AI]         {e.assistant_text}"
            _write_txt_line(path, user_line)
            _write_txt_line(path, ai_line)
            _write_txt_line(path, "")  # blank line between turns
        except Exception as exc:
            log.warning("transcript_txt_write_failed", session_id=label, error=str(exc))

        # ── Sink B: observability ──────────────────────────────────────────────
        # A single TRANSCRIPT_TURN event carries both speaker texts and increments
        # the Prometheus counter. User text maps to the standard `transcript` field;
        # assistant text goes into `extra` so it round-trips cleanly through MongoDB.
        emit(
            ObsEvent(
                kind=EventKind.TRANSCRIPT_TURN,
                service="transcript",
                session_id=e.session_id,
                request_id=e.request_id,
                ts=e.ts,
                transcript=e.user_text,
                transcript_chars=len(e.user_text),
                extra={"assistant_text": e.assistant_text},
            )
        )


# ── module-level singleton ────────────────────────────────────────────────────

transcript_writer = TranscriptWriter()
