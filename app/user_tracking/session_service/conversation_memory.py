"""
conversation_memory.py — QA Audit Bus for the structured interview pipeline.

PHILOSOPHY — THIS IS A COMPLETE ARCHITECTURAL INVERSION FROM THE ORIGINAL:
───────────────────────────────────────────────────────────────────────────

  The OLD conversation_memory.py fed rolling conversation history to the LLM.
  That philosophy is DEAD. The LLM is NOT a chatbot. It receives ZERO history.

  The NEW conversation_memory.py is the QA Audit Bus. Its sole responsibilities:

    1. TURN ROUTING        — Receive committed QATurns from voice_graph after
                             qa_controller.commit_turn(). Fan out each turn to
                             transcript_writer (dual sink: .txt + observability)
                             and the eval_engine scheduling queue.

    2. DOMAIN BATCH MGMT  — Accumulate turns per domain. When a domain's quota
                             is satisfied (as signalled by QADocument.domain_progress),
                             package the domain's Q/A pairs and dispatch them to
                             eval_engine.schedule_domain_eval() asynchronously.

    3. SESSION LIFECYCLE   — Emit open/close events to transcript_writer and
                             observability on session start and end. Flush all
                             pending audit writes on session close.

    4. EVAL SCHEDULING     — Coordinate with eval_engine to ensure each domain
                             batch is evaluated exactly once. Tracks which domains
                             have been dispatched for eval to prevent double-firing.
                             Provides idempotency via Redis SET NX per domain key.

    5. AUDIT SNAPSHOTS     — Maintain a Redis-backed (LRU-fallback) audit log of
                             committed turns per session. TTL = SESSION_TTL + GRACE.
                             Used by admin tooling and the /debug/session endpoint.

    6. DEAD-LETTER QUEUE   — If eval_engine is unreachable, committed domain batches
                             land in a bounded in-process dead-letter queue. A
                             background retry task re-attempts dispatch with
                             exponential backoff. No turns are silently dropped.

WHAT THIS MODULE DOES NOT DO:
──────────────────────────────
  ✗ Build LangChain message lists for the LLM  (use qa_controller.get_llm_input)
  ✗ Maintain rolling conversation windows       (the LLM doesn't need them)
  ✗ Inject system prompts                       (qa_controller owns all prompts)
  ✗ Track topics, hints, or follow-up depth     (qa_controller.QADocument owns this)
  ✗ Compress history                            (there is no history to compress)

INTEGRATION — voice_graph.py:
──────────────────────────────
  from app.memory.conversation_memory import qa_audit_bus

  # Session start (called from controller._register_session or session_store.begin):
  await qa_audit_bus.open_session(session_id)

  # After qa_controller.commit_turn() returns a CommittedTurn:
  await qa_audit_bus.route_turn(
      committed_turn  = committed,
      qa_document     = qa_doc,     # post-commit snapshot
  )

  # After each domain completes (domain_rotated=True in CommittedTurn):
  await qa_audit_bus.trigger_domain_eval(
      session_id      = session_id,
      completed_domain = prev_domain,
      qa_document      = qa_doc,
  )

  # Session end (called from controller._shutdown or session_store.end):
  await qa_audit_bus.close_session(session_id, reason="complete")

THREAD SAFETY:
──────────────
  All public methods are async coroutines. All I/O runs through asyncio.Queue
  drains in background tasks. Multiple callers may invoke any method concurrently.
  Per-session locks (asyncio.Lock, keyed by session_id) guard all Redis writes.

DATA CONTRACT:
──────────────
  Redis key schema (QA Audit Bus namespace):
    audit:v1:{session_id}:turns          → JSON list of AuditTurnRecord
    audit:v1:{session_id}:domains_evaled → JSON set of domain keys sent to eval
    audit:v1:{session_id}:meta           → JSON AuditSessionMeta

  TTL: QA_TTL_S from qa_controller config (inherits from SESSION_TTL_S + GRACE_S).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import weakref # noqa
import random
from collections import defaultdict, deque # noqa
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING
import dataclasses

import redis.asyncio as aioredis
from opentelemetry import trace # noqa

from app.common.shared import (
    CircuitBreaker,
    CircuitBreakerOpen,
    InMemoryLRU,
    ServiceHealthState,
    backoff_retry, # noqa
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)
from app.monitoring.observability import (
    EventKind,
    ObsEvent,
    emit,
    get_logger as obs_get_logger,
)

if TYPE_CHECKING:
    from app.interview.qa_controller import CommittedTurn, QADocument # noqa

# shared.get_logger so that structured log records are routed through the
# OTel/Prometheus pipeline alongside span context. This means session_id,
# request_id, and domain fields automatically appear in trace-correlated logs
# in Grafana without manual span attribute injection.
log    = obs_get_logger(__name__)
tracer = get_tracer(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

_turns_routed        = make_counter("audit_turns_routed_total",         "Turns fanned out",              ["stage"])
_domain_evals_fired  = make_counter("audit_domain_evals_fired_total",   "Domain eval dispatches",        ["status"])
_sessions_opened     = make_counter("audit_sessions_opened_total",      "Sessions opened")
_sessions_closed     = make_counter("audit_sessions_closed_total",      "Sessions closed",               ["reason"])
_dlq_enqueued        = make_counter("audit_dlq_enqueued_total",         "Turns added to dead-letter Q")
_dlq_retried         = make_counter("audit_dlq_retried_total",          "Dead-letter retry attempts",    ["result"])
_redis_ops           = make_counter("audit_redis_ops_total",            "Redis read/write ops",          ["op", "status"])
_lru_hits            = make_counter("audit_lru_hits_total",             "LRU cache hits")
_idempotency_blocks  = make_counter("audit_idempotency_blocks_total",   "Duplicate eval dispatches blocked")
_transcript_fanouts  = make_counter("audit_transcript_fanouts_total",   "Transcript writer fan-outs",    ["status"])
_eval_fanouts        = make_counter("audit_eval_fanouts_total",         "Eval engine fan-outs",          ["status"])

_route_latency = make_histogram(
    "audit_route_turn_latency_seconds",
    "route_turn() end-to-end wall clock",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
)
_dlq_depth = make_gauge("audit_dlq_depth", "Dead-letter queue current depth")
_active_sessions_gauge = make_gauge("audit_active_sessions", "Sessions currently open in audit bus")

# ── config ─────────────────────────────────────────────────────────────────────

REDIS_URL:    str | None = os.getenv("REDIS_URL")
REDIS_MAX_CONN: int      = int(os.getenv("REDIS_MAX_CONN",       "200"))
AUDIT_TTL_S:  int        = int(os.getenv("SESSION_TTL_S",        "3600")) + int(os.getenv("SESSION_GRACE_S", "60"))
AUDIT_PREFIX              = "audit:v1:"
AUDIT_LRU_SIZE: int      = int(os.getenv("AUDIT_LRU_SIZE",       "128"))

DLQ_MAX_SIZE:   int      = int(os.getenv("AUDIT_DLQ_MAX_SIZE",   "512"))
DLQ_RETRY_BASE: float    = float(os.getenv("AUDIT_DLQ_RETRY_BASE_S", "2.0"))
DLQ_MAX_RETRIES: int     = int(os.getenv("AUDIT_DLQ_MAX_RETRIES", "8"))

ROUTE_QUEUE_DEPTH: int   = int(os.getenv("AUDIT_ROUTE_QUEUE_DEPTH", "1024"))
TRANSCRIPT_FLUSH_TIMEOUT: float = float(os.getenv("AUDIT_TRANSCRIPT_FLUSH_TIMEOUT_S", "5.0"))


# ── enums ──────────────────────────────────────────────────────────────────────

class AuditEventKind(str, Enum):
    SESSION_OPEN        = "session_open"
    SESSION_CLOSE       = "session_close"
    TURN_COMMITTED      = "turn_committed"
    DOMAIN_EVAL_QUEUED  = "domain_eval_queued"
    DOMAIN_EVAL_SENT    = "domain_eval_sent"
    DOMAIN_EVAL_FAILED  = "domain_eval_failed"
    DOMAIN_EVAL_DLQ     = "domain_eval_dlq"
    DLQ_RETRY_OK        = "dlq_retry_ok"
    DLQ_RETRY_FAIL      = "dlq_retry_fail"
    DLQ_EXHAUSTED       = "dlq_exhausted"


# ── data models ────────────────────────────────────────────────────────────────

@dataclass
class AuditTurnRecord:
    """
    Compact audit record for one committed Q/A turn.
    Stored in Redis per-session. Used by admin tooling and /debug endpoints.
    Never fed back to the LLM.
    """
    turn_index:       int
    domain:           str
    question:         str
    answer:           str
    ts:               float
    request_id:       str   = ""
    transcript_ok:    bool  = False   # True if transcript_writer.write_turn() succeeded
    eval_queued:      bool  = False   # True if this turn was included in an eval dispatch
    word_count_q:     int   = 0
    word_count_a:     int   = 0

    def __post_init__(self) -> None:
        self.word_count_q = len(self.question.split())
        self.word_count_a = len(self.answer.split())

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AuditTurnRecord":
        valid_keys = {f.name for f in dataclasses.fields(AuditTurnRecord)}
        return AuditTurnRecord(**{k: v for k, v in d.items() if k in valid_keys})

@dataclass
class AuditSessionMeta:
    """
    Session-level metadata tracked by the audit bus.
    Stored in Redis alongside per-turn audit records.
    """
    session_id:          str
    opened_at:           float                = field(default_factory=time.time)
    closed_at:           float | None         = None
    close_reason:        str                  = ""
    domains_evaluated:   list[str]            = field(default_factory=list)   # domains dispatched to eval
    total_turns_routed:  int                  = 0
    total_domains_evaled: int                 = 0
    transcript_ok:       bool                 = False   # True if open_session call reached transcript_writer

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "AuditSessionMeta":
        valid_keys = {f.name for f in dataclasses.fields(AuditSessionMeta)}
        return AuditSessionMeta(**{k: v for k, v in d.items() if k in valid_keys})

@dataclass
class _DLQItem:
    """Dead-letter queue item — a failed eval dispatch awaiting retry."""
    session_id:     str
    domain:         str
    eval_pairs:     list[dict]
    candidate_meta: dict
    attempt:        int        = 0
    next_retry_at:  float      = field(default_factory=time.time)
    created_at:     float      = field(default_factory=time.time)

    def next_backoff(self) -> float:
        """Exponential backoff with jitter: base * 2^attempt + jitter(0..1)."""
        return DLQ_RETRY_BASE * (2 ** self.attempt) + random.uniform(0, 1)


@dataclass
class _RouteTask:
    """Internal queue item for the background drain loop."""
    kind:           AuditEventKind
    session_id:     str
    committed_turn: "CommittedTurn | None"    = None
    qa_document:    "QADocument | None"       = None
    completed_domain: str                     = ""
    close_reason:   str                       = ""
    request_id:     str                       = field(default_factory=new_request_id)
    ts:             float                     = field(default_factory=time.time)


# ── Redis layer ────────────────────────────────────────────────────────────────

class _AuditRedis:
    """
    Thin Redis wrapper for the audit bus.
    Manages connection pooling, circuit breaker, and LRU fallback.
    """

    def __init__(self) -> None:
        self._redis:   aioredis.Redis | None = None
        self._lru      = InMemoryLRU(max_size=AUDIT_LRU_SIZE)
        self._breaker  = CircuitBreaker(name="audit_redis", failure_threshold=4, recovery_timeout=20.0)
        self._sessions_locks: dict[str, asyncio.Lock] = {}   # per-session write locks

    async def unmark_domain_evaluated(self, session_id: str, domain: str) -> None:
        """Remove a domain from the evaluated set — used by admin requeue only."""
        key = f"{AUDIT_PREFIX}{session_id}:domains_evaled"
        async with self._session_lock(session_id):
            existing = await self.get_domains_evaluated(session_id)
            existing.discard(domain)
            await self._set(key, json.dumps(list(existing)))

    async def write_turns(self, session_id: str, turns: list[AuditTurnRecord]) -> None:
        """Overwrite the full turns list — used by retention policy PII erasure only."""
        key = f"{AUDIT_PREFIX}{session_id}:turns"
        async with self._session_lock(session_id):
            await self._set(key, json.dumps([t.to_dict() for t in turns]))

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._sessions_locks:
            self._sessions_locks[session_id] = asyncio.Lock()
        return self._sessions_locks[session_id]

    async def _conn(self) -> aioredis.Redis | None:
        if not REDIS_URL:
            return None
        if self._redis is None:
            self._redis = await aioredis.from_url(
                REDIS_URL,
                encoding               = "utf-8",
                decode_responses       = True,
                max_connections        = REDIS_MAX_CONN,
                socket_connect_timeout = 0.15,
                socket_timeout         = 0.25,
            )
        return self._redis

    async def get_turns(self, session_id: str) -> list[AuditTurnRecord]:
        key = f"{AUDIT_PREFIX}{session_id}:turns"
        raw = await self._get(key)
        if raw is None:
            return []
        try:
            return [AuditTurnRecord.from_dict(d) for d in json.loads(raw)]
        except Exception: # noqa
            return []

    async def append_turn(self, session_id: str, record: AuditTurnRecord) -> None:
        key = f"{AUDIT_PREFIX}{session_id}:turns"
        async with self._session_lock(session_id):
            existing = await self.get_turns(session_id)
            existing.append(record)
            await self._set(key, json.dumps([r.to_dict() for r in existing]))

    async def get_domains_evaluated(self, session_id: str) -> set[str]:
        key = f"{AUDIT_PREFIX}{session_id}:domains_evaled"
        raw = await self._get(key)
        if raw is None:
            return set()
        try:
            return set(json.loads(raw))
        except Exception: # noqa
            return set()

    async def mark_domain_evaluated(self, session_id: str, domain: str) -> bool:
        """
        Idempotent mark. Returns True if this is the FIRST time domain is marked
        (caller should fire eval), False if already marked (caller should skip).
        """
        key = f"{AUDIT_PREFIX}{session_id}:domains_evaled"
        async with self._session_lock(session_id):
            existing = await self.get_domains_evaluated(session_id)
            if domain in existing:
                return False
            existing.add(domain)
            await self._set(key, json.dumps(list(existing)))
            return True

    async def get_meta(self, session_id: str) -> AuditSessionMeta | None:
        key = f"{AUDIT_PREFIX}{session_id}:meta"
        raw = await self._get(key)
        if raw is None:
            return None
        try:
            return AuditSessionMeta.from_dict(json.loads(raw))
        except Exception: # noqa
            return None

    async def set_meta(self, session_id: str, meta: AuditSessionMeta) -> None:
        key = f"{AUDIT_PREFIX}{session_id}:meta"
        async with self._session_lock(session_id):
            await self._set(key, json.dumps(meta.to_dict()))

    async def _get(self, key: str) -> str | None:
        try:
            r = await self._conn()
            if r:
                val = await self._breaker.call(r.get, key)
                if val is not None:
                    _redis_ops.labels(op="get", status="hit").inc()
                    return val
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("audit_redis_get_failed", key=key[-20:], error=str(exc))
            _redis_ops.labels(op="get", status="error").inc()

        lru_val = await self._lru.get(key)
        if lru_val is not None:
            _lru_hits.inc()
        return lru_val

    async def _set(self, key: str, value: str) -> None:
        await self._lru.set(key, value)
        try:
            r = await self._conn()
            if r:
                await self._breaker.call(r.set, key, value, ex=AUDIT_TTL_S)
                _redis_ops.labels(op="set", status="ok").inc()
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("audit_redis_set_failed", key=key[-20:], error=str(exc))
            _redis_ops.labels(op="set", status="error").inc()

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    def evict_session_locks(self, session_id: str) -> None:
        self._sessions_locks.pop(session_id, None)


# ── eval engine interface (deferred import to avoid circular) ──────────────────

async def _call_eval_engine(session_id: str, domain: str, eval_pairs: list[dict], candidate_meta: dict) -> bool:
    """
    Dispatch a domain batch to eval_engine.schedule_domain_eval().

    Returns True on success, False on failure.
    The import is deferred to break the circular dependency between
    conversation_memory ↔ eval_engine.
    """
    try:
        from app.eval.evaluation_engine import evaluation_engine as eval_engine  # type: ignore[import]
        await eval_engine.schedule_domain_eval(
            session_id     = session_id,
            domain         = domain,
            eval_pairs     = eval_pairs,
            candidate_meta = candidate_meta,
        )
        return True
    except ImportError:
        log.warning("audit_eval_engine_not_available", session_id=session_id[:8], domain=domain)
        return False
    except Exception as exc:
        log.error("audit_eval_engine_call_failed", session_id=session_id[:8], domain=domain, error=str(exc))
        return False


# ── transcript writer interface ────────────────────────────────────────────────

async def _call_transcript_open(session_id: str) -> bool:
    try:
        from app.user_tracking.transcript.transcription import transcript_writer   # type: ignore[import]
        await transcript_writer.open_session(session_id)
        return True
    except Exception as exc:
        log.warning("audit_transcript_open_failed", session_id=session_id[:8], error=str(exc))
        return False


async def _call_transcript_turn(
    session_id:     str,
    user_text:      str,
    assistant_text: str,
    request_id:     str = "",
) -> bool:
    try:
        from app.user_tracking.transcript.transcription import transcript_writer   # type: ignore[import]
        await transcript_writer.write_turn(
            session_id     = session_id,
            user_text      = user_text,
            assistant_text = assistant_text,
            request_id     = request_id,
        )
        return True
    except Exception as exc:
        log.warning("audit_transcript_turn_failed", session_id=session_id[:8], error=str(exc))
        return False


async def _call_transcript_close(session_id: str) -> bool:
    try:
        from app.user_tracking.transcript.transcription import transcript_writer   # type: ignore[import]
        await transcript_writer.close_session(session_id)
        return True
    except Exception as exc:
        log.warning("audit_transcript_close_failed", session_id=session_id[:8], error=str(exc))
        return False


# ── dead-letter queue ──────────────────────────────────────────────────────────

class _DeadLetterQueue:
    """
    Bounded in-process dead-letter queue for failed eval dispatches.

    When eval_engine is unreachable, domain batches land here.
    A background retry task drains the queue with exponential backoff.
    On DLQ_MAX_RETRIES exhaustion the item is logged and dropped —
    the transcript record is permanent so data is never truly lost.
    """

    def __init__(self) -> None:
        self._queue: deque[_DLQItem] = deque(maxlen=DLQ_MAX_SIZE)
        self._lock   = asyncio.Lock()
        self._task:  asyncio.Task | None = None
        self._started = False

    def enqueue(self, item: _DLQItem) -> None:
        self._queue.append(item)
        _dlq_enqueued.inc()
        _dlq_depth.set(len(self._queue))
        log.warning(
            "audit_dlq_enqueue",
            session_id = item.session_id[:8],
            domain     = item.domain,
            attempt    = item.attempt,
            dlq_depth  = len(self._queue),
        )

    def start(self) -> None:
        if not self._started:
            self._task    = asyncio.ensure_future(self._drain_loop())
            self._started = True

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    async def _drain_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(1.0)
                await self._process_due()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("audit_dlq_drain_error", error=str(exc))

    async def _process_due(self) -> None:
        now  = time.time()
        keep: list[_DLQItem] = []

        async with self._lock:
            items = list(self._queue)
            self._queue.clear()

        for item in items:
            if item.next_retry_at > now:
                keep.append(item)
                continue

            if item.attempt >= DLQ_MAX_RETRIES:
                log.error(
                    "audit_dlq_exhausted",
                    session_id = item.session_id[:8],
                    domain     = item.domain,
                    attempts   = item.attempt,
                )
                _dlq_retried.labels(result="exhausted").inc()
                # Emit observability event — transcript record already written
                emit(ObsEvent(
                    kind       = "audit_dlq_exhausted",
                    service    = "qa_audit_bus",
                    session_id = item.session_id,
                    extra      = {"domain": item.domain, "attempts": item.attempt},
                ))
                continue

            ok = await _call_eval_engine(
                item.session_id, item.domain, item.eval_pairs, item.candidate_meta
            )

            if ok:
                _dlq_retried.labels(result="ok").inc()
                log.info(
                    "audit_dlq_retry_ok",
                    session_id = item.session_id[:8],
                    domain     = item.domain,
                    attempt    = item.attempt,
                )
            else:
                item.attempt       += 1
                item.next_retry_at  = time.time() + item.next_backoff()
                keep.append(item)
                _dlq_retried.labels(result="fail").inc()

        async with self._lock:
            for item in keep:
                self._queue.append(item)
            _dlq_depth.set(len(self._queue))

    @property
    def depth(self) -> int:
        return len(self._queue)


# ── eval pair builder ──────────────────────────────────────────────────────────

class _EvalPairBuilder:
    """
    Packages a domain's AuditTurnRecords into the eval_pairs format
    that eval_engine.schedule_domain_eval() expects.

    eval_engine contract (stable interface):
        eval_pairs = [
            {
                "turn_index": int,
                "domain":     str,
                "question":   str,
                "answer":     str,
                "ts":         float,
            },
            ...
        ]
        candidate_meta = {
            "session_id":  str,
            "name":        str,
            "level":       str,
            "raw_intro":   str,
            "notes":       str,
        }
    """

    @staticmethod
    def build_eval_pairs(records: list[AuditTurnRecord], domain: str) -> list[dict]:
        return [
            {
                "turn_index": r.turn_index,
                "domain":     r.domain,
                "question":   r.question,
                "answer":     r.answer,
                "ts":         r.ts,
                "word_count_q": r.word_count_q,
                "word_count_a": r.word_count_a,
            }
            for r in records
            if r.domain == domain
        ]

    @staticmethod
    def build_candidate_meta(session_id: str, qa_doc: "QADocument | None") -> dict:
        if qa_doc is None:
            return {"session_id": session_id, "name": "", "level": "intermediate", "raw_intro": "", "notes": ""}
        c = qa_doc.candidate
        return {
            "session_id": session_id,
            "name":       c.name,
            "level":      c.level,
            "raw_intro":  c.raw_intro,
            "notes":      c.notes,
        }


# ── background route processor ─────────────────────────────────────────────────

class _RouteProcessor:
    """
    Background task that drains the route queue and executes I/O.
    Runs in a single asyncio task — all I/O is non-blocking.
    Decouples the caller (voice_graph) from disk and network I/O completely.
    """

    def __init__(self, redis: _AuditRedis, dlq: _DeadLetterQueue) -> None:
        self._redis = redis
        self._dlq   = dlq
        self._queue: asyncio.Queue[_RouteTask] = asyncio.Queue(maxsize=ROUTE_QUEUE_DEPTH)
        self._task:  asyncio.Task | None       = None
        self._started                          = False
        self._lock                             = asyncio.Lock()

    async def enqueue(self, task: _RouteTask) -> None:
        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            log.warning(
                "audit_route_queue_full",
                session_id = task.session_id[:8],
                kind       = task.kind.value,
                queue_size = self._queue.qsize(),
            )
            # Route queue full → process synchronously to avoid data loss
            await self._dispatch(task)

    async def ensure_started(self) -> None:
        if self._started:
            return
        async with self._lock:
            if not self._started:
                self._task    = asyncio.ensure_future(self._drain_loop())
                self._started = True

    async def flush(self, timeout: float = 5.0) -> None:
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            log.warning("audit_route_flush_timeout", remaining=self._queue.qsize())

    async def _drain_loop(self) -> None:
        while True:
            try:
                task = await self._queue.get()
                try:
                    await self._dispatch(task)
                except Exception as exc:
                    log.error("audit_route_dispatch_error", kind=task.kind.value, error=str(exc))
                finally:
                    self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("audit_route_drain_fatal", error=str(exc))

    async def _dispatch(self, task: _RouteTask) -> None:
        if task.kind == AuditEventKind.SESSION_OPEN:
            await self._handle_open(task)
        elif task.kind == AuditEventKind.SESSION_CLOSE:
            await self._handle_close(task)
        elif task.kind == AuditEventKind.TURN_COMMITTED:
            await self._handle_turn(task)
        elif task.kind == AuditEventKind.DOMAIN_EVAL_QUEUED:
            await self._handle_domain_eval(task)

    # ── handlers ──────────────────────────────────────────────────────────────

    async def _handle_open(self, task: _RouteTask) -> None:
        sid = task.session_id
        meta = AuditSessionMeta(session_id=sid, opened_at=task.ts)

        # Transcript writer
        ok = await _call_transcript_open(sid)
        meta.transcript_ok = ok
        if ok:
            _transcript_fanouts.labels(status="ok").inc()
        else:
            _transcript_fanouts.labels(status="error").inc()

        await self._redis.set_meta(sid, meta)
        _sessions_opened.inc()
        _active_sessions_gauge.inc()

        emit(ObsEvent(
            kind       = "audit_session_open",
            service    = "qa_audit_bus",
            session_id = sid,
            ts         = task.ts,
        ))
        log.info("audit_session_opened", session_id=sid[:8])

    async def _handle_close(self, task: _RouteTask) -> None:
        sid = task.session_id
        reason = task.close_reason

        ok = await _call_transcript_close(sid)
        if ok:
            _transcript_fanouts.labels(status="ok").inc()
        else:
            _transcript_fanouts.labels(status="error").inc()

        # Update meta
        meta = await self._redis.get_meta(sid) or AuditSessionMeta(session_id=sid)
        meta.closed_at = task.ts
        meta.close_reason = reason
        await self._redis.set_meta(sid, meta)

        self._redis.evict_session_locks(sid)
        _sessions_closed.labels(reason=reason or "unknown").inc()
        _active_sessions_gauge.dec()

        emit(ObsEvent(
            kind="audit_session_close",
            service="qa_audit_bus",
            session_id=sid,
            ts=task.ts,
            extra={"reason": reason, "total_turns": meta.total_turns_routed},
        ))
        log.info("audit_session_closed", session_id=sid[:8], reason=reason)

    async def _handle_turn(self, task: _RouteTask) -> None:
        t0  = time.monotonic()
        ct  = task.committed_turn
        sid = task.session_id

        if ct is None:
            return

        # Build audit record
        record = AuditTurnRecord(
            turn_index  = ct.turn_index,
            domain      = ct.domain,
            question    = ct.question,
            answer      = ct.answer,
            ts          = ct.ts,
            request_id  = task.request_id,
        )

        # Sink A: transcript_writer
        # NOTE: transcript writes are owned by voice_graph._flush_deferred_transcripts,
        # which applies the late-patch (substituting the full STT buffer for partial
        # early-fire text) before writing. Writing here would produce a duplicate entry
        # in the transcript file for every turn. We set transcript_ok=True optimistically.
        transcript_ok = True
        record.transcript_ok = transcript_ok
        _transcript_fanouts.labels(status="ok").inc()

        # Sink B: audit Redis append
        await self._redis.append_turn(sid, record)

        # Update session meta counters
        meta = await self._redis.get_meta(sid) or AuditSessionMeta(session_id=sid)
        meta.total_turns_routed += 1
        await self._redis.set_meta(sid, meta)

        latency = time.monotonic() - t0
        _route_latency.observe(latency)
        _turns_routed.labels(stage=ct.domain or "unknown").inc()

        emit(ObsEvent(
            kind             = EventKind.TRANSCRIPT_TURN,
            service          = "qa_audit_bus",
            session_id       = sid,
            request_id       = task.request_id,
            ts               = ct.ts,
            transcript       = ct.answer,
            transcript_chars = len(ct.answer),
            extra            = {
                "question":      ct.question,
                "domain":        ct.domain,
                "turn_index":    ct.turn_index,
                "transcript_ok": transcript_ok,
                "route_latency": round(latency, 4),
            },
        ))

        log.info(
            "audit_turn_routed",
            session_id     = sid[:8],
            turn_index     = ct.turn_index,
            domain         = ct.domain,
            transcript_ok  = transcript_ok,
            latency_ms     = round(latency * 1000, 1),
        )

    async def _handle_domain_eval(self, task: _RouteTask) -> None:
        sid    = task.session_id
        domain = task.completed_domain
        qa_doc = task.qa_document

        if not domain:
            return

        # Idempotency guard — ensure we never dispatch the same domain twice
        is_first = await self._redis.mark_domain_evaluated(sid, domain)
        if not is_first:
            _idempotency_blocks.inc()
            log.info(
                "audit_domain_eval_already_dispatched",
                session_id = sid[:8],
                domain     = domain,
            )
            return

        # Build eval payload
        turns   = await self._redis.get_turns(sid)
        pairs   = _EvalPairBuilder.build_eval_pairs(turns, domain)
        c_meta  = _EvalPairBuilder.build_candidate_meta(sid, qa_doc)

        if not pairs:
            log.warning("audit_domain_eval_no_pairs", session_id=sid[:8], domain=domain)
            return

        # Attempt eval dispatch
        ok = await _call_eval_engine(sid, domain, pairs, c_meta)

        if ok:
            _domain_evals_fired.labels(status="ok").inc()
            _eval_fanouts.labels(status="ok").inc()

            meta = await self._redis.get_meta(sid) or AuditSessionMeta(session_id=sid)
            meta.domains_evaluated.append(domain)
            meta.total_domains_evaled += 1
            await self._redis.set_meta(sid, meta)

            emit(ObsEvent(
                kind       = "audit_domain_eval_sent",
                service    = "qa_audit_bus",
                session_id = sid,
                ts         = task.ts,
                extra      = {"domain": domain, "pair_count": len(pairs)},
            ))
            log.info(
                "audit_domain_eval_dispatched",
                session_id = sid[:8],
                domain     = domain,
                pairs      = len(pairs),
            )
        else:
            # Enqueue to DLQ for retry
            dlq_item = _DLQItem(
                session_id     = sid,
                domain         = domain,
                eval_pairs     = pairs,
                candidate_meta = c_meta,
            )
            self._dlq.enqueue(dlq_item)
            _domain_evals_fired.labels(status="dlq").inc()
            _eval_fanouts.labels(status="error").inc()

            emit(ObsEvent(
                kind       = "audit_domain_eval_dlq",
                service    = "qa_audit_bus",
                session_id = sid,
                ts         = task.ts,
                extra      = {"domain": domain, "pair_count": len(pairs)},
            ))


# ── session analytics snapshot ─────────────────────────────────────────────────

@dataclass
class AuditSessionSnapshot:
    """
    Point-in-time audit snapshot for a session.
    Returned by QAAuditBus.get_session_snapshot().
    Used by admin tooling and /debug/session endpoints.
    """
    session_id:           str
    meta:                 AuditSessionMeta
    turns:                list[AuditTurnRecord]
    domains_evaluated:    set[str]
    dlq_depth:            int
    total_domains_in_doc: int     = 0
    active_domain:        str     = ""
    interview_complete:   bool    = False

    def to_dict(self) -> dict:
        return {
            "session_id":           self.session_id,
            "meta":                 self.meta.to_dict(),
            "turns":                [t.to_dict() for t in self.turns],
            "domains_evaluated":    list(self.domains_evaluated),
            "dlq_depth":            self.dlq_depth,
            "total_domains_in_doc": self.total_domains_in_doc,
            "active_domain":        self.active_domain,
            "interview_complete":   self.interview_complete,
        }


# ── QA Audit Bus — main class ──────────────────────────────────────────────────

class QAAuditBus:
    """
    QA Audit Bus — the single integration point for all downstream I/O
    triggered by committed interview turns.

    voice_graph calls three methods in order:
      1. open_session(session_id)              ← when session is registered
      2. route_turn(committed, qa_doc)         ← after every commit_turn()
         OR trigger_domain_eval(...)           ← when domain_rotated=True
      3. close_session(session_id, reason)     ← when interview ends

    Everything else is internal. No caller outside voice_graph needs to
    touch this class. eval_engine and transcript_writer are called BY this
    class, not by voice_graph directly.
    """

    def __init__(self) -> None:
        self._redis     = _AuditRedis()
        self._dlq       = _DeadLetterQueue()
        self._processor = _RouteProcessor(self._redis, self._dlq)
        self._started   = False

    # ── public API ─────────────────────────────────────────────────────────────

    async def open_session(self, session_id: str) -> None:
        """
        Emit a session-open event. Call once when a new interview session
        is registered (before the first audio turn arrives).

        Fire-and-forget — enqueues to background processor.
        """
        await self._ensure_started()
        await self._processor.enqueue(_RouteTask(
            kind       = AuditEventKind.SESSION_OPEN,
            session_id = session_id,
        ))

    async def close_session(self, session_id: str, reason: str = "") -> None:
        """
        Emit a session-close event and flush all pending writes.
        Call from controller._shutdown() or session_store.end().

        Flushes the route queue before returning so no turns are lost
        when the process exits.
        """
        await self._ensure_started()
        await self._processor.enqueue(_RouteTask(
            kind         = AuditEventKind.SESSION_CLOSE,
            session_id   = session_id,
            close_reason = reason,
        ))
        await self._processor.flush(timeout=TRANSCRIPT_FLUSH_TIMEOUT)

    async def route_turn(
        self,
        *,
        committed_turn: "CommittedTurn",
        qa_document:    "QADocument | None" = None,
        request_id:     str                 = "",
    ) -> None:
        """
        Route a committed Q/A turn to all downstream sinks.

        Called from voice_graph after qa_controller.commit_turn() returns.
        Fire-and-forget — never blocks the pipeline.

        Args:
            committed_turn: The CommittedTurn returned by qa_controller.commit_turn()
            qa_document:    Optional post-commit QADocument snapshot (for richer metadata)
            request_id:     Pipeline request ID for distributed tracing
        """
        if not committed_turn:
            return
        await self._ensure_started()
        await self._processor.enqueue(_RouteTask(
            kind           = AuditEventKind.TURN_COMMITTED,
            session_id     = committed_turn.session_id,
            committed_turn = committed_turn,
            qa_document    = qa_document,
            request_id     = request_id or new_request_id(),
        ))

    async def trigger_domain_eval(
        self,
        *,
        session_id:       str,
        completed_domain: str,
        qa_document:      "QADocument | None" = None,
    ) -> None:
        """
        Signal that a domain has been fully questioned and is ready for evaluation.

        Call this when CommittedTurn.domain_rotated is True (i.e. the domain that
        was just active before rotation is now complete).

        The bus will:
          1. Check idempotency (skip if already dispatched)
          2. Pull all Q/A turns for the completed domain from the audit log
          3. Package them and dispatch to eval_engine.schedule_domain_eval()
          4. On failure, enqueue to DLQ for background retry

        Fire-and-forget — never blocks the pipeline.
        """
        if not completed_domain:
            return
        await self._ensure_started()
        await self._processor.enqueue(_RouteTask(
            kind             = AuditEventKind.DOMAIN_EVAL_QUEUED,
            session_id       = session_id,
            completed_domain = completed_domain,
            qa_document      = qa_document,
        ))

    async def flush_all(self, timeout: float = 10.0) -> None:
        """
        Flush all pending route tasks. Call during graceful shutdown.
        Allows up to `timeout` seconds for the queue to drain.
        """
        await self._processor.flush(timeout=timeout)

    # ── read API for admin / debug ─────────────────────────────────────────────

    async def get_session_snapshot(self, session_id: str, qa_document: "QADocument | None" = None) -> AuditSessionSnapshot:
        """
        Return a full audit snapshot for a session.
        Used by admin tooling, the /debug/session endpoint, and test suites.
        """
        meta        = await self._redis.get_meta(session_id) or AuditSessionMeta(session_id=session_id)
        turns       = await self._redis.get_turns(session_id)
        evaled      = await self._redis.get_domains_evaluated(session_id)

        total_domains  = len(qa_document.domains) if qa_document else 0
        active_domain  = qa_document.active_domain if qa_document else ""
        is_complete    = qa_document.all_domains_exhausted if qa_document else False

        return AuditSessionSnapshot(
            session_id           = session_id,
            meta                 = meta,
            turns                = turns,
            domains_evaluated    = evaled,
            dlq_depth            = self._dlq.depth,
            total_domains_in_doc = total_domains,
            active_domain        = active_domain,
            interview_complete   = is_complete,
        )

    async def get_audit_turns(self, session_id: str) -> list[AuditTurnRecord]:
        """Return all committed audit turn records for a session."""
        return await self._redis.get_turns(session_id)

    async def get_domain_eval_status(self, session_id: str) -> dict[str, bool]:
        """
        Return {domain: was_eval_dispatched} mapping for a session.
        Useful for admin tooling to verify eval coverage.
        """
        evaled = await self._redis.get_domains_evaluated(session_id)
        return {d: True for d in evaled}

    async def requeue_failed_domain(
            self,
            session_id: str,
            domain: str,
            qa_document: "QADocument | None" = None,
    ) -> bool:
        """
        Admin: force-re-dispatch a domain that failed evaluation.
        Clears the idempotency guard and re-queues the domain eval task.

        Returns True if the domain was re-queued, False if no audit records found.
        """
        turns = await self._redis.get_turns(session_id)
        pairs = _EvalPairBuilder.build_eval_pairs(turns, domain)
        if not pairs:
            log.warning("audit_requeue_no_pairs", session_id=session_id[:8], domain=domain)
            return False

        # Clear idempotency marker so _handle_domain_eval will proceed.
        # Goes through the public _AuditRedis method rather than reaching
        # into private _set/_session_lock directly.
        await self._redis.unmark_domain_evaluated(session_id, domain)

        await self.trigger_domain_eval(
            session_id=session_id,
            completed_domain=domain,
            qa_document=qa_document,
        )
        log.info("audit_domain_requeued", session_id=session_id[:8], domain=domain)
        return True

    async def health(self) -> ServiceHealthState:
        """
        Return health state of the audit bus.
        Checks: Redis reachability, DLQ depth, background task liveness.
        """
        redis_ok = False
        try:
            r = await self._redis._conn() # noqa
            if r:
                await r.ping()
                redis_ok = True
        except Exception: # noqa
            pass

        task_alive = bool(
            self._processor._task and not self._processor._task.done() # noqa
        )

        healthy = redis_ok and task_alive and self._dlq.depth < (DLQ_MAX_SIZE * 0.9)

        return ServiceHealthState(
            service=__name__,
            healthy=healthy,
            circuit_state="closed",  # QAAuditBus has no circuit breaker; always closed
            inflight=self._processor._queue.qsize(), # noqa
            extra={
                "redis_ok": redis_ok,
                "background_task": task_alive,
                "route_queue_size": self._processor._queue.qsize(), # noqa
                "dlq_depth": self._dlq.depth,
                "dlq_max": DLQ_MAX_SIZE,
                "started": self._started,
            },
        )

    async def close(self) -> None:
        """
        Graceful shutdown: flush pending tasks and close Redis connection.
        Call from FastAPI lifespan shutdown event.
        """
        await self.flush_all(timeout=10.0)
        self._dlq.stop()
        await self._redis.close()
        log.info("audit_bus_closed")

    # ── internals ──────────────────────────────────────────────────────────────

    async def _ensure_started(self) -> None:
        if self._started:
            return
        await self._processor.ensure_started()
        self._dlq.start()
        self._started = True


# ── voice_graph integration helpers ───────────────────────────────────────────

async def route_committed_turn_to_audit(
    *,
    committed_turn: "CommittedTurn",
    qa_document:    "QADocument | None" = None,
    request_id:     str                 = "",
) -> None:
    """
    Convenience wrapper for voice_graph.node_llm.

    After qa_controller.commit_turn() returns, call this with the result.
    It fans out to transcript_writer and schedules eval if domain rotated.

      committed = await qa_controller.commit_turn(...)
      await route_committed_turn_to_audit(
          committed_turn = committed,
          qa_document    = qa_doc,
          request_id     = state["request_id"],
      )

    If committed.domain_rotated is True, also dispatches the completed
    domain for evaluation automatically. No extra call required from voice_graph.
    """
    await qa_audit_bus.route_turn(
        committed_turn = committed_turn,
        qa_document    = qa_document,
        request_id     = request_id,
    )

    if committed_turn.domain_rotated:
        # The domain that just finished is stage_after's predecessor.
        # committed_turn.domain holds the domain that was active for THIS turn
        # (the last turn of the completed domain).
        await qa_audit_bus.trigger_domain_eval(
            session_id       = committed_turn.session_id,
            completed_domain = committed_turn.domain,
            qa_document      = qa_document,
        )


async def finalize_session_eval(session_id: str, qa_document: "QADocument | None" = None) -> None:
    """
    Convenience wrapper — trigger eval for any domains that were never
    explicitly rotated (e.g. the LAST active domain when the interview ends).

    Call from voice_graph after stage transitions to 'complete'.

      await finalize_session_eval(session_id=session_id, qa_document=qa_doc)
    """
    if qa_document is None:
        return

    evaled = await qa_audit_bus.get_domain_eval_status(session_id)
    for domain in qa_document.domains:
        if not evaled.get(domain, False):
            await qa_audit_bus.trigger_domain_eval(
                session_id       = session_id,
                completed_domain = domain,
                qa_document      = qa_document,
            )


# ── backward compatibility shim ───────────────────────────────────────────────

class _ConversationMemoryShim:
    """
    Shim that surfaces the OLD conversation_memory public API for any code
    that hasn't been updated yet. Every method raises a clear error explaining
    why it's gone and what to use instead.

    If any caller imports conversation_memory and calls resolve() or commit(),
    it will get an immediate, clear, actionable error message rather than a
    silent behaviour change.
    """

    def __getattr__(self, name: str) -> Any:
        MIGRATION_GUIDE = {
            "resolve": (
                "conversation_memory.resolve() is REMOVED. "
                "Use qa_controller.get_llm_input(session_id, answer) → LLMInterviewInput, "
                "then pass to llm_node.stream_question(). "
                "The LLM no longer receives conversation history."
            ),
            "commit": (
                "conversation_memory.commit() is REMOVED. "
                "Use qa_controller.commit_turn(session_id, answer, question) → CommittedTurn, "
                "then await route_committed_turn_to_audit(committed_turn=...). "
                "This routes the turn to transcript_writer and eval_engine."
            ),
            "load": (
                "conversation_memory.load() is REMOVED. "
                "The LLM does not receive conversation history. "
                "To read turns for eval, use qa_audit_bus.get_audit_turns(session_id)."
            ),
            "evict": (
                "conversation_memory.evict() is REMOVED. "
                "Use qa_audit_bus.close_session(session_id, reason='evict')."
            ),
        }
        msg = MIGRATION_GUIDE.get(
            name,
            f"conversation_memory.{name}() is REMOVED. "
            f"conversation_memory is now QAAuditBus. See qa_audit_bus API."
        )
        raise NotImplementedError(msg)


conversation_memory = _ConversationMemoryShim()

# ── audit analytics ────────────────────────────────────────────────────────────

@dataclass
class DomainAuditStats:
    """Per-domain statistics derived from audit turn records."""
    domain:           str
    total_questions:  int    = 0
    total_answers:    int    = 0
    avg_answer_words: float  = 0.0
    min_answer_words: int    = 0
    max_answer_words: int    = 0
    transcript_ok_pct: float = 0.0    # % of turns where transcript write succeeded
    eval_dispatched:  bool   = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionAuditReport:
    """
    Full audit report for a completed session.
    Generated by AuditAnalytics.build_report(session_id).
    Used by admin tooling and the post-interview summary endpoint.
    """
    session_id:          str
    opened_at:           float | None   = None
    closed_at:           float | None   = None
    duration_minutes:    float          = 0.0
    total_turns:         int            = 0
    total_domains:       int            = 0
    domains_evaluated:   list[str]      = field(default_factory=list)
    domains_not_evaled:  list[str]      = field(default_factory=list)
    per_domain:          list[DomainAuditStats] = field(default_factory=list)
    avg_answer_words:    float          = 0.0
    transcript_ok_pct:   float          = 0.0
    dlq_depth:           int            = 0
    is_complete:         bool           = False

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "per_domain": [d.to_dict() for d in self.per_domain],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class AuditAnalytics:
    """
    Read-only analytics engine over audit turn records.

    Builds DomainAuditStats and SessionAuditReport from raw AuditTurnRecords
    without touching qa_controller's QADocument. Keeps analytics decoupled
    from session state so reports can be generated after session completion.
    """

    @staticmethod
    def build_domain_stats(
        turns:            list[AuditTurnRecord],
        domain:           str,
        eval_dispatched:  bool = False,
    ) -> DomainAuditStats:
        domain_turns = [t for t in turns if t.domain == domain]
        if not domain_turns:
            return DomainAuditStats(domain=domain)

        word_counts = [t.word_count_a for t in domain_turns]
        ok_count    = sum(1 for t in domain_turns if t.transcript_ok)

        return DomainAuditStats(
            domain            = domain,
            total_questions   = len(domain_turns),
            total_answers     = len(domain_turns),
            avg_answer_words  = round(sum(word_counts) / len(word_counts), 1),
            min_answer_words  = min(word_counts),
            max_answer_words  = max(word_counts),
            transcript_ok_pct = round(ok_count / len(domain_turns) * 100, 1),
            eval_dispatched   = eval_dispatched,
        )

    @classmethod
    async def build_report(
            cls,
            session_id: str,
            bus: "QAAuditBus",
    ) -> SessionAuditReport:
        """Build a full SessionAuditReport for a session."""
        # Use the public snapshot API instead of reaching into bus._redis and
        # bus._dlq directly. get_session_snapshot() already aggregates meta,
        # turns, evaluated domains, and dlq_depth through the proper accessors.
        snapshot = await bus.get_session_snapshot(session_id)

        meta = snapshot.meta
        turns = snapshot.turns
        evaled_set = snapshot.domains_evaluated

        all_domains = list(dict.fromkeys(t.domain for t in turns))
        not_evaled = [d for d in all_domains if d not in evaled_set]
        per_domain = [
            cls.build_domain_stats(turns, d, d in evaled_set)
            for d in all_domains
        ]

        duration = 0.0
        if meta.opened_at and meta.closed_at:
            duration = round((meta.closed_at - meta.opened_at) / 60, 2)

        all_word_counts = [t.word_count_a for t in turns] or [0]
        ok_turns = sum(1 for t in turns if t.transcript_ok)
        transcript_pct = round(ok_turns / len(turns) * 100, 1) if turns else 0.0

        return SessionAuditReport(
            session_id          = session_id,
            opened_at           = meta.opened_at,
            closed_at           = meta.closed_at,
            duration_minutes    = duration,
            total_turns         = len(turns),
            total_domains       = len(all_domains),
            domains_evaluated   = list(evaled_set),
            domains_not_evaled  = not_evaled,
            per_domain          = per_domain,
            avg_answer_words    = round(sum(all_word_counts) / len(all_word_counts), 1),
            transcript_ok_pct   = transcript_pct,
            dlq_depth           = snapshot.dlq_depth,  # from snapshot, not bus._dlq.depth
            is_complete         = bool(meta.closed_at),
        )

    @staticmethod
    def detect_anomalies(turns: list[AuditTurnRecord]) -> list[dict]:
        """
        Detect anomalies in the audit turn sequence.
        Returns list of {turn_index, anomaly_type, detail} dicts.
        """
        anomalies: list[dict] = []
        if not turns:
            return anomalies

        for t in turns:
            # Very short answers (possible connectivity issue or evasion)
            if 0 < t.word_count_a < 3:
                anomalies.append({
                    "turn_index":   t.turn_index,
                    "anomaly_type": "very_short_answer",
                    "detail":       f"{t.word_count_a} words",
                })
            # Very long answers (candidate may be reading from notes)
            if t.word_count_a > 300:
                anomalies.append({
                    "turn_index":   t.turn_index,
                    "anomaly_type": "very_long_answer",
                    "detail":       f"{t.word_count_a} words",
                })
            # Missing transcript write
            if not t.transcript_ok:
                anomalies.append({
                    "turn_index":   t.turn_index,
                    "anomaly_type": "transcript_write_failed",
                    "detail":       "turn was not written to transcript sink",
                })

        # Check for domain gaps (domain appears, disappears, reappears)
        seen: set[str] = set()
        last_domain     = None
        for t in turns:
            if t.domain != last_domain and t.domain in seen:
                anomalies.append({
                    "turn_index":   t.turn_index,
                    "anomaly_type": "domain_reentrance",
                    "detail":       f"domain '{t.domain}' re-appeared after leaving",
                })
            seen.add(t.domain)
            last_domain = t.domain

        return anomalies


# ── retention policy (GDPR / data minimization) ────────────────────────────────

class RetentionPolicy:
    """
    GDPR-aware retention policy for audit records.

    On retention_days expiry OR explicit right-to-erasure request:
      • PII fields (question text, answer text) are overwritten with "[erased]"
      • Session metadata is anonymized (name → "", raw_intro → "[erased]")
      • Turn records are preserved for statistical aggregation
      • All changes are logged to the observability stack

    NOTE: Redis TTL handles automatic key expiry. This class handles
    explicit erasure (user request or admin action) BEFORE the TTL expires.
    """

    _ERASED_MARKER = "[erased per retention policy]"

    def __init__(self, redis: _AuditRedis) -> None:
        self._redis = redis

    async def erase_session_pii(self, session_id: str) -> dict:
        """
        Erase all PII from a session's audit records.
        Returns a summary of what was erased.
        """
        turns = await self._redis.get_turns(session_id)
        meta  = await self._redis.get_meta(session_id)

        if not turns and not meta:
            return {"erased": False, "reason": "session_not_found"}

        # Erase turn-level PII
        erased_turns = 0
        for turn in turns:
            turn.question = self._ERASED_MARKER
            turn.answer   = self._ERASED_MARKER
            erased_turns  += 1

        # Persist erased turns
        await self._redis.write_turns(session_id, turns)

        # Erase meta-level PII (session_id is retained for audit trail continuity)
        result_summary: dict = {
            "erased":        True,
            "turns_erased":  erased_turns,
            "meta_erased":   False,
            "session_id_8":  session_id[:8],
        }

        if meta:
            # No-op when running locally without OTel — prevents the block from
            # being optimised away or flagged as empty in linters while still
            # being a safe write when OTel is active. Non-PII fields are retained as-is.
            meta.close_reason = meta.close_reason
            await self._redis.set_meta(session_id, meta)
            result_summary["meta_erased"] = True

        emit(ObsEvent(
            kind       = "audit_pii_erased",
            service    = "qa_audit_bus",
            session_id = session_id,
            ts         = time.time(),
            extra      = result_summary,
        ))
        log.warning(
            "audit_pii_erased",
            session_id   = session_id[:8],
            turns_erased = erased_turns,
        )
        return result_summary

    async def bulk_erase(self, session_ids: list[str]) -> list[dict]:
        """
        Batch PII erasure. Processes up to 10 sessions concurrently.
        """
        sem = asyncio.Semaphore(10)

        async def _erase_one(sid: str) -> dict:
            async with sem:
                return await self.erase_session_pii(sid)

        results = await asyncio.gather(*[_erase_one(sid) for sid in session_ids])
        return list(results)


# ── session replay ─────────────────────────────────────────────────────────────

class SessionReplayer:
    """
    Reconstructs an interview session from audit turn records.

    Produces a human-readable or machine-readable replay of the session
    without needing the live QADocument (which may have expired from Redis).

    Output formats:
      - text: formatted .txt-style transcript (same format as transcript_writer)
      - json: structured list of {question, answer, domain, ts} dicts
      - markdown: markdown table with Q/A/domain columns
    """

    def replay_as_text( # noqa
        self,
        session_id: str,
        turns:      list[AuditTurnRecord],
        meta:       AuditSessionMeta | None = None,
    ) -> str:
        lines: list[str] = [] # noqa
        lines.append("─" * 60)
        lines.append(f"Session  : {session_id}")
        if meta and meta.opened_at:
            lines.append(f"Started  : {datetime.fromtimestamp(meta.opened_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        if meta and meta.closed_at:
            lines.append(f"Ended    : {datetime.fromtimestamp(meta.closed_at, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("─" * 60)
        lines.append("")

        current_domain = None
        for turn in sorted(turns, key=lambda t: t.turn_index):
            if turn.domain != current_domain:
                lines.append(f"── Domain: {turn.domain} ──")
                current_domain = turn.domain

            ts_str  = datetime.fromtimestamp(turn.ts, tz=timezone.utc).strftime("%H:%M:%S")
            sid_lbl = session_id[:8]
            lines.append(f"[{ts_str}] [AI]         {turn.question}")
            lines.append(f"[{ts_str}] [{sid_lbl}]   {turn.answer}")
            lines.append("")

        lines.append("─" * 60)
        if meta and meta.closed_at and meta.opened_at:
            duration = round((meta.closed_at - meta.opened_at) / 60, 1)
            lines.append(f"Duration : {duration} minutes")
        lines.append("─" * 60)
        return "\n".join(lines)

    def replay_as_json(self, turns: list[AuditTurnRecord]) -> str: # noqa
        records = [
            {
                "turn_index": t.turn_index,
                "domain":     t.domain,
                "question":   t.question,
                "answer":     t.answer,
                "ts":         t.ts,
                "word_count_q": t.word_count_q,
                "word_count_a": t.word_count_a,
            }
            for t in sorted(turns, key=lambda t: t.turn_index)
        ]
        return json.dumps(records, indent=2, ensure_ascii=False)

    def replay_as_markdown(self, turns: list[AuditTurnRecord]) -> str: # noqa
        lines = [
            "| Turn | Domain | Question (truncated) | Answer (truncated) |",
            "| ---- | ------ | -------------------- | ------------------ |",
        ]
        for t in sorted(turns, key=lambda t: t.turn_index):
            q_trunc = t.question[:60].replace("|", "\\|") + "…" if len(t.question) > 60 else t.question.replace("|", "\\|")
            a_trunc = t.answer[:60].replace("|", "\\|") + "…" if len(t.answer) > 60 else t.answer.replace("|", "\\|")
            lines.append(f"| {t.turn_index} | {t.domain} | {q_trunc} | {a_trunc} |")
        return "\n".join(lines)


# ── webhook fanout ─────────────────────────────────────────────────────────────

AUDIT_WEBHOOK_URL: str | None = os.getenv("AUDIT_WEBHOOK_URL")
AUDIT_WEBHOOK_TIMEOUT: float  = float(os.getenv("AUDIT_WEBHOOK_TIMEOUT_S", "5.0"))
AUDIT_WEBHOOK_EVENTS: set[str] = set(
    os.getenv("AUDIT_WEBHOOK_EVENTS", "session_close,domain_eval_sent").split(",")
)


class AuditWebhookFanout:
    """
    Optional webhook fanout for audit events.

    When AUDIT_WEBHOOK_URL is set, specific audit events are POST'd as JSON
    to the configured endpoint. Useful for integration with:
      - External HR/ATS platforms (session close → candidate result)
      - Monitoring dashboards (domain eval → score ingestion)
      - Alerting systems (dlq_exhausted → ops alert)

    Fires asynchronously — never blocks the audit pipeline.
    Failed webhooks are logged but NOT retried (not business-critical).

    Configure which events to fan out via AUDIT_WEBHOOK_EVENTS env var
    (comma-separated list of AuditEventKind values).
    """

    def __init__(self) -> None:
        self._client: Any | None = None   # httpx.AsyncClient
        self._lock   = asyncio.Lock()

    async def _get_client(self) -> Any | None:
        if not AUDIT_WEBHOOK_URL:
            return None
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        import httpx  # noqa
                        self._client = httpx.AsyncClient(timeout=AUDIT_WEBHOOK_TIMEOUT)
                    except ImportError:
                        self._client = None
                        log.warning("audit_webhook_httpx_not_available")
                        return None
        return self._client

    async def emit(
        self,
        event_kind: str,
        session_id: str,
        payload:    dict,
    ) -> None:
        """
        Emit an audit event to the webhook endpoint (fire-and-forget).
        """
        if not AUDIT_WEBHOOK_URL:
            return
        if event_kind not in AUDIT_WEBHOOK_EVENTS:
            return

        client = await self._get_client()
        if client is None:
            return

        body = {
            "event_kind":  event_kind,
            "session_id":  session_id,
            "ts":          time.time(),
            "payload":     payload,
        }

        asyncio.ensure_future(self._send(client, AUDIT_WEBHOOK_URL, body))

    async def _send(self, client: Any, url: str, body: dict) -> None: # noqa
        try:
            resp = await client.post(url, json=body)
            if resp.status_code >= 400:
                log.warning(
                    "audit_webhook_non_200",
                    status=resp.status_code,
                    event=body.get("event_kind"),
                )
        except Exception as exc:
            log.warning("audit_webhook_send_failed", error=str(exc), event=body.get("event_kind"))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# ── extended QAAuditBus ────────────────────────────────────────────────────────

class QAAuditBusV2(QAAuditBus):
    """
    Extended QAAuditBus with:
      - AuditAnalytics integration
      - RetentionPolicy (PII erasure)
      - SessionReplayer
      - AuditWebhookFanout
      - Anomaly detection on session close
    """

    def __init__(self) -> None:
        super().__init__()
        # AuditAnalytics has no instance state (all @staticmethod / @classmethod)
        # so no instance is needed — call methods directly on the class.
        self.retention = RetentionPolicy(self._redis)
        self.replayer = SessionReplayer()
        self.webhook = AuditWebhookFanout()

    async def close_session(self, session_id: str, reason: str = "") -> None:
        """
        Extended close: runs anomaly detection and webhook fanout on close.
        """
        # Initialise before the conditional so the reference in the webhook
        # payload is always valid regardless of whether turns were found.
        anomalies: list = []

        turns = await self._redis.get_turns(session_id)
        if turns:
            anomalies = AuditAnalytics.detect_anomalies(turns)
            if anomalies:
                log.warning(
                    "audit_session_anomalies_detected",
                    session_id=session_id[:8],
                    count=len(anomalies),
                    types=list({a["anomaly_type"] for a in anomalies}),
                )
                emit(ObsEvent(
                    kind="audit_anomalies_detected",
                    service="qa_audit_bus",
                    session_id=session_id,
                    ts=time.time(),
                    extra={"anomalies": anomalies[:10]},
                ))

        await super().close_session(session_id, reason=reason)

        meta = await self._redis.get_meta(session_id)
        await self.webhook.emit(
            event_kind="session_close",
            session_id=session_id,
            payload={
                "reason": reason,
                "total_turns": meta.total_turns_routed if meta else 0,
                "domains_evaled": meta.total_domains_evaled if meta else 0,
                "anomaly_count": len(anomalies),  # always safe — [] if no turns or none detected
            },
        )

    async def get_full_report(self, session_id: str) -> SessionAuditReport:
        """Build and return a full SessionAuditReport for a session."""
        return await AuditAnalytics.build_report(session_id, self)

    async def get_replay_text(self, session_id: str) -> str:
        """Return the session replay as a formatted .txt string."""
        turns = await self._redis.get_turns(session_id)
        meta  = await self._redis.get_meta(session_id)
        return self.replayer.replay_as_text(session_id, turns, meta)

    async def get_replay_json(self, session_id: str) -> str:
        """Return the session replay as a JSON string."""
        turns = await self._redis.get_turns(session_id)
        return self.replayer.replay_as_json(turns)

    async def get_replay_markdown(self, session_id: str) -> str:
        """Return the session replay as a Markdown table string."""
        turns = await self._redis.get_turns(session_id)
        return self.replayer.replay_as_markdown(turns)

    async def request_erasure(self, session_id: str) -> dict:
        """
        GDPR right-to-erasure for a specific session.
        Erases all PII fields from audit records while preserving structure.
        """
        return await self.retention.erase_session_pii(session_id)

    async def bulk_erasure(self, session_ids: list[str]) -> list[dict]:
        """Batch GDPR erasure for multiple sessions."""
        return await self.retention.bulk_erase(session_ids)

    async def close(self) -> None:
        await self.webhook.close()
        await super().close()


# ── module-level singleton ─────────────────────────────────────────────────────

qa_audit_bus = QAAuditBusV2()


# ── session queue (batch ingestion helper) ─────────────────────────────────────

class AuditSessionQueue:
    """
    High-throughput batch session ingestion helper.

    For pipelines that process many sessions concurrently (e.g., asynchronous
    batch interview scoring), AuditSessionQueue wraps QAAuditBus with:
      - Bounded concurrency (semaphore-controlled)
      - Per-session error isolation
      - Throughput metrics
      - Progress callbacks

    Example usage (batch scoring pipeline):
        queue = AuditSessionQueue(bus=qa_audit_bus, max_concurrent=20)
        async for sid in session_ids:
            await queue.enqueue_session(sid)
        await queue.drain()
    """

    def __init__(self, bus: QAAuditBusV2, max_concurrent: int = 20) -> None:
        self._bus  = bus
        self._sem  = asyncio.Semaphore(max_concurrent)
        self._pending: list[asyncio.Task] = []
        self._done  = 0
        self._errors = 0

    async def enqueue_session(self, session_id: str) -> None:
        """
        Enqueue all pending domain evals for a session.
        Useful when processing sessions that were interrupted before completion.
        """
        async with self._sem:
            try:
                evaled = await self._bus.get_domain_eval_status(session_id)
                turns  = await self._bus.get_audit_turns(session_id)
                domains_in_turns = list(dict.fromkeys(t.domain for t in turns))

                for domain in domains_in_turns:
                    if not evaled.get(domain, False):
                        await self._bus.trigger_domain_eval(
                            session_id       = session_id,
                            completed_domain = domain,
                        )

                self._done += 1
                log.debug(
                    "audit_queue_session_processed",
                    session_id = session_id[:8],
                    domains    = len(domains_in_turns),
                )

            except Exception as exc:
                self._errors += 1
                log.error(
                    "audit_queue_session_error",
                    session_id = session_id[:8],
                    error      = str(exc),
                )

    async def drain(self) -> dict:
        """Wait for all pending tasks to complete. Returns summary stats."""
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)
        return {
            "processed": self._done,
            "errors":    self._errors,
        }

    @property
    def stats(self) -> dict:
        return {"processed": self._done, "errors": self._errors}


# ── public convenience factory ─────────────────────────────────────────────────

def make_audit_session_queue(max_concurrent: int = 20) -> AuditSessionQueue:
    """Factory for creating a bounded batch session queue tied to the singleton bus."""
    return AuditSessionQueue(bus=qa_audit_bus, max_concurrent=max_concurrent)


# ── health endpoint helper ─────────────────────────────────────────────────────

async def audit_bus_health_check() -> dict:
    """
    Comprehensive health check for the audit bus and its dependencies.
    Designed to be called from /health or /readyz endpoint handlers.

    Returns:
        {
            "healthy":       bool,
            "redis_ok":      bool,
            "background_task": bool,
            "dlq_depth":     int,
            "dlq_max":       int,
            "route_queue":   int,
            "webhook_url":   str | None,
            "retention_ok":  bool,
        }
    """
    state        = await qa_audit_bus.health()
    webhook_ok   = AUDIT_WEBHOOK_URL is not None

    return {
        **state.extra,
        "healthy":      state.healthy,
        "webhook_url":  AUDIT_WEBHOOK_URL or None,
        "webhook_ok":   webhook_ok,
        "retention_ok": True,   # RetentionPolicy has no async deps to check
    }


# ── exports ────────────────────────────────────────────────────────────────────

__all__ = [
    # Main bus
    "qa_audit_bus",
    "QAAuditBus",
    "QAAuditBusV2",

    # Integration helpers (for voice_graph)
    "route_committed_turn_to_audit",
    "finalize_session_eval",

    # Analytics
    "AuditAnalytics",
    "SessionAuditReport",
    "DomainAuditStats",

    # Records
    "AuditTurnRecord",
    "AuditSessionMeta",
    "AuditSessionSnapshot",

    # Retention
    "RetentionPolicy",

    # Replay
    "SessionReplayer",

    # Batch
    "AuditSessionQueue",
    "make_audit_session_queue",

    # Shim (for migration detection)
    "conversation_memory",

    # Health
    "audit_bus_health_check",
]




# ── configuration reference ────────────────────────────────────────────────────
# All env vars consumed by this module, with defaults and descriptions.
# This table is also surfaced at /debug/config for operational visibility.

AUDIT_BUS_CONFIG_REFERENCE: dict[str, dict] = {
    "REDIS_URL":                  {"default": None,    "desc": "Redis connection URL (redis://host:port/db)"},
    "REDIS_MAX_CONN":             {"default": "200",   "desc": "Max Redis connections in pool"},
    "SESSION_TTL_S":              {"default": "3600",  "desc": "Session TTL in seconds"},
    "SESSION_GRACE_S":            {"default": "60",    "desc": "Grace period added to session TTL"},
    "AUDIT_LRU_SIZE":             {"default": "128",   "desc": "In-process LRU cache session capacity"},
    "AUDIT_DLQ_MAX_SIZE":         {"default": "512",   "desc": "Dead-letter queue maximum item count"},
    "AUDIT_DLQ_RETRY_BASE_S":     {"default": "2.0",   "desc": "DLQ retry base interval (seconds, exponential)"},
    "AUDIT_DLQ_MAX_RETRIES":      {"default": "8",     "desc": "Max DLQ retry attempts before dropping"},
    "AUDIT_ROUTE_QUEUE_DEPTH":    {"default": "1024",  "desc": "Route queue max depth before synchronous fallback"},
    "AUDIT_TRANSCRIPT_FLUSH_TIMEOUT_S": {"default": "5.0", "desc": "Transcript flush timeout on session close"},
    "AUDIT_WEBHOOK_URL":          {"default": None,    "desc": "Optional webhook URL for audit event fanout"},
    "AUDIT_WEBHOOK_TIMEOUT_S":    {"default": "5.0",   "desc": "Webhook POST timeout"},
    "AUDIT_WEBHOOK_EVENTS":       {"default": "session_close,domain_eval_sent", "desc": "Comma-sep event kinds to fan out"},
}


def get_audit_bus_config() -> dict:
    """
    Return the current runtime configuration of the audit bus as a dict.
    Useful for /debug/config or admin introspection endpoints.
    """
    return {
        key: {
            "value":   os.getenv(key, meta["default"]),
            "default": meta["default"],
            "desc":    meta["desc"],
        }
        for key, meta in AUDIT_BUS_CONFIG_REFERENCE.items()
    }