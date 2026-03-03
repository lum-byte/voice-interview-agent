"""
concept_tracker.py  —  ConceptTracker
══════════════════════════════════════

FILE LOCATION:  app/eval/concept_tracker.py

WHAT THIS IS
────────────
A Kafka-driven concept coverage tracker that steers which concepts the
question-engine LLM targets next — without the LLM knowing any of this
reasoning exists. Memory is Redis. Transport is Kafka. The LLM only sees
a slightly extended difficulty suffix that says:
    "Focus on: X. Avoid recently covered: Y."

DESIGN AXIOMS
─────────────
A. Concept extraction is RULE-BASED. Zero LLM calls. Zero latency on critical path.
B. Kafka consumer is async and decoupled. A lag or crash never blocks a pipeline turn.
C. If the steering signal isn't ready (consumer lagged), LLMInputBuilder falls back
   to the existing fingerprint hints. Graceful degradation is the default.
D. Coverage map is per-session per-domain. Redis TTL mirrors session TTL.
E. A concept is "covered" when its keywords appear in a committed question.
   We track the QUESTION, not the answer — the answer is the LLM's business.
F. The steering signal carries two lists: `avoid` (covered) and `focus` (uncovered).
   Both are injected as plain English constraints into the suffix. Nothing else.
G. Cross-session concept state is never shared. Each session starts from zero.
"""

from __future__ import annotations

import asyncio
import hashlib # noqa
import pathlib
import json
from app.monitoring.observability import get_logger
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from app.common.shared import InMemoryLRU

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# § CONCEPT REGISTRY
# Domain → concept clusters → keyword triggers
# Rule-based only. Each keyword is lowercased and substring-matched
# against the committed question text after normalization.
# ══════════════════════════════════════════════════════════════════════════════

CONCEPT_REGISTRY: dict[str, dict[str, list[str]]] = json.loads(
    (pathlib.Path(__file__).parent / "concept_registry.json").read_text(encoding="utf-8")
)

# Minimum keyword hits to tag a concept as "covered" in a single question
CONCEPT_HIT_THRESHOLD: int = 1

# Redis key layout
_COVERAGE_PREFIX = "concept:cov:v1:"
_COVERAGE_TTL_S  = int(os.getenv("SESSION_TTL_S", "3600")) + int(os.getenv("SESSION_GRACE_S", "60"))

# Kafka config
KAFKA_BOOTSTRAP: str       = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_IN:  str       = os.getenv("CONCEPT_TRACKER_TOPIC", "interview.question.committed")
KAFKA_GROUP_ID:  str       = os.getenv("CONCEPT_TRACKER_GROUP",  "concept-tracker-v1")
KAFKA_ENABLED:   bool      = os.getenv("CONCEPT_TRACKER_KAFKA", "true").lower() == "true"

# How many covered concepts to surface in the avoid list
MAX_AVOID_HINTS: int = int(os.getenv("CONCEPT_MAX_AVOID", "4"))
# How many uncovered concepts to surface in the focus list
MAX_FOCUS_HINTS: int = int(os.getenv("CONCEPT_MAX_FOCUS", "2"))

# Minimum turns in a domain before we start steering (avoid noise on first Q)
MIN_TURNS_BEFORE_STEER: int = 2


# ══════════════════════════════════════════════════════════════════════════════
# § DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuestionAskedEvent:
    """
    Produced by qa_controller.commit_turn() → consumed by ConceptTracker.
    Minimal payload — only what concept extraction needs.
    """
    session_id:  str
    domain:      str
    question:    str    # the QUESTION text only — answer never travels this bus
    turn_index:  int
    level:       str    = "intermediate"
    ts:          float  = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps({
            "session_id":  self.session_id,
            "domain":      self.domain,
            "question":    self.question,
            "turn_index":  self.turn_index,
            "level":       self.level,
            "ts":          self.ts,
        })

    @staticmethod
    def from_json(raw: str | bytes) -> "QuestionAskedEvent":
        d = json.loads(raw)
        return QuestionAskedEvent(
            session_id = str(d["session_id"]),
            domain     = str(d["domain"]),
            question   = str(d["question"]),
            turn_index = int(d["turn_index"]),
            level      = str(d.get("level", "intermediate")),
            ts         = float(d.get("ts", time.time())),
        )


@dataclass
class ConceptCoverageMap:
    """
    Tracks which concept clusters have been hit in a session/domain.
    Stored in Redis as JSON. Never sent to the LLM.
    """
    session_id:    str
    domain:        str
    covered:       dict[str, int]   = field(default_factory=dict)
    # concept_name → hit count (how many questions touched it)
    total_tracked: int              = 0
    last_updated:  float            = field(default_factory=time.time)

    def mark_covered(self, concept: str, hits: int = 1) -> None:
        self.covered[concept] = self.covered.get(concept, 0) + hits
        self.total_tracked   += 1
        self.last_updated     = time.time()

    def covered_concepts(self) -> list[str]:
        """Concepts hit at least once, sorted by hit count descending."""
        return sorted(self.covered.keys(), key=lambda c: -self.covered[c])

    def uncovered_concepts(self, domain: str) -> list[str]:
        """Concepts in the registry not yet covered for this domain."""
        all_concepts = set(CONCEPT_REGISTRY.get(domain, {}).keys())
        return [c for c in all_concepts if c not in self.covered]

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "domain":        self.domain,
            "covered":       self.covered,
            "total_tracked": self.total_tracked,
            "last_updated":  self.last_updated,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: dict) -> "ConceptCoverageMap":
        m = ConceptCoverageMap(
            session_id = str(d["session_id"]),
            domain     = str(d["domain"]),
        )
        m.covered       = dict(d.get("covered", {}))
        m.total_tracked = int(d.get("total_tracked", 0))
        m.last_updated  = float(d.get("last_updated", time.time()))
        return m

    @staticmethod
    def from_json(raw: str) -> "ConceptCoverageMap":
        return ConceptCoverageMap.from_dict(json.loads(raw))


@dataclass
class ConceptSteeringSignal:
    """
    Output of ConceptTracker.get_steering_signal().
    Consumed ONLY by LLMInputBuilder to extend the difficulty_prompt_suffix.
    Never passed directly to the LLM — only its plain-text rendering is.
    """
    session_id:      str
    domain:          str
    avoid_concepts:  list[str]   # recently covered — steer away
    focus_concepts:  list[str]   # uncovered — steer toward
    coverage_ratio:  float       # covered / total in registry (0–1)
    has_signal:      bool        # False → no data yet, use fallback
    ts:              float = field(default_factory=time.time)

    def to_suffix_instruction(self) -> str:
        """
        Renders as a plain-English constraint appended to difficulty_prompt_suffix.
        This is the ONLY thing the LLM sees from this entire module.

        Example output:
            "Focus your question on: metaclasses, descriptors, type system.
             Do NOT ask about: concurrency, generators — these have been covered."
        """
        if not self.has_signal:
            return ""

        parts: list[str] = []

        if self.focus_concepts: # noqa
            focus_labels = [c.replace("_", " ") for c in self.focus_concepts]
            parts.append(f"Focus your question on: {', '.join(focus_labels)}.")

        if self.avoid_concepts:
            avoid_labels = [c.replace("_", " ") for c in self.avoid_concepts]
            parts.append(
                f"Do NOT ask about: {', '.join(avoid_labels)} — "
                f"these concepts have already been covered this session."
            )

        return " ".join(parts)

    def is_empty(self) -> bool:
        return not self.avoid_concepts and not self.focus_concepts


# ══════════════════════════════════════════════════════════════════════════════
# § CONCEPT EXTRACTOR
# Rule-based only. Extracts which concept clusters a question touches.
# No LLM. No network call. Runs in <1ms on any modern CPU.
# ══════════════════════════════════════════════════════════════════════════════

class ConceptExtractor:
    """
    Keyword-based concept tagger.

    For each concept cluster in CONCEPT_REGISTRY[domain], counts how many
    of its keyword triggers appear in the normalized question text.
    Returns a dict of {concept_name: hit_count} for concepts above threshold.

    Normalization: lowercase, strip punctuation, collapse whitespace.
    Matching: substring, not word-boundary (intentional — "threading" matches
    "multithreading"). The hit threshold prevents false positives from single
    generic keyword matches.
    """

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def extract(cls, domain: str, question: str) -> dict[str, int]:
        """
        Returns {concept_name: hit_count} for all concepts whose keyword
        hit count meets or exceeds CONCEPT_HIT_THRESHOLD.
        """
        domain_concepts = CONCEPT_REGISTRY.get(domain, {})
        if not domain_concepts:
            return {}

        normalized = cls._normalize(question)
        hits: dict[str, int] = {}

        for concept_name, keywords in domain_concepts.items():
            count = sum(1 for kw in keywords if kw in normalized)
            if count >= CONCEPT_HIT_THRESHOLD:
                hits[concept_name] = count

        return hits

    @classmethod
    def all_concepts_for_domain(cls, domain: str) -> list[str]:
        return list(CONCEPT_REGISTRY.get(domain, {}).keys())


# ══════════════════════════════════════════════════════════════════════════════
# § KAFKA PRODUCER
# Called from qa_controller.commit_turn() — fire-and-forget.
# If Kafka is unavailable, the event is dropped silently.
# The concept tracker degrades to no-signal (fallback path in LLMInputBuilder).
# ══════════════════════════════════════════════════════════════════════════════

class ConceptEventProducer:
    """
    Thin async Kafka producer for QuestionAskedEvent.
    Initialized lazily on first produce() call.
    Swallowed exceptions — a failed produce never affects the pipeline turn.
    """

    def __init__(self) -> None:
        self._producer: Any = None
        self._lock = asyncio.Lock()
        self._enabled = KAFKA_ENABLED

    async def _get_producer(self) -> Any:
        if self._producer is not None:
            return self._producer
        async with self._lock:
            if self._producer is not None:
                return self._producer # noqa | defensive
            try:
                from aiokafka import AIOKafkaProducer  # type: ignore
                p = AIOKafkaProducer(
                    bootstrap_servers = KAFKA_BOOTSTRAP,
                    value_serializer  = lambda v: v.encode() if isinstance(v, str) else v,
                    compression_type  = "gzip",
                    linger_ms         = 5,     # micro-batch: 5ms window to amortize network
                    acks              = 1,     # leader ack — not at-least-once, intentional
                    request_timeout_ms= 3000,
                )
                await p.start()
                self._producer = p
                log.info("concept_producer_started", topic=KAFKA_TOPIC_IN)
            except ImportError:
                log.warning("concept_producer_aiokafka_not_installed")
                self._enabled = False
            except Exception as exc:
                log.warning("concept_producer_start_failed", error=str(exc))
                self._enabled = False
        return self._producer

    async def produce(self, event: QuestionAskedEvent) -> None:
        """
        Fire-and-forget produce. Never raises — swallows all exceptions.
        Called from commit_turn() with no await holding the critical path.
        """
        if not self._enabled:
            return
        try:
            producer = await self._get_producer()
            if producer:
                await producer.send(
                    KAFKA_TOPIC_IN,
                    key=event.session_id.encode(),
                    value = event.to_json(),
                )
                log.debug(
                    "concept_event_produced",
                    session_id  = event.session_id[:8],
                    domain      = event.domain,
                    turn_index  = event.turn_index,
                )
        except Exception as exc:
            log.debug("concept_produce_failed", error=str(exc))

    async def close(self) -> None:
        if self._producer:
            try:
                await self._producer.stop()
            except Exception:  # noqa
                pass
            self._producer = None


# ══════════════════════════════════════════════════════════════════════════════
# § REDIS COVERAGE STORE
# Reads and writes ConceptCoverageMap per session per domain.
# ══════════════════════════════════════════════════════════════════════════════

class ConceptCoverageStore:
    """
    Redis-backed persistence layer for ConceptCoverageMap.
    Falls back to an in-process dict if Redis is unavailable.
    """

    def __init__(self) -> None:
        self._redis: Any      = None
        self._lock            = asyncio.Lock()
        self._local = InMemoryLRU(max_size=int(os.getenv("CONCEPT_LRU_SIZE", "4096")))   # key → JSON blob (fallback)

    def _key(self, session_id: str, domain: str) -> str: # noqa
        return f"{_COVERAGE_PREFIX}{session_id}:{domain}"

    async def _get_redis(self) -> Any:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return None
        async with self._lock:
            if self._redis is None:
                try:
                    import redis.asyncio as aioredis  # type: ignore
                    self._redis = await aioredis.from_url(
                        redis_url,
                        encoding         = "utf-8",
                        decode_responses = True,
                        max_connections=int(os.getenv("CONCEPT_REDIS_MAX_CONN", "200")),
                        socket_timeout   = 0.3,
                    )
                except Exception as exc:
                    log.warning("concept_store_redis_failed", error=str(exc))
        return self._redis

    async def load(self, session_id: str, domain: str) -> ConceptCoverageMap:
        key = self._key(session_id, domain)
        raw: str | None = None

        try:
            r = await self._get_redis()
            if r:
                raw = await r.get(key)
        except Exception as exc:
            log.debug("concept_store_load_redis_error", error=str(exc))

        if raw is None:
            raw = await self._local.get(key)

        if raw:
            try:
                return ConceptCoverageMap.from_json(raw)
            except Exception: # noqa
                pass

        # First access — return empty map
        return ConceptCoverageMap(session_id=session_id, domain=domain)

    async def save(self, coverage: ConceptCoverageMap) -> None:
        key  = self._key(coverage.session_id, coverage.domain)
        blob = coverage.to_json()
        await self._local.set(key, blob)
        try:
            r = await self._get_redis()
            if r:
                await r.setex(key, _COVERAGE_TTL_S, blob)
        except Exception as exc:
            log.debug("concept_store_save_redis_error", error=str(exc))

    async def delete(self, session_id: str, domains: list[str]) -> None:
        """Called on session eviction."""
        for domain in domains:
            key = self._key(session_id, domain)

            # local fallback
            await self._local.delete(key)

            try:
                r = await self._get_redis()
                if r:
                    await r.delete(key)
            except Exception:  # noqa
                pass


# ══════════════════════════════════════════════════════════════════════════════
# § KAFKA CONSUMER
# Runs as a background asyncio task. Pulls QuestionAskedEvents,
# extracts concepts, updates the coverage store.
# Never on the critical path.
# ══════════════════════════════════════════════════════════════════════════════

class ConceptEventConsumer:
    """
    Async Kafka consumer. Started by voice_graph.on_startup().
    Runs forever in the background, consuming QuestionAskedEvents
    and updating ConceptCoverageStore.

    If Kafka is unavailable or aiokafka is not installed, falls back
    to processing events from an asyncio.Queue (in-process fallback).
    The producer writes to this queue directly in that case.
    """

    def __init__(
        self,
        store:     ConceptCoverageStore,
        extractor: ConceptExtractor,
    ) -> None:
        self._store     = store
        self._extractor = extractor
        self._consumer: Any          = None
        self._running:  bool         = False
        self._task:     asyncio.Task | None = None
        self._fallback_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

        self._events_processed: int  = 0
        self._events_failed:    int  = 0

    async def start(self) -> None:
        """Launch consumer in background. Returns immediately."""
        self._running = True
        self._task    = asyncio.create_task(self._run(), name="concept-tracker-consumer")
        log.info("concept_consumer_started", kafka_enabled=KAFKA_ENABLED)

    async def stop(self) -> None:
        self._running = False
        if self._consumer:
            try:
                await self._consumer.stop()
            except Exception: # noqa
                pass
        if self._task and not self._task.done():
            self._task.cancel()
        log.info(
            "concept_consumer_stopped",
            events_processed = self._events_processed,
            events_failed    = self._events_failed,
        )

    async def _run(self) -> None:
        """
        Main loop. Tries Kafka first, falls back to in-process queue.
        Each iteration is wrapped so a single bad message never stops the loop.
        """
        kafka_ok = await self._try_start_kafka()

        if kafka_ok:
            await self._run_kafka()
        else:
            log.info("concept_consumer_fallback_queue_mode")
            await self._run_queue()

    async def _try_start_kafka(self) -> bool:
        if not KAFKA_ENABLED:
            return False
        try:
            from aiokafka import AIOKafkaConsumer  # type: ignore
            self._consumer = AIOKafkaConsumer(
                KAFKA_TOPIC_IN,
                bootstrap_servers  = KAFKA_BOOTSTRAP,
                group_id           = KAFKA_GROUP_ID,
                auto_offset_reset  = "latest",   # don't replay history on restart
                enable_auto_commit = True,
                value_deserializer = lambda v: v.decode("utf-8"),
                session_timeout_ms = 10000,
                heartbeat_interval_ms = 3000,
                fetch_max_wait_ms  = 100,
            )
            await self._consumer.start()
            return True
        except Exception as exc:
            log.warning("concept_kafka_consumer_start_failed", error=str(exc))
            self._consumer = None
            return False

    async def _run_kafka(self) -> None:
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                await self._process_raw(msg.value)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("concept_kafka_consumer_error", error=str(exc))

    async def _run_queue(self) -> None:
        while self._running:
            try:
                raw = await asyncio.wait_for(
                    self._fallback_queue.get(),
                    timeout = 1.0,
                )
                await self._process_raw(raw)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("concept_queue_consume_error", error=str(exc))

    async def _process_raw(self, raw: str) -> None:
        """
        Parse one event and update the coverage map.
        Errors are swallowed — a bad event never stops the loop.
        """
        try:
            event = QuestionAskedEvent.from_json(raw)
            await self._process_event(event)
            self._events_processed += 1
        except Exception as exc:
            self._events_failed += 1
            log.debug("concept_event_process_failed", error=str(exc))

    async def _process_event(self, event: QuestionAskedEvent) -> None:
        """
        Core concept extraction and coverage update for one event.
        Rule-based extraction → Redis update. No LLM involved.
        """
        hits = self._extractor.extract(event.domain, event.question)

        if not hits:
            # No recognizable concept in this question — still update total_tracked
            coverage = await self._store.load(event.session_id, event.domain)
            coverage.total_tracked += 1
            coverage.last_updated   = time.time()
            await self._store.save(coverage)
            return

        coverage = await self._store.load(event.session_id, event.domain)
        for concept, hit_count in hits.items():
            coverage.mark_covered(concept, hit_count)

        await self._store.save(coverage)

        log.debug(
            "concept_coverage_updated",
            session_id = event.session_id[:8],
            domain     = event.domain,
            turn_index = event.turn_index,
            concepts   = list(hits.keys()),
            total_covered = len(coverage.covered),
        )

    def enqueue_fallback(self, event: QuestionAskedEvent) -> None:
        """
        Used by the producer when Kafka is unavailable.
        Drops silently if the queue is full (bounded, non-blocking).
        """
        try:
            self._fallback_queue.put_nowait(event.to_json())
        except asyncio.QueueFull:
            log.debug("concept_fallback_queue_full_drop")


# ══════════════════════════════════════════════════════════════════════════════
# § CONCEPT TRACKER  (orchestrator)
# The single public object. Owns producer, consumer, store, extractor.
# ══════════════════════════════════════════════════════════════════════════════

class ConceptTracker:
    """
    Public interface for the concept tracking system.

    voice_graph calls:
        await concept_tracker.start()         # on_startup()
        await concept_tracker.stop()          # on_shutdown()

    qa_controller.commit_turn() calls:
        concept_tracker.emit(event)           # fire-and-forget, sync

    LLMInputBuilder.build() calls:
        signal = await concept_tracker.get_steering_signal(session_id, domain, n_turns)
        suffix += signal.to_suffix_instruction()

    Nothing else touches this object.
    """

    def __init__(self) -> None:
        self._extractor = ConceptExtractor()
        self._store     = ConceptCoverageStore()
        self._producer  = ConceptEventProducer()
        self._consumer  = ConceptEventConsumer(self._store, self._extractor)
        self._started   = False

    async def start(self) -> None:
        if self._started:
            return
        await self._consumer.start()
        self._started = True
        log.info("concept_tracker_started")

    async def stop(self) -> None:
        await self._consumer.stop()
        await self._producer.close()
        self._started = False
        log.info("concept_tracker_stopped")

    def emit(self, event: QuestionAskedEvent) -> None:
        """
        Called from commit_turn(). Schedules async produce — never blocks.
        Falls back to in-process queue if Kafka producer isn't available yet.
        """
        task = asyncio.ensure_future(self._emit_async(event))
        task.add_done_callback(self._on_emit_done)

    @staticmethod
    def _on_emit_done(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.debug("concept_emit_task_failed", error=str(task.exception()))

    async def _emit_async(self, event: QuestionAskedEvent) -> None:
        """
        Try Kafka produce. On failure, push to consumer's fallback queue
        so the coverage map still gets updated in-process.
        """
        try:
            await self._producer.produce(event)
        except Exception: # noqa
            self._consumer.enqueue_fallback(event)

    async def get_steering_signal(
        self,
        session_id: str,
        domain:     str,
        n_turns:    int = 0,
    ) -> ConceptSteeringSignal:
        """
        Called by LLMInputBuilder before building the prompt suffix.
        Always returns a signal — `has_signal=False` means "use fallback",
        which the caller handles by not appending anything.
        """
        try:
            if n_turns < MIN_TURNS_BEFORE_STEER:
                return self._empty_signal(session_id, domain)

            coverage = await self._store.load(session_id, domain)

            if coverage.total_tracked == 0:
                return self._empty_signal(session_id, domain)

            # Build avoid list: most-hit concepts, capped
            avoid = coverage.covered_concepts()[:MAX_AVOID_HINTS]

            # Build focus list: uncovered concepts, capped
            uncovered = coverage.uncovered_concepts(domain)
            focus     = uncovered[:MAX_FOCUS_HINTS]

            total_in_domain = len(CONCEPT_REGISTRY.get(domain, {}))
            covered_count   = len(coverage.covered)
            ratio           = covered_count / total_in_domain if total_in_domain else 0.0

            signal = ConceptSteeringSignal(
                session_id     = session_id,
                domain         = domain,
                avoid_concepts = avoid,
                focus_concepts = focus,
                coverage_ratio = round(ratio, 3),
                has_signal     = bool(avoid or focus),
            )

            log.debug(
                "concept_signal_built",
                session_id     = session_id[:8],
                domain         = domain,
                avoid          = avoid,
                focus          = focus,
                coverage_ratio = signal.coverage_ratio,
            )

            return signal

        except Exception as exc:
            log.debug("concept_signal_error", error=str(exc))
            return self._empty_signal(session_id, domain)

    async def evict_session(self, session_id: str, domains: list[str]) -> None:
        """Called by QAControllerV2.close_session_v2() to clean up Redis."""
        await self._store.delete(session_id, domains)
        log.debug("concept_session_evicted", session_id=session_id[:8])

    async def get_coverage_report(
        self, session_id: str, domain: str
    ) -> dict[str, Any]:
        """Admin/debug endpoint — full coverage state for a session/domain."""
        coverage = await self._store.load(session_id, domain)
        all_concepts = ConceptExtractor.all_concepts_for_domain(domain)
        return {
            "session_id":      session_id,
            "domain":          domain,
            "covered":         coverage.covered,
            "uncovered":       [c for c in all_concepts if c not in coverage.covered],
            "total_tracked":   coverage.total_tracked,
            "coverage_ratio":  round(len(coverage.covered) / max(len(all_concepts), 1), 3),
            "last_updated":    coverage.last_updated,
        }

    @staticmethod
    def _empty_signal(session_id: str, domain: str) -> ConceptSteeringSignal:
        return ConceptSteeringSignal(
            session_id     = session_id,
            domain         = domain,
            avoid_concepts = [],
            focus_concepts = [],
            coverage_ratio = 0.0,
            has_signal     = False,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

concept_tracker = ConceptTracker()


# ══════════════════════════════════════════════════════════════════════════════
# § INTEGRATION INSTRUCTIONS
#
# [1] qa_controller.py — QAControllerV2.commit_turn()
#     Add import at top of file:
#         from app.eval.concept_tracker import concept_tracker, QuestionAskedEvent
#
#     In commit_turn(), after `committed = await super().commit_turn(...)`:
#         concept_tracker.emit(QuestionAskedEvent(
#             session_id = session_id,
#             domain     = committed.domain,
#             question   = llm_question,
#             turn_index = committed.turn_index,
#             level      = doc.candidate.level,
#         ))
#
# [2] qa_controller.py — LLMInputBuilder.build()
#     Add import at top of file (same import as above).
#
#     In LLMInputBuilder.build(), after computing `sys_text` and before
#     returning the LLMInterviewInput, append the steering suffix:
#
#         steering = await concept_tracker.get_steering_signal(
#             session_id = doc.session_id,
#             domain     = domain,
#             n_turns    = q_index,
#         )
#         steer_suffix = steering.to_suffix_instruction()
#         if steer_suffix:
#             sys_text += f"\n\n{steer_suffix}"
#
#     That's it. The LLM sees a slightly longer suffix. Nothing else changes.
#
# [3] voice_graph.py — on_startup() / on_shutdown()
#     Add import:
#         from app.eval.concept_tracker import concept_tracker
#
#     In on_startup():
#         await concept_tracker.start()
#
#     In on_shutdown():
#         await concept_tracker.stop()
#
# [4] qa_controller.py — QAControllerV2.close_session_v2()
#     After qa_prefetch_buffer.cancel_session() and self._scaler.evict_session():
#         doc = await self.get_document(session_id)
#         if doc:
#             await concept_tracker.evict_session(session_id, doc.domains)
#
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# § ANSWER SIGNAL DATA STRUCTURES
# Per-session per-domain tracking of what the candidate claims to know.
# Lives alongside ConceptCoverageMap. Answer text never stored here.
# ══════════════════════════════════════════════════════════════════════════════

import statistics as _statistics  # noqa — already in stdlib, alias avoids shadowing

# Answer signal config
_SIGNAL_PREFIX   = "concept:sig:v1:"
_SIGNAL_TTL_S    = _COVERAGE_TTL_S

MAX_PROBE_HINTS:  int   = int(os.getenv("CONCEPT_MAX_PROBE",   "3"))
MAX_WEAK_HINTS:   int   = int(os.getenv("CONCEPT_MAX_WEAK",    "2"))
MAX_DYNAMIC_CAPS: int   = int(os.getenv("CONCEPT_MAX_DYNAMIC", "8"))

CLAIM_STRONG_THRESHOLD: float = 0.70
CLAIM_WEAK_THRESHOLD:   float = 0.35
CLAIM_DROP_THRESHOLD:   float = 0.20

PRESS_DECAY_TURNS: int = int(os.getenv("CONCEPT_PRESS_DECAY", "4"))

MIN_ANSWER_CHARS_FOR_EXTRACTION: int = 30

CONTRADICTION_VARIANCE_THRESHOLD: float = 0.18


@dataclass
class CandidateClaimEntry:
    """
    A single concept-level signal extracted from a candidate's answer.
    Never sent to the LLM directly — only its rendered label appears in
    the suffix instruction.

    signal_type values:
      "claimed"       — candidate confidently named/explained this concept
      "weak"          — candidate hedged ("I think it's something like…")
      "slip"          — keyword density suggests concept confusion
      "dynamic"       — term not in registry, discovered live from answer
      "contradiction" — candidate's confidence swung >THRESHOLD across turns
    """
    concept:            str
    raw_term:           str      # exact term candidate used (logging only)
    confidence:         float    # 0.0–1.0
    signal_type:        str
    turn_index:         int
    pressed:            bool           = False
    press_urgency:      float          = 1.0
    contradiction_turn: int | None     = None
    ts:                 float          = field(default_factory=time.time)

    def decay(self, current_turn: int) -> None:
        """Linear urgency decay over PRESS_DECAY_TURNS since claim was made."""
        elapsed = max(0, current_turn - self.turn_index)
        if elapsed >= PRESS_DECAY_TURNS:
            self.press_urgency = 0.0
        else:
            self.press_urgency = 1.0 - (elapsed / PRESS_DECAY_TURNS)

    def to_dict(self) -> dict:
        return {
            "concept":            self.concept,
            "raw_term":           self.raw_term,
            "confidence":         round(self.confidence, 3),
            "signal_type":        self.signal_type,
            "turn_index":         self.turn_index,
            "pressed":            self.pressed,
            "press_urgency":      round(self.press_urgency, 3),
            "contradiction_turn": self.contradiction_turn,
            "ts":                 self.ts,
        }

    @staticmethod
    def from_dict(d: dict) -> "CandidateClaimEntry":
        e = CandidateClaimEntry(
            concept     = str(d["concept"]),
            raw_term    = str(d["raw_term"]),
            confidence  = float(d["confidence"]),
            signal_type = str(d["signal_type"]),
            turn_index  = int(d["turn_index"]),
        )
        e.pressed            = bool(d.get("pressed", False))
        e.press_urgency      = float(d.get("press_urgency", 1.0))
        e.contradiction_turn = d.get("contradiction_turn")
        e.ts                 = float(d.get("ts", time.time()))
        return e


@dataclass
class CandidateSignalMap:
    """
    Per-session per-domain store of all answer signals extracted so far.
    Sits alongside ConceptCoverageMap in Redis.

    Responsibilities:
      - Accumulate CandidateClaimEntry records across turns
      - Track dynamic concepts discovered live from answers
      - Detect cross-turn contradictions
      - Expose ranked probe/weak lists for steering signal assembly
    """
    session_id:    str
    domain:        str
    claims:        list[CandidateClaimEntry]              = field(default_factory=list)
    concept_turns: dict[str, list[int]]                   = field(default_factory=dict)
    dynamic_concepts: dict[str, list[str]]                = field(default_factory=dict)
    claim_history: dict[str, list[tuple[int, float]]]     = field(default_factory=dict)
    contradictions: list[dict[str, Any]]                  = field(default_factory=list)
    total_answers_processed: int                          = 0
    last_updated:  float                                  = field(default_factory=time.time)

    def add_claim(self, entry: CandidateClaimEntry) -> None:
        """
        Merge new claim. If concept already claimed this session, keep the
        higher-confidence one. Flag if confidence swing is large (contradiction).
        """
        concept = entry.concept

        if concept not in self.claim_history:
            self.claim_history[concept] = []
        self.claim_history[concept].append((entry.turn_index, entry.confidence))

        if concept not in self.concept_turns:
            self.concept_turns[concept] = []
        if entry.turn_index not in self.concept_turns[concept]:
            self.concept_turns[concept].append(entry.turn_index)

        existing = [c for c in self.claims if c.concept == concept and not c.pressed]
        if existing:
            best = max(existing, key=lambda c: c.confidence)
            delta = abs(best.confidence - entry.confidence)
            if delta >= CONTRADICTION_VARIANCE_THRESHOLD:
                entry.contradiction_turn = best.turn_index
                entry.signal_type = "contradiction"
                self.contradictions.append({
                    "concept": concept,
                    "turn_a":  best.turn_index,
                    "conf_a":  round(best.confidence, 3),
                    "turn_b":  entry.turn_index,
                    "conf_b":  round(entry.confidence, 3),
                    "delta":   round(delta, 3),
                })
            if entry.confidence > best.confidence:
                best.confidence    = entry.confidence
                best.signal_type   = entry.signal_type
                best.press_urgency = 1.0
                best.ts            = entry.ts
            return

        self.claims.append(entry)
        self.last_updated = time.time()

    def register_dynamic(self, term: str, keywords: list[str]) -> None:
        if len(self.dynamic_concepts) >= MAX_DYNAMIC_CAPS:
            return
        norm = term.lower().replace("-", "_").replace(" ", "_")
        if norm not in self.dynamic_concepts:
            self.dynamic_concepts[norm] = keywords

    def mark_pressed(self, concept: str) -> None:
        for claim in self.claims:
            if claim.concept == concept:
                claim.pressed      = True
                claim.press_urgency = 0.0

    def decay_all(self, current_turn: int) -> None:
        for claim in self.claims:
            if not claim.pressed:
                claim.decay(current_turn)

    def pending_probes(self, current_turn: int) -> list[CandidateClaimEntry]:
        """
        Unpressed claims ranked by interrogation value:
          1. contradictions (story changed — highest value)
          2. slips (demonstrably confused)
          3. claimed (need to verify depth)
          4. weak (hedged, gentle follow-up)
          5. dynamic (live-discovered terms)
        Within tier: sort by press_urgency desc.
        """
        self.decay_all(current_turn)
        active = [c for c in self.claims if not c.pressed and c.press_urgency > 0.0]
        tier = {"contradiction": 0, "slip": 1, "claimed": 2, "weak": 3, "dynamic": 4}
        return sorted(
            active,
            key=lambda c: (tier.get(c.signal_type, 9), -c.press_urgency, -c.confidence),
        )

    def to_dict(self) -> dict:
        return {
            "session_id":              self.session_id,
            "domain":                  self.domain,
            "claims":                  [c.to_dict() for c in self.claims],
            "concept_turns":           self.concept_turns,
            "dynamic_concepts":        self.dynamic_concepts,
            "claim_history":           {
                k: [[ti, conf] for ti, conf in v]
                for k, v in self.claim_history.items()
            },
            "contradictions":          self.contradictions,
            "total_answers_processed": self.total_answers_processed,
            "last_updated":            self.last_updated,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_dict(d: dict) -> "CandidateSignalMap":
        m = CandidateSignalMap(
            session_id = str(d["session_id"]),
            domain     = str(d["domain"]),
        )
        m.claims       = [CandidateClaimEntry.from_dict(c) for c in d.get("claims", [])]
        m.concept_turns = {k: list(v) for k, v in d.get("concept_turns", {}).items()}
        m.dynamic_concepts = dict(d.get("dynamic_concepts", {}))
        m.claim_history = {
            k: [(int(ti), float(conf)) for ti, conf in v]
            for k, v in d.get("claim_history", {}).items()
        }
        m.contradictions          = list(d.get("contradictions", []))
        m.total_answers_processed = int(d.get("total_answers_processed", 0))
        m.last_updated            = float(d.get("last_updated", time.time()))
        return m

    @staticmethod
    def from_json(raw: str) -> "CandidateSignalMap":
        return CandidateSignalMap.from_dict(json.loads(raw))


# ══════════════════════════════════════════════════════════════════════════════
# § UPDATED ConceptSteeringSignal
# Replaces the original. Three new list fields for answer-axis signals.
# ══════════════════════════════════════════════════════════════════════════════

# Monkey-patch the existing ConceptSteeringSignal with the new fields and
# updated to_suffix_instruction. We re-declare it as a new dataclass here;
# the module-level name is rebound below. The old class is discarded.

@dataclass
class ConceptSteeringSignal:  # noqa: F811 — intentional rebind
    """
    Output of ConceptTracker.get_steering_signal().
    Consumed ONLY by LLMInputBuilder to extend difficulty_prompt_suffix.

    Six axes (two original, four new):
      avoid_concepts      — questions already covered
      focus_concepts      — registry concepts not touched
      probe_concepts      — candidate claimed, needs proving       [NEW]
      weak_concepts       — candidate hedged on these             [NEW]
      dynamic_probes      — live-discovered terms to press        [NEW]
      contradiction_flags — candidate's story changed across turns [NEW]
    """
    session_id:          str
    domain:              str
    avoid_concepts:      list[str]
    focus_concepts:      list[str]
    probe_concepts:      list[str]      = field(default_factory=list)
    weak_concepts:       list[str]      = field(default_factory=list)
    dynamic_probes:      list[str]      = field(default_factory=list)
    contradiction_flags: list[str]      = field(default_factory=list)
    coverage_ratio:      float          = 0.0
    has_signal:          bool           = False
    ts:                  float          = field(default_factory=time.time)

    def to_suffix_instruction(self) -> str:
        """
        Renders as a plain-English constraint appended to difficulty_prompt_suffix.
        This is the ONLY thing the LLM sees from this entire module.

        Priority order (highest interrogation value first):
          1. Contradiction flags — candidate's story changed, press directly
          2. Probe concepts — claimed but unproven depth
          3. Weak concepts — hedged, follow-up needed
          4. Dynamic probes — live-discovered terms
          5. Focus concepts — registry gaps
          6. Avoid concepts — already covered, skip

        Budget: ~40-80 words. Every word is load-bearing.
        """
        if not self.has_signal:
            return ""

        parts: list[str] = []

        if self.contradiction_flags:
            flags = [c.replace("_", " ") for c in self.contradiction_flags[:2]]
            parts.append(
                f"IMPORTANT: The candidate gave inconsistent answers about: "
                f"{', '.join(flags)}. Press directly on this — ask them to "
                f"clarify their understanding clearly."
            )

        all_probes = self.probe_concepts + self.dynamic_probes
        if all_probes:
            labels = [c.replace("_", " ") for c in all_probes[:MAX_PROBE_HINTS]]
            parts.append(
                f"The candidate mentioned {', '.join(labels)} — press on these "
                f"specifically to verify actual depth, not just familiarity."
            )

        if self.weak_concepts:
            labels = [c.replace("_", " ") for c in self.weak_concepts[:MAX_WEAK_HINTS]]
            parts.append(
                f"The candidate seemed uncertain about: {', '.join(labels)}. "
                f"A targeted follow-up will reveal real understanding."
            )

        if self.focus_concepts: # noqa
            labels = [c.replace("_", " ") for c in self.focus_concepts]
            parts.append(f"Topics not yet covered: {', '.join(labels)}.")

        if self.avoid_concepts:
            labels = [c.replace("_", " ") for c in self.avoid_concepts]
            parts.append(
                f"Do NOT revisit: {', '.join(labels)} — "
                f"these have already been covered this session."
            )

        return " ".join(parts)

    def is_empty(self) -> bool:
        return (
            not self.avoid_concepts
            and not self.focus_concepts
            and not self.probe_concepts
            and not self.weak_concepts
            and not self.dynamic_probes
            and not self.contradiction_flags
        )


# ══════════════════════════════════════════════════════════════════════════════
# § ANSWER SIGNAL EXTRACTOR
# Four rule-based passes over answer text. In-process only. <1ms total.
# Answer text never leaves this call.
# ══════════════════════════════════════════════════════════════════════════════

class AnswerSignalExtractor:
    """
    Extracts concept-level signals from a candidate's spoken answer.

    Pass 1 — REGISTRY CLAIM DETECTION
        Checks answer against CONCEPT_REGISTRY[domain] keywords.
        Base confidence from keyword hit density. Per-concept negation check
        prevents pressing concepts the candidate explicitly disclaimed.

    Pass 2 — HEDGE + NEGATION SCORING (integrated into Pass 1 via multiplier)
        Global hedge multiplier computed once from linguistic marker patterns.
        Strong hedges ("I'm not sure") cut confidence. Claim phrases
        ("I've built", "we use in production") boost it. Explicit negations
        ("I've never used X") suppress the concept entirely.

    Pass 3 — DYNAMIC CONCEPT DISCOVERY
        Extracts PascalCase terms, ALLCAPS acronyms, dunder attributes,
        quoted terms, and snake_case identifiers not present in registry.
        Each becomes a session-local probe target.

    Pass 4 — SLIP DETECTION
        If candidate appears to explain concept A (≥2 keywords), but the
        answer's dominant keyword density belongs to concept B, flags as
        "slip" — suggests concept confusion rather than knowledge gap.
    """

    # ── Hedge banks: (regex_pattern, confidence_multiplier) ───────────────────

    _HEDGE_STRONG: list[tuple[str, float]] = [
        (r"\bi'?m not sure\b",            0.25),
        (r"\bi don'?t know\b",            0.20),
        (r"\bi might be wrong\b",         0.25),
        (r"\bi'?m guessing\b",            0.20),
        (r"\bnot entirely sure\b",        0.25),
        (r"\bi'?m not confident\b",       0.25),
        (r"\bvaguely\b",                  0.30),
        (r"\bsomething like that\b",      0.30),
        (r"\bi'?m hazy on\b",             0.25),
        (r"\bdon'?t remember exactly\b",  0.25),
        (r"\bno idea\b",                  0.15),
        (r"\bblank on\b",                 0.20),
    ]

    _HEDGE_MEDIUM: list[tuple[str, float]] = [
        (r"\bi think\b",                  0.60),
        (r"\bi believe\b",                0.60),
        (r"\bi assume\b",                 0.55),
        (r"\bi'?m pretty sure\b",         0.65),
        (r"\bif i remember\b",            0.55),
        (r"\bif i recall\b",              0.55),
        (r"\bsomething to do with\b",     0.50),
        (r"\bkind of\b",                  0.60),
        (r"\bsort of\b",                  0.60),
        (r"\bmore or less\b",             0.65),
        (r"\balong the lines of\b",       0.55),
        (r"\bif i'?m not mistaken\b",     0.60),
        (r"\bi think it involves\b",      0.55),
    ]

    _HEDGE_SOFT: list[tuple[str, float]] = [
        (r"\bi think it'?s\b",            0.70),
        (r"\bprobably\b",                 0.72),
        (r"\bmaybe\b",                    0.70),
        (r"\bperhaps\b",                  0.72),
        (r"\bshould be\b",               0.75),
        (r"\btypically\b",               0.82),
        (r"\busually\b",                 0.82),
        (r"\bgenerally\b",               0.85),
        (r"\bmost of the time\b",        0.80),
    ]

    # Explicit disclaimer — SUPPRESS concept entirely (candidate is honest)
    _NEGATION_PATTERNS: list[str] = [
        r"\bi haven'?t (?:used|worked with|tried|seen|touched)\b",
        r"\bi don'?t (?:know|use|have experience with|understand|work with)\b",
        r"\bi'?ve never\b",
        r"\bnot familiar with\b",
        r"\bno experience with\b",
        r"\bhaven'?t really (?:used|worked with)\b",
        r"\bdon'?t really (?:know|understand)\b",
        r"\bout of my (?:depth|knowledge|area|expertise)\b",
        r"\bnever (?:used|worked with|seen|tried)\b",
    ]

    # Direct experience assertion — boost confidence
    _CLAIM_BOOST_PATTERNS: list[tuple[str, float]] = [
        (r"\bi'?(?:ve)? (?:built|implemented|written|created|designed|shipped)\b", 1.35),
        (r"\bwe (?:use|used|built|implemented|deployed)\b",                        1.20),
        (r"\bi (?:use|regularly use|work with|specialize in)\b",                   1.20),
        (r"\bin (?:production|my current project|my last job)\b",                  1.25),
        (r"\bactually\b",                                                           1.08),
        (r"\bi know\b",                                                             1.10),
        (r"\bfamiliar with\b",                                                      1.10),
        (r"\bextensively\b",                                                        1.30),
        (r"\bdeeply\b",                                                             1.25),
        (r"\bthe way it works is\b",                                                1.15),
        (r"\bwhat happens is\b",                                                    1.12),
    ]

    # Dynamic discovery patterns
    _DYNAMIC_PASCAL  = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]*)+)\b')
    _DYNAMIC_ACRONYM = re.compile(r'\b([A-Z]{2,8})\b')
    _DYNAMIC_DUNDER  = re.compile(r'(__[a-z][a-z_]+__)')
    _DYNAMIC_QUOTED  = re.compile(r'["\']([a-zA-Z][a-zA-Z0-9_\-]{2,30})["\']')
    _DYNAMIC_SNAKE   = re.compile(r'\b([a-z][a-z0-9]+(?:_[a-z][a-z0-9]+){1,4})\b')

    _COMMON_WORDS: frozenset[str] = frozenset({
        "the", "and", "that", "this", "with", "for", "from", "have", "has",
        "are", "was", "were", "been", "its", "our", "their", "can", "will",
        "would", "could", "should", "does", "did", "not", "yes", "but",
        "also", "just", "like", "then", "when", "what", "how", "why",
        "which", "where", "who", "all", "any", "some", "more", "than",
        "into", "use", "used", "using", "about", "make", "made", "take",
        "work", "works", "good", "bad", "new", "old", "get", "got", "set",
        "run", "runs", "put", "see", "say", "said", "way", "time", "part",
        "well", "even", "back", "still", "really", "def", "class", "return",
        "import", "pass", "none", "true", "false", "null", "int", "str",
        "list", "dict", "bool", "yeah", "okay", "right", "basically",
        "actually", "thing", "things", "stuff", "like", "mean", "means",
    })

    _STOP_ACRONYMS: frozenset[str] = frozenset({
        "I", "A", "OK", "API", "DB", "UI", "UX", "URL", "HTTP", "HTTPS",
        "REST", "JSON", "XML", "SQL", "AWS", "GCP", "CPU", "RAM", "SSD",
        "IDE", "CLI", "SDK", "MVP", "POC", "SLA", "SLO", "KPI", "ROI",
        "ETL", "SPA", "CMS", "CDN", "DNS", "TCP", "UDP", "IP",
    })

    @staticmethod
    def _normalize(text: str) -> str:
        t = text.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @classmethod
    def _compute_hedge_multiplier(cls, answer_lower: str) -> float:
        """
        Returns a confidence multiplier in [0.15, 1.40].
        All matched hedge/boost patterns stack multiplicatively.
        """
        mult = 1.0
        for pattern, mod in cls._HEDGE_STRONG:
            if re.search(pattern, answer_lower):
                mult *= mod
        for pattern, mod in cls._HEDGE_MEDIUM:
            if re.search(pattern, answer_lower):
                mult *= mod
        for pattern, mod in cls._HEDGE_SOFT:
            if re.search(pattern, answer_lower):
                mult *= mod
        for pattern, boost in cls._CLAIM_BOOST_PATTERNS:
            if re.search(pattern, answer_lower):
                mult *= boost
        return max(0.15, min(1.40, mult))

    @classmethod
    def _is_negated(cls, answer_lower: str) -> bool:
        return any(re.search(p, answer_lower) for p in cls._NEGATION_PATTERNS)

    @classmethod
    def _is_concept_negated(cls, concept_keywords: list[str], answer_lower: str) -> bool:
        """
        Per-concept negation: is a negation pattern within 50 chars of one of
        this concept's keywords? Prevents pressing concepts explicitly disclaimed.
        """
        for neg_pattern in cls._NEGATION_PATTERNS:
            for m in re.finditer(neg_pattern, answer_lower):
                w_start = max(0, m.start() - 50)
                w_end   = min(len(answer_lower), m.end() + 50)
                window  = answer_lower[w_start:w_end]
                if any(kw in window for kw in concept_keywords):
                    return True
        return False

    @classmethod
    def _extract_registry_claims(
        cls,
        domain:           str,
        answer:           str,
        hedge_multiplier: float,
        turn_index:       int,
    ) -> list[CandidateClaimEntry]:
        """Pass 1: registry keyword matching with per-concept negation check."""
        domain_concepts = CONCEPT_REGISTRY.get(domain, {})
        normalized      = cls._normalize(answer)
        answer_lower    = answer.lower()
        entries: list[CandidateClaimEntry] = []

        for concept_name, keywords in domain_concepts.items():
            hits = [kw for kw in keywords if kw in normalized]
            if not hits:
                continue
            if cls._is_concept_negated(keywords, answer_lower):
                continue

            base: float = min(1.0, len(hits) / max(1.0, len(keywords) * 0.4))
            confidence = max(0.0, min(1.0, base * hedge_multiplier))

            if confidence < CLAIM_DROP_THRESHOLD:
                continue

            sig_type = "claimed" if confidence >= CLAIM_STRONG_THRESHOLD else "weak"
            entries.append(CandidateClaimEntry(
                concept     = concept_name,
                raw_term    = hits[0],
                confidence  = round(confidence, 3),
                signal_type = sig_type,
                turn_index  = turn_index,
            ))

        return entries

    @classmethod
    def _extract_dynamic_concepts(
        cls,
        answer:           str,
        existing_dynamic: dict[str, list[str]],
        domain:           str,
    ) -> dict[str, list[str]]:
        """Pass 3: extract technical terms not in the static registry."""
        registry_kws: set[str] = set()
        for keywords in CONCEPT_REGISTRY.get(domain, {}).values():
            registry_kws.update(kw.lower() for kw in keywords)

        discovered: dict[str, list[str]] = {}

        def _add(raw_term: str) -> None:
            norm = raw_term.lower().replace("-", "_").replace(" ", "_")
            if norm in existing_dynamic or norm in discovered:
                return
            if norm in cls._COMMON_WORDS or len(norm) < 3:
                return
            if norm in registry_kws:
                return
            discovered[norm] = list({norm, raw_term.lower()})

        for m in cls._DYNAMIC_PASCAL.finditer(answer):
            _add(m.group(1))
        for m in cls._DYNAMIC_ACRONYM.finditer(answer):
            tok = m.group(1)
            if tok not in cls._STOP_ACRONYMS:
                _add(tok)
        for m in cls._DYNAMIC_DUNDER.finditer(answer):
            _add(m.group(1))
        for m in cls._DYNAMIC_QUOTED.finditer(answer):
            _add(m.group(1))
        for m in cls._DYNAMIC_SNAKE.finditer(answer):
            tok = m.group(1)
            if tok not in cls._COMMON_WORDS and "_" in tok:
                _add(tok)

        return discovered

    @classmethod
    def _detect_slips(
        cls,
        domain:           str,
        answer:           str,
        hedge_multiplier: float,
        turn_index:       int,
    ) -> list[CandidateClaimEntry]:
        """
        Pass 4: slip detection.
        If candidate uses ≥2 keywords for concept A, but the dominant keyword
        density in the answer is 2.5× from concept B, flag concept A as "slip".
        """
        domain_concepts = CONCEPT_REGISTRY.get(domain, {})
        if len(domain_concepts) < 2:
            return []

        normalized    = cls._normalize(answer)
        concept_hits: dict[str, int] = {}
        for cname, keywords in domain_concepts.items():
            count = sum(1 for kw in keywords if kw in normalized)
            if count >= 2:
                concept_hits[cname] = count

        if len(concept_hits) < 2:
            return []

        sorted_hits          = sorted(concept_hits.items(), key=lambda x: -x[1])
        dominant, dom_count  = sorted_hits[0]
        entries: list[CandidateClaimEntry] = []

        for cname, hit_count in sorted_hits[1:]:
            if dom_count >= hit_count * 2.5 and hit_count >= 2:
                slip_conf = min(0.65, 0.40 + 0.05 * hit_count) * hedge_multiplier
                if slip_conf < CLAIM_DROP_THRESHOLD:
                    continue
                entries.append(CandidateClaimEntry(
                    concept     = cname,
                    raw_term    = cname.replace("_", " "),
                    confidence  = round(slip_conf, 3),
                    signal_type = "slip",
                    turn_index  = turn_index,
                ))

        return entries

    @classmethod
    def process_answer(
        cls,
        domain:     str,
        answer:     str,
        question:   str, # noqa
        turn_index: int,
        signal_map: CandidateSignalMap,
    ) -> CandidateSignalMap:
        """
        Main entry point. Runs all four passes. Mutates signal_map in place.
        Fully synchronous — no I/O. Answer text never leaves this call.
        """
        if len(answer.strip()) < MIN_ANSWER_CHARS_FOR_EXTRACTION:
            signal_map.total_answers_processed += 1
            return signal_map

        answer_lower = answer.lower()

        # Global negation fast path: candidate is broadly disclaiming
        if cls._is_negated(answer_lower):
            signal_map.total_answers_processed += 1
            log.debug(
                "answer_signal_negated",
                session_id = signal_map.session_id[:8],
                domain     = domain,
                turn_index = turn_index,
            )
            return signal_map

        hedge_mult = cls._compute_hedge_multiplier(answer_lower)

        # Pass 1: registry claims
        for entry in cls._extract_registry_claims(domain, answer, hedge_mult, turn_index):
            signal_map.add_claim(entry)

        # Pass 3: dynamic discovery
        new_dynamic = cls._extract_dynamic_concepts(answer, signal_map.dynamic_concepts, domain)
        for term, keywords in new_dynamic.items():
            signal_map.register_dynamic(term, keywords)
            signal_map.add_claim(CandidateClaimEntry(
                concept     = term,
                raw_term    = term.replace("_", " "),
                confidence  = round(0.60 * hedge_mult, 3),
                signal_type = "dynamic",
                turn_index  = turn_index,
            ))

        # Pass 4: slip detection
        for entry in cls._detect_slips(domain, answer, hedge_mult, turn_index):
            signal_map.add_claim(entry)

        signal_map.total_answers_processed += 1
        signal_map.last_updated = time.time()

        log.debug(
            "answer_signal_extracted",
            session_id     = signal_map.session_id[:8],
            domain         = domain,
            turn_index     = turn_index,
            hedge_mult     = round(hedge_mult, 3),
            total_claims   = len(signal_map.claims),
            contradictions = len(signal_map.contradictions),
        )

        return signal_map


# ══════════════════════════════════════════════════════════════════════════════
# § CANDIDATE SIGNAL STORE
# Redis-backed persistence for CandidateSignalMap.
# NEVER stores answer text — only derived labels and confidence scores.
# ══════════════════════════════════════════════════════════════════════════════

class CandidateSignalStore:

    def __init__(self) -> None:
        self._redis: Any = None
        self._lock       = asyncio.Lock()
        self._local      = InMemoryLRU(max_size=int(os.getenv("CONCEPT_LRU_SIZE", "4096")))

    def _key(self, session_id: str, domain: str) -> str: # noqa
        return f"{_SIGNAL_PREFIX}{session_id}:{domain}"

    async def _get_redis(self) -> Any:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return None
        async with self._lock:
            if self._redis is None:
                try:
                    import redis.asyncio as aioredis  # type: ignore
                    self._redis = await aioredis.from_url(
                        redis_url,
                        encoding         = "utf-8",
                        decode_responses = True,
                        max_connections  = int(os.getenv("CONCEPT_REDIS_MAX_CONN", "200")),
                        socket_timeout   = 0.3,
                    )
                except Exception as exc:
                    log.warning("candidate_signal_store_redis_failed", error=str(exc))
        return self._redis

    async def load(self, session_id: str, domain: str) -> CandidateSignalMap:
        key = self._key(session_id, domain)
        raw: str | None = None
        try:
            r = await self._get_redis()
            if r:
                raw = await r.get(key)
        except Exception as exc:
            log.debug("signal_store_load_redis_error", error=str(exc))
        if raw is None:
            raw = await self._local.get(key)
        if raw:
            try:
                return CandidateSignalMap.from_json(raw)
            except Exception:  # noqa
                pass
        return CandidateSignalMap(session_id=session_id, domain=domain)

    async def save(self, signal_map: CandidateSignalMap) -> None:
        key  = self._key(signal_map.session_id, signal_map.domain)
        blob = signal_map.to_json()
        await self._local.set(key, blob)
        try:
            r = await self._get_redis()
            if r:
                await r.setex(key, _SIGNAL_TTL_S, blob)
        except Exception as exc:
            log.debug("signal_store_save_redis_error", error=str(exc))

    async def delete(self, session_id: str, domains: list[str]) -> None:
        for domain in domains:
            key = self._key(session_id, domain)
            await self._local.delete(key)
            try:
                r = await self._get_redis()
                if r:
                    await r.delete(key)
            except Exception:  # noqa
                pass


# ══════════════════════════════════════════════════════════════════════════════
# § CONCEPT TRACKER — EXTENDED
# Extends the module-level singleton with answer-signal capabilities.
# We monkey-patch the existing ConceptTracker class with new methods
# rather than redefining it, to avoid touching the existing wiring.
# ══════════════════════════════════════════════════════════════════════════════

def _build_extended_tracker() -> "ConceptTracker":
    """
    Rebuild the singleton with the new signal store and answer extractor wired in.
    Called once at module load. Returns the fully extended ConceptTracker instance.
    """

    class _ExtendedConceptTracker(ConceptTracker):
        """
        ConceptTracker extended with dual-axis signal assembly.
        Replaces the module-level singleton.
        """

        def __init__(self) -> None:
            super().__init__()
            self._signal_store     = CandidateSignalStore()
            self._answer_extractor = AnswerSignalExtractor()

        # ── Answer signal extraction (in-process path) ────────────────────────

        def extract_answer_signals(
            self,
            session_id:  str,
            domain:      str,
            answer:      str,
            question:    str,
            turn_index:  int,
            level:       str = "intermediate", # noqa
        ) -> None:
            """
            Called from commit_turn() after emit().
            Schedules async extraction + Redis persistence. Never blocks.

            ANSWER TEXT BOUNDARY: `answer` processed in-place.
            Never serialized to logs at INFO+, never sent to Kafka,
            never stored in Redis. Only derived labels are persisted.
            """
            task = asyncio.ensure_future(
                self._extract_and_save(session_id, domain, answer, question, turn_index)
            )
            task.add_done_callback(self._on_extract_done)

        @staticmethod
        def _on_extract_done(task: asyncio.Task) -> None:
            if not task.cancelled() and task.exception():
                log.debug("concept_extract_task_failed", error=str(task.exception()))

        async def _extract_and_save(
            self,
            session_id: str,
            domain:     str,
            answer:     str,
            question:   str,
            turn_index: int,
        ) -> None:
            try:
                signal_map = await self._signal_store.load(session_id, domain)
                signal_map = AnswerSignalExtractor.process_answer(
                    domain     = domain,
                    answer     = answer,
                    question   = question,
                    turn_index = turn_index,
                    signal_map = signal_map,
                )
                await self._signal_store.save(signal_map)
            except Exception as exc:
                log.debug("concept_extract_and_save_failed", error=str(exc))

        # ── Override: get_steering_signal with answer-axis merged in ──────────

        async def get_steering_signal(
            self,
            session_id: str,
            domain:     str,
            n_turns:    int = 0,
        ) -> ConceptSteeringSignal:
            """
            Merges coverage map (question axis) + signal map (answer axis).
            Both stores loaded concurrently. Always returns a valid signal.
            """
            try:
                if n_turns < MIN_TURNS_BEFORE_STEER:
                    return self._empty_signal(session_id, domain)

                coverage, signal_map = await asyncio.gather(
                    self._store.load(session_id, domain),
                    self._signal_store.load(session_id, domain),
                    return_exceptions=True,
                )

                if isinstance(coverage, Exception):
                    coverage = ConceptCoverageMap(session_id=session_id, domain=domain)
                if isinstance(signal_map, Exception):
                    signal_map = CandidateSignalMap(session_id=session_id, domain=domain)

                if coverage.total_tracked == 0 and signal_map.total_answers_processed == 0:
                    return self._empty_signal(session_id, domain)

                # Axis 1: coverage
                avoid = coverage.covered_concepts()[:MAX_AVOID_HINTS]
                focus = coverage.uncovered_concepts(domain)[:MAX_FOCUS_HINTS]

                # Axis 2: answer signals
                pending             = signal_map.pending_probes(current_turn=n_turns)
                probe_concepts      = [e.concept for e in pending if e.signal_type in ("claimed", "slip") and e.press_urgency > 0.3][:MAX_PROBE_HINTS]
                weak_concepts       = [e.concept for e in pending if e.signal_type == "weak" and e.press_urgency > 0.3][:MAX_WEAK_HINTS]
                dynamic_probes      = [e.concept for e in pending if e.signal_type == "dynamic" and e.press_urgency > 0.4][:MAX_PROBE_HINTS]
                contradiction_flags = [e.concept for e in pending if e.signal_type == "contradiction"][:2]

                total_in_domain = len(CONCEPT_REGISTRY.get(domain, {}))
                ratio           = len(coverage.covered) / total_in_domain if total_in_domain else 0.0

                has_signal = bool(avoid or focus or probe_concepts or weak_concepts or dynamic_probes or contradiction_flags)

                signal = ConceptSteeringSignal(
                    session_id          = session_id,
                    domain              = domain,
                    avoid_concepts      = avoid,
                    focus_concepts      = focus,
                    probe_concepts      = probe_concepts,
                    weak_concepts       = weak_concepts,
                    dynamic_probes      = dynamic_probes,
                    contradiction_flags = contradiction_flags,
                    coverage_ratio      = round(ratio, 3),
                    has_signal          = has_signal,
                )

                log.debug(
                    "concept_signal_built",
                    session_id     = session_id[:8],
                    domain         = domain,
                    avoid          = avoid,
                    focus          = focus,
                    probe          = probe_concepts,
                    weak           = weak_concepts,
                    dynamic        = dynamic_probes,
                    contradictions = contradiction_flags,
                    coverage_ratio = signal.coverage_ratio,
                    pending_total  = len(pending),
                )

                return signal

            except Exception as exc:
                log.debug("concept_signal_error", error=str(exc))
                return self._empty_signal(session_id, domain)

        # ── Override: evict_session — also clears signal store ────────────────

        async def evict_session(self, session_id: str, domains: list[str]) -> None:
            await asyncio.gather(
                self._store.delete(session_id, domains),
                self._signal_store.delete(session_id, domains),
                return_exceptions=True,
            )
            log.debug("concept_session_evicted", session_id=session_id[:8])

        # ── Optional precision wire ────────────────────────────────────────────

        async def mark_concept_pressed(
            self,
            session_id: str,
            domain:     str,
            concept:    str,
        ) -> None:
            """
            Optional: retire a claim immediately when a question presses it,
            rather than waiting for PRESS_DECAY_TURNS natural decay.
            Best-effort — silent on failure.
            """
            try:
                signal_map = await self._signal_store.load(session_id, domain)
                signal_map.mark_pressed(concept)
                await self._signal_store.save(signal_map)
            except Exception as exc:
                log.debug("concept_mark_pressed_failed", error=str(exc))

        # ── Extended reporting ─────────────────────────────────────────────────

        async def get_coverage_report(
            self, session_id: str, domain: str
        ) -> dict[str, Any]:
            coverage, signal_map = await asyncio.gather(
                self._store.load(session_id, domain),
                self._signal_store.load(session_id, domain),
            )
            all_concepts = ConceptExtractor.all_concepts_for_domain(domain)
            return {
                "session_id":    session_id,
                "domain":        domain,
                "covered":       coverage.covered,
                "uncovered":     [c for c in all_concepts if c not in coverage.covered],
                "total_tracked": coverage.total_tracked,
                "coverage_ratio": round(len(coverage.covered) / max(len(all_concepts), 1), 3),
                "last_updated":  coverage.last_updated,
                "claims":        [c.to_dict() for c in signal_map.claims],
                "dynamic_concepts": signal_map.dynamic_concepts,
                "contradictions":   signal_map.contradictions,
                "total_answers_processed": signal_map.total_answers_processed,
                "pending_probes": [
                    c.to_dict()
                    for c in signal_map.pending_probes(current_turn=coverage.total_tracked)
                ],
            }

        async def get_answer_signal_report(
            self, session_id: str, domain: str
        ) -> dict[str, Any]:
            signal_map = await self._signal_store.load(session_id, domain)
            return {
                "session_id":              session_id,
                "domain":                  domain,
                "total_answers_processed": signal_map.total_answers_processed,
                "claims":                  [c.to_dict() for c in signal_map.claims],
                "contradictions":          signal_map.contradictions,
                "dynamic_concepts":        signal_map.dynamic_concepts,
                "claim_history":           signal_map.claim_history,
                "last_updated":            signal_map.last_updated,
            }

        @staticmethod
        def _empty_signal(session_id: str, domain: str) -> ConceptSteeringSignal:
            return ConceptSteeringSignal(
                session_id          = session_id,
                domain              = domain,
                avoid_concepts      = [],
                focus_concepts      = [],
                probe_concepts      = [],
                weak_concepts       = [],
                dynamic_probes      = [],
                contradiction_flags = [],
                coverage_ratio      = 0.0,
                has_signal          = False,
            )

    return _ExtendedConceptTracker()


# ── Replace module-level singleton with extended version ─────────────────────
# The original `concept_tracker = ConceptTracker()` above is overridden here.
# All existing import sites (`from app.eval.concept_tracker import concept_tracker`)
# pick up this extended instance automatically — no import changes needed.

concept_tracker = _build_extended_tracker()  # noqa: F811

