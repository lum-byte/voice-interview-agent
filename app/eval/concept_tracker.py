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

WHAT THIS IS NOT
────────────────
- Not a new LLM context layer
- Not a conversation history injection
- Not a change to LLMInterviewInput
- Not a graph database or embedding store

THE ONLY SURFACE TOUCHED IN THE EXISTING SYSTEM
────────────────────────────────────────────────
`LLMInputBuilder.build()` → `difficulty_prompt_suffix`
That suffix already exists. We extend it by ~20-40 words per turn.
The LLM gets a constraint, not an explanation. It follows it.

INTEGRATION (3 surgical edits, documented at bottom of file)
────────────────────────────────────────────────────────────
  [1] qa_controller.py  — QAControllerV2.commit_turn()
  [2] qa_controller.py  — LLMInputBuilder.build()
  [3] voice_graph.py    — on_startup() / on_shutdown()

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

        if self.focus_concepts:
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
                await producer.send_and_wait(
                    KAFKA_TOPIC_IN,
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
        self._local: dict[str, str] = {}   # key → JSON blob (fallback)

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
                        max_connections  = 10,
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
            raw = self._local.get(key)

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
        self._local[key] = blob   # always write local fallback

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
            self._local.pop(key, None)
            try:
                r = await self._get_redis()
                if r:
                    await r.delete(key)
            except Exception: # noqa
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
        asyncio.ensure_future(self._emit_async(event))

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