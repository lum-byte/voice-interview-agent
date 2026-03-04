from __future__ import annotations

import asyncio
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import (
    BehavioralVector,
    SessionState,
    TurnRecord,
    _IW,
    _TM,
    _n,
    diff_numeric,
    level_numeric,
)
from app.pgs._signals._velocity     import VelocityEngine
from app.pgs._signals._elaboration  import ElaborationEngine
from app.pgs._signals._hedge        import HedgeEngine
from app.pgs._signals._deflection   import DeflectionEngine
from app.pgs._signals._recovery     import RecoveryEngine
from app.pgs._signals._consistency  import ConsistencyEngine
from app.pgs._signals._comfort      import ComfortEngine
from app.pgs._signals._probe        import ProbeOracle
from app.pgs._store                 import PGSStore

if TYPE_CHECKING:
    pass

_MOMENTUM_ALPHA     = 0.60
_ENTROPY_FLOOR      = 0.15
_MIN_TURNS_ACTIVE   = 3


@dataclass
class _EpistemicMomentum:
    """
    Tracks rate of behavioral vector change.
    Fast change = candidate adapting to system.
    Slow change = stable pattern — _signals are converging.
    """
    history:    list[float] = field(default_factory=list)
    velocity:   float       = 0.0
    prev_vec:   BehavioralVector | None = None

    def update(self, vec: BehavioralVector) -> None:
        if self.prev_vec is not None:
            delta = vec.delta(self.prev_vec)
            self.history.append(delta)
            if len(self.history) > 12:
                self.history.pop(0)
            if self.velocity == 0.0:
                self.velocity = delta
            else:
                self.velocity = _MOMENTUM_ALPHA * delta + (1 - _MOMENTUM_ALPHA) * self.velocity
        self.prev_vec = vec

    def signal(self) -> float:
        if not self.history:
            return 0.5
        v = math.tanh(self.velocity * 6.0) * 0.45
        return max(0.0, min(1.0, 0.5 + v))


class PsychographicEngine:
    """
    Single aggregator. Owns all signal engines. Produces BehavioralVector.
    One public interface. Everything else is sealed.
    """

    def __init__(self, store: PGSStore) -> None:
        self._store = store

        self._velocity    = VelocityEngine()
        self._elaboration = ElaborationEngine()
        self._hedge       = HedgeEngine()
        self._deflection  = DeflectionEngine()
        self._recovery    = RecoveryEngine()
        self._consistency = ConsistencyEngine()
        self._comfort     = ComfortEngine()
        self._probe       = ProbeOracle()

        self._momentum:    dict[str, _EpistemicMomentum]       = defaultdict(_EpistemicMomentum)
        self._sessions:    dict[str, SessionState]             = {}
        self._locks:       dict[str, asyncio.Lock]             = {}
        self._turn_counts: dict[str, int]                      = defaultdict(int)
        # stores (answer_words, difficulty) keyed by (session_id, turn_index)
        # required so push_score can forward to probe without re-reading Redis
        self._turn_meta:   dict[tuple[str, int], tuple[int, float]] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def open_session(
        self,
        session_id:   str,
        stated_level: str,
        domains:      list[str],
    ) -> None:
        async with self._lock(session_id):
            state = SessionState(
                session_id   = session_id,
                stated_level = stated_level,
                domains      = domains,
            )
            self._sessions[session_id] = state
            await self._store.save_state(state)

    async def ingest(
        self,
        session_id:    str,
        turn_index:    int,
        domain:        str,
        level:         str,
        question:      str,
        answer:        str,
        answer_words:  int,
        duration_s:    float,
        difficulty:    float | str,
        score:         float | None = None,
    ) -> None:
        state = await self._ensure_state(session_id, level, [domain])

        turn = TurnRecord(
            session_id   = session_id,
            turn_index   = turn_index,
            domain       = domain,
            level        = level,
            question     = question,
            answer       = answer,
            answer_words = answer_words,
            duration_s   = duration_s,
            difficulty   = diff_numeric(difficulty),
            score        = score,
        )

        async with self._lock(session_id):
            self._velocity.ingest(session_id, turn)
            self._elaboration.ingest(session_id, turn)
            self._hedge.ingest(session_id, turn)
            self._deflection.ingest(session_id, turn)
            self._comfort.ingest(session_id, turn)
            self._turn_counts[session_id] += 1

            # store lightweight meta for probe oracle — avoids Redis round-trip
            self._turn_meta[(session_id, turn_index)] = (
                answer_words,
                diff_numeric(difficulty),
            )
            # cap to avoid unbounded growth on very long sessions
            if len(self._turn_meta) > 2048:
                oldest = sorted(self._turn_meta.keys())[:256]
                for k in oldest:
                    self._turn_meta.pop(k, None)

            await self._store.append_turn(session_id, turn)

            if self._turn_counts[session_id] >= _MIN_TURNS_ACTIVE:
                vec = self._compute_vector(session_id, state)
                self._momentum[session_id].update(vec)
                await self._store.append_vector(session_id, vec)

    async def push_score(
        self,
        session_id:  str,
        turn_index:  int,
        score:       float,
        domain:      str,
    ) -> None:
        async with self._lock(session_id):
            self._recovery.push_score(session_id, turn_index, score, domain)
            self._consistency.push_score(session_id, turn_index, score, domain)

            meta = self._turn_meta.get((session_id, turn_index))
            if meta is not None:
                answer_words, difficulty = meta
                self._probe.push_score(
                    session_id   = session_id,
                    turn_index   = turn_index,
                    score        = score,
                    answer_words = answer_words,
                    difficulty   = difficulty,
                    domain       = domain,
                )

            await self._store.patch_turn_score(session_id, turn_index, score)

            state = self._sessions.get(session_id)
            if state and self._turn_counts[session_id] >= _MIN_TURNS_ACTIVE:
                vec = self._compute_vector(session_id, state)
                self._momentum[session_id].update(vec)
                await self._store.append_vector(session_id, vec)

    def _compute_vector(
        self, session_id: str, state: SessionState
    ) -> BehavioralVector:
        """
        All computation sealed here. Variable names intentionally opaque.
        The only place the eight dimensions are assembled.
        """
        a = self._velocity.compute(session_id)
        b = self._elaboration.compute(session_id)
        c = self._hedge.compute(session_id)
        d = self._deflection.compute(session_id)
        e = self._recovery.compute(session_id)
        f = self._consistency.compute(session_id)
        g = self._comfort.compute(session_id)
        h = self._momentum[session_id].signal()

        raw = [a, b, c, d, e, f, g, h]

        # apply pairwise interference
        adjusted: list[float] = []
        for i in range(8):
            interference = sum(
                _IW[i][j] * (raw[j] - 0.5) for j in range(8) if j != i
            )
            adjusted.append(max(0.0, min(1.0, raw[i] + interference * 0.12)))

        # weight by session-learned weights
        w = state.weights
        weighted = [
            max(0.0, min(1.0, adjusted[i] * (0.6 + w[i] * 3.2)))
            for i in range(8)
        ]

        # add deployment-specific noise before returning
        noisy = tuple(_n(x) for x in weighted)

        return BehavioralVector(v=noisy, session_id=session_id)

    async def current_vector(self, session_id: str) -> BehavioralVector:
        state = self._sessions.get(session_id)
        if state and self._turn_counts[session_id] >= _MIN_TURNS_ACTIVE:
            return self._compute_vector(session_id, state)
        return BehavioralVector.neutral(session_id)

    async def _ensure_state(
        self, session_id: str, level: str, domains: list[str]
    ) -> SessionState:
        if session_id in self._sessions:
            return self._sessions[session_id]
        state = await self._store.load_state(session_id)
        if state is None:
            state = SessionState(
                session_id   = session_id,
                stated_level = level,
                domains      = domains,
            )
            await self._store.save_state(state)
        self._sessions[session_id] = state
        return state

    async def update_weights(self, session_id: str, weights: list[float]) -> None:
        state = self._sessions.get(session_id)
        if state:
            state.weights = weights
            await self._store.save_weights(session_id, weights)

    async def get_weights(self, session_id: str) -> list[float]:
        w = await self._store.load_weights(session_id)
        return w if w else [1/8] * 8

    def signal_context(self, session_id: str) -> dict:
        """
        Returns contextual _signals for the compiler.
        No psychological labels — only behavioral properties.
        """
        return {
            "pressure":       self._velocity.pressure_classification(session_id),
            "verbosity":      self._elaboration.verbosity_profile(session_id),
            "hedge_rising":   self._hedge.rising(session_id),
            "in_drop":        self._recovery.currently_in_drop(session_id),
            "drop_depth":     self._recovery.drop_depth(session_id),
            "warmup":         self._consistency.warmup_detected(session_id),
            "fatigue":        self._consistency.fatigue_detected(session_id),
            "comfort_top":    self._comfort.most_comfortable(session_id),
            "comfort_bot":    self._comfort.least_comfortable(session_id),
            "domain_ranks":   self._comfort.comfort_ranking(session_id),
            "hedge_domains":  self._hedge.domain_hedge_rank(session_id),
            "strongest":      self._consistency.strongest_domain(session_id),
            "weakest":        self._consistency.weakest_domain(session_id),
            # probe oracle _signals — event-class, not continuous
            "should_verify":  self._probe.should_verify(session_id),
            "probe_variance": self._probe.compute(session_id),
            "last_suspect":   self._probe.last_suspect_index(session_id),
        }

    async def evict_session(self, session_id: str) -> None:
        self._velocity.evict(session_id)
        self._elaboration.evict(session_id)
        self._hedge.evict(session_id)
        self._deflection.evict(session_id)
        self._recovery.evict(session_id)
        self._consistency.evict(session_id)
        self._comfort.evict(session_id)
        self._probe.evict(session_id)
        self._momentum.pop(session_id, None)
        self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        self._turn_counts.pop(session_id, None)
        # evict all turn_meta entries for this session
        stale = [k for k in self._turn_meta if k[0] == session_id]
        for k in stale:
            self._turn_meta.pop(k, None)
        await self._store.evict(session_id)
