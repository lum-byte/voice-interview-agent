from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pgs._core      import BehavioralVector, ConstraintSet
from pgs._engine    import PsychographicEngine
from pgs._compiler  import ConstraintCompiler
from pgs._feedback  import FeedbackLoop
from pgs._store     import PGSStore
from pgs._core      import diff_numeric
from pgs._signals._deflection import DeflectionEngine, TurnRecord

if TYPE_CHECKING:
    pass


class _PGS:
    """
    Public interface. Three methods. Everything else is inaccessible.

    ingest()         — called after commit_turn(), fire-and-forget
    push_score()     — called when eval_engine returns a score
    get_constraints()— called in get_llm_input(), returns ConstraintSet
    open_session()   — called at session start
    evict_session()  — called at session close
    """

    def __init__(self) -> None:
        self._store    = PGSStore()
        self._engine   = PsychographicEngine(self._store)
        self._compiler = ConstraintCompiler()
        self._feedback = FeedbackLoop()
        self._deflect  = DeflectionEngine()
        self._locks:   dict[str, asyncio.Lock] = {}

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
        await self._engine.open_session(session_id, stated_level, domains)

    async def ingest(
        self,
        *,
        session_id:   str,
        turn_index:   int,
        domain:       str,
        level:        str,
        question:     str,
        answer:       str,
        answer_words: int,
        duration_s:   float,
        difficulty:   float | str,
        score:        float | None = None,
    ) -> None:
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
        self._deflect.ingest(session_id, turn)

        await self._engine.ingest(
            session_id   = session_id,
            turn_index   = turn_index,
            domain       = domain,
            level        = level,
            question     = question,
            answer       = answer,
            answer_words = answer_words,
            duration_s   = duration_s,
            difficulty   = difficulty,
            score        = score,
        )

        vec = await self._engine.current_vector(session_id)
        cs  = await self.get_constraints(
            session_id     = session_id,
            domain         = domain,
            n_turns        = turn_index + 1,
            stated_level   = level,
            active_domain  = domain,
        )
        self._feedback.record(session_id, turn_index, vec, cs)

    async def push_score(
        self,
        *,
        session_id:  str,
        turn_index:  int,
        score:       float,
        domain:      str,
    ) -> None:
        await self._engine.push_score(session_id, turn_index, score, domain)

        new_weights = self._feedback.push_score(session_id, turn_index, score)
        if new_weights is not None:
            await self._engine.update_weights(session_id, new_weights)

    async def get_constraints(
        self,
        *,
        session_id:    str,
        domain:        str,
        n_turns:       int,
        stated_level:  str,
        active_domain: str,
    ) -> ConstraintSet:
        if n_turns < 3:
            return ConstraintSet.neutral()

        vec = await self._engine.current_vector(session_id)

        if vec.v == (0.5,) * 8:
            cached = await self._store.load_constraints(session_id)
            if cached:
                return cached
            return ConstraintSet.neutral()

        context = self._engine.signal_context(session_id)

        last_turn_domain = domain
        deflection_type = "none"
        turns = await self._store.load_turns(session_id)
        if turns:
            last = sorted(turns, key=lambda t: t.turn_index)[-1]
            from pgs._signals._deflection import DeflectionEngine as _DE
            deflection_type = self._deflect.deflection_type(session_id, last)

        cs = self._compiler.compile(
            vec             = vec,
            context         = context,
            n_turns         = n_turns,
            stated_level    = stated_level,
            active_domain   = active_domain,
            deflection_type = deflection_type,
        )

        await self._store.save_constraints(session_id, cs)
        return cs

    async def evict_session(self, session_id: str) -> None:
        self._deflect.evict(session_id)
        self._feedback.evict(session_id)
        self._locks.pop(session_id, None)
        await self._engine.evict_session(session_id)

    async def health(self) -> bool:
        return await self._store.health()


pgs = _PGS()
