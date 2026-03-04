from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any # noqa

import redis.asyncio as aioredis

from app.pgs._core import (
    BehavioralVector,
    ConstraintSet,
    SessionState,
    TurnRecord,
)

_REDIS_URL      = os.getenv("REDIS_URL")
_REDIS_CONN     = int(os.getenv("REDIS_MAX_CONN", "200"))
_TTL            = int(os.getenv("SESSION_TTL_S", "3600")) + int(os.getenv("SESSION_GRACE_S", "60"))

_PFX_STATE      = "pgs:s:v1:"
_PFX_VEC        = "pgs:v:v1:"
_PFX_CS         = "pgs:c:v1:"
_PFX_TURNS      = "pgs:t:v1:"
_PFX_WEIGHTS    = "pgs:w:v1:"

_LRU_MAX        = 128


class _LRU:
    def __init__(self, cap: int = _LRU_MAX) -> None:
        from collections import OrderedDict
        self._d: OrderedDict = OrderedDict()
        self._cap = cap

    def get(self, k: str) -> str | None:
        if k not in self._d:
            return None
        self._d.move_to_end(k)
        return self._d[k]

    def set(self, k: str, v: str) -> None:
        if k in self._d:
            self._d.move_to_end(k)
        self._d[k] = v
        if len(self._d) > self._cap:
            self._d.popitem(last=False)

    def delete(self, k: str) -> None:
        self._d.pop(k, None)


class PGSStore:

    def __init__(self) -> None:
        self._r: aioredis.Redis | None = None
        self._lock = asyncio.Lock()
        self._lru  = _LRU()

    async def _conn(self) -> aioredis.Redis | None:
        if not _REDIS_URL:
            return None
        async with self._lock:
            if self._r is None:
                self._r = await aioredis.from_url(
                    _REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=_REDIS_CONN,
                    socket_connect_timeout=0.1,
                    socket_timeout=0.25,
                )
        return self._r

    # ── session state ──────────────────────────────────────────────────────────

    async def load_state(self, session_id: str) -> SessionState | None:
        key = f"{_PFX_STATE}{session_id}"
        cached = self._lru.get(key)
        if cached:
            try:
                return self._deserialize_state(session_id, cached)
            except Exception: # noqa
                pass
        try:
            r = await self._conn()
            if r:
                raw = await r.get(key)
                if raw:
                    self._lru.set(key, raw)
                    return self._deserialize_state(session_id, raw)
        except Exception: # noqa
            pass
        return None

    async def save_state(self, state: SessionState) -> None:
        key = f"{_PFX_STATE}{state.session_id}"
        raw = self._serialize_state(state)
        self._lru.set(key, raw)
        try:
            r = await self._conn()
            if r:
                await r.setex(key, _TTL, raw)
        except Exception: # noqa
            pass

    def _serialize_state(self, s: SessionState) -> str: # noqa
        return json.dumps({
            "sl": s.stated_level,
            "dm": s.domains,
            "w":  s.weights,
            "ca": s.created_at,
        })

    def _deserialize_state(self, session_id: str, raw: str) -> SessionState: # noqa
        d = json.loads(raw)
        return SessionState(
            session_id   = session_id,
            stated_level = d["sl"],
            domains      = d["dm"],
            weights      = d.get("w", [1/8]*8),
            created_at   = d.get("ca", time.monotonic()),
        )

    # ── turn log ───────────────────────────────────────────────────────────────

    async def append_turn(self, session_id: str, turn: TurnRecord) -> None:
        key = f"{_PFX_TURNS}{session_id}"
        try:
            r = await self._conn()
            if r:
                await r.rpush(key, turn.serialize())
                await r.expire(key, _TTL)
        except Exception: # noqa
            pass

    async def patch_turn_score(
        self, session_id: str, turn_index: int, score: float
    ) -> bool:
        key = f"{_PFX_TURNS}{session_id}"
        try:
            r = await self._conn()
            if not r:
                return False
            raw_list = await r.lrange(key, 0, -1)
            for i, raw in enumerate(raw_list):
                t = TurnRecord.deserialize(raw)
                if t.turn_index == turn_index:
                    t.score = score
                    await r.lset(key, i, t.serialize())
                    return True
        except Exception: # noqa
            pass
        return False

    async def load_turns(self, session_id: str) -> list[TurnRecord]:
        key = f"{_PFX_TURNS}{session_id}"
        try:
            r = await self._conn()
            if r:
                raw_list = await r.lrange(key, 0, -1)
                turns = []
                for raw in raw_list:
                    try:
                        turns.append(TurnRecord.deserialize(raw))
                    except Exception: # noqa
                        pass
                return turns
        except Exception: # noqa
            pass
        return []

    # ── behavioral vector history ──────────────────────────────────────────────

    async def append_vector(self, session_id: str, vec: BehavioralVector) -> None:
        key = f"{_PFX_VEC}{session_id}"
        try:
            r = await self._conn()
            if r:
                await r.rpush(key, vec.serialize())
                await r.ltrim(key, -32, -1)
                await r.expire(key, _TTL)
        except Exception: # noqa
            pass

    async def load_vectors(self, session_id: str) -> list[BehavioralVector]:
        key = f"{_PFX_VEC}{session_id}"
        try:
            r = await self._conn()
            if r:
                raw_list = await r.lrange(key, 0, -1)
                vecs = []
                for raw in raw_list:
                    try:
                        vecs.append(BehavioralVector.deserialize(raw, session_id))
                    except Exception:
                        pass
                return vecs
        except Exception:
            pass
        return []

    # ── constraint set ─────────────────────────────────────────────────────────

    async def save_constraints(self, session_id: str, cs: ConstraintSet) -> None:
        key = f"{_PFX_CS}{session_id}"
        raw = cs.serialize()
        self._lru.set(key, raw)
        try:
            r = await self._conn()
            if r:
                await r.setex(key, _TTL, raw)
        except Exception:
            pass

    async def load_constraints(self, session_id: str) -> ConstraintSet | None:
        key = f"{_PFX_CS}{session_id}"
        cached = self._lru.get(key)
        if cached:
            try:
                return ConstraintSet.deserialize(cached)
            except Exception:
                pass
        try:
            r = await self._conn()
            if r:
                raw = await r.get(key)
                if raw:
                    self._lru.set(key, raw)
                    return ConstraintSet.deserialize(raw)
        except Exception:
            pass
        return None

    # ── dimension weights (feedback loop) ─────────────────────────────────────

    async def save_weights(self, session_id: str, weights: list[float]) -> None:
        key = f"{_PFX_WEIGHTS}{session_id}"
        raw = json.dumps(weights)
        self._lru.set(key, raw)
        try:
            r = await self._conn()
            if r:
                await r.setex(key, _TTL, raw)
        except Exception:
            pass

    async def load_weights(self, session_id: str) -> list[float] | None:
        key = f"{_PFX_WEIGHTS}{session_id}"
        cached = self._lru.get(key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass
        try:
            r = await self._conn()
            if r:
                raw = await r.get(key)
                if raw:
                    self._lru.set(key, raw)
                    return json.loads(raw)
        except Exception:
            pass
        return None

    # ── eviction ───────────────────────────────────────────────────────────────

    async def evict(self, session_id: str) -> None:
        keys = [
            f"{_PFX_STATE}{session_id}",
            f"{_PFX_VEC}{session_id}",
            f"{_PFX_CS}{session_id}",
            f"{_PFX_TURNS}{session_id}",
            f"{_PFX_WEIGHTS}{session_id}",
        ]
        for k in keys:
            self._lru.delete(k)
        try:
            r = await self._conn()
            if r and keys:
                await r.delete(*keys)
        except Exception:
            pass

    async def health(self) -> bool:
        try:
            r = await self._conn()
            if r:
                await r.ping()
                return True
        except Exception:
            pass
        return False
