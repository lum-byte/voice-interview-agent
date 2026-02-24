"""
IP-locked session store for the voice interview pipeline.

One active session is permitted per IP address at any given time.
A second registration attempt from the same address is rejected
until the first session completes, is ended explicitly, or expires.
Raw IPs are never stored — only an HMAC-SHA256 derived fingerprint.

Architecture
────────────
  Registration       POST /session/register
    ↓  client IP extracted from X-Forwarded-For / CF-Connecting-IP / REMOTE_ADDR
    ↓  ip_hash   = HMAC-SHA256(SESSION_SECRET_KEY, canonical_ip)
    ↓  session_id = sha256(ip_hash + os.urandom(16)) — opaque, non-reversible
    ↓  Redis SETNX  session:lock:{ip_hash}  → rejects duplicate registrations
    ↓  Redis SETEX  session:data:{session_id}  → conversation state JSON
    ↓  Redis SETEX  session:meta:{ip_hash}   → ip_hash → session_id mapping
    ↓  Response includes registered_from_ip (masked display) + session_id

  Voice turn          POST /voice (X-Session-ID header required)
    ↓  Middleware resolves session_id → validates IP still matches
    ↓  Loads conversation history → injected into LLM message chain
    ↓  After LLM responds → appends (user, assistant) turn to history
    ↓  Rolling window: oldest turns pruned beyond SESSION_MAX_TURNS

  Completion          POST /session/end  or  SESSION_TTL_S elapsed
    ↓  ip_hash lock released → IP is free to register again

Redis key layout
────────────────
  session:lock:{ip_hash}       NX sentinel, TTL = SESSION_TTL_S + SESSION_GRACE_S
  session:data:{session_id}    JSON blob,   TTL = SESSION_TTL_S + SESSION_GRACE_S
  session:meta:{ip_hash}       session_id string for lookup, same TTL

InMemoryLRU fallback activates automatically if Redis is unreachable.
Degraded mode reduces concurrency and logs a warning — it does NOT drop
the exclusive-session guarantee (LRU is process-local so cross-process
guarantees cannot be maintained; this is clearly documented below).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import time
import uuid  # noqa
from dataclasses import dataclass, field
from typing import Any, Union

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status, WebSocket
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.common.shared import (
    CircuitBreaker,
    CircuitBreakerOpen,
    InMemoryLRU,
    ServiceHealthState,
    backoff_retry,  # noqa
    degraded_mode,
    get_logger,
    get_tracer,
    make_counter,
    make_gauge,
    make_histogram,
    new_request_id,
)

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── config ─────────────────────────────────────────────────────────────────────
# All tuneable without a code change. SESSION_SECRET_KEY is required
# — the server refuses to start without it so misconfigured deployments
# fail loudly rather than silently degrading to an insecure mode.

_SECRET_KEY_RAW: str = os.environ.get("SESSION_SECRET_KEY", "")
APP_MODE: str = os.getenv("APP_MODE", "desktop")  # default = local desktop

if not _SECRET_KEY_RAW:
    if APP_MODE == "desktop":
        import secrets

        _SECRET_KEY_RAW = secrets.token_hex(32)
        log.warning("SESSION_SECRET_KEY missing — generated ephemeral desktop key.")
    else:
        raise RuntimeError(
            "SESSION_SECRET_KEY must be set in the environment. "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )

SESSION_SECRET_KEY: bytes = _SECRET_KEY_RAW.encode()

REDIS_URL: str | None = os.getenv("REDIS_URL")
REDIS_MAX_CONN: int = int(os.getenv("REDIS_MAX_CONN", "200"))

# How long a session may remain idle before it auto-expires.
SESSION_TTL_S: int = int(os.getenv("SESSION_TTL_S", "1800"))  # 30 Mins default

# Extra grace period so TTL-based expiry never races with an in-flight request.
SESSION_GRACE_S: int = int(os.getenv("SESSION_GRACE_S", "300"))

# Maximum number of (user, assistant) turn pairs kept in rolling history.
SESSION_MAX_TURNS: int = int(os.getenv("SESSION_MAX_TURNS", "20"))

# How far back a single character in the IP string may be preserved in
# the masked display returned to the client ("You are registering from
# 192.168.*.* "). Purely cosmetic — does not affect security.
_IP_MASK_OCTETS: int = int(os.getenv("SESSION_IP_DISPLAY_OCTETS", "2"))

# In-memory LRU size (per process) used when Redis is unavailable.
# Cross-process session uniqueness cannot be guaranteed in this mode.
SESSION_LRU_SIZE: int = int(os.getenv("SESSION_LRU_SIZE", "1024"))

# Maximum number of IP address changes permitted per session before it is
# terminated as suspicious. Modelled after real online exam platforms that
# allow a small number of network changes (WiFi → mobile, NAT reassignment)
# but flag and terminate sessions that change IP too many times.
# Set to 0 to disable re-binding entirely (strict mode).
SESSION_MAX_IP_CHANGES: int = int(os.getenv("SESSION_MAX_IP_CHANGES", "3"))

# Key namespace — bump the version suffix if you change the storage schema
# so old keys don't collide with new ones during rolling deploys.
_LOCK_PREFIX = "session:lock:v1:"
_DATA_PREFIX = "session:data:v1:"
_META_PREFIX = "session:meta:v1:"


# ── exceptions ──────────────────────────────────────────────────────────────────


class SessionLocked(RuntimeError):
    """
    Raised when an IP address already has an active session.

    Carries the masked IP so callers can surface a meaningful message
    without exposing the raw address in log lines or API responses.
    """

    def __init__(self, masked_ip: str) -> None:
        self.masked_ip = masked_ip
        super().__init__(f"An active session already exists for {masked_ip}.")


class SessionNotFound(RuntimeError):
    """Raised when a session_id does not exist or has expired."""


class SessionIPMismatch(RuntimeError):
    """
    Raised when the session_id is valid but the caller's IP no longer
    matches the IP that created the session and strict mode is on.
    """


class SessionIPChangeLimitExceeded(RuntimeError):
    """
    Raised when a session has exceeded SESSION_MAX_IP_CHANGES address changes.
    The session is ended automatically before this is raised.
    """


# ── Prometheus metrics ─────────────────────────────────────────────────────────

_sessions_created = make_counter("session_created_total", "New sessions registered")
_sessions_ended = make_counter(
    "session_ended_total", "Sessions ended (explicit or expiry)", ["reason"]
)
_sessions_rejected = make_counter(
    "session_rejected_total", "Registration rejections", ["reason"]
)
_turns_appended = make_counter("session_turns_total", "Conversation turns persisted")
_lru_hits = make_counter("session_lru_hits_total", "Session reads served from LRU")
_lock_conflicts = make_counter(
    "session_lock_conflicts_total", "IP already-locked conflicts"
)

_active_sessions = make_gauge("session_active", "Sessions currently holding an IP lock")
_history_turns = make_histogram(
    "session_history_turns",
    "Number of turns in history at time of LLM call",
    buckets=(0, 1, 2, 5, 10, 15, 20),
)

# ── IP utilities ───────────────────────────────────────────────────────────────

# Strip port and brackets for IPv6  ( [::1]:12345 → ::1 )
_PORT_STRIP = re.compile(r":\d+$")
_BRACKET_STRIP = re.compile(r"^\[(.+)\]$")  # noqa


def _canonical_ip(raw: str) -> str:
    """
    Normalise an IP string: strip port, brackets, and leading zeros.
    Returns the empty string if the input is unparseable.
    """
    ip = raw.strip()
    ip = _BRACKET_STRIP.sub(r"\1", ip)
    ip = _PORT_STRIP.sub("", ip)
    return ip


def _extract_client_ip(conn: Union[Request, WebSocket]) -> str:
    """
    Walk the standard proxy-header chain to find the real client IP.

    Order of precedence (highest trust first):
      CF-Connecting-IP   — Cloudflare, single real-client IP, most reliable
      X-Real-IP          — nginx proxy_set_header
      X-Forwarded-For    — leftmost entry is the original client
      request.client.host — direct connection fallback

    This covers the vast majority of deployment topologies without
    requiring framework-specific middleware.
    """
    headers = conn.headers

    for header in ("CF-Connecting-IP", "X-Real-IP"):
        val = headers.get(header, "").strip()
        if val:
            return _canonical_ip(val.split(",")[0])

    xff = headers.get("X-Forwarded-For", "").strip()
    if xff:
        return _canonical_ip(xff.split(",")[0])

    if conn.client:
        return _canonical_ip(conn.client.host or "")

    return ""


def _mask_ip(ip: str) -> str:
    """
    Return a partially masked IP for display purposes.

    IPv4: first _IP_MASK_OCTETS octets visible, rest replaced with *.
          e.g. "192.168.1.100" → "192.168.*.*"

    IPv6: first two groups visible, rest replaced with ***::
          e.g. "2001:db8::1"   → "2001:db8:***::"
    """
    if ":" in ip:
        parts = ip.split(":")
        visible = parts[:_IP_MASK_OCTETS]
        return ":".join(visible) + ":***::"
    else:
        parts = ip.split(".")
        visible = parts[:_IP_MASK_OCTETS]
        hidden = ["*"] * (len(parts) - _IP_MASK_OCTETS)
        return ".".join(visible + hidden)


# ── cryptographic helpers ──────────────────────────────────────────────────────


def _ip_hash(ip: str) -> str:
    """
    Derive a fixed-length, non-reversible fingerprint of an IP address.

    Uses HMAC-SHA256 so the same IP always produces the same hash
    (deterministic lock key) but the original address cannot be
    recovered from the hash alone. The HMAC key is SESSION_SECRET_KEY.
    """
    return hmac.new(SESSION_SECRET_KEY, ip.encode(), hashlib.sha256).hexdigest()


def _new_session_id(ip_fingerprint: str) -> str:
    """
    Derive an opaque session token.

    Combines the IP fingerprint with a cryptographically random nonce
    so different registrations from the same IP produce different tokens.
    The token cannot be reversed to the original IP.
    """
    nonce = os.urandom(16).hex()
    raw = f"{ip_fingerprint}:{nonce}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ── session data model ─────────────────────────────────────────────────────────


@dataclass
class ConversationTurn:
    """A single (user, assistant) exchange within a session."""

    user: str
    assistant: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"user": self.user, "assistant": self.assistant, "ts": self.ts}

    @staticmethod
    def from_dict(d: dict) -> "ConversationTurn":
        return ConversationTurn(
            user=d["user"],
            assistant=d["assistant"],
            ts=d.get("ts", 0.0),
        )


@dataclass
class IPChangeEvent:
    """
    A single IP address change event stored in the session audit log.

    Modelled after the event logs kept by online exam platforms: every
    network change is timestamped so reviewers can distinguish a genuine
    reconnect from suspicious token sharing across devices.
    """

    from_hash: str
    to_hash: str
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"from_hash": self.from_hash, "to_hash": self.to_hash, "ts": self.ts}

    @staticmethod
    def from_dict(d: dict) -> "IPChangeEvent":
        return IPChangeEvent(
            from_hash=d["from_hash"],
            to_hash=d["to_hash"],
            ts=d.get("ts", 0.0),
        )


@dataclass
class SessionData:
    """
    Full in-memory representation of a session — serialised to JSON for Redis.

    Fields
    ──────
    session_id        — opaque token returned to the client
    ip_hash           — HMAC fingerprint of the CURRENT IP (updated on re-bind)
    original_ip_hash  — HMAC fingerprint of the REGISTERING IP (never changes, audit trail)
    ip_change_log     — ordered list of every IP change event (exam-style audit log)
    suspended         — True when SESSION_MAX_IP_CHANGES exceeded; blocks further requests
    turns             — rolling conversation history (oldest pruned at SESSION_MAX_TURNS)
    created_at        — epoch seconds, set once at registration
    last_active       — epoch seconds, updated on every turn
    metadata          — arbitrary caller-supplied key-value pairs
    """

    session_id: str
    ip_hash: str
    original_ip_hash: str = ""
    ip_change_log: list[IPChangeEvent] = field(default_factory=list)
    suspended: bool = False
    turns: list[ConversationTurn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.original_ip_hash:
            self.original_ip_hash = self.ip_hash

    @property
    def ip_change_count(self) -> int:
        return len(self.ip_change_log)

    def to_json(self) -> str:
        return json.dumps(
            {
                "session_id": self.session_id,
                "ip_hash": self.ip_hash,
                "original_ip_hash": self.original_ip_hash,
                "ip_change_log": [e.to_dict() for e in self.ip_change_log],
                "suspended": self.suspended,
                "turns": [t.to_dict() for t in self.turns],
                "created_at": self.created_at,
                "last_active": self.last_active,
                "metadata": self.metadata,
            }
        )

    @staticmethod
    def from_json(raw: str) -> "SessionData":
        d = json.loads(raw)
        return SessionData(
            session_id=d["session_id"],
            ip_hash=d["ip_hash"],
            original_ip_hash=d.get("original_ip_hash", d["ip_hash"]),
            ip_change_log=[
                IPChangeEvent.from_dict(e) for e in d.get("ip_change_log", [])
            ],
            suspended=d.get("suspended", False),
            turns=[ConversationTurn.from_dict(t) for t in d.get("turns", [])],
            created_at=d.get("created_at", 0.0),
            last_active=d.get("last_active", 0.0),
            metadata=d.get("metadata", {}),
        )

    def prune(self) -> None:
        """Drop the oldest turns beyond the rolling window."""
        if len(self.turns) > SESSION_MAX_TURNS:
            self.turns = self.turns[-SESSION_MAX_TURNS:]

    def to_langchain_messages(self, system_prompt: str) -> list:
        """
        Build the full LangChain message list for an LLM call.

        Injecting the rolling history into the message chain gives the LLM
        context across turns — the interviewer remembers what was asked,
        the candidate's prior answers, and any hints already given.

        Format
        ──────
          SystemMessage(system_prompt)
          HumanMessage(turn[0].user)
          AIMessage(turn[0].assistant)
          ...
          HumanMessage(turn[n].user)   ← current turn added by caller
        """
        messages: list = [SystemMessage(content=system_prompt)]
        for turn in self.turns:
            # Guard: skip turns where the assistant response is empty or
            # whitespace-only.  These can arise when stream_messages raises
            # ValueError *after* yielding whitespace tokens — the whitespace
            # gets committed before the guard in voice_graph catches it.
            # An AIMessage(content="") or AIMessage(content=" \n") causes
            # modern GPT models to return empty streams on the next call,
            # creating a cascading poisoning of the conversation history.
            if not turn.assistant.strip():
                log.warning(
                    "to_langchain_messages_skipping_empty_assistant_turn",
                    session_id=self.session_id[:8],
                    user_preview=turn.user[:40],
                )
                continue
            messages.append(HumanMessage(content=turn.user))
            messages.append(AIMessage(content=turn.assistant))
        return messages


# ── session store ──────────────────────────────────────────────────────────────


class SessionStore:
    """
    Async session registry backed by Redis with automatic LRU fallback.

    Thread-safety
    ─────────────
    All public methods are coroutines safe to call from multiple asyncio
    tasks concurrently. The exclusive-session guarantee is maintained by
    Redis SETNX (atomic across processes). In degraded (LRU-only) mode,
    the guarantee is process-local — documented via a logged warning.

    Usage
    ─────
        store = SessionStore()

        # Registration (call once per candidate)
        result = await store.register(client_ip)
        session_id = result["session_id"]

        # Each voice turn
        data = await store.load(session_id, client_ip)
        messages = data.to_langchain_messages(SYSTEM_PROMPT)
        # ... run LLM with messages + current user turn appended ...
        await store.append_turn(session_id, user_text, assistant_text)

        # End of interview
        await store.end(session_id, reason="completed")
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None
        self._lru = InMemoryLRU(max_size=SESSION_LRU_SIZE)
        self._breaker = CircuitBreaker(
            name="session_store", failure_threshold=3, recovery_timeout=15.0
        )
        self._lock = asyncio.Lock()  # guards _redis initialisation only

    # ── Redis connection ───────────────────────────────────────────────────────

    async def _get_redis(self) -> aioredis.Redis | None:
        if not REDIS_URL:
            return None
        async with self._lock:
            if self._redis is None:
                self._redis = await aioredis.from_url(
                    REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=REDIS_MAX_CONN,
                    socket_connect_timeout=0.15,
                    socket_timeout=0.5,
                )
        return self._redis

    # ── generic Redis helpers with LRU fallback ────────────────────────────────

    async def _redis_get(self, key: str) -> str | None:
        try:
            r = await self._get_redis()
            if r:
                val = await self._breaker.call(r.get, key)
                if val is not None:
                    return val
                if degraded_mode.active:
                    await degraded_mode.exit("redis_readable")
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("session_redis_read_failed", key=key, error=str(exc))
            await degraded_mode.enter(f"redis_error: {exc}")

        val = await self._lru.get(key)
        if val is not None:
            _lru_hits.inc()
        return val

    async def _redis_set(self, key: str, value: str, ttl: int | None = None) -> None:
        await self._lru.set(key, value)
        try:
            r = await self._get_redis()
            if r:
                if ttl and ttl > 0:
                    await self._breaker.call(r.setex, key, ttl, value)
                else:
                    await self._breaker.call(r.set, key, value)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("session_redis_write_failed", key=key, error=str(exc))

    async def _redis_delete(self, *keys: str) -> None:
        for key in keys:
            await self._lru.set(
                key, ""
            )  # overwrite with sentinel so get() returns miss
        try:
            r = await self._get_redis()
            if r:
                await self._breaker.call(r.delete, *keys)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("session_redis_delete_failed", keys=keys, error=str(exc))

    async def _redis_setnx(self, key: str, value: str, ttl: int) -> bool:
        """
        Atomic set-if-not-exists with TTL.

        Returns True if the key was set (lock acquired),
        False if it already existed (lock held by another session).

        In degraded mode the LRU provides a best-effort check but
        cannot guarantee cross-process exclusivity.
        """
        try:
            r = await self._get_redis()
            if r:
                result = await self._breaker.call(r.set, key, value, nx=True, ex=ttl)
                return bool(result)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning(
                "session_setnx_redis_failed_using_lru",
                key=key,
                error=str(exc),
            )
            await degraded_mode.enter(f"redis_error: {exc}")

        # LRU fallback — process-local only
        existing = await self._lru.get(key)
        if existing:
            return False
        await self._lru.set(key, value)
        log.warning(
            "session_lock_lru_only",
            key=key,
            note="Cross-process exclusivity not guaranteed in degraded mode.",
        )
        return True

    # ── public API ─────────────────────────────────────────────────────────────

    async def register(
        self,
        client_ip: str,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """
        Register a new session for the given IP address.

        Returns
        ───────
        {
            "session_id":         str   — opaque token, include in X-Session-ID header
            "registered_from_ip": str   — masked display e.g. "192.168.*.*"
            "ttl_s":              int   — seconds until expiry if idle
            "max_turns":          int   — rolling history window depth
        }

        Raises
        ───────
        SessionLocked  — same IP already holds an active session
        ValueError     — IP could not be extracted or is empty
        """
        if not client_ip:
            raise ValueError("Cannot register a session without a client IP address.")

        masked = _mask_ip(client_ip)
        fingerprint = _ip_hash(client_ip)
        session_id = _new_session_id(fingerprint)
        rid = new_request_id()

        effective_ttl = SESSION_TTL_S + SESSION_GRACE_S

        with tracer.start_as_current_span("session.register") as span:
            span.set_attribute("request_id", rid)
            span.set_attribute("ip_masked", masked)

            lock_key = f"{_LOCK_PREFIX}{fingerprint}"
            acquired = await self._redis_setnx(lock_key, session_id, ttl=effective_ttl)

            if not acquired:
                _lock_conflicts.inc()
                _sessions_rejected.labels(reason="ip_locked").inc()
                log.warning(
                    "session_register_rejected_locked",
                    request_id=rid,
                    ip_masked=masked,
                )
                raise SessionLocked(masked)

            # Persist session data
            data = SessionData(
                session_id=session_id,
                ip_hash=fingerprint,
                metadata=metadata or {},
            )
            data_key = f"{_DATA_PREFIX}{session_id}"
            meta_key = f"{_META_PREFIX}{fingerprint}"

            await self._redis_set(data_key, data.to_json(), ttl=effective_ttl)
            await self._redis_set(meta_key, session_id, ttl=effective_ttl)

            _sessions_created.inc()
            _active_sessions.inc()

            log.info(
                "session_registered",
                request_id=rid,
                session_id=session_id,
                ip_masked=masked,
                ttl_s=SESSION_TTL_S,
            )

            return {
                "session_id": session_id,
                "registered_from_ip": masked,
                "ttl_s": SESSION_TTL_S,
                "max_turns": SESSION_MAX_TURNS,
            }

    async def load(self, session_id: str, client_ip: str) -> SessionData:
        """
        Load a session, automatically re-binding to a new IP if it has changed.

        Behaviour mirrors real online exam platforms:
          1. Token is valid + IP matches              → normal load, no action.
          2. Token is valid + IP changed              → atomically migrate lock to
                                                        new IP, log the change event,
                                                        persist updated session, continue.
          3. Token is valid + IP changed too many times → suspend + end the session,
                                                          raise SessionIPChangeLimitExceeded.
          4. Token not found / expired                → raise SessionNotFound.
          5. Session already suspended                → raise SessionIPChangeLimitExceeded.
          6. SESSION_MAX_IP_CHANGES == 0              → strict mode, no rebinding.

        Raises
        ───────
        SessionNotFound              — session_id unknown or expired
        SessionIPMismatch            — IP changed but strict mode is on
        SessionIPChangeLimitExceeded — too many IP changes; session terminated
        SessionLocked                — new IP already locked by another session
        """
        with tracer.start_as_current_span("session.load") as span:
            span.set_attribute("session_id", session_id)

            data_key = f"{_DATA_PREFIX}{session_id}"
            raw = await self._redis_get(data_key)

            if not raw:
                log.warning("session_not_found", session_id=session_id)
                raise SessionNotFound(
                    f"Session '{session_id[:8]}…' not found or expired."
                )

            data = SessionData.from_json(raw)

            # Reject already-suspended sessions immediately
            if data.suspended:
                log.warning("session_load_suspended", session_id=session_id)
                raise SessionIPChangeLimitExceeded(
                    f"Session '{session_id[:8]}…' was suspended due to too many IP changes."
                )

            current_fingerprint = _ip_hash(client_ip)
            ip_matched = hmac.compare_digest(data.ip_hash, current_fingerprint)

            if not ip_matched:
                # ── IP has changed — decide whether to re-bind or terminate ──

                if SESSION_MAX_IP_CHANGES == 0:
                    # Strict mode: re-binding disabled entirely
                    log.error(
                        "session_ip_mismatch_strict",
                        session_id=session_id,
                        ip_masked=_mask_ip(client_ip),
                        changes_so_far=data.ip_change_count,
                    )
                    raise SessionIPMismatch(
                        f"Session '{session_id[:8]}…' was created from a different address "
                        "and IP re-binding is disabled."
                    )

                if data.ip_change_count >= SESSION_MAX_IP_CHANGES:
                    # Too many changes — suspend and terminate (exam-style flagging)
                    data.suspended = True
                    effective_ttl = SESSION_TTL_S + SESSION_GRACE_S
                    await self._redis_set(data_key, data.to_json(), ttl=effective_ttl)
                    await self.end(session_id, reason="ip_change_limit_exceeded")
                    log.error(
                        "session_suspended_ip_limit",
                        session_id=session_id,
                        ip_masked=_mask_ip(client_ip),
                        changes=data.ip_change_count,
                        limit=SESSION_MAX_IP_CHANGES,
                    )
                    raise SessionIPChangeLimitExceeded(
                        f"Session '{session_id[:8]}…' exceeded the maximum of "
                        f"{SESSION_MAX_IP_CHANGES} IP address changes and has been terminated."
                    )

                # ── Atomic lock migration ──────────────────────────────────────
                # Without atomicity there is a window between deleting the old
                # lock and setting the new one where another process could grab
                # the new IP lock, leaving an orphaned lock on the previous IP.
                #
                # Redis path — single pipeline(transaction=True) executes all
                #              four commands as one atomic unit on the server.
                # LRU path   — asyncio is single-threaded; sequential ops safe.

                old_fingerprint = data.ip_hash
                new_fingerprint = current_fingerprint
                old_lock_key = f"{_LOCK_PREFIX}{old_fingerprint}"
                new_lock_key = f"{_LOCK_PREFIX}{new_fingerprint}"
                old_meta_key = f"{_META_PREFIX}{old_fingerprint}"
                new_meta_key = f"{_META_PREFIX}{new_fingerprint}"
                effective_ttl = SESSION_TTL_S + SESSION_GRACE_S

                # Build updated session object before writing
                change_event = IPChangeEvent(
                    from_hash=old_fingerprint,
                    to_hash=new_fingerprint,
                )
                data.ip_change_log.append(change_event)
                data.ip_hash = new_fingerprint
                data.last_active = time.time()
                serialised = data.to_json()

                try:
                    r = await self._get_redis()
                    if r:
                        async with r.pipeline(transaction=True) as pipe:
                            await pipe.delete(
                                old_lock_key, old_meta_key
                            )  # release old lock + meta
                            await pipe.set(
                                new_lock_key, session_id, ex=effective_ttl, nx=True
                            )  # acquire new lock (NX = atomic)
                            await pipe.setex(
                                data_key, effective_ttl, serialised
                            )  # persist session JSON
                            await pipe.setex(
                                new_meta_key, effective_ttl, session_id
                            )  # update meta mapping
                            results = await pipe.execute()

                        if not bool(results[1]):
                            # New IP already locked by a different session —
                            # roll back in-memory state and reject the rebind.
                            data.ip_hash = old_fingerprint
                            data.ip_change_log.pop()
                            log.warning(
                                "session_rebind_new_ip_already_locked",
                                session_id=session_id,
                                new_ip_masked=_mask_ip(client_ip),
                            )
                            raise SessionLocked(_mask_ip(client_ip))
                    else:
                        # LRU fallback — sequential is safe under asyncio
                        await self._lru.set( # noqa
                            old_lock_key, ""
                        )  # noqa # sentinel = deleted
                        await self._lru.set(old_meta_key, "")
                        await self._lru.set(new_lock_key, session_id)
                        await self._lru.set(new_meta_key, session_id)
                        await self._lru.set(data_key, serialised)

                except SessionLocked:
                    raise
                except (CircuitBreakerOpen, Exception) as exc:
                    # Redis pipeline failed — fall back to LRU for resilience
                    log.warning(
                        "session_rebind_redis_failed_using_lru",
                        session_id=session_id,
                        error=str(exc),
                    )
                    await self._lru.set(old_lock_key, "")  # noqa
                    await self._lru.set(old_meta_key, "")
                    await self._lru.set(new_lock_key, session_id)
                    await self._lru.set(new_meta_key, session_id)
                    await self._lru.set(data_key, serialised)

                log.warning(
                    "session_ip_rebound",
                    session_id=session_id,
                    new_ip_masked=_mask_ip(client_ip),
                    change_number=data.ip_change_count,
                    changes_remaining=SESSION_MAX_IP_CHANGES - data.ip_change_count,
                )

            span.set_attribute("turns_loaded", len(data.turns))
            span.set_attribute("ip_changes", data.ip_change_count)
            _history_turns.observe(len(data.turns))
            return data

    async def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> SessionData:
        """
        Append a completed (user, assistant) exchange to the session history.

        Enforces the rolling window — oldest turns beyond SESSION_MAX_TURNS
        are silently dropped. Refreshes the session TTL on every write so
        an active conversation never expires mid-interview.
        """
        with tracer.start_as_current_span("session.append_turn") as span:
            span.set_attribute("session_id", session_id)

            data_key = f"{_DATA_PREFIX}{session_id}"
            raw = await self._redis_get(data_key)
            if not raw:
                raise SessionNotFound(
                    f"Session '{session_id[:8]}…' expired before turn could be saved."
                )

            data = SessionData.from_json(raw)
            data.turns.append(
                ConversationTurn(user=user_text, assistant=assistant_text)
            )
            data.last_active = time.time()
            data.prune()

            effective_ttl = SESSION_TTL_S + SESSION_GRACE_S
            await self._redis_set(data_key, data.to_json(), ttl=effective_ttl)

            # Refresh the lock TTL so the ip_hash lock doesn't expire while
            # the session is still active (active conversation resets the clock)
            lock_key = f"{_LOCK_PREFIX}{data.ip_hash}"
            try:
                r = await self._get_redis()
                if r:
                    await r.expire(lock_key, effective_ttl)
            except Exception:  # noqa — best-effort TTL refresh
                pass

            _turns_appended.inc()
            span.set_attribute("total_turns", len(data.turns))
            log.info(
                "session_turn_appended",
                session_id=session_id,
                total_turns=len(data.turns),
                user_chars=len(user_text),
                assistant_chars=len(assistant_text),
            )
            return data

    async def end(self, session_id: str, reason: str = "explicit") -> None:
        """
        Terminate a session and release the IP lock.

        Safe to call even if the session has already expired — redundant
        deletes are silently ignored by Redis.
        """
        with tracer.start_as_current_span("session.end") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("reason", reason)

            data_key = f"{_DATA_PREFIX}{session_id}"
            raw = await self._redis_get(data_key)

            if raw:
                data = SessionData.from_json(raw)
                lock_key = f"{_LOCK_PREFIX}{data.ip_hash}"
                meta_key = f"{_META_PREFIX}{data.ip_hash}"
                await self._redis_delete(data_key, lock_key, meta_key)
                _active_sessions.dec()
                log.info(
                    "session_ended",
                    session_id=session_id,
                    reason=reason,
                    turns=len(data.turns),
                    duration_s=round(time.time() - data.created_at, 1),
                )
            else:
                log.info("session_end_no_data", session_id=session_id, reason=reason)

            _sessions_ended.labels(reason=reason).inc()

    async def health(self) -> ServiceHealthState:
        """Return store health for VoiceGraph routing decisions."""
        try:
            r = await self._get_redis()
            if r:
                await r.ping()
                healthy = True
            else:
                healthy = not degraded_mode.active
        except Exception:  # noqa
            healthy = False

        return ServiceHealthState(
            service="session_store",
            healthy=healthy,
            circuit_state=self._breaker.state,
            inflight=0,
            degraded=degraded_mode.active,
        )

    async def close(self) -> None:
        """Graceful shutdown — close the Redis connection pool."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
        log.info("session_store_closed")


# ── module-level singleton ─────────────────────────────────────────────────────
# Shared across all requests in the process — same pattern as voice_graph,
# stt_node, llm_node etc. Import this from any module that needs sessions.

session_store = SessionStore()


# ── FastAPI utilities ──────────────────────────────────────────────────────────


def _extract_session_id(request: Request) -> str:
    """
    Pull the session token from the X-Session-ID header.

    Using a custom header (rather than a cookie or query param) keeps the
    token out of server logs and browser history, which is the right
    default for an interview platform.
    """
    sid = request.headers.get("X-Session-ID", "").strip()
    if not sid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": "missing_session",
                "message": (
                    "No session token found. "
                    "Register first via POST /session/register, "
                    "then include the returned session_id in the X-Session-ID header."
                ),
            },
        )
    return sid


async def require_session(request: Request) -> SessionData:
    """
    FastAPI dependency — resolves and validates the caller's session.

    IP changes within SESSION_MAX_IP_CHANGES are handled transparently (re-bind).
    Raises HTTP 401 if the header is missing.
    Raises HTTP 403 if the session was suspended or new IP already locked.
    Raises HTTP 404 if the session has expired or never existed.
    """
    session_id = _extract_session_id(request)
    client_ip = _extract_client_ip(request)

    try:
        return await session_store.load(session_id, client_ip)
    except SessionNotFound as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "session_not_found", "message": str(exc)},
        )
    except SessionIPMismatch as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "ip_mismatch", "message": str(exc)},
        )
    except SessionIPChangeLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "session_suspended",
                "message": str(exc),
                "action": "Your session has been terminated. Please register a new session via POST /session/register.",
            },
        )
    except SessionLocked as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "session_locked",
                "message": f"Your new IP address ({exc.masked_ip}) is already in use by another session.",
            },
        )


# ── FastAPI route handlers ─────────────────────────────────────────────────────
# Wire these into your FastAPI app:
#
#   from app.session.session_store import session_router
#   app.include_router(session_router, prefix="/session", tags=["session"])

from fastapi import APIRouter  # noqa: E402 — after class definitions

session_router = APIRouter()


@session_router.post("/register", summary="Register a new session")
async def register_session(request: Request) -> dict[str, Any]:
    """
    Register a new interview session.

    Extracts your IP address automatically from the incoming request —
    you do not need to provide it manually. One session per IP address
    is allowed at a time. If a session is already active for your address,
    this call returns HTTP 409 until that session ends or expires.

    Response
    ────────
    {
        "session_id":         "<opaque token>",
        "registered_from_ip": "192.168.*.*",   ← your IP, partially masked
        "ttl_s":              3600,
        "max_turns":          20,
        "instructions":       "Include session_id in X-Session-ID header for all /voice calls."
    }
    """
    client_ip = _extract_client_ip(request)

    if not client_ip:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ip_unresolvable",
                "message": (
                    "Your IP address could not be determined from the request. "
                    "This can happen behind certain proxy configurations. "
                    "Contact support if this persists."
                ),
            },
        )

    try:
        result = await session_store.register(client_ip)
    except SessionLocked as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "session_locked",
                "message": (
                    f"An active interview session already exists for {exc.masked_ip}. "
                    "Only one session per IP is permitted at a time. "
                    "End your current session via POST /session/end, or wait for it to expire."
                ),
            },
        )

    return {
        **result,
        "instructions": (
            "Your session is active. Include the session_id value in the "
            "X-Session-ID header for every POST /voice call."
        ),
    }


@session_router.post("/end", summary="End the current session")
async def end_session(request: Request) -> dict[str, Any]:
    """
    Explicitly terminate your session and release the IP lock.

    After calling this, the same IP address may immediately register a
    new session. If you don't call this, the session will auto-expire
    after SESSION_TTL_S seconds of inactivity.
    """
    session_id = _extract_session_id(request)
    client_ip = _extract_client_ip(request)

    # Validate ownership before allowing termination
    try:
        await session_store.load(session_id, client_ip)
    except SessionNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail={"error": "session_not_found"}
        )
    except SessionIPMismatch:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail={"error": "ip_mismatch"}
        )

    await session_store.end(session_id, reason="explicit_end")

    return {
        "status": "session_ended",
        "session_id": session_id,
        "message": "Your session has been ended. You may register a new session at any time.",
    }


@session_router.get("/status", summary="Query current session status")
async def session_status(
    session: SessionData = Depends(require_session),
) -> dict[str, Any]:
    """
    Return the current session's status, conversation summary, and IP audit log.

    Useful for client-side UI to show turn count, elapsed time, history
    depth, and network change events without re-running the pipeline.
    """
    return {
        "session_id": session.session_id,
        "turns": len(session.turns),
        "max_turns": SESSION_MAX_TURNS,
        "created_at": session.created_at,
        "last_active": session.last_active,
        "elapsed_s": round(time.time() - session.created_at, 1),
        "suspended": session.suspended,
        "ip_changes": session.ip_change_count,
        "ip_changes_remaining": max(
            0, SESSION_MAX_IP_CHANGES - session.ip_change_count
        ),
        "ip_change_log": [e.to_dict() for e in session.ip_change_log],
        "metadata": session.metadata,
    }


@session_router.get(
    "/health", summary="Session store health check", include_in_schema=False
)
async def session_health() -> dict[str, Any]:
    state = await session_store.health()
    return {
        "service": state.service,
        "healthy": state.healthy,
        "circuit_state": state.circuit_state,
        "degraded": state.degraded,
    }


# ── LLM integration helper ─────────────────────────────────────────────────────


def build_messages_with_history(
    session: SessionData,
    current_user_input: str,
    system_prompt: str,
) -> list:
    """
    Build the full LangChain message list for an LLM call, injecting the
    session's rolling conversation history before the current user turn.

    This is the single integration point between SessionStore and LLMNode.
    Call this in place of the bare [SystemMessage, HumanMessage] pair that
    LLM_service.py currently builds in generate() and stream():

    In LLM_service.py generate():
        # Before (stateless):
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        # After (stateful):
        from app.session.session_store import build_messages_with_history
        messages = build_messages_with_history(session, prompt, SYSTEM_PROMPT)

    Returns
    ───────
        [
            SystemMessage(system_prompt),
            HumanMessage(turn[0].user),    # oldest remembered turn
            AIMessage(turn[0].assistant),
            ...
            HumanMessage(turn[n].user),    # most recent completed turn
            AIMessage(turn[n].assistant),
            HumanMessage(current_user_input),   # ← this turn, not yet answered
        ]
    """
    messages = session.to_langchain_messages(system_prompt)
    messages.append(HumanMessage(content=current_user_input))
    return messages