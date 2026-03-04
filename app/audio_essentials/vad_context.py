"""
vad_context.py — Interview-semantic context injection for PCMVADGate.

Overview
────────
PCMVADGate is a purely signal-processing component.  This module adds a
side-channel that lets the interview orchestration layer bias its hangover
and min-speech parameters to match the expected answer type — without
touching the signal logic at all.

The design principle: biasing, not overriding.  If the audio energy
contradicts the hint (candidate finishes early, stays quiet), the onset/
hangover energy thresholds still do the right thing.  The hint only adjusts
*how long* the VAD waits before it decides silence is silence.

Components
──────────
  VADContextHint        — immutable, validated hint dataclass
  VADBiasProfile        — named (hangover_s, min_speech_s) pair with bounds
  VADParameterTable     — priority-ordered mapping: hint fields → VADBiasProfile
  VADContextHistory     — bounded ring of applied hints for post-mortem diagnostics
  VADContextHintBuilder — factory: QA-layer state dicts → VADContextHint
  ContextGatedVAD       — PCMVADGate subclass; applies hints atomically at onset

Integration
───────────
After ``node_llm`` commits a turn and ``CommittedTurn`` is available::

    hint = VADContextHintBuilder.from_committed_turn(committed_turn)
    vad_gate.apply_context(hint)   # two lines, zero signal-path changes

The hint is queued as ``_pending_hint`` and consumed exactly once at the
next SILENCE→SPEECH onset, so there is no race between a mid-segment
parameter change and a running hangover countdown.

Thread safety
─────────────
``apply_context`` is safe to call from the same asyncio event-loop thread
as ``stream()``.  If you call it from a threadpool executor (e.g., a
sync WebSocket callback), use::

    loop.call_soon_threadsafe(vad_gate.apply_context, hint)
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import collections
import dataclasses # noqa
import enum
import logging # noqa
import threading
import time
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, NamedTuple, Optional, Sequence, Callable # noqa

# ── internal ──────────────────────────────────────────────────────────────────
# Import only what we strictly need from the audio engine.
# The PCMVADGate import is deferred inside ContextGatedVAD to avoid circular
# imports when this module is loaded early during app startup.
from app.audio_essentials.audio_engine import (
    PCMChunk,
    PCMFormat,
    PCMVADGate,
    _VADState,
    _VAD_HANGOVER_S,
    _VAD_MIN_SPEECH_S,
    _VAD_ONSET_RMS,
    _VAD_OFFSET_RMS,
    _VAD_PRE_ROLL_S,
)
from app.common.shared import get_tracer, make_counter, make_gauge, make_histogram
from app.monitoring.observability import get_logger

log = get_logger(__name__)
tracer = get_tracer(__name__)

# ── Prometheus metrics ────────────────────────────────────────────────────────

_vad_context_updates = make_counter(
    "pcm_vad_context_updates_total",
    "VAD context hint applications at segment onset",
    ["qa_stage", "scaler_action"],
)
_vad_context_pending_overwritten = make_counter(
    "pcm_vad_context_pending_overwritten_total",
    "Pending hints overwritten before consumption (rapid turn commits)",
)
_vad_context_domain_switch = make_counter(
    "pcm_vad_context_domain_switches_total",
    "Domain-switch hints applied (no STT expected)",
)
_vad_context_hangover_s = make_histogram(
    "pcm_vad_context_hangover_s",
    "Hangover duration applied per context update (seconds)",
    buckets=(0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0),
)
_vad_context_history_depth = make_gauge(
    "pcm_vad_context_history_depth",
    "Number of hints stored in VADContextHistory",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. ENUMS — stable string values used in hint fields
# ──────────────────────────────────────────────────────────────────────────────

class QAStage(str, enum.Enum):
    """Interview pipeline stage, maps directly to voice_graph session state."""
    GREETING   = "greeting"
    INTRO      = "intro"
    INTERVIEW  = "interview"
    COMPLETE   = "complete"

    @classmethod
    def _missing_(cls, value: object) -> "QAStage":
        # Tolerant parsing: unknown stages fall back to INTERVIEW so that
        # new stages added to voice_graph don't hard-crash the VAD layer.
        log.warning("vad_context_unknown_qa_stage", value=value)
        return cls.INTERVIEW


class ScalerActionKind(str, enum.Enum):
    """
    Maps to ScalerAction.value from the QA controller.

    Only the values that affect VAD tuning are enumerated here.  Unknown
    values are resolved to PROBE_VERIFY (medium hangover) as a safe default.
    """
    PROBE_VERIFY   = "probe_verify"
    PROBE_LATERAL  = "probe_lateral"
    COAST          = "coast"
    ESCALATE       = "escalate"
    DEESCALATE     = "deescalate"
    CLOSE          = "close"
    BRIDGE         = "bridge"        # domain switch — no STT answer expected

    @classmethod
    def _missing_(cls, value: object) -> "ScalerActionKind":
        log.warning("vad_context_unknown_scaler_action", value=value)
        return cls.PROBE_VERIFY


class AnswerDomain(str, enum.Enum):
    """
    Answer domain, loosely maps to interview question categories.
    Used as a secondary lookup key in VADParameterTable.
    """
    SYSTEM_DESIGN  = "system_design"
    BEHAVIORAL     = "behavioral"
    CODING         = "coding"
    ALGORITHM      = "algorithm"
    TRIVIA         = "trivia"
    UNKNOWN        = "unknown"

    @classmethod
    def _missing_(cls, value: object) -> "AnswerDomain":
        return cls.UNKNOWN


# ──────────────────────────────────────────────────────────────────────────────
# 2. VADBiasProfile — named (hangover_s, min_speech_s) pair
# ──────────────────────────────────────────────────────────────────────────────

# Absolute bounds enforced on any profile, regardless of source.
_HANGOVER_MIN_S: Final[float] = 0.10   # below this we'd clip mid-word pauses
_HANGOVER_MAX_S: Final[float] = 2.50   # above this STT latency becomes user-visible
_MIN_SPEECH_MIN_S: Final[float] = 0.05
_MIN_SPEECH_MAX_S: Final[float] = 1.00


@dataclass(frozen=True)
class VADBiasProfile:
    """
    A named VAD parameter profile.

    All values are clamped to physically meaningful bounds at construction
    time so callers cannot accidentally set a 10-second hangover.

    Attributes
    ──────────
    name            Human-readable label for logs and metrics.
    hangover_s      How long to wait after sub-threshold audio before ending
                    the segment.  Longer = more tolerant of mid-answer pauses.
    min_speech_s    Discard segments shorter than this.  Should be shorter
                    than hangover_s to avoid swallowing short interruptions.
    description     Optional documentation string stored for diagnostics.
    """

    name:          str
    hangover_s:    float
    min_speech_s:  float
    description:   str = ""

    # ── validation ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        # frozen=True means we must use object.__setattr__ for clamping.
        clamped_h = float(
            max(_HANGOVER_MIN_S, min(_HANGOVER_MAX_S, self.hangover_s))
        )
        clamped_m = float(
            max(_MIN_SPEECH_MIN_S, min(_MIN_SPEECH_MAX_S, self.min_speech_s))
        )
        if clamped_h != self.hangover_s:
            log.warning(
                "vad_bias_profile_hangover_clamped",
                name=self.name,
                requested=self.hangover_s,
                clamped=clamped_h,
            )
        if clamped_m != self.min_speech_s:
            log.warning(
                "vad_bias_profile_min_speech_clamped",
                name=self.name,
                requested=self.min_speech_s,
                clamped=clamped_m,
            )
        object.__setattr__(self, "hangover_s",   clamped_h)
        object.__setattr__(self, "min_speech_s", clamped_m)

        if self.min_speech_s > self.hangover_s:
            # This is not fatal but it means every segment passes the min gate,
            # which is the expected behaviour for domains where every utterance
            # counts.  Log it so it shows up in diagnostics.
            log.debug(
                "vad_bias_profile_min_speech_exceeds_hangover",
                name=self.name,
                hangover_s=self.hangover_s,
                min_speech_s=self.min_speech_s,
            )

    def __repr__(self) -> str:
        return (
            f"VADBiasProfile({self.name!r}, "
            f"hangover={self.hangover_s:.2f}s, "
            f"min_speech={self.min_speech_s:.2f}s)"
        )


# ── built-in profiles ─────────────────────────────────────────────────────────

PROFILE_DEFAULT = VADBiasProfile(
    name="default",
    hangover_s=_VAD_HANGOVER_S,
    min_speech_s=_VAD_MIN_SPEECH_S,
    description="Env-var defaults; used when no context is available.",
)
PROFILE_GREETING = VADBiasProfile(
    name="greeting",
    hangover_s=1.5,
    min_speech_s=0.5,
    description="Pre-interview greeting; candidate will introduce themselves.",
)
PROFILE_INTRO = VADBiasProfile(
    name="intro",
    hangover_s=1.2,
    min_speech_s=0.4,
    description="Interview intro/role explanation; moderately long answer.",
)
PROFILE_PROBE_VERIFY = VADBiasProfile(
    name="probe_verify",
    hangover_s=0.3,
    min_speech_s=0.2,
    description="Short confirmatory questions; candidate answers briefly.",
)
PROFILE_PROBE_LATERAL = VADBiasProfile(
    name="probe_lateral",
    hangover_s=0.3,
    min_speech_s=0.2,
    description="Lateral probe; similar to verify, usually one-two sentences.",
)
PROFILE_COAST_BEHAVIORAL = VADBiasProfile(
    name="coast_behavioral",
    hangover_s=0.9,
    min_speech_s=0.4,
    description="COAST action on behavioral domain; expect story-form 30-90s answer.",
)
PROFILE_COAST_GENERAL = VADBiasProfile(
    name="coast_general",
    hangover_s=0.7,
    min_speech_s=0.3,
    description="COAST action on non-behavioral domain; medium-length answer.",
)
PROFILE_ESCALATE_SYSDESIGN = VADBiasProfile(
    name="escalate_system_design",
    hangover_s=1.2,
    min_speech_s=0.5,
    description="ESCALATE on system_design; expect architecture walk-through 60-180s.",
)
PROFILE_ESCALATE_GENERAL = VADBiasProfile(
    name="escalate_general",
    hangover_s=0.9,
    min_speech_s=0.4,
    description="ESCALATE on non-system_design domain; longer than probe, shorter than arch.",
)
PROFILE_FIRST_IN_DOMAIN = VADBiasProfile(
    name="first_in_domain",
    hangover_s=0.7,
    min_speech_s=0.3,
    description="First question after a domain switch; candidate establishes context.",
)
PROFILE_DOMAIN_SWITCH = VADBiasProfile(
    name="domain_switch",
    hangover_s=0.15,
    min_speech_s=0.05,
    description=(
        "Bridge utterance between domains. Bot speaks, no STT answer expected. "
        "Hangover kept ultra-short to detect barge-in only."
    ),
)
PROFILE_DEESCALATE = VADBiasProfile(
    name="deescalate",
    hangover_s=0.4,
    min_speech_s=0.25,
    description="DEESCALATE action; similar to default, candidate giving a shorter answer.",
)
PROFILE_CLOSE = VADBiasProfile(
    name="close",
    hangover_s=0.5,
    min_speech_s=0.2,
    description="Interview closing; short acknowledgment expected.",
)


# ──────────────────────────────────────────────────────────────────────────────
# 3. VADParameterTable — priority-ordered profile resolution
# ──────────────────────────────────────────────────────────────────────────────

class VADParameterTable:
    """
    Maps a VADContextHint to the appropriate VADBiasProfile.

    Resolution order (first match wins):
      1. is_domain_switch → PROFILE_DOMAIN_SWITCH always
      2. qa_stage == "greeting" → PROFILE_GREETING
      3. qa_stage == "intro" → PROFILE_INTRO
      4. scaler_action == "close" → PROFILE_CLOSE
      5. scaler_action == "deescalate" → PROFILE_DEESCALATE
      6. scaler_action in {probe_verify, probe_lateral} → PROFILE_PROBE_VERIFY/LATERAL
      7. scaler_action == "escalate" and domain == "system_design" → PROFILE_ESCALATE_SYSDESIGN
      8. scaler_action == "escalate" → PROFILE_ESCALATE_GENERAL
      9. scaler_action == "coast" and domain == "behavioral" → PROFILE_COAST_BEHAVIORAL
     10. scaler_action == "coast" → PROFILE_COAST_GENERAL
     11. is_first_in_domain → PROFILE_FIRST_IN_DOMAIN
     12. fallback → PROFILE_DEFAULT

    The table can be extended at runtime via ``register_override`` for
    A/B testing new profiles without redeploying.
    """

    def __init__(self) -> None:
        # List of (predicate, profile) pairs evaluated in order.
        self._rules: list[tuple[Callable[["VADContextHint"], bool], VADBiasProfile]] = []
        self._lock = threading.Lock()
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """Populate the default priority-ordered rule list."""
        # Each rule is a (predicate_fn, profile) tuple.
        # predicate_fn receives a VADContextHint and returns bool.
        self._rules = [
            (lambda h: h.is_domain_switch,                                          PROFILE_DOMAIN_SWITCH),
            (lambda h: h.qa_stage == QAStage.GREETING,                             PROFILE_GREETING),
            (lambda h: h.qa_stage == QAStage.INTRO,                                PROFILE_INTRO),
            (lambda h: h.scaler_action == ScalerActionKind.CLOSE,                  PROFILE_CLOSE),
            (lambda h: h.scaler_action == ScalerActionKind.DEESCALATE,             PROFILE_DEESCALATE),
            (lambda h: h.scaler_action == ScalerActionKind.PROBE_VERIFY,           PROFILE_PROBE_VERIFY),
            (lambda h: h.scaler_action == ScalerActionKind.PROBE_LATERAL,          PROFILE_PROBE_LATERAL),
            (lambda h: (h.scaler_action == ScalerActionKind.ESCALATE
                        and h.domain == AnswerDomain.SYSTEM_DESIGN),               PROFILE_ESCALATE_SYSDESIGN),
            (lambda h: h.scaler_action == ScalerActionKind.ESCALATE,               PROFILE_ESCALATE_GENERAL),
            (lambda h: (h.scaler_action == ScalerActionKind.COAST
                        and h.domain == AnswerDomain.BEHAVIORAL),                  PROFILE_COAST_BEHAVIORAL),
            (lambda h: h.scaler_action == ScalerActionKind.COAST,                  PROFILE_COAST_GENERAL),
            (lambda h: h.is_first_in_domain,                                       PROFILE_FIRST_IN_DOMAIN),
        ]

    def resolve(self, hint: "VADContextHint") -> VADBiasProfile:
        """
        Return the VADBiasProfile that best matches this hint.

        Never raises; always returns a profile (PROFILE_DEFAULT on no match).
        """
        with self._lock:
            for predicate, profile in self._rules:
                try:
                    if predicate(hint):
                        return profile
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "vad_parameter_table_predicate_error",
                        profile=profile.name,
                        exc=str(exc),
                    )
            return PROFILE_DEFAULT

    def register_override(
        self,
        predicate: Callable[["VADContextHint"], bool],
        profile: VADBiasProfile,
        *,
        priority: int = 0,
    ) -> None:
        """
        Insert a custom rule at ``priority`` (0 = highest, len = lowest).

        Thread-safe.  Primarily for A/B testing in production without a
        full redeploy.

        Example::

            table.register_override(
                lambda h: h.domain == AnswerDomain.CODING,
                VADBiasProfile("coding_custom", hangover_s=0.6, min_speech_s=0.3),
                priority=6,   # after probe rules, before escalate rules
            )
        """
        with self._lock:
            insert_at = max(0, min(priority, len(self._rules)))
            self._rules.insert(insert_at, (predicate, profile))
            log.info(
                "vad_parameter_table_override_registered",
                profile=profile.name,
                priority=insert_at,
            )

    def list_rules(self) -> list[tuple[str, str]]:
        """Return (predicate_repr, profile_name) pairs for diagnostics."""
        with self._lock:
            return [(repr(p), prof.name) for p, prof in self._rules]


# Module-level singleton.  voice_graph imports this directly.
PARAMETER_TABLE: Final[VADParameterTable] = VADParameterTable()


# ──────────────────────────────────────────────────────────────────────────────
# 4. VADContextHint — immutable, validated hint dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VADContextHint:
    """
    Interview-semantic hint injected into ContextGatedVAD before each turn.

    All fields are validated at construction time.  Enum fields accept both
    enum instances and raw strings (tolerant parsing).

    Attributes
    ──────────
    qa_stage
        Current stage of the interview pipeline.
    scaler_action
        The ScalerAction that was committed for this turn.
    domain
        Answer domain inferred by the QA controller.
    is_first_in_domain
        True if this is the first question after a domain transition.
        The candidate typically gives a longer answer to establish context.
    is_domain_switch
        True for bridge utterances where the *bot* speaks and no STT
        answer is expected.  VAD hangover is set to near-zero for barge-in
        detection only.
    expected_duration_s
        Hint from the QA layer about expected answer length.  Not used
        directly for parameter selection (the scaler_action + domain combo
        already encodes this), but logged and stored for analytics.
    turn_index
        Zero-based index of this turn within the current interview session.
        Used by VADContextHistory to correlate hints with transcript events.
    committed_at
        Monotonic timestamp (seconds) when the hint was created.  Used to
        measure how long a hint waited in the pending slot before application.

    Usage::

        hint = VADContextHint(
            qa_stage=QAStage.INTERVIEW,
            scaler_action=ScalerActionKind.ESCALATE,
            domain=AnswerDomain.SYSTEM_DESIGN,
            is_first_in_domain=False,
            is_domain_switch=False,
            expected_duration_s=90.0,
            turn_index=7,
        )
        vad_gate.apply_context(hint)
    """

    qa_stage:             QAStage
    scaler_action:        ScalerActionKind
    domain:               AnswerDomain
    is_first_in_domain:   bool
    is_domain_switch:     bool
    expected_duration_s:  float
    turn_index:           int                = 0
    committed_at:         float              = field(default_factory=time.monotonic)

    # Resolved profile is cached lazily — not part of frozen equality/hash.
    _resolved_profile: ClassVar[dict[int, VADBiasProfile]] = {}

    # noinspection PyUnreachableCode
    def __post_init__(self) -> None:
        # Tolerant enum coercion: accept raw strings from dict-deserialized state.
        if not isinstance(self.qa_stage, QAStage):
            object.__setattr__(self, "qa_stage", QAStage(self.qa_stage))
        if not isinstance(self.scaler_action, ScalerActionKind):
            object.__setattr__(self, "scaler_action", ScalerActionKind(self.scaler_action))
        if not isinstance(self.domain, AnswerDomain):
            object.__setattr__(self, "domain", AnswerDomain(self.domain))

        if self.expected_duration_s < 0.0:
            object.__setattr__(self, "expected_duration_s", 0.0)
        if self.turn_index < 0:
            object.__setattr__(self, "turn_index", 0)

        # Logical consistency check: a domain_switch hint should not also be
        # marked first_in_domain (it's a bot utterance, not a Q&A turn).
        if self.is_domain_switch and self.is_first_in_domain:
            log.warning(
                "vad_context_hint_inconsistent_flags",
                is_domain_switch=True,
                is_first_in_domain=True,
                turn_index=self.turn_index,
            )
            object.__setattr__(self, "is_first_in_domain", False)

    # ── resolution ────────────────────────────────────────────────────────────

    def resolve_profile(
        self,
        table: VADParameterTable = PARAMETER_TABLE,
    ) -> VADBiasProfile:
        """
        Return the VADBiasProfile for this hint via ``table``.

        Result is NOT cached because ``table`` may have dynamic overrides.
        Typically called once per hint application; the overhead is trivial.
        """
        return table.resolve(self)

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def for_greeting(cls, *, turn_index: int = 0) -> "VADContextHint":
        """Pre-built hint for the greeting stage."""
        return cls(
            qa_stage=QAStage.GREETING,
            scaler_action=ScalerActionKind.COAST,
            domain=AnswerDomain.UNKNOWN,
            is_first_in_domain=True,
            is_domain_switch=False,
            expected_duration_s=30.0,
            turn_index=turn_index,
        )

    @classmethod
    def for_intro(cls, *, turn_index: int = 0) -> "VADContextHint":
        """Pre-built hint for the intro stage (candidate self-introduction)."""
        return cls(
            qa_stage=QAStage.INTRO,
            scaler_action=ScalerActionKind.COAST,
            domain=AnswerDomain.UNKNOWN,
            is_first_in_domain=True,
            is_domain_switch=False,
            expected_duration_s=60.0,
            turn_index=turn_index,
        )

    @classmethod
    def for_domain_switch(
        cls,
        *,
        domain: AnswerDomain | str = AnswerDomain.UNKNOWN,
        turn_index: int = 0,
    ) -> "VADContextHint":
        """
        Pre-built hint for a domain-switch bridge utterance.

        No STT answer is expected — VAD is set to barge-in sensitivity only.
        """
        return cls(
            qa_stage=QAStage.INTERVIEW,
            scaler_action=ScalerActionKind.BRIDGE,
            domain=domain if isinstance(domain, AnswerDomain) else AnswerDomain(domain),
            is_first_in_domain=False,
            is_domain_switch=True,
            expected_duration_s=0.0,
            turn_index=turn_index,
        )

    @classmethod
    def for_complete(cls, *, turn_index: int = 0) -> "VADContextHint":
        """Pre-built hint for interview completion (closing pleasantries)."""
        return cls(
            qa_stage=QAStage.COMPLETE,
            scaler_action=ScalerActionKind.CLOSE,
            domain=AnswerDomain.UNKNOWN,
            is_first_in_domain=False,
            is_domain_switch=False,
            expected_duration_s=10.0,
            turn_index=turn_index,
        )

    # ── serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dict for logging and trace attributes."""
        return {
            "qa_stage":            self.qa_stage.value,
            "scaler_action":       self.scaler_action.value,
            "domain":              self.domain.value,
            "is_first_in_domain":  self.is_first_in_domain,
            "is_domain_switch":    self.is_domain_switch,
            "expected_duration_s": round(self.expected_duration_s, 2),
            "turn_index":          self.turn_index,
            "committed_at":        round(self.committed_at, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VADContextHint":
        """Deserialize from a dict produced by ``to_dict``."""
        return cls(
            qa_stage=QAStage(d["qa_stage"]),
            scaler_action=ScalerActionKind(d["scaler_action"]),
            domain=AnswerDomain(d["domain"]),
            is_first_in_domain=bool(d["is_first_in_domain"]),
            is_domain_switch=bool(d["is_domain_switch"]),
            expected_duration_s=float(d.get("expected_duration_s", 0.0)),
            turn_index=int(d.get("turn_index", 0)),
        )

    def __repr__(self) -> str:
        return (
            f"VADContextHint("
            f"stage={self.qa_stage.value}, "
            f"action={self.scaler_action.value}, "
            f"domain={self.domain.value}, "
            f"first={self.is_first_in_domain}, "
            f"switch={self.is_domain_switch}, "
            f"expected={self.expected_duration_s:.1f}s, "
            f"turn={self.turn_index})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. VADContextHistory — bounded ring of applied hints for diagnostics
# ──────────────────────────────────────────────────────────────────────────────

class _HistoryEntry(NamedTuple):
    hint:          VADContextHint
    profile:       VADBiasProfile
    applied_at:    float    # monotonic seconds
    pending_age_s: float    # how long it sat in the pending slot


class VADContextHistory:
    """
    Bounded ring buffer of the last N applied VADContextHints.

    Stored alongside the resolved VADBiasProfile and application timestamp
    so post-mortem analysis can correlate VAD misfires with context changes.

    Thread-safe (used from both asyncio callbacks and diagnostic endpoints).

    Usage::

        history = VADContextHistory(maxlen=50)
        history.record(hint, profile, applied_at=time.monotonic(), pending_age_s=0.02)
        entries = history.recent(n=10)
        summary = history.summary()
    """

    def __init__(self, maxlen: int = 100) -> None:
        self._maxlen = maxlen
        self._entries: collections.deque[_HistoryEntry] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total_recorded: int = 0

    def record(
        self,
        hint: VADContextHint,
        profile: VADBiasProfile,
        *,
        applied_at: float,
        pending_age_s: float,
    ) -> None:
        """Append an entry.  Oldest entry evicted when at capacity."""
        entry = _HistoryEntry(
            hint=hint,
            profile=profile,
            applied_at=applied_at,
            pending_age_s=pending_age_s,
        )
        with self._lock:
            self._entries.append(entry)
            self._total_recorded += 1
        _vad_context_history_depth.set(len(self._entries))

    def recent(self, n: int = 10) -> list[_HistoryEntry]:
        """Return the n most recently applied entries (newest last)."""
        with self._lock:
            entries = list(self._entries)
        return entries[-n:]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        _vad_context_history_depth.set(0)

    def summary(self) -> dict:
        """
        Return aggregate statistics over the full history buffer.

        Useful for health-check endpoints and test assertions.
        """
        with self._lock:
            entries = list(self._entries)

        if not entries:
            return {
                "total_recorded": self._total_recorded,
                "buffered": 0,
                "profile_counts": {},
                "mean_pending_age_s": None,
                "max_pending_age_s": None,
            }

        profile_counts: dict[str, int] = {}
        pending_ages: list[float] = []
        for e in entries:
            profile_counts[e.profile.name] = profile_counts.get(e.profile.name, 0) + 1
            pending_ages.append(e.pending_age_s)

        return {
            "total_recorded":   self._total_recorded,
            "buffered":         len(entries),
            "profile_counts":   profile_counts,
            "mean_pending_age_s": round(sum(pending_ages) / len(pending_ages), 4),
            "max_pending_age_s":  round(max(pending_ages), 4),
        }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"VADContextHistory(buffered={len(self)}, "
            f"total_recorded={self._total_recorded}, "
            f"maxlen={self._maxlen})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. VADContextHintBuilder — factory from QA controller state
# ──────────────────────────────────────────────────────────────────────────────

class VADContextHintBuilder:
    """
    Converts QA controller turn state into a VADContextHint.

    This class exists so that voice_graph does not have to know the enum
    types — it just passes a CommittedTurn (or an equivalent dict).

    All methods are static / class-level; the class is a namespace, not
    a service object.

    Primary API
    ───────────
    ``from_committed_turn(committed_turn)``  — main integration point
    ``from_state_dict(d)``                   — for testing without domain objects
    """

    # Expected answer duration lookup by (scaler_action, domain) — used only
    # for the ``expected_duration_s`` field; does not affect profile resolution.
    _EXPECTED_DURATION: ClassVar[dict[tuple[str, str], float]] = {
        ("probe_verify",  "system_design"): 15.0,
        ("probe_verify",  "behavioral"):    10.0,
        ("probe_verify",  "coding"):        20.0,
        ("probe_verify",  "unknown"):       10.0,
        ("probe_lateral", "system_design"): 15.0,
        ("probe_lateral", "behavioral"):    10.0,
        ("probe_lateral", "unknown"):       10.0,
        ("coast",         "system_design"): 60.0,
        ("coast",         "behavioral"):    60.0,
        ("coast",         "coding"):        45.0,
        ("coast",         "unknown"):       30.0,
        ("escalate",      "system_design"): 120.0,
        ("escalate",      "behavioral"):    90.0,
        ("escalate",      "coding"):        60.0,
        ("escalate",      "unknown"):       60.0,
        ("deescalate",    "system_design"): 30.0,
        ("deescalate",    "behavioral"):    30.0,
        ("deescalate",    "unknown"):       20.0,
        ("close",         "unknown"):       10.0,
        ("bridge",        "unknown"):        0.0,
    }

    @classmethod
    def _lookup_duration(cls, action: str, domain: str) -> float:
        key = (action, domain)
        if key in cls._EXPECTED_DURATION:
            return cls._EXPECTED_DURATION[key]
        # Fallback: try with "unknown" domain, then hard-coded default.
        fallback = cls._EXPECTED_DURATION.get((action, "unknown"), 20.0)
        return fallback

    @classmethod
    def from_committed_turn(
        cls,
        committed_turn: object,
        *,
        turn_index: int = 0,
    ) -> VADContextHint:
        """
        Build a VADContextHint from a CommittedTurn domain object.

        Accesses the following attributes via getattr (duck-typed so this
        module does not import the QA controller directly):

          .qa_stage       → str  e.g. "interview"
          .scaler_action  → str  e.g. "escalate"   (ScalerAction.value)
          .domain         → str  e.g. "system_design"
          .is_first_in_domain  → bool
          .is_domain_switch    → bool

        All fields have safe fallbacks; a missing attribute logs a warning
        and uses the default.
        """
        def _get(attr: str, default: object) -> object:
            val = getattr(committed_turn, attr, None)
            if val is None:
                log.warning(
                    "vad_hint_builder_missing_attr",
                    attr=attr,
                    fallback=default,
                )
                return default
            return val

        qa_stage_raw       = str(_get("qa_stage",          "interview"))
        scaler_action_raw  = str(_get("scaler_action",     "probe_verify"))
        domain_raw         = str(_get("domain",            "unknown"))
        is_first           = bool(_get("is_first_in_domain", False))
        is_switch          = bool(_get("is_domain_switch",   False))

        expected_s = cls._lookup_duration(scaler_action_raw, domain_raw)

        return VADContextHint(
            qa_stage=QAStage(qa_stage_raw),
            scaler_action=ScalerActionKind(scaler_action_raw),
            domain=AnswerDomain(domain_raw),
            is_first_in_domain=is_first,
            is_domain_switch=is_switch,
            expected_duration_s=expected_s,
            turn_index=turn_index,
        )

    @classmethod
    def from_state_dict(cls, d: dict, *, turn_index: int = 0) -> VADContextHint:
        """
        Build from a plain dict.  Useful in tests and when deserializing
        from Redis session state.

        Required keys: ``qa_stage``, ``scaler_action``, ``domain``.
        Optional: ``is_first_in_domain``, ``is_domain_switch``,
                  ``expected_duration_s``.
        """
        action = d.get("scaler_action", "probe_verify")
        domain = d.get("domain", "unknown")
        return VADContextHint(
            qa_stage=QAStage(d.get("qa_stage", "interview")),
            scaler_action=ScalerActionKind(action),
            domain=AnswerDomain(domain),
            is_first_in_domain=bool(d.get("is_first_in_domain", False)),
            is_domain_switch=bool(d.get("is_domain_switch", False)),
            expected_duration_s=float(
                d.get("expected_duration_s", cls._lookup_duration(action, domain))
            ),
            turn_index=turn_index,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 7. ContextGatedVAD — PCMVADGate subclass with pending-hint slot
# ──────────────────────────────────────────────────────────────────────────────

class ContextGatedVAD(PCMVADGate):
    """
    PCMVADGate extended with a runtime-adjustable interview context hint.

    All existing signal-processing behaviour is preserved when no hint has
    been applied.  When a hint is provided via ``apply_context``, the
    hangover and min-speech parameters are updated **atomically at the next
    SILENCE→SPEECH onset** — never mid-segment, never mid-hangover.

    Design invariants
    ─────────────────
    • ``_pending_hint`` is written by ``apply_context`` and consumed exactly
      once by ``_process_chunk`` at the SILENCE→SPEECH transition.
    • If ``apply_context`` is called again before the pending hint is
      consumed (e.g. the bot commits two turns in quick succession during
      interruption handling), the newer hint overwrites the older one and
      the overwrite counter is incremented.
    • Parameter changes are never applied during HANGOVER.  The hangover
      countdown that is already ticking uses the parameters from the onset
      that started it.

    Usage (from voice_graph.stream_full_pcm)::

        vad_gate = ContextGatedVAD(fmt=PCMFormat.whisper())

        # After node_llm commits a turn:
        hint = VADContextHintBuilder.from_committed_turn(committed_turn,
                                                          turn_index=turn_idx)
        vad_gate.apply_context(hint)

        # The existing capture loop is unchanged:
        async for speech_chunk in vad_gate.stream(pcm_source):
            await stt_node.dispatch(speech_chunk)
    """

    def __init__(
        self,
        fmt: PCMFormat,
        onset_rms:    float = _VAD_ONSET_RMS,
        offset_rms:   float = _VAD_OFFSET_RMS,
        hangover_s:   float = _VAD_HANGOVER_S,
        pre_roll_s:   float = _VAD_PRE_ROLL_S,
        min_speech_s: float = _VAD_MIN_SPEECH_S,
        *,
        history_maxlen: int = 100,
        parameter_table: VADParameterTable = PARAMETER_TABLE,
    ) -> None:
        super().__init__(
            fmt=fmt,
            onset_rms=onset_rms,
            offset_rms=offset_rms,
            hangover_s=hangover_s,
            pre_roll_s=pre_roll_s,
            min_speech_s=min_speech_s,
        )
        self._pending_hint:    VADContextHint | None = None
        self._pending_set_at:  float = 0.0           # monotonic; for pending_age_s
        self._active_profile:  VADBiasProfile = VADBiasProfile(
            name="initial",
            hangover_s=hangover_s,
            min_speech_s=min_speech_s,
            description="Parameters from constructor args.",
        )
        self._history:         VADContextHistory = VADContextHistory(
            maxlen=history_maxlen
        )
        self._table:           VADParameterTable = parameter_table

    # ── public API ────────────────────────────────────────────────────────────

    def apply_context(self, hint: VADContextHint) -> None:
        """
        Queue a context hint for application at the next segment onset.

        Thread-safe: safe to call from the same asyncio thread as ``stream``
        or from ``loop.call_soon_threadsafe``.  Do NOT call directly from a
        background thread without going through ``call_soon_threadsafe``.

        If a pending hint already exists (previous turn was not yet picked up
        because no speech has arrived), the newer hint overwrites it and a
        metric is incremented so this shows up in dashboards.
        """
        if self._pending_hint is not None:
            _vad_context_pending_overwritten.inc()
            log.debug(
                "vad_context_pending_hint_overwritten",
                old_hint=repr(self._pending_hint),
                new_hint=repr(hint),
            )

        self._pending_hint   = hint
        self._pending_set_at = time.monotonic()

        log.debug(
            "vad_context_hint_queued",
            hint=repr(hint),
            profile=hint.resolve_profile(self._table).name,
        )

    @property
    def active_profile(self) -> VADBiasProfile:
        """The VADBiasProfile whose parameters are currently active."""
        return self._active_profile

    @property
    def history(self) -> VADContextHistory:
        """Read-only access to the applied-hint history ring."""
        return self._history

    @property
    def pending_hint(self) -> VADContextHint | None:
        """The hint queued for the next onset, or None."""
        return self._pending_hint

    # ── internal: hint application ────────────────────────────────────────────

    def _apply_pending_hint(self) -> None:
        """
        Consume ``_pending_hint`` and update VAD parameters.

        Called exclusively from ``_process_chunk`` at the SILENCE→SPEECH
        transition, so it runs in the same asyncio callchain as ``stream``.
        No locking needed beyond the GIL.
        """
        hint = self._pending_hint
        if hint is None:
            return

        self._pending_hint = None
        applied_at   = time.monotonic()
        pending_age  = applied_at - self._pending_set_at

        profile = hint.resolve_profile(self._table)
        self._active_profile = profile

        # Apply to PCMVADGate internals.
        self._hangover_frames    = self._fmt.frames_for_duration(profile.hangover_s)
        self._min_speech_frames  = self._fmt.frames_for_duration(profile.min_speech_s)

        # Observability.
        _vad_context_updates.labels(
            qa_stage=hint.qa_stage.value,
            scaler_action=hint.scaler_action.value,
        ).inc()
        _vad_context_hangover_s.observe(profile.hangover_s)

        if hint.is_domain_switch:
            _vad_context_domain_switch.inc()

        self._history.record(
            hint,
            profile,
            applied_at=applied_at,
            pending_age_s=pending_age,
        )

        with tracer.start_as_current_span("vad.context_applied") as span:
            span.set_attribute("vad.profile",       profile.name)
            span.set_attribute("vad.hangover_s",    profile.hangover_s)
            span.set_attribute("vad.min_speech_s",  profile.min_speech_s)
            span.set_attribute("vad.pending_age_s", round(pending_age, 4))
            span.set_attribute("vad.turn_index",    hint.turn_index)
            for k, v in hint.to_dict().items():
                span.set_attribute(f"vad.hint.{k}", str(v))

        log.info(
            "vad_context_applied",
            profile=profile.name,
            hangover_s=profile.hangover_s,
            min_speech_s=profile.min_speech_s,
            pending_age_s=round(pending_age, 4),
            turn_index=hint.turn_index,
            is_domain_switch=hint.is_domain_switch,
        )

    # ── override: inject hint at onset ────────────────────────────────────────

    def _process_chunk(self, chunk: PCMChunk) -> PCMChunk | None:
        """
        Override of PCMVADGate._process_chunk.

        Identical to the parent except that when a SILENCE→SPEECH transition
        is detected and a pending hint exists, the hint is applied *before*
        the first speech frame is accumulated.  This ensures the new hangover
        value is in effect for the entire segment.
        """
        import numpy as np  # already imported in audio_engine; local for clarity # noqa

        data = chunk.data
        if data.ndim == 2:
            data = data[:, 0] if self._fmt.channels == 1 else data.reshape(-1)
        rms = self._rms(data)

        if self._state == _VADState.SILENCE:
            self._pre_roll.write(data)
            if rms >= self._onset:
                # Apply pending context *before* accumulating the first frame.
                if self._pending_hint is not None:
                    self._apply_pending_hint()

                # Remainder is identical to PCMVADGate.
                from app.audio_engine import _vad_transitions, _vad_active  # noqa
                _vad_transitions.labels(direction="onset").inc()
                _vad_active.set(1)
                log.debug("pcm_vad_onset", rms=round(rms, 1))
                self._state = _VADState.SPEECH
                pre = self._pre_roll.read(self._pre_roll_frames)
                if len(pre) > 0:
                    self._speech_frames.append(pre)
                    self._speech_frame_count += len(pre)
                self._speech_frames.append(data)
                self._speech_frame_count += len(data)
            return None

        # SPEECH and HANGOVER states are unchanged — delegate to parent.
        # noinspection PyUnreachableCode
        return super()._process_chunk(chunk)