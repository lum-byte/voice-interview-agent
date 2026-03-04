from __future__ import annotations

import hashlib as _hl
import hmac as _hm
import os as _os
import random as _rng
import struct as _st
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal # noqa

# ── process-scoped key, never persisted, never logged ─────────────────────────
_K: bytes = _os.urandom(32)
_R: _rng.Random = _rng.Random(_os.urandom(8))


def _t(i: int, j: int) -> float:
    h = _hm.new(_K, _st.pack(">II", i, j), _hl.sha256).digest()
    return 0.18 + (_st.unpack(">H", h[:2])[0] / 65535.0) * 0.64


def _n(v: float, s: float = 0.022) -> float:
    return max(0.0, min(1.0, v + _R.gauss(0.0, s)))


# threshold matrix — 8 dimensions × 4 breakpoints each
# generated once at import, constant within a process, opaque across deployments
_TM: list[list[float]] = [
    sorted([_t(i, j) for j in range(4)]) for i in range(8)
]

# pairwise interference weights — 8×8, symmetric, seed-derived
_IW: list[list[float]] = [
    [
        (_st.unpack(">H", _hm.new(_K, _st.pack(">II", min(i, j), max(i, j) + 100),
         _hl.sha256).digest()[:2])[0] / 65535.0) * 0.4
        for j in range(8)
    ]
    for i in range(8)
]


# ── vector ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BehavioralVector:
    v: tuple[float, ...]
    session_id: str
    ts: float = field(default_factory=time.monotonic)

    def __len__(self) -> int:
        return len(self.v)

    def delta(self, other: "BehavioralVector") -> float:
        if len(self.v) != len(other.v):
            return 0.0
        return sum((a - b) ** 2 for a, b in zip(self.v, other.v)) ** 0.5

    def entropy(self) -> float:
        s = sum(self.v) + 1e-9
        return -sum((x / s) * (_os.urandom(1)[0] / 255 * 0.01 +
                    max(1e-9, x / s)) ** 0.5 for x in self.v if x > 0)

    def noisy(self) -> "BehavioralVector":
        return BehavioralVector(
            v=tuple(_n(x) for x in self.v),
            session_id=self.session_id,
            ts=self.ts,
        )

    def serialize(self) -> str:
        import json
        return json.dumps(list(self.v))

    @staticmethod
    def deserialize(raw: str, session_id: str) -> "BehavioralVector":
        import json
        return BehavioralVector(v=tuple(json.loads(raw)), session_id=session_id)

    @staticmethod
    def neutral(session_id: str) -> "BehavioralVector":
        return BehavioralVector(v=(0.5,) * 8, session_id=session_id)


# ── constraint set ─────────────────────────────────────────────────────────────
# craft language only — no psychological terminology anywhere in this dataclass

class Pacing(str, Enum):
    MEASURED    = "measured"
    STANDARD    = "standard"
    BRISK       = "brisk"


@dataclass
class ConstraintSet:
    forbidden_tones:    list[str] = field(default_factory=list)
    forbidden_patterns: list[str] = field(default_factory=list)
    required_ack_range: tuple[int, int] = (3, 12)
    pacing:             Pacing = Pacing.STANDARD
    difficulty_gate:    float = 1.0
    max_words:          int = 72
    min_words:          int = 18
    probe_angle:        str = ""
    active:             bool = False

    def to_suffix(self) -> str:
        if not self.active:
            return ""
        parts: list[str] = []

        if "rushed" in self.forbidden_tones:
            parts.append(
                "Allow a full pause between acknowledgment and the question."
            )
        if "cold" in self.forbidden_tones:
            parts.append(
                "Acknowledgment must be a complete sentence, minimum "
                f"{self.required_ack_range[0]} words."
            )
        if "flat" in self.forbidden_tones:
            parts.append("Vary sentence rhythm. Do not open two sentences identically.")
        if "over-affirming" in self.forbidden_tones:
            parts.append(
                "Single-word acknowledgments only. No elaboration on the answer."
            )
        if "multi-part" in self.forbidden_patterns:
            parts.append("One question only. No compound structure.")
        if "immediate-escalation" in self.forbidden_patterns:
            parts.append(
                f"Difficulty ceiling this turn: {self.difficulty_gate:.1f}. "
                "Do not increase complexity from the previous question."
            )
        if "domain-pivot" in self.forbidden_patterns:
            parts.append("Stay strictly within the current domain. No bridging.")
        if "positive-reinforcement" in self.forbidden_patterns:
            parts.append(
                "Do not use affirming qualifiers. No 'great', 'good', 'nice', "
                "'interesting', 'exactly'."
            )
        if self.pacing == Pacing.MEASURED:
            parts.append(
                f"Total response {self.min_words}–{self.max_words} words. "
                "Prefer shorter acknowledgment, longer question."
            )
        elif self.pacing == Pacing.BRISK:
            parts.append(
                f"Total response under {self.max_words} words. "
                "One-sentence acknowledgment maximum."
            )
        if self.probe_angle:
            parts.append(f"Target specifically: {self.probe_angle}.")

        return "  ".join(parts)

    def serialize(self) -> str:
        import json
        return json.dumps({
            "ft":  self.forbidden_tones,
            "fp":  self.forbidden_patterns,
            "ar":  list(self.required_ack_range),
            "pc":  self.pacing.value,
            "dg":  self.difficulty_gate,
            "mx":  self.max_words,
            "mn":  self.min_words,
            "pa":  self.probe_angle,
            "ac":  self.active,
        })

    @staticmethod
    def deserialize(raw: str) -> "ConstraintSet":
        import json
        d = json.loads(raw)
        return ConstraintSet(
            forbidden_tones    = d["ft"],
            forbidden_patterns = d["fp"],
            required_ack_range = tuple(d["ar"]),
            pacing             = Pacing(d["pc"]),
            difficulty_gate    = float(d["dg"]),
            max_words          = int(d["mx"]),
            min_words          = int(d["mn"]),
            probe_angle        = d["pa"],
            active             = bool(d["ac"]),
        )

    @staticmethod
    def neutral() -> "ConstraintSet":
        return ConstraintSet(active=False)


# ── turn record — internal only ────────────────────────────────────────────────

@dataclass
class TurnRecord:
    session_id:    str
    turn_index:    int
    domain:        str
    level:         str
    question:      str
    answer:        str
    answer_words:  int
    duration_s:    float
    difficulty:    float
    score:         float | None = None
    ts:            float = field(default_factory=time.monotonic)

    @property
    def has_score(self) -> bool:
        return self.score is not None

    def serialize(self) -> str:
        import json
        return json.dumps({
            "si": self.session_id,
            "ti": self.turn_index,
            "dm": self.domain,
            "lv": self.level,
            "q":  self.question,
            "a":  self.answer,
            "aw": self.answer_words,
            "ds": self.duration_s,
            "df": self.difficulty,
            "sc": self.score,
            "ts": self.ts,
        })

    @staticmethod
    def deserialize(raw: str) -> "TurnRecord":
        import json
        d = json.loads(raw)
        return TurnRecord(
            session_id   = d["si"],
            turn_index   = d["ti"],
            domain       = d["dm"],
            level        = d["lv"],
            question     = d["q"],
            answer       = d["a"],
            answer_words = d["aw"],
            duration_s   = d["ds"],
            difficulty   = d["df"],
            score        = d.get("sc"),
            ts           = d["ts"],
        )


# ── session state ──────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    session_id:     str
    stated_level:   str
    domains:        list[str]
    turns:          list[TurnRecord]           = field(default_factory=list)
    vectors:        list[BehavioralVector]     = field(default_factory=list)
    baseline:       BehavioralVector | None    = None
    weights:        list[float]                = field(default_factory=lambda: [1/8]*8)
    created_at:     float                      = field(default_factory=time.monotonic)

    def add_turn(self, t: TurnRecord) -> None:
        self.turns.append(t)

    def patch_score(self, turn_index: int, score: float) -> bool:
        for t in self.turns:
            if t.turn_index == turn_index:
                t.score = score
                return True
        return False

    def scored_turns(self, domain: str | None = None) -> list[TurnRecord]:
        return [
            t for t in self.turns
            if t.has_score and (domain is None or t.domain == domain)
        ]

    def domain_turns(self, domain: str) -> list[TurnRecord]:
        return [t for t in self.turns if t.domain == domain]

    def sufficient(self, min_turns: int = 3) -> bool:
        return len(self.turns) >= min_turns


# ── difficulty numeric map ─────────────────────────────────────────────────────

_DIFF_MAP: dict[str, float] = {
    "easy":   0.20,
    "medium": 0.45,
    "hard":   0.72,
    "expert": 0.95,
}

_LEVEL_MAP: dict[str, float] = {
    "beginner":     0.15,
    "intermediate": 0.50,
    "advanced":     0.85,
}


def diff_numeric(d: str | float) -> float:
    if isinstance(d, float):
        return d
    return _DIFF_MAP.get(str(d).lower(), 0.45)


def level_numeric(l: str) -> float:
    return _LEVEL_MAP.get(str(l).lower(), 0.50)
