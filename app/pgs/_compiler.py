from __future__ import annotations

import math
import random as _rng
from typing import TYPE_CHECKING

from app.pgs._core import (
    BehavioralVector,
    ConstraintSet,
    Pacing,
    _TM,
    _K,
    _n,
)

if TYPE_CHECKING:
    pass

# ── conflict priority ordering ────────────────────────────────────────────────
# When constraints conflict, this ordering resolves them.
# Indices match constraint slots — lower index = higher priority.
_PRIORITY = [
    "rushed",             # 0 — pacing always wins
    "multi-part",         # 1 — structure second
    "immediate-escalation", # 2 — difficulty third
    "cold",               # 3 — tone fourth
    "positive-reinforcement", # 4 — affirmation fifth
    "domain-pivot",       # 5 — domain sixth
    "flat",               # 6 — rhythm lowest
    "over-affirming",     # 7 — affirmation inverse lowest
]

# Probe angle vocabulary — craft language only
_PROBE_ANGLES: dict[str, list[str]] = {
    "avoidance": [
        "the specific mechanism, not the general pattern",
        "the concrete implementation, not the concept",
        "what happens at the boundary case",
        "the failure mode specifically",
        "the tradeoff between the two approaches mentioned",
    ],
    "uncertainty": [
        "the definition, stripped of qualifiers",
        "a concrete example from your experience",
        "the simplest version of this",
        "what you are certain about within this topic",
    ],
    "withdrawal": [
        "one specific aspect of this",
        "the part you are most confident about",
        "how you have seen this behave in practice",
    ],
    "vocabulary_gap": [
        "describe the behavior without the technical term",
        "what problem this solves",
        "how you have used this concept, under any name",
    ],
    "none": [],
}


def _pick_angle(deflection_type: str, seed_offset: int) -> str:
    options = _PROBE_ANGLES.get(deflection_type, [])
    if not options:
        return ""
    idx = seed_offset % len(options)
    return options[idx]


class ConstraintCompiler:
    """
    Maps BehavioralVector → ConstraintSet.

    Threshold values are seed-derived at process start.
    The mapping is non-linear and uses pairwise interference from _TM.
    The output is craft language — no psychological terminology.

    Priority resolution is explicit: when multiple constraint slots would
    fire simultaneously, the priority ordering prevents contradictions.
    """

    def __init__(self) -> None:
        import struct
        import hmac as _hm
        import hashlib as _hl
        self._seed_int = struct.unpack(">I", _K[:4])[0]

    def compile(
        self,
        vec:              BehavioralVector,
        context:          dict,
        n_turns:          int,
        stated_level:     str,
        active_domain:    str,
        deflection_type:  str = "none",
    ) -> ConstraintSet:
        if n_turns < 3:
            return ConstraintSet.neutral()

        v = vec.v
        if len(v) < 8:
            return ConstraintSet.neutral()

        cs = ConstraintSet(active=True)

        self._apply_pacing(v, context, cs)
        self._apply_tone(v, context, cs)
        self._apply_structure(v, context, cs)
        self._apply_difficulty_gate(v, context, cs, n_turns)
        self._apply_probe(v, context, cs, deflection_type, n_turns)
        self._resolve_conflicts(cs)

        return cs

    def _apply_pacing(self, v: tuple, ctx: dict, cs: ConstraintSet) -> None:
        a = v[0]  # velocity
        h = v[2]  # hedge

        t_slow = _TM[0][1]
        t_fast = _TM[0][2]

        if a < t_slow or ctx.get("pressure") in ("shutdown", "collapsing"):
            cs.pacing = Pacing.MEASURED
            cs.forbidden_tones.append("rushed")
            cs.min_words = 22
            cs.max_words = 68
            cs.required_ack_range = (8, 18)

        elif a > t_fast and h < _TM[2][1]:
            cs.pacing = Pacing.BRISK
            cs.max_words = 52
            cs.min_words = 14

        else:
            cs.pacing = Pacing.STANDARD
            cs.max_words = 72
            cs.min_words = 18

    def _apply_tone(self, v: tuple, ctx: dict, cs: ConstraintSet) -> None:
        e = v[4]  # recovery
        c = v[2]  # hedge
        f = v[5]  # consistency
        g = v[6]  # comfort

        t_warm  = _TM[4][0]
        t_cool  = _TM[4][3]
        t_hedge = _TM[2][2]
        t_flat  = _TM[5][0]

        if e < t_warm or ctx.get("in_drop"):
            if "cold" not in cs.forbidden_tones:
                cs.forbidden_tones.append("cold")
            cs.required_ack_range = (
                max(cs.required_ack_range[0], 6),
                max(cs.required_ack_range[1], 16),
            )

        if e > t_cool and c < _TM[2][0]:
            cs.forbidden_tones.append("over-affirming")
            cs.forbidden_patterns.append("positive-reinforcement")
            cs.required_ack_range = (
                cs.required_ack_range[0],
                min(cs.required_ack_range[1], 8),
            )

        if c > t_hedge and ctx.get("hedge_rising"):
            if "flat" not in cs.forbidden_tones:
                cs.forbidden_tones.append("flat")

        if f < t_flat:
            if "cold" not in cs.forbidden_tones:
                cs.forbidden_tones.append("cold")

    def _apply_structure(self, v: tuple, ctx: dict, cs: ConstraintSet) -> None:
        b = v[1]  # elaboration

        t_multi = _TM[1][1]

        if b < t_multi:
            cs.forbidden_patterns.append("multi-part")

        if ctx.get("fatigue"):
            if "immediate-escalation" not in cs.forbidden_patterns:
                cs.forbidden_patterns.append("immediate-escalation")
            cs.difficulty_gate = 0.6

        if ctx.get("warmup"):
            pass

    def _apply_difficulty_gate(
        self, v: tuple, ctx: dict, cs: ConstraintSet, n_turns: int
    ) -> None:
        a = v[0]
        e = v[4]

        t_gate = _TM[0][0]

        if a < t_gate and e < _TM[4][1]:
            if "immediate-escalation" not in cs.forbidden_patterns:
                cs.forbidden_patterns.append("immediate-escalation")
            drop_d = ctx.get("drop_depth", 0.0)
            cs.difficulty_gate = max(0.4, 0.8 - drop_d * 0.6)

        if ctx.get("pressure") == "over-deliberating":
            cs.difficulty_gate = min(cs.difficulty_gate, 0.75)

    def _apply_probe(
        self,
        v: tuple,
        ctx: dict,
        cs: ConstraintSet,
        deflection_type: str,
        n_turns: int,
    ) -> None:
        d = v[3]  # deflection

        t_probe = _TM[3][1]

        # path A: deflection-driven probe — candidate answering adjacent
        if d < t_probe and deflection_type != "none":
            offset = (n_turns + self._seed_int) % 31
            angle = _pick_angle(deflection_type, offset)
            if angle:
                cs.probe_angle = angle

        # path B: variance-driven verification — suspect high score pattern
        # probe_variance > 0.65 + should_verify flag = compiler fires verify angle
        # overrides path A since verify is higher epistemic priority
        if ctx.get("should_verify") and ctx.get("probe_variance", 0.5) > 0.62:
            offset = (n_turns + self._seed_int + 7) % 5
            verify_angles = [
                "explain the reasoning behind that answer specifically",
                "walk through a concrete example of that in practice",
                "describe what happens at the edge case for that approach",
                "explain how you would test that claim",
                "describe where that approach breaks down",
            ]
            cs.probe_angle = verify_angles[offset]

        if "domain-pivot" not in cs.forbidden_patterns:
            comfort_bot = ctx.get("comfort_bot")
            comfort_top = ctx.get("comfort_top")
            if comfort_bot and comfort_top and comfort_bot != comfort_top:
                g = v[6]
                if g > _TM[6][2]:
                    cs.forbidden_patterns.append("domain-pivot")

    def _resolve_conflicts(self, cs: ConstraintSet) -> None:
        if "cold" in cs.forbidden_tones and "over-affirming" in cs.forbidden_tones:
            cs.forbidden_tones.remove("over-affirming")

        if cs.pacing == Pacing.BRISK and "rushed" in cs.forbidden_tones:
            cs.pacing = Pacing.STANDARD
            cs.forbidden_tones.remove("rushed")
            cs.max_words = 62

        if (
            "immediate-escalation" in cs.forbidden_patterns
            and "positive-reinforcement" in cs.forbidden_patterns
        ):
            cs.forbidden_patterns.remove("positive-reinforcement")

        cs.forbidden_tones    = list(dict.fromkeys(cs.forbidden_tones))
        cs.forbidden_patterns = list(dict.fromkeys(cs.forbidden_patterns))
