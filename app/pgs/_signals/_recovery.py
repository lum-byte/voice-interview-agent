from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import TurnRecord # noqa

if TYPE_CHECKING:
    pass

_MIN_SCORED_TURNS    = 3
_DROP_THRESHOLD      = 0.18   # NOT 0.22 — eval scores are on 0-1 normalized from 0-10.
                               # A 0.18 drop = 1.8 raw points — meaningful performance dip.
                               # 0.22 required a near-catastrophic drop before logging a drop event,
                               # missing the subtle pressure-response pattern entirely.
                               # Ref: Yerkes-Dodson (1908); Beilock et al. (2004) choking under pressure.
_RECOVERY_THRESHOLD  = 0.14   # how much score must improve to count as recovery
                               # slightly lower to match reduced drop threshold — proportionality
_STABLE_BAND         = 0.14   # std dev threshold for stability classification
                               # 0.12 was too tight — genuinely stable candidates show
                               # ±0.12-0.15 natural scoring variance; this was marking them unstable.
_EWMA_ALPHA          = 0.72   # higher than other engines — score events are sparse and
                               # each one carries high weight. Recency matters more here.
_MAX_RECOVERY_TURNS  = 6      # NOT 8 — domains are 5-8 questions each.
                               # An 8-turn recovery window spans the entire domain.
                               # 6 gives enough runway without crossing domain boundaries.


@dataclass
class _DropEvent:
    turn_index:    int
    pre_score:     float
    drop_score:    float
    domain:        str
    recovery_turn: int | None = None
    recovery_score: float | None = None

    @property
    def recovered(self) -> bool:
        return self.recovery_turn is not None

    @property
    def recovery_speed(self) -> float:
        if not self.recovered:
            return 0.0
        turns_taken = max(1, self.recovery_turn - self.turn_index)
        return 1.0 / turns_taken


@dataclass
class _RecoveryWindow:
    scored_turns:    list[tuple[int, float]] = field(default_factory=list)
    drop_events:     list[_DropEvent]        = field(default_factory=list)
    ewma_score:      float                   = 0.5
    pending_drop:    _DropEvent | None       = None
    n:               int                     = 0


class RecoveryEngine:
    """
    Tracks behavioral recovery after score drops.

    Recovery velocity: how quickly does performance restabilize after a wrong answer.
    Recovery quality: does the recovered score match the pre-drop level or just clear threshold.
    Bounce pattern: score spike immediately after drop = lucky guess, not recovery.

    Fast recovery with quality = resilient.
    Fast recovery without quality = lucky bounce.
    Slow recovery = ruminating, affected.
    No recovery within window = shutdown or knowledge wall.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _RecoveryWindow] = defaultdict(_RecoveryWindow)

    def push_score(self, session_id: str, turn_index: int, score: float, domain: str) -> None:
        w = self._windows[session_id]

        if w.n == 0:
            w.ewma_score = score
        else:
            w.ewma_score = _EWMA_ALPHA * score + (1 - _EWMA_ALPHA) * w.ewma_score

        w.scored_turns.append((turn_index, score))

        if w.pending_drop is not None:
            recovery_gain = score - w.pending_drop.drop_score
            turns_since_drop = turn_index - w.pending_drop.turn_index
            if recovery_gain >= _RECOVERY_THRESHOLD:
                w.pending_drop.recovery_turn  = turn_index
                w.pending_drop.recovery_score = score
                w.drop_events.append(w.pending_drop)
                w.pending_drop = None
            elif turns_since_drop >= _MAX_RECOVERY_TURNS:
                w.drop_events.append(w.pending_drop)
                w.pending_drop = None

        if len(w.scored_turns) >= 2 and w.pending_drop is None:
            prev_score = w.scored_turns[-2][1]
            drop = prev_score - score
            if drop >= _DROP_THRESHOLD:
                w.pending_drop = _DropEvent(
                    turn_index = turn_index,
                    pre_score  = prev_score,
                    drop_score = score,
                    domain     = domain,
                )

        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral / no significant drop events yet.
        > 0.5 = fast, quality recovery (resilient).
        < 0.5 = slow or no recovery (affected by drops).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_SCORED_TURNS:
            return 0.5

        events = w.drop_events
        if w.pending_drop is not None:
            events = events + [w.pending_drop]

        if not events:
            stability = self._stability_signal(w)
            return 0.5 + (stability - 0.5) * 0.3

        speeds  = []
        quality = []

        for ev in events:
            if ev.recovered:
                speeds.append(ev.recovery_speed)
                q = ev.recovery_score / max(0.01, ev.pre_score)
                quality.append(min(1.0, q))
            else:
                speeds.append(0.0)
                quality.append(0.0)

        mean_speed   = statistics.mean(speeds)
        mean_quality = statistics.mean(quality)

        speed_signal   = math.tanh(mean_speed * 3.0) * 0.45
        quality_signal = (mean_quality - 0.5) * 0.45

        composite = 0.5 + speed_signal * 0.55 + quality_signal * 0.45
        return max(0.0, min(1.0, composite))

    def _stability_signal(self, w: _RecoveryWindow) -> float:
        if len(w.scored_turns) < 3:
            return 0.5
        scores = [s for _, s in w.scored_turns]
        if len(scores) < 2:
            return 0.5
        try:
            variance = statistics.variance(scores)
        except statistics.StatisticsError:
            return 0.5
        stable_variance = _STABLE_BAND ** 2    # 0.0196
        if variance < stable_variance:
            return 0.62
        if variance > 0.06:                    # std dev > 0.245 = high instability
            depth = math.tanh((variance - 0.06) / 0.06 * 1.4) * 0.32
            return max(0.18, 0.5 - depth)
        return 0.5

    def currently_in_drop(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        return bool(w and w.pending_drop is not None)

    def drop_depth(self, session_id: str) -> float:
        w = self._windows.get(session_id)
        if not w or w.pending_drop is None:
            return 0.0
        return w.pending_drop.pre_score - w.pending_drop.drop_score

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
