from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import TurnRecord, diff_numeric

if TYPE_CHECKING:
    pass

_MIN_TURNS      = 3
_EWMA_ALPHA     = 0.68        # higher than before — spoken turns are short, recency matters more
_COLLAPSE_RATIO = 0.54        # hard answers at <54% duration of easy = shutdown
                               # Ayres & Hopf (1993): response latency drops 35-45% under interview stress
                               # 0.54 catches genuine collapse without firing on natural pacing variation
_SPIKE_RATIO    = 1.72        # over-deliberation at 1.72x — NOT 2.1x
                               # Beatty et al. (1998): anxious candidates show 1.6-1.8x latency inflation
                               # 2.1x was too conservative, missed the actual signal band
_BUCKET_EASY    = 0.35        # difficulty < 0.35 = easy bucket
_BUCKET_HARD    = 0.65        # difficulty > 0.65 = hard bucket


@dataclass
class _VelocityWindow:
    durations:   list[float] = field(default_factory=list)
    difficulties: list[float] = field(default_factory=list)
    ewma:        float = 0.0
    initialized: bool  = False


class VelocityEngine:
    """
    Tracks answer timing against difficulty trajectory.

    Signal: does time-per-answer collapse as difficulty increases?
    Collapse = candidate shutting down under pressure.
    Inverse spike = candidate over-deliberating (anxiety response).
    Both are distinct pressure responses requiring different constraint profiles.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _VelocityWindow] = defaultdict(
            _VelocityWindow
        )

    def ingest(self, session_id: str, turn: TurnRecord) -> None:
        w = self._windows[session_id]
        d = diff_numeric(turn.difficulty)
        dur = max(0.1, turn.duration_s)

        w.durations.append(dur)
        w.difficulties.append(d)

        if not w.initialized:
            w.ewma = dur
            w.initialized = True
        else:
            w.ewma = _EWMA_ALPHA * dur + (1 - _EWMA_ALPHA) * w.ewma

    def compute(self, session_id: str) -> float:
        """
        Returns a float in [0, 1].
        0.5 = neutral (no timing signal).
        < 0.5 = timing collapse under pressure (shutdown).
        > 0.5 = timing inflation under pressure (anxiety/over-deliberation).
        """
        w = self._windows.get(session_id)
        if not w or len(w.durations) < _MIN_TURNS:
            return 0.5

        easy_durs  = [d for d, diff in zip(w.durations, w.difficulties) if diff < _BUCKET_EASY]
        hard_durs  = [d for d, diff in zip(w.durations, w.difficulties) if diff > _BUCKET_HARD]

        if not easy_durs or not hard_durs:
            return self._regression_signal(w)

        easy_mean = statistics.mean(easy_durs)
        hard_mean = statistics.mean(hard_durs)

        if easy_mean < 0.5:
            return 0.5

        ratio = hard_mean / easy_mean

        if ratio < _COLLAPSE_RATIO:
            # Hard questions answered faster than easy ones — shutdown signal
            collapse_depth = 1.0 - (ratio / _COLLAPSE_RATIO)
            return max(0.0, 0.5 - collapse_depth * 0.45)

        if ratio > _SPIKE_RATIO:
            # Hard questions taking dramatically longer — anxiety signal
            spike_height = min(1.0, (ratio - _SPIKE_RATIO) / _SPIKE_RATIO)
            return min(1.0, 0.5 + spike_height * 0.45)

        return 0.5 + (ratio - 1.0) * 0.08

    def _regression_signal(self, w: _VelocityWindow) -> float:
        if len(w.durations) < _MIN_TURNS:
            return 0.5
        n = len(w.durations)
        diffs = w.difficulties
        durs  = w.durations
        x_mean = statistics.mean(diffs)
        y_mean = statistics.mean(durs)
        num = sum((diffs[i] - x_mean) * (durs[i] - y_mean) for i in range(n))
        den = sum((diffs[i] - x_mean) ** 2 for i in range(n)) + 1e-9
        slope = num / den
        # Negative slope = faster on harder questions = collapse
        # Positive slope = slower on harder questions = inflation
        normalized = math.tanh(slope * 2.2) * 0.4
        return max(0.0, min(1.0, 0.5 + normalized))

    def pressure_classification(self, session_id: str) -> str:
        v = self.compute(session_id)
        if v < 0.28:
            return "shutdown"
        if v < 0.42:
            return "collapsing"
        if v > 0.72:
            return "over-deliberating"
        if v > 0.58:
            return "inflating"
        return "stable"

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
