from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import TurnRecord, diff_numeric
from app.pgs._normalizer import _normalizer

if TYPE_CHECKING:
    pass

_MIN_TURNS          = 2
_BASELINE_TURNS     = 2
_EWMA_ALPHA         = 0.68
_SLOPE_RISE         = 0.042       # slope threshold per turn
                                   # hedge rate rarely changes by more than 0.04/turn
                                   # 0.055 was missing real gradual escalation patterns
_SLOPE_FLAT         = 0.012
_ABSOLUTE_HIGH      = 0.21        # NOT 0.38 — that was calibrated on written text
                                   # spoken hedge density: baseline ~0.06-0.09,
                                   # high-anxiety spoken: 0.15-0.22 (Levelt, 1989)
                                   # above 0.21 = genuinely high uncertainty signal
_ABSOLUTE_LOW       = 0.03        # below 3% hedge rate = unusually assertive
                                   # typical baseline: 6-9% (Hyland, 1998)
_DOMAIN_CONTRAST    = 0.10        # NOT 0.22 — delta between domain hedge rates
                                   # given absolute_high of 0.21, a 0.22 contrast
                                   # would require near-impossible spread
                                   # 0.10 catches real domain comfort asymmetry


@dataclass
class _HedgeWindow:
    rates:          list[float]          = field(default_factory=list)
    difficulties:   list[float]          = field(default_factory=list)
    by_domain:      dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    baseline_rate:  float | None         = None
    ewma_rate:      float                = 0.0
    n:              int                  = 0


class HedgeEngine:
    """
    Tracks hedging language rate across turns.

    Hedge rate = hedge_count / word_count.
    Baseline: average of first _BASELINE_TURNS turns per candidate.
    Signal: deviation from baseline, slope direction, domain asymmetry.

    Key insight: absolute hedge rate matters less than trajectory.
    A candidate who always hedges heavily is different from one whose
    hedging increases under pressure or in unfamiliar domains.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _HedgeWindow] = defaultdict(_HedgeWindow)

    def ingest(self, session_id: str, turn: TurnRecord) -> None:
        w  = self._windows[session_id]
        nr = _normalizer.normalize(turn.answer)
        d  = diff_numeric(turn.difficulty)

        rate = nr.hedge_count / max(1, nr.word_count)
        w.rates.append(rate)
        w.difficulties.append(d)
        w.by_domain[turn.domain].append(rate)

        if w.baseline_rate is None and w.n >= _BASELINE_TURNS - 1:
            w.baseline_rate = statistics.mean(w.rates[:_BASELINE_TURNS])

        if w.n == 0:
            w.ewma_rate = rate
        else:
            w.ewma_rate = _EWMA_ALPHA * rate + (1 - _EWMA_ALPHA) * w.ewma_rate

        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral / stable hedging pattern.
        > 0.5 = hedging increasing (uncertainty escalating).
        < 0.5 = hedging decreasing (candidate becoming more assertive — or shutting down).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_TURNS:
            return 0.5

        slope_signal    = self._slope_signal(w)
        absolute_signal = self._absolute_signal(w)
        domain_signal   = self._domain_asymmetry(w)

        composite = (
            slope_signal    * 0.50 +
            absolute_signal * 0.30 +
            domain_signal   * 0.20
        )
        return max(0.0, min(1.0, composite))

    def _slope_signal(self, w: _HedgeWindow) -> float:
        n = len(w.rates)
        if n < 3:
            if w.baseline_rate is None:
                return 0.5
            current = w.ewma_rate
            delta = current - w.baseline_rate
            return max(0.0, min(1.0, 0.5 + delta * 3.5))

        xs = list(range(n))
        ys = w.rates
        x_mean = statistics.mean(xs)
        y_mean = statistics.mean(ys)
        num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n)) + 1e-9
        slope = num / den

        if slope > _SLOPE_RISE:
            return min(1.0, 0.5 + math.tanh(slope / _SLOPE_RISE) * 0.42)
        if slope < -_SLOPE_RISE:
            return max(0.0, 0.5 + math.tanh(slope / _SLOPE_RISE) * 0.42)
        return 0.5

    def _absolute_signal(self, w: _HedgeWindow) -> float:
        if not w.rates:
            return 0.5
        current = w.ewma_rate
        if current > _ABSOLUTE_HIGH:
            # scale: at 2× absolute_high → signal = 0.90
            excess = (current - _ABSOLUTE_HIGH) / _ABSOLUTE_HIGH
            return min(1.0, 0.5 + math.tanh(excess * 2.2) * 0.48)
        if current < _ABSOLUTE_LOW and w.n > 4:
            deficit = (_ABSOLUTE_LOW - current) / max(_ABSOLUTE_LOW, 1e-6)
            return max(0.0, 0.5 - math.tanh(deficit * 1.8) * 0.28)
        return 0.5

    def _domain_asymmetry(self, w: _HedgeWindow) -> float:
        domains_with_data = {
            dm: rates for dm, rates in w.by_domain.items()
            if len(rates) >= 2
        }
        if len(domains_with_data) < 2:
            return 0.5

        domain_means = {
            dm: statistics.mean(rates)
            for dm, rates in domains_with_data.items()
        }
        vals = list(domain_means.values())
        spread = max(vals) - min(vals)

        if spread > _DOMAIN_CONTRAST:
            # tanh scaling: at 3× threshold → signal = 0.82
            normalized = math.tanh(spread / _DOMAIN_CONTRAST * 1.4) * 0.38
            return min(1.0, 0.5 + normalized)
        return 0.5

    def domain_hedge_rank(self, session_id: str) -> list[tuple[str, float]]:
        w = self._windows.get(session_id)
        if not w:
            return []
        return sorted(
            [(dm, statistics.mean(rates)) for dm, rates in w.by_domain.items() if rates],
            key=lambda x: x[1],
            reverse=True,
        )

    def rising(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        if not w or w.n < 3:
            return False
        return self._slope_signal(w) > 0.62

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
