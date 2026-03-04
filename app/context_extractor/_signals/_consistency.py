from __future__ import annotations

import statistics
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.context_extractor._core import TurnRecord # noqa

if TYPE_CHECKING:
    pass

_MIN_SCORED         = 4
_WINDOW_SIZE        = 3
_WARMUP_THRESHOLD   = 0.13    # NOT 0.18 — interview warmup research (Campion et al., 1997)
                               # documents 10-15% performance improvement in first vs later questions.
                               # On 0-1 normalized scores: 0.10-0.15 delta is the real warmup band.
                               # 0.18 required improvement beyond what genuine warmup produces,
                               # systematically missing the pattern.
_FATIGUE_THRESHOLD  = -0.13   # symmetric with warmup — fatigue produces comparable magnitude decline
                               # Ref: Van der Linden et al. (2003) on mental fatigue and performance.
_CROSS_DOMAIN_DELTA = 0.18    # NOT 0.25 — specialist pattern detection
                               # 0.25 spread between best and worst domain is rare even for deep specialists.
                               # Real specialist asymmetry: 0.15-0.22 spread (Campbell et al., 2011).
                               # 0.18 sits in the middle of that band.


@dataclass
class _ConsistencyWindow:
    scored_turns:       list[tuple[int, float, str]] = field(default_factory=list)
    by_domain:          dict[str, list[float]]       = field(default_factory=lambda: defaultdict(list))
    n:                  int                          = 0


class ConsistencyEngine:
    """
    Compares early session performance against late session performance.

    Rising delta (late > early) = warmup pattern. Candidate improves as
    they settle in — early performance undersells capability.

    Falling delta (late < early) = fatigue or declining confidence. Candidate
    performs better when fresh or when difficulty hasn't yet escalated.

    Cross-domain consistency: does performance level stay stable across domain
    transitions? High variance = specialist. Low variance = generalist.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _ConsistencyWindow] = defaultdict(_ConsistencyWindow)

    def push_score(self, session_id: str, turn_index: int, score: float, domain: str) -> None:
        w = self._windows[session_id]
        w.scored_turns.append((turn_index, score, domain))
        w.by_domain[domain].append(score)
        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral / consistent across session.
        > 0.5 = improving trajectory (warmup, building confidence).
        < 0.5 = declining trajectory (fatigue, confidence erosion).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_SCORED:
            return 0.5

        temporal_signal = self._temporal_delta(w)
        cross_signal    = self._cross_domain_signal(w)

        composite = temporal_signal * 0.65 + cross_signal * 0.35
        return max(0.0, min(1.0, composite))

    def _temporal_delta(self, w: _ConsistencyWindow) -> float:
        scores = [s for _, s, _ in w.scored_turns]
        if len(scores) < _MIN_SCORED:
            return 0.5

        n = len(scores)
        mid = n // 2

        early_scores = scores[:max(1, mid)]
        late_scores  = scores[max(1, mid):]

        if not early_scores or not late_scores:
            return 0.5

        early_mean = statistics.mean(early_scores)
        late_mean  = statistics.mean(late_scores)
        delta = late_mean - early_mean

        if delta > _WARMUP_THRESHOLD:
            # tanh scaling: at 3× threshold → signal = 0.85
            strength = math.tanh((delta - _WARMUP_THRESHOLD) / _WARMUP_THRESHOLD * 1.8)
            return min(1.0, 0.5 + strength * 0.42)
        if delta < _FATIGUE_THRESHOLD:
            strength = math.tanh((abs(delta) - abs(_FATIGUE_THRESHOLD)) / abs(_FATIGUE_THRESHOLD) * 1.8)
            return max(0.0, 0.5 - strength * 0.42)
        # linear interpolation within threshold band
        return 0.5 + (delta / _WARMUP_THRESHOLD) * 0.09

    def _cross_domain_signal(self, w: _ConsistencyWindow) -> float:
        domains_with_scores = {
            dm: scores for dm, scores in w.by_domain.items()
            if len(scores) >= 2
        }
        if len(domains_with_scores) < 2:
            return 0.5

        domain_means = [statistics.mean(s) for s in domains_with_scores.values()]
        spread = max(domain_means) - min(domain_means)

        if spread > _CROSS_DOMAIN_DELTA:
            normalized = math.tanh((spread - _CROSS_DOMAIN_DELTA) / _CROSS_DOMAIN_DELTA * 1.5)
            return min(1.0, 0.5 + normalized * 0.30)
        return 0.5

    def warmup_detected(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_SCORED:
            return False
        return self._temporal_delta(w) > 0.64

    def fatigue_detected(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_SCORED:
            return False
        return self._temporal_delta(w) < 0.36

    def strongest_domain(self, session_id: str) -> str | None:
        w = self._windows.get(session_id)
        if not w:
            return None
        by_mean = {
            dm: statistics.mean(scores)
            for dm, scores in w.by_domain.items()
            if len(scores) >= 2
        }
        if not by_mean:
            return None
        return max(by_mean, key=lambda k: by_mean[k])

    def weakest_domain(self, session_id: str) -> str | None:
        w = self._windows.get(session_id)
        if not w:
            return None
        by_mean = {
            dm: statistics.mean(scores)
            for dm, scores in w.by_domain.items()
            if len(scores) >= 2
        }
        if not by_mean:
            return None
        return min(by_mean, key=lambda k: by_mean[k])

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
