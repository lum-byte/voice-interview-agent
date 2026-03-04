from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.context_extractor._core import TurnRecord, diff_numeric
from app.context_extractor._normalizer import _normalizer

if TYPE_CHECKING:
    pass

_MIN_TURNS_PER_DOMAIN = 2
_COMFORT_CONTRAST     = 0.16    # NOT 0.20 — composite_comfort() produces values in ~[0.32, 0.68]
                                  # for most candidates given the tanh-bounded inputs.
                                  # A 0.20 spread required near-maximal asymmetry to fire.
                                  # 0.16 catches genuine domain comfort divergence at the
                                  # sensitivity level that matches the signal's natural range.
_EWMA_ALPHA           = 0.65


@dataclass
class _DomainComfortProfile:
    word_counts:     list[int]   = field(default_factory=list)
    hedge_rates:     list[float] = field(default_factory=list)
    durations:       list[float] = field(default_factory=list)
    difficulties:    list[float] = field(default_factory=list)
    frag_rates:      list[float] = field(default_factory=list)
    n:               int         = 0

    def composite_comfort(self) -> float:
        if self.n < _MIN_TURNS_PER_DOMAIN:
            return 0.5
        word_mean  = statistics.mean(self.word_counts) if self.word_counts else 60
        hedge_mean = statistics.mean(self.hedge_rates) if self.hedge_rates else 0.08
        frag_mean  = statistics.mean(self.frag_rates)  if self.frag_rates  else 0.12
        dur_mean   = statistics.mean(self.durations)   if self.durations   else 28.0

        # weights sum to 1.0 across the four additive offset terms
        # tanh inputs scaled to produce ±0.20 max offset per dimension
        word_signal  = math.tanh((word_mean  - 60)   / 50)   * 0.22   # 0.28 weight
        hedge_signal = -math.tanh(hedge_mean          * 10)  * 0.26   # 0.33 weight — hedge is strongest
        frag_signal  = -math.tanh(frag_mean            * 6)  * 0.20   # 0.25 weight
        dur_signal   = -math.tanh((dur_mean  - 28)   / 22)   * 0.11   # 0.14 weight — timing noisiest

        return max(0.0, min(1.0, 0.5 + word_signal + hedge_signal + frag_signal + dur_signal))


@dataclass
class _ComfortWindow:
    by_domain: dict[str, _DomainComfortProfile] = field(
        default_factory=lambda: defaultdict(_DomainComfortProfile)
    )
    n: int = 0


class ComfortEngine:
    """
    Computes per-domain comfort signatures by aggregating all behavioral
    dimensions (verbosity, hedging, fragment rate, timing) per domain.

    The result reveals the candidate's actual domain preference ordering
    vs their stated expertise ordering — these frequently diverge.

    High divergence between stated and revealed comfort = the interview
    should pressure-test the stated-primary domains more aggressively.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _ComfortWindow] = defaultdict(_ComfortWindow)

    def ingest(self, session_id: str, turn: TurnRecord) -> None:
        w  = self._windows[session_id]
        nr = _normalizer.normalize(turn.answer)
        p  = w.by_domain[turn.domain]

        p.word_counts.append(nr.word_count)
        p.hedge_rates.append(nr.hedge_count / max(1, nr.word_count))
        p.durations.append(max(0.1, turn.duration_s))
        p.difficulties.append(diff_numeric(turn.difficulty))
        p.frag_rates.append(
            _normalizer.sentence_fragments(nr.clean) / max(1, nr.sentence_count)
        )
        p.n += 1
        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral / uniform comfort across domains.
        > 0.5 = strong domain comfort asymmetry detected (specialist pattern).
        < 0.5 = unusual flatness (masking, uniformly tense, or truly generalist).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_TURNS_PER_DOMAIN * 2:
            return 0.5

        profiles = {
            dm: p for dm, p in w.by_domain.items()
            if p.n >= _MIN_TURNS_PER_DOMAIN
        }
        if len(profiles) < 2:
            return 0.5

        comfort_scores = {dm: p.composite_comfort() for dm, p in profiles.items()}
        vals = list(comfort_scores.values())
        spread = max(vals) - min(vals)

        if spread > _COMFORT_CONTRAST:
            normalized = math.tanh(spread / _COMFORT_CONTRAST) * 0.40
            return min(1.0, 0.5 + normalized)
        if spread < 0.05 and len(vals) >= 3:
            return max(0.0, 0.5 - (0.05 - spread) * 2.0)
        return 0.5

    def comfort_ranking(self, session_id: str) -> list[tuple[str, float]]:
        w = self._windows.get(session_id)
        if not w:
            return []
        return sorted(
            [
                (dm, p.composite_comfort())
                for dm, p in w.by_domain.items()
                if p.n >= _MIN_TURNS_PER_DOMAIN
            ],
            key=lambda x: x[1],
            reverse=True,
        )

    def most_comfortable(self, session_id: str) -> str | None:
        ranking = self.comfort_ranking(session_id)
        return ranking[0][0] if ranking else None

    def least_comfortable(self, session_id: str) -> str | None:
        ranking = self.comfort_ranking(session_id)
        return ranking[-1][0] if ranking else None

    def diverges_from_stated(
        self, session_id: str, stated_primary: str
    ) -> bool:
        ranking = self.comfort_ranking(session_id)
        if not ranking or len(ranking) < 2:
            return False
        revealed_primary = ranking[0][0]
        if revealed_primary == stated_primary:
            return False
        spread = ranking[0][1] - ranking[-1][1]
        return spread > _COMFORT_CONTRAST

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
