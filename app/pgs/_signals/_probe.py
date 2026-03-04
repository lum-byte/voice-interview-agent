from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import TurnRecord, diff_numeric # noqa

if TYPE_CHECKING:
    pass

_VARIANCE_THRESHOLD   = 0.140   # NOT 0.175 — score variance in technical interviews.
                                  # Typical candidate variance on 0-1 scale: 0.04-0.09.
                                  # Genuinely volatile performance: 0.12-0.18.
                                  # 0.175 was requiring near-maximum volatility to trigger.
                                  # 0.140 sits at the real boundary of suspicious variance.
                                  # Ref: Schmidt & Hunter (1998) on predictor validity variance.
_SPIKE_DELTA          = 0.28    # NOT 0.38 — score jump that qualifies as suspicious.
                                  # A 0.38 single-turn spike on 0-1 = 3.8 raw points.
                                  # That only happens on complete knowledge gaps recovering to
                                  # expert answers — not the pattern we're detecting.
                                  # 0.28 = 2.8 raw points = meaningful unexplained jump.
_SHORT_ANSWER_WORDS   = 22      # NOT 18 — minimum answer length for technical content.
                                  # 18 words can contain a complete correct answer for
                                  # definitional questions. 22 is a better floor for anything
                                  # requiring actual reasoning (not just recall).
_STREAK_NEEDED        = 2       # 2 consecutive suspect turns before verify fires — unchanged,
                                  # single suspect turn has too many false positive causes
_EWMA_ALPHA           = 0.72    # high — probe is score-driven, same as recovery.
                                  # Each eval event is high-weight, recency dominant.
_MIN_SCORED           = 3
_LUCK_CONFIDENCE_BAND = 0.12    # tighter band for luck vs knowledge disambiguation


@dataclass
class _ProbeWindow:
    scores:        deque = field(default_factory=lambda: deque(maxlen=12))
    word_counts:   list[int]   = field(default_factory=list)
    difficulties:  list[float] = field(default_factory=list)
    suspect_turns: list[int]   = field(default_factory=list)
    streak:        int         = 0
    ewma_score:    float       = 0.5
    n:             int         = 0


class ProbeOracle:
    """
    Identifies suspect high scores — answers that score well but show
    behavioral markers inconsistent with genuine knowledge.

    Markers of a suspect answer:
      - Short word count relative to question complexity
      - Score spike significantly above recent rolling mean
      - Answer in a domain where comfort signal is low
      - High variance in scores within a domain (lucky pattern)

    When a suspect turn is detected, the probe angle carries a directive
    to the compiler: verify this specific claim, not explore a new concept.
    The LLM is never told why — it receives "ask them to explain their
    reasoning on the previous answer" as a craft constraint.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _ProbeWindow] = defaultdict(_ProbeWindow)

    def push_score(
        self,
        session_id:  str,
        turn_index:  int,
        score:       float,
        answer_words: int,
        difficulty:  float,
        domain:      str,
    ) -> None:
        w = self._windows[session_id]

        if w.n == 0:
            w.ewma_score = score
        else:
            w.ewma_score = _EWMA_ALPHA * score + (1 - _EWMA_ALPHA) * w.ewma_score

        w.scores.append(score)
        w.word_counts.append(answer_words)
        w.difficulties.append(difficulty)

        spike = score - w.ewma_score
        short = answer_words < _SHORT_ANSWER_WORDS
        hard  = difficulty > 0.58      # lowered from 0.60 — medium-hard questions count

        is_suspect = (
            (spike > _SPIKE_DELTA and short) or
            (spike > _SPIKE_DELTA and hard and score > 0.74) or   # was 0.78
            (short and hard and score > 0.78) or                   # was 0.82
            (spike > _SPIKE_DELTA * 1.5 and score > 0.80)         # very large spike alone
        )

        if is_suspect:
            w.suspect_turns.append(turn_index)
            w.streak += 1
        elif score < w.ewma_score - 0.15:
            w.streak = 0

        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral / no suspicious variance.
        > 0.5 = high variance with suspect pattern (luck signal).
        < 0.5 = consistent, low-variance performance (reliable signal).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_SCORED:
            return 0.5

        variance_signal = self._variance_signal(w)
        suspect_signal  = self._suspect_signal(w)
        streak_signal   = self._streak_signal(w)

        composite = (
            variance_signal * 0.40 +
            suspect_signal  * 0.40 +
            streak_signal   * 0.20
        )
        return max(0.0, min(1.0, composite))

    def _variance_signal(self, w: _ProbeWindow) -> float:
        if len(w.scores) < 3:
            return 0.5
        try:
            var = statistics.variance(list(w.scores))
        except statistics.StatisticsError:
            return 0.5
        if var > _VARIANCE_THRESHOLD:
            # tanh: at 2× threshold → 0.82, at 3× → 0.92
            excess = math.tanh((var - _VARIANCE_THRESHOLD) / _VARIANCE_THRESHOLD * 1.6)
            return min(1.0, 0.5 + excess * 0.46)
        if var < 0.03:
            # extremely low variance = consistent = reliable signal (negative probe)
            return max(0.0, 0.5 - (0.03 - var) * 5.0)
        return 0.5

    def _suspect_signal(self, w: _ProbeWindow) -> float:
        if not w.suspect_turns or w.n < _MIN_SCORED:
            return 0.5
        rate = len(w.suspect_turns) / w.n
        if rate > 0.35:
            return min(1.0, 0.5 + rate * 0.55)
        return 0.5

    def _streak_signal(self, w: _ProbeWindow) -> float:
        if w.streak >= _STREAK_NEEDED:
            confidence = min(1.0, w.streak / (_STREAK_NEEDED * 2))
            return min(1.0, 0.5 + confidence * 0.40)
        return 0.5

    def should_verify(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        if not w:
            return False
        return w.streak >= _STREAK_NEEDED

    def last_suspect_index(self, session_id: str) -> int | None:
        w = self._windows.get(session_id)
        if not w or not w.suspect_turns:
            return None
        return w.suspect_turns[-1]

    def domain_variance(self, session_id: str, scores_by_domain: dict[str, list[float]]) -> dict[str, float]:
        result = {}
        for domain, scores in scores_by_domain.items():
            if len(scores) >= 3:
                try:
                    result[domain] = statistics.variance(scores)
                except statistics.StatisticsError:
                    result[domain] = 0.0
        return result

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
