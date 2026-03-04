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

_MIN_TURNS          = 3
_OVER_ELAB_EASY     = 2.8
_UNDER_ELAB_HARD    = 0.6
_FRAG_THRESHOLD     = 0.28    # NOT 0.22 — STT output naturally produces ~15-18% fragment-like
                               # boundaries due to absence of punctuation and run-on structure.
                               # 0.22 was firing on normal spoken output, not degradation.
                               # 0.28 targets genuine structural breakdown under load.
                               # Ref: Shriberg (1994) on disfluency rates in spontaneous speech.
_DENSITY_DROP       = 0.14    # NOT 0.18 — spoken lexical density baseline: 0.35-0.48
                               # (vs written ~0.55-0.65, Halliday 1985).
                               # Under pressure, drop to 0.28-0.32 is observed.
                               # A 0.14 delta from spoken baseline is the meaningful threshold;
                               # 0.18 required a near-collapse before firing.
_EWMA_ALPHA         = 0.62    # slightly lower than before — elaboration changes more gradually
                               # than timing; over-reactive EWMA loses the pattern shape


@dataclass
class _ElabWindow:
    word_counts:   list[int]   = field(default_factory=list)
    difficulties:  list[float] = field(default_factory=list)
    frag_rates:    list[float] = field(default_factory=list)
    densities:     list[float] = field(default_factory=list)
    baseline_density: float | None = None
    ewma_words:    float = 0.0
    n:             int   = 0


class ElaborationEngine:
    """
    Measures word output relative to question difficulty.

    Over-elaboration on easy questions: anxiety/performance signal.
    Under-elaboration on hard questions: could be knowledge gap or communication gap.
    Structure degradation (fragment rate rise) under pressure: cognitive load indicator.
    Lexical density drop: candidate switching from content words to filler structure.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _ElabWindow] = defaultdict(_ElabWindow)

    def ingest(self, session_id: str, turn: TurnRecord) -> None:
        w   = self._windows[session_id]
        nr  = _normalizer.normalize(turn.answer)
        d   = diff_numeric(turn.difficulty)

        w.word_counts.append(nr.word_count)
        w.difficulties.append(d)

        frag_rate = (
            _normalizer.sentence_fragments(nr.clean) / max(1, nr.sentence_count)
        )
        w.frag_rates.append(frag_rate)

        density = _normalizer.lexical_density(nr.clean)
        w.densities.append(density)

        if w.baseline_density is None and w.n >= 2:
            w.baseline_density = statistics.mean(w.densities[:2])

        if w.n == 0:
            w.ewma_words = float(nr.word_count)
        else:
            w.ewma_words = _EWMA_ALPHA * nr.word_count + (1 - _EWMA_ALPHA) * w.ewma_words

        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral.
        < 0.5 = under-elaboration bias (terse under pressure or across all).
        > 0.5 = over-elaboration bias (verbose on easy, structured breakdown on hard).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_TURNS:
            return 0.5

        elab_signal  = self._elaboration_ratio(w)
        frag_signal  = self._fragment_signal(w)
        density_signal = self._density_signal(w)

        composite = (
            elab_signal    * 0.50 +
            frag_signal    * 0.30 +
            density_signal * 0.20
        )
        return max(0.0, min(1.0, composite))

    def _elaboration_ratio(self, w: _ElabWindow) -> float:
        easy_words = [
            wc for wc, d in zip(w.word_counts, w.difficulties) if d < 0.40
        ]
        hard_words = [
            wc for wc, d in zip(w.word_counts, w.difficulties) if d > 0.60
        ]

        if not easy_words and not hard_words:
            overall_mean = statistics.mean(w.word_counts) if w.word_counts else 60
            if overall_mean > 120:
                return 0.72
            if overall_mean < 25:
                return 0.28
            return 0.5

        if easy_words and hard_words:
            easy_mean = statistics.mean(easy_words)
            hard_mean = statistics.mean(hard_words)
            if easy_mean < 1:
                return 0.5
            ratio = easy_mean / max(1, hard_mean)
            # ratio > 1: says more on easy than hard = over-elab on easy
            # ratio < 1: says more on hard than easy = appropriate scaling
            signal = math.tanh((ratio - 1.0) * 1.2) * 0.4
            return max(0.0, min(1.0, 0.5 + signal))

        if easy_words:
            m = statistics.mean(easy_words)
            if m > 160:
                return 0.80
            if m > 100:
                return 0.65
            return 0.5

        m = statistics.mean(hard_words)
        if m < 20:
            return 0.22
        if m < 40:
            return 0.35
        return 0.5

    def _fragment_signal(self, w: _ElabWindow) -> float:
        if len(w.frag_rates) < 2:
            return 0.5
        recent = w.frag_rates[-3:]
        early  = w.frag_rates[:3]
        recent_mean = statistics.mean(recent)
        early_mean  = statistics.mean(early)
        delta = recent_mean - early_mean
        if delta > _FRAG_THRESHOLD:
            return max(0.0, 0.5 - delta * 1.5)
        return 0.5

    def _density_signal(self, w: _ElabWindow) -> float:
        if w.baseline_density is None or len(w.densities) < 4:
            return 0.5
        recent_density = statistics.mean(w.densities[-3:])
        drop = w.baseline_density - recent_density
        if drop > _DENSITY_DROP:
            # tanh scaling: 2× threshold → signal = 0.26 (meaningful degradation marker)
            depth = math.tanh((drop - _DENSITY_DROP) / _DENSITY_DROP * 1.6) * 0.42
            return max(0.0, 0.5 - depth)
        return 0.5

    def structure_degraded(self, session_id: str) -> bool:
        w = self._windows.get(session_id)
        if not w or w.n < 4:
            return False
        recent = statistics.mean(w.frag_rates[-3:])
        return recent > _FRAG_THRESHOLD

    def verbosity_profile(self, session_id: str) -> str:
        w = self._windows.get(session_id)
        if not w or not w.word_counts:
            return "unknown"
        m = statistics.mean(w.word_counts)
        if m < 25:
            return "terse"
        if m < 60:
            return "concise"
        if m < 110:
            return "moderate"
        if m < 180:
            return "verbose"
        return "excessive"

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
