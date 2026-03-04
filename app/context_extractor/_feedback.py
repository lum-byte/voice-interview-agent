from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.context_extractor._core import BehavioralVector, ConstraintSet

if TYPE_CHECKING:
    pass

_LR             = 0.035
_MIN_RECORDS    = 4
_SIMPLEX_EPS    = 1e-6
_EWMA_ALPHA     = 0.55
_NOISE_SIGMA    = 0.008


@dataclass
class _Record:
    vec:        tuple[float, ...]
    cs_active:  list[str]
    eval_score: float | None = None
    ts:         float = field(default_factory=time.monotonic)

    @property
    def complete(self) -> bool:
        return self.eval_score is not None


def _project_simplex(w: list[float]) -> list[float]:
    """
    Euclidean projection onto the probability simplex.
    Guarantees w[i] >= 0, sum(w) = 1 after projection.
    """
    n = len(w)
    u = sorted(w, reverse=True)
    cssv = 0.0
    rho  = 0
    for i in range(n):
        cssv += u[i]
        if u[i] - (cssv / (i + 1)) > 0:
            rho = i
    theta = (sum(u[:rho + 1]) - 1.0) / (rho + 1)
    return [max(0.0, x - theta) for x in w]


class FeedbackLoop:
    """
    Online weight calibration for BehavioralVector dimensions.

    After each scored turn arrives, we compute prediction error between
    the composite vector score and the eval_engine outcome. A projected
    gradient ascent step updates the per-dimension weights on the
    probability simplex — ensuring weights stay non-negative and sum to 1.

    This makes the compiler adaptive: dimensions that actually predict
    eval scores for this candidate gain weight; noisy dimensions shrink.

    After _MIN_RECORDS complete records, the loop switches from flat
    weights to learned weights. Until then, equal weighting applies.
    """

    def __init__(self) -> None:
        self._records:  dict[str, list[_Record]]  = defaultdict(list)
        self._weights:  dict[str, list[float]]    = {}
        self._ewma_err: dict[str, float]          = defaultdict(float)
        self._n:        dict[str, int]            = defaultdict(int)

    def record(
        self,
        session_id:  str,
        turn_index:  int,
        vec:         BehavioralVector,
        cs:          ConstraintSet,
    ) -> None:
        rec = _Record(
            vec       = vec.v,
            cs_active = list(cs.forbidden_tones) + list(cs.forbidden_patterns),
        )
        records = self._records[session_id]
        if len(records) > 48:
            records.pop(0)
        records.append(rec)

    def push_score(
        self,
        session_id: str,
        turn_index: int,
        score:      float,
    ) -> list[float] | None:
        records = self._records[session_id]
        unscored = [r for r in records if not r.complete]
        if not unscored:
            return None

        target = unscored[0]
        target.eval_score = score

        complete = [r for r in records if r.complete]
        if len(complete) < _MIN_RECORDS:
            return None

        return self._step(session_id, complete)

    def _step(self, session_id: str, records: list[_Record]) -> list[float]:
        weights = self._weights.get(session_id, [1/8] * 8)

        for rec in records[-6:]:
            v = rec.vec
            y = rec.eval_score

            predicted = sum(weights[i] * v[i] for i in range(min(8, len(v))))

            error = y - predicted
            self._ewma_err[session_id] = (
                _EWMA_ALPHA * abs(error) +
                (1 - _EWMA_ALPHA) * self._ewma_err[session_id]
            )

            import random as _r
            grad = [
                error * v[i] + _r.gauss(0, _NOISE_SIGMA)
                for i in range(min(8, len(v)))
            ]

            raw = [weights[i] + _LR * grad[i] for i in range(8)]
            weights = _project_simplex(raw)

        self._weights[session_id] = weights
        return weights

    def current_weights(self, session_id: str) -> list[float]:
        return self._weights.get(session_id, [1/8] * 8)

    def prediction_error(self, session_id: str) -> float:
        return self._ewma_err.get(session_id, 0.0)

    def calibrated(self, session_id: str) -> bool:
        complete = [r for r in self._records.get(session_id, []) if r.complete]
        return len(complete) >= _MIN_RECORDS

    def evict(self, session_id: str) -> None:
        self._records.pop(session_id, None)
        self._weights.pop(session_id, None)
        self._ewma_err.pop(session_id, None)
        self._n.pop(session_id, None)
