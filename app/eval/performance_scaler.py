"""
performance_scaler.py  ─  PerformanceScaler
════════════════════════════════════════════

DROP LOCATION: qa_controller.py, line 3628 — paste the entire contents
of this file directly above the comment block:
    # ── voice_graph integration: build_next_llm_input ─────────────────────────────

CONNECT (3 surgical edits to qa_controller.py):
────────────────────────────────────────────────
  [1] QAControllerV2.__init__() — add at end of method body:
          self._scaler = PerformanceScaler()

  [2] QAControllerV2.seed_from_intro() — after Redis write succeeds, add:
          await self._scaler.initialize_session(
              session_id=session_id,
              stated_level=ats_result.level,
              domains=doc.domains,
          )

  [3] QAControllerV2.get_llm_input() — just before constructing LLMInterviewInput,
      replace the raw `doc.candidate.level` usage with:
          _sig = await self._scaler.get_current_signal(session_id, doc.active_domain)
          _effective_level = _sig.effective_level
          _probe_flag      = _sig.probe_flag
      Then pass `_effective_level` where `doc.candidate.level` was used.

  [EVAL HOOK] In evaluation_engine.py — after scoring a turn, call:
          await qa_controller._scaler.push_eval_score(
              session_id=session_id,
              turn_index=turn_index,
              normalized_score=score_0_to_1,
              domain=domain,
              answer_text=answer_text,
          )

Architecture
────────────
A candidate's stated level is a PRIOR, not ground truth.
PerformanceScaler treats the interview as a Bayesian inference problem
where every eval-scored answer is evidence. Seven interoperating engines
maintain a probabilistic belief about the candidate's TRUE skill level
per domain, and continuously converge on the Zone of Proximal Development.

  1. BayesianBeliefEngine       — P(true_level | stated_level, scored_answers)
  2. DoubtEngine                — Calibrated skepticism toward self-reported levels
  3. ZPDHunter                  — Tracks zone of proximal development in real time
  4. TrajectoryAnalyzer         — Detects score momentum shapes: rise/plateau/collapse
  5. InformationGainOracle      — Picks difficulty that maximally resolves uncertainty
  6. ProbeOracle                — Distinguishes genuine knowledge from lucky answers
  7. DomainCrossInferenceGraph  — Shares skill evidence across correlated domains

Design axioms
─────────────
A. "beginner" stated → start at intermediate. Prove downward, not upward.
B. No level stated  → assume intermediate-to-advanced. Silence ≠ humility.
C. The scaler operates on EVAL SCORES, not raw answers. It trusts eval_engine.
D. Plateau at high scores → level is confirmed. Stop escalating.
E. ZPD target: 68%. Boredom (>85%) and panic (<45%) both kill signal quality.
F. Cross-domain inference uses correlation coefficients, never identity.
G. Probe fires on variance spikes — variance reveals luck, not knowledge.
H. The minimal context contract (domain|level|last_q|last_a|switch_flag) is
   preserved. `level` is already in the contract; we just make it dynamic.
"""

from __future__ import annotations

import asyncio
import json # noqa
from app.monitoring.observability import get_logger
import math
import statistics
import time
from collections import defaultdict, deque # noqa
from dataclasses import dataclass, field, asdict # noqa
from enum import Enum
from typing import Any, Literal, Optional # noqa

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# § CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Level → numeric mapping for continuous math
LEVEL_NUMERIC: dict[str, float] = {
    "beginner":     0.0,
    "intermediate": 0.5,
    "advanced":     1.0,
}
LEVELS: list[str] = ["beginner", "intermediate", "advanced"]
N_LEVELS = 3

# ── ZPD (Zone of Proximal Development) thresholds ─────────────────────────────
# Based on mastery-learning research: ~68% success rate maximizes learning signal.
# Below 45%: frustration zone (too hard — candidate shuts down).
# Above 85%: comfort zone  (too easy — no discriminative signal).
ZPD_TARGET      = 0.68
ZPD_IDEAL_LOW   = 0.58
ZPD_IDEAL_HIGH  = 0.78
ZPD_FLOOR       = 0.42   # hard lower bound before de-escalation fires
ZPD_CEILING     = 0.86   # hard upper bound before escalation fires

# ── Bayesian likelihood parameters ────────────────────────────────────────────
# Expected score = sigmoid(SLOPE * (true_level_numeric - question_difficulty) + BIAS)
# SLOPE: steepness of the S-curve mapping skill-gap → expected performance
# BIAS:  small positive offset → partial credit baseline even on hard questions
LIKELIHOOD_SLOPE = 3.2
LIKELIHOOD_BIAS  = 0.08
LIKELIHOOD_SIGMA = 0.16   # Gaussian noise around expected score

# ── Doubt / skepticism parameters ────────────────────────────────────────────
# How much we compress the prior UPWARD from stated level.
# Key insight: people systematically understate when modest, rarely overstate
# when being evaluated, so we assume upward bias in true ability.
STATED_SKEPTICISM: dict[str | None, float] = {
    "beginner":     0.82,   # Strong skepticism: freshers often undersell badly
    "intermediate": 0.55,   # Moderate: could genuinely be int or be adv-modest
    "advanced":     0.20,   # Low: advanced claims are hard to fake across domains
    None:           0.65,   # No claim → lean intermediate-high
}
# Number of scored turns before doubt fully dissolves into evidence
DOUBT_EVIDENCE_HORIZON = 6

# ── Trajectory parameters ─────────────────────────────────────────────────────
EWM_ALPHA               = 0.72   # Exponential weight decay for recent scores
MIN_TURNS_FOR_TREND     = 3      # Need at least this many before trusting slope
SLOPE_RISE_THRESHOLD    = 0.045  # slope/turn above this → RISING trajectory
SLOPE_FALL_THRESHOLD    = -0.045 # slope/turn below this → FALLING trajectory
PLATEAU_SLOPE_BAND      = 0.025  # |slope| < this → PLATEAU

# ── Information gain parameters ───────────────────────────────────────────────
IG_CERTAINTY_ENTROPY_THRESHOLD = 0.25  # below this → already certain, skip IG
IG_SCORE_SAMPLES = 11                  # resolution for score expectation integral

# ── Probe oracle parameters ───────────────────────────────────────────────────
PROBE_VARIANCE_THRESHOLD    = 0.175  # score variance above this → suspect luck
PROBE_SHORT_ANSWER_WORDS    = 18     # answers below this word count get probed
PROBE_SPIKE_DELTA           = 0.38   # single-turn score jump above this → verify
PROBE_STREAK_NEEDED         = 2      # N consecutive high scores before we trust plateau

# ── ZPD momentum: N questions needed to confirm a band change ─────────────────
ZPD_CONFIRMATION_TURNS = 2

# ── Domain correlation graph ─────────────────────────────────────────────────
# Partial evidence transfer: strong performance in domain A partially
# elevates the prior in domain B by this correlation coefficient.
DOMAIN_CORRELATION: dict[tuple[str, str], float] = {
    ("python",        "javascript"):    0.28,
    ("python",        "dsa"):           0.42,
    ("python",        "databases"):     0.33,
    ("python",        "ml"):            0.55,
    ("javascript",    "typescript"):    0.82,
    ("javascript",    "react"):         0.72,
    ("javascript",    "nodejs"):        0.66,
    ("javascript",    "dsa"):           0.30,
    ("java",          "cpp"):           0.38,
    ("java",          "csharp"):        0.45,
    ("java",          "dsa"):           0.52,
    ("java",          "system_design"): 0.35,
    ("cpp",           "rust"):          0.46,
    ("cpp",           "dsa"):           0.62,
    ("cpp",           "os_concepts"):   0.50,
    ("rust",          "golang"):        0.35,
    ("dsa",           "system_design"): 0.42,
    ("databases",     "system_design"): 0.52,
    ("system_design", "devops"):        0.44,
    ("devops",        "cloud"):         0.60,
    ("cloud",         "system_design"): 0.48,
    ("ml",            "dsa"):           0.44,
    ("ml",            "python"):        0.55,
    ("golang",        "system_design"): 0.38,
    ("os_concepts",   "system_design"): 0.46,
}

# ── Difficulty resolution within a level ─────────────────────────────────────
# Allows sub-level granularity without changing the level string.
# The numeric offset is added to level_numeric before the sigmoid.
DIFFICULTY_OFFSET: dict[str, float] = {
    "low":  -0.15,
    "mid":   0.00,
    "high": +0.15,
}


# ══════════════════════════════════════════════════════════════════════════════
# § DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class ScalerAction(str, Enum):
    """The concrete instruction the scaler sends to the QA controller."""
    HOLD           = "hold"          # same level, same difficulty band
    ESCALATE       = "escalate"      # move up within level or to next level
    DE_ESCALATE    = "de_escalate"   # move down within level or to previous
    PROBE_DEEP     = "probe_deep"    # stay at level, ask harder follow-up
    PROBE_LATERAL  = "probe_lateral" # stay at level, change angle/concept
    PROBE_VERIFY   = "probe_verify"  # ask candidate to explain their reasoning
    COAST          = "coast"         # level confirmed, ease slightly for closure


class Trajectory(str, Enum):
    RISING   = "rising"
    PLATEAU  = "plateau"
    FALLING  = "falling"
    UNKNOWN  = "unknown"    # not enough data yet


@dataclass
class ScoreEvent:
    """A single scored answer event. Immutable once written."""
    session_id:    str
    turn_index:    int
    domain:        str
    score:         float          # 0.0–1.0 from eval_engine
    difficulty:    float          # numeric difficulty the question was asked at
    answer_words:  int            # word count of candidate answer
    ts:            float = field(default_factory=time.monotonic)

    # Derived lazily
    @property
    def is_short_answer(self) -> bool:
        return self.answer_words < PROBE_SHORT_ANSWER_WORDS


@dataclass
class BeliefState:
    """
    Probability distribution over the three latent skill levels.
    Represents P(true_level) as a normalized probability vector.
    """
    probs: list[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])

    def entropy(self) -> float:
        """Shannon entropy of the belief distribution. Range: 0 (certain) → log2(3) (uniform)."""
        h = 0.0
        for p in self.probs:
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    @property
    def map_level(self) -> str:
        """Maximum A Posteriori level — the most probable single level."""
        idx = max(range(N_LEVELS), key=lambda i: self.probs[i])
        return LEVELS[idx]

    @property
    def expected_numeric(self) -> float:
        """Expected value of true level on [0, 0.5, 1.0] scale."""
        return sum(LEVEL_NUMERIC[LEVELS[i]] * self.probs[i] for i in range(N_LEVELS))

    def is_certain(self) -> bool:
        return self.entropy() < IG_CERTAINTY_ENTROPY_THRESHOLD

    def clone(self) -> "BeliefState":
        return BeliefState(probs=list(self.probs))

    def to_dict(self) -> dict:
        return {"probs": self.probs}

    @staticmethod
    def from_dict(d: dict) -> "BeliefState":
        return BeliefState(probs=list(d.get("probs", [1/3, 1/3, 1/3])))


@dataclass
class DomainScalerState:
    """
    All per-domain scaler state for a single session.
    Stored in-memory; reconstructible from ScoreEvent log.
    """
    domain:              str
    belief:              BeliefState = field(default_factory=BeliefState)
    score_history:       list[float] = field(default_factory=list)
    difficulty_history:  list[float] = field(default_factory=list)
    confirmed_level:     str | None  = None    # set once plateau detected
    zpd_consecutive:     int         = 0       # turns inside ZPD band
    current_difficulty:  float       = 0.5     # starts at intermediate
    probe_pending:       bool        = False
    probe_type:          ScalerAction | None = None
    last_action:         ScalerAction = ScalerAction.HOLD
    turns_at_level:      int         = 0       # turns since last level change
    initialized:         bool        = False


@dataclass
class SessionScalerState:
    """All scaler state for a session across domains."""
    session_id:    str
    stated_level:  str | None                     # raw from ATS, may be None
    doubt_coeff:   float                          # 0=trust stated, 1=full doubt
    domains:       list[str]                      = field(default_factory=list)
    domain_states: dict[str, DomainScalerState]  = field(default_factory=dict)
    global_belief: BeliefState                   = field(default_factory=BeliefState)
    total_scored:  int                            = 0
    created_at:    float                          = field(default_factory=time.monotonic)
    unscored_turns: dict[int, dict]               = field(default_factory=dict)
    # unscored_turns: turn_index → {domain, answer_words, difficulty}
    # Holds metadata for turns that haven't received eval scores yet.


@dataclass
class DifficultySignal:
    """
    The output of PerformanceScaler.get_current_signal().
    Everything the QA controller needs to adapt the next question.
    """
    session_id:       str
    domain:           str
    effective_level:  str          # the level string to pass into LLMInterviewInput
    difficulty_hint:  str          # "low" | "mid" | "high" within the level
    action:           ScalerAction
    probe_flag:       bool         # True → tell LLM to dig deeper on last answer
    probe_type:       ScalerAction | None
    belief_entropy:   float        # how uncertain we are (0=certain, 1.58=max)
    trajectory:       Trajectory
    confidence:       float        # 0.0–1.0: how much we trust this signal
    reasoning:        str          # one-line human-readable explanation
    engine_contributions: dict[str, Any] = field(default_factory=dict)
    # Per-engine attribution for observability and debugging.
    # Keys: "bayes", "doubt", "zpd", "trajectory", "ig", "probe", "cross"
    # Values: the dominant scalar or label each engine contributed.
    # Example: {"bayes": "intermediate(H=0.82)", "zpd": "escalate",
    #           "trajectory": "rising(slope=0.04,r2=0.91)",
    #           "ig": "diff=0.62", "probe": "none", "doubt": 0.55,
    #           "cross": "python→dsa(corr=0.70)"}
    ts:               float = field(default_factory=time.monotonic)

    def to_log_dict(self) -> dict:
        return {
            "session_id":      self.session_id[:8],
            "domain":          self.domain,
            "effective_level": self.effective_level,
            "difficulty_hint": self.difficulty_hint,
            "action":          self.action.value,
            "probe_flag":      self.probe_flag,
            "trajectory":      self.trajectory.value,
            "confidence":      round(self.confidence, 3),
            "entropy":         round(self.belief_entropy, 3),
            "reasoning":       self.reasoning,
            "engines":         self.engine_contributions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 1: BayesianBeliefEngine
# ══════════════════════════════════════════════════════════════════════════════

class BayesianBeliefEngine:
    """
    Maintains P(true_level | evidence) using Bayes' theorem.

    Likelihood model
    ─────────────────
    For a candidate at true level L (numeric), answering a question at
    difficulty D (numeric), the expected score is:

        expected = sigmoid(SLOPE * (L - D) + BIAS)

    meaning: if your true level is well above the question difficulty,
    you should score near 1.0. If you're below the difficulty, near 0.1.

    Score variance is modeled as Gaussian noise with sigma=LIKELIHOOD_SIGMA,
    so the likelihood of observing score s is:

        P(s | L, D) ∝ exp(-0.5 * ((s - expected) / sigma)^2)

    We never observe true level directly. We only observe (score, difficulty)
    pairs. This engine accumulates evidence and narrows the posterior.

    Doubt integration
    ─────────────────
    Prior is NOT uniform. It is biased upward from stated_level using
    the DoubtEngine's skepticism coefficient. As scored turns accumulate,
    the prior's influence shrinks and the likelihood dominates.
    """

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable sigmoid
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    @staticmethod
    def expected_score(true_level_numeric: float, question_difficulty: float) -> float:
        """Predicted score for a candidate at true_level facing question_difficulty."""
        gap = true_level_numeric - question_difficulty
        return BayesianBeliefEngine._sigmoid(LIKELIHOOD_SLOPE * gap + LIKELIHOOD_BIAS)

    @staticmethod
    def log_likelihood(score: float, true_level_numeric: float, difficulty: float) -> float:
        """
        Log P(score | true_level, difficulty).
        Gaussian likelihood around expected_score.
        """
        expected = BayesianBeliefEngine.expected_score(true_level_numeric, difficulty)
        residual = score - expected
        return -0.5 * (residual / LIKELIHOOD_SIGMA) ** 2

    @staticmethod
    def update(belief: BeliefState, score: float, difficulty: float) -> BeliefState:
        """
        Bayesian update: posterior ∝ prior × likelihood.
        Returns a new BeliefState.
        """
        log_probs = []
        for i, level_name in enumerate(LEVELS):
            l_numeric = LEVEL_NUMERIC[level_name]
            log_p = math.log(max(belief.probs[i], 1e-12))
            log_p += BayesianBeliefEngine.log_likelihood(score, l_numeric, difficulty)
            log_probs.append(log_p)

        # Numerically stable softmax normalization
        max_lp = max(log_probs)
        unnorm = [math.exp(lp - max_lp) for lp in log_probs]
        total  = sum(unnorm)
        new_probs = [u / total for u in unnorm]

        return BeliefState(probs=new_probs)

    @staticmethod
    def build_prior(stated_level: str | None, doubt_coeff: float) -> BeliefState:
        """
        Build the initial prior from stated_level and doubt coefficient.

        Doubt coefficient controls how much we push upward from stated level:
          doubt_coeff=0.0 → prior exactly on stated_level
          doubt_coeff=1.0 → prior fully on one-level-above stated_level

        The "upward push" encodes the insight that candidates systematically
        understate in evaluation contexts. A stated "beginner" with high
        doubt gets a prior that peaks at intermediate, not beginner.
        """
        if stated_level is None or stated_level not in LEVEL_NUMERIC:
            # No stated level → lean intermediate-to-advanced
            return BeliefState(probs=[0.05, 0.55, 0.40])

        stated_idx = LEVELS.index(stated_level)
        above_idx  = min(stated_idx + 1, N_LEVELS - 1)

        # Base prior: narrow peak at stated level
        base = [0.0] * N_LEVELS
        base[stated_idx] = 1.0

        # Shifted prior: mass moved one level up
        shifted = [0.0] * N_LEVELS
        shifted[above_idx] = 1.0
        if above_idx > 0:
            shifted[above_idx - 1] = 0.15   # small residual below
            shifted[above_idx]    -= 0.15

        # Interpolate between base and shifted by doubt_coeff
        raw = [
            (1.0 - doubt_coeff) * base[i] + doubt_coeff * shifted[i]
            for i in range(N_LEVELS)
        ]

        # Normalize and add a small floor (Laplace smoothing)
        floor = 0.03
        raw   = [max(r, floor) for r in raw]
        total = sum(raw)
        return BeliefState(probs=[r / total for r in raw])

    @staticmethod
    def batch_update(
        prior: BeliefState,
        events: list[ScoreEvent],
        doubt_coeff: float, # noqa
        n_total_scored: int,
    ) -> BeliefState:
        """
        Apply multiple score events sequentially.
        Doubt weight decays as n_total_scored grows toward DOUBT_EVIDENCE_HORIZON.
        """
        belief = prior.clone()
        for ev in events:
            # Blend Bayesian update with prior pull (doubt decay)
            trust = min(1.0, n_total_scored / max(DOUBT_EVIDENCE_HORIZON, 1))
            updated = BayesianBeliefEngine.update(belief, ev.score, ev.difficulty)

            if trust < 1.0:
                # Partial update: blend posterior toward prior
                blend = [
                    trust * updated.probs[i] + (1.0 - trust) * prior.probs[i]
                    for i in range(N_LEVELS)
                ]
                total = sum(blend)
                belief = BeliefState(probs=[b / total for b in blend])
            else:
                belief = updated

        return belief


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 2: DoubtEngine
# ══════════════════════════════════════════════════════════════════════════════

class DoubtEngine:
    """
    Calibrates skepticism toward the candidate's stated level.

    Core idea: stated levels are unreliable priors. The doubt coefficient
    controls how aggressively we push the belief upward from the stated value.

    Doubt DECREASES as evidence accumulates (the scaler gains confidence).
    Doubt INCREASES if the stated level consistently mispredicts performance
    (e.g., stated "intermediate" but scoring 0.9 on hard questions repeatedly).

    The doubt engine also detects level OVERSTATEMENT (stated "advanced"
    but scoring poorly), in which case it adjusts downward instead.
    """

    @staticmethod
    def initial_doubt(stated_level: str | None) -> float:
        """Starting skepticism from the stated level alone."""
        return STATED_SKEPTICISM.get(stated_level, 0.60)

    @staticmethod
    def compute_doubt(
        stated_level:  str | None,
        score_events:  list[ScoreEvent],
        n_total:       int,
    ) -> float:
        """
        Dynamic doubt: starts at STATED_SKEPTICISM[stated_level],
        decays as evidence aligns, sharpens if stated level is clearly wrong.

        Returns the current doubt coefficient ∈ [0.0, 1.0].
        """
        base_doubt = DoubtEngine.initial_doubt(stated_level)

        if n_total == 0 or not score_events:
            return base_doubt

        # Evidence decay: doubt linearly diminishes as turns accumulate
        evidence_factor = min(1.0, n_total / DOUBT_EVIDENCE_HORIZON)
        decayed_doubt   = base_doubt * (1.0 - evidence_factor * 0.70)

        # Mismatch amplifier: if performance strongly diverges from stated level,
        # don't let doubt decay too fast — it means our initial adjustment was right
        if stated_level is not None and len(score_events) >= 2:
            stated_numeric = LEVEL_NUMERIC.get(stated_level, 0.5)
            recent         = score_events[-min(4, len(score_events)):]
            mean_score     = statistics.mean(ev.score for ev in recent)
            mean_diff      = statistics.mean(ev.difficulty for ev in recent)

            # Predicted mean if stated level were true
            predicted_score = BayesianBeliefEngine.expected_score(stated_numeric, mean_diff)
            mismatch        = abs(mean_score - predicted_score)

            # If mismatch is high, the stated level was wrong → maintain higher doubt
            mismatch_amplifier = min(0.55, mismatch * 1.8)
            decayed_doubt = max(decayed_doubt, mismatch_amplifier)

        return max(0.0, min(1.0, decayed_doubt))

    @staticmethod
    def detect_overstatement(
        stated_level:  str | None,
        score_events:  list[ScoreEvent],
    ) -> bool:
        """
        Returns True if there's strong evidence the stated level is OVERSTATED.
        Triggers downward bias in the belief prior instead of upward.
        """
        if stated_level != "advanced" or len(score_events) < 3:
            return False
        recent    = score_events[-4:]
        mean_sc   = statistics.mean(ev.score for ev in recent)
        mean_diff = statistics.mean(ev.difficulty for ev in recent)
        # At difficulty 0.7+ (advanced) and scoring < 0.45: likely overstated
        return mean_diff >= 0.65 and mean_sc < 0.45


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 3: ZPDHunter
# ══════════════════════════════════════════════════════════════════════════════

class ZPDHunter:
    """
    Tracks whether the current difficulty is inside the candidate's
    Zone of Proximal Development.

    ZPD is not a binary condition. This engine models it as a pressure
    gradient: the further from the ZPD target, the stronger the push
    toward recentering. Consecutive questions in the comfort or panic
    zone each add pressure; questions inside the ideal band discharge it.

    Algorithm
    ─────────
    1. Compute ZPD pressure from recent scores vs ZPD_TARGET.
    2. Accumulate pressure across turns (with decay for older turns).
    3. When pressure exceeds escalate/de-escalate threshold, fire action.
    4. Require ZPD_CONFIRMATION_TURNS consecutive questions in the new
       band before confirming the level change (prevents flip-flopping).
    """

    @staticmethod
    def zpd_pressure(score: float) -> float:
        """
        Signed pressure from a single score.
        Positive → pressure to escalate (score too high).
        Negative → pressure to de-escalate (score too low).
        Zero     → inside ideal band.
        """
        if score > ZPD_IDEAL_HIGH:
            return +(score - ZPD_IDEAL_HIGH) / (1.0 - ZPD_IDEAL_HIGH)
        elif score < ZPD_IDEAL_LOW:
            return -(ZPD_IDEAL_LOW - score) / ZPD_IDEAL_LOW
        return 0.0

    @staticmethod
    def accumulated_pressure(
        score_history:  list[float],
        window:         int = 5,
    ) -> float:
        """
        Exponentially-weighted accumulated ZPD pressure over recent scores.
        Uses EWM_ALPHA decay so that the most recent score has highest weight.
        """
        recent = score_history[-window:]
        if not recent:
            return 0.0

        weighted_pressure = 0.0
        total_weight      = 0.0
        n = len(recent)

        for i, s in enumerate(recent):
            w = EWM_ALPHA ** (n - 1 - i)
            weighted_pressure += w * ZPDHunter.zpd_pressure(s)
            total_weight      += w

        return weighted_pressure / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def zpd_action(
        score_history:  list[float],
        current_difficulty: float, # noqa
        turns_at_level: int,
    ) -> tuple[ScalerAction, str]:
        """
        Recommend an action based on accumulated ZPD pressure.

        Returns (action, reasoning).
        Requires at least 2 turns at current level before firing changes
        (prevents jitter from single-question noise).
        """
        if len(score_history) < 2 or turns_at_level < ZPD_CONFIRMATION_TURNS:
            return ScalerAction.HOLD, "too few turns to assess ZPD"

        pressure = ZPDHunter.accumulated_pressure(score_history)
        recent   = score_history[-3:]
        mean_r   = statistics.mean(recent)

        # Hard floor/ceiling override (extreme pressure → act immediately)
        if mean_r > ZPD_CEILING and turns_at_level >= 1:
            return ScalerAction.ESCALATE, f"mean_score={mean_r:.2f} above ceiling {ZPD_CEILING}"

        if mean_r < ZPD_FLOOR and turns_at_level >= 1:
            return ScalerAction.DE_ESCALATE, f"mean_score={mean_r:.2f} below floor {ZPD_FLOOR}"

        # Pressure-based gradual action
        if pressure > 0.35:
            return ScalerAction.ESCALATE, f"ZPD pressure={pressure:.2f} → escalate"
        elif pressure < -0.35:
            return ScalerAction.DE_ESCALATE, f"ZPD pressure={pressure:.2f} → de-escalate"
        else:
            return ScalerAction.HOLD, f"ZPD pressure={pressure:.2f} → inside band"


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 4: TrajectoryAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryAnalyzer:
    """
    Detects the SHAPE of the score sequence, not just its mean.

    Score shape carries information the mean cannot:
      - Rising trajectory: candidate is warming up / knowledge is breadth-first.
        Escalate faster than ZPD alone suggests.
      - Plateau at high score: true level found. Stop escalating. Gentle coast.
      - Plateau at low score: genuine skill floor. Back off, don't crush.
      - Falling trajectory: fatigue OR question difficulty jumped too fast.
        Check difficulty delta — if it was stable, falling = real ceiling hit.
      - Collapse: sudden sharp drop after plateau. Classic overextension.

    Algorithm: Exponentially-Weighted Regression
    ────────────────────────────────────────────
    Fits a weighted linear regression y = a + bx to the score history,
    where weight of data point i = EWM_ALPHA^(n-1-i).
    Recent points have weight ~1.0; oldest point has weight ~EWM_ALPHA^(n-1).
    Slope b is the trajectory rate in score units per turn.
    """

    @staticmethod
    def _ew_linear_regression(
        scores: list[float],
        alpha:  float = EWM_ALPHA,
    ) -> tuple[float, float, float]:
        """
        Exponentially weighted linear regression.
        Returns (intercept, slope, r_squared).
        slope > 0 = scores trending up.
        """
        n = len(scores)
        if n < 2:
            return (scores[0] if scores else 0.5, 0.0, 0.0) # noqa

        weights = [alpha ** (n - 1 - i) for i in range(n)]
        xs      = list(range(n))

        W  = sum(weights)
        Wx = sum(w * x for w, x in zip(weights, xs))
        Wy = sum(w * y for w, y in zip(weights, scores))
        Wx2 = sum(w * x ** 2 for w, x in zip(weights, xs))
        Wxy = sum(w * x * y for w, x, y in zip(weights, xs, scores))

        denom     = W * Wx2 - Wx ** 2
        if abs(denom) < 1e-12:
            return (Wy / W, 0.0, 0.0) # noqa

        slope     = (W * Wxy - Wx * Wy) / denom
        intercept = (Wy - slope * Wx) / W

        # Weighted R²
        y_pred   = [intercept + slope * x for x in xs]
        ss_res   = sum(w * (y - yp) ** 2 for w, y, yp in zip(weights, scores, y_pred))
        y_mean_w = Wy / W
        ss_tot   = sum(w * (y - y_mean_w) ** 2 for w, y in zip(weights, scores))
        r2       = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

        return (intercept, slope, max(0.0, r2)) # noqa

    @staticmethod
    def analyze(
        score_history: list[float],
        difficulty_history: list[float] | None = None, # noqa
    ) -> tuple[Trajectory, float, float]:
        """
        Analyze the score sequence.
        Returns (trajectory, slope_per_turn, confidence).

        Confidence here is R² from the regression — how well the linear
        model fits the actual score sequence. Low R² = volatile/noisy.
        """
        n = len(score_history)
        if n < MIN_TURNS_FOR_TREND:
            return (Trajectory.UNKNOWN, 0.0, 0.0) # noqa

        intercept, slope, r2 = TrajectoryAnalyzer._ew_linear_regression(score_history)
        mean_score = statistics.mean(score_history[-4:]) if n >= 4 else statistics.mean(score_history) # noqa

        # Collapse detection: rapid drop in most recent 2 turns
        if n >= 3:
            recent_slope_raw = score_history[-1] - score_history[-2]
            if recent_slope_raw < -0.30 and slope > SLOPE_FALL_THRESHOLD:
                # Override: sudden collapse despite generally OK trend
                return (Trajectory.FALLING, recent_slope_raw, r2) # noqa

        if slope > SLOPE_RISE_THRESHOLD:
            return (Trajectory.RISING, slope, r2) # noqa
        elif slope < SLOPE_FALL_THRESHOLD:
            return (Trajectory.FALLING, slope, r2) # noqa
        else:
            # Plateau: check if it's a high-plateau (ceiling found) or low (floor)
            return (Trajectory.PLATEAU, slope, r2) # noqa

    @staticmethod
    def plateau_level(
        score_history:   list[float],
        trajectory:      Trajectory,
    ) -> str | None:
        """
        If trajectory is PLATEAU, estimate the level at which the plateau formed.
        Returns the confirmed level string, or None if not a reliable plateau.
        High plateau (>= ZPD_IDEAL_HIGH): level is confirmed current difficulty.
        Low plateau (< ZPD_IDEAL_LOW):    actual ceiling is one level below.
        """
        if trajectory != Trajectory.PLATEAU or len(score_history) < PROBE_STREAK_NEEDED:
            return None
        mean_sc = statistics.mean(score_history[-PROBE_STREAK_NEEDED:])
        if mean_sc >= ZPD_IDEAL_HIGH:
            return "plateau_high"
        elif mean_sc < ZPD_IDEAL_LOW:
            return "plateau_low"
        return "plateau_mid"


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 5: InformationGainOracle
# ══════════════════════════════════════════════════════════════════════════════

class InformationGainOracle:
    """
    Selects the next question difficulty that maximizes information gain
    about the candidate's true skill level.

    This is ACTIVE LEARNING applied to interview evaluation.
    Instead of always following ZPD (which optimizes for learning),
    the IG Oracle optimizes for MEASUREMENT efficiency.

    When the scaler is uncertain about the candidate's level (high entropy),
    it asks: "which difficulty value, if asked next, would most shrink
    our uncertainty about true level?" This is the difficulty that places
    the expected posterior at minimum expected entropy.

    Algorithm (Expected Information Gain)
    ──────────────────────────────────────
    For each candidate difficulty d ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:
      1. Sample IG_SCORE_SAMPLES scores uniformly in [0, 1].
      2. For each sampled score s, compute the posterior P(L | current + (d, s)).
      3. Compute expected entropy = weighted average of H(posterior_s),
         where weights are P(s | current_belief, d) — i.e., how likely
         each score is given what we currently believe.
      4. Information gain for d = H(current) - expected_entropy(d).
    Choose the d with highest information gain.

    Ceiling constraint: if we're already certain (entropy < threshold),
    skip the IG calculation entirely and fall back to ZPD guidance.
    """

    CANDIDATE_DIFFICULTIES = [0.0, 0.25, 0.5, 0.75, 1.0]
    SCORE_SAMPLES          = [i / (IG_SCORE_SAMPLES - 1) for i in range(IG_SCORE_SAMPLES)]

    @staticmethod
    def _expected_score_weight(
        score:   float,
        belief:  BeliefState,
        difficulty: float,
    ) -> float:
        """
        P(score | current_belief, difficulty).
        Marginalizes over true level using current belief as weights.
        """
        # Gaussian PDF value (unnormalized is fine — we normalize at the end)
        total = 0.0
        for i, level_name in enumerate(LEVELS):
            l_num    = LEVEL_NUMERIC[level_name]
            expected = BayesianBeliefEngine.expected_score(l_num, difficulty)
            gauss    = math.exp(-0.5 * ((score - expected) / LIKELIHOOD_SIGMA) ** 2)
            total   += belief.probs[i] * gauss
        return total

    @staticmethod
    def best_difficulty(belief: BeliefState) -> tuple[float, str]:
        """
        Returns (optimal_difficulty_numeric, reasoning).
        Skips computation if belief is already certain.
        """
        current_entropy = belief.entropy()

        if current_entropy < IG_CERTAINTY_ENTROPY_THRESHOLD:
            # Already certain — use ZPD target, not information gain
            best_d = LEVEL_NUMERIC.get(belief.map_level, 0.5)
            return (best_d, f"already certain (H={current_entropy:.3f}), using ZPD") # noqa

        best_d     = 0.5
        best_ig    = -1.0
        best_reason = ""

        for d in InformationGainOracle.CANDIDATE_DIFFICULTIES:
            # Compute score weights P(s | belief, d)
            raw_weights = [
                InformationGainOracle._expected_score_weight(s, belief, d)
                for s in InformationGainOracle.SCORE_SAMPLES
            ]
            total_w = sum(raw_weights)
            if total_w < 1e-12:
                continue
            weights = [w / total_w for w in raw_weights]

            # Expected posterior entropy under difficulty d
            expected_entropy = 0.0
            for s, w in zip(InformationGainOracle.SCORE_SAMPLES, weights):
                posterior = BayesianBeliefEngine.update(belief, s, d)
                expected_entropy += w * posterior.entropy()

            ig = current_entropy - expected_entropy

            if ig > best_ig:
                best_ig     = ig
                best_d      = d
                best_reason = (
                    f"IG={ig:.3f} bits at difficulty={d:.2f} "
                    f"(H_before={current_entropy:.3f}, H_after≈{expected_entropy:.3f})"
                )

        return (best_d, best_reason) # noqa

    @staticmethod
    def difficulty_to_level_hint(d: float) -> tuple[str, str]:
        """Convert numeric difficulty to (level_name, difficulty_hint) tuple."""
        if d < 0.20:
            return ("beginner", "low") # noqa
        elif d < 0.37:
            return ("beginner", "high") # noqa
        elif d < 0.50:
            return ("intermediate", "low") # noqa
        elif d < 0.63:
            return ("intermediate", "mid") # noqa
        elif d < 0.80:
            return ("intermediate", "high") # noqa
        elif d < 0.90:
            return ("advanced", "low") # noqa
        else:
            return ("advanced", "high") # noqa


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 6: ProbeOracle
# ══════════════════════════════════════════════════════════════════════════════

class ProbeOracle:
    """
    Decides when to PROBE instead of advance to the next question.

    A probe fires when the evidence suggests the last answer was unreliable:
      - Variance spike: scores fluctuating wildly → lucky guesses likely
      - Short answer: candidate answered briefly → surface knowledge only
      - Score spike after low: sudden jump → verify, might be a fluke
      - First question in domain: always probe second question to confirm

    Probe types
    ───────────
    PROBE_DEEP:     Ask a harder question on the SAME concept.
                    Use when: short answer + high score (suspicious).
    PROBE_LATERAL:  Ask about a RELATED concept at the same level.
                    Use when: variance spike (might know one sub-topic only).
    PROBE_VERIFY:   Ask the candidate to EXPLAIN their reasoning.
                    Use when: score spike after collapse (possible memorized answer).

    The oracle returns a (should_probe, probe_type, confidence) triple.
    Confidence reflects how certain we are that the probe is warranted.
    """

    @staticmethod
    def _score_variance(score_history: list[float], window: int = 5) -> float:
        recent = score_history[-window:]
        if len(recent) < 2:
            return 0.0
        return statistics.variance(recent)

    @staticmethod
    def evaluate(
        score_history:    list[float],
        last_answer_words: int,
        q_index_in_domain: int,
        last_score:        float,
    ) -> tuple[bool, ScalerAction | None, float, str]:
        """
        Returns: (should_probe, probe_type, confidence, reasoning)
        """
        n = len(score_history)
        if n == 0:
            return (False, None, 0.0, "no score history") # noqa

        reasons: list[str] = []
        probe_scores: dict[ScalerAction, float] = defaultdict(float)

        # ── Signal 1: Answer brevity ────────────────────────────────────────
        if last_answer_words < PROBE_SHORT_ANSWER_WORDS and last_score > 0.65:
            weight = (0.65 - (last_answer_words / PROBE_SHORT_ANSWER_WORDS)) * 0.8
            probe_scores[ScalerAction.PROBE_DEEP] += weight
            reasons.append(f"short_answer({last_answer_words}w) + high_score({last_score:.2f})")

        # ── Signal 2: Score variance spike ──────────────────────────────────
        variance = ProbeOracle._score_variance(score_history)
        if variance > PROBE_VARIANCE_THRESHOLD:
            weight = min(1.0, (variance - PROBE_VARIANCE_THRESHOLD) / 0.10)
            probe_scores[ScalerAction.PROBE_LATERAL] += weight * 0.7
            reasons.append(f"variance_spike({variance:.3f})")

        # ── Signal 3: Score spike after low (anomaly detection) ─────────────
        if n >= 2:
            prev_score  = score_history[-2]
            delta       = last_score - prev_score
            if delta > PROBE_SPIKE_DELTA and prev_score < 0.50:
                weight = min(1.0, delta / 0.50)
                probe_scores[ScalerAction.PROBE_VERIFY] += weight * 0.9
                reasons.append(f"score_spike({prev_score:.2f}→{last_score:.2f}, Δ={delta:.2f})")

        # ── Signal 4: First domain question always probes second ─────────────
        if q_index_in_domain == 0 and last_score > 0.70:
            # First question went well — second should probe deeper to verify
            probe_scores[ScalerAction.PROBE_DEEP] += 0.60
            reasons.append("first_question_verify")

        # ── Signal 5: Consecutively perfect then perfect (suspicious ceiling) ─
        if n >= PROBE_STREAK_NEEDED:
            streak_scores = score_history[-PROBE_STREAK_NEEDED:]
            if all(s > 0.88 for s in streak_scores):
                probe_scores[ScalerAction.PROBE_DEEP] += 0.50
                reasons.append(f"perfect_streak({PROBE_STREAK_NEEDED})")

        if not probe_scores:
            return (False, None, 0.0, "no probe signals") # noqa

        best_action = max(probe_scores, key=lambda a: probe_scores[a])
        confidence  = min(1.0, probe_scores[best_action])

        if confidence < 0.40:
            return (False, None, confidence, f"below threshold: {'; '.join(reasons)}") # noqa

        return (True, best_action, confidence, f"probe_warranted: {'; '.join(reasons)}") # noqa


# ══════════════════════════════════════════════════════════════════════════════
# § ENGINE 7: DomainCrossInferenceGraph
# ══════════════════════════════════════════════════════════════════════════════

class DomainCrossInferenceGraph:
    """
    Transfers skill evidence across correlated domains.

    When the scaler has scored evidence in domain A, and domain B is
    correlated with A at coefficient r, then domain B's belief prior
    is initialized with a partial update from A's posterior.

    This matters most when the candidate is interviewed on a NEW domain
    before we have any direct evidence in that domain. Without cross-inference,
    we always start from the stated_level prior (which may be poorly calibrated).
    With cross-inference, we start from a belief that's already been partially
    updated by correlated evidence.

    Algorithm
    ─────────
    For target domain T with no direct evidence yet:
      1. Find all domains D where corr(D, T) > 0 and D has a trained belief.
      2. For each such D, weight its contribution by:
           contribution_weight = corr(D, T) * evidence_weight(D)
         where evidence_weight(D) = min(1, len(D.score_history) / 4)
      3. Blend the contributions into an initial belief for T:
           prior_T = normalize(Σ_D [w_D * belief_D.probs])
      4. Mix with the doubt-adjusted stated_level prior:
           final_prior = α * cross_prior + (1 - α) * stated_prior
         where α = min(0.45, max_contribution_weight)
         (we cap α at 0.45 so direct stated level never fully disappears)
    """

    @staticmethod
    def _get_correlation(d1: str, d2: str) -> float:
        if d1 == d2:
            return 1.0
        return (
            DOMAIN_CORRELATION.get((d1, d2))
            or DOMAIN_CORRELATION.get((d2, d1))
            or 0.0
        )

    @staticmethod
    def build_cross_prior(
        target_domain:  str,
        domain_states:  dict[str, "DomainScalerState"],
        stated_prior:   BeliefState,
    ) -> BeliefState:
        """
        Build a cross-inferred prior for target_domain based on
        evidence already available in other domains.
        """
        blended_probs  = [0.0] * N_LEVELS
        total_weight   = 0.0

        for other_domain, other_state in domain_states.items():
            if other_domain == target_domain:
                continue
            if not other_state.score_history:
                continue

            corr = DomainCrossInferenceGraph._get_correlation(target_domain, other_domain)
            if corr < 0.05:
                continue

            # Evidence weight: how much we trust other_domain's belief
            ev_weight = min(1.0, len(other_state.score_history) / 4.0)
            w = corr * ev_weight

            for i in range(N_LEVELS):
                blended_probs[i] += w * other_state.belief.probs[i]
            total_weight += w

        if total_weight < 0.05:
            # No useful cross-domain evidence → return stated_prior unchanged
            return stated_prior

        # Normalize cross prior
        cross_probs = [p / total_weight for p in blended_probs]

        # Blend with stated prior
        alpha = min(0.45, total_weight / 2.0)  # cap influence of cross-inference
        final = [
            alpha * cross_probs[i] + (1.0 - alpha) * stated_prior.probs[i]
            for i in range(N_LEVELS)
        ]
        total = sum(final)
        return BeliefState(probs=[p / total for p in final])


# ══════════════════════════════════════════════════════════════════════════════
# § ORCHESTRATOR: PerformanceScaler
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceScaler:
    """
    PerformanceScaler — Adaptive difficulty engine for qa_controller.py.

    Orchestrates all seven sub-engines to produce a DifficultySignal
    at each turn. The signal drives two things in qa_controller:
      1. The `level` field in LLMInterviewInput (which question difficulty to use).
      2. The `probe_flag` (whether to ask a probing follow-up vs a fresh question).

    State management
    ─────────────────
    All state is held in memory in `_sessions` (dict keyed by session_id).
    State survives for the duration of the session.
    No external storage is required — the scaler reconstructs its state
    from score events alone if the process is restarted (via rebuild()).

    Thread/async safety
    ────────────────────
    All public methods are async. Internal state mutations are protected
    by per-session asyncio.Lock to prevent concurrent update races in
    the rare case where two eval scores arrive simultaneously (e.g., fast
    back-to-back turns from a WebSocket stream).

    Default signal
    ───────────────
    If no state exists for a session (e.g., score hasn't arrived yet),
    get_current_signal() returns a safe default that reflects the
    stated level adjusted for doubt. It never blocks or raises.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionScalerState] = {}
        self._locks:    dict[str, asyncio.Lock]       = {}
        log.info("performance_scaler_initialized")

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    # ── Public API ─────────────────────────────────────────────────────────────

    async def initialize_session(
        self,
        session_id:    str,
        stated_level:  str | None,
        domains:       list[str],
    ) -> None:
        """
        Called from QAControllerV2.seed_from_intro() after ATS extraction.

        Sets up the doubt coefficient, builds the initial Bayesian prior,
        and initializes per-domain state for all extracted domains.
        """
        async with self._get_lock(session_id):
            doubt = DoubtEngine.initial_doubt(stated_level)
            prior = BayesianBeliefEngine.build_prior(stated_level, doubt)

            session_state = SessionScalerState(
                session_id   = session_id,
                stated_level = stated_level,
                doubt_coeff  = doubt,
                domains      = list(domains),
                global_belief = prior.clone(),
            )

            # Pre-initialize per-domain states with the global prior
            for domain in domains:
                d_state = DomainScalerState(domain=domain)
                d_state.belief             = prior.clone()
                d_state.current_difficulty = prior.expected_numeric
                d_state.initialized        = True
                session_state.domain_states[domain] = d_state

            self._sessions[session_id] = session_state

            log.info(
                "scaler_session_initialized",
                session_id    = session_id[:8],
                stated_level  = stated_level,
                doubt_coeff   = round(doubt, 3),
                prior_map     = prior.map_level,
                prior_entropy = round(prior.entropy(), 3),
                domains       = domains,
            )

    async def push_eval_score(
        self,
        session_id:       str,
        turn_index:       int,
        normalized_score: float,
        domain:           str,
        answer_text:      str = "",
    ) -> None:
        """
        Called by eval_engine after scoring a turn.
        This is the primary data ingestion point.

        normalized_score: 0.0–1.0 (eval_engine output, already normalized).
        """
        async with self._get_lock(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                # Session not initialized (e.g., process restart) — auto-init
                log.warning(
                    "scaler_auto_init_on_score",
                    session_id = session_id[:8],
                    domain     = domain,
                )
                await self._auto_init(session_id, domain)
                state = self._sessions[session_id]

            # Resolve current difficulty at time of this turn
            d_state = self._ensure_domain(state, domain)
            difficulty = d_state.current_difficulty

            # Resolve unscored metadata if available
            answer_words = 0
            if turn_index in state.unscored_turns:
                meta         = state.unscored_turns.pop(turn_index)
                answer_words = meta.get("answer_words", 0)
                difficulty   = meta.get("difficulty", difficulty)

            if answer_text and not answer_words:
                answer_words = len(answer_text.split())

            event = ScoreEvent(
                session_id   = session_id,
                turn_index   = turn_index,
                domain       = domain,
                score        = max(0.0, min(1.0, normalized_score)),
                difficulty   = difficulty,
                answer_words = answer_words,
            )

            # Update domain state
            d_state.score_history.append(event.score)
            d_state.difficulty_history.append(event.difficulty)
            state.total_scored += 1

            # Recompute doubt coefficient
            state.doubt_coeff = DoubtEngine.compute_doubt(
                stated_level  = state.stated_level,
                score_events  = self._domain_events(d_state),
                n_total       = state.total_scored,
            )

            # Bayesian update for this domain
            updated_belief = BayesianBeliefEngine.update(
                belief     = d_state.belief,
                score      = event.score,
                difficulty = event.difficulty,
            )

            # Blend with doubt (trust decay)
            trust = min(1.0, state.total_scored / max(DOUBT_EVIDENCE_HORIZON, 1))
            prior = BayesianBeliefEngine.build_prior(state.stated_level, state.doubt_coeff)
            blended_probs = [
                trust * updated_belief.probs[i] + (1.0 - trust) * prior.probs[i]
                for i in range(N_LEVELS)
            ]
            total = sum(blended_probs)
            d_state.belief = BeliefState(probs=[p / total for p in blended_probs])

            # Update global belief: weighted average across all domains with evidence
            self._update_global_belief(state)

            # Update difficulty for next turn
            self._recompute_difficulty(state, d_state, event)

            log.debug(
                "scaler_score_ingested",
                session_id    = session_id[:8],
                domain        = domain,
                turn_index    = turn_index,
                score         = round(event.score, 3),
                difficulty    = round(event.difficulty, 3),
                map_level     = d_state.belief.map_level,
                entropy       = round(d_state.belief.entropy(), 3),
                doubt         = round(state.doubt_coeff, 3),
                next_diff     = round(d_state.current_difficulty, 3),
            )

    async def notify_turn_committed(
        self,
        session_id:    str,
        turn_index:    int,
        domain:        str,
        answer_text:   str,
        difficulty:    float | None = None,
    ) -> None:
        """
        Called from QAControllerV2.commit_turn() before eval score arrives.
        Stores turn metadata so push_eval_score() can use it when the score arrives.
        """
        async with self._get_lock(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                return

            d_state = state.domain_states.get(domain)
            current_diff = d_state.current_difficulty if d_state else 0.5

            state.unscored_turns[turn_index] = {
                "domain":       domain,
                "answer_words": len(answer_text.split()) if answer_text else 0,
                "difficulty":   difficulty if difficulty is not None else current_diff,
                "committed_at": time.monotonic(),
            }

    async def get_current_signal(
        self,
        session_id: str,
        domain:     str,
    ) -> DifficultySignal:
        """
        Called from QAControllerV2.get_llm_input() to get the current
        difficulty recommendation for the next question.

        Always returns a valid DifficultySignal. Never raises.
        Falls back to a safe default if no state exists.
        """
        try:
            async with self._get_lock(session_id):
                state = self._sessions.get(session_id)
                if state is None:
                    return self._default_signal(session_id, domain)

                d_state = self._ensure_domain(state, domain)
                return self._compute_signal(state, d_state, domain)

        except Exception as exc:
            log.error(
                "scaler_signal_error",
                session_id = session_id[:8],
                domain     = domain,
                error      = str(exc),
            )
            return self._default_signal(session_id, domain)

    async def get_session_report(self, session_id: str) -> dict[str, Any]:
        """
        Returns a diagnostic report for a session. Used in admin/eval tools.
        """
        async with self._get_lock(session_id):
            state = self._sessions.get(session_id)
            if state is None:
                return {"session_id": session_id, "error": "not_found"}

            domain_reports = {}
            for domain, d_state in state.domain_states.items():
                traj, slope, r2 = TrajectoryAnalyzer.analyze(
                    d_state.score_history,
                    d_state.difficulty_history,
                )
                domain_reports[domain] = {
                    "belief_probs":       [round(p, 3) for p in d_state.belief.probs],
                    "map_level":          d_state.belief.map_level,
                    "entropy":            round(d_state.belief.entropy(), 3),
                    "score_history":      [round(s, 3) for s in d_state.score_history],
                    "current_difficulty": round(d_state.current_difficulty, 3),
                    "confirmed_level":    d_state.confirmed_level,
                    "trajectory":         traj.value,
                    "slope_per_turn":     round(slope, 4),
                    "regression_r2":      round(r2, 3),
                    "last_action":        d_state.last_action.value,
                    "probe_pending":      d_state.probe_pending,
                    "turns_at_level":     d_state.turns_at_level,
                }

            return {
                "session_id":    session_id,
                "stated_level":  state.stated_level,
                "doubt_coeff":   round(state.doubt_coeff, 3),
                "total_scored":  state.total_scored,
                "global_belief": {
                    "probs":     [round(p, 3) for p in state.global_belief.probs],
                    "map_level": state.global_belief.map_level,
                    "entropy":   round(state.global_belief.entropy(), 3),
                },
                "domains": domain_reports,
            }

    async def evict_session(self, session_id: str) -> None:
        """Called on session close to free memory."""
        async with self._get_lock(session_id):
            self._sessions.pop(session_id, None)
        self._locks.pop(session_id, None)
        log.debug("scaler_session_evicted", session_id=session_id[:8])

    # ── Internal computation ───────────────────────────────────────────────────

    def _ensure_domain( # noqa
        self,
        state:  SessionScalerState,
        domain: str,
    ) -> DomainScalerState:
        """
        Gets or creates a DomainScalerState for a domain.
        On first creation, uses cross-domain inference to build a smarter prior.
        """
        if domain not in state.domain_states:
            stated_prior = BayesianBeliefEngine.build_prior(
                state.stated_level, state.doubt_coeff
            )
            cross_prior = DomainCrossInferenceGraph.build_cross_prior(
                target_domain = domain,
                domain_states = state.domain_states,
                stated_prior  = stated_prior,
            )

            d_state = DomainScalerState(domain=domain)
            d_state.belief             = cross_prior
            d_state.current_difficulty = cross_prior.expected_numeric
            d_state.initialized        = True
            state.domain_states[domain] = d_state

            log.debug(
                "scaler_domain_initialized",
                session_id   = state.session_id[:8],
                domain       = domain,
                prior_level  = cross_prior.map_level,
                prior_entropy= round(cross_prior.entropy(), 3),
            )

        return state.domain_states[domain]

    def _domain_events(self, d_state: DomainScalerState) -> list[ScoreEvent]: # noqa
        """
        Reconstruct minimal ScoreEvent list from stored history for Engines 2/3/4.
        Note: used for engines that only need score + difficulty, not full metadata.
        """
        events = []
        for sc, diff in zip(d_state.score_history, d_state.difficulty_history):
            events.append(ScoreEvent(
                session_id   = "",
                turn_index   = len(events),
                domain       = d_state.domain,
                score        = sc,
                difficulty   = diff,
                answer_words = PROBE_SHORT_ANSWER_WORDS,  # assume OK for history
            ))
        return events

    def _recompute_difficulty( # noqa
        self,
        state:   SessionScalerState,
        d_state: DomainScalerState,
        event:   ScoreEvent,
    ) -> None:
        """
        After ingesting a new score event, decide the difficulty for the
        NEXT question in this domain. This is the core decision logic.

        Decision priority (highest wins):
          1. Confirmed level (plateau found) → coast or light probe
          2. ZPD pressure exceeds threshold → escalate/de-escalate
          3. Information gain oracle → pick most informative difficulty
          4. Hold → no action needed
        """
        # ── Check for confirmed plateau ────────────────────────────────────────
        if d_state.confirmed_level is not None:
            # Level confirmed: gentle coast, no further escalation
            d_state.last_action = ScalerAction.COAST
            d_state.turns_at_level += 1
            return

        traj, slope, r2 = TrajectoryAnalyzer.analyze(
            d_state.score_history,
            d_state.difficulty_history,
        )
        plateau_type = TrajectoryAnalyzer.plateau_level(d_state.score_history, traj)

        if plateau_type == "plateau_high" and d_state.turns_at_level >= PROBE_STREAK_NEEDED:
            # High plateau: candidate has mastered this level
            d_state.confirmed_level = d_state.belief.map_level
            log.info(
                "scaler_level_confirmed",
                session_id      = state.session_id[:8],
                domain          = d_state.domain,
                confirmed_level = d_state.confirmed_level,
                mean_score      = round(statistics.mean(d_state.score_history[-3:]), 3),
            )
            d_state.last_action = ScalerAction.COAST
            return

        if plateau_type == "plateau_low" and d_state.turns_at_level >= PROBE_STREAK_NEEDED:
            # Low plateau: genuine floor found, confirmed level is one down
            current_idx   = LEVELS.index(d_state.belief.map_level)
            floor_level   = LEVELS[max(0, current_idx - 1)]
            d_state.confirmed_level = floor_level
            d_state.current_difficulty = LEVEL_NUMERIC[floor_level]
            d_state.last_action = ScalerAction.COAST
            log.info(
                "scaler_floor_confirmed",
                session_id   = state.session_id[:8],
                domain       = d_state.domain,
                floor_level  = floor_level,
            )
            return

        # ── ZPD action ─────────────────────────────────────────────────────────
        zpd_action, zpd_reason = ZPDHunter.zpd_action(
            score_history       = d_state.score_history,
            current_difficulty  = d_state.current_difficulty,
            turns_at_level      = d_state.turns_at_level,
        )

        # ── Trajectory modifier ────────────────────────────────────────────────
        # Rising trajectory: accelerate escalation (candidate is warming up fast)
        # Falling trajectory: slow down or reverse escalation
        if traj == Trajectory.RISING and zpd_action == ScalerAction.ESCALATE:
            # Boost: jump directly to MAP level, not just one step up
            target_diff = d_state.belief.expected_numeric + 0.12
        elif traj == Trajectory.FALLING and zpd_action == ScalerAction.ESCALATE:
            # Override: falling + escalate signal is contradictory → hold
            zpd_action  = ScalerAction.HOLD
            zpd_reason += " [override: falling traj]"
            target_diff = d_state.current_difficulty
        else:
            target_diff = d_state.current_difficulty

        # ── Information gain: use when still uncertain ─────────────────────────
        if (not d_state.belief.is_certain()
                and zpd_action == ScalerAction.HOLD
                and state.total_scored >= 2):
            ig_diff, ig_reason = InformationGainOracle.best_difficulty(d_state.belief)
            # Only override if IG suggests a meaningfully different difficulty
            if abs(ig_diff - d_state.current_difficulty) > 0.12:
                target_diff = ig_diff
                zpd_action  = ScalerAction.HOLD
                zpd_reason  = f"IG override: {ig_reason}"

        # ── Apply difficulty step ─────────────────────────────────────────────
        step = 0.15   # standard difficulty step per level change
        if zpd_action == ScalerAction.ESCALATE:
            d_state.current_difficulty = min(1.0, d_state.current_difficulty + step)
            d_state.turns_at_level     = 0
            d_state.last_action        = ScalerAction.ESCALATE
        elif zpd_action == ScalerAction.DE_ESCALATE:
            d_state.current_difficulty = max(0.0, d_state.current_difficulty - step)
            d_state.turns_at_level     = 0
            d_state.last_action        = ScalerAction.DE_ESCALATE
        else:
            # Subtle drift toward target_diff (soft targeting)
            drift = (target_diff - d_state.current_difficulty) * 0.25
            d_state.current_difficulty = max(0.0, min(1.0, d_state.current_difficulty + drift))
            d_state.turns_at_level    += 1
            d_state.last_action        = ScalerAction.HOLD

        # ── Probe oracle ───────────────────────────────────────────────────────
        last_answer_words = event.answer_words
        q_index = len(d_state.score_history) - 1

        should_probe, probe_type, probe_conf, probe_reason = ProbeOracle.evaluate(
            score_history     = d_state.score_history,
            last_answer_words = last_answer_words,
            q_index_in_domain = q_index,
            last_score        = event.score,
        )

        d_state.probe_pending = should_probe
        d_state.probe_type    = probe_type

        log.debug(
            "scaler_difficulty_updated",
            session_id   = state.session_id[:8],
            domain       = d_state.domain,
            new_diff     = round(d_state.current_difficulty, 3),
            action       = d_state.last_action.value,
            trajectory   = traj.value,
            zpd_reason   = zpd_reason,
            probe        = should_probe,
            probe_type   = probe_type.value if probe_type else None,
        )

    def _compute_signal( # noqa
        self,
        state:   SessionScalerState,
        d_state: DomainScalerState,
        domain:  str,
    ) -> DifficultySignal:
        """
        Compute the final DifficultySignal to return to the QA controller.
        This is read-only — no state mutations.
        """
        belief   = d_state.belief
        traj, slope, r2 = TrajectoryAnalyzer.analyze(
            d_state.score_history,
            d_state.difficulty_history,
        )

        # Effective level from MAP estimate of belief
        effective_level, diff_hint = InformationGainOracle.difficulty_to_level_hint(
            d_state.current_difficulty
        )

        # If level is confirmed, use that directly
        if d_state.confirmed_level is not None:
            effective_level = d_state.confirmed_level
            diff_hint       = "mid"
            action          = ScalerAction.COAST
        else:
            action = d_state.last_action

        # Confidence: 1 - entropy (scaled), boosted by R² of trajectory fit
        raw_conf   = 1.0 - (belief.entropy() / math.log2(N_LEVELS))
        confidence = min(1.0, raw_conf * (0.7 + 0.3 * r2))

        # Build human-readable reasoning
        scored_n = len(d_state.score_history)
        if scored_n == 0:
            reasoning = (
                f"no scores yet; stated={state.stated_level}; "
                f"doubt={state.doubt_coeff:.2f}; starting at {effective_level}-{diff_hint}"
            )
        else:
            mean_sc = round(statistics.mean(d_state.score_history[-3:]), 2)
            reasoning = (
                f"n={scored_n}; mean3={mean_sc}; traj={traj.value}; "
                f"belief_map={belief.map_level}(H={belief.entropy():.2f}); "
                f"doubt={state.doubt_coeff:.2f}; action={action.value}"
            )

        # ── Per-engine attribution ─────────────────────────────────────────────
        # One entry per engine. Gives exact visibility into which engine drove
        # any given level decision. Logged via to_log_dict() → structured logs.
        engine_contributions = {
            # Engine 1: Bayesian belief — MAP estimate + uncertainty
            "bayes": f"{belief.map_level}(H={belief.entropy():.2f},"
                     f"p={belief.probs[LEVELS.index(belief.map_level)]:.2f})",

            # Engine 2: DoubtEngine — skepticism coefficient
            "doubt": round(state.doubt_coeff, 3),

            # Engine 3: ZPDHunter — action + accumulated pressure
            "zpd": d_state.last_action.value,

            # Engine 4: TrajectoryAnalyzer — shape + regression quality
            "trajectory": f"{traj.value}(slope={slope:.3f},r2={r2:.2f})",

            # Engine 5: InformationGainOracle — difficulty it resolved to
            "ig": f"diff={d_state.current_difficulty:.2f}→{effective_level}-{diff_hint}",

            # Engine 6: ProbeOracle — pending probe or none
            "probe": (
                f"{d_state.probe_type.value}" if d_state.probe_pending and d_state.probe_type
                else "none"
            ),

            # Engine 7: DomainCrossInferenceGraph — noted at domain init;
            # confirmed_level shows if cross-prior was validated by scores
            "cross": (
                f"confirmed={d_state.confirmed_level}" if d_state.confirmed_level
                else f"map={belief.map_level}(unconfirmed)"
            ),
        }

        return DifficultySignal(
            session_id           = state.session_id,
            domain               = domain,
            effective_level      = effective_level,
            difficulty_hint      = diff_hint,
            action               = action,
            probe_flag           = d_state.probe_pending,
            probe_type           = d_state.probe_type,
            belief_entropy       = belief.entropy(),
            trajectory           = traj,
            confidence           = confidence,
            reasoning            = reasoning,
            engine_contributions = engine_contributions,
        )

    def _default_signal(self, session_id: str, domain: str) -> DifficultySignal: # noqa
        """
        Safe fallback signal when no state is available.
        Starts at intermediate — the stated-level-agnostic safe default.
        """
        return DifficultySignal(
            session_id           = session_id,
            domain               = domain,
            effective_level      = "intermediate",
            difficulty_hint      = "mid",
            action               = ScalerAction.HOLD,
            probe_flag           = False,
            probe_type           = None,
            belief_entropy       = math.log2(N_LEVELS),   # maximum uncertainty
            trajectory           = Trajectory.UNKNOWN,
            confidence           = 0.0,
            reasoning            = "default: no state found, safe intermediate fallback",
            engine_contributions = {"default": "no_state"},
        )

    def _update_global_belief(self, state: SessionScalerState) -> None: # noqa
        """
        Recompute global belief as evidence-weighted average of domain beliefs.
        Domains with more score history contribute more to global belief.
        """
        total_weight = 0.0
        blended      = [0.0] * N_LEVELS

        for d_state in state.domain_states.values():
            if not d_state.score_history:
                continue
            w = min(1.0, len(d_state.score_history) / 4.0)
            for i in range(N_LEVELS):
                blended[i] += w * d_state.belief.probs[i]
            total_weight += w

        if total_weight < 1e-6:
            return

        total = sum(blended)
        if total > 1e-12:
            state.global_belief = BeliefState(probs=[b / total for b in blended])

    async def _auto_init(self, session_id: str, domain: str) -> None: # noqa
        """Fallback init when session state is missing (e.g. after restart)."""
        state = SessionScalerState(
            session_id   = session_id,
            stated_level = None,
            doubt_coeff  = DoubtEngine.initial_doubt(None),
            global_belief = BayesianBeliefEngine.build_prior(None, 0.65),
        )
        self._sessions[session_id] = state


# ══════════════════════════════════════════════════════════════════════════════
# § INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_signal_to_llm_input(
    signal:     DifficultySignal,
    llm_input:  Any,  # LLMInterviewInput
) -> Any:
    """
    Utility: overwrite `level` on an LLMInterviewInput with the scaler's
    effective_level, and mark probe_flag if warranted.

    Use in get_llm_input() AFTER building the LLMInterviewInput but BEFORE
    returning it. This keeps the mutation co-located with the input builder.

    Example usage in qa_controller.py:
        llm_input = LLMInputBuilder(...).build(doc)
        signal    = await self._scaler.get_current_signal(session_id, domain)
        llm_input = apply_signal_to_llm_input(signal, llm_input)
        return llm_input
    """
    if llm_input is None:
        return None

    try:
        # Mutate level in place — dataclass field is not frozen
        object.__setattr__(llm_input, "level", signal.effective_level)

        # probe_flag informs the LLM system prompt if wired in voice_graph
        # It is also readable directly from the signal without touching LLMInterviewInput
        log.debug(
            "scaler_signal_applied",
            domain          = signal.domain,
            effective_level = signal.effective_level,
            action          = signal.action.value,
            probe           = signal.probe_flag,
            confidence      = round(signal.confidence, 3),
        )

    except Exception as exc:
        log.warning("scaler_signal_apply_failed", error=str(exc))

    return llm_input


# ══════════════════════════════════════════════════════════════════════════════
# § PROBE INSTRUCTION BUILDER
# Translates DifficultySignal.probe_flag into concrete LLM prompt directives.
# Injected into the system prompt in voice_graph.node_llm() when probe_flag=True.
# ══════════════════════════════════════════════════════════════════════════════

class ProbeInstructionBuilder:
    """
    Converts a DifficultySignal into a natural-language directive that
    the LLM question-engine understands.

    These instructions are appended to the system prompt ONLY when
    probe_flag=True. They are brief — the LLM already knows the domain,
    level, and last answer. The probe instruction only changes the
    INTENT of the next question without polluting the minimal context.

    Usage in voice_graph.node_llm() system prompt:
        if signal.probe_flag:
            extra = ProbeInstructionBuilder.build(signal)
            system_prompt += f"\\n\\nDIRECTIVE: {extra}"
    """

    _DEEP_TEMPLATES = [
        "The candidate's last answer was brief. Ask a significantly harder question "
        "on the same concept that requires them to demonstrate depth, not recall.",

        "The last answer seemed surface-level. Follow up with a question that asks "
        "the candidate to explain the WHY behind what they just said.",

        "Push deeper on the last concept. Ask a question that a junior engineer "
        "could not answer without real hands-on experience.",
    ]

    _LATERAL_TEMPLATES = [
        "The candidate's score pattern suggests they may know one angle of this topic. "
        "Ask about a closely related but distinct concept within the same domain.",

        "Probe laterally: move to an adjacent part of {domain} that typically "
        "trips up candidates who know the surface but not the internals.",

        "Ask about the failure modes or edge cases of the concept just discussed.",
    ]

    _VERIFY_TEMPLATES = [
        "Ask the candidate to walk you through HOW they would apply the concept "
        "they just described to a real production scenario.",

        "The last answer was unusually strong after a weak start. Ask the candidate "
        "to explain their reasoning step by step — not the answer, the process.",

        "Ask a concrete 'what would happen if...' question that tests whether the "
        "candidate truly understands the last concept or only recognized it.",
    ]

    import random as _rnd # noqa

    @staticmethod
    def build(signal: DifficultySignal) -> str:
        """
        Returns the probe instruction string for a given signal.
        Returns empty string if probe_flag is False.
        """
        if not signal.probe_flag or signal.probe_type is None:
            return ""

        import random

        domain_label = signal.domain.replace("_", " ").title()

        templates: list[str] = [] # noqa
        if signal.probe_type == ScalerAction.PROBE_DEEP:
            templates = ProbeInstructionBuilder._DEEP_TEMPLATES
        elif signal.probe_type == ScalerAction.PROBE_LATERAL:
            templates = ProbeInstructionBuilder._LATERAL_TEMPLATES
        elif signal.probe_type == ScalerAction.PROBE_VERIFY:
            templates = ProbeInstructionBuilder._VERIFY_TEMPLATES
        else:
            return ""

        chosen = random.choice(templates)
        return chosen.replace("{domain}", domain_label)

    @staticmethod
    def difficulty_hint_to_instruction(hint: str, level: str) -> str:
        """
        Translates difficulty_hint into a prompt instruction.
        Appended to system prompt alongside level.

        "low"  → ask a foundational question that tests basics of the level
        "mid"  → standard question for this level
        "high" → ask the hardest question appropriate for this level
        """
        level_label = level.capitalize()
        if hint == "low":
            return (
                f"Ask an {level_label}-level question on the easier end — "
                "foundational knowledge, not edge cases."
            )
        elif hint == "high":
            return (
                f"Ask a challenging {level_label}-level question — "
                "the kind that separates strong from weak candidates at this level."
            )
        else:
            return f"Ask a standard {level_label}-level question."


# ══════════════════════════════════════════════════════════════════════════════
# § SCALER DIAGNOSTICS
# Standalone diagnostic tools for debugging / Grafana / admin endpoints.
# ══════════════════════════════════════════════════════════════════════════════

class ScalerDiagnostics:
    """
    Diagnostic utilities. All methods are static and pure — no side effects.

    Usage:
        report = ScalerDiagnostics.belief_report(d_state.belief)
        print(ScalerDiagnostics.render_belief_bar(d_state.belief))
    """

    @staticmethod
    def belief_report(belief: BeliefState) -> dict[str, Any]:
        """Human-readable report of a BeliefState."""
        return {
            "distribution": {
                LEVELS[i]: round(belief.probs[i], 4)
                for i in range(N_LEVELS)
            },
            "MAP_level":    belief.map_level,
            "expected":     round(belief.expected_numeric, 4),
            "entropy":      round(belief.entropy(), 4),
            "is_certain":   belief.is_certain(),
        }

    @staticmethod
    def render_belief_bar(belief: BeliefState, width: int = 40) -> str:
        """
        ASCII bar chart of belief distribution for terminal/log output.

        Example:
          beginner:     ████░░░░░░░░░░░░░░░░░░░░░░░░░░  0.31
          intermediate: ████████████████████░░░░░░░░░░  0.52
          advanced:     █████░░░░░░░░░░░░░░░░░░░░░░░░░  0.17
        """
        lines = []
        max_label = max(len(L) for L in LEVELS)
        for i, level in enumerate(LEVELS):
            p    = belief.probs[i]
            fill = int(round(p * width))
            bar  = "█" * fill + "░" * (width - fill)
            lines.append(f"  {level:<{max_label}}: {bar}  {p:.3f}")
        lines.append(f"  entropy: {belief.entropy():.3f} / {math.log2(N_LEVELS):.3f}")
        return "\n".join(lines)

    @staticmethod
    def simulate_session(
        stated_level:  str,
        true_level:    str,
        n_turns:       int = 10,
        noise_sigma:   float = 0.12,
    ) -> list[dict]:
        """
        Dry-run simulation: given stated and true levels, simulate n_turns
        of scoring and show how the PerformanceScaler converges.

        Returns a list of per-turn state snapshots for plotting or testing.

        This is a SYNCHRONOUS simulation (no async) for use in tests/evals.
        """
        import random

        true_numeric   = LEVEL_NUMERIC.get(true_level, 0.5)
        doubt          = DoubtEngine.initial_doubt(stated_level)
        belief         = BayesianBeliefEngine.build_prior(stated_level, doubt)
        score_history  = []
        diff_history   = []
        current_diff   = belief.expected_numeric
        snapshots      = []
        total_scored   = 0

        for turn in range(n_turns):
            # Simulate score from true level + noise
            expected = BayesianBeliefEngine.expected_score(true_numeric, current_diff)
            score    = max(0.0, min(1.0, expected + random.gauss(0, noise_sigma)))
            score_history.append(score)
            diff_history.append(current_diff)
            total_scored += 1

            # Bayesian update
            updated = BayesianBeliefEngine.update(belief, score, current_diff)
            trust   = min(1.0, total_scored / DOUBT_EVIDENCE_HORIZON)
            prior   = BayesianBeliefEngine.build_prior(stated_level, doubt)
            blended_probs = [
                trust * updated.probs[i] + (1.0 - trust) * prior.probs[i]
                for i in range(N_LEVELS)
            ]
            total   = sum(blended_probs)
            belief  = BeliefState(probs=[p / total for p in blended_probs])

            # Update doubt
            mock_events = [
                ScoreEvent("sim", i, "python", score_history[i], diff_history[i], 20)
                for i in range(len(score_history))
            ]
            doubt = DoubtEngine.compute_doubt(stated_level, mock_events, total_scored)

            # Update difficulty
            zpd_action, _ = ZPDHunter.zpd_action(score_history, current_diff, turn + 1)
            step = 0.15
            if zpd_action == ScalerAction.ESCALATE:
                current_diff = min(1.0, current_diff + step)
            elif zpd_action == ScalerAction.DE_ESCALATE:
                current_diff = max(0.0, current_diff - step)

            traj, slope, r2 = TrajectoryAnalyzer.analyze(score_history)

            snapshots.append({
                "turn":            turn,
                "score":           round(score, 3),
                "difficulty":      round(current_diff, 3),
                "map_level":       belief.map_level,
                "entropy":         round(belief.entropy(), 3),
                "doubt":           round(doubt, 3),
                "trajectory":      traj.value,
                "zpd_action":      zpd_action.value,
                "belief_probs": {
                    LEVELS[i]: round(belief.probs[i], 3) for i in range(N_LEVELS)
                },
            })

        return snapshots

    @staticmethod
    def entropy_convergence_turns(
        stated_level: str,
        true_level:   str,
        target_entropy: float = 0.40,
        max_trials: int = 200,
        noise_sigma: float = 0.12,
    ) -> float:
        """
        Monte Carlo estimate of how many turns it takes for the scaler to
        converge to within target_entropy for a given stated→true level pair.

        Useful for calibrating DOUBT_EVIDENCE_HORIZON and ZPD thresholds.
        Returns the mean turns-to-convergence over max_trials runs.
        """
        import random # noqa
        convergence_turns: list[int] = []

        for _ in range(max_trials):
            snaps = ScalerDiagnostics.simulate_session(
                stated_level = stated_level,
                true_level   = true_level,
                n_turns      = 25,
                noise_sigma  = noise_sigma,
            )
            for snap in snaps:
                if snap["entropy"] <= target_entropy:
                    convergence_turns.append(snap["turn"])
                    break
            else:
                convergence_turns.append(25)   # did not converge

        return statistics.mean(convergence_turns) if convergence_turns else 25.0


# ── Module-level singleton ────────────────────────────────────────────────────

performance_scaler = PerformanceScaler()