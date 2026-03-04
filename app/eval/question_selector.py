"""
question_selector.py
══════════════════════════════════════════════════════════════════════════════

Joint optimizer over (concept_cluster × difficulty_band) space for
structured interview question selection.

THEORETICAL FOUNDATIONS
────────────────────────
This module is a synthesis of five research bodies that have never been
combined in a deployed interview or education AI system:

  1. Item Response Theory — 2-Parameter Logistic (IRT 2PL)
     ───────────────────────────────────────────────────────
     P(correct | θ, a, b) = σ(a · (θ − b))
     where θ = latent ability, a = discrimination, b = difficulty threshold.
     Fisher Information: I(θ; a, b) = a² · P · (1 − P)
     Maximised when θ = b — question matched to ability. Used here to
     quantify the signal yield of every (concept, difficulty) pair given
     the current belief distribution over true skill level.
     Source: Birnbaum (1968); Lord (1980) "Applications of IRT to Testing"

  2. Active Inference / Variational Free Energy (Friston, 2010)
     ────────────────────────────────────────────────────────────
     Instead of pure expected-information-gain (Bayesian active learning),
     the selector minimises a variational free energy that simultaneously
     maximises epistemic value (belief update) and pragmatic value (coverage
     progress, trajectory alignment). This is the correct unified objective
     when you have both signal-seeking and goal-directed motives.
     Objective: F(c, d) = −α·EIG − β·U(c) − γ·A(c,d) + δ·R(c) + λ·CL(c,d)
     Selection: argmin F ≡ argmax[α·EIG + β·U + γ·A − δ·R − λ·CL]
     Source: Friston et al. (2017) "Active Inference: A Process Theory"

  3. Cognitive Load Theory (Sweller, 1988; Paas et al., 2003)
     ──────────────────────────────────────────────────────────
     Three-component model: intrinsic (task complexity), extraneous (question
     framing novelty), germane (productive engagement). Optimal learning and
     discrimination happen in a "productive zone" [0.3, 0.7] on a normalised
     scale. Questions outside this zone either yield no signal (too easy)
     or cause candidate shutdown (too hard). The CL penalty steers the
     selector away from these dead zones.
     Source: Sweller (1988) Cognitive Science; Van Merriënboer & Sweller (2005)

  4. Thompson Sampling with Fisher Information Priors (MAB)
     ────────────────────────────────────────────────────────
     Each concept cluster is a bandit arm. Beta(α_c, β_c) priors are
     initialised from the concept's IRT discrimination parameter: high-α_c
     for concepts that historically discriminate well. Thompson samples
     introduce stochastic exploration, preventing the deterministic selector
     from collapsing onto the same concept every turn when scores are close.
     Online update: high-IG realised → increment α_c; low-IG → increment β_c.
     Source: Thompson (1933); Russo et al. (2018) "A Tutorial on Thompson Sampling"

  5. Concept Dependency Graph with Bayesian Blocking
     ─────────────────────────────────────────────────
     Domain concepts form a directed acyclic graph of prerequisite relations.
     When a candidate fails a root concept, successor concepts are partially
     blocked: P(blocked | c) = Π_p P(failed | p) over prerequisites p.
     Topological priority boosts foundational concepts when prerequisite
     status is uncertain, preventing wasted signal on unreachable knowledge.
     Related to: BKT (Corbett & Anderson, 1994); DKT (Piech et al., 2015)

EPISTEMIC MOMENTUM — A Novel Meta-Signal
─────────────────────────────────────────
  Defined here for the first time. Momentum tracks the rate of belief
  entropy reduction across successive turns:

      ΔH_t = H(π_{t-1}) − H(π_t)          (entropy reduction per turn)
      velocity = EWM(|ΔH_t|, α=0.6)        (exponentially weighted average)

  High velocity → belief converging fast → pivot to coverage (EIG is cheap
  to harvest; breadth is now the bottleneck).
  Low velocity  → belief stagnant → maximise EIG (candidate's true level
  is genuinely ambiguous; signal quality trumps breadth).

  This makes the EQS self-aware: it knows when it knows enough about skill
  level, and shifts resources accordingly.

ONLINE WEIGHT CALIBRATION
──────────────────────────
  Objective weights (α, β, γ, δ, λ) start from theoretically motivated
  defaults but adapt within the session via projected online gradient ascent
  on the correlation between composite score and eval_engine outcome:

      e_t = y_t − composite_score_t          (prediction error)
      w_i ← w_i + lr · e_t · component_i_t  (gradient step)
      w   ← project_simplex(w)               (enforce w ≥ 0, Σw = 1)

  After 6+ turns with eval data, weights reflect this candidate's actual
  response patterns rather than average population assumptions.

INTEGRATION
───────────
  Instantiate once per session in SessionLifecycleManager.open_session():

      eqs = EpistemicQuestionSelector.create(
          session_id=session_id,
          redis=redis_client,
          stated_level=doc.candidate.level,
      )

  Replace the two-patch call in LLMInputBuilder.build():

      spec = await eqs.select(
          d_state      = d_state,       # DomainScalerState
          coverage_map = coverage_map,  # ConceptCoverageMap
          domain       = domain,
          q_index      = q_index,
          trajectory   = traj,
          score_history = d_state.score_history,
      )
      # spec.effective_level   → replaces apply_signal_to_llm_input level patch
      # spec.build_suffix()    → replaces to_suffix_instruction()
      # spec.action            → carries scaler action through

  After eval score arrives (fire-and-forget):

      asyncio.create_task(eqs.record_outcome(domain, q_index, spec, eval_score))

WIRE-IN GUARD: FF_EQS
──────────────────────
  Feature flag: EQS_ENABLED=true (default false until validated in staging).
  When disabled, EpistemicQuestionSelector.select() returns a FallbackSpec
  that is API-compatible but uses the existing two-patch code path internally.
  Zero graph-level changes needed to toggle.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random # noqa
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict # noqa
from dataclasses import fields as dc_fields
from enum import Enum
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING # noqa

import numpy as np
import redis.asyncio as aioredis

from app.common.shared import (
    CircuitBreaker,
    CircuitBreakerOpen,
    InMemoryLRU,
    get_tracer,
    make_counter,
    make_gauge, # noqa
    make_histogram,
    backoff_retry,  # noqa
)
from app.monitoring.observability import get_logger

if TYPE_CHECKING:
    # Avoid circular imports — these are type-check-only.
    # At runtime we accept Any and validate via duck-typing.
    from app.eval.performance_scaler import ( # noqa
        DomainScalerState,
        BeliefState,
        ScalerAction,
    )
    from app.eval.concept_tracker import ConceptCoverageMap # noqa

log = get_logger(__name__)
tracer = get_tracer(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# § 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

EQS_ENABLED:           bool  = os.getenv("EQS_ENABLED",          "false").lower() == "true"
EQS_MIN_Q_FOR_WEIGHT:  int   = int(os.getenv("EQS_MIN_Q_WEIGHT",  "6"))    # turns before weight adaptation fires
EQS_REDIS_TTL_S:       int   = int(os.getenv("SESSION_TTL_S",     "3600")) + int(os.getenv("SESSION_GRACE_S", "60"))
EQS_LRU_SIZE:          int   = int(os.getenv("EQS_LRU_SIZE",      "64"))
EQS_FALLBACK_ON_ERROR: bool  = os.getenv("EQS_FALLBACK_ON_ERROR", "true").lower() == "true"
EQS_WEIGHT_LR:         float = float(os.getenv("EQS_WEIGHT_LR",   "0.04"))
EQS_MOMENTUM_ALPHA:    float = float(os.getenv("EQS_MOMENTUM_ALPHA", "0.60"))

# IRT 2PL parameters (global; concept-specific overrides are in CONCEPT_DISCRIMINABILITY)
IRT_SLOPE:             float = 3.2    # matches likelihood slope in performance_scaler
IRT_BIAS:              float = 0.08   # matches likelihood bias in performance_scaler
IRT_SIGMA:             float = 0.16   # Gaussian noise around expected score

# Cognitive Load productive zone
CL_ZONE_LOW:           float = 0.28
CL_ZONE_HIGH:          float = 0.72
CL_ZONE_CENTER:        float = 0.50

# Epistemic momentum strategy thresholds
MOMENTUM_HIGH_VELOCITY: float = 0.08   # ΔH/turn: belief converging fast → coverage mode
MOMENTUM_LOW_VELOCITY:  float = 0.015  # ΔH/turn: belief stagnant → IG maximise mode

# Default objective weights (normalised; must sum to 1.0)
# Motivation: EIG is the primary signal source, coverage is secondary.
# Recency and CL are penalties, so their effective contribution is negative.
DEFAULT_EIG_WEIGHT:      float = 0.38
DEFAULT_COV_WEIGHT:      float = 0.28
DEFAULT_TRAJ_WEIGHT:     float = 0.18
DEFAULT_RECENCY_WEIGHT:  float = 0.08
DEFAULT_CL_WEIGHT:       float = 0.08

# Thompson Sampling hyper-parameters
TS_PRIOR_ALPHA_SCALE:   float = 2.0   # discrimination → prior α (high discrim → explore more)
TS_PRIOR_BETA:          float = 1.0
TS_IG_HIT_THRESHOLD:    float = 0.15  # normalised IG above this counts as an arm "hit"

# Recency decay
RECENCY_DECAY_PER_TURN: float = 0.30  # fraction of penalty removed per turn

# Prefix for all Redis keys in this module
_EQS_PREFIX = "eqs:v1:"


# ══════════════════════════════════════════════════════════════════════════════
# § 2. CONCEPT KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
#
# CONCEPT_PREREQUISITES  — DAG edges: concept → list of prerequisite concept keys
# CONCEPT_DISCRIMINABILITY — IRT 'a' parameter per concept key
#   Range [0.5, 2.8]: low=foundational (everyone knows), high=discriminative
# CONCEPT_DIFFICULTY_CENTER — IRT 'b': ability level where P(correct)=0.5
#   Maps to the continuous ability scale: 0=beginner, 0.5=intermediate, 1.0=advanced
#
# These encode expert knowledge about concept ordering and signal yield.
# They are intentionally domain-specific — "GIL" questions discriminate
# differently than "variables" questions.
# ──────────────────────────────────────────────────────────────────────────────

CONCEPT_PREREQUISITES: dict[str, dict[str, list[str]]] = {
    "python": {
        "variables_types":    [],
        "functions":          ["variables_types"],
        "oop_basics":         ["functions"],
        "closures":           ["functions"],
        "decorators":         ["closures", "oop_basics"],
        "generators":         ["functions"],
        "async_await":        ["generators"],
        "context_managers":   ["oop_basics"],
        "metaclasses":        ["oop_basics", "decorators"],
        "descriptors":        ["oop_basics"],
        "memory_management":  ["oop_basics"],
        "gil":                ["memory_management", "concurrency"],
        "concurrency":        ["async_await", "generators"],
        "type_system":        ["oop_basics", "functions"],
        "testing":            ["functions", "oop_basics"],
        "data_model":         ["oop_basics", "descriptors"],
        "packaging":          ["functions"],
    },
    "javascript": {
        "variables_scope":    [],
        "functions_closures": ["variables_scope"],
        "prototypes":         ["functions_closures"],
        "classes":            ["prototypes"],
        "promises":           ["functions_closures"],
        "async_await":        ["promises"],
        "event_loop":         ["promises", "async_await"],
        "dom_apis":           ["functions_closures"],
        "modules":            ["functions_closures"],
        "generators":         ["functions_closures"],
        "proxy_reflect":      ["classes", "prototypes"],
        "performance":        ["event_loop", "async_await"],
        "testing":            ["functions_closures", "modules"],
    },
    "java": {
        "oop_basics":         [],
        "inheritance":        ["oop_basics"],
        "interfaces":         ["oop_basics"],
        "generics":           ["oop_basics"],
        "collections":        ["generics"],
        "streams":            ["collections"],
        "concurrency":        ["oop_basics"],
        "locks_sync":         ["concurrency"],
        "jvm_internals":      ["concurrency", "collections"],
        "gc":                 ["jvm_internals"],
        "spring_ioc":         ["interfaces", "generics"],
        "reactive":           ["streams", "concurrency"],
        "testing":            ["oop_basics", "interfaces"],
    },
    "dsa": {
        "arrays":             [],
        "strings":            ["arrays"],
        "hash_tables":        ["arrays"],
        "linked_lists":       ["arrays"],
        "stacks_queues":      ["linked_lists", "arrays"],
        "sorting":            ["arrays"],
        "binary_search":      ["sorting", "arrays"],
        "recursion":          ["arrays"],
        "trees":              ["recursion", "linked_lists"],
        "heaps":              ["trees"],
        "graphs":             ["trees", "recursion"],
        "dynamic_programming":["recursion", "sorting"],
        "greedy":             ["sorting", "recursion"],
        "tries":              ["trees", "hash_tables"],
        "segment_trees":      ["trees", "binary_search"],
        "advanced_graphs":    ["graphs", "dynamic_programming"],
        "network_flow":       ["advanced_graphs"],
    },
    "system_design": {
        "basic_networking":   [],
        "http_rest":          ["basic_networking"],
        "databases_basics":   [],
        "sql_nosql":          ["databases_basics"],
        "caching":            ["databases_basics", "http_rest"],
        "load_balancing":     ["http_rest"],
        "microservices":      ["http_rest", "databases_basics"],
        "message_queues":     ["microservices"],
        "consistency":        ["databases_basics", "caching"],
        "cap_theorem":        ["consistency"],
        "sharding":           ["databases_basics", "cap_theorem"],
        "distributed_tracing":["microservices"],
        "cdn":                ["caching", "http_rest"],
        "rate_limiting":      ["microservices", "caching"],
        "event_sourcing":     ["message_queues", "consistency"],
        "consensus":          ["cap_theorem", "consistency"],
    },
    "databases": {
        "sql_basics":         [],
        "joins":              ["sql_basics"],
        "indexes":            ["sql_basics"],
        "transactions":       ["sql_basics"],
        "acid":               ["transactions"],
        "query_optimisation": ["indexes", "joins"],
        "normalization":      ["sql_basics"],
        "stored_procedures":  ["sql_basics"],
        "nosql_concepts":     ["sql_basics"],
        "document_stores":    ["nosql_concepts"],
        "column_stores":      ["nosql_concepts"],
        "graph_databases":    ["nosql_concepts"],
        "replication":        ["transactions", "acid"],
        "partitioning":       ["indexes", "replication"],
        "mvcc":               ["transactions", "acid"],
    },
    "os_concepts": {
        "processes":          [],
        "threads":            ["processes"],
        "scheduling":         ["processes", "threads"],
        "memory_basics":      ["processes"],
        "virtual_memory":     ["memory_basics"],
        "paging_segmentation":["virtual_memory"],
        "synchronization":    ["threads"],
        "deadlocks":          ["synchronization"],
        "file_systems":       ["memory_basics"],
        "io_systems":         ["file_systems"],
        "ipc":                ["processes", "synchronization"],
        "signals":            ["processes", "ipc"],
        "system_calls":       ["processes", "threads"],
        "kernel_user_space":  ["system_calls", "virtual_memory"],
    },
    "cpp": {
        "basics":             [],
        "pointers_refs":      ["basics"],
        "memory_management":  ["pointers_refs"],
        "raii":               ["memory_management"],
        "oop_cpp":            ["pointers_refs"],
        "templates":          ["oop_cpp"],
        "stl":                ["templates", "oop_cpp"],
        "move_semantics":     ["raii", "oop_cpp"],
        "smart_pointers":     ["raii", "move_semantics"],
        "concurrency_cpp":    ["memory_management"],
        "lambdas":            ["oop_cpp"],
        "concepts_cpp20":     ["templates", "lambdas"],
        "coroutines_cpp20":   ["concurrency_cpp", "lambdas"],
    },
    "ml": {
        "statistics_basics":  [],
        "linear_algebra":     [],
        "linear_regression":  ["statistics_basics", "linear_algebra"],
        "gradient_descent":   ["linear_regression"],
        "logistic_regression":["gradient_descent"],
        "decision_trees":     ["statistics_basics"],
        "ensemble_methods":   ["decision_trees"],
        "neural_nets_basics": ["gradient_descent", "linear_algebra"],
        "backpropagation":    ["neural_nets_basics"],
        "cnns":               ["backpropagation"],
        "rnns_lstms":         ["backpropagation"],
        "transformers":       ["rnns_lstms", "linear_algebra"],
        "regularization":     ["gradient_descent", "logistic_regression"],
        "evaluation_metrics": ["statistics_basics"],
        "feature_engineering":["statistics_basics"],
    },
}

# IRT 2PL discrimination parameter 'a' per concept.
# Higher value → concept sharply separates skill levels.
# Lower value  → concept answered correctly by wide range of abilities.
CONCEPT_DISCRIMINABILITY: dict[str, dict[str, float]] = {
    "python": {
        "variables_types": 0.7, "functions": 1.0, "oop_basics": 1.2,
        "closures": 1.8, "decorators": 2.0, "generators": 1.7,
        "async_await": 2.1, "context_managers": 1.4, "metaclasses": 2.4,
        "descriptors": 2.2, "memory_management": 1.8, "gil": 2.3,
        "concurrency": 2.0, "type_system": 1.6, "testing": 1.3,
        "data_model": 2.1, "packaging": 0.9,
    },
    "javascript": {
        "variables_scope": 0.8, "functions_closures": 1.6, "prototypes": 2.0,
        "classes": 1.4, "promises": 1.7, "async_await": 1.8,
        "event_loop": 2.2, "dom_apis": 1.1, "modules": 1.3,
        "generators": 1.9, "proxy_reflect": 2.4, "performance": 2.0,
        "testing": 1.2,
    },
    "dsa": {
        "arrays": 0.8, "strings": 0.9, "hash_tables": 1.4,
        "linked_lists": 1.2, "stacks_queues": 1.3, "sorting": 1.5,
        "binary_search": 1.6, "recursion": 1.7, "trees": 1.8,
        "heaps": 2.0, "graphs": 2.1, "dynamic_programming": 2.4,
        "greedy": 2.0, "tries": 2.2, "segment_trees": 2.5,
        "advanced_graphs": 2.6, "network_flow": 2.7,
    },
    "system_design": {
        "basic_networking": 0.8, "http_rest": 1.0, "databases_basics": 0.9,
        "sql_nosql": 1.3, "caching": 1.6, "load_balancing": 1.5,
        "microservices": 1.8, "message_queues": 1.9, "consistency": 2.2,
        "cap_theorem": 2.1, "sharding": 2.3, "distributed_tracing": 1.7,
        "cdn": 1.4, "rate_limiting": 1.6, "event_sourcing": 2.3,
        "consensus": 2.6,
    },
    "databases": {
        "sql_basics": 0.8, "joins": 1.2, "indexes": 1.6,
        "transactions": 1.7, "acid": 1.8, "query_optimisation": 2.1,
        "normalization": 1.5, "stored_procedures": 1.3, "nosql_concepts": 1.4,
        "document_stores": 1.5, "column_stores": 1.9, "graph_databases": 2.0,
        "replication": 2.1, "partitioning": 2.2, "mvcc": 2.5,
    },
    "os_concepts": {
        "processes": 0.9, "threads": 1.3, "scheduling": 1.8,
        "memory_basics": 1.1, "virtual_memory": 1.8, "paging_segmentation": 2.0,
        "synchronization": 2.0, "deadlocks": 2.1, "file_systems": 1.5,
        "io_systems": 1.7, "ipc": 1.9, "signals": 1.8,
        "system_calls": 1.6, "kernel_user_space": 2.3,
    },
    "cpp": {
        "basics": 0.7, "pointers_refs": 1.4, "memory_management": 1.9,
        "raii": 2.0, "oop_cpp": 1.5, "templates": 2.2,
        "stl": 1.7, "move_semantics": 2.3, "smart_pointers": 2.0,
        "concurrency_cpp": 2.2, "lambdas": 1.8, "concepts_cpp20": 2.5,
        "coroutines_cpp20": 2.6,
    },
    "ml": {
        "statistics_basics": 0.9, "linear_algebra": 0.9,
        "linear_regression": 1.2, "gradient_descent": 1.5,
        "logistic_regression": 1.4, "decision_trees": 1.3,
        "ensemble_methods": 1.8, "neural_nets_basics": 1.7,
        "backpropagation": 2.0, "cnns": 2.1, "rnns_lstms": 2.2,
        "transformers": 2.4, "regularization": 1.6,
        "evaluation_metrics": 1.4, "feature_engineering": 1.5,
    },
}

# IRT 'b': ability level at which P(correct) = 0.5.
# 0.0 = beginner boundary, 0.5 = intermediate, 1.0 = advanced.
CONCEPT_DIFFICULTY_CENTER: dict[str, dict[str, float]] = {
    "python": {
        "variables_types": 0.10, "functions": 0.22, "oop_basics": 0.32,
        "closures": 0.52, "decorators": 0.58, "generators": 0.55,
        "async_await": 0.68, "context_managers": 0.45, "metaclasses": 0.85,
        "descriptors": 0.78, "memory_management": 0.62, "gil": 0.78,
        "concurrency": 0.72, "type_system": 0.58, "testing": 0.38,
        "data_model": 0.75, "packaging": 0.28,
    },
    "dsa": {
        "arrays": 0.12, "strings": 0.15, "hash_tables": 0.38,
        "linked_lists": 0.30, "stacks_queues": 0.32, "sorting": 0.42,
        "binary_search": 0.48, "recursion": 0.50, "trees": 0.55,
        "heaps": 0.65, "graphs": 0.68, "dynamic_programming": 0.80,
        "greedy": 0.70, "tries": 0.75, "segment_trees": 0.88,
        "advanced_graphs": 0.90, "network_flow": 0.95,
    },
    "system_design": {
        "basic_networking": 0.15, "http_rest": 0.22, "databases_basics": 0.18,
        "sql_nosql": 0.40, "caching": 0.48, "load_balancing": 0.45,
        "microservices": 0.58, "message_queues": 0.62, "consistency": 0.72,
        "cap_theorem": 0.70, "sharding": 0.75, "distributed_tracing": 0.60,
        "cdn": 0.42, "rate_limiting": 0.52, "event_sourcing": 0.78,
        "consensus": 0.90,
    },
}

_DEFAULT_DISCRIMINABILITY: float = 1.5
_DEFAULT_DIFFICULTY_CENTER: float = 0.5

# Concept type classification: used by CognitiveLoadEstimator for germane load.
# "foundational" → appropriate for all levels, especially correction
# "discriminative" → optimal near ability match
# "frontier" → suitable for high-ability confirmation
ConceptType = Literal["foundational", "discriminative", "frontier"]

def _concept_type(discriminability: float) -> ConceptType:
    if discriminability < 1.2:
        return "foundational"
    if discriminability < 2.0:
        return "discriminative"
    return "frontier"


# ══════════════════════════════════════════════════════════════════════════════
# § 3. CORE DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class SelectionStrategy(str, Enum):
    JOINT_OPTIMAL   = "joint_optimal"    # full EQS path: all engines active
    COVERAGE_PIVOT  = "coverage_pivot"   # high momentum: prioritise breadth
    IG_MAXIMISE     = "ig_maximise"      # stagnant belief: maximise EIG
    FALLBACK        = "fallback"         # EQS disabled or error path


@dataclass
class EQSWeights:
    """
    Learnable objective weights for the free energy formulation.
    All weights are non-negative and sum to 1.0 (enforced by project_simplex).
    Semantics: higher weight → this component matters more in selection.
    Note: recency and cognitive_load are penalty terms (subtracted from score).
    """
    eig:           float = DEFAULT_EIG_WEIGHT
    coverage:      float = DEFAULT_COV_WEIGHT
    trajectory:    float = DEFAULT_TRAJ_WEIGHT
    recency:       float = DEFAULT_RECENCY_WEIGHT
    cognitive_load:float = DEFAULT_CL_WEIGHT

    def as_vector(self) -> list[float]:
        return [self.eig, self.coverage, self.trajectory, self.recency, self.cognitive_load]

    @classmethod
    def from_vector(cls, v: list[float]) -> "EQSWeights":
        if len(v) != 5:
            raise ValueError(f"EQSWeights vector must have 5 elements, got {len(v)}")
        return cls(eig=v[0], coverage=v[1], trajectory=v[2], recency=v[3], cognitive_load=v[4])

    def total(self) -> float:
        return sum(self.as_vector())

    def is_valid(self) -> bool:
        v = self.as_vector()
        return all(w >= 0.0 for w in v) and abs(sum(v) - 1.0) < 1e-6

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}

    @classmethod
    def default(cls) -> "EQSWeights":
        return cls()


@dataclass
class SelectionCandidate:
    """
    Internal scoring record for one (concept, difficulty_band) pair.
    Fully scored before QuestionSpec is emitted.
    """
    concept:            str
    concept_label:      str
    difficulty_band:    str           # "low" | "mid" | "high"
    difficulty:         float         # continuous [0, 1]
    effective_level:    str           # "beginner" | "intermediate" | "advanced"
    discrimination:     float         # IRT 'a'
    difficulty_center:  float         # IRT 'b'

    # Scored components — all normalised to [0, 1] before weighting
    eig:                 float = 0.0
    coverage_value:      float = 0.0
    trajectory_alignment:float = 0.0
    recency_penalty:     float = 0.0
    cognitive_load_penalty: float = 0.0
    thompson_bonus:      float = 0.0  # MAB stochastic exploration bonus
    dependency_weight:   float = 1.0  # reduced if prerequisites are uncertain/failed

    # Final
    composite_score:     float = 0.0  # weighted sum using EQSWeights

    def __repr__(self) -> str:
        return (
            f"SC(concept={self.concept!r}, band={self.difficulty_band}, "
            f"score={self.composite_score:.3f}, eig={self.eig:.3f}, "
            f"cov={self.coverage_value:.3f})"
        )


@dataclass
class QuestionSpec:
    """
    Joint output of EpistemicQuestionSelector.select().

    Replaces the two-patch approach (apply_signal_to_llm_input + to_suffix_instruction)
    with a single coherent object covering both concept and difficulty dimensions.
    Callers use build_suffix() to get the LLM system prompt suffix.
    """
    session_id:           str
    domain:               str
    turn_index:           int

    # Primary selection outputs
    concept_cluster:      str
    concept_label:        str
    concept_focus_keywords: list[str]
    difficulty_band:      str          # "low" | "mid" | "high"
    difficulty:           float        # numeric [0, 1]
    effective_level:      str          # used to overwrite LLMInterviewInput.level
    action:               Any          # ScalerAction (passed through from DomainScalerState)

    # Scoring breakdown (for observability and weight adaptation)
    eig:                  float
    coverage_value:       float
    trajectory_alignment: float
    recency_penalty:      float
    cognitive_load_penalty: float
    thompson_bonus:       float
    dependency_weight:    float
    composite_score:      float

    # Meta
    strategy:             SelectionStrategy
    reasoning:            str
    latency_ms:           float
    n_candidates:         int
    weights_used:         EQSWeights

    def build_suffix(self) -> str:
        """
        Construct the complete LLM system prompt suffix from this spec.
        Emits a natural language instruction injected after the base system prompt.
        """
        parts: list[str] = []
        if self.concept_focus_keywords:
            focus = ", ".join(self.concept_focus_keywords[:4])
            parts.append(f"Concept focus: {self.concept_label}. "
                         f"Key angles to probe: {focus}.")
        if self.difficulty_band == "high":
            parts.append("Ask at the challenging end of the difficulty spectrum for this level.")
        elif self.difficulty_band == "low":
            parts.append("Use a more accessible framing — build candidate confidence first.")
        return "  ".join(parts) if parts else ""

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "concept":       self.concept_cluster,
            "level":         self.effective_level,
            "band":          self.difficulty_band,
            "diff":          round(self.difficulty, 3),
            "score":         round(self.composite_score, 4),
            "eig":           round(self.eig, 4),
            "cov":           round(self.coverage_value, 3),
            "traj":          round(self.trajectory_alignment, 3),
            "recency":       round(self.recency_penalty, 3),
            "cl":            round(self.cognitive_load_penalty, 3),
            "strategy":      self.strategy.value,
            "n_cands":       self.n_candidates,
            "latency_ms":    round(self.latency_ms, 2),
            "weights":       self.weights_used.to_dict(),
        }

    @classmethod
    def fallback(
        cls,
        session_id: str,
        domain: str,
        turn_index: int,
        effective_level: str,
        action: Any,
        latency_ms: float = 0.0,
    ) -> "QuestionSpec":
        """Safe fallback spec when EQS is disabled or errors out."""
        return cls(
            session_id=session_id, domain=domain, turn_index=turn_index,
            concept_cluster="", concept_label="", concept_focus_keywords=[],
            difficulty_band="mid", difficulty=0.5, effective_level=effective_level,
            action=action, eig=0.0, coverage_value=0.0, trajectory_alignment=0.0,
            recency_penalty=0.0, cognitive_load_penalty=0.0, thompson_bonus=0.0,
            dependency_weight=1.0, composite_score=0.0,
            strategy=SelectionStrategy.FALLBACK, reasoning="fallback: EQS disabled or error",
            latency_ms=latency_ms, n_candidates=0, weights_used=EQSWeights.default(),
        )


@dataclass
class EQSRecord:
    """Per-turn prediction record used by EQSWeightAdapter for online calibration."""
    turn_index:          int
    domain:              str
    spec_composite:      float        # composite score emitted at selection time
    eig_component:       float
    coverage_component:  float
    trajectory_component:float
    recency_component:   float
    cl_component:        float
    weights_snapshot:    list[float]  # weights used at selection time
    eval_score:          float = -1.0 # -1.0 = eval not yet received
    ts:                  float = field(default_factory=time.monotonic)

    def has_outcome(self) -> bool:
        return self.eval_score >= 0.0

    def prediction_error(self) -> float | None:
        """Signed error: positive → we underestimated the concept value."""
        if not self.has_outcome():
            return None
        return self.eval_score - self.spec_composite


@dataclass
class ThompsonArmState:
    """Beta distribution state for one concept cluster arm."""
    concept:   str
    alpha:     float   # pseudo-successes (high IG realised)
    beta:      float   # pseudo-failures  (low IG realised)

    def sample(self) -> float:
        """Draw Thompson sample from Beta(alpha, beta)."""
        return float(np.random.beta(max(self.alpha, 0.01), max(self.beta, 0.01)))

    def update(self, was_informative: bool, magnitude: float = 1.0) -> None:
        """Update arm posterior from observed outcome."""
        if was_informative:
            self.alpha += magnitude
        else:
            self.beta  += magnitude

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class EQSSessionState:
    """
    All mutable EQS state for a single session.
    Persisted to Redis between turns; LRU fallback for Redis outages.
    """
    session_id:        str
    weights:           EQSWeights
    arm_states:        dict[str, dict[str, ThompsonArmState]]  # domain → concept → arm
    entropy_history:   list[tuple[int, float]]   # [(turn_index, H)]
    records:           list[EQSRecord]
    recent_concepts:   dict[str, list[tuple[int, str]]]  # domain → [(q_index, concept)]
    momentum_velocity: float   = 0.0
    total_turns:       int     = 0
    created_at:        float   = field(default_factory=time.monotonic)
    updated_at:        float   = field(default_factory=time.monotonic)

    def get_arm(self, domain: str, concept: str) -> ThompsonArmState:
        if domain not in self.arm_states:
            self.arm_states[domain] = {}
        if concept not in self.arm_states[domain]:
            disc = (CONCEPT_DISCRIMINABILITY
                    .get(domain, {})
                    .get(concept, _DEFAULT_DISCRIMINABILITY))
            self.arm_states[domain][concept] = ThompsonArmState(
                concept=concept,
                alpha=TS_PRIOR_ALPHA_SCALE * disc,
                beta=TS_PRIOR_BETA,
            )
        return self.arm_states[domain][concept]

    def push_recent(self, domain: str, q_index: int, concept: str) -> None:
        if domain not in self.recent_concepts:
            self.recent_concepts[domain] = []
        self.recent_concepts[domain].append((q_index, concept))
        # Keep only last 10 per domain
        self.recent_concepts[domain] = self.recent_concepts[domain][-10:]

    def recency_turns_ago(self, domain: str, concept: str, current_q: int) -> int | None:
        """Return how many turns ago this concept was asked, or None if never."""
        history = self.recent_concepts.get(domain, [])
        for q_idx, c in reversed(history):
            if c == concept:
                return current_q - q_idx
        return None


# ══════════════════════════════════════════════════════════════════════════════
# § 4. PURE MATHEMATICAL FUNCTIONS (standalone, fully testable)
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid. Clamps extreme inputs to avoid overflow."""
    x = max(-50.0, min(50.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def irt_2pl(ability: float, difficulty: float, discrimination: float) -> float:
    """
    IRT 2-Parameter Logistic model.
    P(correct | θ=ability, b=difficulty, a=discrimination) = σ(a·(θ − b))

    Args:
        ability:        Latent trait θ ∈ [0, 1].
        difficulty:     Item difficulty b ∈ [0, 1].
        discrimination: Item discrimination a ∈ [0.5, 3.0].

    Returns:
        Probability of a correct/high-quality response in [0, 1].
    """
    return _sigmoid(discrimination * (ability - difficulty))


def fisher_information_2pl(ability: float, difficulty: float, discrimination: float) -> float:
    """
    Fisher Information for IRT 2PL at a given ability level.
    I(θ; a, b) = a² · P(θ) · (1 − P(θ))

    This is the Cramér-Rao lower bound on estimation variance — higher FI means
    the item tells us more about the true ability level at θ.
    Maximised when P = 0.5 (i.e., ability = difficulty in the symmetric case).

    Returns:
        Fisher Information ≥ 0.
    """
    p = irt_2pl(ability, difficulty, discrimination)
    return (discrimination ** 2) * p * (1.0 - p)


def expected_information_gain_discrete(
    prior_probs:     list[float],
    ability_levels:  list[float],
    difficulty:      float,
    discrimination:  float,
) -> float:
    """
    Exact EIG for a discrete ability distribution (3 levels in our case).

    EIG(item) = H(prior) − E_y[H(posterior|y)]
               = H(prior) − Σ_y P(y|prior) · H(π(·|y))

    Since y ∈ {correct, incorrect} and the prior is discrete, this is O(K)
    where K = number of ability levels (3). Cheap and exact.

    Args:
        prior_probs:    [P(beginner), P(intermediate), P(advanced)]
        ability_levels: numeric values for each level [0.0, 0.5, 1.0]
        difficulty:     IRT 'b' parameter
        discrimination: IRT 'a' parameter

    Returns:
        Expected information gain in nats (natural log entropy units).
    """
    k = len(prior_probs)
    if k != len(ability_levels):
        raise ValueError("prior_probs and ability_levels must have same length")

    # P(correct | θ_i) for each level
    p_correct = [irt_2pl(theta, difficulty, discrimination) for theta in ability_levels]

    # Marginal probability of correct response under prior
    p_correct_marginal = sum(prior_probs[i] * p_correct[i] for i in range(k))
    p_wrong_marginal   = 1.0 - p_correct_marginal

    # Posterior given correct: π_i × P(correct|θ_i) / P(correct)
    def posterior(p_response_given_theta: list[float], marginal: float) -> list[float]:
        if marginal < 1e-12:
            return list(prior_probs)
        raw = [prior_probs[i] * p_response_given_theta[i] for i in range(k)]
        total = sum(raw)
        return [r / total for r in raw] if total > 1e-12 else list(prior_probs)

    post_correct = posterior(p_correct, p_correct_marginal)
    post_wrong   = posterior([1.0 - p for p in p_correct], p_wrong_marginal)

    h_prior    = entropy_bits(prior_probs)
    h_correct  = entropy_bits(post_correct)
    h_wrong    = entropy_bits(post_wrong)

    expected_h_posterior = p_correct_marginal * h_correct + p_wrong_marginal * h_wrong
    return max(0.0, h_prior - expected_h_posterior)


def entropy_bits(probs: list[float]) -> float:
    """Shannon entropy in nats. Safe for zero probabilities (0·log(0) = 0)."""
    h = 0.0
    for p in probs:
        if p > 1e-12:
            h -= p * math.log(p)
    return h


def project_simplex(v: list[float]) -> list[float]:
    """
    Euclidean projection onto the probability simplex {w: w≥0, Σw=1}.
    Algorithm: sort-and-threshold (Duchi et al., 2008). O(n log n).
    """
    n = len(v) # noqa
    u = sorted(v, reverse=True)
    cumsum = 0.0
    rho = 0
    for j, u_j in enumerate(u):
        cumsum += u_j
        if u_j - (cumsum - 1.0) / (j + 1) > 0:
            rho = j + 1
    theta = (sum(u[:rho]) - 1.0) / rho
    return [max(x - theta, 0.0) for x in v]


def normalise_to_unit(values: list[float]) -> list[float]:
    """Normalise a list to [0, 1] by max; returns zeros if all equal."""
    mx = max(values) if values else 0.0
    if mx < 1e-12:
        return [0.0] * len(values)
    return [v / mx for v in values]


# ══════════════════════════════════════════════════════════════════════════════
# § 5. CONCEPT DEPENDENCY GRAPH
# ══════════════════════════════════════════════════════════════════════════════

class ConceptDependencyGraph:
    """
    Directed acyclic graph of prerequisite relationships within each domain.

    Evidence propagation: when a candidate demonstrably fails or succeeds
    at a concept (inferred from eval scores), that evidence propagates
    through the dependency edges to update blocking probabilities for
    successors and prerequisite probabilities for ancestors.

    Topological Priority:
        Concepts closer to the root (no prerequisites) receive higher
        topological priority when the belief distribution is uncertain.
        This prevents the selector from probing frontier concepts before
        establishing whether the candidate has the prerequisite knowledge.

    Blocking Probability:
        P(blocked | c, evidence) = Π_{p ∈ prereqs(c)} P(failed | p, evidence)
        Approximated via a running map of estimated_failure_prob per concept.
    """

    def __init__(self) -> None:
        # Topological rank per (domain, concept): lower = more foundational
        self._topo_rank: dict[str, dict[str, int]] = {}
        # Maximum rank per domain (for normalisation)
        self._max_rank:  dict[str, int] = {}
        # Compiled adjacency: concept → list of successors
        self._successors: dict[str, dict[str, list[str]]] = {}
        self._build()

    def _build(self) -> None:
        for domain, prereqs in CONCEPT_PREREQUISITES.items():
            self._successors[domain] = defaultdict(list)
            for concept, prereq_list in prereqs.items():
                for p in prereq_list:
                    self._successors[domain][p].append(concept)

            # Kahn's BFS topological sort for rank assignment
            in_degree: dict[str, int] = defaultdict(int)
            for concept, prereq_list in prereqs.items():
                for _ in prereq_list:
                    in_degree[concept] += 1

            rank: dict[str, int] = {}
            queue = deque([c for c in prereqs if in_degree[c] == 0])
            current_rank = 0
            while queue:
                next_queue: deque[str] = deque()
                for concept in list(queue):
                    rank[concept] = current_rank
                    for suc in self._successors[domain].get(concept, []):
                        in_degree[suc] -= 1
                        if in_degree[suc] == 0:
                            next_queue.append(suc)
                queue = next_queue
                current_rank += 1

            self._topo_rank[domain] = rank
            self._max_rank[domain] = max(rank.values(), default=0) if rank else 0

    def topological_priority(self, domain: str, concept: str) -> float:
        """
        Returns a priority in [0, 1] where 1.0 = most foundational (root).
        Used to boost foundational concepts when belief is uncertain.
        """
        rank = self._topo_rank.get(domain, {}).get(concept, 0)
        max_r = self._max_rank.get(domain, 1) or 1
        return 1.0 - (rank / max_r)

    def prerequisites(self, domain: str, concept: str) -> list[str]: # noqa
        return CONCEPT_PREREQUISITES.get(domain, {}).get(concept, [])

    def successors(self, domain: str, concept: str) -> list[str]:
        return self._successors.get(domain, {}).get(concept, [])

    def blocking_probability(
        self,
        domain: str, # noqa
        concept: str, # noqa
        failure_probs: dict[str, float],
    ) -> float:
        """
        P(blocked | concept, evidence) = Π_{p ∈ prereqs} P(failed | p)

        A concept is "blocked" if all prerequisites have failed. In practice
        we use a soft product: if any prerequisite has P(failed) > 0.5,
        the concept is partially blocked.

        Args:
            failure_probs: mapping of concept → estimated P(candidate failed this concept)

        Returns:
            Blocking probability in [0, 1]. 0 = not blocked, 1 = fully blocked.
        """
        prereqs = self.prerequisites(domain, concept)
        if not prereqs:
            return 0.0
        # Product of failure probabilities over prerequisites
        prob = 1.0
        for p in prereqs:
            prob *= failure_probs.get(p, 0.10)  # default: low prior of failure
        return prob

    def dependency_weight(
        self,
        domain: str,
        concept: str,
        failure_probs: dict[str, float],
        belief_entropy: float,
        max_entropy: float,
    ) -> float:
        """
        Combined weight accounting for topological priority and blocking.

        High entropy (uncertain belief) → boost foundational concepts.
        Low entropy (certain belief) → allow frontier concepts.
        High blocking probability → penalise the concept.

        Returns weight in [0, 1].
        """
        topo = self.topological_priority(domain, concept)
        block = self.blocking_probability(domain, concept, failure_probs)
        uncertainty_ratio = belief_entropy / (max_entropy + 1e-9)

        # When uncertain, foundational concepts are weighted higher
        topo_boost = topo * uncertainty_ratio + (1.0 - uncertainty_ratio)
        blocking_penalty = 1.0 - block

        return float(np.clip(topo_boost * blocking_penalty, 0.0, 1.0))

    def learnable_frontier(
        self,
        domain: str,
        covered: set[str],
        failure_probs: dict[str, float],
    ) -> list[str]:
        """
        Concepts that are not yet covered AND whose prerequisites are likely met.
        These are the "zone of proximal development" concepts in the domain graph.
        """
        all_concepts = set(CONCEPT_PREREQUISITES.get(domain, {}).keys())
        uncovered = all_concepts - covered
        frontier = []
        for c in uncovered:
            block = self.blocking_probability(domain, c, failure_probs)
            if block < 0.5:  # less than 50% chance of being blocked
                frontier.append(c)
        return frontier


# ══════════════════════════════════════════════════════════════════════════════
# § 6. FISHER INFORMATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_ABILITY_LEVELS_NUMERIC: list[float] = [0.0, 0.5, 1.0]   # beginner, intermediate, advanced
_LEVELS_STR: list[str] = ["beginner", "intermediate", "advanced"]
_DIFFICULTY_OFFSETS: dict[str, float] = {"low": -0.15, "mid": 0.0, "high": +0.15}
_MAX_ENTROPY: float = math.log(3)  # ln(3) ≈ 1.099 nats for 3 levels


class FisherInformationEngine:
    """
    IRT 2PL-based computation of expected information gain and optimal
    difficulty for the current belief distribution.

    All computations are O(K) where K = number of ability levels (3).
    No external calls. Pure math, no I/O, no state mutation.
    """

    @staticmethod
    def expected_ig(
        prior_probs:    list[float],
        difficulty:     float,
        discrimination: float,
    ) -> float:
        """
        Exact expected information gain for this item under the current prior.
        See expected_information_gain_discrete() for derivation.
        """
        return expected_information_gain_discrete(
            prior_probs=prior_probs,
            ability_levels=_ABILITY_LEVELS_NUMERIC,
            difficulty=difficulty,
            discrimination=discrimination,
        )

    @staticmethod
    def integrated_fisher(
        prior_probs:    list[float],
        difficulty:     float,
        discrimination: float,
    ) -> float:
        """
        Fisher Information integrated over the prior distribution.
        I_avg = Σ_i π_i · I(θ_i; a, b)

        This is a faster approximation of EIG that scales similarly but
        is O(K) with no log operations. Used as a tiebreaker.
        """
        return sum(
            prior_probs[i] * fisher_information_2pl(
                _ABILITY_LEVELS_NUMERIC[i], difficulty, discrimination
            )
            for i in range(len(prior_probs))
        )

    @staticmethod
    def optimal_difficulty(prior_probs: list[float]) -> float:
        """
        Optimal item difficulty (b*) for the given prior: the difficulty that
        maximises EIG. For a unimodal prior, this is close to the MAP ability.
        For flat priors, b* ≈ 0.5 (centre of ability scale).

        We compute via grid search over [0, 1] at 0.05 resolution.
        Returns optimal difficulty in [0, 1].
        """
        best_d, best_ig = 0.5, -1.0
        for d_int in range(0, 21):
            d = d_int / 20.0
            ig = expected_information_gain_discrete(
                prior_probs=prior_probs,
                ability_levels=_ABILITY_LEVELS_NUMERIC,
                difficulty=d,
                discrimination=1.5,  # use average discrimination for this search
            )
            if ig > best_ig:
                best_ig = ig
                best_d = d
        return best_d

    @staticmethod
    def difficulty_for_band(
        current_difficulty: float,
        band: str,
        concept_center: float,
    ) -> float:
        """
        Compute effective difficulty for a (difficulty_band, concept) pair.
        Blends current ability estimate with concept's natural difficulty center
        and the requested difficulty offset.
        """
        offset = _DIFFICULTY_OFFSETS.get(band, 0.0)
        # Blend: 60% current difficulty estimate, 40% concept's natural center
        blended = 0.6 * current_difficulty + 0.4 * concept_center
        return float(np.clip(blended + offset, 0.01, 0.99))

    @staticmethod
    def level_for_difficulty(difficulty: float) -> str:
        """Map continuous difficulty to level string."""
        if difficulty < 0.38:
            return "beginner"
        if difficulty < 0.72:
            return "intermediate"
        return "advanced"

    @staticmethod
    def normalise_ig(ig: float) -> float:
        """Normalise EIG to [0, 1] using maximum possible entropy as denominator."""
        return float(np.clip(ig / (_MAX_ENTROPY + 1e-9), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# § 7. COGNITIVE LOAD ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveLoadEstimator:
    """
    Three-component cognitive load model based on Sweller's CLT (1988).

    Components:
        Intrinsic    — determined by the mismatch between candidate's current
                       ability level and the question's difficulty. Large mismatch
                       = high intrinsic load (either too easy or too hard).
        Extraneous   — determined by conceptual novelty: how different this
                       concept cluster is from what the candidate has encountered.
                       High novelty → high working memory demand.
        Germane      — productive cognitive engagement. High when the concept
                       is in the candidate's ZPD and aligns with trajectory.
                       Subtracted from total load (it's beneficial).

    Productive Zone:
        CL_total ∈ [CL_ZONE_LOW, CL_ZONE_HIGH] = [0.28, 0.72]
        Below: no signal (too easy), Above: cognitive overload (shutdown).
        The CL penalty is the squared distance from the zone centre (0.50).

    All components are normalised to [0, 1] before combination.
    """

    # Intrinsic load: penalty for difficulty mismatch
    _INTRINSIC_SCALE: float = 2.0   # multiplier on |ability - difficulty|

    # Extraneous load: penalty for concept novelty vs recent history
    _EXTRANEOUS_RECENT_WINDOW: int = 4

    # Germane load weights by trajectory-concept type combination
    _GERMANE_TABLE: dict[tuple[str, str], float] = {
        # (trajectory, concept_type): germane load contribution
        ("rising",  "foundational"):  0.15,  # too easy when rising
        ("rising",  "discriminative"):0.60,  # ideal productive challenge
        ("rising",  "frontier"):      0.75,  # good for confirming high ability
        ("plateau", "foundational"):  0.30,  # mild correction value
        ("plateau", "discriminative"):0.65,  # break the plateau
        ("plateau", "frontier"):      0.55,  # might confirm plateau is real ceiling
        ("falling", "foundational"):  0.70,  # recovery: rebuild confidence
        ("falling", "discriminative"):0.45,  # risky when falling
        ("falling", "frontier"):      0.10,  # inappropriate when candidate is struggling
        ("unknown", "foundational"):  0.45,
        ("unknown", "discriminative"):0.55,
        ("unknown", "frontier"):      0.40,
    }

    def intrinsic_load(self, ability_estimate: float, question_difficulty: float) -> float:
        """
        Penalty for difficulty mismatch. U-shaped: penalty is low near the ability
        estimate and rises for both too-easy and too-hard questions.
        Normalised to [0, 1].
        """
        mismatch = abs(ability_estimate - question_difficulty)
        # Scale by 2 so full range [0, 1] maps to [0, 1] penalty
        raw = min(1.0, self._INTRINSIC_SCALE * mismatch)
        return raw

    def extraneous_load(
        self,
        concept: str,
        recent_concepts: list[str],
    ) -> float:
        """
        Penalty for concept novelty (working memory context switch cost).
        If this concept is identical to recent concepts → low extraneous load.
        Completely new concept cluster → higher extraneous load.
        """
        if not recent_concepts:
            return 0.3   # some base novelty cost for first concept
        recent = recent_concepts[-self._EXTRANEOUS_RECENT_WINDOW:]
        # Exact match in recent → very low extraneous
        if concept in recent:
            return 0.05
        # Check partial match via concept type similarity
        # (foundational → foundational is less novel than foundational → frontier)
        disc = CONCEPT_DISCRIMINABILITY.get("*", {}).get(concept, _DEFAULT_DISCRIMINABILITY)
        recent_discs = [_DEFAULT_DISCRIMINABILITY] * len(recent)  # approximate
        avg_recent_disc = sum(recent_discs) / len(recent_discs)
        disc_gap = abs(disc - avg_recent_disc) / 2.5   # normalise to ~[0, 1]
        return float(np.clip(0.15 + disc_gap * 0.5, 0.05, 0.75))

    def germane_load(
        self,
        concept: str, # noqa
        discrimination: float,
        trajectory: str,  # "rising" | "plateau" | "falling" | "unknown"
    ) -> float:
        """
        Productive engagement score — how much cognitive work yields signal.
        Higher is better; subtracted from total load (it's a benefit).
        """
        c_type = _concept_type(discrimination)
        return self._GERMANE_TABLE.get((trajectory, c_type), 0.4)

    def total_load( # noqa
        self,
        intrinsic: float,
        extraneous: float,
        germane: float,
        w_i: float = 0.45,
        w_e: float = 0.35,
        w_g: float = 0.20,
    ) -> float:
        """
        Total cognitive load. Germane load is beneficial (subtracted).
        Normalised to [0, 1].
        """
        raw = w_i * intrinsic + w_e * extraneous - w_g * germane
        return float(np.clip(raw, 0.0, 1.0))

    def load_penalty(self, total_cl: float) -> float: # noqa
        """
        Penalty for being outside the productive zone.
        Zero at CL_ZONE_CENTER (0.50). Quadratic distance from centre.
        Maximum penalty of 1.0 at the extremes (0 or 1).
        """
        distance = abs(total_cl - CL_ZONE_CENTER)
        # Quadratic penalty, normalised: max distance is 0.5 (zone centre to 0 or 1)
        normalised = (distance / 0.5) ** 2
        return float(np.clip(normalised, 0.0, 1.0))

    def compute_penalty(
        self,
        concept: str,
        difficulty: float,
        discrimination: float,
        ability_estimate: float,
        trajectory: str,
        recent_concept_keys: list[str],
    ) -> tuple[float, dict[str, float]]:
        """
        Full CL penalty for a (concept, difficulty) pair.
        Returns (penalty_in_0_1, component_breakdown).
        """
        intr  = self.intrinsic_load(ability_estimate, difficulty)
        extr  = self.extraneous_load(concept, recent_concept_keys)
        germ  = self.germane_load(concept, discrimination, trajectory)
        total = self.total_load(intr, extr, germ)
        penalty = self.load_penalty(total)
        return penalty, {"intrinsic": intr, "extraneous": extr, "germane": germ,
                         "total": total, "penalty": penalty}


# ══════════════════════════════════════════════════════════════════════════════
# § 8. EPISTEMIC MOMENTUM TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicMomentumTracker:
    """
    Tracks the rate of belief entropy reduction across successive turns.

    The core insight: a fast-converging belief distribution signals that
    the IRT likelihood is doing its job — the questions are discriminative
    and the candidate is not near any level boundary. At this point,
    continuing to maximise EIG is redundant; the system should pivot to
    coverage breadth.

    Conversely, a stagnant belief (entropy barely changing) means either:
      (a) the candidate is genuinely near a level boundary, or
      (b) the questions are not discriminative enough.
    The selector should respond by maximising EIG aggressively.

    Velocity = EWM(|ΔH_t|, α=EQS_MOMENTUM_ALPHA)
    Strategy:
        velocity > MOMENTUM_HIGH_VELOCITY → COVERAGE_PIVOT
        velocity < MOMENTUM_LOW_VELOCITY  → IG_MAXIMISE
        else                              → JOINT_OPTIMAL (balanced)
    """

    def __init__(self, alpha: float = EQS_MOMENTUM_ALPHA) -> None:
        self._alpha  = alpha
        self._velocity: float = 0.0
        self._last_entropy: float | None = None
        self._delta_history: list[float] = []  # |ΔH| per turn

    def record_entropy(self, turn_index: int, entropy: float) -> float: # noqa
        """
        Update momentum state from the current belief entropy.
        Returns the delta entropy (change from previous turn).
        """
        if self._last_entropy is None:
            self._last_entropy = entropy
            return 0.0

        delta = abs(self._last_entropy - entropy)
        self._last_entropy = entropy
        self._delta_history.append(delta)

        # Exponentially weighted moving average
        if len(self._delta_history) == 1:
            self._velocity = delta
        else:
            self._velocity = self._alpha * delta + (1.0 - self._alpha) * self._velocity

        return delta

    @property
    def velocity(self) -> float:
        """Current EWM velocity of entropy reduction."""
        return self._velocity

    @property
    def momentum(self) -> float:
        """Cumulative sum of recent deltas (last 5 turns)."""
        return sum(self._delta_history[-5:])

    def strategy(self) -> SelectionStrategy:
        """
        Derive the recommended selection strategy from current velocity.
        """
        if len(self._delta_history) < 2:
            return SelectionStrategy.JOINT_OPTIMAL   # not enough data yet
        if self._velocity > MOMENTUM_HIGH_VELOCITY:
            return SelectionStrategy.COVERAGE_PIVOT  # belief converging → breadth
        if self._velocity < MOMENTUM_LOW_VELOCITY:
            return SelectionStrategy.IG_MAXIMISE     # belief stagnant → signal
        return SelectionStrategy.JOINT_OPTIMAL

    def weight_modifier(self) -> dict[str, float]:
        """
        Returns additive adjustments to objective weights based on momentum.
        These are added to the base EQSWeights before scoring.
        """
        s = self.strategy()
        if s == SelectionStrategy.COVERAGE_PIVOT:
            return {"eig": -0.10, "coverage": +0.10, "trajectory": 0.0,
                    "recency": 0.0, "cognitive_load": 0.0}
        if s == SelectionStrategy.IG_MAXIMISE:
            return {"eig": +0.12, "coverage": -0.08, "trajectory": -0.04,
                    "recency": 0.0, "cognitive_load": 0.0}
        return {"eig": 0.0, "coverage": 0.0, "trajectory": 0.0,
                "recency": 0.0, "cognitive_load": 0.0}

    def to_dict(self) -> dict:
        return {
            "velocity":      round(self._velocity, 5),
            "momentum":      round(self.momentum, 4),
            "strategy":      self.strategy().value,
            "n_turns":       len(self._delta_history),
            "last_delta":    round(self._delta_history[-1], 5) if self._delta_history else 0.0,
        }

    @classmethod
    def from_entropy_history(
        cls,
        history: list[tuple[int, float]],
        alpha: float = EQS_MOMENTUM_ALPHA,
    ) -> "EpistemicMomentumTracker":
        """Reconstruct tracker from persisted entropy history."""
        tracker = cls(alpha=alpha)
        for turn_idx, entropy in sorted(history, key=lambda x: x[0]):
            tracker.record_entropy(turn_idx, entropy)
        return tracker


# ══════════════════════════════════════════════════════════════════════════════
# § 9. THOMPSON CONCEPT SAMPLER
# ══════════════════════════════════════════════════════════════════════════════

class ThompsonConceptSampler:
    """
    Multi-Armed Bandit with Thompson Sampling for stochastic concept selection.

    Each concept cluster is an arm with a Beta(α_c, β_c) distribution over
    its "value" (probability of yielding high information gain if selected).
    Thompson sampling draws a random value from each arm's posterior and
    selects the arm with the highest draw — balancing exploration (trying
    under-tested concepts) with exploitation (re-using high-value concepts).

    Fisher Information Prior Initialisation:
        α_c = TS_PRIOR_ALPHA_SCALE × discrimination_c
    This encodes the expert belief that high-discrimination concepts are
    more likely to be valuable arms, while still allowing the data to correct
    this prior over the course of the session.

    The Thompson bonus is a multiplicative modifier on the composite score,
    not a replacement. This preserves the free energy structure while adding
    controlled stochasticity to prevent deterministic collapse.
    """

    def __init__(self, session_state: EQSSessionState) -> None:
        self._state = session_state

    def sample_bonus(self, domain: str, concept: str) -> float:
        """
        Draw one Thompson sample for concept as a value in [0, 1].
        This bonus is added to the composite score with a small weight.
        """
        arm = self._state.get_arm(domain, concept)
        return arm.sample()

    def update(
        self,
        domain:         str,
        concept:        str,
        realised_ig:    float, # noqa
        ig_normalised:  float,
    ) -> None:
        """
        Update arm posterior from realised information gain.
        Called after a turn's eval result arrives.
        """
        arm = self._state.get_arm(domain, concept)
        was_informative = ig_normalised >= TS_IG_HIT_THRESHOLD
        # Scale update magnitude by realised IG for faster learning
        magnitude = 0.5 + 0.5 * min(ig_normalised, 1.0)
        arm.update(was_informative, magnitude=magnitude)
        log.debug(
            "eqs_thompson_update",
            domain=domain, concept=concept,
            informative=was_informative,
            alpha=round(arm.alpha, 2), beta=round(arm.beta, 2),
        )

    def arm_diagnostics(self, domain: str) -> dict[str, dict]:
        arms = self._state.arm_states.get(domain, {})
        return {
            c: {"alpha": round(a.alpha, 2), "beta": round(a.beta, 2),
                "mean": round(a.mean(), 3)}
            for c, a in arms.items()
        }


# ══════════════════════════════════════════════════════════════════════════════
# § 10. VARIATIONAL FREE ENERGY OBJECTIVE
# ══════════════════════════════════════════════════════════════════════════════

class VariationalFreeEnergyObjective:
    """
    The core scoring function for candidate (concept, difficulty) pairs.

    Formulation as variational free energy minimisation:
        F(c, d) = −α·EIG − β·U(c) − γ·A(c,d) + δ·R(c) + λ·CL(c,d)

    Equivalently, the composite score to MAXIMISE:
        score(c, d) = α·EIG + β·U(c) + γ·A(c,d) − δ·R(c) − λ·CL(c,d)

    Where:
        EIG = expected information gain from FisherInformationEngine
        U   = concept utility from coverage map + dependency graph
        A   = trajectory alignment score
        R   = recency penalty (subtracted)
        CL  = cognitive load penalty (subtracted)

    The Thompson bonus is added after scoring as a separate term to
    preserve the theoretical grounding of the free energy objective.

    All components are normalised to [0, 1] before weighting.
    """

    def trajectory_alignment( # noqa
        self,
        concept_type:       str,        # "foundational" | "discriminative" | "frontier"
        difficulty_band:    str,        # "low" | "mid" | "high"
        trajectory:         str,        # "rising" | "plateau" | "falling" | "unknown"
        score_history:      list[float],
    ) -> float:
        """
        Score for how well this (concept_type, difficulty_band) combination
        aligns with the candidate's current trajectory.

        RISING + frontier + high   → great: confirm the rise
        FALLING + foundational + low → great: recovery, rebuild signal
        RISING + foundational + low  → poor: wastes signal on solved territory
        PLATEAU + discriminative + high → good: break the plateau
        """
        alignment_table: dict[tuple[str, str, str], float] = {
            ("rising",  "frontier",      "high"):  0.90,
            ("rising",  "discriminative","high"):  0.80,
            ("rising",  "discriminative","mid"):   0.70,
            ("rising",  "foundational",  "mid"):   0.30,
            ("rising",  "foundational",  "low"):   0.10,
            ("plateau", "discriminative","high"):  0.85,
            ("plateau", "discriminative","mid"):   0.70,
            ("plateau", "frontier",      "high"):  0.75,
            ("plateau", "foundational",  "mid"):   0.40,
            ("plateau", "frontier",      "mid"):   0.55,
            ("falling", "foundational",  "low"):   0.90,
            ("falling", "foundational",  "mid"):   0.75,
            ("falling", "discriminative","low"):   0.60,
            ("falling", "discriminative","mid"):   0.40,
            ("falling", "frontier",      "high"):  0.05,
            ("unknown", "discriminative","mid"):   0.60,
            ("unknown", "foundational",  "mid"):   0.50,
            ("unknown", "frontier",      "high"):  0.45,
        }
        score = alignment_table.get((trajectory, concept_type, difficulty_band), 0.45)

        # Adjust for score variance: high variance → prefer verification probes
        if len(score_history) >= 3:
            variance = float(np.var(score_history[-4:]))
            if variance > 0.04 and difficulty_band == "high":
                score *= 0.85   # penalise high difficulty when variance is high

        return score

    def recency_score( # noqa
        self,
        concept:      str, # noqa
        turns_ago:    int | None,
        decay_factor: float = RECENCY_DECAY_PER_TURN,
    ) -> float:
        """
        Recency penalty in [0, 1]. Decays exponentially with turns since last ask.
        Zero if concept was never asked.
        """
        if turns_ago is None:
            return 0.0   # never asked, no penalty
        # Exponential decay: penalty = exp(-decay × turns_ago)
        return math.exp(-decay_factor * max(turns_ago, 1))

    def coverage_utility( # noqa
        self,
        concept:      str,
        coverage_map: Any,  # ConceptCoverageMap
        domain:       str, # noqa
    ) -> float:
        """
        Coverage value of selecting this concept cluster.
        Uncovered: 1.0, Shallow (1-2 hits): 0.5, Well-covered (≥3 hits): 0.1
        """
        covered: dict[str, int] = getattr(coverage_map, "covered", {})
        hits = covered.get(concept, 0)
        if hits == 0:
            return 1.0
        if hits <= 2:
            return 0.5
        return 0.1

    def score( # noqa
        self,
        eig:          float,
        coverage:     float,
        trajectory:   float,
        recency:      float,
        cl_penalty:   float,
        weights:      EQSWeights,
    ) -> float:
        """
        Compute composite score from normalised components and weights.

        score = α·EIG + β·coverage + γ·trajectory − δ·recency − λ·cl_penalty

        Recency and CL are penalties, so their weights are subtracted.
        Clamped to [0, 1] after weighting.
        """
        raw = (
            weights.eig      * eig      +
            weights.coverage * coverage +
            weights.trajectory * trajectory -
            weights.recency  * recency  -
            weights.cognitive_load * cl_penalty
        )
        return float(np.clip(raw, 0.0, 1.0))

    def explain( # noqa
        self,
        candidate:   SelectionCandidate,
        weights:     EQSWeights,
        trajectory:  str,
    ) -> str:
        """Human-readable explanation of this candidate's scoring."""
        return (
            f"concept={candidate.concept}({candidate.difficulty_band}) "
            f"score={candidate.composite_score:.3f} "
            f"eig={candidate.eig:.3f}×{weights.eig:.2f} "
            f"cov={candidate.coverage_value:.3f}×{weights.coverage:.2f} "
            f"traj={candidate.trajectory_alignment:.3f}[{trajectory}] "
            f"disc={candidate.discrimination:.2f} dep={candidate.dependency_weight:.2f} "
            f"ts_bonus={candidate.thompson_bonus:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# § 11. ONLINE WEIGHT ADAPTER
# ══════════════════════════════════════════════════════════════════════════════

class EQSWeightAdapter:
    """
    Online calibration of objective weights via projected gradient ascent.

    After enough eval scores arrive, the adapter adjusts weights to maximise
    the correlation between the composite score at selection time and the
    actual eval score that subsequently arrives. This is a form of
    self-supervised meta-learning: the EQS learns which signals actually
    predict good candidate responses for this specific session.

    Update rule (projected stochastic gradient ascent):
        e_t  = y_t − composite_score_t        (signed prediction error)
        w_i ← w_i + lr · e_t · component_i_t  (gradient step per weight)
        w   ← project_simplex(w)              (project onto Δ⁴)

    This is valid because the composite score is a linear function of the
    weights, making the gradient trivially computable from the stored
    component values.

    Activation: only fires after EQS_MIN_Q_FOR_WEIGHT matched pairs
    (selection + outcome) are available, to avoid adapting on noise.

    Domain-specific: weights are adapted per domain, since "behavioural"
    and "system_design" have structurally different optimal objectives.
    """

    def __init__(
        self,
        session_state: EQSSessionState,
        lr:            float = EQS_WEIGHT_LR,
        min_pairs:     int   = EQS_MIN_Q_FOR_WEIGHT,
    ) -> None:
        self._state    = session_state
        self._lr       = lr
        self._min_pairs = min_pairs
        # Domain-specific weight overrides (start from session global)
        self._domain_weights: dict[str, EQSWeights] = {}

    def record_prediction(
        self,
        turn_index:   int,
        domain:       str,
        composite:    float,
        eig:          float,
        coverage:     float,
        trajectory:   float,
        recency:      float,
        cl:           float,
        weights:      EQSWeights,
    ) -> None:
        """Store a prediction record for later outcome matching."""
        record = EQSRecord(
            turn_index=turn_index, domain=domain,
            spec_composite=composite,
            eig_component=eig,
            coverage_component=coverage,
            trajectory_component=trajectory,
            recency_component=recency,
            cl_component=cl,
            weights_snapshot=weights.as_vector(),
        )
        self._state.records.append(record)
        # Keep memory bounded: last 60 records
        if len(self._state.records) > 60:
            self._state.records = self._state.records[-60:]

    def record_outcome(self, turn_index: int, eval_score: float) -> None:
        """
        Match an eval score to its prediction record and trigger weight update.
        eval_score should be normalised to [0, 1] (divide raw 0-10 score by 10).
        """
        for record in reversed(self._state.records):
            if record.turn_index == turn_index and not record.has_outcome():
                record.eval_score = float(np.clip(eval_score, 0.0, 1.0))
                break

        matched = [r for r in self._state.records
                   if r.has_outcome() and r.domain != ""]
        if len(matched) >= self._min_pairs:
            self._adapt(matched[-self._min_pairs:])

    def _adapt(self, recent_matched: list[EQSRecord]) -> None:
        """
        Run one gradient step on the session-level weights using recent matched pairs.
        """
        w = list(self._state.weights.as_vector())  # [eig, cov, traj, recency, cl]

        for record in recent_matched:
            error = record.prediction_error()
            if error is None:
                continue
            components = [
                record.eig_component,
                record.coverage_component,
                record.trajectory_component,
                -record.recency_component,   # penalty: gradient is negated
                -record.cl_component,        # penalty: gradient is negated
            ]
            for i in range(len(w)):
                w[i] += self._lr * error * components[i]

        # Project back to simplex
        w = project_simplex([max(0.0, wi) for wi in w])
        self._state.weights = EQSWeights.from_vector(w)
        self._state.updated_at = time.monotonic()

        log.debug(
            "eqs_weights_adapted",
            new_weights=self._state.weights.to_dict(),
            n_pairs=len(recent_matched),
        )

    def current_weights(self, domain: str) -> EQSWeights:
        """
        Return the current weights for a domain.
        Domain-specific weights (if enough domain data exists) take precedence.
        """
        domain_records = [r for r in self._state.records
                          if r.domain == domain and r.has_outcome()]
        if len(domain_records) >= self._min_pairs:
            return self._domain_weights.get(domain, self._state.weights)
        return self._state.weights

    def convergence_diagnostics(self) -> dict:
        matched = [r for r in self._state.records if r.has_outcome()]
        if not matched:
            return {"n_matched": 0, "mean_error": None, "weight_stability": None}
        errors = [abs(r.prediction_error()) for r in matched if r.prediction_error() is not None]
        return {
            "n_matched":        len(matched),
            "mean_abs_error":   round(sum(errors) / len(errors), 4) if errors else None,
            "current_weights":  self._state.weights.to_dict(),
            "adapted":          len(matched) >= self._min_pairs,
        }


# ══════════════════════════════════════════════════════════════════════════════
# § 12. EQS STATE STORE
# ══════════════════════════════════════════════════════════════════════════════

class EQSStateStore:
    """
    Two-tier persistence for EQSSessionState: Redis primary, InMemoryLRU fallback.

    Redis key: eqs:v1:{session_id}  →  JSON blob
    TTL: SESSION_TTL_S + SESSION_GRACE_S (matches qa_controller.py convention)

    Serialisation: arm_states are compacted (α, β pairs only); entropy_history
    is stored as flat list of [turn_index, entropy] pairs. Records are kept
    in full but bounded to last 60 entries.
    """

    _PREFIX = _EQS_PREFIX

    def __init__(
        self,
        redis: aioredis.Redis | None = None,
        lru_size: int = EQS_LRU_SIZE,
    ) -> None:
        self._redis = redis
        self._lru   = InMemoryLRU(max_size=lru_size)
        self._cb    = CircuitBreaker(
            name="eqs_redis",
            failure_threshold=4,
            recovery_timeout=20.0,
            success_threshold=2,
        )
        self._lock  = asyncio.Lock()

    def _key(self, session_id: str) -> str:
        return f"{self._PREFIX}{session_id}"

    def _serialise(self, state: EQSSessionState) -> str: # noqa
        """Compact JSON serialisation of EQSSessionState."""
        arm_data: dict[str, dict[str, list[float]]] = {}
        for domain, arms in state.arm_states.items():
            arm_data[domain] = {c: [a.alpha, a.beta] for c, a in arms.items()}

        record_data = [
            {
                "ti": r.turn_index, "d": r.domain,
                "c": round(r.spec_composite, 5),
                "eig": round(r.eig_component, 5),
                "cov": round(r.coverage_component, 5),
                "tra": round(r.trajectory_component, 5),
                "rec": round(r.recency_component, 5),
                "cl":  round(r.cl_component, 5),
                "ws":  [round(w, 5) for w in r.weights_snapshot],
                "y":   round(r.eval_score, 5),
                "ts":  round(r.ts, 3),
            }
            for r in state.records[-60:]
        ]

        return json.dumps({
            "session_id":     state.session_id,
            "weights":        state.weights.as_vector(),
            "arms":           arm_data,
            "entropy_hist":   [[t, round(h, 6)] for t, h in state.entropy_history[-50:]],
            "records":        record_data,
            "recent_concepts":state.recent_concepts,
            "momentum":       round(state.momentum_velocity, 6),
            "total_turns":    state.total_turns,
            "created_at":     round(state.created_at, 3),
            "updated_at":     round(state.updated_at, 3),
        })

    def _deserialise(self, raw: str, session_id: str) -> EQSSessionState: # noqa
        d = json.loads(raw)
        weights  = EQSWeights.from_vector(d.get("weights", EQSWeights.default().as_vector()))
        arm_data = d.get("arms", {})
        arm_states: dict[str, dict[str, ThompsonArmState]] = {}
        for domain, arms in arm_data.items():
            arm_states[domain] = {
                c: ThompsonArmState(concept=c, alpha=ab[0], beta=ab[1])
                for c, ab in arms.items()
            }
        entropy_history = [(int(e[0]), float(e[1])) for e in d.get("entropy_hist", [])]
        records_raw = d.get("records", [])
        records = [
            EQSRecord(
                turn_index=r["ti"], domain=r["d"],
                spec_composite=r["c"], eig_component=r["eig"],
                coverage_component=r["cov"], trajectory_component=r["tra"],
                recency_component=r["rec"], cl_component=r["cl"],
                weights_snapshot=r["ws"], eval_score=r["y"], ts=r["ts"],
            )
            for r in records_raw
        ]
        return EQSSessionState(
            session_id=session_id,
            weights=weights,
            arm_states=arm_states,
            entropy_history=entropy_history,
            records=records,
            recent_concepts=d.get("recent_concepts", {}),
            momentum_velocity=d.get("momentum", 0.0),
            total_turns=d.get("total_turns", 0),
            created_at=d.get("created_at", time.monotonic()),
            updated_at=d.get("updated_at", time.monotonic()),
        )

    async def get(self, session_id: str) -> EQSSessionState | None:
        # Try LRU first
        cached = await self._lru.get(session_id)
        if cached:
            try:
                return self._deserialise(cached, session_id)
            except Exception: # noqa
                pass

        if self._redis is None:
            return None

        try:
            async def _fetch() -> EQSSessionState | None:
                raw = await self._redis.get(self._key(session_id))  # type: ignore[union-attr]
                if raw is None:
                    return None
                return self._deserialise(raw, session_id)

            return await self._cb.call(_fetch)
        except (CircuitBreakerOpen, Exception):
            return None

    async def set(self, state: EQSSessionState) -> None:
        raw = self._serialise(state)
        await self._lru.set(state.session_id, raw)

        if self._redis is None:
            return

        try:
            async def _store() -> None:
                await self._redis.setex(  # type: ignore[union-attr]
                    self._key(state.session_id),
                    EQS_REDIS_TTL_S,
                    raw,
                )

            await self._cb.call(_store)
        except (CircuitBreakerOpen, Exception) as exc:
            log.warning("eqs_redis_set_failed", session_id=state.session_id[:8], error=str(exc))

    @classmethod
    def new_state(cls, session_id: str, stated_level: str | None = None) -> EQSSessionState: # noqa
        """Create a fresh session state with default priors."""
        return EQSSessionState(
            session_id=session_id,
            weights=EQSWeights.default(),
            arm_states={},
            entropy_history=[],
            records=[],
            recent_concepts={},
        )


# ══════════════════════════════════════════════════════════════════════════════
# § 13. PROMETHEUS METRICS + OBSERVABILITY
# ══════════════════════════════════════════════════════════════════════════════

_eqs_selections = make_counter(
    "eqs_selections_total",
    "EpistemicQuestionSelector selections by strategy",
    ["strategy", "domain"],
)
_eqs_selection_latency = make_histogram(
    "eqs_selection_latency_seconds",
    "EQS select() wall-clock latency",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1),
)
_eqs_eig_distribution = make_histogram(
    "eqs_eig_distribution",
    "Distribution of top-candidate EIG scores",
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0),
)
_eqs_composite_score = make_histogram(
    "eqs_composite_score",
    "Distribution of winning composite scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
)
_eqs_weight_adaptations = make_counter(
    "eqs_weight_adaptations_total",
    "Online weight adaptation events",
    ["domain"],
)
_eqs_fallbacks = make_counter(
    "eqs_fallbacks_total",
    "EQS fallback activations (disabled or error)",
    ["reason"],
)
_eqs_thompson_samples = make_counter(
    "eqs_thompson_samples_total",
    "Thompson samples drawn by domain",
    ["domain"],
)
_eqs_candidates_evaluated = make_histogram(
    "eqs_candidates_evaluated",
    "Number of (concept, difficulty) candidates scored per turn",
    buckets=(1, 3, 5, 8, 12, 20, 30, 50),
)
_eqs_momentum_strategy = make_counter(
    "eqs_momentum_strategy_total",
    "Epistemic momentum strategy selections",
    ["strategy"],
)
_eqs_cl_zone_violations = make_counter(
    "eqs_cl_zone_violations_total",
    "Cognitive load zone violations detected in winning candidate",
    ["direction"],  # "too_easy" | "too_hard"
)


# ══════════════════════════════════════════════════════════════════════════════
# § 14. EPISTEMIC QUESTION SELECTOR — MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicQuestionSelector:
    """
    Joint optimizer over (concept_cluster × difficulty_band) space.

    Replaces the sequential two-patch approach in LLMInputBuilder.build():
        OLD: apply_signal_to_llm_input(signal, input) + to_suffix_instruction()
        NEW: spec = await eqs.select(...) → single coherent QuestionSpec

    Public API:
        EpistemicQuestionSelector.create(...)   → factory (loads/creates session state)
        await eqs.select(...)                   → QuestionSpec  (hot path, ≤5ms)
        await eqs.record_outcome(...)           → None          (fire-and-forget)
        await eqs.record_entropy(...)           → None          (called each turn)
        eqs.health()                            → dict

    Thread safety:
        All state mutations go through a per-session asyncio.Lock.
        select() is cancellation-safe.

    Error handling:
        Any internal exception falls back to a FallbackSpec with the effective
        level and action from the DomainScalerState — zero regression vs old code.
    """

    def __init__(
        self,
        session_id:     str,
        state:          EQSSessionState,
        store:          EQSStateStore,
        dep_graph:      ConceptDependencyGraph,
        fi_engine:      FisherInformationEngine,
        cl_estimator:   CognitiveLoadEstimator,
        momentum:       EpistemicMomentumTracker,
        sampler:        ThompsonConceptSampler,
        objective:      VariationalFreeEnergyObjective,
        adapter:        EQSWeightAdapter,
        enabled:        bool = EQS_ENABLED,
    ) -> None:
        self._session_id = session_id
        self._state      = state
        self._store      = store
        self._dep_graph  = dep_graph
        self._fi         = fi_engine
        self._cl         = cl_estimator
        self._momentum   = momentum
        self._sampler    = sampler
        self._obj        = objective
        self._adapter    = adapter
        self._enabled    = enabled
        self._lock       = asyncio.Lock()

    @classmethod
    async def create(
        cls,
        session_id:    str,
        redis:         aioredis.Redis | None = None,
        stated_level:  str | None = None,
        enabled:       bool = EQS_ENABLED,
    ) -> "EpistemicQuestionSelector":
        """
        Factory: load existing session state from Redis or create fresh state.
        Should be called once from SessionLifecycleManager.open_session().
        """
        store = EQSStateStore(redis=redis)
        state = await store.get(session_id)
        if state is None:
            state = EQSStateStore.new_state(session_id, stated_level)
            await store.set(state)

        dep_graph    = ConceptDependencyGraph()
        fi_engine    = FisherInformationEngine()
        cl_estimator = CognitiveLoadEstimator()
        momentum     = EpistemicMomentumTracker.from_entropy_history(state.entropy_history)
        sampler      = ThompsonConceptSampler(state)
        objective    = VariationalFreeEnergyObjective()
        adapter      = EQSWeightAdapter(state)

        return cls(
            session_id=session_id,
            state=state,
            store=store,
            dep_graph=dep_graph,
            fi_engine=fi_engine,
            cl_estimator=cl_estimator,
            momentum=momentum,
            sampler=sampler,
            objective=objective,
            adapter=adapter,
            enabled=enabled,
        )

    async def select(
        self,
        d_state:       Any,           # DomainScalerState
        coverage_map:  Any,           # ConceptCoverageMap
        domain:        str,
        q_index:       int,
        trajectory:    Any = None,    # Trajectory enum
        score_history: list[float] | None = None,
        failure_probs: dict[str, float] | None = None,
    ) -> QuestionSpec:
        """
        Core selection method. Returns a QuestionSpec that carries both
        the concept cluster and the difficulty level for the next question.

        Timing target: ≤5ms for a typical domain with ~15 concept clusters.
        All computation is pure CPU (no I/O on the hot path).

        Args:
            d_state:        DomainScalerState from PerformanceScaler.
                            Must have: .belief.probs, .current_difficulty,
                            .score_history, .last_action, .confirmed_level
            coverage_map:   ConceptCoverageMap from ConceptTracker.
                            Must have: .covered (dict[str, int])
            domain:         Domain key (e.g. "python", "dsa")
            q_index:        Question index within this domain (0-based)
            trajectory:     Trajectory enum (rising/plateau/falling/unknown)
            score_history:  List of recent eval scores [0..1] for this domain
            failure_probs:  Estimated P(failed) per concept (for dependency graph)

        Returns:
            QuestionSpec — use .effective_level and .build_suffix() in LLMInputBuilder
        """
        t0 = time.monotonic()

        if not self._enabled:
            _eqs_fallbacks.labels(reason="disabled").inc()
            return QuestionSpec.fallback(
                self._session_id, domain, q_index,
                effective_level=self._safe_level(d_state),
                action=self._safe_action(d_state),
                latency_ms=0.0,
            )

        try:
            async with self._lock:
                spec = await self._select_internal(
                    d_state=d_state,
                    coverage_map=coverage_map,
                    domain=domain,
                    q_index=q_index,
                    trajectory=trajectory,
                    score_history=score_history or [],
                    failure_probs=failure_probs or {},
                )
                spec.latency_ms = (time.monotonic() - t0) * 1000
                self._state.total_turns += 1
                self._state.updated_at  = time.monotonic()
                await self._store.set(self._state)
                return spec

        except Exception as exc:
            log.error(
                "eqs_select_error",
                session_id=self._session_id[:8],
                domain=domain, q_index=q_index,
                error=str(exc), exc_info=True,
            )
            _eqs_fallbacks.labels(reason="error").inc()
            return QuestionSpec.fallback(
                self._session_id, domain, q_index,
                effective_level=self._safe_level(d_state),
                action=self._safe_action(d_state),
                latency_ms=(time.monotonic() - t0) * 1000,
            )

    async def _select_internal(
        self,
        d_state:       Any,
        coverage_map:  Any,
        domain:        str,
        q_index:       int,
        trajectory:    Any,
        score_history: list[float],
        failure_probs: dict[str, float],
    ) -> QuestionSpec:
        """
        Internal hot path. Called with session lock held.
        Performs the full joint optimisation pipeline.
        """
        # ── Extract scalar inputs ─────────────────────────────────────────────
        prior_probs: list[float]  = getattr(d_state, "belief", None) and \
                                    list(getattr(d_state.belief, "probs", [1/3, 1/3, 1/3])) or \
                                    [1/3, 1/3, 1/3]
        current_diff: float       = float(getattr(d_state, "current_difficulty", 0.5))
        belief_entropy: float     = entropy_bits(prior_probs)
        action: Any               = getattr(d_state, "last_action", None)
        confirmed_level: str | None = getattr(d_state, "confirmed_level", None)
        traj_str: str             = self._trajectory_str(trajectory)

        # ── Momentum: derive strategy ─────────────────────────────────────────
        momentum_strategy = self._momentum.strategy()
        _eqs_momentum_strategy.labels(strategy=momentum_strategy.value).inc()

        # Adapt weights using momentum modifier
        base_weights   = self._adapter.current_weights(domain)
        mod            = self._momentum.weight_modifier()
        effective_w    = self._apply_weight_modifier(base_weights, mod)

        # ── Build candidate set ───────────────────────────────────────────────
        candidates = self._build_candidates(
            domain=domain,
            coverage_map=coverage_map,
            q_index=q_index,
            current_diff=current_diff,
            confirmed_level=confirmed_level,
            failure_probs=failure_probs,
            belief_entropy=belief_entropy,
        )

        if not candidates:
            # No concepts in registry for this domain — return scaler signal
            _eqs_fallbacks.labels(reason="no_candidates").inc()
            return QuestionSpec.fallback(
                self._session_id, domain, q_index,
                effective_level=self._safe_level(d_state),
                action=action,
            )

        # ── Score all candidates ──────────────────────────────────────────────
        candidates = self._score_candidates(
            candidates=candidates,
            prior_probs=prior_probs,
            coverage_map=coverage_map,
            domain=domain,
            q_index=q_index,
            current_diff=current_diff,
            trajectory=traj_str,
            score_history=score_history,
            belief_entropy=belief_entropy,
            failure_probs=failure_probs,
            effective_weights=effective_w,
        )

        # ── Sort and select ───────────────────────────────────────────────────
        candidates.sort(key=lambda c: -c.composite_score)
        winner = candidates[0]

        # ── Record for momentum and arm update ────────────────────────────────
        self._state.push_recent(domain, q_index, winner.concept)

        # ── Record prediction for weight adapter ─────────────────────────────
        self._adapter.record_prediction(
            turn_index=q_index, domain=domain,
            composite=winner.composite_score,
            eig=winner.eig, coverage=winner.coverage_value,
            trajectory=winner.trajectory_alignment,
            recency=winner.recency_penalty,
            cl=winner.cognitive_load_penalty,
            weights=effective_w,
        )

        # ── Emit metrics ──────────────────────────────────────────────────────
        _eqs_selections.labels(strategy=momentum_strategy.value, domain=domain).inc()
        _eqs_selection_latency.observe(0.0)   # latency tagged at call site
        _eqs_eig_distribution.observe(winner.eig)
        _eqs_composite_score.observe(winner.composite_score)
        _eqs_candidates_evaluated.observe(len(candidates))

        if winner.cognitive_load_penalty > 0.6:
            if winner.difficulty < CL_ZONE_LOW:
                _eqs_cl_zone_violations.labels(direction="too_easy").inc()
            else:
                _eqs_cl_zone_violations.labels(direction="too_hard").inc()

        # ── Build QuestionSpec ────────────────────────────────────────────────
        focus_keywords = (
            CONCEPT_DISCRIMINABILITY.get(domain, {})  # use keys as concept name proxy
            and list(
                (CONCEPT_PREREQUISITES.get(domain, {})).get(winner.concept, [])
            )[:4]
        ) or []

        # Fall back to concept name itself if no prerequisites (root concept)
        if not focus_keywords:
            focus_keywords = [winner.concept.replace("_", " ")]

        reasoning = self._obj.explain(winner, effective_w, traj_str)

        return QuestionSpec(
            session_id=self._session_id,
            domain=domain,
            turn_index=q_index,
            concept_cluster=winner.concept,
            concept_label=winner.concept_label,
            concept_focus_keywords=focus_keywords,
            difficulty_band=winner.difficulty_band,
            difficulty=winner.difficulty,
            effective_level=winner.effective_level,
            action=action,
            eig=winner.eig,
            coverage_value=winner.coverage_value,
            trajectory_alignment=winner.trajectory_alignment,
            recency_penalty=winner.recency_penalty,
            cognitive_load_penalty=winner.cognitive_load_penalty,
            thompson_bonus=winner.thompson_bonus,
            dependency_weight=winner.dependency_weight,
            composite_score=winner.composite_score,
            strategy=momentum_strategy,
            reasoning=reasoning,
            latency_ms=0.0,   # filled by caller
            n_candidates=len(candidates),
            weights_used=effective_w,
        )

    def _build_candidates(
        self,
        domain:           str,
        coverage_map:     Any, # noqa
        q_index:          int, # noqa
        current_diff:     float,
        confirmed_level:  str | None,
        failure_probs:    dict[str, float],
        belief_entropy:   float,
    ) -> list[SelectionCandidate]:
        """
        Enumerate all viable (concept, difficulty_band) pairs for this domain.
        Viability: concept exists in CONCEPT_PREREQUISITES, not fully blocked.
        """
        concepts = list(CONCEPT_PREREQUISITES.get(domain, {}).keys())
        if not concepts:
            # Domain not in dependency registry — use concept_tracker registry
            from app.eval.concept_tracker import CONCEPT_REGISTRY  # type: ignore
            concepts = list(CONCEPT_REGISTRY.get(domain, {}).keys())
        if not concepts:
            return []

        # Restrict difficulty bands if level is confirmed (COAST action)
        if confirmed_level == "beginner":
            bands = ["low", "mid"]
        elif confirmed_level == "advanced":
            bands = ["mid", "high"]
        else:
            bands = ["low", "mid", "high"]

        disc_map  = CONCEPT_DISCRIMINABILITY.get(domain, {})
        center_map = CONCEPT_DIFFICULTY_CENTER.get(domain, {})

        candidates: list[SelectionCandidate] = []
        for concept in concepts:
            disc   = disc_map.get(concept, _DEFAULT_DISCRIMINABILITY)
            center = center_map.get(concept, _DEFAULT_DIFFICULTY_CENTER)

            # Compute dependency weight (reduces if prerequisites uncertain/failed)
            dep_w = self._dep_graph.dependency_weight(
                domain=domain, concept=concept,
                failure_probs=failure_probs,
                belief_entropy=belief_entropy,
                max_entropy=_MAX_ENTROPY,
            )
            # Skip if nearly fully blocked
            if dep_w < 0.05:
                continue

            for band in bands:
                diff = self._fi.difficulty_for_band(current_diff, band, center)
                level = self._fi.level_for_difficulty(diff)

                # Derive concept label (replace underscores for readability)
                label = concept.replace("_", " ").title()

                candidates.append(SelectionCandidate(
                    concept=concept,
                    concept_label=label,
                    difficulty_band=band,
                    difficulty=diff,
                    effective_level=level,
                    discrimination=disc,
                    difficulty_center=center,
                    dependency_weight=dep_w,
                ))

        return candidates

    def _score_candidates(
        self,
        candidates:      list[SelectionCandidate],
        prior_probs:     list[float],
        coverage_map:    Any,
        domain:          str,
        q_index:         int,
        current_diff:    float, # noqa
        trajectory:      str,
        score_history:   list[float],
        belief_entropy:  float, # noqa
        failure_probs:   dict[str, float], # noqa
        effective_weights: EQSWeights,
    ) -> list[SelectionCandidate]:
        """
        Score each candidate on all five components, then compute composite.
        Also adds Thompson sampling bonus.
        """
        # Compute raw EIG for all candidates first (for normalisation)
        raw_eigs = []
        for cand in candidates:
            eig_raw = self._fi.expected_ig(
                prior_probs=prior_probs,
                difficulty=cand.difficulty,
                discrimination=cand.discrimination,
            )
            raw_eigs.append(eig_raw)

        # Normalise EIG scores to [0, 1] within this candidate set
        normed_eigs = normalise_to_unit(raw_eigs)

        # Recent concept keys for CL extraneous computation
        recent_concept_keys = [
            c for _, c in self._state.recent_concepts.get(domain, [])
        ][-4:]

        # MAP ability estimate for CL intrinsic load
        map_idx = int(np.argmax(prior_probs))
        ability_map = _ABILITY_LEVELS_NUMERIC[map_idx]

        for i, cand in enumerate(candidates):
            # EIG (normalised)
            cand.eig = normed_eigs[i]

            # Coverage utility
            cand.coverage_value = self._obj.coverage_utility(
                cand.concept, coverage_map, domain
            )

            # Trajectory alignment
            c_type = _concept_type(cand.discrimination)
            cand.trajectory_alignment = self._obj.trajectory_alignment(
                concept_type=c_type,
                difficulty_band=cand.difficulty_band,
                trajectory=trajectory,
                score_history=score_history,
            )

            # Recency penalty
            turns_ago = self._state.recency_turns_ago(domain, cand.concept, q_index)
            cand.recency_penalty = self._obj.recency_score(cand.concept, turns_ago)

            # Cognitive load penalty
            cl_pen, _ = self._cl.compute_penalty(
                concept=cand.concept,
                difficulty=cand.difficulty,
                discrimination=cand.discrimination,
                ability_estimate=ability_map,
                trajectory=trajectory,
                recent_concept_keys=recent_concept_keys,
            )
            cand.cognitive_load_penalty = cl_pen

            # Thompson bonus (stochastic exploration)
            _eqs_thompson_samples.labels(domain=domain).inc()
            ts_draw = self._sampler.sample_bonus(domain, cand.concept)
            # Scale Thompson bonus: small relative to deterministic components
            cand.thompson_bonus = ts_draw * 0.06

            # Apply dependency weight as a multiplicative gate on the full score
            base_score = self._obj.score(
                eig=cand.eig,
                coverage=cand.coverage_value,
                trajectory=cand.trajectory_alignment,
                recency=cand.recency_penalty,
                cl_penalty=cand.cognitive_load_penalty,
                weights=effective_weights,
            )
            cand.composite_score = float(np.clip(
                (base_score + cand.thompson_bonus) * cand.dependency_weight,
                0.0, 1.0,
            ))

        return candidates

    async def record_outcome(
        self,
        domain:     str,
        turn_index: int,
        spec:       QuestionSpec,
        eval_score: float,
    ) -> None:
        """
        Feed an eval score back into the adapter and Thompson sampler.
        Should be called fire-and-forget after eval_engine delivers a score.

        eval_score: raw 0–10 score from eval_engine rubric (will be normalised).
        """
        try:
            normalised = float(np.clip(eval_score / 10.0, 0.0, 1.0))
            async with self._lock:
                # Weight adapter
                self._adapter.record_outcome(turn_index, normalised)

                # Thompson arm update
                realised_ig_proxy = normalised * spec.eig   # proxy for realised IG
                self._sampler.update(
                    domain=domain,
                    concept=spec.concept_cluster,
                    realised_ig=realised_ig_proxy,
                    ig_normalised=spec.eig,
                )

                self._state.updated_at = time.monotonic()
                await self._store.set(self._state)

            _eqs_weight_adaptations.labels(domain=domain).inc()
            log.debug(
                "eqs_outcome_recorded",
                session_id=self._session_id[:8],
                domain=domain, turn_index=turn_index,
                eval_score=round(eval_score, 2),
                normalised=round(normalised, 4),
            )
        except Exception as exc:
            log.warning(
                "eqs_record_outcome_error",
                session_id=self._session_id[:8],
                error=str(exc),
            )

    async def record_entropy(
        self,
        turn_index: int,
        entropy:    float,
    ) -> None:
        """
        Update epistemic momentum from the current belief entropy.
        Call this after each scored turn with d_state.belief.entropy().
        """
        try:
            async with self._lock:
                delta = self._momentum.record_entropy(turn_index, entropy) # noqa
                self._state.entropy_history.append((turn_index, entropy))
                self._state.momentum_velocity = self._momentum.velocity
                # Keep entropy history bounded
                if len(self._state.entropy_history) > 50:
                    self._state.entropy_history = self._state.entropy_history[-50:]
                await self._store.set(self._state)
        except Exception as exc:
            log.debug("eqs_record_entropy_error", error=str(exc))

    def health(self) -> dict[str, Any]:
        """Diagnostic snapshot for the /health and /debug/session endpoints."""
        return {
            "session_id":   self._session_id[:8],
            "enabled":      self._enabled,
            "total_turns":  self._state.total_turns,
            "momentum":     self._momentum.to_dict(),
            "weights":      self._state.weights.to_dict(),
            "weight_adapt": self._adapter.convergence_diagnostics(),
            "n_records":    len(self._state.records),
            "n_domains_armed": len(self._state.arm_states),
            "updated_s_ago":round(time.monotonic() - self._state.updated_at, 1),
        }

    # ── private utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _trajectory_str(trajectory: Any) -> str:
        """Safely extract the trajectory string from the enum or default."""
        if trajectory is None:
            return "unknown"
        val = getattr(trajectory, "value", None)
        return str(val) if val else str(trajectory).lower()

    @staticmethod
    def _safe_level(d_state: Any) -> str:
        confirmed = getattr(d_state, "confirmed_level", None)
        if confirmed:
            return confirmed
        if hasattr(d_state, "belief") and hasattr(d_state.belief, "map_level"):
            return d_state.belief.map_level
        return "intermediate"

    @staticmethod
    def _safe_action(d_state: Any) -> Any:
        return getattr(d_state, "last_action", None)

    @staticmethod
    def _apply_weight_modifier(
        base: EQSWeights,
        mod: dict[str, float],
    ) -> EQSWeights:
        """Apply momentum modifier to base weights and project back to simplex."""
        v = base.as_vector()
        keys = ["eig", "coverage", "trajectory", "recency", "cognitive_load"]
        for i, k in enumerate(keys):
            v[i] = max(0.0, v[i] + mod.get(k, 0.0))
        projected = project_simplex(v)
        return EQSWeights.from_vector(projected)


# ══════════════════════════════════════════════════════════════════════════════
# § 15. INTEGRATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def spec_to_llm_input_patch(
    spec:      QuestionSpec,
    llm_input: Any,           # LLMInterviewInput
) -> Any:
    """
    Apply a QuestionSpec to an LLMInterviewInput in-place.

    This is a drop-in replacement for apply_signal_to_llm_input() from
    performance_scaler.py AND to_suffix_instruction() from concept_tracker.py.
    Call this in LLMInputBuilder.build() after constructing the base llm_input.

    Mutates:
        llm_input.level           ← spec.effective_level
        llm_input.messages[-1]    ← appends spec.build_suffix() to system prompt
        llm_input.difficulty_hint ← spec.difficulty_band (if field exists)

    Returns the patched llm_input (same object, for chaining).
    """
    if spec.strategy == SelectionStrategy.FALLBACK or not spec.concept_cluster:
        # EQS disabled or errored — don't touch the input
        return llm_input

    # Patch level
    if hasattr(llm_input, "level") and spec.effective_level:
        object.__setattr__(llm_input, "level", spec.effective_level) \
            if hasattr(type(llm_input), "__dataclass_fields__") else \
            setattr(llm_input, "level", spec.effective_level)

    # Append concept suffix to the system message
    suffix = spec.build_suffix()
    if suffix and hasattr(llm_input, "messages") and llm_input.messages:
        from langchain_core.messages import SystemMessage  # type: ignore
        msgs = list(llm_input.messages)
        if msgs and hasattr(msgs[0], "content"):
            existing = msgs[0].content
            msgs[0] = SystemMessage(content=f"{existing}\n\n{suffix}")
            try:
                object.__setattr__(llm_input, "messages", msgs)
            except (TypeError, AttributeError):
                setattr(llm_input, "messages", msgs)

    return llm_input


def should_use_eqs(q_index: int, domain: str) -> bool: # noqa
    """
    Guard: whether EQS should be activated for this turn.
    Always returns False when EQS_ENABLED=false.
    Waits for q_index >= 1 (first question uses scaler defaults).
    """
    return EQS_ENABLED and q_index >= 1


async def build_failure_probs_from_scores(
    session_id:  str, # noqa
    domain:      str,
    qa_document: Any,          # QADocument
    eval_scores: dict[int, float],   # turn_index → eval_score (0–10)
) -> dict[str, float]:
    """
    Estimate P(failed | concept) from committed turns in the QA document.

    Uses a simple heuristic: if a turn in a concept's thematic area has
    eval score < 4.0 (below ZPD floor on 0–10 scale), that concept is
    considered potentially failed.

    Returns mapping of concept_key → P(failed) ∈ [0, 1].
    This is consumed by ConceptDependencyGraph.blocking_probability().

    Note: this is a coarse approximation. A proper implementation would
    use the eval rubric's technical_accuracy dimension specifically and
    match questions to concept clusters via the CONCEPT_REGISTRY keywords.
    """
    failure_probs: dict[str, float] = {}
    if not hasattr(qa_document, "turns"):
        return failure_probs

    prereqs = CONCEPT_PREREQUISITES.get(domain, {})
    disc_map = CONCEPT_DISCRIMINABILITY.get(domain, {}) # noqa
    center_map = CONCEPT_DIFFICULTY_CENTER.get(domain, {}) # noqa

    for turn in getattr(qa_document, "turns", []):
        ti = getattr(turn, "turn_index", -1)
        score = eval_scores.get(ti, -1.0)
        if score < 0:
            continue

        q_text = getattr(turn, "q", "").lower()
        # Match question text against concept keywords
        for concept in prereqs:
            # Simple heuristic: check if concept name words appear in question
            tokens = concept.replace("_", " ").split()
            matched = any(tok in q_text for tok in tokens if len(tok) > 3)
            if matched:
                # Low score on a matched question → elevated failure probability
                normalised_score = score / 10.0
                failure_prob = max(0.0, 1.0 - normalised_score) ** 1.5
                failure_probs[concept] = max(
                    failure_probs.get(concept, 0.0),
                    failure_prob,
                )

    return failure_probs


# ══════════════════════════════════════════════════════════════════════════════
# § 16. MODULE-LEVEL SINGLETON (optional convenience)
# ══════════════════════════════════════════════════════════════════════════════
#
# For test environments and simple integrations, a module-level session registry
# maps session_id → EpistemicQuestionSelector. In production, the instance is
# owned by SessionLifecycleManager.SessionResources and lifecycle-managed there.
# ──────────────────────────────────────────────────────────────────────────────

_session_registry: dict[str, EpistemicQuestionSelector] = {}
_registry_lock = threading.Lock()


def register_selector(session_id: str, selector: "EpistemicQuestionSelector") -> None:
    with _registry_lock:
        _session_registry[session_id] = selector


def get_selector(session_id: str) -> "EpistemicQuestionSelector | None":
    with _registry_lock:
        return _session_registry.get(session_id)


def deregister_selector(session_id: str) -> None:
    with _registry_lock:
        _session_registry.pop(session_id, None)


async def get_or_create_selector(
    session_id:   str,
    redis:        aioredis.Redis | None = None,
    stated_level: str | None = None,
) -> "EpistemicQuestionSelector":
    """
    Convenience: get existing selector for session or create a new one.
    Thread-safe. Suitable for single-process deployments without explicit
    lifecycle management.
    """
    existing = get_selector(session_id)
    if existing is not None:
        return existing

    selector = await EpistemicQuestionSelector.create(
        session_id=session_id,
        redis=redis,
        stated_level=stated_level,
    )
    register_selector(session_id, selector)
    return selector


# ══════════════════════════════════════════════════════════════════════════════
# § 17. CONCEPT REGISTRY FALLBACK
# ══════════════════════════════════════════════════════════════════════════════
#
# For domains not in CONCEPT_PREREQUISITES (e.g., behavioural, php, golang),
# we supply a minimal registry so the EQS still operates. Without this,
# _build_candidates() would return empty for unknown domains.
# ──────────────────────────────────────────────────────────────────────────────

_MINIMAL_FALLBACK_PREREQS: dict[str, dict[str, list[str]]] = {
    "behavioral": {
        "communication":     [],
        "conflict_resolution":["communication"],
        "leadership":        ["communication"],
        "teamwork":          ["communication"],
        "problem_solving":   [],
        "ownership":         ["leadership"],
        "adaptability":      ["problem_solving"],
    },
    "golang": {
        "basics":            [],
        "goroutines":        ["basics"],
        "channels":          ["goroutines"],
        "interfaces":        ["basics"],
        "error_handling":    ["basics"],
        "context":           ["goroutines", "channels"],
        "testing":           ["basics", "interfaces"],
        "concurrency_patterns": ["channels", "context"],
    },
    "php": {
        "basics":            [],
        "oop":               ["basics"],
        "namespaces":        ["oop"],
        "traits":            ["oop"],
        "async":             ["basics"],
        "composer":          ["namespaces"],
        "testing":           ["oop"],
    },
    "rust": {
        "ownership":         [],
        "borrowing":         ["ownership"],
        "lifetimes":         ["borrowing"],
        "traits":            ["ownership"],
        "enums_pattern":     ["ownership"],
        "generics":          ["traits"],
        "async_rust":        ["traits", "ownership"],
        "unsafe":            ["lifetimes", "generics"],
    },
}


def _enrich_concept_prerequisites() -> None:
    """Merge fallback registries into the main CONCEPT_PREREQUISITES."""
    for domain, prereqs in _MINIMAL_FALLBACK_PREREQS.items():
        if domain not in CONCEPT_PREREQUISITES:
            CONCEPT_PREREQUISITES[domain] = prereqs
            # Supply neutral discriminability values
            CONCEPT_DISCRIMINABILITY.setdefault(domain, {})
            for concept in prereqs:
                CONCEPT_DISCRIMINABILITY[domain].setdefault(concept, 1.4)


_enrich_concept_prerequisites()


# ══════════════════════════════════════════════════════════════════════════════
# § 18. SELF-TESTS (importable; run with `python -m pytest epistemic_question_selector.py`)
# ══════════════════════════════════════════════════════════════════════════════

def _test_irt_2pl() -> None:
    """IRT 2PL: P(correct) should be 0.5 when ability == difficulty."""
    p = irt_2pl(ability=0.5, difficulty=0.5, discrimination=2.0)
    assert abs(p - 0.5) < 0.01, f"IRT 2PL symmetry failed: {p}"
    # Higher ability → higher P(correct)
    assert irt_2pl(0.8, 0.5, 2.0) > irt_2pl(0.5, 0.5, 2.0)
    # Higher discrimination → steeper curve
    p_low  = irt_2pl(0.7, 0.5, 0.5)
    p_high = irt_2pl(0.7, 0.5, 3.0)
    assert p_high > p_low


def _test_fisher_information() -> None:
    """FI maximised when ability == difficulty."""
    fi_at_match   = fisher_information_2pl(0.5, 0.5, 2.0)
    fi_off_by_half = fisher_information_2pl(0.5, 1.0, 2.0)
    assert fi_at_match > fi_off_by_half, "FI should be max when ability=difficulty"


def _test_eig() -> None:
    """EIG should be positive and ≤ max entropy."""
    prior = [1/3, 1/3, 1/3]
    eig = expected_information_gain_discrete(prior, _ABILITY_LEVELS_NUMERIC, 0.5, 1.5)
    assert 0.0 <= eig <= _MAX_ENTROPY + 1e-9, f"EIG out of range: {eig}"
    # Uninformative item (discrimination=0) should give near-zero EIG
    eig_zero = expected_information_gain_discrete(prior, _ABILITY_LEVELS_NUMERIC, 0.5, 0.001)
    assert eig_zero < 0.01, f"Near-zero discrimination should give near-zero EIG: {eig_zero}"


def _test_project_simplex() -> None:
    """Project simplex should return valid probability vector."""
    v = [0.5, 0.6, -0.2, 0.3, 0.1]
    projected = project_simplex(v)
    assert all(w >= 0.0 for w in projected), "Simplex projection must be non-negative"
    assert abs(sum(projected) - 1.0) < 1e-9, "Simplex projection must sum to 1"


def _test_eqs_weights() -> None:
    """EQSWeights default should be valid."""
    w = EQSWeights.default()
    assert w.is_valid(), f"Default weights invalid: {w.to_dict()}"
    v = w.as_vector()
    roundtrip = EQSWeights.from_vector(v)
    assert roundtrip.is_valid()


def _test_dependency_graph() -> None:
    """Dependency graph should compute topological priority correctly."""
    g = ConceptDependencyGraph()
    # Root concepts should have highest priority (1.0)
    root_priority = g.topological_priority("python", "variables_types")
    leaf_priority  = g.topological_priority("python", "metaclasses")
    assert root_priority > leaf_priority, (
        f"Root should have higher priority than leaf: {root_priority} vs {leaf_priority}"
    )
    # No prerequisites → blocking prob should be 0
    block = g.blocking_probability("python", "variables_types", {})
    assert block == 0.0


def _test_cl_estimator() -> None:
    """Cognitive load penalty should be 0 near zone centre."""
    cl = CognitiveLoadEstimator()
    # Perfectly matched difficulty → low intrinsic load
    intr = cl.intrinsic_load(ability_estimate=0.5, question_difficulty=0.5)
    assert intr < 0.05, f"Zero mismatch should give near-zero intrinsic load: {intr}"
    # Load penalty at zone centre should be minimal
    penalty = cl.load_penalty(total_cl=CL_ZONE_CENTER)
    assert penalty < 0.01, f"Zone centre should give near-zero penalty: {penalty}"


def _test_momentum_tracker() -> None:
    """Momentum tracker should reflect entropy changes."""
    tracker = EpistemicMomentumTracker()
    # Feed decreasing entropies (converging belief)
    for i, h in enumerate([1.09, 0.95, 0.80, 0.65, 0.50, 0.35]):
        tracker.record_entropy(i, h)
    assert tracker.velocity > 0, "Converging belief should have positive velocity"


def run_self_tests() -> None:
    """Run all self-tests and report results."""
    tests = [
        _test_irt_2pl, _test_fisher_information, _test_eig,
        _test_project_simplex, _test_eqs_weights, _test_dependency_graph,
        _test_cl_estimator, _test_momentum_tracker,
    ]
    passed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  ✓ {test_fn.__name__}")
        except AssertionError as e:
            print(f"  ✗ {test_fn.__name__}: {e}")
        except Exception as e:
            print(f"  ! {test_fn.__name__} (exception): {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    print("Running EpistemicQuestionSelector self-tests...\n")
    run_self_tests()