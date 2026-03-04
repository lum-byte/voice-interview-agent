from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.pgs._core import TurnRecord
from app.pgs._normalizer import _normalizer

if TYPE_CHECKING:
    pass

_MIN_TURNS          = 2
_DEFLECT_THRESHOLD  = 0.25    # concept overlap below 25% = deflection
                               # genuine technical interview answers cluster 40-70% overlap;
                               # 0.28 was catching candidates who use synonyms, not deflectors.
                               # 0.25 targets actual avoidance without synonym penalization.
_PARTIAL_THRESHOLD  = 0.58    # NOT 0.52 — direct engagement threshold
                               # 52% overlap is a borderline answer, not clear engagement.
                               # 0.58 correctly separates "sort of answered it" from
                               # "directly addressed the question" in technical domains.
_EWMA_ALPHA         = 0.58    # lower than velocity/hedge — deflection patterns are sticky,
                               # a candidate who deflects continues to deflect for several turns.
                               # Lower alpha preserves history weight over recency.
_OPENER_WEIGHT      = 0.32    # slight reduction — opener detection is a strong prior
                               # but shouldn't dominate when overlap signal is clear


_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "python": [
        "list", "dict", "tuple", "set", "function", "class", "object",
        "inherit", "module", "import", "decorator", "generator", "iterator",
        "exception", "thread", "async", "await", "gil", "memory", "garbage",
        "reference", "scope", "closure", "lambda", "comprehension", "slice",
        "mutable", "immutable", "namespace", "attribute", "method", "instance",
        "metaclass", "descriptor", "pickle", "pickle", "bytecode", "cpython",
    ],
    "java": [
        "jvm", "class", "interface", "abstract", "inherit", "polymorphism",
        "encapsulation", "static", "final", "thread", "synchronize", "lock",
        "concurrent", "collection", "generic", "annotation", "reflection",
        "garbage", "heap", "stack", "bytecode", "compile", "runtime", "jit",
        "stream", "lambda", "optional", "exception", "checked", "unchecked",
    ],
    "dsa": [
        "array", "linked list", "tree", "graph", "hash", "stack", "queue",
        "heap", "sort", "search", "binary", "recursion", "dynamic programming",
        "greedy", "backtrack", "complexity", "time", "space", "big o",
        "node", "edge", "path", "cycle", "depth", "breadth", "traversal",
        "insert", "delete", "lookup", "balance", "rotate", "partition",
    ],
    "system_design": [
        "scale", "load balance", "cache", "database", "replication",
        "partition", "shard", "consistency", "availability", "partition tolerance",
        "cap theorem", "microservice", "api", "rest", "message queue", "kafka",
        "redis", "cdn", "rate limit", "circuit breaker", "failover",
        "horizontal", "vertical", "stateless", "idempotent", "eventual",
    ],
    "databases": [
        "sql", "table", "index", "query", "join", "transaction", "acid",
        "normalize", "schema", "primary key", "foreign key", "constraint",
        "aggregate", "group by", "having", "subquery", "view", "stored procedure",
        "trigger", "lock", "deadlock", "isolation", "dirty read", "phantom",
        "nosql", "document", "key value", "columnar", "graph", "replication",
    ],
    "os_concepts": [
        "process", "thread", "scheduler", "memory", "virtual", "page", "frame",
        "segment", "deadlock", "mutex", "semaphore", "monitor", "interrupt",
        "system call", "kernel", "user space", "file system", "inode",
        "buffer", "cache", "swap", "context switch", "preempt", "priority",
        "starvation", "synchronization", "race condition", "critical section",
    ],
    "javascript": [
        "closure", "prototype", "scope", "hoisting", "event loop", "callback",
        "promise", "async", "await", "dom", "bom", "window", "this", "bind",
        "call", "apply", "spread", "destructure", "module", "import", "export",
        "typescript", "class", "inherit", "getter", "setter", "proxy", "reflect",
    ],
    "ml": [
        "model", "train", "test", "validate", "overfit", "underfit", "bias",
        "variance", "gradient", "backprop", "loss", "activation", "layer",
        "weight", "epoch", "batch", "learning rate", "regularize", "dropout",
        "convolution", "attention", "transformer", "embedding", "feature",
        "classification", "regression", "cluster", "dimension reduction",
    ],
}

_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "it", "this", "that",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "be", "have", "has", "had", "will", "would", "could", "should",
    "do", "does", "did", "not", "and", "or", "but", "so", "if",
    "you", "i", "we", "they", "he", "she", "me", "my", "your",
    "what", "when", "where", "how", "why", "which", "who",
    "can", "may", "might", "let", "just", "also", "even", "then",
    "use", "used", "using", "used", "get", "got", "put", "say", "said",
    "think", "know", "mean", "like", "want", "need", "make", "take",
})


def _tokens(text: str) -> set[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _bigrams(text: str) -> set[str]:
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return {f"{words[i]} {words[i+1]}" for i in range(len(words)-1)
            if words[i] not in _STOPWORDS or words[i+1] not in _STOPWORDS}


def _concept_overlap(q: str, a: str, domain: str) -> float:
    domain_kws = set(_DOMAIN_KEYWORDS.get(domain, []))
    q_tokens   = _tokens(q) | _bigrams(q)
    a_tokens   = _tokens(a) | _bigrams(a)
    q_domain   = q_tokens & domain_kws
    a_domain   = a_tokens & domain_kws

    if not q_domain:
        # Question didn't use domain keywords — use general token overlap
        union = q_tokens | a_tokens
        if not union:
            return 0.5
        return len(q_tokens & a_tokens) / len(union)

    if not a_domain:
        return 0.0

    q_domain_in_bigrams = _bigrams(q) & domain_kws
    a_domain_in_bigrams = _bigrams(a) & domain_kws
    q_concept = q_domain | q_domain_in_bigrams
    a_concept = a_domain | a_domain_in_bigrams

    if not q_concept:
        return 0.5

    direct_overlap = len(q_concept & a_concept) / len(q_concept)
    general_overlap = len(q_tokens & a_tokens) / max(1, len(q_tokens | a_tokens))
    return direct_overlap * 0.70 + general_overlap * 0.30


@dataclass
class _DeflectWindow:
    overlap_scores: list[float] = field(default_factory=list)
    opener_flags:   list[bool]  = field(default_factory=list)
    ewma_overlap:   float       = 0.6
    n:              int         = 0


class DeflectionEngine:
    """
    Measures concept overlap between question keywords and answer content.

    Low overlap: candidate answered adjacent to the question — deflection.
    High deflection + opener flag: deliberate avoidance.
    High deflection + no opener + no hedging: vocabulary gap (candidate
    knows the concept but not the terminology).

    These two deflection types require different constraint responses.
    """

    def __init__(self) -> None:
        self._windows: dict[str, _DeflectWindow] = defaultdict(_DeflectWindow)

    def ingest(self, session_id: str, turn: TurnRecord) -> None:
        w   = self._windows[session_id]
        nr  = _normalizer.normalize(turn.answer)

        overlap = _concept_overlap(turn.question, nr.clean, turn.domain)
        opener  = _normalizer.is_deflection_open(turn.answer)

        w.overlap_scores.append(overlap)
        w.opener_flags.append(opener)

        if w.n == 0:
            w.ewma_overlap = overlap
        else:
            w.ewma_overlap = _EWMA_ALPHA * overlap + (1 - _EWMA_ALPHA) * w.ewma_overlap

        w.n += 1

    def compute(self, session_id: str) -> float:
        """
        Returns float in [0, 1].
        0.5 = neutral.
        < 0.5 = deflection (low overlap, answering adjacent).
        > 0.5 = direct engagement (high overlap, directly addressing Q).
        """
        w = self._windows.get(session_id)
        if not w or w.n < _MIN_TURNS:
            return 0.5

        base_signal   = self._overlap_signal(w)
        opener_signal = self._opener_signal(w)

        composite = base_signal * (1 - _OPENER_WEIGHT) + opener_signal * _OPENER_WEIGHT
        return max(0.0, min(1.0, composite))

    def _overlap_signal(self, w: _DeflectWindow) -> float:
        mean_overlap = statistics.mean(w.overlap_scores)
        if mean_overlap < _DEFLECT_THRESHOLD:
            depth = (_DEFLECT_THRESHOLD - mean_overlap) / _DEFLECT_THRESHOLD
            return max(0.0, 0.5 - depth * 0.45)
        if mean_overlap > _PARTIAL_THRESHOLD:
            height = (mean_overlap - _PARTIAL_THRESHOLD) / (1.0 - _PARTIAL_THRESHOLD)
            return min(1.0, 0.5 + height * 0.45)
        ratio = (mean_overlap - _DEFLECT_THRESHOLD) / (_PARTIAL_THRESHOLD - _DEFLECT_THRESHOLD)
        return 0.5 - 0.10 + ratio * 0.20

    def _opener_signal(self, w: _DeflectWindow) -> float:
        if not w.opener_flags:
            return 0.5
        recent_opens = sum(w.opener_flags[-4:]) / min(4, len(w.opener_flags))
        if recent_opens > 0.5:
            return max(0.0, 0.5 - recent_opens * 0.40)
        return 0.5

    def deflection_type(self, session_id: str, turn: TurnRecord) -> str:
        """
        Classify the most recent deflection type if deflection is occurring.
        Used by the compiler to choose probe angle.
        """
        w = self._windows.get(session_id)
        if not w or not w.overlap_scores:
            return "none"

        last_overlap = w.overlap_scores[-1]
        last_opener  = w.opener_flags[-1] if w.opener_flags else False

        if last_overlap > _PARTIAL_THRESHOLD:
            return "none"
        if last_opener:
            return "avoidance"

        nr = _normalizer.normalize(turn.answer)
        if nr.hedge_count > 2:
            return "uncertainty"
        if nr.word_count < 20:
            return "withdrawal"
        return "vocabulary_gap"

    def evict(self, session_id: str) -> None:
        self._windows.pop(session_id, None)
