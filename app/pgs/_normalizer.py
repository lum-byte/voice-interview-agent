from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import NamedTuple

# ── filler corpus — spoken language specific ──────────────────────────────────
_FILLERS: frozenset[str] = frozenset({
    "um", "uh", "uhh", "umm", "hmm", "hm", "ah", "ahh", "er", "err",
    "like", "you know", "you know what i mean", "i mean", "kind of",
    "sort of", "basically", "literally", "actually", "obviously",
    "so", "right", "okay", "ok", "well", "alright", "anyway",
    "let me think", "let me see", "give me a second", "hold on",
    "how do i put this", "how do i say this",
})

_FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in sorted(_FILLERS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ── hedge corpus — preserved, not removed ────────────────────────────────────
# these are extracted and counted BEFORE filler removal
_HEDGE_PHRASES: list[str] = [
    "i think", "i believe", "i guess", "i suppose", "i assume",
    "i'm not sure", "i'm not entirely sure", "i'm not certain",
    "i could be wrong", "i might be wrong", "don't quote me",
    "from what i recall", "if i remember correctly", "if i recall correctly",
    "as far as i know", "to the best of my knowledge", "to my knowledge",
    "i think it might be", "i think it could be", "i think it's probably",
    "i'm fairly certain", "i'm pretty sure", "i'm somewhat sure",
    "i would say", "i would think", "i would guess",
    "maybe", "perhaps", "possibly", "probably", "likely",
    "might", "could", "should probably", "would probably",
    "something like", "something along the lines of",
    "roughly", "approximately", "around", "sort of like",
    "kind of like", "more or less", "in a way",
    "i think what happens is", "i believe what happens is",
    "if i'm not mistaken", "unless i'm wrong",
    "i'm not a hundred percent sure", "i'm not totally sure",
    "i vaguely remember", "i faintly recall",
]

_HEDGE_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(p) for p in sorted(_HEDGE_PHRASES, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)

# ── deflection phrase markers — answers that avoid the question ───────────────
_DEFLECT_OPENERS: list[str] = [
    "that's a great question", "that's an interesting question",
    "it depends", "it really depends", "it depends on",
    "there are many ways", "there are several ways", "there are different ways",
    "that's a broad topic", "that's a complex topic", "that's a big topic",
    "it's complicated", "it's complex", "it's hard to say",
    "in general", "generally speaking", "broadly speaking",
    "from a high level", "at a high level", "from a high-level perspective",
    "so the thing is", "the thing about that is",
    "well it depends", "well that depends",
]

_DEFLECT_RE = re.compile(
    r"(?:^|\.\s+)(" + "|".join(re.escape(p) for p in sorted(_DEFLECT_OPENERS, key=len, reverse=True)) + r")",
    re.IGNORECASE,
)

# ── sentence boundary inference for STT output ───────────────────────────────
# STT often produces run-on text with no punctuation
_BOUNDARY_MARKERS = re.compile(
    r"\b(so|then|and then|therefore|however|but|although|whereas|because|"
    r"which means|which is why|the reason is|basically|ultimately|"
    r"at the end of the day|in other words|to summarize)\b",
    re.IGNORECASE,
)

_MULTI_SPACE = re.compile(r"\s{2,}")
_NON_ALPHA   = re.compile(r"[^\w\s\'-]")


# ── output ────────────────────────────────────────────────────────────────────

class NormResult(NamedTuple):
    clean:          str
    hedge_count:    int
    hedge_positions: list[int]
    deflect_opens:  int
    filler_count:   int
    word_count:     int
    sentence_count: int
    raw_words:      int


@dataclass
class TextNormalizer:

    def normalize(self, text: str) -> NormResult:
        if not text or not text.strip():
            return NormResult("", 0, [], 0, 0, 0, 0, 0)

        text = unicodedata.normalize("NFKC", text)
        text = text.strip()

        raw_words = len(text.split())

        hedge_matches = list(_HEDGE_RE.finditer(text))
        hedge_count = len(hedge_matches)
        hedge_positions = [m.start() for m in hedge_matches]

        deflect_opens = len(_DEFLECT_RE.findall(text))

        filler_matches = _FILLER_RE.findall(text)
        filler_count = len(filler_matches)
        clean = _FILLER_RE.sub(" ", text)

        clean = _NON_ALPHA.sub(" ", clean)
        clean = _MULTI_SPACE.sub(" ", clean).strip().lower()

        word_count = len(clean.split()) if clean else 0

        sentence_count = max(1, self._infer_sentences(clean))

        return NormResult(
            clean           = clean,
            hedge_count     = hedge_count,
            hedge_positions = hedge_positions,
            deflect_opens   = deflect_opens,
            filler_count    = filler_count,
            word_count      = word_count,
            sentence_count  = sentence_count,
            raw_words       = raw_words,
        )

    def _infer_sentences(self, text: str) -> int:
        explicit = text.count(".") + text.count("?") + text.count("!")
        if explicit > 1:
            return explicit
        boundaries = len(_BOUNDARY_MARKERS.findall(text))
        words = len(text.split())
        implicit = max(0, words // 22)
        return max(1, explicit + boundaries + implicit)

    def extract_hedges(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in _HEDGE_RE.finditer(text)]

    def is_deflection_open(self, text: str) -> bool:
        stripped = text.strip()
        for p in _DEFLECT_OPENERS:
            if stripped.lower().startswith(p):
                return True
        return bool(_DEFLECT_RE.search(stripped[:120]))

    def word_count(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def sentence_fragments(self, text: str) -> int:
        words = text.split()
        if not words:
            return 0
        chunks = re.split(r"[.!?]|\band\b|\bbut\b|\bso\b", text)
        return sum(1 for c in chunks if 0 < len(c.split()) < 4)

    def lexical_density(self, text: str) -> float:
        words = text.lower().split()
        if not words:
            return 0.0
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "need", "dare", "ought", "used", "to", "of", "in", "on",
            "at", "by", "for", "with", "about", "as", "into", "through",
            "during", "before", "after", "above", "below", "from", "up",
            "down", "out", "off", "over", "under", "again", "then",
            "once", "here", "there", "when", "where", "why", "how",
            "all", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "i", "me", "my",
            "myself", "we", "our", "you", "your", "he", "she", "it",
            "they", "them", "their", "this", "that", "these", "those",
            "and", "but", "or", "if", "because", "while", "although",
        }
        content = [w for w in words if w not in stopwords]
        return len(content) / len(words)

    def repetition_ratio(self, text: str) -> float:
        words = [w for w in text.lower().split() if len(w) > 3]
        if not words:
            return 0.0
        unique = len(set(words))
        return 1.0 - (unique / len(words))

    def concept_keywords(self, text: str, domain_keywords: list[str]) -> set[str]:
        clean = text.lower()
        return {kw for kw in domain_keywords if kw in clean}


_normalizer = TextNormalizer()
