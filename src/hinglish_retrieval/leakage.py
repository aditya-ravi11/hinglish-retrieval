"""Text normalisation and overlap checks between training and evaluation queries.

normalise(): NFKC, Devanagari mapped to the machine romanization (so a
Hindi query and its Latin version compare equal), lowercase, keep only
a-z0-9 and spaces, collapse runs of the same letter ("kyaaa" -> "kya").
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

from hinglish_retrieval.data.translit import machine_romanize

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_NON = re.compile(r"[^a-z0-9 ]+")
_REPEAT = re.compile(r"(.)\1+")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    if _DEVANAGARI.search(t):
        t = machine_romanize(t)
    t = _NON.sub(" ", t.lower())
    t = _REPEAT.sub(r"\1", t)
    return _SPACE.sub(" ", t).strip()


def digest(norm: str) -> str:
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def grams(s: str, n: int = 5) -> set[str]:
    s = f" {s} "
    return {s[i : i + n] for i in range(max(1, len(s) - n + 1))}


def near_duplicates(train: dict[str, str], test: dict[str, str], threshold: float = 0.8):
    """Pairs (train_id, test_id, jaccard) of normalised strings with char-5-gram Jaccard >= threshold."""
    index: dict[str, list[str]] = defaultdict(list)
    test_grams = {}
    for tid, s in test.items():
        g = grams(s)
        test_grams[tid] = g
        for x in g:
            index[x].append(tid)
    hits = []
    for rid, s in train.items():
        g = grams(s)
        shared: dict[str, int] = defaultdict(int)
        for x in g:
            for tid in index.get(x, ()):
                shared[tid] += 1
        for tid, k in shared.items():
            j = k / (len(g) + len(test_grams[tid]) - k)
            if j >= threshold:
                hits.append((rid, tid, round(j, 3)))
    return hits
