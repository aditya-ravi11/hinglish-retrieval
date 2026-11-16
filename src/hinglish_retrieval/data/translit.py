"""Deterministic machine romanization of Devanagari Hindi.

This is the "one clean spelling" condition that earlier work evaluates on.
It is ITRANS with the marks people never type removed: long vowels lose
their capital letters, anusvara becomes n, and the inherent final "a" is
dropped when a word ends in a bare consonant (kisa -> kis), which is how
Hindi is pronounced and usually typed.
"""

from __future__ import annotations

import re

from indic_transliteration import sanscript

_CONSONANT = re.compile(r"[क-हक़-य़]$")
_KEEP = re.compile(r"[^a-z0-9]")


def _word(w: str) -> str:
    core = w.strip("?!.,;:\"'()[]{}।॥")
    if not core:
        return ""
    t = sanscript.transliterate(core, sanscript.DEVANAGARI, sanscript.ITRANS)
    t = t.replace(".N", "n").replace("~N", "n").replace("~n", "n").replace("M", "n")
    t = t.replace("Sh", "sh").replace("RRi", "ri").replace("R^i", "ri")
    t = _KEEP.sub("", t.lower())
    # final schwa is dropped after a bare consonant (kisa -> kis), but kept
    # after a cluster (yantra, chandra) unless the last letter is ha (chinh)
    cluster = len(core) >= 2 and core[-2] == "्" and core[-1] != "ह"
    # -iya endings keep the a (rashtriya, kriya)
    cluster = cluster or (len(core) >= 2 and core[-1] == "य" and core[-2] in "िी")
    if _CONSONANT.search(core) and not cluster and len(t) > 2 and t.endswith("a"):
        t = t[:-1]
    return t


def machine_romanize(text: str) -> str:
    return " ".join(x for x in (_word(w) for w in text.split()) if x)
