"""Rule-based Hinglish spelling noise for training queries.

Turns a Devanagari query into one informal Latin-script spelling, the way
people type Hindi on a phone. The rules are written by hand from common
chat conventions (kya/kia, hai/h, nahin/nhi, medial vowel dropping, v/w,
ph/f, a few English words swapped in). They are NOT fitted on any human
rewrite: the leakage rule forbids using the human test set for spelling-noise
fitting, and this keeps the dev split clean for model selection too.

Each call takes a random.Random, so a fixed seed gives a fixed spelling.
"""

from __future__ import annotations

import random
import re

from hinglish_retrieval.data.translit import machine_romanize

# Devanagari word -> English word people often type instead (code-mixing).
EN_SWAP = {
    "प्रसिद्ध": "famous",
    "महत्वपूर्ण": "important",
    "राज्य": "state",
    "जिला": "district",
    "ज़िला": "district",
    "शहर": "city",
    "नदी": "river",
    "वर्ष": "year",
    "साल": "year",
    "भारत": "india",
    "स्थित": "located",
    "अर्थ": "meaning",
    "प्रक्रिया": "process",
    "प्रथम": "first",
    "पहला": "first",
    "पहली": "first",
    "राजधानी": "capital",
    "मुख्यमंत्री": "cm",
    "प्रधानमंत्री": "pm",
    "स्थापना": "establishment",
    "जन्म": "birth",
    "मृत्यु": "death",
    "युद्ध": "war",
    "खेल": "game",
    "भाषा": "language",
    "पुस्तक": "book",
    "किताब": "book",
    "लेखक": "writer",
    "फिल्म": "film",
    "फ़िल्म": "film",
    "विश्वविद्यालय": "university",
    "विज्ञान": "science",
    "संख्या": "number",
    "जनसंख्या": "population",
    "क्षेत्र": "area",
    "सरकार": "government",
    "विद्यालय": "school",
    "अस्पताल": "hospital",
    "नाम": "name",
    "उपयोग": "use",
    "इतिहास": "history",
    "देश": "country",
    "दुनिया": "world",
    "विश्व": "world",
    "कंपनी": "company",
    "समय": "time",
    "रंग": "color",
    "शब्द": "word",
    "मंदिर": "temple",
    "संविधान": "constitution",
    "कानून": "law",
    "क़ानून": "law",
    "चुनाव": "election",
    "पार्टी": "party",
    "अध्यक्ष": "president",
    "आविष्कार": "invention",
    "खोज": "discovery",
    "वैज्ञानिक": "scientist",
    "उत्तर": "north",
    "दक्षिण": "south",
    "पूर्व": "east",
    "पश्चिम": "west",
}

# Machine-romanized word -> spellings people type for it (first = machine form).
WORD_VARIANTS = {
    "hai": ["hai", "h", "he", "hae"],
    "hain": ["hain", "hai", "hn", "h"],
    "kya": ["kya", "kia", "kyaa"],
    "kyon": ["kyon", "kyun", "kyu", "kyo"],
    "nahin": ["nahin", "nahi", "nhi", "nai"],
    "men": ["men", "me", "mein", "mai", "main"],
    "main": ["main", "mai", "me"],
    "kaun": ["kaun", "kon", "kaon"],
    "kaunsa": ["kaunsa", "konsa", "kaun sa"],
    "kahan": ["kahan", "kaha", "kahaan"],
    "kaise": ["kaise", "kese", "kaisey", "kaese"],
    "aur": ["aur", "or", "aor"],
    "bhi": ["bhi", "bi"],
    "yah": ["yah", "ye", "yeh", "ya"],
    "vah": ["vah", "woh", "wo", "vo"],
    "dvara": ["dvara", "dwara", "duara"],
    "lie": ["lie", "liye", "liae"],
    "jnan": ["jnan", "gyan", "gyaan"],
}

_VOWELS = "aeiou"
# medial schwa people drop: karana -> karna, sabase -> sabse, pahala -> pahla
_MEDIAL_SCHWA = re.compile(r"([aeiou][^aeiou\s])a([^aeiou\s][aeiou]+)$")
_FINAL_CLUSTER_A = re.compile(r"(?<=[^aeiou](?:dh|th|bh|kh|gh|ph))a$")
_DROP_A = re.compile(r"(?<=[^aeiou\s])a(?=[^aeiou\s])")


def _chars(w: str, rng: random.Random, p: float) -> str:
    """Character-level variation inside one Latin word."""
    if rng.random() < 0.8:
        w = _MEDIAL_SCHWA.sub(r"\1\2", w)
    if rng.random() < 0.7:  # prasiddha -> prasiddh, artha -> arth
        w = _FINAL_CLUSTER_A.sub("", w)
    if "jn" in w and rng.random() < 0.9:
        w = w.replace("jn", "gy")
    if "v" in w and rng.random() < p:
        w = w.replace("v", "w")
    if "ph" in w and rng.random() < p:
        w = w.replace("ph", "f")
    if "sh" in w and rng.random() < p / 3:
        w = w.replace("sh", "s")
    if "ksh" in w and rng.random() < p / 2:
        w = w.replace("ksh", "x")
    if w.endswith("i") and len(w) > 3 and rng.random() < p / 2:
        w = w[:-1] + "ee"
    if "u" in w and len(w) > 3 and rng.random() < p / 3:
        w = w.replace("u", "oo", 1)
    if len(w) > 4 and rng.random() < p / 2:  # SMS style: drop an inner a
        w = _DROP_A.sub("", w, count=1)
    if w.endswith("a") and len(w) > 3 and rng.random() < p / 4:
        w += "a"
    return w


def hinglish_noisy(native: str, rng: random.Random, p: float = 0.3, p_en: float = 0.25) -> str:
    """One noisy Latin-script spelling of a Devanagari query.

    p: chance of each character rule firing; p_en: chance of swapping a word
    from EN_SWAP for its English equivalent.
    """
    out = []
    for t in native.split():
        core = t.strip("?!.,;:\"'()[]{}।॥")
        if core in EN_SWAP and rng.random() < p_en:
            out.append(EN_SWAP[core])  # English words keep their spelling
            continue
        w = machine_romanize(t)
        if not w:
            continue
        if w in WORD_VARIANTS and rng.random() < 0.7:
            w = rng.choice(WORD_VARIANTS[w])
        elif any(c in _VOWELS for c in w):
            w = _chars(w, rng, p)
        out.append(w)
    return " ".join(out)
