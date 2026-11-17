"""Back-transliteration: Latin-script Hinglish -> Devanagari, before retrieval.

This is the "fix the input, keep the model" baseline: turn the query back into
Devanagari, then search with an unchanged retriever. Earlier work on MIRACL
Hindi (Found in Transliteration) used IndicXlit for this, which needs fairseq.

The first choice here was sk-community/romanized-hindi-transliterator-60M
(MarianMT, IndoTranslit paper). It does not work as published: under
transformers 4.51 and 5.22 alike it drops leading words and returns mixed
script ("tum kaisa ho" -> "sa हो", the example from its own model card), and
the sibling indotranslit-multitransliterator-60M behaves the same.

So the step is an instruction-tuned LLM (Qwen3, Apache-2.0) asked to rewrite
the query in Devanagari, greedy decoding. English words inside a Hinglish query
get written in Devanagari too ("captain" -> कैप्टन); that is what this pipeline
would do in production. If an answer has no Devanagari, the query is kept as
typed.

Set HR_BT_LLM to use another model. Outputs are cached per model in
data/cache/backtranslit-<model>.json, so each query is converted once across
runs and roles.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MODEL_ID = os.environ.get("HR_BT_LLM", "Qwen/Qwen3-4B")
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

PROMPT = """The search query below is Hindi typed in Latin letters (Hinglish), possibly with English words mixed in. Write the same query in Hindi using Devanagari script. Transliterate it; do not translate it into other words, answer it, or explain anything. English words that Hindi speakers use can stay as they are said, written in Devanagari.

Reply with the Devanagari query only.

Query: {query}"""

_DEVA = re.compile(r"[ऀ-ॿ]")


def _cache_path() -> Path:
    return CACHE_DIR / f"backtranslit-{MODEL_ID.replace('/', '--')}.json"


def n_params() -> int:
    from hinglish_retrieval.llm import n_params as count

    return count(MODEL_ID)


def clean(answer: str, query: str) -> str:
    line = next((x.strip() for x in answer.splitlines() if x.strip()), "")
    line = re.sub(r"^(query|output|devanagari)\s*[:：]\s*", "", line, flags=re.IGNORECASE).strip("\"'` ")
    if not _DEVA.search(line):
        return query
    # small models sometimes loop ("जान जान जान ..."): keep at most a few words more than typed
    return " ".join(line.split()[: len(query.split()) + 3])


def back_transliterate(texts: list[str], batch_size: int | None = None) -> list[str]:
    from hinglish_retrieval.llm import chat_generate

    out = list(texts)
    prompts = [PROMPT.format(query=t) for t in texts]
    for i, a in chat_generate(MODEL_ID, prompts, max_new_tokens=96, batch_size=batch_size):
        out[i] = clean(a, texts[i])
    return out


def back_transliterate_cached(texts: list[str]) -> list[str]:
    """Same as back_transliterate, reading and extending the on-disk cache."""
    path = _cache_path()
    cache: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = sorted({t for t in texts if t not in cache})
    if todo:
        cache.update(zip(todo, back_transliterate(todo)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    return [cache[t] for t in texts]
