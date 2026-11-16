"""Back-transliteration: Latin-script Hinglish -> Devanagari, before retrieval.

This is the "fix the input, keep the model" baseline: transliterate the query
back to Devanagari, then search with an unchanged retriever. Earlier work on
MIRACL Hindi (Found in Transliteration) used IndicXlit for this, which needs
fairseq; we use sk-community/romanized-hindi-transliterator-60M (MIT, MarianMT,
from the IndoTranslit paper, arXiv 2511.22769), which loads with transformers.

English words inside a Hinglish query get transliterated too ("captain" ->
कैप्टन). That is expected: it is what this pipeline would do in production.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_ID = "sk-community/romanized-hindi-transliterator-60M"


@lru_cache(maxsize=1)
def _load(device: str):
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    tok = MarianTokenizer.from_pretrained(MODEL_ID)
    model = MarianMTModel.from_pretrained(MODEL_ID).to(device).eval()
    return tok, model, torch


def back_transliterate(texts: list[str], device: str = "cuda", batch_size: int = 64) -> list[str]:
    tok, model, torch = _load(device)
    out: list[str] = []
    for i in range(0, len(texts), batch_size):
        batch = [t.lower() for t in texts[i : i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=4, max_new_tokens=160)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out
