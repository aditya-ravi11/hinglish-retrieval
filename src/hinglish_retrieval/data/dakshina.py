"""Google Dakshina: Hindi Wikipedia sentences romanized by native speakers.

Dakshina (Roark et al., LREC 2020; CC BY-SA 4.0) asked native speakers to type
10,000 Hindi Wikipedia sentences in Latin script, with no fixed spelling. That
gives real human romanizations ("Jabki yah Jainon se km hai.") paired with the
native sentence, which is what this benchmark needs and could not collect.

Task built from it: the query is the human romanization (or, for comparison,
our machine romanization of the same sentence) and the system must find the
native sentence among all 10,000. One relevant item per query, so nDCG@10 is a
rank-discounted hit rate. Dakshina is used for evaluation only, never for
training or spelling-noise fitting.

Splits (fixed): sentences are ordered by the SHA-1 of their native text; the
first 1,000 are dev queries (model selection), the next 4,000 are test
queries. The corpus is all ~10,000 native sentences, the same for both roles.

Source: hi/romanized/hi.romanized.rejoined.tsv of the official release
(storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar, 1.9 GB
for all languages), read here from a Parquet copy of the same rows on Hugging
Face, pinned to one revision.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

from hinglish_retrieval.leakage import normalise

REPO = "Anvesh-Lankala/Copy_Dakshina_Google_research_dataset"
REVISION = "765309607a263b86d25a5bd1a4fa20ef34a87a85"
FILE = "data/hi-00000-of-00001.parquet"
RAW = Path(__file__).resolve().parents[3] / "data" / "raw" / "dakshina"
N_DEV, N_TEST = 1000, 4000

_DEVA = re.compile(r"[ऀ-ॿ]")
_SENT = re.compile(r"[।॥.!?]+")


def order(natives: list[str]) -> list[int]:
    """Indices sorted by the SHA-1 of the native text: the fixed split order."""
    return sorted(range(len(natives)), key=lambda i: hashlib.sha1(natives[i].encode("utf-8")).hexdigest())


@lru_cache(maxsize=1)
def load() -> list[dict]:
    """[{id, native, human}], one row per distinct native sentence, in split order."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION, local_dir=RAW)
    df = pd.read_parquet(p, columns=["native", "romanized"])
    seen, rows = set(), []
    for nat, rom in zip(df["native"], df["romanized"]):
        nat, rom = str(nat).strip(), str(rom).strip()
        if not nat or not rom or nat in seen or _DEVA.search(rom) or not _DEVA.search(nat):
            continue
        seen.add(nat)
        rows.append({"native": nat, "human": rom})
    rows = [rows[i] for i in order([r["native"] for r in rows])]
    for k, r in enumerate(rows):
        r["id"] = f"dk-{k:05d}"
    return rows


def split(role: str) -> list[dict]:
    rows = load()
    return rows[:N_DEV] if role == "dev" else rows[N_DEV : N_DEV + N_TEST]


def sentence_keys(text: str) -> set[str]:
    """Normalised sentences of a passage, to spot Dakshina sentences inside it."""
    return {k for k in (normalise(s) for s in _SENT.split(text)) if len(k) > 20}


def native_keys() -> set[str]:
    return {k for r in load() for k in sentence_keys(r["native"])}
