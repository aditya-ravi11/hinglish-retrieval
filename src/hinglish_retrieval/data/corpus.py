"""Corpus loaders.

hi: MIRACL Hindi corpus (hiwiki-20220301, 506k passages), full or pooled.
en: MTEB MIRACLRetrievalHardNegatives English dev corpus (178k passages),
    which already covers every English source query we use.
Passage text is "title text", the same join MTEB uses.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
POOL_IDS = ROOT / "data" / "corpus" / "hi_pool_ids.txt"


def load_hi_full() -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for part in sorted((RAW / "miracl-corpus" / "miracl-corpus-v1.0-hi").glob("docs-*.jsonl.gz")):
        with gzip.open(part, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ids.append(r["docid"])
                texts.append(f"{r['title']} {r['text']}".strip())
    return ids, texts


def load_hi_pooled() -> tuple[list[str], list[str]]:
    keep = set(POOL_IDS.read_text(encoding="utf-8").split())
    ids, texts = load_hi_full()
    pairs = [(i, t) for i, t in zip(ids, texts) if i in keep]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def load_en_hn() -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    import pandas as pd

    base = RAW / "mteb-miracl-hn"
    c = pd.read_parquet(base / "en-corpus" / "dev-00000-of-00001.parquet")
    r = pd.read_parquet(base / "en-qrels" / "dev-00000-of-00001.parquet")
    texts = [f"{t} {x}".strip() for t, x in zip(c["title"].fillna(""), c["text"])]
    qrels: dict[str, dict[str, int]] = {}
    for q, d, s in zip(r["query-id"].astype(str), r["corpus-id"].astype(str), r["score"]):
        qrels.setdefault(q, {})[d] = int(s)
    return c["id"].astype(str).tolist(), texts, qrels
