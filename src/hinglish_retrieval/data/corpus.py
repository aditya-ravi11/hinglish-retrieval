"""Corpus loaders.

MIRACL Hindi corpus (hiwiki-20220301, 506k passages), full or pooled.
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
