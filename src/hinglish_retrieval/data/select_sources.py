"""Pick the MIRACL queries used for evaluation, and the training pool.

Roles (fixed seed, written once and committed):
  test     1,000 Hindi queries: all 349 unique MIRACL-hi dev texts + 651 from train
  dev        150 Hindi queries sampled from train (model selection only)
  test_en    150 English queries sampled from MIRACL-en dev (planned code-mixed
             subset; not scored since human collection was dropped)
  train pool: every remaining MIRACL-hi train query, the only MIRACL queries
             training may use

Why all of MIRACL-hi dev goes to test: some baselines (bge-m3, multilingual-e5)
list MIRACL train in their fine-tuning data, so dev-origin queries are the
cleaner half. Each source keeps `origin_split` so results can be broken down.

Usage: python -m hinglish_retrieval.data.select_sources
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from hinglish_retrieval.data.miracl import load_qrels, load_topics
from hinglish_retrieval.leakage import near_duplicates, normalise

SEED = 20261112
N_TEST_FROM_TRAIN = 651  # 349 unique dev-origin texts + 651 = 1,000
N_DEV = 150
N_TEST_EN = 150

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "data" / "sources"
FORM_COPY = ROOT / "form" / "sources.jsonl"


def _with_positive(topics: dict[str, str], qrels: dict[str, dict[str, int]]) -> list[str]:
    """Query ids that have at least one relevant passage, sorted for determinism."""
    return sorted(q for q in topics if any(r > 0 for r in qrels.get(q, {}).values()))


def select() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)

    hi_dev_t, hi_dev_q = load_topics("hi", "dev"), load_qrels("hi", "dev")
    hi_tr_t, hi_tr_q = load_topics("hi", "train"), load_qrels("hi", "train")
    en_dev_t, en_dev_q = load_topics("en", "dev"), load_qrels("en", "dev")

    hi_dev_ids = _with_positive(hi_dev_t, hi_dev_q)
    hi_tr_ids = _with_positive(hi_tr_t, hi_tr_q)
    en_dev_ids = _with_positive(en_dev_t, en_dev_q)

    # MIRACL repeats some question texts under different ids. Keep one id per
    # normalised text (dev-origin first) so no text sits in two roles.
    seen: set[str] = set()

    def first_of_text(ids: list[str], topics: dict[str, str]) -> list[str]:
        kept = []
        for q in ids:
            n = normalise(topics[q])
            if n not in seen:
                seen.add(n)
                kept.append(q)
        return kept

    hi_dev_ids = first_of_text(hi_dev_ids, hi_dev_t)
    hi_tr_ids = first_of_text(hi_tr_ids, hi_tr_t)

    rng.shuffle(hi_tr_ids)
    test_from_train = hi_tr_ids[:N_TEST_FROM_TRAIN]
    dev_ids = hi_tr_ids[N_TEST_FROM_TRAIN : N_TEST_FROM_TRAIN + N_DEV]
    # duplicates dropped above never reach the training pool either
    pool_ids = hi_tr_ids[N_TEST_FROM_TRAIN + N_DEV :]
    en_ids = rng.sample(en_dev_ids, N_TEST_EN)

    sources: list[dict] = []

    def add(qid: str, text: str, lang: str, origin: str, role: str) -> None:
        sources.append(
            {
                "source_id": f"{lang}-{origin}-{qid}",
                "miracl_qid": qid,
                "lang": lang,
                "origin_split": origin,
                "role": role,
                "text": text,
            }
        )

    for q in hi_dev_ids:
        add(q, hi_dev_t[q], "hi", "dev", "test")
    for q in test_from_train:
        add(q, hi_tr_t[q], "hi", "train", "test")
    for q in dev_ids:
        add(q, hi_tr_t[q], "hi", "train", "dev")
    for q in en_ids:
        add(q, en_dev_t[q], "en", "dev", "test_en")

    # Priority: one fixed random order over all sources. It was the order of the
    # retired collection form; kept so the committed sources file is reproducible.
    order = list(range(len(sources)))
    rng.shuffle(order)
    for rank, i in enumerate(order):
        sources[i]["priority"] = rank

    # Conservative: also drop pool queries that are near paraphrases of an
    # evaluation query (char-5-gram Jaccard >= 0.6), since they tend to share
    # relevant passages.
    ev_norm = {s["source_id"]: normalise(s["text"]) for s in sources}
    pool_norm = {q: normalise(hi_tr_t[q]) for q in pool_ids}
    near = {a for a, _, _ in near_duplicates(pool_norm, ev_norm, threshold=0.6)}
    pool = [
        {"miracl_qid": q, "lang": "hi", "origin_split": "train", "text": hi_tr_t[q]}
        for q in sorted(pool_ids)
        if q not in near
    ]
    return sources, pool


def main() -> None:
    sources, pool = select()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in [
        (OUT_DIR / "sources.jsonl", sources),
        (FORM_COPY, sources),
        (OUT_DIR / "train_pool_hi.jsonl", pool),
    ]:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    roles: dict[str, int] = {}
    for s in sources:
        roles[s["role"]] = roles.get(s["role"], 0) + 1
    print(f"sources: {len(sources)} {roles}; train pool: {len(pool)}")


if __name__ == "__main__":
    main()
