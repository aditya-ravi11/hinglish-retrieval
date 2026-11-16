"""Build the pooled Hindi corpus used by the benchmark.

Pool = every passage judged for any Hindi source query (relevant or not)
     + top-100 from BM25 (native queries)
     + top-100 from multilingual-e5-base and bge-m3 (native and machine-romanized queries)
over the full 506k-passage MIRACL Hindi corpus.

Why pool: scoring 10 models x several query conditions on 506k passages is
slow on a laptop, and MTEB prefers small retrieval tasks. Pooling with
several systems keeps the hard distractors. Pooling with romanized queries
too means distractors that only a Latin-script query would pull in are
also present. Caveat: MIRACL treats unjudged passages as not relevant.

Output: data/corpus/hi_pool_ids.txt (sorted ids, committed).
Usage: python -m hinglish_retrieval.data.build_corpus
"""

from __future__ import annotations

import json
import time

from hinglish_retrieval.data.corpus import POOL_IDS, ROOT, load_hi_full
from hinglish_retrieval.data.miracl import load_qrels
from hinglish_retrieval.data.translit import machine_romanize
from hinglish_retrieval.eval.models import DENSE
from hinglish_retrieval.eval.retrieve import BM25, Dense

K = 100


def main() -> None:
    t0 = time.time()
    sources = [json.loads(l) for l in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8")]
    hi = [s for s in sources if s["lang"] == "hi"]
    native = {s["source_id"]: s["text"] for s in hi}
    roman = {s["source_id"]: machine_romanize(s["text"]) for s in hi}

    qrels = {**load_qrels("hi", "train"), **load_qrels("hi", "dev")}
    pool: set[str] = set()
    for s in hi:
        pool.update(qrels[s["miracl_qid"]].keys())
    print(f"judged passages: {len(pool)}")

    ids, texts = load_hi_full()
    print(f"full corpus: {len(ids)} passages ({time.time() - t0:.0f}s)")

    bm = BM25(ids, texts)
    for run in bm.search(native, k=K).values():
        pool.update(run)
    del bm
    print(f"after BM25: {len(pool)} ({time.time() - t0:.0f}s)")

    for key in ["me5-base", "bge-m3"]:
        d = Dense(DENSE[key])
        d.index("hi-full", ids, texts)
        for qs in (native, roman):
            for run in d.search(qs, k=K).values():
                pool.update(run)
        d.close()
        print(f"after {key}: {len(pool)} ({time.time() - t0:.0f}s)")

    POOL_IDS.parent.mkdir(parents=True, exist_ok=True)
    POOL_IDS.write_text("\n".join(sorted(pool)) + "\n", encoding="utf-8")
    print(f"wrote {len(pool)} ids to {POOL_IDS}")


if __name__ == "__main__":
    main()
