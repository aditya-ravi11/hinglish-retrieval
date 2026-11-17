"""Prove that no training query overlaps any test or dev query.

Two modes:

  python scripts/check_leakage.py --full
      Needs the local data. Loads every evaluation text (MIRACL source queries
      of all roles, and the ~20,000 Dakshina sentences, native and
      human-romanized) and every training text (MIRACL-hi train pool,
      synthetic native queries and their Hinglish versions in
      data/train/*.jsonl). Checks:
        1. no exact match after normalisation (MIRACL and Dakshina)
        2. no char-5-gram Jaccard >= 0.8 between normalised strings (MIRACL
           queries; Dakshina is too large for the pairwise check and its
           sentences are not questions, so it gets checks 1 and 4)
        3. no training query was generated from a passage judged for an
           evaluation query
        4. no seed passage contains a Dakshina sentence
      Then rewrites the hash manifests in data/manifests/ and writes
      results/leakage_report.json.

  python scripts/check_leakage.py
      CI mode. Uses only the committed manifests: asserts the training and
      evaluation hash sets are disjoint and the seed-passage set does not
      touch the judged-passage set. Exits 1 on any overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglish_retrieval.leakage import digest, near_duplicates, normalise  # noqa: E402

MAN = ROOT / "data" / "manifests"
FILES = {
    "eval_hashes": MAN / "eval_query_hashes.txt",
    "train_hashes": MAN / "train_query_hashes.txt",
    "judged": MAN / "eval_judged_passages.txt",
    "seeds": MAN / "train_seed_passages.txt",
}


def _jsonl(p: Path):
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def collect_full():
    from hinglish_retrieval.data import dakshina
    from hinglish_retrieval.data.miracl import load_qrels

    sources = _jsonl(ROOT / "data/sources/sources.jsonl")
    ev: dict[str, str] = {s["source_id"]: s["text"] for s in sources}
    dk: dict[str, str] = {}
    for r in dakshina.load():
        dk[f"{r['id']}:native"] = r["native"]
        dk[f"{r['id']}:human"] = r["human"]

    tr: dict[str, str] = {}
    seeds: set[str] = set()
    for r in _jsonl(ROOT / "data/sources/train_pool_hi.jsonl"):
        tr[f"pool:{r['miracl_qid']}"] = r["text"]
    for p in sorted((ROOT / "data" / "train").glob("*.jsonl")):
        for r in _jsonl(p):
            qid = r["qid"]
            tr[f"{p.stem}:{qid}:native"] = r["text_native"]
            for k, h in enumerate(r.get("text_hinglish", [])):
                tr[f"{p.stem}:{qid}:hing{k}"] = h
            if r.get("seed_passage_id"):
                seeds.add(r["seed_passage_id"])

    qrels = {**load_qrels("hi", "train"), **load_qrels("hi", "dev")}
    en_qrels = load_qrels("en", "dev")
    judged: set[str] = set()
    for s in sources:
        q = (qrels if s["lang"] == "hi" else en_qrels)[s["miracl_qid"]]
        judged.update(q)
    return ev, dk, tr, judged, seeds


def seeds_with_dakshina(seeds: set[str]) -> list[str]:
    if not seeds:
        return []
    from hinglish_retrieval.data.corpus import load_hi_full
    from hinglish_retrieval.data.dakshina import native_keys, sentence_keys

    keys = native_keys()
    ids, texts = load_hi_full()
    return sorted(d for d, t in zip(ids, texts) if d in seeds and sentence_keys(t) & keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    if args.full:
        ev, dk, tr, judged, seeds = collect_full()
        ev_n = {k: normalise(v) for k, v in ev.items()}
        dk_n = {k: normalise(v) for k, v in dk.items()}
        tr_n = {k: normalise(v) for k, v in tr.items()}
        ev_by = {}
        for k, v in {**ev_n, **dk_n}.items():
            ev_by.setdefault(v, k)
        exact = [(k, ev_by[v]) for k, v in tr_n.items() if v in ev_by]
        review = near_duplicates(tr_n, ev_n, threshold=0.6)
        near = [h for h in review if h[2] >= 0.8]  # fails the check
        review = [(a, b, j, tr[a], ev[b]) for a, b, j in review if j < 0.8]  # for a human to read
        seed_hits = sorted(seeds & judged)
        seed_dk = seeds_with_dakshina(seeds)
        report = {
            "eval_texts": len(ev),
            "dakshina_texts": len(dk),
            "train_texts": len(tr),
            "judged_passages": len(judged),
            "seed_passages": len(seeds),
            "exact_overlaps": exact[:50],
            "n_exact": len(exact),
            "near_duplicates_jaccard_0.8": near[:50],
            "n_near": len(near),
            "n_review_0.6_to_0.8": len(review),
            "review_0.6_to_0.8": review[:200],
            "seed_passage_overlaps": seed_hits[:50],
            "n_seed_overlaps": len(seed_hits),
            "seeds_containing_dakshina": seed_dk[:50],
            "n_seed_dakshina": len(seed_dk),
        }
        MAN.mkdir(parents=True, exist_ok=True)
        FILES["eval_hashes"].write_text("\n".join(sorted({digest(v) for v in {**ev_n, **dk_n}.values()})) + "\n")
        FILES["train_hashes"].write_text("\n".join(sorted({digest(v) for v in tr_n.values()})) + "\n")
        FILES["judged"].write_text("\n".join(sorted(judged)) + "\n")
        FILES["seeds"].write_text("\n".join(sorted(seeds)) + ("\n" if seeds else ""))
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / "leakage_report.json").write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in report.items() if k.startswith("n_") or k.endswith("s")}, indent=1))
        bad = report["n_exact"] or report["n_near"] or report["n_seed_overlaps"] or report["n_seed_dakshina"]
    else:
        read = {k: set(p.read_text().split()) for k, p in FILES.items()}
        e = read["eval_hashes"] & read["train_hashes"]
        s = read["judged"] & read["seeds"]
        print(
            f"eval hashes {len(read['eval_hashes'])}, train hashes {len(read['train_hashes'])}, "
            f"overlap {len(e)}; judged {len(read['judged'])}, seeds {len(read['seeds'])}, overlap {len(s)}"
        )
        bad = bool(e or s)
    if bad:
        print("LEAKAGE FOUND")
        sys.exit(1)
    print("OK: no overlap")


if __name__ == "__main__":
    main()
