"""One command for the full benchmark table.

    python scripts/reproduce.py [--models bm25,me5-small,...] [--role test]

Scores every baseline on every query condition that exists:
  native      Devanagari MIRACL query
  machine     deterministic machine romanization (one clean spelling)
  human_w1/2  human Hinglish rewrites, writer slot 1 and 2 (only sources
              with both slots; appears once rewrites are collected)
  en_native / en_human_w1 / en_human_w2 for the English subset

Writes results/benchmark.json (all per-query scores) and results/table.md.
Every number in the README must come from these files.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglish_retrieval.data.corpus import load_en_hn, load_hi_pooled  # noqa: E402
from hinglish_retrieval.data.miracl import load_qrels  # noqa: E402
from hinglish_retrieval.data.rewrites import load_rows, rewrite_text, writer_slots  # noqa: E402
from hinglish_retrieval.data.translit import machine_romanize  # noqa: E402
from hinglish_retrieval.eval.metrics import mean, ndcg_at_k, paired_bootstrap, recall_at_k  # noqa: E402
from hinglish_retrieval.eval.models import DENSE  # noqa: E402
from hinglish_retrieval.eval.retrieve import BM25, Dense  # noqa: E402

ALL_MODELS = ["bm25", *DENSE]


def build_conditions(role: str):
    sources = [json.loads(l) for l in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8")]
    slots = writer_slots(load_rows())
    hi = [s for s in sources if s["lang"] == "hi" and s["role"] == role]
    en = [s for s in sources if s["lang"] == "en"] if role == "test" else []

    hi_q = {**load_qrels("hi", "train"), **load_qrels("hi", "dev")}
    conds: dict[str, tuple[str, dict[str, str]]] = {}  # name -> (corpus, {source_id: text})
    conds["native"] = ("hi", {s["source_id"]: s["text"] for s in hi})
    conds["machine"] = ("hi", {s["source_id"]: machine_romanize(s["text"]) for s in hi})
    # human_w1: every source with at least one human rewrite (first writer).
    # human_w2: sources that also have a second, different writer.
    def human(prefix: str, group: list[dict], corpus: str) -> None:
        for w in (0, 1):
            have = [s for s in group if len(slots.get(s["source_id"], [])) > w]
            if have:
                conds[f"{prefix}human_w{w + 1}"] = (
                    corpus,
                    {s["source_id"]: rewrite_text(slots[s["source_id"]][w]) for s in have},
                )

    human("", hi, "hi")
    if en:
        conds["en_native"] = ("en", {s["source_id"]: s["text"] for s in en})
        human("en_", en, "en")
    qrels = {s["source_id"]: hi_q[s["miracl_qid"]] for s in hi}
    origin = {s["source_id"]: s["origin_split"] for s in hi + en}
    return conds, qrels, en, origin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--role", default="test", choices=["test", "dev"])
    args = ap.parse_args()
    models = args.models.split(",")

    conds, qrels, en_sources, origin = build_conditions(args.role)
    hi_ids, hi_texts = load_hi_pooled()
    corpora = {"hi": (hi_ids, hi_texts)}
    if en_sources:
        en_ids, en_texts, en_q = load_en_hn()
        corpora["en"] = (en_ids, en_texts)
        qrels.update({s["source_id"]: en_q[s["miracl_qid"]] for s in en_sources})

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": args.role,
        "machine": platform.platform(),
        "corpus_sizes": {k: len(v[0]) for k, v in corpora.items()},
        "conditions": {c: len(q) for c, (_, q) in conds.items()},
        "models": {},
    }
    for m in models:
        t0 = time.time()
        res: dict = {"per_query": {}}
        dense = None
        try:
            if m != "bm25":
                dense = Dense(DENSE[m])
                res["params"] = dense.n_params
            for name, (ids, texts) in corpora.items():
                if dense is None:
                    retriever = BM25(ids, texts)
                else:
                    dense.index(f"{name}-pooled", ids, texts)  # one index at a time
                    retriever = dense
                for c, (corp, qs) in conds.items():
                    if corp == name:
                        res["per_query"][c] = _score(retriever.search(qs), qrels)
        except Exception as e:  # e.g. gated model without login
            res["error"] = repr(e)
            print(f"{m}: FAILED {e!r}")
        finally:
            if dense is not None:
                dense.close()
        res["seconds"] = round(time.time() - t0, 1)
        out["models"][m] = res
        print(f"{m}: done in {res['seconds']}s")

    (ROOT / "results").mkdir(exist_ok=True)
    json_path = ROOT / "results" / f"benchmark_{args.role}.json"
    # A run over a subset of models keeps the other models' rows, but only
    # when the query conditions and corpora are identical to the saved run.
    if json_path.exists() and set(models) != set(ALL_MODELS):
        prev = json.loads(json_path.read_text(encoding="utf-8"))
        if prev["conditions"] == out["conditions"] and prev["corpus_sizes"] == out["corpus_sizes"]:
            merged = {**prev["models"], **out["models"]}
            out["models"] = {m: merged[m] for m in ALL_MODELS if m in merged}
        else:
            print("conditions changed since the saved run: rerun all models for a full table")
    json_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    md = render(out, origin)
    (ROOT / "results" / f"table_{args.role}.md").write_text(md, encoding="utf-8")
    print(md)


def _score(run, qrels):
    sub = {q: qrels[q] for q in run}
    nd = ndcg_at_k(run, sub, 10)
    rc = recall_at_k(run, sub, 100)
    return {q: {"ndcg10": nd[q], "recall100": rc[q]} for q in nd}


def human_scores(nd: dict[str, dict[str, float]], prefix: str = "") -> dict[str, float]:
    """Per query, the mean over the human writers available for it."""
    w1, w2 = nd.get(f"{prefix}human_w1", {}), nd.get(f"{prefix}human_w2", {})
    return {q: (v + w2[q]) / 2 if q in w2 else v for q, v in w1.items()}


def _ci(g) -> str:
    return f"{100 * g[0]:.1f} [{100 * g[1]:.1f}, {100 * g[2]:.1f}]"


def render(out: dict, origin: dict[str, str]) -> str:
    conds = list(out["conditions"])
    lines = [
        f"Role: {out['role']}. Generated {out['generated_at']} by scripts/reproduce.py.",
        f"Corpus sizes: {out['corpus_sizes']}. Queries per condition: {out['conditions']}.",
        "",
        "nDCG@10 (x100), each condition over its own queries. Gap = native minus condition,",
        "mean with a 95% paired bootstrap CI over the queries both conditions share.",
        "",
        "| model | params (M) | " + " | ".join(conds) + " | gap native-machine |",
        "|---|---|" + "---|" * len(conds) + "---|",
    ]
    for m, r in out["models"].items():
        if "error" in r:
            lines.append(f"| {m} | | failed: {r['error'][:60]} |")
            continue
        nd = {c: {q: v["ndcg10"] for q, v in pq.items()} for c, pq in r["per_query"].items()}
        cells = [f"{100 * mean(nd[c]):.1f}" if c in nd else "" for c in conds]
        lines.append(
            f"| {m} | {r.get('params', 0) / 1e6:.0f} | " + " | ".join(cells)
            + f" | {_ci(paired_bootstrap(nd['native'], nd['machine']))} |"
        )

    if "human_w1" in conds:
        n = out["conditions"]["human_w1"]
        lines += [
            "",
            f"Human Hinglish subset: the {n} queries with at least one human rewrite. Native, machine and",
            "human are scored on these same queries. Human = mean over the writers each query has.",
            "",
            "| model | native | machine | human | gap native-human | gap machine-human |",
            "|---|---|---|---|---|---|",
        ]
        for m, r in out["models"].items():
            if "error" in r:
                continue
            nd = {c: {q: v["ndcg10"] for q, v in pq.items()} for c, pq in r["per_query"].items()}
            hu = human_scores(nd)
            nat = {q: nd["native"][q] for q in hu}
            mac = {q: nd["machine"][q] for q in hu}
            lines.append(
                f"| {m} | {100 * mean(nat):.1f} | {100 * mean(mac):.1f} | {100 * mean(hu):.1f} | "
                f"{_ci(paired_bootstrap(nat, hu))} | {_ci(paired_bootstrap(mac, hu))} |"
            )

    lines += ["", "Native nDCG@10 (x100) by MIRACL origin split of the query (contamination check):", ""]
    lines += ["| model | dev-origin | train-origin |", "|---|---|---|"]
    for m, r in out["models"].items():
        if "error" in r:
            continue
        nat = r["per_query"]["native"]
        parts = []
        for o in ("dev", "train"):
            v = [x["ndcg10"] for q, x in nat.items() if origin.get(q) == o]
            parts.append(f"{100 * sum(v) / len(v):.1f} (n={len(v)})" if v else "")
        lines.append(f"| {m} | {parts[0]} | {parts[1]} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
