"""One command for the full benchmark table.

    python scripts/reproduce.py [--models bm25,me5-small,...] [--role test]

Two tasks; every model is scored on every query condition:
  MIRACL Hindi retrieval (pooled hiwiki corpus, MIRACL relevance labels)
    native      the Devanagari MIRACL question
    machine     our deterministic machine romanization of it (one clean spelling)
  Dakshina sentence retrieval (find the native sentence among ~10,000)
    dk_human    the romanization a native speaker typed (Google Dakshina)
    dk_machine  our machine romanization of the same sentence

Models: BM25, the dense baselines, three back-transliteration pipelines
(bt-*: Latin-script Hindi queries are turned back into Devanagari, then an
unchanged retriever searches) and the fine-tuned models in models/. The dev
role scores every fine-tuned model (for selection); the test role scores only
the one named in models/SELECTED.

Writes results/benchmark_<role>.json (all per-query scores) and
results/table_<role>.md. Every number in the README must come from these files.

--corpus-limit N keeps the relevant items plus N others per corpus: a quick end-to-end
check that writes results/smoke_<role>.* instead and never touches the real
results.
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

from hinglish_retrieval.data import dakshina  # noqa: E402
from hinglish_retrieval.data.corpus import load_hi_pooled  # noqa: E402
from hinglish_retrieval.data.miracl import load_qrels  # noqa: E402
from hinglish_retrieval.data.translit import machine_romanize  # noqa: E402
from hinglish_retrieval.eval.metrics import mean, ndcg_at_k, paired_bootstrap, recall_at_k  # noqa: E402
from hinglish_retrieval.eval.models import BACKTRANSLIT, DENSE, trained  # noqa: E402
from hinglish_retrieval.eval.retrieve import BM25, Dense  # noqa: E402

ALL_MODELS = ["bm25", *DENSE, *BACKTRANSLIT]  # baselines; fine-tuned models are added per role


def model_list(role: str) -> list[str]:
    return [*ALL_MODELS, *trained(role)]


def backtranslit_conditions(conds):
    """The same conditions with every Latin-script query back-transliterated.

    Native queries are already Devanagari and pass through unchanged.
    """
    from hinglish_retrieval.data.backtranslit import back_transliterate_cached, n_params
    from hinglish_retrieval.llm import unload

    out = {}
    for c, (corp, qs) in conds.items():
        if c != "native":
            ids = list(qs)
            out[c] = (corp, dict(zip(ids, back_transliterate_cached([qs[i] for i in ids]))))
        else:
            out[c] = (corp, qs)
    unload()  # free the GPU for the retrievers
    return out, n_params()


def build_conditions(role: str):
    sources = [json.loads(l) for l in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8")]
    hi = [s for s in sources if s["lang"] == "hi" and s["role"] == role]
    hi_q = {**load_qrels("hi", "train"), **load_qrels("hi", "dev")}
    conds: dict[str, tuple[str, dict[str, str]]] = {}  # name -> (corpus, {query_id: text})
    conds["native"] = ("hi", {s["source_id"]: s["text"] for s in hi})
    conds["machine"] = ("hi", {s["source_id"]: machine_romanize(s["text"]) for s in hi})
    qrels = {s["source_id"]: hi_q[s["miracl_qid"]] for s in hi}
    origin = {s["source_id"]: s["origin_split"] for s in hi}

    dk = dakshina.split(role)  # the query id is the id of its own native sentence
    conds["dk_machine"] = ("dk", {r["id"]: machine_romanize(r["native"]) for r in dk})
    conds["dk_human"] = ("dk", {r["id"]: r["human"] for r in dk})
    qrels.update({r["id"]: {r["id"]: 1} for r in dk})
    return conds, qrels, origin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated; default: every baseline and fine-tuned model")
    ap.add_argument("--role", default="test", choices=["test", "dev"])
    ap.add_argument("--corpus-limit", type=int, help="smoke test: judged passages plus this many others")
    args = ap.parse_args()
    every = model_list(args.role)
    models = args.models.split(",") if args.models else every
    specs = {**DENSE, **trained(args.role if not args.models else None)}
    unknown = [m for m in models if BACKTRANSLIT.get(m, m) not in {"bm25", *specs}]
    if unknown:
        sys.exit(f"unknown models: {unknown}. Known: {every}")

    conds, qrels, origin = build_conditions(args.role)
    hi_ids, hi_texts = load_hi_pooled()
    dk_rows = dakshina.load()
    corpora = {
        "hi": (hi_ids, hi_texts),
        "dk": ([r["id"] for r in dk_rows], [r["native"] for r in dk_rows]),
    }
    if args.corpus_limit is not None:
        judged = {d for q in qrels.values() for d in q}
        for name, (ids, texts) in list(corpora.items()):
            keep = [i for i, d in enumerate(ids) if d in judged]
            keep += [i for i, d in enumerate(ids) if d not in judged][: args.corpus_limit]
            keep.sort()
            corpora[name] = ([ids[i] for i in keep], [texts[i] for i in keep])
    bt_conds, bt_params = backtranslit_conditions(conds) if any(m in BACKTRANSLIT for m in models) else (None, 0)

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
        base = BACKTRANSLIT.get(m, m)
        queries = bt_conds if m in BACKTRANSLIT else conds
        try:
            if base != "bm25":
                dense = Dense(specs[base])
                res["params"] = dense.n_params
            if m in BACKTRANSLIT:
                res["params"] = res.get("params", 0) + bt_params
            for name, (ids, texts) in corpora.items():
                if dense is None:
                    retriever = BM25(ids, texts)
                else:
                    tag = "pooled" if args.corpus_limit is None else f"smoke{args.corpus_limit}"
                    dense.index(f"{name}-{tag}", ids, texts)  # one index at a time
                    retriever = dense
                for c, (corp, qs) in queries.items():
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
    smoke = args.corpus_limit is not None
    json_path = ROOT / "results" / (f"smoke_{args.role}.json" if smoke else f"benchmark_{args.role}.json")
    # A run over a subset of models keeps the other models' rows, but only
    # when the query conditions and corpora are identical to the saved run.
    # Rows of models that are no longer listed (e.g. a fine-tuned model that
    # was not selected) are dropped.
    if json_path.exists() and set(models) != set(every):
        prev = json.loads(json_path.read_text(encoding="utf-8"))
        if prev["conditions"] == out["conditions"] and prev["corpus_sizes"] == out["corpus_sizes"]:
            merged = {**prev["models"], **out["models"]}
            order = every + [m for m in models if m not in every]
            out["models"] = {m: merged[m] for m in order if m in merged}
        else:
            print("conditions changed since the saved run: rerun all models for a full table")
    json_path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    md = render(out, origin)
    md_name = f"smoke_{args.role}.md" if smoke else f"table_{args.role}.md"
    (ROOT / "results" / md_name).write_text(md, encoding="utf-8")
    print(md)


def _score(run, qrels):
    sub = {q: qrels[q] for q in run}
    nd = ndcg_at_k(run, sub, 10)
    rc = recall_at_k(run, sub, 100)
    return {q: {"ndcg10": nd[q], "recall100": rc[q]} for q in nd}


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

    if {"dk_human", "dk_machine"} <= set(conds):
        lines += [
            "",
            f"Dakshina: the same {out['conditions']['dk_human']} sentences typed by a native speaker (human) and",
            "romanized by rule (machine); the system must find the Devanagari sentence among",
            f"{out['corpus_sizes']['dk']}. Gap = machine minus human, 95% paired bootstrap CI.",
            "",
            "| model | machine | human | gap machine-human |",
            "|---|---|---|---|",
        ]
        for m, r in out["models"].items():
            if "error" in r:
                continue
            nd = {c: {q: v["ndcg10"] for q, v in pq.items()} for c, pq in r["per_query"].items()}
            mac, hu = nd["dk_machine"], nd["dk_human"]
            lines.append(
                f"| {m} | {100 * mean(mac):.1f} | {100 * mean(hu):.1f} | {_ci(paired_bootstrap(mac, hu))} |"
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
