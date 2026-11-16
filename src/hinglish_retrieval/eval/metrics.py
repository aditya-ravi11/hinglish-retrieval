"""Retrieval metrics, computed per query so we can bootstrap over queries.

nDCG uses linear gain (rel / log2(rank + 1)), the same as pytrec_eval's
ndcg_cut, which MTEB uses. Ties in score are broken by doc id so runs are
deterministic. Queries without any relevant passage are skipped.
"""

from __future__ import annotations

import math

import numpy as np

Run = dict[str, dict[str, float]]  # qid -> {doc_id: score}
Qrels = dict[str, dict[str, int]]  # qid -> {doc_id: relevance}


def _ranked(scores: dict[str, float]) -> list[str]:
    return [d for d, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def ndcg_at_k(run: Run, qrels: Qrels, k: int = 10) -> dict[str, float]:
    out = {}
    for qid, rels in qrels.items():
        ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:k]
        if not ideal:
            continue
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        ranked = _ranked(run.get(qid, {}))[:k]
        dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked))
        out[qid] = dcg / idcg
    return out


def recall_at_k(run: Run, qrels: Qrels, k: int = 100) -> dict[str, float]:
    out = {}
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = set(_ranked(run.get(qid, {}))[:k])
        out[qid] = len(relevant & ranked) / len(relevant)
    return out


def mean(per_query: dict[str, float]) -> float:
    return float(np.mean(list(per_query.values()))) if per_query else float("nan")


def paired_bootstrap(
    a: dict[str, float], b: dict[str, float], n: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) over shared queries with a 95% percentile interval."""
    qids = sorted(set(a) & set(b))
    diff = np.array([a[q] - b[q] for q in qids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    boots = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
