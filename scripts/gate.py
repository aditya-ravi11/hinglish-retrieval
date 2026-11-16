"""Go/no-go gate: script gap on the human queries collected so far.

    python scripts/gate.py

Reads results/benchmark_test.json (run scripts/reproduce.py first). For each
baseline, on the test queries that have at least one human rewrite: native
minus human nDCG@10 with a 95% paired bootstrap interval. Rule from the brief:
if the best baseline with at most 300M parameters loses less than about 3
points, the model angle changes and Aditya decides.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hinglish_retrieval.eval.metrics import mean, paired_bootstrap  # noqa: E402
from reproduce import human_scores  # noqa: E402

SMALL = 300e6
THRESHOLD = 3.0


def main() -> None:
    out = json.loads((ROOT / "results" / "benchmark_test.json").read_text(encoding="utf-8"))
    if "human_w1" not in out["conditions"]:
        sys.exit("No human rewrites yet. Import rows and rerun scripts/reproduce.py.")
    n1 = out["conditions"]["human_w1"]
    n2 = out["conditions"].get("human_w2", 0)
    print(f"Test queries with a human rewrite: {n1} (with a second writer: {n2})")
    print("model | params M | native | human | gap [95% CI]")
    best_small = None
    for m, r in out["models"].items():
        if "error" in r:
            continue
        nd = {c: {q: v["ndcg10"] for q, v in pq.items()} for c, pq in r["per_query"].items()}
        hu = human_scores(nd)
        nat = {q: nd["native"][q] for q in hu}
        g, lo, hi = paired_bootstrap(nat, hu)
        params = r.get("params", 0)  # 0 for BM25
        print(f"{m} | {params / 1e6:.0f} | {100 * mean(nat):.1f} | {100 * mean(hu):.1f} | "
              f"{100 * g:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]")
        if m != "bm25" and params <= SMALL and (best_small is None or mean(hu) > best_small[1]):
            best_small = (m, mean(hu), 100 * g, 100 * lo)
    if best_small:
        m, _, gap, lo = best_small
        verdict = "GO (model angle holds)" if lo >= THRESHOLD else "BELOW OR UNCLEAR (Aditya decides)"
        print(f"Best small model on human queries: {m}, gap {gap:.1f} points "
              f"(CI lower bound {lo:.1f}) -> {verdict}")


if __name__ == "__main__":
    main()
