"""Pick the fine-tuned model to report, using the dev split only.

    python scripts/select_model.py

Reads results/benchmark_dev.json (run scripts/reproduce.py --role dev first).
Score per fine-tuned model = mean nDCG@10 over the dev conditions (MIRACL
native and machine-romanized questions, Dakshina machine and human
romanizations), each condition weighted equally. Writes the winner's key to models/SELECTED, which
is the only fine-tuned model scripts/reproduce.py --role test will score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglish_retrieval.eval.metrics import mean  # noqa: E402
from hinglish_retrieval.eval.models import SELECTED  # noqa: E402

CONDS = ["native", "machine", "dk_machine", "dk_human"]


def score(res: dict) -> tuple[float, dict[str, float]]:
    per = {c: mean({q: v["ndcg10"] for q, v in res["per_query"][c].items()}) for c in CONDS if c in res["per_query"]}
    return sum(per.values()) / len(per), per


def main() -> None:
    path = ROOT / "results" / "benchmark_dev.json"
    if not path.exists():
        sys.exit("results/benchmark_dev.json missing: run scripts/reproduce.py --role dev")
    out = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for m, r in out["models"].items():
        if "error" not in r:
            s, per = score(r)
            rows.append((s, m, per))
    rows.sort(reverse=True)
    print("dev split, nDCG@10 x100 | " + " | ".join(CONDS))
    for s, m, per in rows:
        cells = " | ".join(f"{100 * per[c]:.1f}" if c in per else "" for c in CONDS)
        print(f"{m}: {100 * s:.1f} | {cells}")
    trained = [x for x in rows if x[1].startswith("hr-")]
    if not trained:
        sys.exit("no fine-tuned model in the dev results")
    best = trained[0][1]
    SELECTED.parent.mkdir(parents=True, exist_ok=True)
    SELECTED.write_text(best + "\n", encoding="utf-8")
    print(f"selected {best} -> {SELECTED}")


if __name__ == "__main__":
    main()
