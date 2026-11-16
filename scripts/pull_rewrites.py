"""Download collected rows from the private HF dataset and print progress.

    python scripts/pull_rewrites.py --repo <user>/hinglish-rewrites-private

Rows land in data/collected/rows/ (git-ignored). Needs `hf auth login`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglish_retrieval.data.rewrites import COLLECTED, load_rows, writer_slots  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="private dataset id; omit to only summarise local rows")
    args = ap.parse_args()
    if args.repo:
        from huggingface_hub import snapshot_download

        snapshot_download(
            args.repo, repo_type="dataset", local_dir=COLLECTED.parent, allow_patterns=["rows/*.jsonl"]
        )
    rows = load_rows()
    import json

    role = {
        s["source_id"]: s["role"]
        for s in (json.loads(l) for l in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8"))
    }
    rw = [r for r in rows if r.get("type") == "rewrite"]
    slots = writer_slots(rows)
    pairs = Counter(role[s] for s, v in slots.items() if len(v) == 2)
    print(f"annotators (consented): {len({r['anon_id'] for r in rows if r.get('type') == 'consent'})}")
    print(f"rewrites: {sum(not r['skipped'] for r in rw)}, skips: {sum(r['skipped'] for r in rw)}")
    print(f"sources with 2 rewrites by role: {dict(pairs)}")


if __name__ == "__main__":
    main()
