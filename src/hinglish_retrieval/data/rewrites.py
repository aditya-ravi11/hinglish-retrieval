"""Load collected human rewrites (rows written by form/store.py).

Rows live in data/collected/rows/*.jsonl (downloaded from the private HF
dataset, never committed until release). For each source we keep the first
two non-skipped rewrites by submission time, from two different annotators,
as writer slot 1 and writer slot 2.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COLLECTED = ROOT / "data" / "collected" / "rows"


def load_rows(folder: Path = COLLECTED) -> list[dict]:
    rows = []
    for p in sorted(folder.glob("*.jsonl")) if folder.exists() else []:
        with p.open(encoding="utf-8") as f:
            rows.extend(json.loads(l) for l in f if l.strip())
    return rows


def rewrite_text(row: dict) -> str:
    """Text used for evaluation: the cleaned `rewrite` when the import added one, else the raw text."""
    return row.get("rewrite") or row["rewrite_raw"]


def writer_slots(rows: list[dict]) -> dict[str, list[dict]]:
    """source_id -> up to 2 rewrite rows from distinct annotators, oldest first."""
    by_src: dict[str, list[dict]] = {}
    for r in sorted(
        (r for r in rows if r.get("type") == "rewrite" and not r.get("skipped")),
        key=lambda r: r["submitted_at"],
    ):
        slot = by_src.setdefault(r["source_id"], [])
        if len(slot) < 2 and all(x["anon_id"] != r["anon_id"] for x in slot):
            slot.append(r)
    return by_src
