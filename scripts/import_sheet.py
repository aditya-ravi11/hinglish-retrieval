"""Convert the Google Sheet "rows" tab to the JSONL row format.

    python scripts/import_sheet.py data/collected/sheet-latest.xlsx   (or a CSV of the rows tab)

Output: data/collected/rows/sheet-export.jsonl (git-ignored), same schema as
the rows form/store.py writes, so data/rewrites.py reads both.

  * Rows from anon ids in data/excluded_anon_ids.txt are dropped.
  * rewrite_raw keeps the text exactly as submitted. rewrite is the text used
    for evaluation: identical, except that a leading "[synthetic]" label is
    removed (see data/annotator_notes.txt for why it appears).
  * seconds = time from the question appearing to the answer arriving.
  * provenance = the note for that anon id in data/annotator_notes.txt, if any.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "collected" / "rows" / "sheet-export.jsonl"
EXCLUDE = ROOT / "data" / "excluded_anon_ids.txt"
NOTES = ROOT / "data" / "annotator_notes.txt"
TAG = re.compile(r"^\s*\[synthetic\]\s*", re.IGNORECASE)


def _id_file(path: Path) -> dict[str, str]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                aid, _, note = line.partition("#")
                out[aid.strip()] = note.strip()
    return out


def _read(path: str) -> list[dict]:
    """Rows of the "rows" tab from a CSV of that tab or an .xlsx of the whole workbook."""
    if path.lower().endswith(".xlsx"):
        import pandas as pd

        df = pd.read_excel(path, sheet_name="rows", dtype=str).fillna("")
        return df.to_dict("records")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _seconds(a: str, b: str) -> float | None:
    try:
        p = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return round((p(b) - p(a)).total_seconds(), 1)
    except ValueError:
        return None


def main(path: str) -> None:
    excluded = _id_file(EXCLUDE)
    notes = _id_file(NOTES)
    out = []
    n_tag = 0
    for r in _read(path):
        if r["anon_id"] in excluded:
            continue
        base = {"type": r["type"], "anon_id": r["anon_id"], "app_version": r["app_version"]}
        if r["type"] == "rewrite":
            skipped = r["skipped"] == "true"
            raw = None if skipped else json.loads(r["rewrite_json"])
            clean = None if raw is None else TAG.sub("", raw)
            n_tag += raw is not None and clean != raw
            base |= {
                "source_id": r["source_id"],
                "rewrite_raw": raw,
                "rewrite": clean,
                "skipped": skipped,
                "shown_at": r["shown_at"],
                "submitted_at": r["at"],
                "seconds": _seconds(r["shown_at"], r["at"]),
            }
            if r["anon_id"] in notes:
                base["provenance"] = notes[r["anon_id"]]
        elif r["type"] == "consent":
            base |= {
                "consent_version": r["consent_version"],
                "hindi_latin_typing_freq": r["typing_freq"] or None,
                "devanagari_reading": r["devanagari_reading"] or None,
                "at": r["at"],
            }
        else:  # "shown" rows only matter for assignment, keep them for auditing
            base |= {"source_id": r["source_id"], "at": r["at"]}
        out.append(base)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_rw = sum(1 for r in out if r["type"] == "rewrite" and not r["skipped"])
    print(f"wrote {len(out)} rows ({n_rw} rewrites, {n_tag} tags stripped, "
          f"{len(excluded)} excluded ids) to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
