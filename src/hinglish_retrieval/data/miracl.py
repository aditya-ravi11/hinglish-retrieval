"""Download and parse MIRACL topics and qrels (v1.0) from the official HF repo.

The repo `miracl/miracl` (Apache-2.0) stores plain TSV files, so we fetch them
with hf_hub_download instead of a dataset loading script.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "miracl/miracl"
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "miracl"


def _fetch(path_in_repo: str) -> Path:
    return Path(
        hf_hub_download(REPO, path_in_repo, repo_type="dataset", local_dir=RAW_DIR)
    )


def load_topics(lang: str, split: str) -> dict[str, str]:
    """Return {query_id: query_text} for a language and split (train, dev)."""
    p = _fetch(f"miracl-v1.0-{lang}/topics/topics.miracl-v1.0-{lang}-{split}.tsv")
    out: dict[str, str] = {}
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 2:
                out[row[0]] = row[1]
    return out


def load_qrels(lang: str, split: str) -> dict[str, dict[str, int]]:
    """Return {query_id: {doc_id: relevance}} from a TREC-format qrels file."""
    p = _fetch(f"miracl-v1.0-{lang}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv")
    out: dict[str, dict[str, int]] = defaultdict(dict)
    with p.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 4:
                qid, _, did, rel = parts
                out[qid][did] = int(rel)
    return dict(out)
