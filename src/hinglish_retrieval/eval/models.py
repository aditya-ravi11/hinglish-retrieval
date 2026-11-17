"""Baseline registry. Prompts are copied from each model card (checked 12 Nov 2026).

Parameter counts are not typed in here: retrieve.py counts them from the
loaded weights and writes them into the results file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"  # fine-tuned models, git-ignored
SELECTED = MODELS_DIR / "SELECTED"  # key of the model chosen on the dev split


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    licence: str
    query_prompt: str = ""
    doc_prompt: str = ""
    max_len: int = 512
    gated: bool = False


QWEN_TASK = "Given a web search query, retrieve relevant passages that answer the query"

DENSE = {
    s.key: s
    for s in [
        ModelSpec("me5-small", "intfloat/multilingual-e5-small", "MIT", "query: ", "passage: "),
        ModelSpec("me5-base", "intfloat/multilingual-e5-base", "MIT", "query: ", "passage: "),
        ModelSpec("bge-m3", "BAAI/bge-m3", "MIT"),
        ModelSpec(
            "embeddinggemma-300m",
            "google/embeddinggemma-300m",
            "Gemma Terms of Use",
            "task: search result | query: ",
            "title: none | text: ",
            gated=True,
        ),
        ModelSpec(
            "qwen3-emb-0.6b",
            "Qwen/Qwen3-Embedding-0.6B",
            "Apache-2.0",
            f"Instruct: {QWEN_TASK}\nQuery:",
            "",
        ),
        ModelSpec("bhasha-embed-v0", "AkshitaS/bhasha-embed-v0", "Apache-2.0"),
    ]
}

# Back-transliteration pipelines: Latin-script queries go through
# data/backtranslit.py first, then an unchanged retriever.
BACKTRANSLIT = {"bt-bm25": "bm25", "bt-me5-base": "me5-base", "bt-bge-m3": "bge-m3"}


def trained(role: str | None = None) -> dict[str, ModelSpec]:
    """Fine-tuned models in models/<key>/, written by hinglish_retrieval.train.train.

    role "dev": every trained model, so one can be selected on the dev split.
    role "test": only the model named in models/SELECTED, so the test split
    never takes part in model selection.
    """
    if not MODELS_DIR.exists():
        return {}
    keys = sorted(p.parent.name for p in MODELS_DIR.glob("*/hr_spec.json"))
    if role == "test":
        chosen = SELECTED.read_text(encoding="utf-8").strip() if SELECTED.exists() else ""
        keys = [k for k in keys if k == chosen]
    out = {}
    for k in keys:
        meta = json.loads((MODELS_DIR / k / "hr_spec.json").read_text(encoding="utf-8"))
        out[k] = ModelSpec(
            k,
            str(MODELS_DIR / k),
            meta["licence"],
            meta["query_prompt"],
            meta["doc_prompt"],
            meta.get("max_len", 512),
        )
    return out
