"""Baseline registry. Prompts are copied from each model card (checked 12 Nov 2026).

Parameter counts are not typed in here: retrieve.py counts them from the
loaded weights and writes them into the results file.
"""

from __future__ import annotations

from dataclasses import dataclass


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
