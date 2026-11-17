"""BM25 and dense retrieval with cached corpus embeddings."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np

from hinglish_retrieval.eval.metrics import Run
from hinglish_retrieval.eval.models import ModelSpec

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "cache" / "emb"

# Letters, digits and the whole Devanagari block except the danda marks.
# Python's \w alone splits Hindi words at vowel signs, so it is not enough.
_TOKEN = re.compile(r"[\wऀ-ॣ०-ॿ]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFC", text).lower())


class BM25:
    def __init__(self, ids: list[str], texts: list[str]):
        import bm25s

        self.ids = ids
        self.model = bm25s.BM25()
        self.model.index([tokenize(t) for t in texts], show_progress=False)

    def search(self, queries: dict[str, str], k: int = 100) -> Run:
        qids = list(queries)
        toks = [tokenize(queries[q]) or ["<empty>"] for q in qids]
        docs, scores = self.model.retrieve(toks, k=k, show_progress=False)
        return {
            q: {self.ids[int(d)]: float(s) for d, s in zip(docs[i], scores[i]) if s > 0}
            for i, q in enumerate(qids)
        }


class Dense:
    def __init__(self, spec: ModelSpec, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        self.model = SentenceTransformer(
            spec.hf_id,
            device=device,
            # bf16 on GPU as before; CPU (tests only) is much faster in fp32
            model_kwargs={"torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32},
        )
        self.model.max_seq_length = spec.max_len
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.ids: list[str] = []
        self.emb = None

    def _encode(self, texts: list[str], prompt: str, batch_size: int):
        return self.model.encode(
            texts,
            prompt=prompt or None,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 5000,
        ).astype(np.float16)

    def index(self, corpus_name: str, ids: list[str], texts: list[str], batch_size: int = 128):
        import torch

        digest = hashlib.sha1("\n".join(ids).encode()).hexdigest()[:12]
        path = CACHE / self.spec.key / f"{corpus_name}-{digest}.npy"
        if path.exists():
            emb = np.load(path)
        else:
            emb = self._encode(texts, self.spec.doc_prompt, batch_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, emb)
        self.ids = ids
        self.emb = torch.from_numpy(emb).to(self.device)

    def search(self, queries: dict[str, str], k: int = 100, chunk: int = 256) -> Run:
        import torch

        qids = list(queries)
        q = self._encode([queries[x] for x in qids], self.spec.query_prompt, 64)
        q = torch.from_numpy(q).to(self.device)
        run: Run = {}
        for s in range(0, len(qids), chunk):
            sc = q[s : s + chunk] @ self.emb.T
            top = torch.topk(sc.float(), k=min(k, sc.shape[1]), dim=1)
            for i, (vals, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
                run[qids[s + i]] = {self.ids[j]: v for j, v in zip(idx, vals)}
        return run

    def close(self):
        import gc

        import torch

        self.emb = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
