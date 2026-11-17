"""Fine-tune small embedding models on data/train/*.jsonl.

    python -m hinglish_retrieval.train.train [--bases bhasha-embed-v0,me5-base,me5-small]

Every training item has one positive passage and several query versions: the
Devanagari question and its Hinglish spellings. Each version is paired with
the passage, and the loss is in-batch contrastive (cached multiple-negatives
ranking, so a large batch fits in memory). The no-duplicates batch sampler
keeps two versions of the same item out of one batch, where they would count
as false negatives for each other.

Each base model ends up in models/hr-<base>/ (a SentenceTransformer folder
plus hr_spec.json with the prompts and licence), and that model's cached
corpus embeddings are cleared so evaluation re-encodes with the new weights.
Bases are the baselines with at most 300M parameters, the project's size
limit for the trained model.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from hinglish_retrieval.eval.models import DENSE, MODELS_DIR
from hinglish_retrieval.eval.retrieve import CACHE

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data" / "train"
BASES = ["bhasha-embed-v0", "me5-base", "me5-small"]


def load_items(limit: int | None = None) -> list[dict]:
    items = []
    for p in sorted(TRAIN.glob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            items.extend(json.loads(l) for l in f if l.strip())
    if not items:
        raise SystemExit("no training data in data/train/: run python -m hinglish_retrieval.train.generate")
    random.Random(0).shuffle(items)
    return items[:limit] if limit else items


def pairs(items: list[dict], passage_text: dict[str, str]) -> tuple[list[str], list[str]]:
    anchors, positives = [], []
    for it in items:
        pos = [p for p in it["positive_ids"] if p in passage_text]
        if not pos:
            continue
        for k, q in enumerate([it["text_native"], *it["text_hinglish"]]):
            anchors.append(q)
            positives.append(passage_text[pos[k % len(pos)]])
    return anchors, positives


def train_one(base: str, anchors, positives, args) -> Path:
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
        SentenceTransformerTrainingArguments,
    )

    spec = DENSE[base]
    key = f"hr-{base}"
    out_dir = MODELS_DIR / key
    cuda = torch.cuda.is_available()
    model = SentenceTransformer(spec.hf_id, device="cuda" if cuda else "cpu")
    model.max_seq_length = args.max_len
    prompts = {}
    if spec.query_prompt:
        prompts["anchor"] = spec.query_prompt
    if spec.doc_prompt:
        prompts["positive"] = spec.doc_prompt

    ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(MODELS_DIR / "_checkpoints" / key),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=cuda,
        use_cpu=not cuda,  # never Apple MPS: its attention kernel has no dropout
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts=prompts or None,
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        dataloader_drop_last=True,
    )
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss).train()
    secs = round(time.time() - t0)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    model.save(str(out_dir))
    meta = {
        "key": key,
        "base": base,
        "base_hf_id": spec.hf_id,
        "licence": spec.licence,
        "query_prompt": spec.query_prompt,
        "doc_prompt": spec.doc_prompt,
        "max_len": spec.max_len,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pairs": len(anchors),
        "train_seconds": secs,
        "args": vars(args),
    }
    (out_dir / "hr_spec.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(CACHE / key, ignore_errors=True)  # stale corpus embeddings
    shutil.rmtree(MODELS_DIR / "_checkpoints" / key, ignore_errors=True)
    print(f"{key}: {len(anchors)} pairs, {secs}s -> {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default=",".join(BASES))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mini-batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-steps", type=int, help="stop early (smoke test)")
    ap.add_argument("--limit-items", type=int, help="use only this many training items (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bases = args.bases.split(",")
    bad = [b for b in bases if b not in DENSE]
    if bad:
        raise SystemExit(f"unknown bases {bad}; choose from {list(DENSE)}")

    from hinglish_retrieval.data.corpus import load_hi_full

    items = load_items(args.limit_items)
    need = {p for it in items for p in it["positive_ids"]}
    ids, texts = load_hi_full()
    passage_text = {d: t for d, t in zip(ids, texts) if d in need}
    anchors, positives = pairs(items, passage_text)
    print(f"{len(items)} items -> {len(anchors)} (query, passage) pairs")
    for b in bases:
        train_one(b, anchors, positives, args)


if __name__ == "__main__":
    main()
