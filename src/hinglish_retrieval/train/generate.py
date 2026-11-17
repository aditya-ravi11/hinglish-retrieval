"""Training data: synthetic Hindi questions from passages, with Hinglish versions.

    python -m hinglish_retrieval.train.generate [--n 20000] [--llm Qwen/Qwen3-8B]

Seeds are passages of the full MIRACL Hindi corpus that are NOT in the pooled
evaluation corpus, so no seed was ever judged for an evaluation query, and that
contain no sentence of the Dakshina evaluation set. For each
seed an instruction-tuned LLM (Apache-2.0 Qwen3 by default) writes one Hindi
question the passage answers and two Hinglish versions of it. A third Hinglish
version comes from the rule-based noise in noise.py. The MIRACL-hi training
pool (data/sources/train_pool_hi.jsonl, queries never used for evaluation) is
added with its relevant passages, minus any passage judged for an evaluation
query, and rule-based Hinglish versions.

Before writing, items whose native or Hinglish text is an exact or near
duplicate (char-5-gram Jaccard >= 0.8 after normalisation) of any MIRACL
evaluation query, or an exact duplicate of any Dakshina sentence, are dropped,
so scripts/check_leakage.py --full passes.

Raw LLM outputs are appended to data/cache/train_gen/raw.jsonl as they come, so
a rerun after a crash continues where it stopped.

Output:
  data/train/synth.jsonl        {qid, text_native, text_hinglish[], seed_passage_id, positive_ids[], source}
  data/train/miracl_pool.jsonl  same fields, no seed_passage_id
  data/train/meta.json          generator settings and counts
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from hinglish_retrieval.data.corpus import POOL_IDS, load_hi_full
from hinglish_retrieval.leakage import near_duplicates, normalise
from hinglish_retrieval.train.noise import hinglish_noisy

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "data" / "train"
RAW = ROOT / "data" / "cache" / "train_gen" / "raw.jsonl"

PROMPT = """Read this Hindi Wikipedia passage and write one question in Hindi (Devanagari script) that the passage answers. Make it sound like something a person would type into a search engine: 4 to 20 words, and do not copy a whole sentence from the passage.

Then write the same question twice the way people in India type Hindi on a phone in Latin letters (Hinglish): informal spellings, no accents or special characters, and common English words mixed in where people would naturally use them. The two versions should spell some words differently.

Answer in exactly this format and nothing else:
HINDI: <question>
HINGLISH 1: <question>
HINGLISH 2: <question>

Passage:
{passage}"""

_DEVA = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_LINE = re.compile(r"^\W*(HINDI|HINGLISH\s*1|HINGLISH\s*2)\W*?[:：]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _share(rx: re.Pattern, s: str) -> float:
    letters = [c for c in s if c.isalpha() or _DEVA.match(c)]
    return sum(1 for c in letters if rx.match(c)) / max(1, len(letters))


def parse(raw: str, passage: str = "") -> tuple[str, list[str]] | None:
    """(hindi, [hinglish1, hinglish2]) from one LLM answer, or None if it does not pass the checks."""
    got: dict[str, str] = {}
    for key, text in _LINE.findall(raw):
        k = re.sub(r"\s+", "", key.upper())
        got.setdefault(k, text.strip().strip('"*').strip())
    hi = got.get("HINDI", "")
    hs = [got.get("HINGLISH1", ""), got.get("HINGLISH2", "")]
    if not (4 <= len(hi.split()) <= 25) or _share(_DEVA, hi) < 0.8:
        return None
    if normalise(hi) and normalise(hi) in normalise(passage):  # copied a sentence
        return None
    good = []
    for h in hs:
        if 3 <= len(h.split()) <= 25 and not _DEVA.search(h) and _share(_LATIN, h) > 0.95:
            if h.lower() not in {g.lower() for g in good}:
                good.append(h)
    return (hi, good) if good else None


def eval_texts() -> dict[str, str]:
    """Every MIRACL evaluation query (all roles)."""
    ev = {}
    for line in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8"):
        s = json.loads(line)
        ev[s["source_id"]] = s["text"]
    return ev


def dakshina_texts() -> set[str]:
    """Normalised Dakshina sentences, native and human-romanized (exact-match check only)."""
    from hinglish_retrieval.data import dakshina

    return {normalise(t) for r in dakshina.load() for t in (r["native"], r["human"])}


def judged_passages() -> set[str]:
    from hinglish_retrieval.data.miracl import load_qrels

    qrels = {**load_qrels("hi", "train"), **load_qrels("hi", "dev")}
    out: set[str] = set()
    for line in open(ROOT / "data/sources/sources.jsonl", encoding="utf-8"):
        s = json.loads(line)
        if s["lang"] == "hi":
            out.update(qrels[s["miracl_qid"]])
    return out


def pick_seeds(ids: list[str], texts: list[str], k: int, seed: int) -> list[int]:
    from hinglish_retrieval.data.dakshina import native_keys, sentence_keys

    banned = set(POOL_IDS.read_text(encoding="utf-8").split()) | judged_passages()
    ok = [
        i
        for i, (d, t) in enumerate(zip(ids, texts))
        if d not in banned and 200 <= len(t) <= 1500 and _share(_DEVA, t) > 0.7
    ]
    random.Random(seed).shuffle(ok)
    dk = native_keys()
    # passages are shuffled first, so the Dakshina check only runs on what is needed
    out = []
    for i in ok:
        if not (sentence_keys(texts[i]) & dk):
            out.append(i)
            if len(out) == k:
                break
    return out


def llm_generate(todo: list[tuple[str, str]], model_id: str, batch_size: int | None, seed: int):
    """Yield (passage_id, raw_answer) for [(passage_id, passage_text)], one batch at a time."""
    from hinglish_retrieval.llm import chat_generate

    prompts = [PROMPT.format(passage=t[:1200]) for _, t in todo]
    for i, a in chat_generate(model_id, prompts, 200, batch_size, sample=True, seed=seed, progress=True):
        yield todo[i][0], a


def _variants(native: str, llm: list[str], qid: str, k_noise: int) -> list[str]:
    rng = random.Random(qid)
    out = list(llm)
    for _ in range(k_noise * 3):
        if len(out) >= len(llm) + k_noise:
            break
        v = hinglish_noisy(native, rng)
        if v and v not in out:
            out.append(v)
    return out


def drop_leaks(items: list[dict], ev: dict[str, str], exact_only: set[str] = frozenset()) -> tuple[list[dict], int]:
    """Remove items with any text that matches or nearly matches an evaluation text.

    ev: texts checked for exact and near duplicates; exact_only: normalised
    texts checked for exact duplicates only (the 20,000 Dakshina strings).
    """
    ev_n = {k: normalise(v) for k, v in ev.items()}
    ev_set = set(ev_n.values()) | set(exact_only)
    tr_n = {}
    for i, it in enumerate(items):
        for j, t in enumerate([it["text_native"], *it["text_hinglish"]]):
            tr_n[f"{i}:{j}"] = normalise(t)
    bad = {k.split(":")[0] for k, v in tr_n.items() if v in ev_set}
    bad |= {a.split(":")[0] for a, _, _ in near_duplicates(tr_n, ev_n, threshold=0.8)}
    keep = [it for i, it in enumerate(items) if str(i) not in bad]
    return keep, len(items) - len(keep)


def pool_items(texts_by_id: dict[str, str]) -> list[dict]:
    from hinglish_retrieval.data.miracl import load_qrels

    qrels = load_qrels("hi", "train")
    judged = judged_passages()
    out = []
    for line in open(ROOT / "data/sources/train_pool_hi.jsonl", encoding="utf-8"):
        r = json.loads(line)
        pos = [d for d, rel in qrels.get(r["miracl_qid"], {}).items() if rel > 0 and d not in judged and d in texts_by_id]
        if pos:
            qid = f"pool-{r['miracl_qid']}"
            out.append({
                "qid": qid, "text_native": r["text"], "text_hinglish": _variants(r["text"], [], qid, 3),
                "positive_ids": pos, "source": "miracl-train-pool",
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000, help="target number of synthetic items")
    ap.add_argument("--llm", default="Qwen/Qwen3-8B")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--allow-few", action="store_true", help="smoke test: do not fail on a low yield")
    args = ap.parse_args()

    t0 = time.time()
    ids, texts = load_hi_full()
    if not ids:
        raise SystemExit("MIRACL Hindi corpus missing: see SETUP.md (hf download miracl/miracl-corpus ...)")
    text_of = dict(zip(ids, texts))
    seeds = pick_seeds(ids, texts, int(args.n * 1.15) + 10, args.seed)
    print(f"{len(ids)} passages, {len(seeds)} seeds picked ({time.time() - t0:.0f}s)")

    RAW.parent.mkdir(parents=True, exist_ok=True)
    have = {}
    if RAW.exists():
        for line in RAW.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("llm") == args.llm:
                have[r["pid"]] = r["raw"]
    todo = [(ids[i], texts[i]) for i in seeds if ids[i] not in have]
    print(f"{len(have)} raw answers cached, {len(todo)} to generate with {args.llm}")
    if todo:
        fresh: list[str] = []
        with RAW.open("a", encoding="utf-8") as f:
            for pid, raw in llm_generate(todo, args.llm, args.batch_size, args.seed):
                have[pid] = raw
                f.write(json.dumps({"pid": pid, "llm": args.llm, "raw": raw}, ensure_ascii=False) + "\n")
                f.flush()
                fresh.append(pid)
                if len(fresh) == min(200, len(todo)) and not args.allow_few:
                    # stop early, before hours of GPU time, if the model ignores the format
                    ok = sum(parse(have[p], text_of[p]) is not None for p in fresh)
                    print(f"  early check: {ok}/{len(fresh)} answers usable", flush=True)
                    if ok < 0.3 * len(fresh):
                        sample = "\n---\n".join(have[p][:300] for p in fresh[:3])
                        raise SystemExit(f"too few usable answers from {args.llm}; examples:\n{sample}")

    items, seen, failed = [], set(), 0
    for i in seeds:
        pid = ids[i]
        got = parse(have.get(pid, ""), texts[i])
        if got is None:
            failed += 1
            continue
        hi, hs = got
        key = normalise(hi)
        if key in seen:
            continue
        seen.add(key)
        qid = f"syn-{pid}"
        items.append({
            "qid": qid, "text_native": hi, "text_hinglish": _variants(hi, hs, qid, 1),
            "seed_passage_id": pid, "positive_ids": [pid], "source": f"llm:{args.llm}",
        })
        if len(items) >= args.n:
            break
    pool = pool_items(text_of)

    ev, dk = eval_texts(), dakshina_texts()
    items, leak_syn = drop_leaks(items, ev, dk)
    pool, leak_pool = drop_leaks(pool, ev, dk)
    TRAIN.mkdir(parents=True, exist_ok=True)
    for name, rows in (("synth", items), ("miracl_pool", pool)):
        with (TRAIN / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm": args.llm, "seed": args.seed, "target_n": args.n,
        "seeds_tried": len(seeds), "parse_failures": failed,
        "synth_items": len(items), "pool_items": len(pool),
        "dropped_as_near_eval": {"synth": leak_syn, "pool": leak_pool},
        "hinglish_per_item": "2 from the LLM (when both pass the checks) + 1 rule-based (noise.py); pool: 3 rule-based",
    }
    (TRAIN / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    if len(items) < 0.5 * args.n and not args.allow_few:
        raise SystemExit(f"only {len(items)} usable items out of {args.n}: check the raw answers in {RAW}")


if __name__ == "__main__":
    main()
