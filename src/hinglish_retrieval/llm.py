"""Batched chat generation with a local instruction-tuned model (transformers).

Used for the synthetic training data (train/generate.py) and for the
LLM back-transliteration baseline (data/backtranslit.py).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from functools import lru_cache

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@lru_cache(maxsize=1)
def load(model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model


def n_params(model_id: str) -> int:
    """Parameter count from the checkpoint header, without loading the weights."""
    try:
        from huggingface_hub import get_safetensors_metadata

        return sum(get_safetensors_metadata(model_id).parameter_count.values())
    except Exception:  # offline: load and count
        _, model = load(model_id)
        return sum(p.numel() for p in model.parameters())


def unload() -> None:
    """Free the loaded model's GPU memory."""
    import gc

    import torch

    load.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def default_batch_size(model) -> int:
    import torch

    if model.device.type != "cuda":
        return 2
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    return 192 if gb >= 70 else 96 if gb >= 40 else 24 if gb >= 22 else 4


def chat_generate(
    model_id: str,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int | None = None,
    sample: bool = False,
    seed: int = 0,
    progress: bool = False,
) -> Iterator[tuple[int, str]]:
    """Yield (index into prompts, answer) one batch at a time, shortest prompts first."""
    import torch

    tok, model = load(model_id)
    bs = batch_size or default_batch_size(model)
    torch.manual_seed(seed)
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for p in prompts
    ]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding per batch
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False)
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        answers = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, a in zip(idx, answers):
            yield i, _THINK.sub("", a).strip()
        if progress:
            done = s + len(idx)
            rate = done / (time.time() - t0)
            left = (len(order) - done) / rate / 60
            print(f"  {done}/{len(order)} ({rate:.2f}/s, ~{left:.0f} min left)", flush=True)
