# Hinglish retrieval

A retrieval benchmark for Hindi typed in Latin script (Hinglish), and a small embedding model trained to find the same text whether a query is typed in Devanagari or in Hinglish.

Status: first full results are in (17 November 2026, below). The fine-tuned model is not yet published on Hugging Face. Every number in this file comes from `results/`, which `bash scripts/vast_run.sh` (or `python scripts/reproduce.py`) regenerates.

## The problem

Many people type Hindi in Latin letters with no fixed spelling: "kya", "kia" and "kyaa" are the same word. Embedding models are trained mostly on native-script text, so a question typed as "bharat ka rashtriya pashu kaun sa hai" may not find the passages that "भारत का राष्ट्रीय पशु कौन सा है" finds. Earlier evaluations of this gap use machine transliteration, which produces one clean spelling per word. This project also measures it with spellings people actually typed, from Google's Dakshina dataset.

## Related work

- Found in Transliteration (Dhruvgohil07, GitHub, 2026): MIRACL Hindi with native and romanized queries, including a 50-query human-romanized set, and a back-transliteration pipeline.
- Hinglish-CSR (Shashankshekhar13, GitHub): LLM-written Hinglish rewrites of English BEIR queries with synthetic spelling noise, 120 human-verified.
- Singh, "One Meaning, Many Romanizations" (MRL 2026): informal romanization across four Indic languages in multilingual representations.
- Chari, Ounis and MacAvaney, "Lost in Transliteration" (SIGIR 2025, arXiv 2505.08411): the script gap in neural IR with transliterated queries.
- Roark et al., "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset" (LREC 2020): native-speaker romanizations of Wikipedia sentences.
- `AkshitaS/bhasha-embed-v0`: an embedding model for Hindi, English and romanized Hindi, evaluated on machine-transliterated text.

What this project adds: machine and human romanization compared on the same 4,000 sentences, so the cost of real spelling variation is measured rather than simulated; MIRACL Hindi retrieval with native and machine-romanized questions; ten baselines, including three back-transliteration pipelines; and a small model (at most 300M parameters) fine-tuned only on synthetic data, with a leakage check against every evaluation text. The benchmark is meant for MTEB.

## Benchmark design

| Part | Choice | Why |
|---|---|---|
| MIRACL queries | 1,000 MIRACL Hindi test queries (all 349 unique dev-origin texts and 651 from train) and 150 dev queries, each as the Devanagari question and its machine romanization | MIRACL already has relevance labels. Dev-origin queries are kept separate in results because some baselines were fine-tuned on MIRACL train. |
| Corpus | Pooled subset of the 506k-passage MIRACL Hindi corpus: all judged passages plus the top 100 from BM25, multilingual-e5-base and bge-m3 for native and machine-romanized queries | Ten models times several query conditions on 506k passages is slow, and MTEB prefers small tasks. Pooling with several systems keeps the hard distractors. |
| Human romanization | Google Dakshina (CC BY-SA 4.0): 10,000 Hindi Wikipedia sentences typed in Latin script by native speakers. Queries are the typed sentences (1,000 dev, 4,000 test); the target is the Devanagari sentence among all ~10,000. The same sentences are also machine-romanized | Real spelling variation at scale, and a paired machine-versus-human comparison on identical content. |
| Metric | nDCG@10, with Recall@100 | nDCG@10 is the MTEB retrieval headline. Gaps are reported with 95% paired bootstrap intervals over queries. |
| Deduplication | MIRACL repeats some question texts under different ids. Selection keeps one id per normalised text, and drops training-pool queries that are near paraphrases of evaluation queries | Found by the leakage check on its first run: 4 training-pool queries were identical to test queries and 2 test queries were identical to dev queries. |

## Baselines and the trained model

- BM25 and six dense models: multilingual-e5-small and -base, bge-m3, EmbeddingGemma-300m, Qwen3-Embedding-0.6B, bhasha-embed-v0.
- Back-transliteration: Qwen3-4B rewrites each Latin-script query in Devanagari, then BM25, multilingual-e5-base or bge-m3 searches unchanged.
- Fine-tuned: bhasha-embed-v0, multilingual-e5-base and -small, trained with an in-batch contrastive loss on synthetic data. Qwen3-8B writes a Hindi question for passages outside the evaluation pool, plus two Hinglish versions; a third spelling comes from hand-written rules (`src/hinglish_retrieval/train/noise.py`). The MIRACL-hi training pool (queries never evaluated) adds real questions with rule-based spellings. The fine-tuned model with the best dev score is the only one scored on test.

## Leakage rule

Evaluation data (MIRACL test and dev queries, Dakshina sentences) is never used for training or spelling-noise fitting, and model selection uses only the dev splits. `scripts/check_leakage.py --full` normalises every text (Devanagari mapped to Latin, lowercase, repeated letters collapsed) and fails on any exact match between a training text and an evaluation text, on any char-5-gram Jaccard of 0.8 or more between a training text and a MIRACL query, on any training query generated from a passage judged for an evaluation query, and on any seed passage that contains a Dakshina sentence. CI re-runs the check from committed hash manifests.

## Reproduce

On a Linux GPU, after the setup in [SETUP.md](SETUP.md) section 5, the whole run (training data, leakage check, fine-tuning, dev selection, test table) is one command:

```
bash scripts/vast_run.sh
```

The baseline tables alone:

```
python scripts/reproduce.py --role dev
python scripts/reproduce.py --role test
```

## Results

One run of `bash scripts/vast_run.sh` on a single RTX A6000 (48 GB, rented on Vast.ai), about 3.5 hours. Test split, nDCG@10 (x100). Full tables with 95% paired bootstrap intervals: [results/table_test.md](results/table_test.md) and [results/table_dev.md](results/table_dev.md); per-query scores and Recall@100 in `results/benchmark_*.json`.

| model | params (M) | MIRACL native | MIRACL machine | Dakshina machine | Dakshina human |
|---|---|---|---|---|---|
| BM25 | 0 | 37.2 | 0.1 | 8.9 | 10.4 |
| multilingual-e5-small | 118 | 56.0 | 0.1 | 24.4 | 43.9 |
| multilingual-e5-base | 278 | 57.9 | 0.0 | 21.7 | 41.2 |
| bge-m3 | 568 | 67.4 | 1.1 | 47.4 | 63.0 |
| EmbeddingGemma-300m | 308 | 64.6 | 11.7 | 65.7 | 75.5 |
| Qwen3-Embedding-0.6B | 596 | 49.7 | 12.5 | 67.9 | 76.8 |
| bhasha-embed-v0 | 238 | 44.8 | 23.2 | 96.1 | 96.3 |
| back-transliteration + BM25 | 4,022 | 37.2 | 20.1 | 94.7 | 95.0 |
| back-transliteration + e5-base | 4,301 | 57.9 | 43.4 | 98.3 | 98.9 |
| back-transliteration + bge-m3 | 4,590 | 67.4 | 54.3 | 98.9 | 99.3 |
| **hr-bhasha-embed-v0 (ours)** | 238 | 43.5 | 31.1 | 99.0 | 99.0 |

MIRACL: 1,000 questions, target among 171,873 passages. Dakshina: 4,000 sentences, target among 9,899. Back-transliteration parameter counts include the 4B LLM.

What the numbers say:

1. **The script gap is close to total for most models.** Typed in machine-romanized Latin script, the same MIRACL questions drop from 56 to 67 to at most 1.1 for multilingual-e5 and bge-m3, and to about 12 for EmbeddingGemma and Qwen3-Embedding. Only models built for romanized Hindi keep a usable score.
2. **Real typing is easier than our machine romanization.** On the same 4,000 Dakshina sentences, every general model scores higher on the native speakers' spellings than on the machine romanization: +19.5 points for both e5 models, +15.6 for bge-m3, +9.8 for EmbeddingGemma and +8.9 for Qwen3-Embedding (all intervals exclude zero). People keep English loanwords in Latin script ("ribosome", "feet") and use common spellings, while rule-based romanization produces rarer ones. So machine-romanized benchmarks overstate the gap, at least for careful typing: Dakshina's annotators romanized Wikipedia sentences, not casual search queries.
3. **Back-transliteration recovers most of the gap, at a cost.** An LLM that rewrites the query in Devanagari, in front of bge-m3, reaches 54.3 on machine-romanized MIRACL against 67.4 native, but it adds a 4B-parameter model to every query.
4. **The fine-tuned model helps on romanized queries without losing native ones.** Against its base, hr-bhasha-embed-v0 gains +7.9 [+6.1, +9.6] on machine-romanized MIRACL and +2.6 [+2.2, +3.1] on Dakshina human typing; native changes by -1.3 [-2.9, +0.3], not significant. It has the smallest native-to-romanized gap of any system (12.4 points) and beats back-transliteration with BM25 by +11.0 [+8.5, +13.5], but trails back-transliteration with bge-m3 by 23.2 points on romanized MIRACL, a pipeline about 19 times its size.
5. **Dakshina saturates for romanization-aware systems** (96 to 99): with sentence-length queries, finding the matching sentence is easy once the script is handled. It separates general models well, but not the strongest ones.
6. **Contamination check:** bge-m3 and EmbeddingGemma score 10 to 12 points higher on MIRACL questions from its train split than from its dev split (71.7 against 59.4, and 68.2 against 57.7), consistent with MIRACL train in their training data. The other models differ by about 2 points.

Training data for the fine-tuned model ([data/train/](data/train/)): 9,739 synthetic items from Qwen3-8B (15% of its answers failed the format checks and were dropped) and 327 MIRACL training-pool questions, 39,379 (query, passage) pairs, one epoch. The full leakage check passed: no exact or near duplicates of any evaluation text, no seed passage judged for an evaluation query, none containing a Dakshina sentence ([results/leakage_report.json](results/leakage_report.json)).

## What did not work

- `sk-community/romanized-hindi-transliterator-60M`, the planned back-transliteration model, returns broken output as published (its own model-card example "tum kaisa ho" comes back as "sa हो", under transformers 4.51 and 5.22 alike), and so does its sibling `indotranslit-multitransliterator-60M`. The back-transliteration step uses an instruction-tuned LLM (Qwen3-4B, Apache-2.0) instead; see `src/hinglish_retrieval/data/backtranslit.py`.
- Collecting our own Hinglish rewrites of the MIRACL questions with a web form was dropped: it was slow, and the first answers could not be verified as typed by hand. Dakshina's native-speaker romanizations replace it as the human-typed evaluation.
- Fine-tuning multilingual-e5 on the same data cost native accuracy. On the dev split, native MIRACL fell from 63.6 to 43.6 (e5-base) and from 63.5 to 42.7 (e5-small), while romanized queries improved. bhasha-embed-v0, already trained on romanized Hindi, did not lose (48.4 to 49.5), so dev selection picked it and the e5 runs were never scored on test. Next to try: more native pairs in the mix, or a distillation term that keeps the student close to its base on native text.
- Training-data generation ran out of GPU memory on the longest passages at a batch of 96. Generation now halves a batch and retries when that happens.

## Licences

See [LICENCES.md](LICENCES.md).
