# Hinglish retrieval

A retrieval benchmark for Hindi typed in Latin script (Hinglish), and a small embedding model trained to find the same text whether a query is typed in Devanagari or in Hinglish.

Status: in progress. Evaluation, ten baselines (including back-transliteration), training-data generation and fine-tuning are built; one GPU run (`bash scripts/vast_run.sh`) produces the first full results. No results are reported yet. Every number in this file will come from `results/`, which `python scripts/reproduce.py` regenerates.

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

Not yet reported.

## What did not work

- `sk-community/romanized-hindi-transliterator-60M`, the planned back-transliteration model, returns broken output as published (its own model-card example "tum kaisa ho" comes back as "sa हो", under transformers 4.51 and 5.22 alike), and so does its sibling `indotranslit-multitransliterator-60M`. The back-transliteration step uses an instruction-tuned LLM (Qwen3-4B, Apache-2.0) instead; see `src/hinglish_retrieval/data/backtranslit.py`.
- Collecting our own Hinglish rewrites of the MIRACL questions with a web form was dropped: it was slow, and the first answers could not be verified as typed by hand. Dakshina's native-speaker romanizations replace it as the human-typed evaluation.

The rest is to be written from measurements.

## Licences

See [LICENCES.md](LICENCES.md).
