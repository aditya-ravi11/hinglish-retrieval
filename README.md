# Hinglish retrieval

A retrieval benchmark of Hindi questions typed in Latin script by real people, and a small embedding model trained to retrieve the same passages whether a question is typed in Devanagari or in Hinglish.

Status: in progress. The collection form and evaluation harness are built. No results are reported yet. Every number in this file will come from `results/`, which `python scripts/reproduce.py` regenerates.

## The problem

Many people type Hindi in Latin letters with no fixed spelling: "kya", "kia" and "kyaa" are the same word. Embedding models are trained mostly on native-script text, so a question typed as "bharat ka rashtriya pashu kaun sa hai" may not find the passages that "भारत का राष्ट्रीय पशु कौन सा है" finds. Earlier evaluations of this gap use machine transliteration, which produces one clean spelling per word. This project measures it with spellings people actually type.

## Related work

- Found in Transliteration (Dhruvgohil07, GitHub, 2026): MIRACL Hindi with native and romanized queries, including a 50-query human-romanized set, and a back-transliteration pipeline.
- Hinglish-CSR (Shashankshekhar13, GitHub): LLM-written Hinglish rewrites of English BEIR queries with synthetic spelling noise, 120 human-verified.
- Singh, "One Meaning, Many Romanizations" (MRL 2026): informal romanization across four Indic languages in multilingual representations.
- Chari, Ounis and MacAvaney, "Lost in Transliteration" (SIGIR 2025, arXiv 2505.08411): the script gap in neural IR with transliterated queries.
- `AkshitaS/bhasha-embed-v0`: an embedding model for Hindi, English and romanized Hindi, evaluated on machine-transliterated text.

What this project adds: about 1,000 MIRACL Hindi questions each rewritten by at least two different people, so spelling variance itself can be measured; a direct comparison of machine and human romanization on the same queries; a code-mixed English subset; ten baselines including a back-transliteration pipeline; and a small trained model. The benchmark is meant for MTEB.

## Benchmark design

| Part | Choice | Why |
|---|---|---|
| Source queries | 1,000 MIRACL Hindi test queries (all 349 unique dev-origin texts and 651 from train), 150 for a human dev split, 150 English dev queries | MIRACL already has relevance labels, so only the queries need writing. Dev-origin queries are kept separate in results because some baselines were fine-tuned on MIRACL train. |
| Rewrites | At least 2 per query from different anonymous annotators, typed without autocorrect, stored exactly as typed | One writer's spelling is one sample; two show how much spelling varies for the same question. |
| Corpus | Pooled subset of the 506k-passage MIRACL Hindi corpus: all judged passages plus the top 100 from BM25, multilingual-e5-base and bge-m3 for native and machine-romanized queries | Ten models times several query conditions on 506k passages is slow on a laptop, and MTEB prefers small tasks. Pooling with several systems keeps the hard distractors. |
| Metric | nDCG@10, with Recall@100 | nDCG@10 is the MTEB retrieval headline. Gaps are reported with 95% paired bootstrap intervals over queries. |
| Deduplication | MIRACL repeats some question texts under different ids. Selection keeps one id per normalised text, and drops training-pool queries that are near paraphrases of evaluation queries | Found by the leakage check on its first run: 4 training-pool queries were identical to test queries and 2 test queries were identical to dev queries. |

## Leakage rule

The human test set is never used for training, spelling-noise fitting, or model selection. `scripts/check_leakage.py --full` normalises every text (Devanagari mapped to Latin, lowercase, repeated letters collapsed) and fails on any exact match or char-5-gram Jaccard of 0.8 or more between a training query and an evaluation query, and on any training query generated from a passage judged for an evaluation query. CI re-runs the check from committed hash manifests.

## Reproduce

```
python -m venv .venv
.venv\Scripts\pip install -e ".[dev,models]"
python -m hinglish_retrieval.data.select_sources
python -m hinglish_retrieval.data.build_corpus
python scripts/reproduce.py
python scripts/check_leakage.py --full
```

## Results

Not yet reported.

## What did not work

To be written from measurements.

## Licences

See [LICENCES.md](LICENCES.md).
