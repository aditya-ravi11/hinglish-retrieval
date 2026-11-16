Role: test. Generated 2026-11-16T17:45:39+00:00 by scripts/reproduce.py.
Corpus sizes: {'hi': 171873, 'en': 178768}. Queries per condition: {'native': 1000, 'machine': 1000, 'human_w1': 172, 'en_native': 150, 'en_human_w1': 23}.

nDCG@10 (x100), each condition over its own queries. Gap = native minus condition,
mean with a 95% paired bootstrap CI over the queries both conditions share.

| model | params (M) | native | machine | human_w1 | en_native | en_human_w1 | gap native-machine |
|---|---|---|---|---|---|---|---|
| bm25 | 0 | 37.2 | 0.1 | 0.1 | 28.4 | 11.8 | 37.1 [34.8, 39.4] |
| me5-small | 118 | 55.9 | 0.1 | 3.0 | 53.6 | 13.4 | 55.9 [53.7, 58.0] |
| me5-base | 278 | 57.7 | 0.0 | 0.9 | 56.0 | 15.5 | 57.7 [55.6, 59.8] |
| bge-m3 | 568 | 67.4 | 1.1 | 8.5 | 61.3 | 33.9 | 66.3 [64.3, 68.2] |
| embeddinggemma-300m | 308 | 64.5 | 11.7 | 26.1 | 58.0 | 46.3 | 52.8 [50.4, 55.0] |
| qwen3-emb-0.6b | 596 | 49.7 | 12.5 | 20.1 | 55.5 | 42.6 | 37.1 [34.9, 39.4] |
| bhasha-embed-v0 | 238 | 44.8 | 23.2 | 24.1 | 43.5 | 33.1 | 21.6 [19.7, 23.7] |

Human Hinglish subset: the 172 queries with at least one human rewrite. Native, machine and
human are scored on these same queries. Human = mean over the writers each query has.

| model | native | machine | human | gap native-human | gap machine-human |
|---|---|---|---|---|---|
| bm25 | 37.5 | 0.1 | 0.1 | 37.4 [32.2, 42.9] | -0.1 [-0.2, 0.1] |
| me5-small | 59.0 | 0.2 | 3.0 | 56.0 [51.2, 61.1] | -2.8 [-4.9, -1.1] |
| me5-base | 56.1 | 0.0 | 0.9 | 55.2 [50.0, 60.4] | -0.9 [-2.1, -0.0] |
| bge-m3 | 67.3 | 1.3 | 8.5 | 58.8 [53.7, 63.9] | -7.1 [-10.4, -4.2] |
| embeddinggemma-300m | 64.7 | 11.7 | 26.1 | 38.6 [33.1, 44.2] | -14.3 [-18.4, -10.5] |
| qwen3-emb-0.6b | 48.5 | 13.4 | 20.1 | 28.3 [23.2, 33.5] | -6.7 [-10.5, -3.2] |
| bhasha-embed-v0 | 41.1 | 21.5 | 24.1 | 17.1 [13.2, 21.0] | -2.6 [-5.6, 0.3] |

Native nDCG@10 (x100) by MIRACL origin split of the query (contamination check):

| model | dev-origin | train-origin |
|---|---|---|
| bm25 | 35.9 (n=349) | 37.9 (n=651) |
| me5-small | 55.6 (n=349) | 56.1 (n=651) |
| me5-base | 58.2 (n=349) | 57.5 (n=651) |
| bge-m3 | 59.4 (n=349) | 71.7 (n=651) |
| embeddinggemma-300m | 57.6 (n=349) | 68.2 (n=651) |
| qwen3-emb-0.6b | 51.0 (n=349) | 49.0 (n=651) |
| bhasha-embed-v0 | 43.7 (n=349) | 45.4 (n=651) |
