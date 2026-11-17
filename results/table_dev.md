Role: dev. Generated 2026-11-17T15:11:31+00:00 by scripts/reproduce.py.
Corpus sizes: {'hi': 171873, 'dk': 9899}. Queries per condition: {'native': 150, 'machine': 150, 'dk_machine': 1000, 'dk_human': 1000}.

nDCG@10 (x100), each condition over its own queries. Gap = native minus condition,
mean with a 95% paired bootstrap CI over the queries both conditions share.

| model | params (M) | native | machine | dk_machine | dk_human | gap native-machine |
|---|---|---|---|---|---|---|
| bm25 | 0 | 36.7 | 0.0 | 8.6 | 10.0 | 36.7 [31.0, 42.5] |
| me5-small | 118 | 63.5 | 0.0 | 23.7 | 44.8 | 63.5 [58.5, 68.6] |
| me5-base | 278 | 63.6 | 0.0 | 21.3 | 42.1 | 63.6 [58.2, 68.8] |
| bge-m3 | 568 | 74.7 | 0.7 | 47.4 | 63.5 | 74.1 [69.5, 78.4] |
| embeddinggemma-300m | 308 | 69.3 | 12.7 | 66.4 | 76.2 | 56.7 [50.9, 62.4] |
| qwen3-emb-0.6b | 596 | 52.9 | 16.3 | 70.2 | 78.4 | 36.5 [30.6, 42.5] |
| bhasha-embed-v0 | 238 | 48.4 | 27.2 | 96.0 | 96.5 | 21.2 [16.0, 26.5] |
| bt-bm25 | 4022 | 36.7 | 26.1 | 93.2 | 94.6 | 10.7 [5.9, 15.4] |
| bt-me5-base | 4301 | 63.6 | 52.1 | 98.4 | 99.2 | 11.5 [7.1, 16.2] |
| bt-bge-m3 | 4590 | 74.7 | 61.4 | 99.3 | 99.7 | 13.4 [9.3, 17.7] |
| hr-bhasha-embed-v0 | 238 | 49.5 | 35.6 | 99.1 | 99.1 | 14.0 [9.6, 18.7] |
| hr-me5-base | 278 | 43.6 | 15.5 | 63.4 | 75.4 | 28.1 [22.9, 33.4] |
| hr-me5-small | 118 | 42.7 | 16.4 | 59.1 | 68.9 | 26.3 [20.7, 32.0] |

Dakshina: the same 1000 sentences typed by a native speaker (human) and
romanized by rule (machine); the system must find the Devanagari sentence among
9899. Gap = machine minus human, 95% paired bootstrap CI.

| model | machine | human | gap machine-human |
|---|---|---|---|
| bm25 | 8.6 | 10.0 | -1.4 [-2.2, -0.7] |
| me5-small | 23.7 | 44.8 | -21.1 [-23.4, -18.8] |
| me5-base | 21.3 | 42.1 | -20.8 [-23.1, -18.5] |
| bge-m3 | 47.4 | 63.5 | -16.1 [-18.4, -13.8] |
| embeddinggemma-300m | 66.4 | 76.2 | -9.8 [-11.8, -8.0] |
| qwen3-emb-0.6b | 70.2 | 78.4 | -8.2 [-10.2, -6.3] |
| bhasha-embed-v0 | 96.0 | 96.5 | -0.5 [-1.7, 0.7] |
| bt-bm25 | 93.2 | 94.6 | -1.4 [-2.4, -0.3] |
| bt-me5-base | 98.4 | 99.2 | -0.8 [-1.5, -0.1] |
| bt-bge-m3 | 99.3 | 99.7 | -0.4 [-0.9, 0.0] |
| hr-bhasha-embed-v0 | 99.1 | 99.1 | -0.1 [-0.7, 0.5] |
| hr-me5-base | 63.4 | 75.4 | -11.9 [-13.9, -10.0] |
| hr-me5-small | 59.1 | 68.9 | -9.9 [-11.6, -8.2] |

Native nDCG@10 (x100) by MIRACL origin split of the query (contamination check):

| model | dev-origin | train-origin |
|---|---|---|
| bm25 |  | 36.7 (n=150) |
| me5-small |  | 63.5 (n=150) |
| me5-base |  | 63.6 (n=150) |
| bge-m3 |  | 74.7 (n=150) |
| embeddinggemma-300m |  | 69.3 (n=150) |
| qwen3-emb-0.6b |  | 52.9 (n=150) |
| bhasha-embed-v0 |  | 48.4 (n=150) |
| bt-bm25 |  | 36.7 (n=150) |
| bt-me5-base |  | 63.6 (n=150) |
| bt-bge-m3 |  | 74.7 (n=150) |
| hr-bhasha-embed-v0 |  | 49.5 (n=150) |
| hr-me5-base |  | 43.6 (n=150) |
| hr-me5-small |  | 42.7 (n=150) |
