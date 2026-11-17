Role: test. Generated 2026-11-17T16:14:34+00:00 by scripts/reproduce.py.
Corpus sizes: {'hi': 171873, 'dk': 9899}. Queries per condition: {'native': 1000, 'machine': 1000, 'dk_machine': 4000, 'dk_human': 4000}.

nDCG@10 (x100), each condition over its own queries. Gap = native minus condition,
mean with a 95% paired bootstrap CI over the queries both conditions share.

| model | params (M) | native | machine | dk_machine | dk_human | gap native-machine |
|---|---|---|---|---|---|---|
| bm25 | 0 | 37.2 | 0.1 | 8.9 | 10.4 | 37.1 [34.8, 39.4] |
| me5-small | 118 | 56.0 | 0.1 | 24.4 | 43.9 | 55.9 [53.7, 58.1] |
| me5-base | 278 | 57.9 | 0.0 | 21.7 | 41.2 | 57.9 [55.7, 60.0] |
| bge-m3 | 568 | 67.4 | 1.1 | 47.4 | 63.0 | 66.3 [64.3, 68.2] |
| embeddinggemma-300m | 308 | 64.6 | 11.7 | 65.7 | 75.5 | 52.8 [50.5, 55.1] |
| qwen3-emb-0.6b | 596 | 49.7 | 12.5 | 67.9 | 76.8 | 37.3 [35.0, 39.5] |
| bhasha-embed-v0 | 238 | 44.8 | 23.2 | 96.1 | 96.3 | 21.6 [19.7, 23.6] |
| bt-bm25 | 4022 | 37.2 | 20.1 | 94.7 | 95.0 | 17.1 [15.1, 19.2] |
| bt-me5-base | 4301 | 57.9 | 43.4 | 98.3 | 98.9 | 14.5 [12.7, 16.3] |
| bt-bge-m3 | 4590 | 67.4 | 54.3 | 98.9 | 99.3 | 13.1 [11.3, 14.8] |
| hr-bhasha-embed-v0 | 238 | 43.5 | 31.1 | 99.0 | 99.0 | 12.4 [10.9, 14.1] |

Dakshina: the same 4000 sentences typed by a native speaker (human) and
romanized by rule (machine); the system must find the Devanagari sentence among
9899. Gap = machine minus human, 95% paired bootstrap CI.

| model | machine | human | gap machine-human |
|---|---|---|---|
| bm25 | 8.9 | 10.4 | -1.5 [-1.9, -1.2] |
| me5-small | 24.4 | 43.9 | -19.5 [-20.6, -18.3] |
| me5-base | 21.7 | 41.2 | -19.5 [-20.7, -18.4] |
| bge-m3 | 47.4 | 63.0 | -15.6 [-16.8, -14.6] |
| embeddinggemma-300m | 65.7 | 75.5 | -9.8 [-10.8, -8.9] |
| qwen3-emb-0.6b | 67.9 | 76.8 | -8.9 [-9.9, -7.9] |
| bhasha-embed-v0 | 96.1 | 96.3 | -0.2 [-0.9, 0.4] |
| bt-bm25 | 94.7 | 95.0 | -0.3 [-0.9, 0.2] |
| bt-me5-base | 98.3 | 98.9 | -0.6 [-0.9, -0.3] |
| bt-bge-m3 | 98.9 | 99.3 | -0.4 [-0.7, -0.2] |
| hr-bhasha-embed-v0 | 99.0 | 99.0 | 0.1 [-0.3, 0.4] |

Native nDCG@10 (x100) by MIRACL origin split of the query (contamination check):

| model | dev-origin | train-origin |
|---|---|---|
| bm25 | 35.9 (n=349) | 37.9 (n=651) |
| me5-small | 55.6 (n=349) | 56.2 (n=651) |
| me5-base | 58.5 (n=349) | 57.6 (n=651) |
| bge-m3 | 59.4 (n=349) | 71.7 (n=651) |
| embeddinggemma-300m | 57.7 (n=349) | 68.2 (n=651) |
| qwen3-emb-0.6b | 51.0 (n=349) | 49.0 (n=651) |
| bhasha-embed-v0 | 43.7 (n=349) | 45.4 (n=651) |
| bt-bm25 | 35.9 (n=349) | 37.9 (n=651) |
| bt-me5-base | 58.5 (n=349) | 57.6 (n=651) |
| bt-bge-m3 | 59.4 (n=349) | 71.7 (n=651) |
| hr-bhasha-embed-v0 | 45.2 (n=349) | 42.7 (n=651) |
