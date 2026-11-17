# Licences of everything this project uses

Checked on 17 Nov 2026 from each Hugging Face page or repository. Re-check before release.

## Data

| Resource | Used for | Licence | Redistributed in our release? |
|---|---|---|---|
| `miracl/miracl` (topics and qrels v1.0) | Source queries and relevance labels | Apache-2.0 | Yes: query ids, query text, qrels |
| `miracl/miracl-corpus` Hindi (hiwiki-20220301) | Hindi passages; seed passages for synthetic training data | Apache-2.0 tag on the dataset; the text is Wikipedia, CC BY-SA | Yes, pooled subset, under CC BY-SA 4.0 |
| Google Dakshina, Hindi romanized sentences (read from the Parquet copy `Anvesh-Lankala/Copy_Dakshina_Google_research_dataset`, pinned revision) | Human-romanized evaluation queries and their native targets | CC BY-SA 4.0 | Yes, the sentence pairs used, under CC BY-SA 4.0 |
| Synthetic training data (`data/train/`, written by Qwen3-8B and our rules) | Training only | Released by us under CC BY-SA 4.0 (derived from Wikipedia passages) | Yes |

Not used: L3Cube and LIPI resources under CC BY-NC-SA, because the benchmark must be redistributable without a non-commercial clause.

## Models

| Model | Role | Licence |
|---|---|---|
| `intfloat/multilingual-e5-small` | Baseline, fine-tuning candidate | MIT |
| `intfloat/multilingual-e5-base` | Baseline, pooling, back-transliteration retriever, fine-tuning candidate | MIT |
| `BAAI/bge-m3` | Baseline, pooling, back-transliteration retriever | MIT |
| `google/embeddinggemma-300m` | Baseline (gated) | Gemma Terms of Use |
| `Qwen/Qwen3-Embedding-0.6B` | Baseline | Apache-2.0 |
| `AkshitaS/bhasha-embed-v0` | Baseline, fine-tuning candidate | Apache-2.0 |
| `Qwen/Qwen3-8B` | Writes the synthetic training questions and their Hinglish versions | Apache-2.0 |
| `Qwen/Qwen3-4B` | Back-transliteration step of the bt-* baselines | Apache-2.0 |

A fine-tuned model keeps its base model's licence (MIT or Apache-2.0).
