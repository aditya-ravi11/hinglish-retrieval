# Licences of everything this project uses

Checked on 12 Nov 2026 from each Hugging Face page or repository. Re-check before release.

## Data

| Resource | Used for | Licence | Redistributed in our release? |
|---|---|---|---|
| `miracl/miracl` (topics and qrels v1.0) | Source queries and relevance labels | Apache-2.0 | Yes: query ids, query text, qrels |
| `miracl/miracl-corpus` Hindi (hiwiki-20220301) | Hindi passages | Apache-2.0 tag on the dataset; the text is Wikipedia, CC BY-SA | Yes, pooled subset, under CC BY-SA 4.0 |
| `mteb/MIRACLRetrievalHardNegatives` English | English corpus for the code-mixed subset | CC BY-SA 4.0 | Yes, under CC BY-SA 4.0 |
| Human Hinglish rewrites (collected by this project) | Test and dev queries | Released by us under CC BY-SA 4.0 (stated on the consent screen) | Yes |
| Google Dakshina | Fitting spelling noise | CC BY-SA 4.0 | No, only fitted statistics |
| `ai4bharat/Aksharantar` | Fitting spelling noise | CC-BY (manually collected part), CC0 (mined and existing sources) | No, only fitted statistics |

Not used: L3Cube and LIPI resources under CC BY-NC-SA, because the benchmark must be redistributable without a non-commercial clause.

## Models

| Model | Role | Licence |
|---|---|---|
| `intfloat/multilingual-e5-small` | Baseline, candidate student | MIT |
| `intfloat/multilingual-e5-base` | Baseline, pooling, default student | MIT |
| `BAAI/bge-m3` | Baseline, pooling | MIT |
| `google/embeddinggemma-300m` | Baseline (gated) | Gemma Terms of Use |
| `Qwen/Qwen3-Embedding-0.6B` | Baseline | Apache-2.0 |
| `AkshitaS/bhasha-embed-v0` | Baseline | Apache-2.0 |
| Local LLM for synthetic rewrites | Training data only | To be confirmed before use (permissive licence required) |
