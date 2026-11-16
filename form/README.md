---
title: Hinglish queries
emoji: ⌨️
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
---

# Hinglish query collection form

Shows one MIRACL source query at a time and asks people to type it the way they would type it to a friend or in a search bar. Rows go to a private HF dataset.

Space settings needed before it goes live:

- Variable `HR_DATA_REPO`: the private dataset that stores rows, for example `<your-hf-username>/hinglish-rewrites-private`.
- Secret `HF_TOKEN`: a fine-grained token with write access to that dataset only.

Local run: `python app.py` (rows go to `collected/`, nothing is uploaded).
