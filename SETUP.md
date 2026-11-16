# Setting up on a new machine

These steps rebuild everything that was not copied: the Python environment, downloaded data and models, and cached embeddings. Commands are for Windows PowerShell, run from the project folder.

## 1. Python environment

Python 3.12 is required.

```
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e ".[dev,models,form]"
```

Check the GPU is visible:

```
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Sign-ins (one time per machine)

- Hugging Face: `.venv\Scripts\hf.exe auth login`, with a token of type Write. EmbeddingGemma also needs its licence accepted on its model page.
- Google, for the form and for pulling answers: `npx.cmd @google/clasp@3.4.1 login`, and tick every permission on the Google page, including Google Drive.
- GitHub, to push: the first `git push` opens a sign-in window.

If Hugging Face downloads fail with "Access is denied" on `.cache\huggingface`, set a project-local cache before running scripts:

```
$env:HF_HOME = "$PWD\data\cache\hf"
$env:HF_TOKEN_PATH = "$env:USERPROFILE\.cache\huggingface\token"
```

## 3. Rebuild data and caches

The MIRACL files download automatically on first use. The pooled corpus id list is already in `data/corpus/`, so only the embeddings need rebuilding. The first full run takes about 1 to 2 hours on a 12 GB GPU and longer on 8 GB.

```
.venv\Scripts\python scripts\reproduce.py --role test
```

To recreate the source selection or the pooled corpus from scratch (not needed normally):

```
.venv\Scripts\python -m hinglish_retrieval.data.select_sources
.venv\Scripts\python -m hinglish_retrieval.data.build_corpus
```

## 4. Day-to-day commands

| Task | Command |
|---|---|
| Pull new form answers, import them, check typing speed | `.venv\Scripts\python scripts\pull_sheet.py` |
| Full benchmark table | `.venv\Scripts\python scripts\reproduce.py --role test` |
| Go/no-go check | `.venv\Scripts\python scripts\gate.py` |
| Leakage check | `.venv\Scripts\python scripts\check_leakage.py --full` |
| Tests | `.venv\Scripts\python -m pytest -q` and `node --test tests\test_gas_assign.mjs tests\test_gas_code.mjs` |
| Push form code changes | `npx.cmd @google/clasp@3.4.1 push --force`, then `npx.cmd @google/clasp@3.4.1 update-deployment REDACTED_DEPLOYMENT_ID` (keeps the same form link) |

## 5. Private files

`data/collected/` holds the human answers and is git-ignored on purpose: it is not published until the release is approved. `.clasp.json` links this folder to the live Apps Script project and is also git-ignored. Keep both when moving machines, and do not commit them.
