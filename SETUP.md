# Setting up on a new machine

These steps rebuild everything that is not in git: the Python environment, downloaded data and models, and cached embeddings. Sections 1 to 4 are for the original Windows machine (PowerShell, run from the project folder); section 5 is the rented Linux GPU run.

## 1. Python environment

Python 3.12 is required.

```
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e ".[dev,models]"
```

Check the GPU is visible:

```
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Sign-ins (one time per machine)

- Hugging Face: `.venv\Scripts\hf.exe auth login`, with a token of type Read. EmbeddingGemma also needs its licence accepted on its model page.
- GitHub, to push: the first `git push` opens a sign-in window.

If Hugging Face downloads fail with "Access is denied" on `.cache\huggingface`, set a project-local cache before running scripts:

```
$env:HF_HOME = "$PWD\data\cache\hf"
$env:HF_TOKEN_PATH = "$env:USERPROFILE\.cache\huggingface\token"
```

## 3. Rebuild data and caches

MIRACL topics and qrels, and the Dakshina sentences, download automatically on first use. The MIRACL Hindi passage corpus does not: download it once (about 100 MB):

```
.venv\Scripts\hf.exe download miracl/miracl-corpus --repo-type dataset --include "miracl-corpus-v1.0-hi/*" --local-dir data\raw\miracl-corpus
```

The pooled corpus id list is already in `data/corpus/`, so only the embeddings need rebuilding. The baseline tables take about 1 to 2 hours on a 12 GB GPU.

```
.venv\Scripts\python scripts\reproduce.py --role dev
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
| Benchmark tables | `.venv\Scripts\python scripts\reproduce.py --role dev`, then `--role test` |
| Training data (needs a 24 GB+ GPU) | `.venv\Scripts\python -m hinglish_retrieval.train.generate` |
| Fine-tuning | `.venv\Scripts\python -m hinglish_retrieval.train.train` |
| Pick the fine-tuned model on dev | `.venv\Scripts\python scripts\select_model.py` |
| Leakage check | `.venv\Scripts\python scripts\check_leakage.py --full` |
| Tests | `.venv\Scripts\python -m pytest -q` |

## 5. Linux GPU machine (Vast.ai)

For a rented Linux GPU. Pick an instance whose driver supports CUDA 13.0 or newer (the CUDA column in the Vast search), with 48 GB of GPU memory (an RTX A6000, L40S or A100; 24 GB works but the generation stage is about three times slower) and 100 GB of disk. Run long jobs inside `tmux` so they survive an SSH disconnect.

The repository is private, so `git clone` asks for your GitHub username and a personal access token with read access to it.

```
git clone https://github.com/aditya-ravi11/hinglish-retrieval.git
cd hinglish-retrieval
```

Python 3.12 environment. If the image has no `python3.12`, create the venv with `pip install uv && uv venv --seed --python 3.12 .venv` instead of the first line.

```
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e ".[dev,models]"
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Hugging Face sign-in (a Read token is enough; EmbeddingGemma's licence must be accepted on the same account), then the passage corpus:

```
.venv/bin/hf auth login
.venv/bin/hf download miracl/miracl-corpus --repo-type dataset --include "miracl-corpus-v1.0-hi/*" --local-dir data/raw/miracl-corpus
```

Start a `tmux` session and run the smoke test first. It runs every stage at tiny scale with the real models, in a scratch copy (15 to 25 minutes, most of it model downloads), and catches setup problems before the long run. If it ends with "smoke passed", start the real run:

```
tmux new -s run
bash scripts/vast_run.sh smoke
BUNDLE_CACHE=1 bash scripts/vast_run.sh
```

The real run takes about 3 to 5 hours on a 48 GB GPU. Its stages are: checks, training-data generation with Qwen3-8B (the longest), the full leakage check, fine-tuning three small models, the dev table and model selection, then the test table. Detach with Ctrl-b then d, and reattach with `tmux attach -t run`. If it stops for any reason, run the same command again: finished stages are skipped and generation resumes where it stopped. Logs are in `logs/`.

When it finishes, copy the bundle back. Run this on your own machine, in the project folder, then unpack it there (the port and host are on the instance's Connect panel):

```
scp -P <port> root@<host>:~/hinglish-retrieval/out/hinglish-run-*.tgz .
tar xzf hinglish-run-*.tgz
```

The bundle holds `results/`, `data/train/` and `data/manifests/` (commit these three), the selected model in `models/` (git-ignored; upload it to Hugging Face to publish it) and the logs. With `BUNDLE_CACHE=1` there is also `out/emb-cache-*.tgz` (about 5 GB) with the corpus embeddings, so the tables can be re-scored later without re-encoding; copy it back the same way if you want that. Then destroy the instance.
