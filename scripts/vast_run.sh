#!/usr/bin/env bash
# The whole GPU run in one command, for a rented Linux GPU (e.g. Vast.ai).
# Set up the machine first: SETUP.md, section 6.
#
#   bash scripts/vast_run.sh smoke   # 15-25 minutes: every stage at tiny scale with the real models, in a scratch copy
#   bash scripts/vast_run.sh         # the real run
#
# Real-run stages. Each one leaves logs/done-<stage> when it finishes and is
# skipped on the next start, so after a crash or a dropped SSH session just run
# the same command again. Delete a marker to redo that stage.
#   checks    GPU, corpus, Hugging Face login, Dakshina download, unit tests
#   generate  synthetic training data with the LLM (the longest stage)
#   leakage   scripts/check_leakage.py --full; stops the run on any overlap
#   train     fine-tune the small models
#   dev       every baseline and fine-tuned model on the dev split, then pick one
#   test      the full test table with the picked model
#   bundle    out/hinglish-run-<time>.tgz to copy back to your own machine
#
# Settings (environment variables): N (training items, default 20000),
# LLM (generator, default Qwen/Qwen3-8B), HR_BT_LLM (back-transliteration,
# default Qwen/Qwen3-4B), BUNDLE_CACHE=1 to also pack the corpus embeddings
# (about 5 GB) so the table can be rerun later without re-encoding.

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PY:-$ROOT/.venv/bin/python}
N=${N:-20000}
LLM=${LLM:-Qwen/Qwen3-8B}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p logs out
exec > >(tee -a "logs/run-$STAMP.log") 2>&1

say() { printf '\n=== %s  %s\n' "$(date +%H:%M:%S)" "$*"; }

stage() {  # stage <name> <command...>: run once, leave a marker
    local name=$1; shift
    if [[ -f logs/done-$name ]]; then say "$name: already done, skipping"; return; fi
    say "$name"
    "$@"
    touch "logs/done-$name"
}

checks() {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
    local f=data/raw/miracl-corpus/miracl-corpus-v1.0-hi/docs-0.jsonl.gz
    [[ -f $f ]] || { echo "missing $f: run the hf download line in SETUP.md section 6"; exit 1; }
    "$PY" -c "from hinglish_retrieval.data import dakshina as d; print('Dakshina sentences:', len(d.load()))"
    "$PY" -c "from huggingface_hub import whoami; print('Hugging Face user:', whoami()['name'])" \
        || echo "WARNING: not logged in to Hugging Face; embeddinggemma-300m (gated) will fail"
    "$PY" -m pytest -q
}

bundle() {
    local sel files
    sel=$(cat models/SELECTED)
    files=(results data/train data/manifests models/SELECTED "models/$sel" logs)
    compgen -G "data/cache/backtranslit-*.json" > /dev/null && files+=(data/cache/backtranslit-*.json)
    # exit status 1 only means the live log grew while being packed
    tar czf "out/hinglish-run-$STAMP.tgz" --warning=no-file-changed "${files[@]}" || [[ $? == 1 ]]
    if [[ ${BUNDLE_CACHE:-0} == 1 ]]; then tar czf "out/emb-cache-$STAMP.tgz" data/cache/emb; fi
    ls -lh out/
    echo "Copy back from your own machine:  scp -P <port> root@<host>:$ROOT/out/hinglish-run-$STAMP.tgz ."
}

smoke() {
    # Scratch copy, so nothing here touches the real data/train, models or results.
    local S=/tmp/hr-smoke
    rm -rf "$S" && mkdir -p "$S/data"
    tar cf - --exclude=./.venv --exclude=./data --exclude=./models --exclude=./logs --exclude=./out . | tar xf - -C "$S"
    cp -r data/sources data/corpus data/manifests "$S/data/"
    ln -s "$ROOT/data/raw" "$S/data/raw"
    cd "$S"
    export PYTHONPATH="$S/src"
    say "smoke: checks"; checks
    # the real generator on a few passages: fails here if its answers are unusable
    say "smoke: generate"; "$PY" -m hinglish_retrieval.train.generate --n 30 --llm "$LLM"
    say "smoke: leakage"; "$PY" scripts/check_leakage.py --full
    say "smoke: train"; "$PY" -m hinglish_retrieval.train.train --bases me5-small --max-steps 5 --batch 16 --mini-batch 8
    say "smoke: dev"
    "$PY" scripts/reproduce.py --role dev --models bm25,bt-bm25,me5-small,hr-me5-small --corpus-limit 3000
    cp results/smoke_dev.json results/benchmark_dev.json && "$PY" scripts/select_model.py
    say "smoke: test"
    "$PY" scripts/reproduce.py --role test --models bm25,bt-me5-base,hr-me5-small --corpus-limit 3000
    say "smoke passed: every stage ran. Start the real run with: bash scripts/vast_run.sh"
}

if [[ ${1:-} == smoke ]]; then smoke; exit 0; fi

stage checks checks
stage generate "$PY" -m hinglish_retrieval.train.generate --n "$N" --llm "$LLM"
stage leakage "$PY" scripts/check_leakage.py --full
stage train "$PY" -m hinglish_retrieval.train.train
stage dev bash -c "'$PY' scripts/reproduce.py --role dev && '$PY' scripts/select_model.py"
stage test "$PY" scripts/reproduce.py --role test
say "bundle"; bundle
say "done"
