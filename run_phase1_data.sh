#!/usr/bin/env bash
# Phase 1 data preparation: StockTwits -> labeled -> deduplicated -> canonical.
#
# Must complete before any of E0-E5 can train. Downloads the StockTwits source
# files from HuggingFace on first run.

set -eu

PY=".venv/Scripts/python.exe"
LOGDIR="results/_phase1_logs"
mkdir -p "$LOGDIR"

export PYTHONIOENCODING=utf-8
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

echo "### [1/3] Recovering StockTwits labels from source files ($(date +%H:%M:%S))"
$PY -m src.data.stocktwits_adapter --conflict-strategy drop 2>&1 | tee "$LOGDIR/01_adapter.log" | tail -20

echo ""
echo "### [2/3] Building final deduplicated dataset ($(date +%H:%M:%S))"
$PY -m src.data.build_final_dataset 2>&1 | tee "$LOGDIR/02_final_dataset.log" | tail -25

echo ""
echo "### [3/3] Building canonical JSONL splits ($(date +%H:%M:%S))"
$PY -c "
from src.data.preprocessing import build_final_canonical
build_final_canonical('data/processed', 'data/processed/canonical')
" 2>&1 | tee "$LOGDIR/03_canonical.log" | tail -10

echo ""
echo "### DATA PREP COMPLETE ($(date +%H:%M:%S))"
ls -la data/processed/canonical/
