#!/usr/bin/env bash
# Phase 1 tournament: E0-E5 on StockTwits, then the final analysis pipeline
# that produces the CSVs the Streamlit dashboard reads.
#
# Prerequisite: bash run_phase1_data.sh   (canonical splits must exist)
#
# Ordering matters:
#   1. Shared featurization  — computed once, hard-linked into all six caches.
#                              Without it the suite featurizes 123,763 texts
#                              six times (~15 h on CPU instead of ~2.6 h).
#   2. E2 emoji artifact     — E2/E4/E5 all refuse to start without it.
#   3. E0..E5                — head training only; minutes each once cached.
#   4. Analysis pipeline     — writes results/final_analysis/*.csv for the UI.

set -u

PY=".venv/Scripts/python.exe"
LOGDIR="results/_phase1_logs"
mkdir -p "$LOGDIR"

export PYTHONIOENCODING=utf-8
export TQDM_DISABLE=1
export TOKENIZERS_PARALLELISM=false

step() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  echo ""
  echo "=================================================================="
  echo ">>> $name   ($(date +%H:%M:%S))   disk: $(df -h /c | tail -1 | awk '{print $4}') free"
  echo "=================================================================="
  if "$@" >"$log" 2>&1; then
    grep -E "Accuracy|Macro F1|Macro-F1|TEST RESULTS|Saved|Transferred|linked|Best val|Done|COMPLETED|\[OK\]" "$log" \
      | grep -vE "^\s*$" | tail -14
    echo "--- ok: $log"
  else
    echo "!!! FAILED: $log"
    tail -30 "$log"
    return 1
  fi
}

echo "##### PHASE 1: E0-E5 + final analysis #####"

step 10_precompute_embeddings $PY -m src.data.precompute_embeddings || exit 1
step 11_e2_emoji_artifact     $PY -m src.data.build_e2_emoji_artifact || exit 1

step 20_E0 env RUN_E0=1 $PY run_e0.py
step 21_E1 env RUN_E1=1 $PY -m src.train_e1
step 22_E2 env RUN_E2=1 $PY -m src.train_e2_run
step 23_E3 env RUN_E3=1 $PY -m src.train_e3
step 24_E4 env RUN_E4=1 $PY -m src.train_e4
step 25_E5 env RUN_E5=1 $PY -m src.train_e5

step 30_final_analysis $PY -m src.analysis.run_pipeline

echo ""
echo "##### PHASE 1 COMPLETE ($(date +%H:%M:%S)) #####"
echo "Results:"
ls -1 results/final_analysis/ 2>/dev/null | head -20
echo ""
df -h /c | tail -1
