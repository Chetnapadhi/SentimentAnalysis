#!/usr/bin/env bash
# Phase 1, unblocked subset: shared featurization + E0, E1, E3.
#
# These three need nothing beyond the canonical splits. E2/E4/E5 are held back
# because they all require the pretrained emoji artifact, which the repository
# never produced (see src/data/build_e2_emoji_artifact.py for the full story).
#
# The featurization here is shared, so once this finishes E2/E4/E5 only need
# minutes of head training each whenever they are approved to run.

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
  echo ">>> $name   ($(date +%H:%M:%S))   disk free: $(df -h /c | tail -1 | awk '{print $4}')"
  echo "=================================================================="
  if "$@" >"$log" 2>&1; then
    grep -E "Accuracy|Macro F1|Macro-F1|TEST RESULTS|saved|linked|Done|\[OK\]" "$log" \
      | grep -vE "^\s*$" | tail -12
    echo "--- ok: $log"
  else
    echo "!!! FAILED: $log"
    tail -30 "$log"
    return 1
  fi
}

echo "##### PHASE 1 CORE: featurization + E0/E1/E3 #####"

step 10_precompute_embeddings $PY -m src.data.precompute_embeddings || exit 1

step 20_E0 env RUN_E0=1 $PY run_e0.py
step 21_E1 env RUN_E1=1 $PY -m src.train_e1
step 23_E3 env RUN_E3=1 $PY -m src.train_e3

echo ""
echo "##### PHASE 1 CORE COMPLETE ($(date +%H:%M:%S)) #####"
df -h /c | tail -1
